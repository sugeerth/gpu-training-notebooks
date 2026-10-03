// 06_flash_decode.cu — decode attention: online softmax, split-K, and a paged KV cache, in
// CUDA. This is the kernel from Attention_Kernels_From_Scratch.ipynb, which develops the same
// four algorithms in NumPy; the Python there is the specification and this is the machine.
//
//     nvcc -O3 -arch=native 06_flash_decode.cu -o build/06_flash_decode && build/06_flash_decode
//     make check
//
// The decode-time problem
// -----------------------
// One query token attends to a KV cache of S positions. Per head:
//
//     s = q·Kᵀ / sqrt(D)   [S]      p = softmax(s)   [S]      o = p·V   [D]
//
// 4·S·D FLOPs against 2·S·D·4 bytes of KV: 0.5 FLOP/byte, hopelessly memory-bound, exactly
// like the GEMV in 05_dequant_gemv.cu. Which means the goals are (a) read K and V exactly
// once, (b) never materialize the S-element score vector, and (c) keep every SM busy even
// though there is only one query.
//
// Those are the four variants below, in order. Each is a different failure to fix:
//
//   1. two-pass  — softmax needs the max before it can normalize, so the obvious kernel
//                  writes S scores to global memory and reads them back twice.
//   2. online    — keep a running max and running sum, and *rescale* the accumulator whenever
//                  the max moves. One pass, no scratch, mathematically identical.
//   3. split-K   — one query means one block means one SM busy and 131 idle. Split the KV
//                  cache across blocks, let each produce a partial (m, ℓ, acc), and merge them
//                  with the same rescaling rule. This is FlashDecoding.
//   4. paged     — the KV cache is not contiguous; it is 16-token pages scattered across a
//                  pool. The only change to the inner loop is one indirection through a block
//                  table. This is PagedAttention, and the cost is a pointer chase per page.
//
// The rescaling identity all of this rests on, for two partial results (m₁,ℓ₁,acc₁) and
// (m₂,ℓ₂,acc₂) with m = max(m₁,m₂):
//
//     ℓ   = ℓ₁·e^(m₁-m) + ℓ₂·e^(m₂-m)
//     acc = acc₁·e^(m₁-m) + acc₂·e^(m₂-m)
//
// It is associative and commutative, which is exactly why the KV cache can be split any way
// the scheduler likes and merged in any order.
//
// One honest caveat before the numbers arrive
// -------------------------------------------
// In *prefill*, the intermediate that online softmax eliminates is the N x N score matrix, and
// avoiding it is worth an order of magnitude — that is the FlashAttention result, developed in
// Attention_Kernels_From_Scratch.ipynb. In *decode*, which is what this file measures, the
// intermediate is a vector of S scores against a KV cache of S x D. The traffic saved is
//
//     3·S / (2·S·D) = 3 / (2·D) ≈ 1.2% at D = 128
//
// — nearly nothing. So do not expect variant 2 to be several times faster than variant 1 here,
// and be suspicious of any writeup that promises it. What variant 1 actually costs you at
// decode time is not bandwidth:
//
//   * S floats of scratch per head per layer, which must be *reserved* for the longest
//     sequence the server will admit. At S=128k and 32 heads that is 16 MB per layer of
//     memory the KV cache does not get to use.
//   * a hard barrier between the passes, so nothing downstream can be fused into it
//   * a footprint that grows with context, in a regime where context length is exactly what
//     you are trying to grow
//
// The large win in this file is variant 3, and it is an occupancy win, not a traffic win.
#include "common.cuh"

#if SHIM_BUILD
constexpr int D = 32;       // head dim
constexpr int TILE = 32;    // KV positions per tile == page size
constexpr int NSPLIT = 4;
#else
constexpr int D = 128;
constexpr int TILE = 32;
constexpr int NSPLIT = 8;
#endif

// Running-max sentinel, and a large finite negative rather than -INFINITY.
//
// Being precise about when this matters, because it is usually stated too strongly: the
// rescale computes exp(m_old - m_new), and (-inf) - (-inf) is NaN. For that you need *both*
// terms to still be the sentinel, which means a tile in which every score is -inf. The
// unmasked decode kernels in this file never produce one — every tile they see has at least
// one real key, so m_new is always finite by the end of the first tile and -INFINITY would
// work here.
//
// It stops working the moment there is a mask. A causal or sliding-window kernel sets masked
// scores to -inf, and a query near the start of a window gets tiles that are masked out
// *entirely*; a merge across an empty split (see the short-sequence check in main()) is the
// same situation. Then m_old and m_new are both -inf, exp() returns NaN, and the NaN
// propagates through the accumulator for the rest of the sequence and out into the logits.
//
// -1e30f gives exp(m_old - m_new) == 0 in exactly those cases, which is the answer you wanted.
// It costs nothing, so production flash kernels carry it whether or not they mask today.
constexpr float NEG = -1e30f;

// ---------------------------------------------------------------------------------------
// Variant 1: the textbook three passes, with the score vector in global memory.
//
// One block per head, D threads. `scores` is [heads x S] of scratch that exists only because
// softmax was written as three separate reductions.
//
// The two reductions here are properly parallel — a block-wide max and a block-wide sum, both
// built from the shuffle primitive of 02_reduce.cu. That is deliberate: a serial scan in
// thread 0 would make this variant lose for a reason that has nothing to do with the
// algorithm, and a benchmark whose baseline is a strawman teaches the wrong lesson. The only
// thing separating this from variant 2 is the three passes and the scratch.
// ---------------------------------------------------------------------------------------
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ float block_reduce(float v, float* scratch, bool take_max) {
  const int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    const float o = __shfl_down_sync(FULL, v, off);
    v = take_max ? fmaxf(v, o) : v + o;
  }
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
  if (warp == 0) {
    v = (lane < nwarps) ? scratch[lane] : (take_max ? NEG : 0.0f);
#pragma unroll
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const float o = __shfl_down_sync(FULL, v, off);
      v = take_max ? fmaxf(v, o) : v + o;
    }
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  const float out = scratch[0];
  __syncthreads();            // scratch may be reused by the next reduction
  return out;
}

__global__ void decode_two_pass(const float* __restrict__ Q, const float* __restrict__ Kc,
                                const float* __restrict__ Vc, float* __restrict__ scores,
                                float* __restrict__ O, int S) {
  SHARED(float, sq, D);
  SHARED(float, red, 32);
  const int h = blockIdx.x, d = threadIdx.x;
  const float scale = rsqrtf((float)D);

  sq[d] = Q[(size_t)h * D + d];
  __syncthreads();

  float* sc = scores + (size_t)h * S;

  // pass 1: all S scores, out to global
  for (int j = d; j < S; j += blockDim.x) {
    const float* kj = Kc + ((size_t)h * S + j) * D;
    float acc = 0.0f;
    for (int i = 0; i < D; ++i) acc += sq[i] * kj[i];
    sc[j] = acc * scale;
  }
  __syncthreads();

  // pass 2: the max, read back from global
  float part = NEG;
  for (int j = d; j < S; j += blockDim.x) part = fmaxf(part, sc[j]);
  const float m = block_reduce(part, red, true);

  // pass 2b: the sum, read back from global again
  part = 0.0f;
  for (int j = d; j < S; j += blockDim.x) part += __expf(sc[j] - m);
  const float l = block_reduce(part, red, false);

  // pass 3: weighted sum of V, reading the scores a third time
  float acc = 0.0f;
  for (int j = 0; j < S; ++j) acc += __expf(sc[j] - m) * Vc[((size_t)h * S + j) * D + d];
  O[(size_t)h * D + d] = acc / l;
}

// ---------------------------------------------------------------------------------------
// Variant 2: online softmax, one pass, no scratch.
//
// The block stages a TILE of K and V into shared memory, computes TILE scores, then folds
// them into the running (m, ℓ, acc) with the rescaling identity above. K and V are each read
// exactly once from HBM, and the S-element score vector never exists.
//
// Two things worth reading closely:
//
//   * every thread computes the same m, ℓ, and correction factor, from the same shared-memory
//     values, in the same order. That is deliberate: TILE is small enough that a serial scan
//     is cheaper than a block reduction plus two barriers, and the reads are broadcasts (all
//     32 lanes read sp[i]) which shared memory serves in a single cycle with no bank conflict.
//     Only `acc` differs per thread, because only `acc` is indexed by d.
//
//   * the __syncthreads() at the bottom of the loop is defensive, not load-bearing, and it is
//     worth understanding why — this is the kind of reasoning a barrier audit consists of.
//     Thread d stages column d of sK and sV, and in the accumulation it reads sV[i·D+d]:
//     the same column. So a thread racing ahead into tile n+1 can only overwrite its own
//     column, which no other thread reads. The two genuinely shared arrays are sp (written by
//     thread d, read by everyone) and sK (staged by column, read by row) — and both are
//     already fenced by the barrier at the top of the next iteration, which a fast thread
//     cannot pass until the slow ones arrive.
//
//     It is kept anyway because that argument depends entirely on the staging pattern. Change
//     the loads to vectorized or swizzled staging — as any tuned version of this kernel would
//     — and thread d no longer owns column d, at which point the barrier becomes mandatory and
//     its absence becomes a rare, input-dependent wrong answer. One barrier per tile is a
//     cheap price for not having to redo this analysis after every edit.
// ---------------------------------------------------------------------------------------
__global__ void decode_online(const float* __restrict__ Q, const float* __restrict__ Kc,
                              const float* __restrict__ Vc, float* __restrict__ O, int S) {
  SHARED(float, sq, D);
  SHARED(float, sK, TILE * D);
  SHARED(float, sV, TILE * D);
  SHARED(float, sp, TILE);

  const int h = blockIdx.x, d = threadIdx.x;
  const float scale = rsqrtf((float)D);
  sq[d] = Q[(size_t)h * D + d];
  __syncthreads();

  float m = NEG, l = 0.0f, acc = 0.0f;

  for (int j0 = 0; j0 < S; j0 += TILE) {
    const int n = (S - j0 < TILE) ? (S - j0) : TILE;
    // Cooperative, coalesced stage: thread d fetches column d of each row.
    for (int r = 0; r < n; ++r) {
      size_t base = ((size_t)h * S + j0 + r) * D;
      sK[r * D + d] = Kc[base + d];
      sV[r * D + d] = Vc[base + d];
    }
    __syncthreads();

    if (d < n) {
      float s = 0.0f;
      for (int i = 0; i < D; ++i) s += sq[i] * sK[d * D + i];
      sp[d] = s * scale;
    }
    __syncthreads();

    float mt = NEG;
    for (int i = 0; i < n; ++i) mt = fmaxf(mt, sp[i]);
    const float mnew = fmaxf(m, mt);
    const float corr = __expf(m - mnew);      // rescale everything accumulated so far

    float lt = 0.0f, at = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float p = __expf(sp[i] - mnew);
      lt += p;
      at += p * sV[i * D + d];
    }
    l = l * corr + lt;
    acc = acc * corr + at;
    m = mnew;
    __syncthreads();                          // tile n is finished with; now it may be reused
  }
  O[(size_t)h * D + d] = acc / l;
}

// ---------------------------------------------------------------------------------------
// Variant 3: FlashDecoding — split the KV cache across NSPLIT blocks.
//
// Variant 2 launches one block per head. At batch 1 with 32 heads that is 32 blocks on a GPU
// with 132 SMs: three quarters of the machine idle while a single thread block walks a 128k
// KV cache. The arithmetic is trivially parallel over the sequence; only the softmax
// normalizer couples it, and the rescaling identity decouples that.
//
// Each block writes an *unnormalized* partial: acc scaled by e^(-m_s), plus its own m_s and
// ℓ_s. The combine kernel merges them pairwise. Writing normalized partials instead would
// lose the information needed to merge, which is the single most common way this gets built
// wrong.
// ---------------------------------------------------------------------------------------
__global__ void decode_split(const float* __restrict__ Q, const float* __restrict__ Kc,
                             const float* __restrict__ Vc, float* __restrict__ pacc,
                             float* __restrict__ pm, float* __restrict__ pl, int S,
                             int nsplit) {
  SHARED(float, sq, D);
  SHARED(float, sK, TILE * D);
  SHARED(float, sV, TILE * D);
  SHARED(float, sp, TILE);

  const int h = blockIdx.x, sp_id = blockIdx.y, d = threadIdx.x;
  const int chunk = (S + nsplit - 1) / nsplit;
  const int begin = sp_id * chunk;
  const int end = (begin + chunk < S) ? begin + chunk : S;

  const float scale = rsqrtf((float)D);
  sq[d] = Q[(size_t)h * D + d];
  __syncthreads();

  float m = NEG, l = 0.0f, acc = 0.0f;
  for (int j0 = begin; j0 < end; j0 += TILE) {
    const int n = (end - j0 < TILE) ? (end - j0) : TILE;
    for (int r = 0; r < n; ++r) {
      size_t base = ((size_t)h * S + j0 + r) * D;
      sK[r * D + d] = Kc[base + d];
      sV[r * D + d] = Vc[base + d];
    }
    __syncthreads();
    if (d < n) {
      float s = 0.0f;
      for (int i = 0; i < D; ++i) s += sq[i] * sK[d * D + i];
      sp[d] = s * scale;
    }
    __syncthreads();

    float mt = NEG;
    for (int i = 0; i < n; ++i) mt = fmaxf(mt, sp[i]);
    const float mnew = fmaxf(m, mt);
    const float corr = __expf(m - mnew);
    float lt = 0.0f, at = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float p = __expf(sp[i] - mnew);
      lt += p;
      at += p * sV[i * D + d];
    }
    l = l * corr + lt;
    acc = acc * corr + at;
    m = mnew;
    __syncthreads();
  }

  const size_t slot = (size_t)h * nsplit + sp_id;
  pacc[slot * D + d] = acc;      // unnormalized, still carrying its own m
  if (d == 0) {
    // An empty split (begin == end) falls straight through the loop and publishes its
    // initializers — m = NEG, l = 0, acc = 0 — which is exactly the identity element of the
    // merge. No special case is needed here, but only because those initializers are right.
    pm[slot] = m;
    pl[slot] = l;
  }
}

__global__ void decode_combine(const float* __restrict__ pacc, const float* __restrict__ pm,
                               const float* __restrict__ pl, float* __restrict__ O,
                               int nsplit) {
  const int h = blockIdx.x, d = threadIdx.x;
  float m = NEG, l = 0.0f, acc = 0.0f;
  for (int s = 0; s < nsplit; ++s) {
    const size_t slot = (size_t)h * nsplit + s;
    const float ms = pm[slot], ls = pl[slot], as = pacc[slot * D + d];
    const float mn = fmaxf(m, ms);
    const float c1 = __expf(m - mn), c2 = __expf(ms - mn);
    acc = acc * c1 + as * c2;
    l = l * c1 + ls * c2;
    m = mn;
  }
  O[(size_t)h * D + d] = acc / l;
}

// ---------------------------------------------------------------------------------------
// Variant 4: PagedAttention. The KV cache lives in a pool of fixed-size pages, and each
// sequence owns a *block table* mapping its logical page index to a physical page.
//
// The entire difference from variant 3 is this line:
//
//     size_t page = block_table[h * npages + (j0 / TILE)];
//
// — one extra load, once per page, off a table small enough to be L2-resident forever. In
// exchange the allocator never has to find S contiguous tokens' worth of memory, so there is
// no external fragmentation and no over-reservation for a sequence's *maximum* length. That
// trade is the reason vLLM exists, and the cost of it is visible right here: an indirection
// and a load that the contiguous version does not have.
//
// The pages are deliberately shuffled in main(), because a block table that happens to be the
// identity permutation tests nothing.
// ---------------------------------------------------------------------------------------
__global__ void decode_paged(const float* __restrict__ Q, const float* __restrict__ Kpool,
                             const float* __restrict__ Vpool, const int* __restrict__ table,
                             float* __restrict__ pacc, float* __restrict__ pm,
                             float* __restrict__ pl, int S, int nsplit, int npages) {
  SHARED(float, sq, D);
  SHARED(float, sK, TILE * D);
  SHARED(float, sV, TILE * D);
  SHARED(float, sp, TILE);

  const int h = blockIdx.x, sp_id = blockIdx.y, d = threadIdx.x;
  const int chunk = ((S + nsplit - 1) / nsplit + TILE - 1) / TILE * TILE;   // page-aligned
  const int begin = sp_id * chunk;
  const int end = (begin + chunk < S) ? begin + chunk : S;

  const float scale = rsqrtf((float)D);
  sq[d] = Q[(size_t)h * D + d];
  __syncthreads();

  float m = NEG, l = 0.0f, acc = 0.0f;
  for (int j0 = begin; j0 < end; j0 += TILE) {
    const int n = (end - j0 < TILE) ? (end - j0) : TILE;
    const int page = table[(size_t)h * npages + j0 / TILE];   // <- the whole of PagedAttention
    for (int r = 0; r < n; ++r) {
      size_t base = ((size_t)page * TILE + r) * D;
      sK[r * D + d] = Kpool[base + d];
      sV[r * D + d] = Vpool[base + d];
    }
    __syncthreads();
    if (d < n) {
      float s = 0.0f;
      for (int i = 0; i < D; ++i) s += sq[i] * sK[d * D + i];
      sp[d] = s * scale;
    }
    __syncthreads();

    float mt = NEG;
    for (int i = 0; i < n; ++i) mt = fmaxf(mt, sp[i]);
    const float mnew = fmaxf(m, mt);
    const float corr = __expf(m - mnew);
    float lt = 0.0f, at = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float p = __expf(sp[i] - mnew);
      lt += p;
      at += p * sV[i * D + d];
    }
    l = l * corr + lt;
    acc = acc * corr + at;
    m = mnew;
    __syncthreads();
  }

  const size_t slot = (size_t)h * nsplit + sp_id;
  pacc[slot * D + d] = acc;
  if (d == 0) {
    pm[slot] = m;        // identity element when this split was empty; see decode_split
    pl[slot] = l;
  }
}

// ---------------------------------------------------------------------------------------

// Plain attention in double, on the host. The online algorithm is algebraically identical to
// this, so the tolerance can be tight and any real drift is a bug rather than a rounding
// difference.
static void attention_ref(int H, int S, const float* Q, const float* K, const float* V,
                          float* out) {
  const double scale = 1.0 / std::sqrt((double)D);
  std::vector<double> sc(S);
  for (int h = 0; h < H; ++h) {
    double mx = -1e300;
    for (int j = 0; j < S; ++j) {
      double a = 0;
      for (int i = 0; i < D; ++i) a += (double)Q[(size_t)h * D + i] * K[((size_t)h * S + j) * D + i];
      sc[j] = a * scale;
      mx = std::max(mx, sc[j]);
    }
    double l = 0;
    for (int j = 0; j < S; ++j) { sc[j] = std::exp(sc[j] - mx); l += sc[j]; }
    for (int d = 0; d < D; ++d) {
      double a = 0;
      for (int j = 0; j < S; ++j) a += sc[j] * V[((size_t)h * S + j) * D + d];
      out[(size_t)h * D + d] = (float)(a / l);
    }
  }
}

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  int H = 2, S = 128;
#else
  int H = argc > 1 ? std::atoi(argv[1]) : 32;      // heads (batch 1 — the decode case)
  int S = argc > 2 ? std::atoi(argv[2]) : 4096;    // KV cache length
#endif
  (void)argc; (void)argv;
  S = (S / TILE) * TILE;
  const int npages = S / TILE;
  const size_t KV = (size_t)H * S * D;

  std::vector<float> hQ((size_t)H * D), hK(KV), hV(KV), hO((size_t)H * D);
  bench::fill(hQ.data(), hQ.size(), 5);
  bench::fill(hK.data(), KV, 6);
  bench::fill(hV.data(), KV, 7);

  // A block table that is a genuine permutation of the pool, per head. Page p of head h lives
  // at physical page table[h][p], and no two logical pages share a physical one.
  std::vector<int> table((size_t)H * npages);
  std::vector<float> hKpool(KV), hVpool(KV);
  {
    unsigned s = 99;
    std::vector<int> perm((size_t)H * npages);
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = (int)i;
    for (size_t i = perm.size(); i > 1; --i) {   // Fisher-Yates, deterministic
      s = s * 1664525u + 1013904223u;
      size_t j = s % i;
      std::swap(perm[i - 1], perm[j]);
    }
    for (int h = 0; h < H; ++h)
      for (int p = 0; p < npages; ++p) {
        int phys = perm[(size_t)h * npages + p];
        table[(size_t)h * npages + p] = phys;
        std::memcpy(&hKpool[(size_t)phys * TILE * D], &hK[((size_t)h * S + (size_t)p * TILE) * D],
                    (size_t)TILE * D * sizeof(float));
        std::memcpy(&hVpool[(size_t)phys * TILE * D], &hV[((size_t)h * S + (size_t)p * TILE) * D],
                    (size_t)TILE * D * sizeof(float));
      }
  }

  std::vector<float> want((size_t)H * D);
  attention_ref(H, S, hQ.data(), hK.data(), hV.data(), want.data());

  float *dQ, *dK, *dV, *dO, *dscratch, *dpacc, *dpm, *dpl, *dKp, *dVp;
  int* dtable;
  auto A = [&](void** p, size_t n) { CUDA_CHECK(cudaMalloc(p, n)); };
  A((void**)&dQ, hQ.size() * sizeof(float));
  A((void**)&dK, KV * sizeof(float));
  A((void**)&dV, KV * sizeof(float));
  A((void**)&dKp, KV * sizeof(float));
  A((void**)&dVp, KV * sizeof(float));
  A((void**)&dO, hO.size() * sizeof(float));
  A((void**)&dscratch, (size_t)H * S * sizeof(float));
  A((void**)&dpacc, (size_t)H * NSPLIT * D * sizeof(float));
  A((void**)&dpm, (size_t)H * NSPLIT * sizeof(float));
  A((void**)&dpl, (size_t)H * NSPLIT * sizeof(float));
  A((void**)&dtable, table.size() * sizeof(int));
  CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), hQ.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, hK.data(), KV * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, hV.data(), KV * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dKp, hKpool.data(), KV * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVp, hVpool.data(), KV * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dtable, table.data(), table.size() * sizeof(int), cudaMemcpyHostToDevice));

  const double kv_bytes = 2.0 * KV * sizeof(float);
  const double flops = 4.0 * (double)H * S * D;
  std::vector<bench::Row> rows;

  auto run = [&](const char* name, double traffic, const char* note, auto&& launch) {
    CUDA_CHECK(cudaMemset(dO, 0, hO.size() * sizeof(float)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev, 30, 10);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, hO.size() * sizeof(float), cudaMemcpyDeviceToHost));
    r.err = bench::max_rel_err(hO.data(), want.data(), hO.size());
    r.checksum = bench::checksum_of(hO);
    r.bytes = traffic;
    r.flops = flops;
    r.note = note;
    rows.push_back(r);
  };

  std::printf("problem   : %d heads, KV length %d, head dim %d, tile/page %d, %d splits\n", H, S,
              D, TILE, NSPLIT);
  std::printf("KV cache  : %.1f MB fp32, minimum traffic %.1f MB per decode step\n",
              kv_bytes / 1048576.0, kv_bytes / 1048576.0);
  bench::header(dev);

  run("1 two-pass, global scores", kv_bytes + 3.0 * (double)H * S * sizeof(float),
      "S floats of scratch per head", [&] {
        KERNEL_LAUNCH(decode_two_pass, dim3(H), dim3(D), 0, dQ, dK, dV, dscratch, dO, S);
      });
  run("2 online softmax, 1 block/head", kv_bytes, "1 block per head", [&] {
    KERNEL_LAUNCH(decode_online, dim3(H), dim3(D), 0, dQ, dK, dV, dO, S);
  });
  run("3 + split-K (FlashDecoding)", kv_bytes, "NSPLIT x more blocks", [&] {
    KERNEL_LAUNCH(decode_split, dim3(H, NSPLIT), dim3(D), 0, dQ, dK, dV, dpacc, dpm, dpl, S,
                  NSPLIT);
    KERNEL_LAUNCH(decode_combine, dim3(H), dim3(D), 0, dpacc, dpm, dpl, dO, NSPLIT);
  });
  run("4 + paged KV (PagedAttention)", kv_bytes, "+ block-table indirection", [&] {
    KERNEL_LAUNCH(decode_paged, dim3(H, NSPLIT), dim3(D), 0, dQ, dKp, dVp, dtable, dpacc, dpm,
                  dpl, S, NSPLIT, npages);
    KERNEL_LAUNCH(decode_combine, dim3(H), dim3(D), 0, dpacc, dpm, dpl, dO, NSPLIT);
  });

  // Online softmax is algebraically exact, so the only error is fp32 summation order. If a
  // variant drifts past this it has a bug, not a rounding difference.
  const double tol = 1e-4;
  bool edge_ok = true;
  bench::rows_out(rows, dev, tol);

  std::printf(
      "\nAll four compute the same function.\n"
      "  1 vs 2: only %.1f%% more traffic (3/(2·D) — the score vector is small next to the KV\n"
      "          cache), but it needs %.2f MB of scratch that scales with context length.\n"
      "          The dramatic version of this argument is prefill, where the intermediate is\n"
      "          N x N rather than N; see Attention_Kernels_From_Scratch.ipynb.\n"
      "  2 vs 3: identical traffic. The difference is occupancy — %d blocks versus %d, on a\n"
      "          GPU that wants hundreds. This is the one that matters at decode time.\n"
      "  3 vs 4: identical traffic and occupancy, plus one block-table load per page. That is\n"
      "          what paging costs; what it buys is not needing %.1f MB contiguous.\n",
      100.0 * 3.0 / (2.0 * D), (double)H * S * sizeof(float) / 1048576.0, H, H * NSPLIT,
      kv_bytes / 1048576.0 / H);

  // -------------------------------------------------------------------------------------
  // Edge case: a sequence shorter than the splits can divide.
  //
  // A 20-token conversation has one page. Split it NSPLIT ways with page-aligned chunks and
  // split 0 gets everything while splits 1..NSPLIT-1 get nothing at all. Those blocks still
  // run, still write a slot, and the combine kernel still reads it — so an empty split must
  // publish the *identity* of the merge (m = NEG, ℓ = 0, acc = 0), not whatever was in the
  // buffer from the last request.
  //
  // This is not a contrived input. Short sequences are the common case in chat serving, and
  // a scheduler that sizes its split count for the longest sequence in the batch hands every
  // short one to this path. It is also where the NEG sentinel earns its keep: two -inf maxima
  // meeting in the combine loop give exp(-inf - -inf) = NaN.
  // -------------------------------------------------------------------------------------
  {
    const int S2 = TILE;                       // exactly one page: NSPLIT-1 splits get nothing
    std::vector<float> hK2((size_t)H * S2 * D), hV2((size_t)H * S2 * D),
        want2((size_t)H * D), got2((size_t)H * D);
    std::vector<int> table2((size_t)H * 1);
    for (int h = 0; h < H; ++h) {
      std::memcpy(&hK2[(size_t)h * S2 * D], &hK[(size_t)h * S * D],
                  (size_t)S2 * D * sizeof(float));
      std::memcpy(&hV2[(size_t)h * S2 * D], &hV[(size_t)h * S * D],
                  (size_t)S2 * D * sizeof(float));
      table2[h] = H - 1 - h;                   // not the identity permutation
    }
    std::vector<float> pool2K((size_t)H * S2 * D), pool2V((size_t)H * S2 * D);
    for (int h = 0; h < H; ++h) {
      std::memcpy(&pool2K[(size_t)table2[h] * TILE * D], &hK2[(size_t)h * S2 * D],
                  (size_t)TILE * D * sizeof(float));
      std::memcpy(&pool2V[(size_t)table2[h] * TILE * D], &hV2[(size_t)h * S2 * D],
                  (size_t)TILE * D * sizeof(float));
    }
    attention_ref(H, S2, hQ.data(), hK2.data(), hV2.data(), want2.data());

    CUDA_CHECK(cudaMemcpy(dK, hK2.data(), hK2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV2.data(), hV2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKp, pool2K.data(), pool2K.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVp, pool2V.data(), pool2V.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dtable, table2.data(), table2.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Poison the partial buffers first: a kernel that fails to write an empty split's slot
    // must not accidentally pass because the buffer happened to hold zeros.
    std::vector<float> poison((size_t)H * NSPLIT * D, 1e9f);
    CUDA_CHECK(cudaMemcpy(dpacc, poison.data(), poison.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpm, poison.data(), (size_t)H * NSPLIT * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpl, poison.data(), (size_t)H * NSPLIT * sizeof(float),
                          cudaMemcpyHostToDevice));

    KERNEL_LAUNCH(decode_split, dim3(H, NSPLIT), dim3(D), 0, dQ, dK, dV, dpacc, dpm, dpl, S2,
                  NSPLIT);
    KERNEL_LAUNCH(decode_combine, dim3(H), dim3(D), 0, dpacc, dpm, dpl, dO, NSPLIT);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got2.data(), dO, got2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double e_split = bench::max_rel_err(got2.data(), want2.data(), got2.size());

    CUDA_CHECK(cudaMemcpy(dpacc, poison.data(), poison.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    KERNEL_LAUNCH(decode_paged, dim3(H, NSPLIT), dim3(D), 0, dQ, dKp, dVp, dtable, dpacc, dpm,
                  dpl, S2, NSPLIT, 1);
    KERNEL_LAUNCH(decode_combine, dim3(H), dim3(D), 0, dpacc, dpm, dpl, dO, NSPLIT);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(got2.data(), dO, got2.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double e_paged = bench::max_rel_err(got2.data(), want2.data(), got2.size());

    std::printf("\nshort-sequence check (S=%d, one page, %d splits — %d of them empty):\n"
                "  split-K  max rel err %.2e  %s\n"
                "  paged    max rel err %.2e  %s\n",
                S2, NSPLIT, NSPLIT - 1, e_split, e_split <= tol ? "ok" : "WRONG", e_paged,
                e_paged <= tol ? "ok" : "WRONG");
    if (!(e_split <= tol) || !(e_paged <= tol)) edge_ok = false;
  }

  for (void* p : {(void*)dQ, (void*)dK, (void*)dV, (void*)dKp, (void*)dVp, (void*)dO,
                  (void*)dscratch, (void*)dpacc, (void*)dpm, (void*)dpl, (void*)dtable})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev) || !edge_ok;
}
