// 13_prefix_attention.cu — cascade attention: N agents sharing one prefix, and why prefix
// caching is a *kernel* result and not just an allocator result.
//
//     nvcc -O3 -arch=native 13_prefix_attention.cu -o build/13 && build/13
//     make check
//
// The agent workload
// ------------------
// A chat turn is a fresh conversation with a bit of history. An agent turn is the entire
// previous conversation plus every tool result so far, re-sent, plus a few new tokens. And a
// fan-out — a planner spawning N sub-agents, or N samples of the same step — is N sequences
// that are **identical for their first P tokens** and differ only in a short suffix.
//
// That shape is not what decode attention was designed for. Run each sequence independently
// and the prefix is read N times:
//
//     independent:  N x (P + S) x D x 2 x bytes
//     cascade:      P x D x 2 x bytes  +  N x S x D x 2 x bytes
//
// At P=2048, S=256, N=16 that is 37.7 MB against 6.3 MB — a 6x cut in the traffic of the one
// kernel that is already the memory-bound part of decoding. Prefix *caching* saves recomputing
// the prefix's KV. Cascade attention saves re-*reading* it, which is a different win and the
// one that shows up on every subsequent token rather than once at admission.
//
// The mechanism is already in this directory
// ------------------------------------------
// Attend to the shared prefix once, producing a partial (m, ℓ, acc) per query; attend to each
// suffix separately, producing another; merge them with
//
//     ℓ   = ℓ₁·e^(m₁-m) + ℓ₂·e^(m₂-m)
//     acc = acc₁·e^(m₁-m) + acc₂·e^(m₂-m)
//
// which is character-for-character the identity `decode_combine` uses in 06_flash_decode.cu to
// merge split-K partials. Same algebra, different reason for splitting: there, to fill idle
// SMs; here, because a prefix is shared and a suffix is not.
//
// That is the general lesson this file exists for. The online-softmax merge lets you compute
// attention over *any* partition of the keys, in any order, and combine the pieces afterwards.
// Once you have it, "these N requests share a prefix" stops being a scheduling curiosity and
// becomes a partition you can exploit in the kernel.
//
// Prerequisite: 06_flash_decode.cu. This file assumes its online-softmax structure.
#include "common.cuh"

#if SHIM_BUILD
constexpr int NSEQ = 4;      // sequences sharing the prefix
constexpr int D = 32;        // head dim
constexpr int TILE = 16;
constexpr int NCHUNK = 4;    // prefix chunks, i.e. how many blocks share the prefix work
constexpr int BLOCK = 32;
#else
constexpr int NSEQ = 16;
constexpr int D = 128;
constexpr int TILE = 16;
constexpr int NCHUNK = 8;
constexpr int BLOCK = 128;
#endif
constexpr float NEG = -1e30f;

// ---------------------------------------------------------------------------------------
// Variant 1: independent. Every sequence attends over its own copy of [prefix ; suffix].
//
// This is what you get from an engine that has prefix *caching* — the prefix's K and V were
// computed once and stored — but no prefix-aware attention kernel. The allocator is happy;
// the memory controller is not, because those same P·D floats are streamed once per sequence
// on every single decode step.
// ---------------------------------------------------------------------------------------
__global__ void attend_independent(const float* __restrict__ Q,      // [NSEQ][D]
                                   const float* __restrict__ Kp,     // [P][D]  shared
                                   const float* __restrict__ Vp,
                                   const float* __restrict__ Ks,     // [NSEQ][S][D]
                                   const float* __restrict__ Vs,
                                   float* __restrict__ out, int P, int S) {
  SHARED(float, sq, D);
  SHARED(float, sK, TILE * D);
  SHARED(float, sV, TILE * D);
  SHARED(float, sp, TILE);

  const int b = blockIdx.x, d = threadIdx.x;
  const float scale = rsqrtf((float)D);
  sq[d] = Q[(size_t)b * D + d];
  __syncthreads();

  float m = NEG, l = 0.0f, acc = 0.0f;

  // Two passes over two different arrays, but one running (m, ℓ, acc): the online softmax does
  // not care that the keys came from separate allocations.
  for (int phase = 0; phase < 2; ++phase) {
    const int n_tok = phase == 0 ? P : S;
    const float* Kb = phase == 0 ? Kp : Ks + (size_t)b * S * D;
    const float* Vb = phase == 0 ? Vp : Vs + (size_t)b * S * D;

    for (int j0 = 0; j0 < n_tok; j0 += TILE) {
      const int n = (n_tok - j0 < TILE) ? (n_tok - j0) : TILE;
      for (int r = 0; r < n; ++r) {
        sK[r * D + d] = Kb[(size_t)(j0 + r) * D + d];
        sV[r * D + d] = Vb[(size_t)(j0 + r) * D + d];
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
  }
  out[(size_t)b * D + d] = acc / l;
}

// ---------------------------------------------------------------------------------------
// Variant 2, phase 1: the prefix, read once for all NSEQ queries.
//
// One block owns a chunk of the prefix and holds a running (m, ℓ, acc) for **every** sequence
// at once. A tile of prefix K/V is staged into shared memory and then consumed NSEQ times, so
// the HBM traffic for the prefix is independent of how many sequences share it.
//
// That is the whole trick, and it is worth being clear about what pays for it: NSEQ
// accumulators live in registers (acc, m and ℓ per sequence per thread), so the fan-out width
// this kernel can handle is set by the register file. A production cascade kernel tiles that
// dimension too; here it is a compile-time constant so the accounting stays legible.
//
// The scores for the whole tile are computed for all sequences at once into sp[NSEQ][TILE],
// which keeps it to one barrier per tile rather than one per (tile, sequence).
// ---------------------------------------------------------------------------------------
__global__ void prefix_pass(const float* __restrict__ Q, const float* __restrict__ Kp,
                            const float* __restrict__ Vp, float* __restrict__ pacc,
                            float* __restrict__ pm, float* __restrict__ pl, int P) {
  SHARED(float, sq, NSEQ * D);
  SHARED(float, sK, TILE * D);
  SHARED(float, sV, TILE * D);
  SHARED(float, sp, NSEQ * TILE);

  const int chunk = blockIdx.x, d = threadIdx.x;
  const float scale = rsqrtf((float)D);

  const int per = (P + NCHUNK - 1) / NCHUNK;
  const int begin = chunk * per;
  const int end = (begin + per < P) ? begin + per : P;

  for (int b = 0; b < NSEQ; ++b) sq[b * D + d] = Q[(size_t)b * D + d];
  __syncthreads();

  float acc[NSEQ], m[NSEQ], l[NSEQ];
#pragma unroll
  for (int b = 0; b < NSEQ; ++b) { acc[b] = 0.0f; m[b] = NEG; l[b] = 0.0f; }

  for (int j0 = begin; j0 < end; j0 += TILE) {
    const int n = (end - j0 < TILE) ? (end - j0) : TILE;
    for (int r = 0; r < n; ++r) {
      sK[r * D + d] = Kp[(size_t)(j0 + r) * D + d];      // read ONCE, used NSEQ times
      sV[r * D + d] = Vp[(size_t)(j0 + r) * D + d];
    }
    __syncthreads();

    // One thread per prefix token in the tile, computing that token's score against every
    // sequence's query. NSEQ x TILE scores, one barrier.
    if (d < n) {
      for (int b = 0; b < NSEQ; ++b) {
        float s = 0.0f;
        for (int i = 0; i < D; ++i) s += sq[b * D + i] * sK[d * D + i];
        sp[b * TILE + d] = s * scale;
      }
    }
    __syncthreads();

    for (int b = 0; b < NSEQ; ++b) {
      float mt = NEG;
      for (int i = 0; i < n; ++i) mt = fmaxf(mt, sp[b * TILE + i]);
      const float mnew = fmaxf(m[b], mt);
      const float corr = __expf(m[b] - mnew);
      float lt = 0.0f, at = 0.0f;
      for (int i = 0; i < n; ++i) {
        const float p = __expf(sp[b * TILE + i] - mnew);
        lt += p;
        at += p * sV[i * D + d];
      }
      l[b] = l[b] * corr + lt;
      acc[b] = acc[b] * corr + at;
      m[b] = mnew;
    }
    __syncthreads();
  }

  for (int b = 0; b < NSEQ; ++b) {
    const size_t slot = (size_t)chunk * NSEQ + b;
    pacc[slot * D + d] = acc[b];
    if (d == 0) { pm[slot] = m[b]; pl[slot] = l[b]; }
  }
}

// Phase 2: each sequence's own suffix. Ordinary decode attention, one block per sequence.
__global__ void suffix_pass(const float* __restrict__ Q, const float* __restrict__ Ks,
                            const float* __restrict__ Vs, float* __restrict__ sacc,
                            float* __restrict__ sm, float* __restrict__ sl, int S) {
  SHARED(float, sq, D);
  SHARED(float, sK, TILE * D);
  SHARED(float, sV, TILE * D);
  SHARED(float, sp, TILE);

  const int b = blockIdx.x, d = threadIdx.x;
  const float scale = rsqrtf((float)D);
  sq[d] = Q[(size_t)b * D + d];
  __syncthreads();

  const float* Kb = Ks + (size_t)b * S * D;
  const float* Vb = Vs + (size_t)b * S * D;
  float m = NEG, l = 0.0f, acc = 0.0f;

  for (int j0 = 0; j0 < S; j0 += TILE) {
    const int n = (S - j0 < TILE) ? (S - j0) : TILE;
    for (int r = 0; r < n; ++r) {
      sK[r * D + d] = Kb[(size_t)(j0 + r) * D + d];
      sV[r * D + d] = Vb[(size_t)(j0 + r) * D + d];
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
  sacc[(size_t)b * D + d] = acc;
  if (d == 0) { sm[b] = m; sl[b] = l; }
}

// Phase 3: merge. NCHUNK prefix partials plus one suffix partial, per sequence.
//
// This is `decode_combine` from 06_flash_decode.cu with a different set of partials fed into
// it. Nothing about the merge knows or cares that some of its inputs came from a shared
// prefix and one did not — which is exactly why the optimization is available at all.
__global__ void cascade_combine(const float* __restrict__ pacc, const float* __restrict__ pm,
                                const float* __restrict__ pl, const float* __restrict__ sacc,
                                const float* __restrict__ sm, const float* __restrict__ sl,
                                float* __restrict__ out) {
  const int b = blockIdx.x, d = threadIdx.x;
  float m = NEG, l = 0.0f, acc = 0.0f;

  for (int c = 0; c < NCHUNK; ++c) {
    const size_t slot = (size_t)c * NSEQ + b;
    const float ms = pm[slot], ls = pl[slot], as = pacc[slot * D + d];
    const float mn = fmaxf(m, ms);
    const float c1 = __expf(m - mn), c2 = __expf(ms - mn);
    acc = acc * c1 + as * c2;
    l = l * c1 + ls * c2;
    m = mn;
  }
  {
    const float ms = sm[b], ls = sl[b], as = sacc[(size_t)b * D + d];
    const float mn = fmaxf(m, ms);
    const float c1 = __expf(m - mn), c2 = __expf(ms - mn);
    acc = acc * c1 + as * c2;
    l = l * c1 + ls * c2;
    m = mn;
  }
  out[(size_t)b * D + d] = acc / l;
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  int P = 128, S = 32;
#else
  int P = argc > 1 ? std::atoi(argv[1]) : 2048;   // shared prefix: system prompt + history
  int S = argc > 2 ? std::atoi(argv[2]) : 256;    // per-branch suffix
#endif
  (void)argc; (void)argv;
  P = (P / TILE) * TILE;
  S = (S / TILE) * TILE;
  const int B = NSEQ;

  std::vector<float> hQ((size_t)B * D), hKp((size_t)P * D), hVp((size_t)P * D),
      hKs((size_t)B * S * D), hVs((size_t)B * S * D), hout((size_t)B * D);
  bench::fill(hQ.data(), hQ.size(), 1);
  bench::fill(hKp.data(), hKp.size(), 2);
  bench::fill(hVp.data(), hVp.size(), 3);
  bench::fill(hKs.data(), hKs.size(), 4);
  bench::fill(hVs.data(), hVs.size(), 5);

  // Reference in double: plain attention over the concatenation, per sequence.
  std::vector<float> want((size_t)B * D);
  {
    const double scale = 1.0 / std::sqrt((double)D);
    std::vector<double> sc(P + S);
    for (int b = 0; b < B; ++b) {
      double mx = -1e300;
      for (int j = 0; j < P + S; ++j) {
        const float* k = (j < P) ? &hKp[(size_t)j * D]
                                 : &hKs[((size_t)b * S + (j - P)) * D];
        double a = 0;
        for (int i = 0; i < D; ++i) a += (double)hQ[(size_t)b * D + i] * k[i];
        sc[j] = a * scale;
        mx = std::max(mx, sc[j]);
      }
      double l = 0;
      for (int j = 0; j < P + S; ++j) { sc[j] = std::exp(sc[j] - mx); l += sc[j]; }
      for (int d = 0; d < D; ++d) {
        double o = 0;
        for (int j = 0; j < P + S; ++j) {
          const float* v = (j < P) ? &hVp[(size_t)j * D]
                                   : &hVs[((size_t)b * S + (j - P)) * D];
          o += sc[j] * v[d];
        }
        want[(size_t)b * D + d] = (float)(o / l);
      }
    }
  }

  float *dQ, *dKp, *dVp, *dKs, *dVs, *dout, *dpacc, *dpm, *dpl, *dsacc, *dsm, *dsl;
  auto A = [&](float** p, size_t n) { CUDA_CHECK(cudaMalloc((void**)p, n * sizeof(float))); };
  A(&dQ, hQ.size()); A(&dKp, hKp.size()); A(&dVp, hVp.size());
  A(&dKs, hKs.size()); A(&dVs, hVs.size()); A(&dout, hout.size());
  A(&dpacc, (size_t)NCHUNK * B * D); A(&dpm, (size_t)NCHUNK * B); A(&dpl, (size_t)NCHUNK * B);
  A(&dsacc, (size_t)B * D); A(&dsm, B); A(&dsl, B);
  auto up = [&](float* d, const std::vector<float>& h) {
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
  };
  up(dQ, hQ); up(dKp, hKp); up(dVp, hVp); up(dKs, hKs); up(dVs, hVs);

  const double Bf = sizeof(float);
  const double indep_bytes = 2.0 * B * (P + S) * D * Bf;
  const double casc_bytes = 2.0 * P * D * Bf + 2.0 * (double)B * S * D * Bf;
  std::vector<bench::Row> rows;

  auto run = [&](const char* name, double traffic, const char* note, auto&& launch) {
    CUDA_CHECK(cudaMemset(dout, 0, hout.size() * sizeof(float)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev, 20, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hout.data(), dout, hout.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    r.err = bench::max_rel_err(hout.data(), want.data(), hout.size());
    r.checksum = bench::checksum_of(hout);
    r.bytes = traffic;
    r.flops = 4.0 * B * (P + S) * D;
    r.note = note;
    rows.push_back(r);
  };

  std::printf("problem   : %d sequences sharing a %d-token prefix, %d-token suffixes each,"
              " head dim %d\n", B, P, S, D);
  std::printf("scenario  : an agent fan-out — same system prompt, history and tool results,\n"
              "            different branch. Or one agent's N sampled continuations.\n");
  bench::header(dev);

  run("1 independent per sequence", indep_bytes, "prefix read N times", [&] {
    KERNEL_LAUNCH(attend_independent, dim3(B), dim3(BLOCK), 0, dQ, dKp, dVp, dKs, dVs, dout,
                  P, S);
  });
  run("2 cascade (prefix read once)", casc_bytes, "prefix read once", [&] {
    KERNEL_LAUNCH(prefix_pass, dim3(NCHUNK), dim3(BLOCK), 0, dQ, dKp, dVp, dpacc, dpm, dpl, P);
    KERNEL_LAUNCH(suffix_pass, dim3(B), dim3(BLOCK), 0, dQ, dKs, dVs, dsacc, dsm, dsl, S);
    KERNEL_LAUNCH(cascade_combine, dim3(B), dim3(BLOCK), 0, dpacc, dpm, dpl, dsacc, dsm, dsl,
                  dout);
  });

  const double tol = 2e-4;
  bench::rows_out(rows, dev, tol);

  std::printf("\nBoth compute the same function — checked against plain attention over the\n"
              "concatenated [prefix ; suffix] in double, so the partition-and-merge is\n"
              "verified rather than assumed.\n");
  std::printf("\nHBM traffic per decode step:\n"
              "  independent  %8.2f MB\n"
              "  cascade      %8.2f MB   (%.1fx less)\n",
              indep_bytes / 1048576.0, casc_bytes / 1048576.0, indep_bytes / casc_bytes);

  std::printf("\nHow the saving scales with fan-out (P=%d, S=%d, D=%d):\n\n", P, S, D);
  std::printf("  %8s %14s %14s %10s\n", "branches", "independent", "cascade", "saving");
  for (int n : {1, 2, 4, 8, 16, 32, 64}) {
    const double a = 2.0 * n * (P + S) * D * Bf, c = 2.0 * P * D * Bf + 2.0 * n * S * D * Bf;
    std::printf("  %8d %11.2f MB %11.2f MB %9.1fx\n", n, a / 1048576.0, c / 1048576.0, a / c);
  }
  std::printf("\n  The ratio tends to (P+S)/S as the fan-out grows — %.1fx here. The prefix\n"
              "  becomes free; only the divergent suffix costs anything.\n",
              (double)(P + S) / S);

  std::printf("\nWhy this is an agent result rather than a chat result: a chat turn's shared\n"
              "prefix is a system prompt, maybe 200 tokens against 4k of conversation. An\n"
              "agent's shared prefix is the system prompt AND the tool definitions AND every\n"
              "tool result so far — usually most of the context — and a fan-out replicates all\n"
              "of it N ways. The bigger P/S gets, the more of the step this recovers.\n");

  for (void* p : {(void*)dQ, (void*)dKp, (void*)dVp, (void*)dKs, (void*)dVs, (void*)dout,
                  (void*)dpacc, (void*)dpm, (void*)dpl, (void*)dsacc, (void*)dsm, (void*)dsl})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
