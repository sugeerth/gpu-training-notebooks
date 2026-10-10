// 17_ragged_batch.cu — decode attention over a batch whose sequences are wildly different
// lengths, which is what an agent batch always is.
//
//     nvcc -O3 -arch=native 17_ragged_batch.cu -o build/17 && build/17
//     make check
//
// A chat batch is roughly uniform: everyone is a few thousand tokens into a conversation, and
// the longest sequence is maybe three times the shortest. An **agent** batch is not, because
// the batch is a mix of turn numbers. One agent is on turn 2 with 4k of context; another is on
// turn 40 with 60k, having accumulated thirty-eight tool results. They are the same workload
// at different points in its life, and they land in the same batch.
//
// That changes which inefficiency dominates.
//
//   padding to the longest.  The textbook batched-attention kernel gives every sequence the
//                            same loop bound and masks the overhang. Work is B x Lmax. With a
//                            15x spread that is mostly masked-out arithmetic on garbage.
//
//   ragged (cu_seqlens).     Give each sequence its own bound, read from a prefix-sum of the
//                            lengths. Work drops to sum(L_i) — exactly what is needed. This is
//                            the layout every serious engine uses, and it is the reason
//                            `cu_seqlens` appears in every FlashAttention signature.
//
//   ragged + split.          Removing the wasted work does not remove the *imbalance*. One
//                            block per sequence means the 60k sequence runs 15x longer than
//                            everyone else, and the step is not over until it finishes, so
//                            most of the GPU idles waiting for one block. Splitting long
//                            sequences across several blocks and merging with the online-
//                            softmax identity from 06_flash_decode makes the makespan depend
//                            on total work rather than on the longest sequence.
//
// The distinction between the last two is worth being precise about, because they are often
// conflated. Ragged fixes **total work**. Splitting fixes **critical path**. A batch can have
// no wasted work and still spend most of its time with one SM busy and the rest parked, and
// the second and third rows below differ by exactly that.
//
// Prerequisite: 06_flash_decode.cu, whose split-K merge this reuses verbatim.
#include "common.cuh"

#if SHIM_BUILD
constexpr int B = 6;          // sequences in the batch
constexpr int D = 32;         // head dim
constexpr int TILE = 8;       // keys staged per pass
constexpr int NSPLIT = 4;     // chunks a long sequence is cut into
constexpr int BLOCK = 32;
constexpr int LMAX = 96;
#else
constexpr int B = 32;
constexpr int D = 128;
constexpr int TILE = 32;
constexpr int NSPLIT = 8;
constexpr int BLOCK = 128;
constexpr int LMAX = 65536;
#endif
constexpr float NEG = -1e30f;

// ---------------------------------------------------------------------------------------
// Variant 1 — padded. Every block loops to `lmax` and masks positions past its own length.
//
// The masking is correct; it is the work that is wasted. Note also that the mask is not free:
// the loop still loads the K and V tiles it is about to discard, because the load happens
// before the bound is known at tile granularity.
// ---------------------------------------------------------------------------------------
__global__ void attend_padded(const float* __restrict__ Q, const float* __restrict__ K,
                              const float* __restrict__ V, const int* __restrict__ len,
                              float* __restrict__ out, int lmax, int kv_stride) {
  SHARED(float, sq, D);
  SHARED(float, sK, TILE * D);
  SHARED(float, sV, TILE * D);
  SHARED(float, sp, TILE);

  const int b = blockIdx.x, d = threadIdx.x;
  const float scale = rsqrtf((float)D);
  sq[d] = Q[(size_t)b * D + d];
  __syncthreads();

  const int L = len[b];
  float m = NEG, l = 0.0f, acc = 0.0f;
  for (int j0 = 0; j0 < lmax; j0 += TILE) {
    for (int r = 0; r < TILE; ++r) {
      const size_t base = ((size_t)b * kv_stride + j0 + r) * D;
      sK[r * D + d] = (j0 + r < lmax) ? K[base + d] : 0.0f;
      sV[r * D + d] = (j0 + r < lmax) ? V[base + d] : 0.0f;
    }
    __syncthreads();
    if (d < TILE) {
      const int j = j0 + d;
      if (j < L) {
        float s = 0.0f;
        for (int i = 0; i < D; ++i) s += sq[i] * sK[d * D + i];
        sp[d] = s * scale;
      } else {
        sp[d] = NEG;              // past this sequence's end: masked out
      }
    }
    __syncthreads();

    float mt = NEG;
    for (int i = 0; i < TILE; ++i) mt = fmaxf(mt, sp[i]);
    const float mnew = fmaxf(m, mt);
    if (mnew > NEG) {
      const float corr = __expf(m - mnew);
      float lt = 0.0f, at = 0.0f;
      for (int i = 0; i < TILE; ++i) {
        const float p = __expf(sp[i] - mnew);
        lt += p;
        at += p * sV[i * D + d];
      }
      l = l * corr + lt;
      acc = acc * corr + at;
      m = mnew;
    }
    __syncthreads();
  }
  out[(size_t)b * D + d] = acc / l;
}

// ---------------------------------------------------------------------------------------
// Variant 2 — ragged. `cu` is the exclusive prefix sum of the lengths, so sequence b's keys
// live at [cu[b], cu[b+1]) in one packed buffer and its loop bound is its own length.
//
// Two things follow. The obvious one is that the wasted work is gone. The less obvious one is
// that the KV buffer is now packed, so there is no per-sequence stride to leave holes in the
// address space — a batch of 32 sequences averaging 20k tokens occupies what 32 x 20k needs,
// not 32 x 60k.
// ---------------------------------------------------------------------------------------
__global__ void attend_ragged(const float* __restrict__ Q, const float* __restrict__ K,
                              const float* __restrict__ V, const int* __restrict__ cu,
                              float* __restrict__ out) {
  SHARED(float, sq, D);
  SHARED(float, sK, TILE * D);
  SHARED(float, sV, TILE * D);
  SHARED(float, sp, TILE);

  const int b = blockIdx.x, d = threadIdx.x;
  const float scale = rsqrtf((float)D);
  sq[d] = Q[(size_t)b * D + d];
  __syncthreads();

  const int begin = cu[b], L = cu[b + 1] - cu[b];
  float m = NEG, l = 0.0f, acc = 0.0f;
  for (int j0 = 0; j0 < L; j0 += TILE) {
    const int n = (L - j0 < TILE) ? (L - j0) : TILE;
    for (int r = 0; r < n; ++r) {
      const size_t base = (size_t)(begin + j0 + r) * D;
      sK[r * D + d] = K[base + d];
      sV[r * D + d] = V[base + d];
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
  out[(size_t)b * D + d] = acc / l;
}

// ---------------------------------------------------------------------------------------
// Variant 3 — ragged, and every sequence split into NSPLIT chunks regardless of length.
//
// Each (sequence, chunk) pair is its own block and produces a partial (m, l, acc). The merge
// afterwards is the online-softmax identity from 06_flash_decode.cu, unchanged:
//
//     l = l1*exp(m1-m) + l2*exp(m2-m)      acc = acc1*exp(m1-m) + acc2*exp(m2-m)
//
// A fixed split count per sequence — rather than a chunk size — is what balances the batch:
// the long sequence's chunks are large and the short one's are small, but every block does
// L_b/NSPLIT work, so no block is 15x the others. The merge is the same algebra 13's cascade
// uses for a completely different partition, which is the third time this identity has earned
// its keep in this directory.
// ---------------------------------------------------------------------------------------
__global__ void attend_split(const float* __restrict__ Q, const float* __restrict__ K,
                             const float* __restrict__ V, const int* __restrict__ cu,
                             float* __restrict__ pm, float* __restrict__ pl,
                             float* __restrict__ pacc) {
  SHARED(float, sq, D);
  SHARED(float, sK, TILE * D);
  SHARED(float, sV, TILE * D);
  SHARED(float, sp, TILE);

  const int b = blockIdx.x, c = blockIdx.y, d = threadIdx.x;
  const float scale = rsqrtf((float)D);
  sq[d] = Q[(size_t)b * D + d];
  __syncthreads();

  const int begin = cu[b], L = cu[b + 1] - cu[b];
  // Split by *count*, not by size: this is what equalizes the blocks across a ragged batch.
  const int lo = (int)((long long)L * c / NSPLIT);
  const int hi = (int)((long long)L * (c + 1) / NSPLIT);

  float m = NEG, l = 0.0f, acc = 0.0f;
  for (int j0 = lo; j0 < hi; j0 += TILE) {
    const int n = (hi - j0 < TILE) ? (hi - j0) : TILE;
    for (int r = 0; r < n; ++r) {
      const size_t base = (size_t)(begin + j0 + r) * D;
      sK[r * D + d] = K[base + d];
      sV[r * D + d] = V[base + d];
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
  const size_t o = (size_t)b * NSPLIT + c;
  // An empty chunk — possible when L < NSPLIT — must contribute nothing, not NaN. It carries
  // m = -inf and l = 0, which the merge below weights to exactly zero.
  if (d == 0) { pm[o] = m; pl[o] = l; }
  pacc[o * D + d] = acc;
}

__global__ void merge_splits(const float* __restrict__ pm, const float* __restrict__ pl,
                             const float* __restrict__ pacc, float* __restrict__ out) {
  const int b = blockIdx.x, d = threadIdx.x;
  float m = NEG;
  for (int c = 0; c < NSPLIT; ++c) m = fmaxf(m, pm[(size_t)b * NSPLIT + c]);
  float l = 0.0f, acc = 0.0f;
  for (int c = 0; c < NSPLIT; ++c) {
    const size_t o = (size_t)b * NSPLIT + c;
    const float w = __expf(pm[o] - m);
    l += pl[o] * w;
    acc += pacc[o * D + d] * w;
  }
  out[(size_t)b * D + d] = acc / l;
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
  (void)argc; (void)argv;

  // A batch of agents at different points in their runs. The spread — not the mean — is what
  // this kernel is about, so the lengths are deliberately not clustered.
  std::vector<int> len(B);
  for (int b = 0; b < B; ++b) {
    // Turn number 1..B, context growing roughly linearly with it, as an agent's does.
    const double frac = (double)(b + 1) / B;
    len[b] = (int)(LMAX * (0.06 + 0.94 * frac * frac));
    if (len[b] < TILE) len[b] = TILE;
  }
  std::vector<int> cu(B + 1, 0);
  for (int b = 0; b < B; ++b) cu[b + 1] = cu[b] + len[b];
  const int total = cu[B];
  const int lmax = *std::max_element(len.begin(), len.end());
  const int lmin = *std::min_element(len.begin(), len.end());

  std::vector<float> Q((size_t)B * D), out((size_t)B * D), want((size_t)B * D);
  std::vector<float> Kr((size_t)total * D), Vr((size_t)total * D);     // ragged, packed
  bench::fill(Q.data(), Q.size(), 1);
  bench::fill(Kr.data(), Kr.size(), 2);
  bench::fill(Vr.data(), Vr.size(), 3);

  // The padded layout holds the same data at a fixed stride, with garbage past each length —
  // garbage on purpose, so a masking bug shows up as a wrong answer rather than a lucky zero.
  std::vector<float> Kp((size_t)B * lmax * D), Vp((size_t)B * lmax * D);
  bench::fill(Kp.data(), Kp.size(), 77);
  bench::fill(Vp.data(), Vp.size(), 78);
  for (int b = 0; b < B; ++b)
    for (int j = 0; j < len[b]; ++j) {
      std::memcpy(&Kp[((size_t)b * lmax + j) * D], &Kr[(size_t)(cu[b] + j) * D],
                  D * sizeof(float));
      std::memcpy(&Vp[((size_t)b * lmax + j) * D], &Vr[(size_t)(cu[b] + j) * D],
                  D * sizeof(float));
    }

  // -- reference, in double ---------------------------------------------------------------
  {
    const double scale = 1.0 / std::sqrt((double)D);
    for (int b = 0; b < B; ++b) {
      const int L = len[b];
      std::vector<double> sc(L);
      double mx = -1e300;
      for (int j = 0; j < L; ++j) {
        double a = 0;
        for (int i = 0; i < D; ++i)
          a += (double)Q[(size_t)b * D + i] * Kr[(size_t)(cu[b] + j) * D + i];
        sc[j] = a * scale;
        mx = std::max(mx, sc[j]);
      }
      double l = 0;
      for (int j = 0; j < L; ++j) { sc[j] = std::exp(sc[j] - mx); l += sc[j]; }
      for (int d = 0; d < D; ++d) {
        double o = 0;
        for (int j = 0; j < L; ++j) o += sc[j] * Vr[(size_t)(cu[b] + j) * D + d];
        want[(size_t)b * D + d] = (float)(o / l);
      }
    }
  }

  float *dQ, *dKr, *dVr, *dKp, *dVp, *dout, *dpm, *dpl, *dpacc;
  int *dlen, *dcu;
  CUDA_CHECK(cudaMalloc((void**)&dQ, Q.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dKr, Kr.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dVr, Vr.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dKp, Kp.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dVp, Vp.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dout, out.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dpm, (size_t)B * NSPLIT * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dpl, (size_t)B * NSPLIT * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dpacc, (size_t)B * NSPLIT * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dlen, len.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dcu, cu.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dKr, Kr.data(), Kr.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVr, Vr.data(), Vr.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dKp, Kp.data(), Kp.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVp, Vp.data(), Vp.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dlen, len.data(), len.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dcu, cu.data(), cu.size() * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<bench::Row> rows;
  auto finish = [&](const char* name, double keys_read, const char* note) {
    CUDA_CHECK(cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    bench::Row r;
    r.name = name;
    r.err = bench::max_rel_err(out.data(), want.data(), out.size());
    r.checksum = bench::checksum_of(out);
    r.bytes = keys_read * D * 2.0 * sizeof(float);
    r.flops = keys_read * D * 4.0;
    r.note = note;
    rows.push_back(r);
  };

  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(attend_padded, dim3(B), dim3(BLOCK), 0, dQ, dKp, dVp, dlen, dout, lmax, lmax);
    }, dev, 20, 5);
    finish("1 padded to the longest", (double)B * lmax, "reads B x Lmax");
    rows.back().st = st;
  }
  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(attend_ragged, dim3(B), dim3(BLOCK), 0, dQ, dKr, dVr, dcu, dout);
    }, dev, 20, 5);
    finish("2 ragged (cu_seqlens)", (double)total, "reads only what exists");
    rows.back().st = st;
  }
  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(attend_split, dim3(B, NSPLIT), dim3(BLOCK), 0, dQ, dKr, dVr, dcu, dpm, dpl,
                    dpacc);
      KERNEL_LAUNCH(merge_splits, dim3(B), dim3(BLOCK), 0, dpm, dpl, dpacc, dout);
    }, dev, 20, 5);
    finish("3 ragged + split, balanced", (double)total, "same bytes, shorter critical path");
    rows.back().st = st;
  }

  std::printf("problem   : %d agent sequences in one decode step, %d to %d tokens\n",
              B, lmin, lmax);
  std::printf("spread    : longest / shortest = %.1fx — a mix of turn numbers, not a mix of\n"
              "            users\n", (double)lmax / lmin);
  bench::header(dev);
  const double tol = 2e-4;
  bench::rows_out(rows, dev, tol);

  std::printf("\nAll three match a per-sequence reference computed in double, so the packing,\n"
              "the masking and the split-and-merge are each verified rather than assumed.\n");

  // -- what padding costs -----------------------------------------------------------------
  std::printf("\nWork, three ways:\n\n");
  std::printf("  %-26s %14s %12s\n", "layout", "keys read", "vs needed");
  std::printf("  %-26s %14d %11.2fx\n", "padded to Lmax", B * lmax, (double)(B * lmax) / total);
  std::printf("  %-26s %14d %11.2fx\n", "ragged", total, 1.0);
  std::printf("  %-26s %14d %11.2fx\n", "ragged + split", total, 1.0);
  std::printf("\n  Padding wastes %.0f%% of the reads here. The waste is set by the *spread*,\n"
              "  not the mean: mean/max = %.2f, and 1 - that is exactly the fraction thrown\n"
              "  away.\n", 100.0 * (1.0 - (double)total / (B * lmax)),
              (double)total / ((double)B * lmax));

  // -- and what splitting costs, which is different ---------------------------------------
  std::printf("\nCritical path, in units of \"keys one block must walk\":\n\n");
  std::printf("  %-26s %16s %14s\n", "layout", "busiest block", "vs balanced");
  const double balanced = (double)total / (B * NSPLIT);
  std::printf("  %-26s %16d %13.1fx\n", "padded", lmax, lmax / balanced);
  std::printf("  %-26s %16d %13.1fx\n", "ragged, one block/seq", lmax, lmax / balanced);
  int worst_split = 0;
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < NSPLIT; ++c) {
      const int lo = (int)((long long)len[b] * c / NSPLIT);
      const int hi = (int)((long long)len[b] * (c + 1) / NSPLIT);
      worst_split = std::max(worst_split, hi - lo);
    }
  std::printf("  %-26s %16d %13.1fx\n", "ragged + split", worst_split, worst_split / balanced);
  std::printf("\n  Note that rows 1 and 2 are identical here. Going ragged removed the wasted\n"
              "  work and left the imbalance untouched — the longest sequence still sets when\n"
              "  the step ends, and every other block finishes early and waits. Splitting is a\n"
              "  separate fix for a separate problem, and it is the one that matters once the\n"
              "  batch is ragged.\n");

  // -- how the spread grows with agent turn count -----------------------------------------
  std::printf("\nThe same accounting for workloads you might actually batch (uniform over the\n"
              "range, %d sequences):\n\n", B);
  std::printf("  %-34s %10s %10s %8s %14s\n", "batch", "shortest", "longest", "waste",
              "tokens wasted");
  struct { const char* what; double lo, hi; } mixes[] = {
      {"chat, everyone mid-conversation", 2000, 6000},
      {"chat, some long some short", 500, 16000},
      {"agents, turns 1-10", 3700, 12250},
      {"agents, turns 1-40", 3700, 40750},
      {"agents, turns 1-40, 2k tool results", 3700, 87700},
  };
  for (auto& m : mixes) {
    // Uniform over [lo, hi]: mean is the midpoint, so waste = 1 - mean/max.
    const double mean = 0.5 * (m.lo + m.hi);
    std::printf("  %-34s %10.0f %10.0f %7.0f%% %14.0f\n", m.what, m.lo, m.hi,
                100.0 * (1.0 - mean / m.hi), B * (m.hi - mean));
  }
  std::printf("\n  The *percentage* barely moves: any spread wide enough to matter lands\n"
              "  between a third and a half, because for a uniform mix the waste is\n"
              "  1 - mean/max, which saturates fast. Chat is not exempt, and a table of\n"
              "  ratios would have said the opposite of the truth here.\n"
              "\n  What separates the workloads is the last column. The same 45%% is 45%% of\n"
              "  4k tokens in one case and of 40k in another — an order of magnitude more\n"
              "  bytes moved per step, from the same mistake. The critical path above scales\n"
              "  the same way: the longest sequence sets when the step ends, and in an agent\n"
              "  batch it is long because somebody is on turn 40, not because somebody typed a\n"
              "  lot. Turn count has no natural ceiling; message length does.\n");

  for (void* p : {(void*)dQ, (void*)dKr, (void*)dVr, (void*)dKp, (void*)dVp, (void*)dout,
                  (void*)dpm, (void*)dpl, (void*)dpacc, (void*)dlen, (void*)dcu})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
