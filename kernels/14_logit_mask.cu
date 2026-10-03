// 14_logit_mask.cu — constrained decoding: the kernel an agent runs on almost every token,
// and the one nobody counts.
//
//     nvcc -O3 -arch=native 14_logit_mask.cu -o build/14 && build/14
//     make check
//
// An agent's output is not prose. It is a tool call: JSON with a fixed schema, a known set of
// function names, typed arguments. Serving stacks enforce that with a grammar — at each step,
// the parser says which tokens are legal, everything else is masked to -inf, and the model
// samples from what is left.
//
// That mask is applied to the **logits**, one float per vocabulary entry per sequence, and
// modern vocabularies are large: 128k for Llama 3, 152k for Qwen, 256k for Gemma. A dense
// fp32 mask at V = 128k and batch 32 is 16.4 MB read per decode step.
//
// Which sounds alarming, and is the wrong thing to be alarmed about. This file exists partly
// to make that concrete, because it is a good example of a trap:
//
//     dense mask                      16 MB     0.05% of the step
//     model weights, 8B in bf16    16,000 MB
//     KV read, 4k ctx, batch 32    17,180 MB
//     ---------------------------------------
//     whole decode step            ~34,200 MB   ~10 ms on an H100
//
// **The mask is five microseconds of a ten-millisecond step.** The kernel your instinct says
// to optimize is not where the time is, and the program below prints that comparison rather
// than asserting it. If you take one thing from this file, take the habit of dividing by the
// step before optimizing anything.
//
// So why does it still deserve a kernel? Three reasons, in increasing order of importance:
//
//   1. The bitset is *free*. Same semantics, same code shape, 1/32 the bytes. There is no
//      trade to weigh — you either wrote it that way or you did not.
//   2. The fraction is not always 0.05%. Speculative decoding samples k+1 positions per step,
//      a draft model has a fraction of the weights but the same vocabulary, and a large batch
//      multiplies the mask while the weight read stays fixed. Push those together and the
//      sampling step stops being noise.
//   3. **The real cost is not on the GPU at all.** Advancing a grammar and computing the
//      allowed set is CPU work, per sequence, per step. If it is not overlapped it serializes
//      with the GPU, and then constrained decoding costs milliseconds rather than
//      microseconds — a thousand times the number this kernel is about.
//
// That third point is the one that bites in production, and no kernel fixes it. It is covered
// in Structured_Output_Guided_Decoding.ipynb.
//
// The variants below are the three representations, plus the one that matters most in
// practice: at temperature 0 — which is what most agents use for tool calls — you do not need
// the softmax at all, only the argmax over the allowed set.
//
// Prerequisite: 02_reduce.cu, for the block reductions this is built from.
#include "common.cuh"
#include <cstdint>

#if SHIM_BUILD
constexpr int VOCAB = 4096;
constexpr int BATCH = 4;
constexpr int BLOCK = 64;
#else
constexpr int VOCAB = 128000;    // Llama-3 class
constexpr int BATCH = 32;
constexpr int BLOCK = 256;
#endif
constexpr int MAXR = 8;          // ranges per sequence in the compact representation
constexpr unsigned FULL = 0xffffffffu;
constexpr float NEG = -1e30f;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = warpSize / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(FULL, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_down_sync(FULL, v, o);
  return v;
}
__device__ __forceinline__ float block_reduce(float v, float* sc, bool take_max) {
  const int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  v = take_max ? warp_max(v) : warp_sum(v);
  if (lane == 0) sc[warp] = v;
  __syncthreads();
  const int nw = blockDim.x / warpSize;
  if (warp == 0) {
    v = (lane < nw) ? sc[lane] : (take_max ? NEG : 0.0f);
    v = take_max ? warp_max(v) : warp_sum(v);
    if (lane == 0) sc[0] = v;
  }
  __syncthreads();
  const float out = sc[0];
  __syncthreads();
  return out;
}

// Argmax needs the index too, so it carries a (value, index) pair through the same tree.
// Ties break toward the lower index in every variant, which is what makes them comparable —
// an argmax that broke ties by whichever lane happened to win would be non-deterministic in
// exactly the way 07_rmsnorm_backward.cu is about.
__device__ __forceinline__ void block_argmax(float v, int i, float* sv, int* si) {
  const int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
#pragma unroll
  for (int o = warpSize / 2; o > 0; o >>= 1) {
    const float ov = __shfl_down_sync(FULL, v, o);
    const int oi = __shfl_down_sync(FULL, i, o);
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
  }
  if (lane == 0) { sv[warp] = v; si[warp] = i; }
  __syncthreads();
  const int nw = blockDim.x / warpSize;
  if (warp == 0) {
    v = (lane < nw) ? sv[lane] : NEG;
    i = (lane < nw) ? si[lane] : 0x7fffffff;
#pragma unroll
    for (int o = warpSize / 2; o > 0; o >>= 1) {
      const float ov = __shfl_down_sync(FULL, v, o);
      const int oi = __shfl_down_sync(FULL, i, o);
      if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if (lane == 0) { sv[0] = v; si[0] = i; }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------------------
// Variant 1: a dense fp32 mask, added to the logits.
//
// The obvious implementation, and the one a framework gives you for free: build a
// [batch, vocab] tensor of 0 and -inf, add, softmax. It is four bytes per vocabulary entry per
// sequence per step, and every one of them is read from HBM.
//
// Note what the mask actually contains: for a JSON grammar mid-string, almost all of it is
// zeros. For a grammar expecting one of five function names, almost all of it is -inf. Either
// way it is 128k floats of mostly-identical values, streamed every step.
// ---------------------------------------------------------------------------------------
__global__ void mask_dense(const float* __restrict__ logits, const float* __restrict__ mask,
                           int* __restrict__ tok, float* __restrict__ prob, int V) {
  SHARED(float, sc, 32);
  SHARED(int, si, 32);
  const int b = blockIdx.x;
  const float* lg = logits + (size_t)b * V;
  const float* mk = mask + (size_t)b * V;

  float mx = NEG;
  for (int i = threadIdx.x; i < V; i += blockDim.x) mx = fmaxf(mx, lg[i] + mk[i]);
  mx = block_reduce(mx, sc, true);

  float sum = 0.0f;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float m = lg[i] + mk[i];
    if (m > NEG) sum += __expf(m - mx);
  }
  sum = block_reduce(sum, sc, false);

  float bv = NEG;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float m = lg[i] + mk[i];
    if (m > bv || (m == bv && i < bi)) { bv = m; bi = i; }
  }
  block_argmax(bv, bi, sc, si);
  if (threadIdx.x == 0) { tok[b] = si[0]; prob[b] = __expf(sc[0] - mx) / sum; }
}

// ---------------------------------------------------------------------------------------
// Variant 2: the same mask as a bitset, one bit per token.
//
// 32x less mask traffic, and the only cost is a shift and an AND per token. The logits still
// have to be read in full — they came from the LM head and they are V floats no matter what —
// so this does not eliminate the pass, it eliminates the *second* stream running alongside it.
//
// This is the representation most serving stacks actually use, and it is worth noticing that
// it is a pure win: same semantics, same code shape, 1/32 the bytes.
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ bool allowed(const uint32_t* bits, int i) {
  return (bits[i >> 5] >> (i & 31)) & 1u;
}

__global__ void mask_bitset(const float* __restrict__ logits, const uint32_t* __restrict__ bits,
                            int* __restrict__ tok, float* __restrict__ prob, int V) {
  SHARED(float, sc, 32);
  SHARED(int, si, 32);
  const int b = blockIdx.x;
  const float* lg = logits + (size_t)b * V;
  const uint32_t* bt = bits + (size_t)b * ((V + 31) / 32);

  float mx = NEG;
  for (int i = threadIdx.x; i < V; i += blockDim.x)
    if (allowed(bt, i)) mx = fmaxf(mx, lg[i]);
  mx = block_reduce(mx, sc, true);

  float sum = 0.0f;
  for (int i = threadIdx.x; i < V; i += blockDim.x)
    if (allowed(bt, i)) sum += __expf(lg[i] - mx);
  sum = block_reduce(sum, sc, false);

  float bv = NEG;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < V; i += blockDim.x)
    if (allowed(bt, i)) {
      const float m = lg[i];
      if (m > bv || (m == bv && i < bi)) { bv = m; bi = i; }
    }
  block_argmax(bv, bi, sc, si);
  if (threadIdx.x == 0) { tok[b] = si[0]; prob[b] = __expf(sc[0] - mx) / sum; }
}

// ---------------------------------------------------------------------------------------
// Variant 3: the allowed set as token ranges.
//
// Grammar-allowed sets are almost never scattered. "any digit", "any lowercase letter", "the
// closing quote", "one of these four function names" — after tokenization these land in a
// handful of contiguous id ranges, because tokenizers group similar strings together. A few
// (start, end) pairs describe the whole set.
//
// Now the kernel touches only the allowed logits. When the grammar is tight — and mid-tool-call
// it usually is, often a single-digit number of legal tokens — that is a hundred bytes instead
// of half a megabyte, and the sampling step effectively disappears.
//
// The catch is that it is only a win when the set is small. A grammar mid-string allows most
// of the vocabulary, the ranges cover nearly everything, and this degenerates to variant 2
// with extra bookkeeping. Production stacks pick per step, which is why the table at the end
// of this file reports both regimes.
// ---------------------------------------------------------------------------------------
__global__ void mask_ranges(const float* __restrict__ logits, const int* __restrict__ ranges,
                            const int* __restrict__ nranges, int* __restrict__ tok,
                            float* __restrict__ prob, int V) {
  SHARED(float, sc, 32);
  SHARED(int, si, 32);
  const int b = blockIdx.x;
  const float* lg = logits + (size_t)b * V;
  const int* rg = ranges + (size_t)b * MAXR * 2;
  const int nr = nranges[b];

  float mx = NEG;
  for (int r = 0; r < nr; ++r)
    for (int i = rg[2 * r] + threadIdx.x; i < rg[2 * r + 1]; i += blockDim.x)
      mx = fmaxf(mx, lg[i]);
  mx = block_reduce(mx, sc, true);

  float sum = 0.0f;
  for (int r = 0; r < nr; ++r)
    for (int i = rg[2 * r] + threadIdx.x; i < rg[2 * r + 1]; i += blockDim.x)
      sum += __expf(lg[i] - mx);
  sum = block_reduce(sum, sc, false);

  float bv = NEG;
  int bi = 0x7fffffff;
  for (int r = 0; r < nr; ++r)
    for (int i = rg[2 * r] + threadIdx.x; i < rg[2 * r + 1]; i += blockDim.x) {
      const float m = lg[i];
      if (m > bv || (m == bv && i < bi)) { bv = m; bi = i; }
    }
  block_argmax(bv, bi, sc, si);
  if (threadIdx.x == 0) { tok[b] = si[0]; prob[b] = __expf(sc[0] - mx) / sum; }
}

// ---------------------------------------------------------------------------------------
// Variant 4: greedy over the ranges — one pass, no normalization pass at all.
//
// Most agents run tool calls at temperature 0, and an argmax does not need a softmax. Fold the
// max and the argmax into one traversal and you have the whole sampling step in a single pass
// over the allowed set.
//
// (This still computes the chosen token's probability so it can be compared against the other
// variants — but that is a second pass over the allowed set only, not over the vocabulary. A
// deployment that does not log probabilities can skip it entirely.)
// ---------------------------------------------------------------------------------------
__global__ void mask_ranges_greedy(const float* __restrict__ logits,
                                   const int* __restrict__ ranges,
                                   const int* __restrict__ nranges, int* __restrict__ tok,
                                   float* __restrict__ prob, int V) {
  SHARED(float, sc, 32);
  SHARED(int, si, 32);
  const int b = blockIdx.x;
  const float* lg = logits + (size_t)b * V;
  const int* rg = ranges + (size_t)b * MAXR * 2;
  const int nr = nranges[b];

  // One traversal produces both the max and the argmax.
  float bv = NEG;
  int bi = 0x7fffffff;
  for (int r = 0; r < nr; ++r)
    for (int i = rg[2 * r] + threadIdx.x; i < rg[2 * r + 1]; i += blockDim.x) {
      const float m = lg[i];
      if (m > bv || (m == bv && i < bi)) { bv = m; bi = i; }
    }
  block_argmax(bv, bi, sc, si);
  const float mx = sc[0];
  const int arg = si[0];
  __syncthreads();

  float sum = 0.0f;
  for (int r = 0; r < nr; ++r)
    for (int i = rg[2 * r] + threadIdx.x; i < rg[2 * r + 1]; i += blockDim.x)
      sum += __expf(lg[i] - mx);
  sum = block_reduce(sum, sc, false);
  if (threadIdx.x == 0) { tok[b] = arg; prob[b] = 1.0f / sum; }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
  (void)argc; (void)argv;
  const int V = VOCAB, B = BATCH;
  const int NW = (V + 31) / 32;

  std::vector<float> logits((size_t)B * V);
  bench::fill(logits.data(), logits.size(), 3);
  for (auto& v : logits) v *= 8.0f;                 // logits are not in [-1, 1)

  // Allowed sets, per sequence, as ranges. Half the batch is mid-tool-call with a tight
  // grammar (a handful of legal tokens); half is mid-string, where almost anything goes.
  // Real traffic is a mix, and the two regimes behave completely differently.
  std::vector<int> ranges((size_t)B * MAXR * 2, 0), nranges(B, 0);
  std::vector<float> mask((size_t)B * V, NEG);
  std::vector<uint32_t> bits((size_t)B * NW, 0u);
  int tight_seqs = 0;
  size_t allowed_total = 0;
  {
    unsigned s = 4242;
    for (int b = 0; b < B; ++b) {
      const bool tight = (b % 2 == 0);
      tight_seqs += tight;
      const int nr = tight ? 3 : 2;
      nranges[b] = nr;
      int cursor = 1;
      for (int r = 0; r < nr; ++r) {
        s = s * 1664525u + 1013904223u;
        const int width = tight ? (int)(1 + s % 6)          // a few legal tokens
                                : (int)(V / 3 + s % (V / 8));  // most of the vocabulary
        s = s * 1664525u + 1013904223u;
        const int gap = (int)(1 + s % 64);
        int begin = cursor + gap;
        int end = begin + width;
        if (end > V) { end = V; begin = end > width ? end - width : 0; }
        if (begin >= end) continue;
        ranges[((size_t)b * MAXR + r) * 2] = begin;
        ranges[((size_t)b * MAXR + r) * 2 + 1] = end;
        cursor = end;
        for (int i = begin; i < end; ++i) {
          mask[(size_t)b * V + i] = 0.0f;
          bits[(size_t)b * NW + (i >> 5)] |= 1u << (i & 31);
          ++allowed_total;
        }
      }
      // Ranges must be non-overlapping and sorted for the compact variants to be equivalent
      // to the dense one; `cursor` guarantees that by construction.
    }
  }

  // Reference in double: the masked softmax's argmax and that token's probability.
  std::vector<int> want_tok(B);
  std::vector<float> want_prob(B);
  for (int b = 0; b < B; ++b) {
    double mx = -1e300;
    int arg = 0x7fffffff;
    for (int i = 0; i < V; ++i)
      if (mask[(size_t)b * V + i] == 0.0f) {
        const double v = logits[(size_t)b * V + i];
        if (v > mx || (v == mx && i < arg)) { mx = v; arg = i; }
      }
    double sum = 0.0;
    for (int i = 0; i < V; ++i)
      if (mask[(size_t)b * V + i] == 0.0f) sum += std::exp(logits[(size_t)b * V + i] - mx);
    want_tok[b] = arg;
    want_prob[b] = (float)(1.0 / sum);
  }

  float *dlg, *dmask, *dprob;
  uint32_t* dbits;
  int *dranges, *dnr, *dtok;
  CUDA_CHECK(cudaMalloc((void**)&dlg, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dmask, mask.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dbits, bits.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc((void**)&dranges, ranges.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dnr, nranges.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dtok, (size_t)B * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dprob, (size_t)B * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dlg, logits.data(), logits.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dmask, mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dbits, bits.data(), bits.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dranges, ranges.data(), ranges.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dnr, nranges.data(), nranges.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  std::vector<int> gtok(B);
  std::vector<float> gprob(B);
  std::vector<bench::Row> rows;
  const double logit_bytes = (double)B * V * sizeof(float);

  auto run = [&](const char* name, double traffic, const char* note, auto&& launch) {
    CUDA_CHECK(cudaMemset(dtok, 0, (size_t)B * sizeof(int)));
    CUDA_CHECK(cudaMemset(dprob, 0, (size_t)B * sizeof(float)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev, 20, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gtok.data(), dtok, (size_t)B * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gprob.data(), dprob, (size_t)B * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // The token id must match exactly — a sampling kernel that picks a different token has
    // not made a rounding error, it has produced a different output.
    double worst = 0;
    for (int b = 0; b < B; ++b) {
      if (gtok[b] != want_tok[b]) worst = 1e9;
      worst = std::max(worst, bench::max_rel_err_scalar(gprob[b], want_prob[b]));
    }
    r.err = worst;
    r.checksum = bench::hash_bytes(gtok.data(), gtok.size() * sizeof(int)) * 1099511628211ull
                 ^ bench::checksum_of(gprob);
    r.bytes = traffic;
    r.flops = 3.0 * allowed_total;
    r.note = note;
    rows.push_back(r);
  };

  std::printf("problem   : vocab %d, batch %d, %d sequences on a tight grammar and %d on a "
              "loose one\n", V, B, tight_seqs, B - tight_seqs);
  std::printf("allowed   : %.1f%% of the vocabulary on average\n",
              100.0 * allowed_total / ((double)B * V));
  bench::header(dev);

  run("1 dense fp32 mask", 2.0 * logit_bytes, "mask = 4 B/token", [&] {
    KERNEL_LAUNCH(mask_dense, dim3(B), dim3(BLOCK), 0, dlg, dmask, dtok, dprob, V);
  });
  run("2 bitset mask", logit_bytes + (double)B * NW * 4, "mask = 1 bit/token", [&] {
    KERNEL_LAUNCH(mask_bitset, dim3(B), dim3(BLOCK), 0, dlg, dbits, dtok, dprob, V);
  });
  run("3 token ranges", (double)allowed_total * 4 + (double)B * MAXR * 8,
      "touches allowed only", [&] {
        KERNEL_LAUNCH(mask_ranges, dim3(B), dim3(BLOCK), 0, dlg, dranges, dnr, dtok, dprob, V);
      });
  run("4 ranges, greedy (temp 0)", (double)allowed_total * 4 + (double)B * MAXR * 8,
      "no softmax pass", [&] {
        KERNEL_LAUNCH(mask_ranges_greedy, dim3(B), dim3(BLOCK), 0, dlg, dranges, dnr, dtok,
                      dprob, V);
      });

  const double tol = 1e-5;
  bench::rows_out(rows, dev, tol);

  std::printf("\nAll four pick the same token — the check requires exact agreement on the id,\n"
              "because a sampler that picks differently has not rounded, it has diverged.\n");

  std::printf("\nMask traffic per decode step, by representation and vocabulary:\n\n");
  std::printf("  %-28s %12s %12s %12s\n", "vocabulary", "dense fp32", "bitset", "ranges");
  struct Vv { const char* name; int v; };
  const Vv vocabs[] = {{"Llama-3  (128k)", 128000}, {"Qwen-2.5 (152k)", 151936},
                       {"Gemma-2  (256k)", 256000}};
  for (const Vv& vv : vocabs) {
    const double dense = (double)B * vv.v * 4, bs = (double)B * ((vv.v + 31) / 32) * 4;
    std::printf("  %-28s %9.2f MB %9.3f MB %9.0f B\n", vv.name, dense / 1048576.0,
                bs / 1048576.0, (double)B * MAXR * 8);
  }
  std::printf("\n  (batch %d. The dense column is what a framework hands you by default.)\n", B);

  // ---- and now the part that stops you optimizing the wrong kernel ---------------------
  // Computed, not asserted, because the instinct this corrects is a strong one.
  {
    const int b32 = 32, v128 = 128000, layers = 32, kvheads = 8, hdim = 128, ctx = 4096;
    const double dense = (double)b32 * v128 * 4;
    const double kv = (double)b32 * ctx * 2 * kvheads * hdim * 2 * layers;
    const double lmhead = 4096.0 * v128 * 2;
    const double weights = 8e9 * 2;
    const double step = weights + lmhead + kv;
    std::printf("\nAgainst the rest of one decode step — Llama-3-8B, batch 32, 4k context:\n\n");
    std::printf("  %-32s %12.1f MB\n", "dense fp32 mask", dense / 1e6);
    std::printf("  %-32s %12.1f MB\n", "LM head weights", lmhead / 1e6);
    std::printf("  %-32s %12.1f MB\n", "model weights (bf16)", weights / 1e6);
    std::printf("  %-32s %12.1f MB\n", "KV cache read", kv / 1e6);
    std::printf("  %-32s %12.1f MB   ~%.0f ms on an H100\n", "whole step", step / 1e6,
                1e3 * step / 3350e9);
    std::printf("\n  The dense mask is %.3f%% of the step — about %.0f microseconds. Switching\n"
                "  it to a bitset saves %.0f of those microseconds. Worth doing because it is\n"
                "  free, not because it is urgent.\n",
                100.0 * dense / step, 1e6 * dense / 3350e9,
                1e6 * (dense - (double)b32 * ((v128 + 31) / 32) * 4) / 3350e9);
  }

  std::printf("\nWhere it stops being noise:\n"
              "  * speculative decoding samples k+1 positions per step, so the mask runs k+1\n"
              "    times against one weight read\n"
              "  * a small draft model has a fraction of the weights and the same vocabulary\n"
              "  * batch scales the mask and the KV read, but NOT the weight read — so the\n"
              "    sampling step's share grows with batch size\n"
              "  * and the CPU-side grammar advance, which this kernel does not measure, is\n"
              "    milliseconds if it is not overlapped — a thousand times everything above\n");

  std::printf("\nWhy it is an agent problem at all:\n"
              "  * a chat reply is prose, and prose is unconstrained — the mask never runs\n"
              "  * an agent's output is a tool call, so the grammar is active on nearly every\n"
              "    token of nearly every turn\n"
              "  * tool calls run at temperature 0, so variant 4 applies almost always\n"
              "  * a fan-out multiplies the batch, and the mask scales with it\n");

  for (void* p : {(void*)dlg, (void*)dmask, (void*)dbits, (void*)dranges, (void*)dnr,
                  (void*)dtok, (void*)dprob})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
