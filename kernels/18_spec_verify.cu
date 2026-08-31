// 18_spec_verify.cu — accepting or rejecting a draft, and why speculation is unusually good at
// exactly the thing agents spend their output tokens on.
//
//     nvcc -O3 -arch=native 18_spec_verify.cu -o build/18 && build/18
//     make check
//
// Speculative decoding: a small draft model proposes k tokens, the big model scores all k+1
// positions in **one** forward pass, and verification keeps the longest leading run the big
// model agrees with. One weight read, up to k+1 tokens out. The weight read is the entire cost
// of a memory-bound decode step, so the speedup is the acceptance length.
//
// Why this belongs in the agent set rather than the general serving set:
//
//   1. An agent's output is a **tool call**, and a tool call is the most predictable text a
//      model ever emits. `{"name": "search", "arguments": {"query": ` is fixed by the schema
//      before the model has decided anything. A tiny draft model — or a lookup table, or the
//      grammar itself — gets those right almost every time. Acceptance rates that would be
//      optimistic for prose are conservative here.
//
//   2. Agents run at **temperature 0** for reliability, and greedy verification is both the
//      cheapest and the strictest form: accept exactly where the target's argmax equals the
//      draft token. No sampling, no rejection-sampling correction, no dependence on the
//      draft's distribution.
//
//   3. It is where 14_logit_mask stops being a rounding error. That kernel's whole point was
//      that a grammar mask is 0.05% of a decode step. Speculation runs the mask **k+1 times**
//      per weight read, so its share is multiplied by k+1 while the weights are read once.
//      Variant 3 below applies the mask and the arithmetic is at the bottom.
//
// Verification itself is not free of interest. The obvious implementation walks positions in
// order and stops at the first mismatch, which is a serial chain of dependent argmaxes over a
// 128k vocabulary. The parallel one computes every position's argmax at once and then reduces
// to find the first mismatch — more total work, one dependent step of latency, and the
// mismatched positions were going to be thrown away in either case.
//
// Prerequisite: 14_logit_mask.cu, whose block_argmax and mask representations this reuses.
#include "common.cuh"

#if SHIM_BUILD
constexpr int VOCAB = 1024;
constexpr int K = 4;            // draft tokens proposed
constexpr int BLOCK = 32;
#else
constexpr int VOCAB = 128000;
constexpr int K = 5;
constexpr int BLOCK = 256;
#endif
constexpr int POS = K + 1;      // k drafted positions plus the free bonus token

// ---------------------------------------------------------------------------------------
// A block-wide argmax with a **lower-index tie-break**, so two implementations that disagree
// only about ties still produce the same token. Ties are not hypothetical: a masked vocabulary
// has long runs of identical -inf, and "whichever thread got there first" is a
// non-reproducible sampler.
// ---------------------------------------------------------------------------------------
__device__ inline void block_argmax(const float* row, int n, float* sval, int* sidx,
                                    float* best_v, int* best_i) {
  float bv = -INFINITY;
  int bi = n;
  for (int v = threadIdx.x; v < n; v += blockDim.x) {
    const float x = row[v];
    if (x > bv || (x == bv && v < bi)) { bv = x; bi = v; }
  }
  sval[threadIdx.x] = bv;
  sidx[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float o = sval[threadIdx.x + s];
      const int oi = sidx[threadIdx.x + s];
      if (o > sval[threadIdx.x] || (o == sval[threadIdx.x] && oi < sidx[threadIdx.x])) {
        sval[threadIdx.x] = o;
        sidx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  *best_v = sval[0];
  *best_i = sidx[0];
}

// ---------------------------------------------------------------------------------------
// Variant 1 — serial verification. One block, positions walked in order, stopping at the
// first mismatch.
//
// This is the textbook description of the algorithm turned directly into code, and it is
// correct. It is also a chain of POS dependent argmaxes, each of which reads a 128k-entry
// logit row — so its latency is POS x (a full vocabulary scan), all of it on the critical
// path of the decode step it is supposed to be accelerating.
// ---------------------------------------------------------------------------------------
__global__ void verify_serial(const float* __restrict__ logits, const int* __restrict__ draft,
                              int* __restrict__ out_tokens, int* __restrict__ n_accepted) {
  SHARED(float, sval, BLOCK);
  SHARED(int, sidx, BLOCK);

  int n = 0;
  for (int p = 0; p < POS; ++p) {
    float bv;
    int bi;
    block_argmax(logits + (size_t)p * VOCAB, VOCAB, sval, sidx, &bv, &bi);
    __syncthreads();
    if (threadIdx.x == 0) out_tokens[p] = bi;
    // Position K has no draft token to compare against: it is the bonus token, always emitted.
    if (p < K) {
      if (bi != draft[p]) break;      // rejected here; everything after is discarded
      ++n;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *n_accepted = n;
}

// ---------------------------------------------------------------------------------------
// Variant 2 — every position argmaxed at once, then a min-reduction over the mismatches.
//
// Total work goes up: the positions after the first rejection are computed and thrown away.
// Latency goes down to one argmax plus a reduction, because the positions are independent —
// the big model already computed all POS logit rows in a single forward, so nothing about
// position p+1 depends on the verdict at p.
//
// The reduction operator is min over integers: associative, commutative, and exact. That
// matters more than it looks. A verifier whose answer depends on block scheduling would make
// the *number of tokens emitted* non-deterministic, and an agent replaying a trajectory would
// diverge on the first step where the timing differed.
// ---------------------------------------------------------------------------------------
__global__ void argmax_all(const float* __restrict__ logits, int* __restrict__ tokens) {
  SHARED(float, sval, BLOCK);
  SHARED(int, sidx, BLOCK);
  const int p = blockIdx.x;
  float bv;
  int bi;
  block_argmax(logits + (size_t)p * VOCAB, VOCAB, sval, sidx, &bv, &bi);
  if (threadIdx.x == 0) tokens[p] = bi;
}

__global__ void first_mismatch(const int* __restrict__ tokens, const int* __restrict__ draft,
                               int* __restrict__ n_accepted) {
  SHARED(int, red, BLOCK);
  int local = K;
  for (int p = threadIdx.x; p < K; p += blockDim.x)
    if (tokens[p] != draft[p] && p < local) local = p;
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = min(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) *n_accepted = red[0];
}

// ---------------------------------------------------------------------------------------
// Variant 3 — the same, with the grammar mask applied at every position.
//
// This is what an agent actually runs: a tool call is being generated under a JSON schema, so
// every one of the POS positions is constrained. The mask is a bitset, as 14_logit_mask
// established — 1 bit per token instead of a float — and it is applied inside the argmax scan
// rather than as a separate pass over the logits, so it costs no extra traffic on the logits
// themselves.
//
// The masks differ per position, because the grammar advances: after `{"name":` only string
// tokens are legal, after `"search"` only `,` or `}`. That is the part a serving stack
// computes on the CPU, and the part that actually costs milliseconds.
// ---------------------------------------------------------------------------------------
__global__ void argmax_masked(const float* __restrict__ logits,
                              const unsigned int* __restrict__ bitset, int* __restrict__ tokens) {
  SHARED(float, sval, BLOCK);
  SHARED(int, sidx, BLOCK);
  const int p = blockIdx.x;
  const float* row = logits + (size_t)p * VOCAB;
  const unsigned int* mask = bitset + (size_t)p * ((VOCAB + 31) / 32);

  float bv = -INFINITY;
  int bi = VOCAB;
  for (int v = threadIdx.x; v < VOCAB; v += blockDim.x) {
    if (!((mask[v >> 5] >> (v & 31)) & 1u)) continue;   // illegal under the grammar
    const float x = row[v];
    if (x > bv || (x == bv && v < bi)) { bv = x; bi = v; }
  }
  sval[threadIdx.x] = bv;
  sidx[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float o = sval[threadIdx.x + s];
      const int oi = sidx[threadIdx.x + s];
      if (o > sval[threadIdx.x] || (o == sval[threadIdx.x] && oi < sidx[threadIdx.x])) {
        sval[threadIdx.x] = o;
        sidx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) tokens[p] = sidx[0];
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
  (void)argc; (void)argv;

  const int WORDS = (VOCAB + 31) / 32;
  std::vector<float> logits((size_t)POS * VOCAB);
  bench::fill(logits.data(), logits.size(), 7);

  // Plant an EXACT tie for the maximum in every row: two entries, bit-identical, both the
  // largest. Random floats essentially never tie, so without this the tie-break rule is
  // untested — and mutation testing says so: flipping the comparison to `>=` passed happily
  // until this existed. A masked vocabulary is full of exact ties, and a sampler that resolves
  // them by "whichever thread got there first" is not reproducible.
  for (int p = 0; p < POS; ++p) {
    float mx = -1e30f;
    for (int v = 0; v < VOCAB; ++v) mx = std::max(mx, logits[(size_t)p * VOCAB + v]);
    // The low index is deliberately >= 16 within its 32-bit mask word. A masked argmax that
    // computes its bit position with `v & 15` instead of `v & 31` then looks at the wrong bit
    // and skips the true winner — a bug mutation testing found hiding behind ties planted at
    // low indices, where the two expressions happen to agree.
    const int lo = 20 + 32 * p, hi = VOCAB - 17 - p;
    logits[(size_t)p * VOCAB + lo] = mx + 1.0f;
    logits[(size_t)p * VOCAB + hi] = mx + 1.0f;   // identical: the tie-break decides
  }

  // The grammar mask, one per position. Deliberately tight — a tool call under a fixed schema
  // usually has a handful of legal continuations, not thousands.
  std::vector<unsigned int> bitset((size_t)POS * WORDS, 0u);
  std::vector<int> allowed_count(POS, 0);
  for (int p = 0; p < POS; ++p) {
    unsigned s = 1234u + 77u * p;
    for (int v = 0; v < VOCAB; ++v) {
      s = s * 1664525u + 1013904223u;
      // ~3% of the vocabulary legal, plus a guaranteed floor so no position is empty.
      const bool tied = (v == 20 + 32 * p) || (v == VOCAB - 17 - p);
      if ((s >> 8) % 100u < 3u || v % (VOCAB / 8) == 0 || tied)
        bitset[(size_t)p * WORDS + (v >> 5)] |= 1u << (v & 31);
    }
    // Isolate the winning token inside its mask word, so that reading the wrong bit of that
    // word cannot accidentally still find a set bit.
    const int lo = 20 + 32 * p;
    bitset[(size_t)p * WORDS + (lo >> 5)] = 1u << (lo & 31);
  }
  for (int p = 0; p < POS; ++p)
    for (int v = 0; v < VOCAB; ++v)
      if ((bitset[(size_t)p * WORDS + (v >> 5)] >> (v & 31)) & 1u) ++allowed_count[p];

  // Reference argmaxes, in double, both unmasked and masked.
  std::vector<int> want_tok(POS), want_tok_masked(POS);
  for (int p = 0; p < POS; ++p) {
    double bv = -1e300, bvm = -1e300;
    int bi = VOCAB, bim = VOCAB;
    for (int v = 0; v < VOCAB; ++v) {
      const double x = logits[(size_t)p * VOCAB + v];
      if (x > bv) { bv = x; bi = v; }
      if ((bitset[(size_t)p * WORDS + (v >> 5)] >> (v & 31)) & 1u)
        if (x > bvm) { bvm = x; bim = v; }
    }
    want_tok[p] = bi;
    want_tok_masked[p] = bim;
  }

  // The draft. Make it agree with the target for the first ACCEPT positions and diverge after,
  // which is what a real accept/reject boundary looks like.
  const int ACCEPT = K - 2 >= 0 ? K - 2 : 0;
  std::vector<int> draft(K);
  for (int p = 0; p < K; ++p)
    draft[p] = (p < ACCEPT) ? want_tok[p] : (want_tok[p] + 1) % VOCAB;

  int want_n = 0;
  while (want_n < K && draft[want_n] == want_tok[want_n]) ++want_n;

  // Under the mask the target's choice changes, so the same draft is accepted differently.
  std::vector<int> draft_masked(K);
  for (int p = 0; p < K; ++p)
    draft_masked[p] = (p < ACCEPT) ? want_tok_masked[p] : (want_tok_masked[p] + 1) % VOCAB;
  int want_n_masked = 0;
  while (want_n_masked < K && draft_masked[want_n_masked] == want_tok_masked[want_n_masked])
    ++want_n_masked;

  float* dlogits;
  unsigned int* dbitset;
  int *ddraft, *ddraft_m, *dtokens, *dn;
  CUDA_CHECK(cudaMalloc((void**)&dlogits, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dbitset, bitset.size() * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc((void**)&ddraft, K * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&ddraft_m, K * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dtokens, POS * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dn, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dlogits, logits.data(), logits.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dbitset, bitset.data(), bitset.size() * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ddraft, draft.data(), K * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ddraft_m, draft_masked.data(), K * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<bench::Row> rows;
  std::vector<int> tokens(POS);
  int got_n = 0;

  // The check is on the accepted count AND the emitted ids, exactly — a verifier that emits a
  // different token has not rounded differently, it has produced a different conversation.
  auto record = [&](const char* name, const std::vector<int>& want_ids, int want_count,
                    double bytes, const char* note) {
    double err = bench::max_rel_err_scalar((float)got_n, (float)want_count);
    for (int p = 0; p <= want_count && p < POS; ++p)
      err = std::max(err, bench::max_rel_err_scalar((float)tokens[p], (float)want_ids[p]));
    bench::Row r;
    r.name = name;
    r.err = err;
    std::vector<int> sig(tokens.begin(), tokens.begin() + POS);
    sig.push_back(got_n);
    r.checksum = bench::checksum_of(sig);
    r.bytes = bytes;
    r.flops = 0;
    r.note = note;
    rows.push_back(r);
  };

  const double logit_bytes = (double)POS * VOCAB * sizeof(float);
  const double mask_bytes = (double)POS * WORDS * sizeof(unsigned int);

  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(verify_serial, dim3(1), dim3(BLOCK), 0, dlogits, ddraft, dtokens, dn);
    }, dev, 20, 5);
    CUDA_CHECK(cudaMemcpy(tokens.data(), dtokens, POS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&got_n, dn, sizeof(int), cudaMemcpyDeviceToHost));
    record("1 serial, stop at first reject", want_tok, want_n,
           (double)(want_n + 1) * VOCAB * sizeof(float), "POS dependent argmaxes");
    rows.back().st = st;
  }
  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(argmax_all, dim3(POS), dim3(BLOCK), 0, dlogits, dtokens);
      KERNEL_LAUNCH(first_mismatch, dim3(1), dim3(BLOCK), 0, dtokens, ddraft, dn);
    }, dev, 20, 5);
    CUDA_CHECK(cudaMemcpy(tokens.data(), dtokens, POS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&got_n, dn, sizeof(int), cudaMemcpyDeviceToHost));
    record("2 all positions at once", want_tok, want_n, logit_bytes, "one step of depth");
    rows.back().st = st;
  }
  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(argmax_masked, dim3(POS), dim3(BLOCK), 0, dlogits, dbitset, dtokens);
      KERNEL_LAUNCH(first_mismatch, dim3(1), dim3(BLOCK), 0, dtokens, ddraft_m, dn);
    }, dev, 20, 5);
    CUDA_CHECK(cudaMemcpy(tokens.data(), dtokens, POS * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&got_n, dn, sizeof(int), cudaMemcpyDeviceToHost));
    record("3 + grammar mask, per position", want_tok_masked, want_n_masked,
           logit_bytes + mask_bytes, "what an agent runs");
    rows.back().st = st;
  }

  double avg_allowed = 0;
  for (int p = 0; p < POS; ++p) avg_allowed += allowed_count[p];
  avg_allowed /= POS;

  std::printf("problem   : a %d-token draft verified against the target's %d logit rows\n",
              K, POS);
  std::printf("grammar   : %.1f%% of the %d-token vocabulary legal on average\n",
              100.0 * avg_allowed / VOCAB, VOCAB);
  std::printf("outcome   : %d of %d draft tokens accepted, plus the bonus token\n", want_n, K);
  std::printf("ties      : every row has two bit-identical maxima, so the lower-index\n"
              "            tie-break is exercised rather than assumed\n");
  bench::header(dev);
  const double tol = 1e-6;
  bench::rows_out(rows, dev, tol);

  std::printf("\nEvery variant is checked on the accepted count *and* the emitted token ids,\n"
              "against argmaxes computed in double. A verifier that picks a different token has\n"
              "not rounded differently — it has produced a different conversation.\n");

  // -- what acceptance buys ----------------------------------------------------------------
  std::printf("\nExpected tokens per weight read, E = (1 - a^(k+1)) / (1 - a):\n\n");
  std::printf("  %-37s", "what is being drafted    a");
  for (int k : {1, 2, 3, 4, 6, 8}) std::printf(" %7d", k);
  std::printf("\n");
  struct { const char* what; double a; } regimes[] = {
      {"prose", 0.60},
      {"code", 0.75},
      {"JSON tool call", 0.90},
      {"a tool call's fixed scaffolding", 0.97},
  };
  for (auto& g : regimes) {
    std::printf("  %-32s %.2f", g.what, g.a);
    for (int k : {1, 2, 3, 4, 6, 8}) {
      const double e = (1.0 - std::pow(g.a, k + 1)) / (1.0 - g.a);
      std::printf(" %7.2f", e);
    }
    std::printf("\n");
  }
  std::printf("\n  Read the bottom two rows against the top one. Speculation is a modest win on\n"
              "  prose and a large one on structured output, because acceptance enters the\n"
              "  formula as a^(k+1) — a small change in a is a large change in how long a draft\n"
              "  it is worth running. An agent emits almost nothing but structured output.\n");

  // -- where 14_logit_mask's 0.05% goes ----------------------------------------------------
  std::printf("\nWhat speculation does to the mask's share of a step (Llama-3-8B, batch 32):\n\n");
  {
    const double weights_mb = 16000.0, kv_mb = 17179.9, head_mb = 1048.6;
    const double dense_mask_mb = 128000.0 * 4.0 * 32 / 1e6;
    std::printf("  %-34s %14s %14s %12s\n", "configuration", "step traffic", "mask traffic",
                "mask share");
    struct { const char* what; double k; double model_scale; int bitset; } cases[] = {
        {"no speculation", 0, 1.0, 0},
        {"k=5 draft", 5, 1.0, 0},
        {"the 1B draft model's own step", 5, 0.125, 0},
        {"same, with a bitset mask", 5, 0.125, 1},
    };
    for (auto& c : cases) {
      const double reps = c.k + 1;
      const double mask = dense_mask_mb * reps / (c.bitset ? 32.0 : 1.0);
      const double step = weights_mb * c.model_scale + kv_mb + head_mb * c.model_scale + mask;
      std::printf("  %-34s %11.0f MB %11.2f MB %11.3f%%\n", c.what, step, mask,
                  100.0 * mask / step);
    }
  }
  std::printf("\n  14_logit_mask put the dense mask at 0.05%% of a step and said not to bother.\n"
              "  Multiplying it by k+1 takes it to 0.3%%, and running it against a draft\n"
              "  model's eighth of the weights takes it to 0.5%% — still small, but now the\n"
              "  same order as things people do optimize. The bitset takes it back to nothing,\n"
              "  for free.\n"
              "\n  The lesson is not that the first answer was wrong. It is that \"what fraction\n"
              "  of the step is this\" has no fixed answer: it is a ratio, and speculation\n"
              "  changes the denominator.\n");

  for (void* p : {(void*)dlogits, (void*)dbitset, (void*)ddraft, (void*)ddraft_m,
                  (void*)dtokens, (void*)dn})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
