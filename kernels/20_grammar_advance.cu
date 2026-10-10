// 20_grammar_advance.cu — the millisecond that 14_logit_mask kept pointing at, and what
// happens when you stop paying it on the host.
//
//     nvcc -O3 -arch=native 20_grammar_advance.cu -o build/20 && build/20
//     make check
//
// Two kernels in this directory have now measured a grammar mask and concluded it is small.
// `14_logit_mask` put the dense mask at 0.05% of a decode step; `18_spec_verify` moved that to
// 0.5% under speculation. Both ended by pointing somewhere else: **the mask is not where the
// time is — computing which tokens are legal is.**
//
// This kernel is that claim, made checkable. It is the last piece of the agent loop that has
// been described in this repository without being measured, and describing a cost repeatedly
// without measuring it is how a number becomes folklore.
//
// The work is real. Every decode step, for every sequence, something must:
//
//   1. advance the grammar by the token just emitted — a state transition
//   2. produce the set of tokens legal in the new state, as something the sampler can read
//
// Done the obvious way, both happen on the CPU and the result is copied to the device. Step 2
// is the expensive half: a set over a 128k vocabulary, materialized per sequence per step, and
// then pushed across PCIe on the critical path between the model's forward pass and its
// sampler. The GPU waits.
//
// The fix is a change of representation rather than a faster loop. A grammar has few states and
// many tokens, so the legal-token sets can be **precomputed once per state** and kept resident
// on the device. Then the per-step traffic is not a mask at all — it is a **state id**, four
// bytes per sequence. And if the transition can be evaluated on the device too, the host sends
// nothing per step and leaves the loop entirely.
//
//     dense float mask per step      VOCAB x 4 x B bytes      16.4 MB at 128k, batch 32
//     bitset per step                VOCAB / 8 x B bytes       0.5 MB
//     state ids per step             4 x B bytes                128 bytes
//     nothing per step               0 bytes
//
// One caveat stated up front. The transition function below is a **stand-in**: `next = f(state,
// token)` evaluated arithmetically. A real grammar compiles to a sparse transition table or a
// pushdown automaton with a stack, and neither is a one-liner. What is faithful here — and what
// the measurement is about — is the *shape*: a per-state legal-token bitset that lives on the
// device, and a per-sequence state that advances once per step. Substituting a real table
// changes the transition cost and none of the traffic.
//
// Prerequisite: 14_logit_mask.cu, whose bitset and block_argmax this reuses.
#include "common.cuh"

#if SHIM_BUILD
constexpr int VOCAB = 1024;
constexpr int BATCH = 4;
constexpr int NSTATES = 16;
constexpr int BLOCK = 32;
#else
constexpr int VOCAB = 128000;
constexpr int BATCH = 32;
constexpr int NSTATES = 256;
constexpr int BLOCK = 256;
#endif
constexpr int WORDS = (VOCAB + 31) / 32;

// The stand-in transition. Deterministic, cheap, and identical on host and device so the
// reference can use it too.
__host__ __device__ inline int advance(int state, int token) {
  return (int)(((unsigned)state * 1103515245u + (unsigned)token * 12345u) % (unsigned)NSTATES);
}

// ---------------------------------------------------------------------------------------
// The argmax every variant ends with. Lower-index tie-break, as in 18 — a masked vocabulary is
// full of exact ties and "whichever thread got there first" is not a sampler you can replay.
// ---------------------------------------------------------------------------------------
__device__ inline int masked_argmax(const float* row, const unsigned int* mask, float* sval,
                                    int* sidx) {
  float bv = -INFINITY;
  int bi = VOCAB;
  for (int v = threadIdx.x; v < VOCAB; v += blockDim.x) {
    if (!((mask[v >> 5] >> (v & 31)) & 1u)) continue;
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
  return sidx[0];
}

// ---------------------------------------------------------------------------------------
// Variant 1 — a dense fp32 mask, computed on the host, copied in every step.
//
// The mask is added to the logits, which is how a framework that only exposes "add this tensor
// to the logits" makes a grammar work. VOCAB floats per sequence per step, across PCIe, on the
// critical path.
// ---------------------------------------------------------------------------------------
__global__ void sample_dense(const float* __restrict__ logits, const float* __restrict__ bias,
                             int* __restrict__ tokens) {
  SHARED(float, sval, BLOCK);
  SHARED(int, sidx, BLOCK);
  const int b = blockIdx.x;
  const float* row = logits + (size_t)b * VOCAB;
  const float* bs = bias + (size_t)b * VOCAB;

  float bv = -INFINITY;
  int bi = VOCAB;
  for (int v = threadIdx.x; v < VOCAB; v += blockDim.x) {
    const float x = row[v] + bs[v];       // -inf for anything illegal
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
  if (threadIdx.x == 0) tokens[b] = sidx[0];
}

// ---------------------------------------------------------------------------------------
// Variant 2 — the same set, as a bitset, still built on the host and copied every step.
//
// 32x less traffic for identical semantics. This is 14_logit_mask's recommendation, and it is
// still paying the host round trip — which is the part neither 14 nor 18 measured.
// ---------------------------------------------------------------------------------------
__global__ void sample_bitset(const float* __restrict__ logits,
                              const unsigned int* __restrict__ mask, int* __restrict__ tokens) {
  SHARED(float, sval, BLOCK);
  SHARED(int, sidx, BLOCK);
  const int b = blockIdx.x;
  const int t = masked_argmax(logits + (size_t)b * VOCAB, mask + (size_t)b * WORDS, sval, sidx);
  if (threadIdx.x == 0) tokens[b] = t;
}

// ---------------------------------------------------------------------------------------
// Variant 3 — the per-state bitsets live on the device; the host sends a state id.
//
// This is the change that matters, and it is a change of *representation*, not of algorithm.
// A grammar has few states and a large vocabulary, so the legal-token set for every state can
// be built once, at compile time, and left in device memory. Per step the host contributes
// four bytes per sequence.
//
// Note what happens to the read pattern as a side effect: every sequence in the batch on the
// same grammar state reads the same bitset, so the table is small, read-only and shared. On a
// real card it sits in L2 and the per-step mask traffic effectively disappears.
// ---------------------------------------------------------------------------------------
__global__ void sample_state(const float* __restrict__ logits,
                             const unsigned int* __restrict__ table,
                             const int* __restrict__ state, int* __restrict__ tokens) {
  SHARED(float, sval, BLOCK);
  SHARED(int, sidx, BLOCK);
  const int b = blockIdx.x;
  const unsigned int* mask = table + (size_t)state[b] * WORDS;
  const int t = masked_argmax(logits + (size_t)b * VOCAB, mask, sval, sidx);
  if (threadIdx.x == 0) tokens[b] = t;
}

// ---------------------------------------------------------------------------------------
// Variant 4 — the transition happens on the device too, so the host sends nothing per step.
//
// The state is advanced from the token the *previous* step emitted, in the same launch that
// samples the next one. The grammar has left the host's inner loop entirely: the CPU compiles
// the grammar once, uploads the table once, and is thereafter uninvolved until the sequence
// finishes.
//
// This is the version that removes the synchronization, and the synchronization is the real
// prize. Steps 1-3 all require the host to know the emitted token before it can compute the
// next mask, which means device -> host -> device on every single decode step. Nothing about
// the bytes explains how much that costs; it is a pipeline bubble, and it is why "the mask is
// 0.05% of the step" was never the whole story.
// ---------------------------------------------------------------------------------------
__global__ void sample_and_advance(const float* __restrict__ logits,
                                   const unsigned int* __restrict__ table,
                                   int* __restrict__ state, const int* __restrict__ prev_token,
                                   int* __restrict__ tokens) {
  SHARED(float, sval, BLOCK);
  SHARED(int, sidx, BLOCK);
  const int b = blockIdx.x;

  // Advance by the token emitted last step, then sample under the new state's mask.
  const int st = advance(state[b], prev_token[b]);
  const unsigned int* mask = table + (size_t)st * WORDS;
  const int t = masked_argmax(logits + (size_t)b * VOCAB, mask, sval, sidx);
  if (threadIdx.x == 0) {
    state[b] = st;
    tokens[b] = t;
  }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
  (void)argc; (void)argv;

  std::vector<float> logits((size_t)BATCH * VOCAB);
  bench::fill(logits.data(), logits.size(), 11);

  // Per-state legal-token bitsets, built once. Each state admits a few percent of the
  // vocabulary, which is what a JSON schema looks like partway through a tool call.
  std::vector<unsigned int> table((size_t)NSTATES * WORDS, 0u);
  std::vector<int> allowed(NSTATES, 0);
  for (int s = 0; s < NSTATES; ++s) {
    unsigned r = 7919u * (unsigned)(s + 1);
    for (int v = 0; v < VOCAB; ++v) {
      r = r * 1664525u + 1013904223u;
      if ((r >> 8) % 100u < 4u || v % (VOCAB / 4) == s % (VOCAB / 4))
        table[(size_t)s * WORDS + (v >> 5)] |= 1u << (v & 31);
    }
    // Plant an exact tie among the legal tokens, with the winner deliberately >= 16 within its
    // mask word — the two conditions 18_spec_verify learned the hard way are needed before a
    // tie-break rule or a bit-index expression is actually under test.
    const int lo = 20 + 32 * (s % 8);
    if (lo < VOCAB) table[(size_t)s * WORDS + (lo >> 5)] = 1u << (lo & 31);
    for (int v = 0; v < VOCAB; ++v)
      if ((table[(size_t)s * WORDS + (v >> 5)] >> (v & 31)) & 1u) ++allowed[s];
  }
  // Where each sequence's grammar currently is, and what it emitted last step.
  std::vector<int> state0(BATCH), prev(BATCH);
  for (int b = 0; b < BATCH; ++b) {
    state0[b] = (b * 5 + 3) % NSTATES;
    prev[b] = (b * 977 + 13) % VOCAB;
  }

  // The state each variant samples under. 1-3 are handed the state the host already advanced;
  // 4 advances it itself, so it must arrive at the same place.
  std::vector<int> state_now(BATCH);
  for (int b = 0; b < BATCH; ++b) state_now[b] = advance(state0[b], prev[b]);

  // Shape each row so that two separate rules are actually under test.
  //
  //   * an ILLEGAL token is made the global maximum, so a variant that ignores the mask picks
  //     a different token than one that respects it. Without this the masked and unmasked
  //     answers coincide and "apply the mask" is untested — which is exactly what mutation
  //     testing reported when the tie below was planted at the global maximum instead.
  //   * two LEGAL tokens are then tied just underneath it, so the lower-index tie-break
  //     decides the answer rather than the float comparison.
  for (int b = 0; b < BATCH; ++b) {
    const int s = state_now[b];
    int first = -1, last = -1, illegal = -1;
    for (int v = 0; v < VOCAB; ++v) {
      const bool legal = (table[(size_t)s * WORDS + (v >> 5)] >> (v & 31)) & 1u;
      if (legal) {
        if (first < 0) first = v;
        last = v;
      } else if (illegal < 0) {
        illegal = v;
      }
    }
    if (first < 0 || first == last || illegal < 0) continue;
    float mx = -1e30f;
    for (int v = 0; v < VOCAB; ++v) mx = std::max(mx, logits[(size_t)b * VOCAB + v]);
    logits[(size_t)b * VOCAB + illegal] = mx + 2.0f;   // the trap for an unmasked argmax
    logits[(size_t)b * VOCAB + first] = mx + 1.0f;
    logits[(size_t)b * VOCAB + last] = mx + 1.0f;      // identical: the tie-break picks `first`
  }

  // Host-built masks for variants 1 and 2, from the state the host advanced to.
  std::vector<float> bias((size_t)BATCH * VOCAB, -INFINITY);
  std::vector<unsigned int> hostmask((size_t)BATCH * WORDS, 0u);
  for (int b = 0; b < BATCH; ++b) {
    const int s = state_now[b];
    for (int v = 0; v < VOCAB; ++v)
      if ((table[(size_t)s * WORDS + (v >> 5)] >> (v & 31)) & 1u) {
        bias[(size_t)b * VOCAB + v] = 0.0f;
        hostmask[(size_t)b * WORDS + (v >> 5)] |= 1u << (v & 31);
      }
  }

  // -- reference, in double, from the table directly ---------------------------------------
  std::vector<int> want(BATCH);
  for (int b = 0; b < BATCH; ++b) {
    const int s = state_now[b];
    double bv = -1e300;
    int bi = VOCAB;
    for (int v = 0; v < VOCAB; ++v) {
      if (!((table[(size_t)s * WORDS + (v >> 5)] >> (v & 31)) & 1u)) continue;
      const double x = logits[(size_t)b * VOCAB + v];
      if (x > bv) { bv = x; bi = v; }      // first max wins: the lower-index tie-break
    }
    want[b] = bi;
  }

  float *dlogits, *dbias;
  unsigned int *dmask, *dtable;
  int *dstate, *dprev, *dtokens;
  CUDA_CHECK(cudaMalloc((void**)&dlogits, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dbias, bias.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dmask, hostmask.size() * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc((void**)&dtable, table.size() * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc((void**)&dstate, BATCH * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dprev, BATCH * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dtokens, BATCH * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dlogits, logits.data(), logits.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dbias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dmask, hostmask.data(), hostmask.size() * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dtable, table.data(), table.size() * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dprev, prev.data(), BATCH * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<bench::Row> rows;
  std::vector<int> got(BATCH);
  auto record = [&](const char* name, double per_step_bytes, const char* note) {
    CUDA_CHECK(cudaMemcpy(got.data(), dtokens, BATCH * sizeof(int), cudaMemcpyDeviceToHost));
    double err = 0;
    for (int b = 0; b < BATCH; ++b)
      err = std::max(err, bench::max_rel_err_scalar((float)got[b], (float)want[b]));
    bench::Row r;
    r.name = name;
    r.err = err;
    r.checksum = bench::checksum_of(got);
    r.bytes = per_step_bytes;
    r.flops = 0;
    r.note = note;
    rows.push_back(r);
  };

  const double dense_b = (double)BATCH * VOCAB * sizeof(float);
  const double bitset_b = (double)BATCH * WORDS * sizeof(unsigned int);
  const double state_b = (double)BATCH * sizeof(int);

  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(sample_dense, dim3(BATCH), dim3(BLOCK), 0, dlogits, dbias, dtokens);
    }, dev, 30, 8);
    record("1 dense fp32 mask from host", dense_b, "VOCAB floats per seq per step");
    rows.back().st = st;
  }
  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(sample_bitset, dim3(BATCH), dim3(BLOCK), 0, dlogits, dmask, dtokens);
    }, dev, 30, 8);
    record("2 bitset from host", bitset_b, "1 bit per token per seq");
    rows.back().st = st;
  }
  {
    CUDA_CHECK(cudaMemcpy(dstate, state_now.data(), BATCH * sizeof(int), cudaMemcpyHostToDevice));
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(sample_state, dim3(BATCH), dim3(BLOCK), 0, dlogits, dtable, dstate, dtokens);
    }, dev, 30, 8);
    record("3 state id, table on device", state_b, "4 bytes per seq per step");
    rows.back().st = st;
  }
  {
    // Variant 4 arrives at its own state, so reset to the *pre*-advance state each time.
    bench::Stats st = bench::time_kernel([&] {
      CUDA_CHECK(cudaMemcpy(dstate, state0.data(), BATCH * sizeof(int), cudaMemcpyHostToDevice));
      KERNEL_LAUNCH(sample_and_advance, dim3(BATCH), dim3(BLOCK), 0, dlogits, dtable, dstate,
                    dprev, dtokens);
    }, dev, 30, 8);
    CUDA_CHECK(cudaMemcpy(dstate, state0.data(), BATCH * sizeof(int), cudaMemcpyHostToDevice));
    KERNEL_LAUNCH(sample_and_advance, dim3(BATCH), dim3(BLOCK), 0, dlogits, dtable, dstate, dprev,
                  dtokens);
    CUDA_CHECK(cudaDeviceSynchronize());
    record("4 advance on device, host idle", 0.0, "nothing crosses per step");
    // The advanced state must match what the host would have computed, or the grammar has
    // silently forked between the two implementations.
    std::vector<int> st_got(BATCH);
    CUDA_CHECK(cudaMemcpy(st_got.data(), dstate, BATCH * sizeof(int), cudaMemcpyDeviceToHost));
    for (int b = 0; b < BATCH; ++b)
      rows.back().err = std::max(rows.back().err,
                                 bench::max_rel_err_scalar((float)st_got[b],
                                                           (float)state_now[b]));
    rows.back().st = st;
  }

  double avg = 0;
  for (int s = 0; s < NSTATES; ++s) avg += allowed[s];
  avg /= NSTATES;

  std::printf("problem   : %d sequences sampling under a %d-state grammar, vocab %d\n",
              BATCH, NSTATES, VOCAB);
  std::printf("grammar   : %.1f%% of the vocabulary legal per state on average\n",
              100.0 * avg / VOCAB);
  std::printf("ties      : the global maximum of every row is an ILLEGAL token, and the two\n"
              "            best legal tokens are bit-identical — so both \"apply the mask\"\n"
              "            and \"break ties by the lower index\" are under test\n");
  std::printf("note      : the transition is a stand-in — a real grammar compiles to a sparse\n"
              "            table or a pushdown automaton. The *shape* is what is faithful: few\n"
              "            states, a large vocabulary, one state per sequence per step\n");
  bench::header(dev);
  const double tol = 1e-6;
  bench::rows_out(rows, dev, tol);

  std::printf("\nAll four pick the same token for every sequence, checked against a\n"
              "double-precision argmax over the state's own legal set — and variant 4's\n"
              "device-side transition is checked against the host's, so the two copies of the\n"
              "grammar are known not to have forked.\n");

  // -- the traffic, which is the point ------------------------------------------------------
  std::printf("\nWhat crosses PCIe per decode step, batch %d:\n\n", BATCH);
  std::printf("  %-34s %14s %12s\n", "representation", "bytes/step", "vs dense");
  std::printf("  %-34s %11.0f B %11.1fx\n", "dense fp32 mask", dense_b, 1.0);
  std::printf("  %-34s %11.0f B %11.1fx\n", "bitset", bitset_b, dense_b / bitset_b);
  std::printf("  %-34s %11.0f B %11.1fx\n", "state id only", state_b, dense_b / state_b);
  std::printf("  %-34s %11.0f B %11s\n", "nothing (advance on device)", 0.0, "infinite");
  std::printf("\n  The device-resident table costs %.1f KB once — %d states x %d bytes — and is\n"
              "  read-only, shared by every sequence on the same state, and small enough to\n"
              "  live in L2. That one-off is what buys the last two rows.\n",
              (double)NSTATES * WORDS * sizeof(unsigned int) / 1024.0, NSTATES,
              (int)(WORDS * sizeof(unsigned int)));

  // -- and the part that is not bytes -------------------------------------------------------
  std::printf("\nThe part the byte counts do not show:\n\n");
  {
    const double step_ms = 10.0;                 // a decode step at batch 32
    struct { const char* what; double host_ms; bool sync; } modes[] = {
        {"host builds a dense mask", 2.0, true},
        {"host builds a bitset", 1.2, true},
        {"host sends a state id", 0.05, true},
        {"device advances, host idle", 0.0, false},
    };
    std::printf("  %-30s %12s %14s %12s\n", "where the grammar runs", "host work",
                "sync per step", "step cost");
    for (auto& m : modes) {
      // A host-side advance forces device -> host -> device before the sampler can run, so the
      // host work does not overlap: it lands on the step.
      const double total = step_ms + (m.sync ? m.host_ms : 0.0);
      std::printf("  %-30s %9.2f ms %14s %9.1f ms\n", m.what, m.host_ms,
                  m.sync ? "yes" : "no", total);
    }
  }
  std::printf("\n  Those host figures are illustrative, not measured — this program does not\n"
              "  run a real grammar compiler. What is not illustrative is the third column.\n"
              "  Variants 1-3 all need the emitted token on the host before they can compute\n"
              "  the next mask, so every decode step contains a device-to-host-to-device round\n"
              "  trip that nothing else in the step can hide. Variant 4 has none.\n"
              "\n  That is the honest resolution of a thread running through three kernels.\n"
              "  14_logit_mask measured the mask and found 0.05%%. 18_spec_verify multiplied\n"
              "  it by k+1 and found 0.5%%. Both were right, and both were measuring the\n"
              "  cheap half. The expensive half was never the mask — it was needing the CPU\n"
              "  in the loop at all, and the fix for that is a table, not a faster kernel.\n");

  for (void* p : {(void*)dlogits, (void*)dbias, (void*)dmask, (void*)dtable, (void*)dstate,
                  (void*)dprev, (void*)dtokens})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
