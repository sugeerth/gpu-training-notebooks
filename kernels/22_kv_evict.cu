// 22_kv_evict.cu — choosing who loses their KV cache when the pool fills, and why an agent
// platform's answer is not a chat platform's.
//
//     nvcc -O3 -arch=native 22_kv_evict.cu -o build/22 && build/22
//     make check
//
// Part 3 of the agent notebook established that an agent holds its whole KV cache through every
// tool call and produces nothing while it does. Multiply that by a few hundred concurrent runs
// and the pool fills with sequences that are not asking for anything. Something has to go.
//
// The kernel here is the selection: given a pool that is short by `need` bytes and a set of
// sequences with their sizes and their scores, pick the victims. Three implementations, one
// answer — the policy is a parameter, and the interesting part is what the policy should be.
//
// **What makes an agent's answer different.** A chat server evicting a sequence throws away
// work it will have to redo, so the classic scores — least recently used, largest first — are
// really proxies for "who will least regret this". For an agent, two things change that
// calculus, and they pull in the same direction:
//
//   1. **The victim is probably idle anyway.** A sequence in the middle of a two-second tool
//      call is not going to notice being evicted for the next two seconds. Its idle time is
//      knowable: the scheduler issued the tool call and knows roughly how long tools take.
//
//   2. **Coming back is cheap, because of 16_prefix_match.** Re-admitting an evicted agent is
//      not a full prefill — it is a prefix-cache lookup that will hit on nearly the whole
//      context, because the context is exactly what it was. The restore cost is
//      `(1 - hit_rate) x tokens`, not `tokens`, and at a 95% hit rate that is a twentieth.
//
// Put those together and the agent-aware score is roughly
//
//     value_of_evicting = bytes_freed x idle_time_remaining / expected_restore_cost
//
// which is a completely different ranking from LRU, and the table at the bottom shows by how
// much. The point is not that this exact formula is right; it is that **LRU is answering a
// question about the past when the scheduler already knows the future**, and for an agent
// workload it knows it unusually well.
//
// Selection must also be *deterministic*. Two runs of the same server state must evict the same
// sequences, or a replayed trajectory diverges for a reason that has nothing to do with the
// model — which is the same argument 19_batch_invariant makes about reductions, arriving from a
// completely different direction. Ties break by index, everywhere.
//
// Prerequisites: 02_reduce.cu for the reductions, 16_prefix_match.cu for why restore is cheap.
#include <functional>

#include "common.cuh"

#if SHIM_BUILD
constexpr int NSEQ = 64;         // sequences resident in the pool
constexpr int BLOCK = 32;
#else
constexpr int NSEQ = 4096;
constexpr int BLOCK = 256;
#endif

// ---------------------------------------------------------------------------------------
// Variant 1 — one thread walks the list, repeatedly taking the best remaining candidate until
// enough bytes are freed.
//
// O(victims x NSEQ) and entirely serial. It is here because it is what the first version looks
// like, and because it is the unambiguous definition of the answer the other two must match.
// ---------------------------------------------------------------------------------------
__global__ void select_serial(const float* __restrict__ score, const int* __restrict__ bytes,
                              long long need, int* __restrict__ victim, int* __restrict__ nvic) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  long long freed = 0;
  int n = 0;
  bool taken[NSEQ];
  for (int i = 0; i < NSEQ; ++i) taken[i] = false;
  while (freed < need) {
    float best = -INFINITY;
    int bi = -1;
    for (int i = 0; i < NSEQ; ++i) {
      if (taken[i]) continue;
      // Ties break by the lower index, so the answer does not depend on scan order.
      if (score[i] > best) { best = score[i]; bi = i; }
    }
    if (bi < 0) break;                      // nothing left to evict
    taken[bi] = true;
    victim[n++] = bi;
    freed += bytes[bi];
  }
  *nvic = n;
}

// ---------------------------------------------------------------------------------------
// Variant 2 — the same greedy loop, but each argmax is a parallel block reduction.
//
// Still `victims` sequential rounds, but each round is O(NSEQ / threads) instead of O(NSEQ).
// This is the shape most engines actually ship, because the victim count is small — you evict
// two or three sequences, not two hundred — so the outer loop is short and the inner one is the
// whole cost.
// ---------------------------------------------------------------------------------------
__global__ void select_parallel(const float* __restrict__ score, const int* __restrict__ bytes,
                                long long need, int* __restrict__ victim,
                                int* __restrict__ nvic) {
  SHARED(float, sval, BLOCK);
  SHARED(int, sidx, BLOCK);
  SHARED(int, staken, NSEQ);

  for (int i = threadIdx.x; i < NSEQ; i += blockDim.x) staken[i] = 0;
  __syncthreads();

  long long freed = 0;
  int n = 0;
  while (freed < need) {
    float bv = -INFINITY;
    int bi = NSEQ;
    for (int i = threadIdx.x; i < NSEQ; i += blockDim.x) {
      if (staken[i]) continue;
      const float s = score[i];
      if (s > bv || (s == bv && i < bi)) { bv = s; bi = i; }
    }
    sval[threadIdx.x] = bv;
    sidx[threadIdx.x] = bi;
    __syncthreads();
    for (int t = blockDim.x / 2; t > 0; t >>= 1) {
      if (threadIdx.x < t) {
        const float o = sval[threadIdx.x + t];
        const int oi = sidx[threadIdx.x + t];
        if (o > sval[threadIdx.x] || (o == sval[threadIdx.x] && oi < sidx[threadIdx.x])) {
          sval[threadIdx.x] = o;
          sidx[threadIdx.x] = oi;
        }
      }
      __syncthreads();
    }
    const int pick = sidx[0];
    if (pick >= NSEQ) break;
    if (threadIdx.x == 0) {
      staken[pick] = 1;
      victim[n] = pick;
    }
    freed += bytes[pick];
    ++n;
    __syncthreads();
  }
  if (threadIdx.x == 0) *nvic = n;
}

// ---------------------------------------------------------------------------------------
// Variant 3 — rank by counting, then take a prefix. No repeated passes.
//
// Each sequence counts how many sequences outrank it, which is its position in the sorted
// order. That is O(NSEQ^2) work but O(1) rounds, and it is fully parallel: for the few-thousand
// sequences a pool actually holds it beats the iterative version by a wide margin, and it has
// no data-dependent loop bound at all.
//
// The comparison `(s > mine) || (s == mine && j < i)` is what makes the rank a strict total
// order even with duplicate scores — without the index tie-break, two equal scores would each
// count the other and both land on the same rank, silently losing a victim.
// ---------------------------------------------------------------------------------------
__global__ void rank_by_counting(const float* __restrict__ score, int* __restrict__ rank) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NSEQ) return;
  const float mine = score[i];
  int r = 0;
  for (int j = 0; j < NSEQ; ++j) {
    const float s = score[j];
    if (s > mine || (s == mine && j < i)) ++r;
  }
  rank[i] = r;
}

__global__ void take_prefix(const int* __restrict__ rank, const int* __restrict__ bytes,
                            long long need, int* __restrict__ victim, int* __restrict__ nvic) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  // Invert the rank into sorted order, then walk it. Both loops are over NSEQ and neither
  // depends on the data, which is what makes this the branch-free version.
  int order[NSEQ];
  for (int i = 0; i < NSEQ; ++i) order[rank[i]] = i;
  long long freed = 0;
  int n = 0;
  for (int r = 0; r < NSEQ && freed < need; ++r) {
    victim[n++] = order[r];
    freed += bytes[order[r]];
  }
  *nvic = n;
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
  (void)argc; (void)argv;

  // The resident set. Every sequence carries what a scheduler would actually know about it.
  std::vector<int> ctx(NSEQ), bytes(NSEQ);
  std::vector<float> last_used(NSEQ), idle_left(NSEQ), hit_rate(NSEQ);
  {
    unsigned r = 20240917u;
    auto next = [&]() { r = r * 1664525u + 1013904223u; return (double)(r >> 8) / 16777216.0; };
    for (int i = 0; i < NSEQ; ++i) {
      // Context grows with turn number, as an agent's does.
      ctx[i] = 2000 + (int)(next() * 60000);
      bytes[i] = ctx[i] * 128;                 // ~128 KB/token at fp16 GQA, scaled down
      last_used[i] = (float)(next() * 10.0);   // seconds since this sequence last ran
      // How much longer it will be idle: most sequences are mid-tool-call, some are not.
      idle_left[i] = next() < 0.6 ? (float)(next() * 5.0) : 0.0f;
      hit_rate[i] = (float)(0.80 + next() * 0.19);
    }
  }

  long long pool = 0;
  for (int i = 0; i < NSEQ; ++i) pool += bytes[i];
  const long long need = pool / 8;             // the pool is short by an eighth

  // The three policies. Only the score changes; the selection kernels are policy-agnostic.
  auto score_lru = [&](int i) { return last_used[i]; };
  auto score_largest = [&](int i) { return (float)bytes[i]; };
  auto score_agent = [&](int i) {
    // Freeing a lot, from something that will not miss it, that is cheap to bring back.
    const float restore = (1.0f - hit_rate[i]) * (float)ctx[i] + 1.0f;
    return (float)bytes[i] * (idle_left[i] + 0.05f) / restore;
  };

  std::vector<float> score(NSEQ);
  for (int i = 0; i < NSEQ; ++i) score[i] = score_agent(i);

  // Plant exact ties AT THE TOP of the ranking. Four sequences are given one bit-identical
  // score, above every other, so they are the first victims and the index tie-break is what
  // decides their order. Ties further down the list would never be reached — which is what
  // mutation testing found when they were planted by giving four sequences identical inputs:
  // their score was unremarkable, they were never selected, and two tie-break bugs passed.
  const int TIED[] = {5, 17, 29, 41};
  {
    float mx = -1e30f;
    for (int i = 0; i < NSEQ; ++i) mx = std::max(mx, score[i]);
    for (int i : TIED)
      if (i < NSEQ) score[i] = mx * 1.5f;
  }
  auto score_final = [&](int i) { return score[i]; };

  // -- reference: the same greedy selection, on the host, in double -------------------------
  std::vector<int> want;
  {
    std::vector<char> taken(NSEQ, 0);
    long long freed = 0;
    while (freed < need) {
      double best = -1e300;
      int bi = -1;
      for (int i = 0; i < NSEQ; ++i) {
        if (taken[i]) continue;
        if ((double)score[i] > best) { best = score[i]; bi = i; }
      }
      if (bi < 0) break;
      taken[bi] = 1;
      want.push_back(bi);
      freed += bytes[bi];
    }
  }

  float* dscore;
  int *dbytes, *dvictim, *dnvic, *drank;
  CUDA_CHECK(cudaMalloc((void**)&dscore, NSEQ * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dbytes, NSEQ * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dvictim, NSEQ * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dnvic, sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&drank, NSEQ * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dscore, score.data(), NSEQ * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dbytes, bytes.data(), NSEQ * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<bench::Row> rows;
  std::vector<int> victim(NSEQ);
  int nvic = 0;
  auto record = [&](const char* name, const char* note) {
    CUDA_CHECK(cudaMemcpy(&nvic, dnvic, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(victim.data(), dvictim, NSEQ * sizeof(int), cudaMemcpyDeviceToHost));
    // The victim *list*, in order, must match exactly — not just the set, and not just the
    // count. Two servers that evict the same sequences in a different order have still made
    // the same decision, but a checksum over the ordered list is the strictest thing available
    // and costs nothing to require.
    double err = bench::max_rel_err_scalar((float)nvic, (float)want.size());
    for (size_t k = 0; k < want.size() && k < (size_t)nvic; ++k)
      err = std::max(err, bench::max_rel_err_scalar((float)victim[k], (float)want[k]));
    std::vector<int> sig(victim.begin(), victim.begin() + nvic);
    bench::Row r;
    r.name = name;
    r.err = err;
    r.checksum = bench::checksum_of(sig);
    r.bytes = (double)NSEQ * (sizeof(float) + sizeof(int));
    r.flops = 0;
    r.note = note;
    rows.push_back(r);
  };

  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(select_serial, dim3(1), dim3(1), 0, dscore, dbytes, need, dvictim, dnvic);
    }, dev, 20, 5);
    record("1 serial greedy", "O(victims x NSEQ), one thread");
    rows.back().st = st;
  }
  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(select_parallel, dim3(1), dim3(BLOCK), 0, dscore, dbytes, need, dvictim,
                    dnvic);
    }, dev, 20, 5);
    record("2 parallel argmax per round", "victims rounds, each parallel");
    rows.back().st = st;
  }
  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(rank_by_counting, dim3((NSEQ + BLOCK - 1) / BLOCK), dim3(BLOCK), 0, dscore,
                    drank);
      KERNEL_LAUNCH(take_prefix, dim3(1), dim3(1), 0, drank, dbytes, need, dvictim, dnvic);
    }, dev, 20, 5);
    record("3 rank by counting, take prefix", "O(1) rounds, no data-dependent loop");
    rows.back().st = st;
  }

  std::printf("problem   : %d sequences resident, pool short by %.1f%% (%lld MB of %lld MB)\n",
              NSEQ, 100.0 * need / pool, need / 1000000, pool / 1000000);
  std::printf("policy    : agent-aware — bytes freed x idle time remaining / restore cost\n");
  std::printf("ties      : four sequences share one bit-identical score at the TOP of the\n"
              "            ranking, so they are the first victims and the index tie-break —\n"
              "            not the float comparison — decides their order\n");
  bench::header(dev);
  const double tol = 1e-6;
  bench::rows_out(rows, dev, tol);

  std::printf("\nAll three select the same victims, in the same order, checked against a host\n"
              "greedy selection. Ties break by index everywhere, so the answer does not depend\n"
              "on scan order or on which block finished first — an eviction decision that\n"
              "varied run to run would make a replayed trajectory diverge for a reason that has\n"
              "nothing to do with the model.\n");

  // -- what the policy is worth --------------------------------------------------------------
  std::printf("\nThe same shortfall, three policies:\n\n");
  std::printf("  %-24s %10s %14s %16s %16s\n", "policy", "victims", "freed", "restore cost",
              "still-busy hit");
  struct Result { int victims; long long freed; double restore; int busy; };
  auto simulate = [&](std::function<float(int)> sc) {
    std::vector<char> taken(NSEQ, 0);
    Result out{0, 0, 0.0, 0};
    while (out.freed < need) {
      double best = -1e300;
      int bi = -1;
      for (int i = 0; i < NSEQ; ++i) {
        if (taken[i]) continue;
        const double v = sc(i);
        if (v > best) { best = v; bi = i; }
      }
      if (bi < 0) break;
      taken[bi] = 1;
      ++out.victims;
      out.freed += bytes[bi];
      // What it will cost to bring this sequence back: only the tokens the prefix cache misses.
      out.restore += (1.0 - hit_rate[bi]) * ctx[bi];
      if (idle_left[bi] <= 0.0f) ++out.busy;    // evicted something that wanted to run now
    }
    return out;
  };
  struct { const char* name; std::function<float(int)> sc; } policies[] = {
      {"least recently used", score_lru},
      {"largest first", score_largest},
      {"agent-aware", score_final},
  };
  Result base{};
  for (int pi = 0; pi < 3; ++pi) {
    Result r = simulate(policies[pi].sc);
    if (pi == 0) base = r;
    std::printf("  %-24s %10d %11lld MB %13.0f tok %11d / %d\n", policies[pi].name, r.victims,
                r.freed / 1000000, r.restore, r.busy, r.victims);
  }
  std::printf("\n  Read the last two columns. \"Restore cost\" is the tokens that will have to\n"
              "  be re-prefilled when the victims come back — and it is small for all three\n"
              "  only because the prefix cache exists. Without 16_prefix_match's lookup the\n"
              "  numbers here would be the *full* contexts, and eviction would be a decision\n"
              "  about which user to punish rather than a routine reclaim.\n"
              "\n  \"Still-busy hit\" is the count of victims that were not idle — sequences\n"
              "  evicted while they actually wanted to run. That is the column LRU cannot see:\n"
              "  recency is a guess about the future, and here the scheduler is not guessing,\n"
              "  because it issued the tool calls itself and knows which sequences are parked.\n");

  // -- and the reason all of this is cheap ---------------------------------------------------
  std::printf("\nWhy eviction is a routine operation for agents and a crisis for chat:\n\n");
  std::printf("  %-30s %16s %16s %14s\n", "hit rate on return", "restore, agent",
              "restore, no cache", "ratio");
  {
    const double avg_ctx = [&] {
      double t = 0;
      for (int i = 0; i < NSEQ; ++i) t += ctx[i];
      return t / NSEQ;
    }();
    for (double h : {0.0, 0.50, 0.80, 0.95, 0.99}) {
      const double with = (1.0 - h) * avg_ctx;
      std::printf("  %29.0f%% %13.0f tok %13.0f tok %12.0fx\n", h * 100.0, with, avg_ctx,
                  h < 1.0 ? avg_ctx / (with > 0 ? with : 1.0) : 0.0);
    }
  }
  std::printf("\n  At a 95%% hit rate an evicted agent costs a twentieth of its context to\n"
              "  restore, which is why \"drop the sequence and let the prefix cache pay for the\n"
              "  return\" was the fourth option in the tool-gap table and usually the best one.\n"
              "  Every other option there — hold, evict-and-recompute, offload — is arithmetic\n"
              "  about memory. This one is arithmetic about the cache, and the cache is a hash\n"
              "  lookup.\n");

  for (void* p : {(void*)dscore, (void*)dbytes, (void*)dvictim, (void*)dnvic, (void*)drank})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
