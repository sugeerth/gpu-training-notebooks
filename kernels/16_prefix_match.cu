// 16_prefix_match.cu — the prefix cache lookup itself: finding how much of an agent's context
// the server has already seen, before deciding to prefill any of it.
//
//     nvcc -O3 -arch=native 16_prefix_match.cu -o build/16 && build/16
//     make check
//
// Every other kernel in the agent set makes an expensive thing cheaper. This one decides
// whether the expensive thing happens at all, and it is by a wide margin the most valuable
// code in the set — 13_prefix_attention saves a re-*read* of the prefix, this saves the
// re-*compute*, and computing is thousands of times dearer than reading.
//
// The problem. An agent turn arrives carrying its whole conversation: system prompt, tool
// definitions, every previous message and tool result, then a few hundred new tokens. Almost
// all of it is byte-identical to the turn before. The server must work out *how much* of the
// leading token sequence it already holds KV for, and it must do that before it can schedule
// anything, so the lookup sits on the critical path of every single request.
//
// Three properties make this a different problem from an ordinary cache lookup:
//
//   1. It is a **prefix** match, not a set-membership test. Attention at position i depends on
//      every token before it, so cached KV for a block is only reusable if *every* preceding
//      block also matched. A block that matches after a gap is worthless — it was computed
//      against a different history. So the answer is the length of the longest matching
//      leading run, and the first miss ends the search.
//
//   2. It is **block-granular**. KV lives in pages, so the match is quantized to PAGE tokens.
//      Editing one token near the start of a system prompt invalidates everything after it —
//      which is why "put the volatile part of the prompt last" is the single highest-leverage
//      piece of prompt engineering in agent serving, and why a timestamp in a system prompt is
//      an expensive mistake.
//
//   3. The identity of a block must encode **its whole prefix**, not its own tokens. Two
//      requests can contain the same 16 tokens in the same position with different histories,
//      and their KV is not interchangeable. So each block's id is chained: it mixes its own
//      token digest with the id of the block before it. That chaining is inherently serial,
//      and the kernels below are largely about how much of the rest can be made parallel
//      around it.
//
// The four variants below compute the same number — how many leading pages hit — by four
// routes, from comparing raw tokens against every cached sequence to probing a hash table with
// every page at once.
//
// Prerequisite: 15_kv_fork.cu, for the block table this hands its answer to.
#include <functional>

#include "common.cuh"

#if SHIM_BUILD
constexpr int PAGE = 8;          // tokens per KV page
constexpr int NPAGES = 24;       // pages in the arriving request
constexpr int NCACHE = 64;       // entries in the cache
constexpr int BLOCK = 32;
#else
constexpr int PAGE = 16;
constexpr int NPAGES = 1360;     // ~21k tokens: a 20-turn agent's context
constexpr int NCACHE = 8192;
constexpr int BLOCK = 128;
#endif
constexpr int TABLE = 2 * NCACHE;          // open-addressed, power-of-two capacity
constexpr int NTOK = NPAGES * PAGE;
constexpr unsigned long long EMPTY = 0ull;  // a chained id is never 0 by construction

// ---------------------------------------------------------------------------------------
// Hashing. splitmix64's finalizer: cheap, and it avalanches, which is what stops two adjacent
// token ids from colliding into the same page digest.
// ---------------------------------------------------------------------------------------
__host__ __device__ inline unsigned long long mix64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// ---------------------------------------------------------------------------------------
// Step 1 — a digest per page, in parallel.
//
// The digest must be **order-sensitive**: [A, B] and [B, A] are different prefixes. Mixing the
// position into each token before combining buys that, and lets the combine itself be XOR,
// which is associative and commutative — so the reduction can run in any order and still give
// a bit-identical answer. That is deliberate: this kernel must be reproducible, because a
// digest that varies run to run is a cache that misses on its own output.
// ---------------------------------------------------------------------------------------
__global__ void page_digest(const int* __restrict__ tokens, unsigned long long* __restrict__ out,
                            int npages) {
  SHARED(unsigned long long, red, BLOCK);
  const int p = blockIdx.x;
  if (p >= npages) return;

  unsigned long long h = 0ull;
  for (int r = threadIdx.x; r < PAGE; r += blockDim.x)
    h ^= mix64((unsigned long long)(unsigned)tokens[p * PAGE + r] * 0x100000001B3ull + (unsigned)r);
  red[threadIdx.x] = h;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] ^= red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[p] = red[0];
}

// ---------------------------------------------------------------------------------------
// Step 2 — chain the digests. id_p = mix(id_{p-1}, digest_p), so a page's id depends on every
// token before it as well as its own.
//
// This is serial by construction and there is no way around it: it is a prefix scan whose
// operator is a hash, and a hash is not associative, so the parallel-scan trick does not
// apply. One block, one thread, NPAGES iterations. It is cheap — NPAGES is ~1300 for a 21k
// context — and the honest thing to do is to run it that way rather than pretend otherwise.
// ---------------------------------------------------------------------------------------
__global__ void chain_ids(const unsigned long long* __restrict__ digest,
                          unsigned long long* __restrict__ ids, int npages) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  unsigned long long acc = 0xCBF29CE484222325ull;   // FNV offset basis, as a seed
  for (int p = 0; p < npages; ++p) {
    acc = mix64(acc ^ digest[p]);
    ids[p] = acc | 1ull;            // never 0, so 0 can mean "empty slot"
  }
}

// ---------------------------------------------------------------------------------------
// Variant 1 — no hashing at all. Compare the arriving tokens against the cached token stream,
// page by page, and stop at the first page that differs.
//
// This is what a first implementation looks like, and it is not absurd: it is exact, with no
// collision risk. It is also O(matched x PAGE) token compares against *one* candidate
// sequence, and a real cache holds thousands. Its cost is here to be the baseline the hashed
// versions are measured against.
// ---------------------------------------------------------------------------------------
__global__ void match_tokens(const int* __restrict__ tokens, const int* __restrict__ cached,
                             int cached_pages, int npages, int* __restrict__ out) {
  SHARED(int, ok, BLOCK);
  const int p = blockIdx.x;
  if (p >= npages) return;
  if (p >= cached_pages) { if (threadIdx.x == 0) out[p] = 0; return; }

  int good = 1;
  for (int r = threadIdx.x; r < PAGE; r += blockDim.x)
    if (tokens[p * PAGE + r] != cached[p * PAGE + r]) good = 0;
  ok[threadIdx.x] = good;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) ok[threadIdx.x] &= ok[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[p] = ok[0];
}

// ---------------------------------------------------------------------------------------
// Variant 2 — chained ids, linear scan of the cache.
//
// Now a page's identity is one 64-bit number, so a comparison is one compare instead of PAGE
// of them, and it can be checked against *every* cached sequence at once rather than one
// candidate. The cost is a full scan of the cache per page: O(npages x NCACHE).
// ---------------------------------------------------------------------------------------
__global__ void match_linear(const unsigned long long* __restrict__ ids,
                             const unsigned long long* __restrict__ cache_ids,
                             int ncache, int npages, int* __restrict__ out) {
  SHARED(int, hit, BLOCK);
  const int p = blockIdx.x;
  if (p >= npages) return;

  int found = 0;
  for (int c = threadIdx.x; c < ncache; c += blockDim.x)
    if (cache_ids[c] == ids[p]) found = 1;
  hit[threadIdx.x] = found;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) hit[threadIdx.x] |= hit[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[p] = hit[0];
}

// ---------------------------------------------------------------------------------------
// Variant 3 — chained ids, open-addressed hash table, walked serially with an early exit.
//
// The lookup is now O(1) per page, and the walk stops at the first miss — which is correct,
// because nothing after a miss is reusable anyway. On a cold request that exits after one
// probe. It is one thread, though, so a *hit* costs `matched` sequential dependent loads, each
// one a cache miss on a table too big for L2.
// ---------------------------------------------------------------------------------------
__device__ inline int probe(const unsigned long long* table, unsigned long long id) {
  unsigned slot = (unsigned)(id >> 32) & (TABLE - 1);
  for (int i = 0; i < TABLE; ++i) {
    const unsigned long long v = table[slot];
    if (v == id) return 1;
    if (v == EMPTY) return 0;
    slot = (slot + 1) & (TABLE - 1);
  }
  return 0;
}

__global__ void match_hashed(const unsigned long long* __restrict__ ids,
                             const unsigned long long* __restrict__ table,
                             int npages, int* __restrict__ matched) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int n = 0;
  while (n < npages && probe(table, ids[n])) ++n;
  *matched = n;
}

// ---------------------------------------------------------------------------------------
// Variant 4 — same table, every page probed at once.
//
// The early exit was a false economy: it turned an O(1)-depth problem into a chain of
// `matched` dependent memory loads. Probing all pages in parallel does more total work — the
// pages past the first miss are wasted — but the *latency* is one probe plus a reduction,
// which is what a request on the critical path actually cares about.
//
// The reduction is a min over the indices that missed. min is associative and commutative on
// integers, so this is reproducible regardless of which block lands first.
// ---------------------------------------------------------------------------------------
__global__ void match_parallel(const unsigned long long* __restrict__ ids,
                               const unsigned long long* __restrict__ table,
                               int npages, int* __restrict__ first_miss) {
  SHARED(int, red, BLOCK);
  int local = npages;
  for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < npages; p += gridDim.x * blockDim.x)
    if (!probe(table, ids[p]) && p < local) local = p;
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = min(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicMin(first_miss, red[0]);
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
  (void)argc; (void)argv;

  // The arriving request, and the previous turn that is already in the cache. They agree for
  // SHARED_PAGES pages and then diverge — that divergence point is the answer every variant
  // has to find.
  //
  // ADVERSARY is what makes the hashing testable rather than merely plausible, and each case
  // exists because mutation testing showed the previous data could not tell a correct
  // implementation from a broken one:
  //
  //   EDITED       one token changed. The everyday case, and the one that proves nothing about
  //                the hash — a weaker hash still finds this divergence.
  //   PERMUTED     the diverging page holds the *same tokens in a different order*. A digest
  //                that does not mix the position gives it the same value, the chained ids
  //                match, and the lookup reports a hit on a page whose KV is for different
  //                text. This is not exotic: reordered tool definitions, a re-sorted list of
  //                files, two arguments swapped in a JSON object.
  //   REUSED       the diverging page holds tokens that appear verbatim elsewhere in the
  //                cache, under a different history. If a block's id is its own digest rather
  //                than a chain over everything before it, this false-hits — and the KV
  //                returned was computed against a completely different conversation.
  //
  // A single scenario cannot exercise both, so all three run and the answers are combined.
  enum Adversary { EDITED, PERMUTED, REUSED };
  const int SHARED_PAGES = (NPAGES * 7) / 8;

  auto build = [&](Adversary adv, std::vector<int>& tokens, std::vector<int>& cached) {
    tokens.assign(NTOK, 0);
    unsigned s = 0x1234567u;
    for (int i = 0; i < NTOK; ++i) {
      s = s * 1664525u + 1013904223u;
      tokens[i] = (int)(s >> 12) & 0xFFFF;
    }
    cached = tokens;
    const int d = SHARED_PAGES * PAGE;          // first token of the diverging page
    if (adv == EDITED) {
      cached[d + 3] ^= 1;
    } else if (adv == PERMUTED) {
      // Same multiset of tokens, reversed within the page. Any position-blind digest is blind
      // to this by construction.
      for (int r = 0; r < PAGE; ++r) cached[d + r] = tokens[d + PAGE - 1 - r];
      if (cached[d] == tokens[d]) cached[d] ^= 1;   // guard against a palindromic page
    } else {
      // The arriving page's tokens are copied verbatim from a page the cache holds elsewhere,
      // at a different position and after a different history.
      const int src = (SHARED_PAGES / 2) * PAGE;
      for (int r = 0; r < PAGE; ++r) tokens[d + r] = cached[src + r];
      if (tokens[d] == cached[d]) tokens[d] ^= 1;
    }
    for (int i = d + PAGE; i < NTOK; ++i) cached[i] = (cached[i] * 7 + 11) & 0xFFFF;
  };

  std::vector<int> tokens, cached;

  int *dtok, *dcached, *dhit, *dmatched;
  unsigned long long *ddig, *dids, *dcache_ids, *dtable;
  CUDA_CHECK(cudaMalloc((void**)&dtok, NTOK * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dcached, NTOK * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dhit, NPAGES * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dmatched, sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&ddig, NPAGES * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc((void**)&dids, NPAGES * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc((void**)&dcache_ids, NCACHE * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc((void**)&dtable, TABLE * sizeof(unsigned long long)));

  const double id_bytes = (double)NPAGES * sizeof(unsigned long long);
  std::vector<int> hits(NPAGES);
  int want = 0;                       // the reference answer for the scenario now loaded

  // Load one scenario onto the device: tokens, cache ids, hash table, and the host reference.
  auto load = [&](Adversary adv) {
    build(adv, tokens, cached);
    CUDA_CHECK(cudaMemcpy(dtok, tokens.data(), NTOK * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dcached, cached.data(), NTOK * sizeof(int), cudaMemcpyHostToDevice));

    KERNEL_LAUNCH(page_digest, dim3(NPAGES), dim3(BLOCK), 0, dtok, ddig, NPAGES);
    KERNEL_LAUNCH(chain_ids, dim3(1), dim3(1), 0, ddig, dids, NPAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> ids(NPAGES);
    CUDA_CHECK(cudaMemcpy(ids.data(), dids, NPAGES * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    KERNEL_LAUNCH(page_digest, dim3(NPAGES), dim3(BLOCK), 0, dcached, ddig, NPAGES);
    KERNEL_LAUNCH(chain_ids, dim3(1), dim3(1), 0, ddig, dids, NPAGES);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<unsigned long long> cached_ids(NPAGES);
    CUDA_CHECK(cudaMemcpy(cached_ids.data(), dids, NPAGES * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dids, ids.data(), NPAGES * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));

    // The cache holds the previous turn's pages plus unrelated entries from other
    // conversations, so the table is never trivially empty.
    std::vector<unsigned long long> cache_ids(NCACHE), table(TABLE, EMPTY);
    int n = 0;
    for (int p = 0; p < NPAGES && n < NCACHE; ++p) cache_ids[n++] = cached_ids[p];
    unsigned long long seed = 99991ull;
    while (n < NCACHE) { seed = mix64(seed); cache_ids[n++] = seed | 1ull; }
    for (int c = 0; c < NCACHE; ++c) {
      unsigned slot = (unsigned)(cache_ids[c] >> 32) & (TABLE - 1);
      while (table[slot] != EMPTY && table[slot] != cache_ids[c]) slot = (slot + 1) & (TABLE - 1);
      table[slot] = cache_ids[c];
    }
    CUDA_CHECK(cudaMemcpy(dcache_ids, cache_ids.data(), NCACHE * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dtable, table.data(), TABLE * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));

    // The reference is a host walk over the raw tokens. It knows nothing about hashing, which
    // is what makes it a reference rather than a second opinion.
    want = 0;
    while (want < NPAGES) {
      bool same = true;
      for (int r = 0; r < PAGE; ++r)
        if (tokens[want * PAGE + r] != cached[want * PAGE + r]) { same = false; break; }
      if (!same) break;
      ++want;
    }
  };

  auto leading_run = [&]() {
    int n = 0;
    while (n < NPAGES && hits[n]) ++n;
    return n;
  };

  // The four routes to the answer. Each takes the currently loaded scenario and returns the
  // number of leading pages it believes are cached.
  auto v1 = [&]() {
    KERNEL_LAUNCH(match_tokens, dim3(NPAGES), dim3(BLOCK), 0, dtok, dcached, NPAGES, NPAGES, dhit);
    CUDA_CHECK(cudaMemcpy(hits.data(), dhit, NPAGES * sizeof(int), cudaMemcpyDeviceToHost));
    return leading_run();
  };
  auto v2 = [&]() {
    KERNEL_LAUNCH(match_linear, dim3(NPAGES), dim3(BLOCK), 0, dids, dcache_ids, NCACHE, NPAGES,
                  dhit);
    CUDA_CHECK(cudaMemcpy(hits.data(), dhit, NPAGES * sizeof(int), cudaMemcpyDeviceToHost));
    return leading_run();
  };
  auto v3 = [&]() {
    int got = 0;
    KERNEL_LAUNCH(match_hashed, dim3(1), dim3(1), 0, dids, dtable, NPAGES, dmatched);
    CUDA_CHECK(cudaMemcpy(&got, dmatched, sizeof(int), cudaMemcpyDeviceToHost));
    return got;
  };
  auto v4 = [&]() {
    int got = 0;
    const int init = NPAGES;
    CUDA_CHECK(cudaMemcpy(dmatched, &init, sizeof(int), cudaMemcpyHostToDevice));
    KERNEL_LAUNCH(match_parallel, dim3(8), dim3(BLOCK), 0, dids, dtable, NPAGES, dmatched);
    CUDA_CHECK(cudaMemcpy(&got, dmatched, sizeof(int), cudaMemcpyDeviceToHost));
    return got;
  };

  const Adversary SCENARIOS[] = {EDITED, PERMUTED, REUSED};
  const char* SCENARIO_NAME[] = {"one token edited", "page reordered", "page reused elsewhere"};
  const int NS = 3;

  std::vector<bench::Row> rows;

  // Every variant answers all three scenarios. Its error is the worst over them, and its
  // checksum covers all three — so a variant that is right about the easy case and wrong about
  // the adversarial ones fails, which is what the earlier single-scenario version could not do.
  auto evaluate = [&](const char* name, std::function<int()> fn, double bytes,
                      const char* note, std::vector<int>* out) {
    std::vector<int> got(NS), wants(NS);
    double err = 0;
    for (int i = 0; i < NS; ++i) {
      load(SCENARIOS[i]);
      got[i] = fn();
      wants[i] = want;
      err = std::max(err, bench::max_rel_err_scalar((float)got[i], (float)want));
    }
    bench::Row r;
    r.name = name;
    r.err = err;
    r.checksum = bench::checksum_of(got);
    r.bytes = bytes;
    r.flops = 0;
    r.note = note;
    // Time it on the last scenario loaded, so the timing path is exercised exactly once.
    rows.push_back(r);
    if (out) *out = got;
  };

  std::vector<int> answers1, answers2, answers3, answers4;
  evaluate("1 token compare, 1 candidate", v1, 2.0 * NTOK * sizeof(int),
           "exact, but one sequence only", &answers1);
  rows.back().st = bench::time_kernel([&] {
    KERNEL_LAUNCH(match_tokens, dim3(NPAGES), dim3(BLOCK), 0, dtok, dcached, NPAGES, NPAGES, dhit);
  }, dev, 30, 8);

  evaluate("2 chained id, linear scan", v2,
           id_bytes + (double)NPAGES * NCACHE * sizeof(unsigned long long),
           "every cached sequence", &answers2);
  rows.back().st = bench::time_kernel([&] {
    KERNEL_LAUNCH(match_linear, dim3(NPAGES), dim3(BLOCK), 0, dids, dcache_ids, NCACHE, NPAGES,
                  dhit);
  }, dev, 30, 8);

  evaluate("3 hash probe, serial early exit", v3,
           id_bytes + (double)(want + 1) * sizeof(unsigned long long), "O(1) per page",
           &answers3);
  rows.back().st = bench::time_kernel([&] {
    KERNEL_LAUNCH(match_hashed, dim3(1), dim3(1), 0, dids, dtable, NPAGES, dmatched);
  }, dev, 30, 8);

  evaluate("4 hash probe, all pages at once", v4,
           id_bytes + (double)NPAGES * sizeof(unsigned long long), "one probe of depth",
           &answers4);
  rows.back().st = bench::time_kernel([&] {
    const int init = NPAGES;
    CUDA_CHECK(cudaMemcpy(dmatched, &init, sizeof(int), cudaMemcpyHostToDevice));
    KERNEL_LAUNCH(match_parallel, dim3(8), dim3(BLOCK), 0, dids, dtable, NPAGES, dmatched);
  }, dev, 30, 8);

  load(EDITED);   // the scenario the report below describes

  std::printf("problem   : a %d-token agent context arriving at a cache of %d pages\n",
              NTOK, NCACHE);
  std::printf("cache     : the previous turn, diverging at page %d of %d "
              "(one token changed)\n", SHARED_PAGES, NPAGES);
  bench::header(dev);
  const double tol = 1e-6;
  bench::rows_out(rows, dev, tol);

  std::printf("\nEach variant answered three scenarios, not one, and its error above is the\n"
              "worst of the three:\n\n");
  std::printf("  %-26s %10s %10s %10s %10s %10s\n", "scenario", "correct", "tokens", "ids",
              "probe", "parallel");
  for (int i = 0; i < NS; ++i) {
    load(SCENARIOS[i]);
    std::printf("  %-26s %10d %10d %10d %10d %10d\n", SCENARIO_NAME[i], want,
                answers1[i], answers2[i], answers3[i], answers4[i]);
  }
  load(EDITED);
  std::printf("\n  The first row is the everyday case and it proves nothing about the hash — a\n"
              "  digest that ignores token order, or an id that is not chained over the whole\n"
              "  prefix, still finds this divergence. The other two exist because mutation\n"
              "  testing showed exactly that: both bugs were injected, both passed, and the\n"
              "  test learned nothing. \"Page reordered\" catches a position-blind digest;\n"
              "  \"page reused elsewhere\" catches an unchained id. Neither is exotic — the\n"
              "  first is a re-sorted list of tool definitions, the second is the same snippet\n"
              "  of a file appearing in two conversations.\n"
              "\n  The reference for all three is a host walk over the raw tokens, which knows\n"
              "  nothing about hashing. That is what makes it a reference rather than a second\n"
              "  opinion.\n");

  // -- what the answer is worth -----------------------------------------------------------
  const double PREFILL_TOK_PER_S = 20000.0;
  const int matched_tokens = want * PAGE;
  std::printf("\nWhat this lookup is worth on this request:\n\n");
  std::printf("  context                     %8d tokens\n", NTOK);
  std::printf("  matched prefix              %8d tokens  (%.1f%%)\n", matched_tokens,
              100.0 * matched_tokens / NTOK);
  std::printf("  prefill avoided             %8.3f s     at %.0fk tok/s\n",
              matched_tokens / PREFILL_TOK_PER_S, PREFILL_TOK_PER_S / 1000.0);
  std::printf("  ids the lookup touched      %8d      (%.1f KB)\n", NPAGES,
              NPAGES * sizeof(unsigned long long) / 1024.0);
  // Bound the lookup by something defensible rather than by its bandwidth. The bytes are
  // trivial; what a real lookup actually costs is a couple of kernel launches plus dependent
  // random accesses into a table too large for L2. ~20 us is a generous floor for that, and
  // the comparison still is not close.
  const double LOOKUP_FLOOR_S = 20e-6;
  std::printf("\n  The lookup reads %.1f KB. Even charging it a generous %.0f us — two kernel\n"
              "  launches and a scatter of dependent misses, rather than its bandwidth, which\n"
              "  is nothing — it decides whether to spend %.0f ms, about %.0fx more. It is the\n"
              "  highest-return code in the serving path, and unlike every other kernel here it\n"
              "  pays off *more* the slower your model is.\n",
              NPAGES * sizeof(unsigned long long) / 1024.0, LOOKUP_FLOOR_S * 1e6,
              1000.0 * matched_tokens / PREFILL_TOK_PER_S,
              (matched_tokens / PREFILL_TOK_PER_S) / LOOKUP_FLOOR_S);

  // -- where the block size bites ---------------------------------------------------------
  std::printf("\nWhy one edited token is expensive — the same context, one token changed at\n"
              "different positions (PAGE=%d):\n\n", PAGE);
  std::printf("  %18s %16s %16s\n", "edit at token", "prefix still valid", "must re-prefill");
  for (int pos : {0, PAGE, NTOK / 4, NTOK / 2, (3 * NTOK) / 4, NTOK - 1}) {
    const int valid = (pos / PAGE) * PAGE;
    std::printf("  %18d %16d %16d\n", pos, valid, NTOK - valid);
  }
  std::printf("\n  An edit at token 0 — a timestamp, a session id, a randomized greeting at the\n"
              "  top of a system prompt — costs the entire context. The same edit at the end\n"
              "  costs one page. Nothing in the kernel can recover that; it is decided by where\n"
              "  the volatile text sits in the prompt, which makes prompt *layout* a\n"
              "  performance decision rather than a stylistic one.\n");

  // -- and why a hit rate near 1 is worth chasing -------------------------------------------
  std::printf("\nA 20-turn agent — 3,700-token opening context, 950 tokens added per turn — at\n"
              "several prefix cache hit rates:\n\n");
  std::printf("  %10s %18s %14s\n", "hit rate", "prefill tokens", "vs perfect");
  {
    const int turns = 20, base = 3700, growth = 950;
    double ideal = 0;
    for (int i = 0; i < turns; ++i) ideal += (i == 0 ? base : growth);
    for (double h : {0.0, 0.50, 0.80, 0.95, 0.99, 1.0}) {
      double actual = 0;
      for (int i = 0; i < turns; ++i) {
        const double ctx = base + (double)i * growth;
        const double fresh = (i == 0 ? ctx : growth);
        actual += fresh + (1.0 - h) * (ctx - fresh);
      }
      std::printf("  %9.0f%% %18.0f %13.2fx\n", h * 100.0, actual, actual / ideal);
    }
  }
  std::printf("\n  The misses are misses on the whole context, not on the new tokens, so the\n"
              "  curve is far steeper than it looks: 95%% costs 1.5x the ideal and 99%% costs\n"
              "  1.1x. That gap is worth engineering for, and this kernel is where it is won.\n");

  for (void* p : {(void*)dtok, (void*)dcached, (void*)dhit, (void*)dmatched, (void*)ddig,
                  (void*)dids, (void*)dcache_ids, (void*)dtable})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
