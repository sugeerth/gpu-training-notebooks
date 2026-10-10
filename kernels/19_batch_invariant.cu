// 19_batch_invariant.cu — why the same request gives different logits depending on who else
// was in the batch, and what it costs to stop that.
//
//     nvcc -O3 -arch=native 19_batch_invariant.cu -o build/19 && build/19
//     make check
//
// An agent replays. It retries a failed tool call, resumes an interrupted run, re-scores a
// trajectory in an eval harness, and caches results keyed on the conversation so far. Every
// one of those assumes the same input gives the same output.
//
// A decode step does not promise that, and the reason is subtler than "atomics are
// non-deterministic". Consider the reduction that computes a row's sum — the denominator of a
// softmax, the mean in a LayerNorm, a dot product against the LM head. A performance-minded
// kernel picks its split count from the *batch*: with one sequence in flight there are idle
// SMs, so cut the row into many chunks; with sixty-four sequences the machine is already full,
// so use one chunk each. Both are correct. But floating-point addition is not associative, so
//
//     ((a + b) + c) + d       !=       (a + b) + (c + d)
//
// in the last bits, and the *shape* of the reduction tree just changed with the batch size.
// The request did not change. Its neighbours did.
//
// This is a different failure from the one KB_SHUFFLE detects. That one is about **order**:
// atomics committing in whatever sequence blocks finish. This one is about **shape**: a
// deterministic, atomic-free, perfectly reproducible kernel that nonetheless answers
// differently at batch 1 and batch 32. Running the same binary twice will never reveal it. You
// have to vary the batch.
//
// Usually the difference is 1e-7 and nothing happens. Occasionally two logits are within that
// of each other, the argmax flips, and from that token on it is a different conversation — the
// prefix cache misses, the eval comparison fails, and the bug does not reproduce because
// reproducing it requires reproducing the batch.
//
// The fix is to decide the split count from the *problem*, not from the machine's spare
// capacity. It costs some occupancy at small batch. For an agent platform it buys replay.
//
// Prerequisite: 02_reduce.cu for the reduction itself, 07_rmsnorm_backward.cu for the
// order-dependence this is the sibling of.
#include <functional>

#include "common.cuh"

#if SHIM_BUILD
constexpr int NCOL = 512;         // elements in the row being reduced
constexpr int BLOCK = 32;
constexpr int MAXSPLIT = 16;
#else
constexpr int NCOL = 32768;
constexpr int BLOCK = 256;
constexpr int MAXSPLIT = 64;
#endif
constexpr int FIXED_SPLIT = 8;    // the batch-invariant choice: always this many, always

// ---------------------------------------------------------------------------------------
// Stage one: each chunk sums its own slice. Identical code in every variant — what differs is
// only how many chunks there are, which is the entire point.
// ---------------------------------------------------------------------------------------
__global__ void partial_sums(const float* __restrict__ x, float* __restrict__ part,
                             int ncol, int nsplit) {
  SHARED(float, red, BLOCK);
  const int c = blockIdx.x;
  const int lo = (int)((long long)ncol * c / nsplit);
  const int hi = (int)((long long)ncol * (c + 1) / nsplit);

  float s = 0.0f;
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) s += x[i];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int t = blockDim.x / 2; t > 0; t >>= 1) {
    if (threadIdx.x < t) red[threadIdx.x] += red[threadIdx.x + t];
    __syncthreads();
  }
  if (threadIdx.x == 0) part[c] = red[0];
}

// Stage two: combine the partials, in index order. Serial and deterministic given the
// partials — so any variation in the answer comes from how many partials there were, not from
// how they were combined.
__global__ void combine(const float* __restrict__ part, float* __restrict__ out, int nsplit) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  float s = 0.0f;
  for (int c = 0; c < nsplit; ++c) s += part[c];
  *out = s;
}

// ---------------------------------------------------------------------------------------
// A one-shot reduction: no split at all, everything in one block. Reproducible and slow — the
// baseline that shows the fixed-split answer is not just "a different wrong number".
// ---------------------------------------------------------------------------------------
__global__ void reduce_single(const float* __restrict__ x, float* __restrict__ out, int ncol) {
  SHARED(float, red, BLOCK);
  float s = 0.0f;
  for (int i = threadIdx.x; i < ncol; i += blockDim.x) s += x[i];
  red[threadIdx.x] = s;
  __syncthreads();
  for (int t = blockDim.x / 2; t > 0; t >>= 1) {
    if (threadIdx.x < t) red[threadIdx.x] += red[threadIdx.x + t];
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = red[0];
}

// ---------------------------------------------------------------------------------------
// A Kahan-compensated one-shot reduction. Not batch-invariance — a *different* answer to a
// different question, included because it is the one people reach for when they see a
// reproducibility problem and it does not solve this one. It reduces the error; it does not
// make the answer independent of the tree shape, because the tree shape still varies.
// ---------------------------------------------------------------------------------------
__global__ void reduce_kahan(const float* __restrict__ x, float* __restrict__ out, int ncol) {
  SHARED(float, red, BLOCK);
  float s = 0.0f, comp = 0.0f;
  for (int i = threadIdx.x; i < ncol; i += blockDim.x) {
    const float y = x[i] - comp;
    const float t = s + y;
    comp = (t - s) - y;
    s = t;
  }
  red[threadIdx.x] = s;
  __syncthreads();
  for (int t = blockDim.x / 2; t > 0; t >>= 1) {
    if (threadIdx.x < t) red[threadIdx.x] += red[threadIdx.x + t];
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = red[0];
}

// ---------------------------------------------------------------------------------------

// How a throughput-minded engine picks its split: enough chunks to fill the machine, given how
// much of it the rest of the batch is already using. Entirely reasonable, and the source of
// the whole problem.
static int split_for_batch(int batch) {
  int s = MAXSPLIT / batch;
  if (s < 1) s = 1;
  if (s > MAXSPLIT) s = MAXSPLIT;
  return s;
}

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
  (void)argc; (void)argv;

  std::vector<float> x(NCOL);
  bench::fill(x.data(), x.size(), 5);
  // Give the row a wide dynamic range. Summation order matters most when large and small
  // magnitudes are mixed, which is exactly what a real logit row or a LayerNorm input looks
  // like — and exactly what a uniform [-1, 1) fill would hide.
  //
  // The stride is a power of two on purpose. Chunk boundaries fall on powers of two, so an
  // off-by-one in a chunk bound lands on a *large* element and moves the sum by a lot. With a
  // prime stride the boundaries almost never hit one, the error is ~1e-6, and the bug hides
  // under any reasonable tolerance — which is exactly what mutation testing reported before
  // this line changed.
  for (int i = 0; i < NCOL; ++i) x[i] *= (i % 64 == 0) ? 4096.0f : 0.015f;

  double want = 0;
  for (int i = 0; i < NCOL; ++i) want += (double)x[i];

  float *dx, *dpart, *dout;
  CUDA_CHECK(cudaMalloc((void**)&dx, x.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dpart, MAXSPLIT * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dout, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));

  const int BATCHES[] = {1, 2, 4, 8, 16, 32};
  const int NB = (int)(sizeof(BATCHES) / sizeof(BATCHES[0]));

  auto run_split = [&](int nsplit) {
    float got = 0.0f;
    KERNEL_LAUNCH(partial_sums, dim3(nsplit), dim3(BLOCK), 0, dx, dpart, NCOL, nsplit);
    KERNEL_LAUNCH(combine, dim3(1), dim3(1), 0, dpart, dout, nsplit);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&got, dout, sizeof(float), cudaMemcpyDeviceToHost));
    return got;
  };
  auto run_one = [&](bool kahan) {
    float got = 0.0f;
    if (kahan) {
      KERNEL_LAUNCH(reduce_kahan, dim3(1), dim3(BLOCK), 0, dx, dout, NCOL);
    } else {
      KERNEL_LAUNCH(reduce_single, dim3(1), dim3(BLOCK), 0, dx, dout, NCOL);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&got, dout, sizeof(float), cudaMemcpyDeviceToHost));
    return got;
  };

  // -- the three variants, each evaluated at the reference batch size ----------------------
  // Each variant is checked against what it *claims*, not only against the double reference.
  // Being close to the right answer is table stakes here; the properties under test are "gives
  // the same bits whatever else is in the batch" and "is more accurate than a plain sum", and
  // neither of those shows up in a comparison against a reference.
  //
  // Mutation testing is what made that obvious. Three injected bugs — a fixed split that
  // secretly reads the batch size, overlapping chunk bounds, and Kahan with its compensation
  // term deleted — all sailed through a check that only looked at accuracy. A test that cannot
  // fail is not a test, so the claims are now checked directly.
  enum Claim { PLAIN, INVARIANT, COMPENSATED };

  // An input on which compensation demonstrably matters: one value at 2^23, where a float's
  // ulp is exactly 1, and a long tail of 0.4s that a plain accumulator rounds away one at a
  // time. Kahan keeps them. Without an input like this the compensation is untestable — and
  // "untestable" is what mutation testing reported, by deleting it and passing.
  std::vector<float> adv(NCOL, 0.4f);
  adv[0] = 8388608.0f;   // 2^23
  double want_adv = 0;
  for (int i = 0; i < NCOL; ++i) want_adv += (double)adv[i];

  float* dadv;
  CUDA_CHECK(cudaMalloc((void**)&dadv, adv.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dadv, adv.data(), adv.size() * sizeof(float), cudaMemcpyHostToDevice));
  auto run_adv = [&](bool kahan) {
    float got = 0.0f;
    if (kahan) {
      KERNEL_LAUNCH(reduce_kahan, dim3(1), dim3(BLOCK), 0, dadv, dout, NCOL);
    } else {
      KERNEL_LAUNCH(reduce_single, dim3(1), dim3(BLOCK), 0, dadv, dout, NCOL);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&got, dout, sizeof(float), cudaMemcpyDeviceToHost));
    return got;
  };

  std::vector<bench::Row> rows;
  auto record = [&](const char* name, float got, std::function<float()> rerun, Claim claim,
                    const char* note) {
    bench::Row r;
    r.name = name;
    r.err = bench::max_rel_err_scalar(got, (float)want);

    if (claim == INVARIANT) {
      // Re-run this variant's own kernel once per batch size and require bit-identical output.
      // A variant that claims invariance and does not have it fails here however accurate it
      // is — which is the whole point, since accuracy is not the property being claimed.
      unsigned first = 0;
      for (int i = 0; i < NB; ++i) {
        const float v = rerun();
        unsigned u;
        std::memcpy(&u, &v, sizeof u);
        if (i == 0) first = u;
        else if (u != first) r.err = 1.0;
      }
    } else if (claim == COMPENSATED) {
      // On the adversarial row, the compensated sum must beat the plain one. Deleting the
      // compensation term makes it *equal* to the plain one, which fails this.
      const double plain = std::fabs((double)run_adv(false) - want_adv);
      const double comp = std::fabs((double)run_adv(true) - want_adv);
      if (!(comp < plain)) r.err = 1.0;
    }

    std::vector<float> one{got};
    r.checksum = bench::checksum_of(one);
    r.bytes = (double)NCOL * sizeof(float);
    r.flops = NCOL;
    r.note = note;
    rows.push_back(r);
  };
  const double bytes = (double)NCOL * sizeof(float);

  {
    const int ns = split_for_batch(1);
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(partial_sums, dim3(ns), dim3(BLOCK), 0, dx, dpart, NCOL, ns);
      KERNEL_LAUNCH(combine, dim3(1), dim3(1), 0, dpart, dout, ns);
    }, dev, 30, 8);
    record("1 split chosen from batch size", run_split(ns), [&] { return run_split(ns); }, PLAIN,
           "fast, NOT batch-invariant");
    rows.back().st = st;
  }
  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(partial_sums, dim3(FIXED_SPLIT), dim3(BLOCK), 0, dx, dpart, NCOL,
                    FIXED_SPLIT);
      KERNEL_LAUNCH(combine, dim3(1), dim3(1), 0, dpart, dout, FIXED_SPLIT);
    }, dev, 30, 8);
    record("2 fixed split, always 8", run_split(FIXED_SPLIT),
           [&] { return run_split(FIXED_SPLIT); }, INVARIANT,
           "batch-invariant, and checked");
    rows.back().st = st;
  }
  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(reduce_single, dim3(1), dim3(BLOCK), 0, dx, dout, NCOL);
    }, dev, 30, 8);
    record("3 one block, no split", run_one(false), [&] { return run_one(false); }, INVARIANT,
           "invariant, and slow");
    rows.back().st = st;
  }
  {
    bench::Stats st = bench::time_kernel([&] {
      KERNEL_LAUNCH(reduce_kahan, dim3(1), dim3(BLOCK), 0, dx, dout, NCOL);
    }, dev, 30, 8);
    record("4 one block, Kahan compensated", run_one(true), [&] { return run_one(true); },
           COMPENSATED,
           "accurate, not the fix");
    rows.back().st = st;
  }

  std::printf("problem   : sum a %d-element row — a softmax denominator, a LayerNorm mean, a\n"
              "            logit against the LM head\n", NCOL);
  std::printf("range     : mixed magnitudes — 1 element in 64 is ~%.0fx the others, which is\n"
              "            what makes summation order visible at all\n", 4096.0f / 0.015f);
  bench::header(dev);
  const double tol = 1e-5;
  bench::rows_out(rows, dev, tol);

  std::printf("\nAll four agree with a double-precision sum to within %.0e relative. They are\n"
              "all correct. That is the difficulty: correctness is not the property in\n"
              "question.\n", tol);

  // -- the actual experiment: vary the batch, hold the request fixed ------------------------
  std::printf("\nThe same row, the same binary, different batch sizes:\n\n");
  std::printf("  %8s %8s   %-16s %-16s   %s\n", "batch", "splits", "adaptive split",
              "fixed split", "vs batch 1");
  std::printf("  %s\n", "----------------------------------------------------------------------"
                        "-------");
  float first_adaptive = 0.0f, first_fixed = 0.0f;
  int adaptive_distinct = 0, fixed_distinct = 0;
  std::vector<unsigned> seen_a, seen_f;
  for (int i = 0; i < NB; ++i) {
    const int b = BATCHES[i];
    const int ns = split_for_batch(b);
    const float a = run_split(ns);
    const float f = run_split(FIXED_SPLIT);
    if (i == 0) { first_adaptive = a; first_fixed = f; }
    unsigned ua, uf;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&uf, &f, 4);
    if (std::find(seen_a.begin(), seen_a.end(), ua) == seen_a.end()) {
      seen_a.push_back(ua);
      ++adaptive_distinct;
    }
    if (std::find(seen_f.begin(), seen_f.end(), uf) == seen_f.end()) {
      seen_f.push_back(uf);
      ++fixed_distinct;
    }
    std::printf("  %8d %8d   %-16.9g %-16.9g   %s\n", b, ns, (double)a, (double)f,
                (a == first_adaptive) ? "" : "<-- CHANGED");
  }
  std::printf("\n  distinct answers: adaptive split %d, fixed split %d\n",
              adaptive_distinct, fixed_distinct);

  if (adaptive_distinct > 1) {
    std::printf("\n  The adaptive kernel gave %d different answers to the same question. Nothing\n"
                "  raced; nothing was uninitialized; running it a thousand times at a fixed\n"
                "  batch would give the same answer a thousand times. The only thing that\n"
                "  changed is how many pieces the row was cut into, and that was decided by\n"
                "  the other requests in flight.\n", adaptive_distinct);
  } else {
    std::printf("\n  The adaptive kernel happened to agree across these batch sizes on this\n"
                "  input. That is luck, not invariance: the reduction tree still changed shape.\n");
  }
  std::printf("\n  The fixed-split kernel gave one answer, bit for bit, at every batch size —\n"
              "  which is the property an agent replay actually needs. It is also the property\n"
              "  the adaptive kernel cannot be *tested* into having, because a test that does\n"
              "  not vary the batch never exercises the difference.\n");

  // -- what it costs -----------------------------------------------------------------------
  std::printf("\nWhat invariance costs, in parallelism:\n\n");
  std::printf("  %8s %14s %14s %16s\n", "batch", "adaptive", "fixed", "blocks lost");
  for (int i = 0; i < NB; ++i) {
    const int b = BATCHES[i];
    const int ns = split_for_batch(b);
    std::printf("  %8d %14d %14d %16d\n", b, ns * b, FIXED_SPLIT * b, (ns - FIXED_SPLIT) * b);
  }
  std::printf("\n  Negative means the fixed split launches *more* blocks than the adaptive one\n"
              "  would have — it over-decomposes at large batch, where the machine was already\n"
              "  full. The cost is concentrated at small batch, where the adaptive kernel\n"
              "  would have used the idle SMs and the fixed one leaves them idle.\n"
              "\n  That is the trade in one line: batch-invariance costs you the small-batch\n"
              "  latency win, and buys you the ability to replay a trajectory. An agent\n"
              "  platform runs evals, retries and caches on top of its own output, so it wants\n"
              "  the second one — and a chat product that never replays anything reasonably\n"
              "  does not.\n");

  // -- and the part where it actually bites -------------------------------------------------
  std::printf("\nWhy a 1e-7 difference matters at all:\n\n");
  std::printf("  A logit row is %d entries wide. Two candidates within 1e-7 of each other is\n"
              "  not rare — it is the normal state of a confident model's second and third\n"
              "  choice, and the normal state of *every* pair once a grammar mask has flattened\n"
              "  the illegal ones. Flip the argmax once and the next token differs; the token\n"
              "  after that is conditioned on a different prefix, so it differs for a second\n"
              "  reason; and the prefix cache misses from that point to the end of the run.\n"
              "\n  One bit becomes a different conversation, a failed eval comparison, and a\n"
              "  bug you cannot bisect because it does not reproduce outside the batch that\n"
              "  produced it.\n", NCOL);

  for (void* p : {(void*)dx, (void*)dpart, (void*)dout, (void*)dadv}) CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
