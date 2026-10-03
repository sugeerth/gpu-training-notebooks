// 10_collectives.cu — all-reduce, and the communication tax that shapes every parallelism plan.
//
//     nvcc -O3 -arch=native 10_collectives.cu -o build/10 && build/10
//     make check
//
// Data-parallel training ends every step by summing gradients across all ranks. For an 8B
// model in bf16 that is 16 GB of gradient, all-reduced, on every step. Whether that costs 5%
// or 50% of the step is decided by an algorithm, not by the network — and the algorithm is
// simple enough to write down here and check.
//
// The naive version — every rank fetches everyone else's buffer and sums locally — moves
// **R·N** bytes per rank. The ring version moves **2(R−1)/R · N**, which for R = 8 is 1.75·N
// against 8·N: a 4.6x reduction, and it gets better as R grows because the ring's factor tends
// to 2 while the naive one grows without bound. That is why every collectives library is built
// around reduce-scatter followed by all-gather.
//
// The ring, in two phases:
//
//   reduce-scatter (R−1 steps): rank r sends chunk (r−s) mod R to rank r+1, which adds it to
//                               its own copy. After R−1 steps rank r owns the fully summed
//                               chunk (r+1) mod R, and nobody else has a complete copy of it.
//   all-gather     (R−1 steps): the same ring, forwarding the finished chunks instead of
//                               accumulating them, until everyone has all of them.
//
// Each step moves N/R bytes, so 2(R−1) steps move 2(R−1)/R·N. Every link carries traffic at
// every step, which is what makes it bandwidth-optimal.
//
// What it is not is latency-optimal: 2(R−1) sequential steps means 14 round trips at R = 8.
// For a small tensor the whole transfer is latency, and a direct algorithm that finishes in 2
// steps wins while moving exactly the same bytes. Under the standard alpha-beta cost model the
// direct version is therefore never slower — which raises the obvious question of why rings
// are the default, and the answer is topology rather than arithmetic. The cost model at the
// end of this file works it through.
//
// The ranks here are simulated as R buffers on one device, so the *algorithm* and its exact
// arithmetic are what get checked; the bandwidth numbers come from the cost model rather than
// from a real interconnect.
#include "common.cuh"

#if SHIM_BUILD
constexpr int RANKS = 4;
constexpr int BLOCK = 64;
constexpr int GRID = 8;
#else
constexpr int RANKS = 8;
constexpr int BLOCK = 256;
constexpr int GRID = 512;
#endif

// ---------------------------------------------------------------------------------------
// Variant 1: every rank sums every other rank's buffer.
//
// One step, trivially correct, and it moves R·N bytes into every rank. At R = 8 that is 8x
// the payload; at R = 64 it is 64x. The traffic grows linearly in the number of ranks, which
// is the property that makes it unusable at scale rather than merely slow.
// ---------------------------------------------------------------------------------------
__global__ void allreduce_naive(const float* __restrict__ orig, float* __restrict__ data,
                                int R, size_t N) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
    float acc = 0.0f;
    for (int k = 0; k < R; ++k) acc += orig[(size_t)k * N + i];   // fixed order: reproducible
    for (int r = 0; r < R; ++r) data[(size_t)r * N + i] = acc;
  }
}

// ---------------------------------------------------------------------------------------
// Variant 2, phase 1: one reduce-scatter step of the ring.
//
// Rank r sends chunk c = (r − s) mod R to rank r+1, which accumulates it. All R sends happen
// in the same kernel because they touch disjoint memory: the chunk rank r is *reading* is
// (r−s) mod R, and the chunk anyone writes into rank r is (r−1−s) mod R. Different index, no
// race, no barrier needed — which is exactly why the ring maps onto real hardware where the
// ranks genuinely are simultaneous.
// ---------------------------------------------------------------------------------------
__global__ void ring_reduce_scatter(float* __restrict__ data, int R, size_t N, size_t chunk,
                                    int s) {
  const size_t total = (size_t)R * chunk;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const int r = (int)(idx / chunk);
    const size_t j = idx % chunk;
    const int c = ((r - s) % R + R) % R;
    const int dst = (r + 1) % R;
    data[(size_t)dst * N + (size_t)c * chunk + j] += data[(size_t)r * N + (size_t)c * chunk + j];
  }
}

// Phase 2: the same ring, forwarding finished chunks instead of accumulating them.
__global__ void ring_all_gather(float* __restrict__ data, int R, size_t N, size_t chunk,
                                int s) {
  const size_t total = (size_t)R * chunk;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const int r = (int)(idx / chunk);
    const size_t j = idx % chunk;
    const int c = ((r + 1 - s) % R + R) % R;
    const int dst = (r + 1) % R;
    data[(size_t)dst * N + (size_t)c * chunk + j] = data[(size_t)r * N + (size_t)c * chunk + j];
  }
}

// ---------------------------------------------------------------------------------------
// Variant 3: direct reduce-scatter and direct all-gather — two steps instead of 2(R−1).
//
// Rank r owns chunk r. In one step it pulls that chunk from every other rank and sums;
// in a second step everyone reads every owner's finished chunk. The bytes moved per rank are
// identical to the ring's 2(R−1)/R·N — every rank still sends and receives (R−1)/R of the
// payload in each phase — but it completes in 2 synchronisations rather than 14.
//
// The catch is not in the byte count — it is in the topology. This needs every rank talking
// to R−1 partners at once. Inside a node, where NVLink gives every pair a fast path, that is
// free and this algorithm wins outright. Across nodes, where many GPUs share a smaller number
// of NICs, all-to-all traffic congests the links the ring would have used one at a time.
//
// So the rule is not "ring is better" or "direct is better"; it is that the ring's structure
// survives being embedded in a hierarchy and the direct algorithm's does not. NCCL chooses by
// message size and topology, and the model at the end of this file shows both terms.
// ---------------------------------------------------------------------------------------
__global__ void direct_reduce_scatter(const float* __restrict__ orig, float* __restrict__ data,
                                      int R, size_t N, size_t chunk) {
  const size_t total = (size_t)R * chunk;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const int r = (int)(idx / chunk);          // rank r owns chunk r
    const size_t j = idx % chunk;
    float acc = 0.0f;
    for (int k = 0; k < R; ++k) acc += orig[(size_t)k * N + (size_t)r * chunk + j];
    data[(size_t)r * N + (size_t)r * chunk + j] = acc;
  }
}

__global__ void direct_all_gather(float* __restrict__ data, int R, size_t N, size_t chunk) {
  const size_t total = (size_t)R * chunk;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const int c = (int)(idx / chunk);          // chunk c, owned by rank c
    const size_t j = idx % chunk;
    const float v = data[(size_t)c * N + (size_t)c * chunk + j];
    for (int r = 0; r < R; ++r)
      if (r != c) data[(size_t)r * N + (size_t)c * chunk + j] = v;
  }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  size_t N = 1024;
#else
  size_t N = argc > 1 ? (size_t)std::atoll(argv[1]) : (1u << 22);   // 4M floats per rank
#endif
  (void)argc; (void)argv;
  const int R = RANKS;
  N = (N / R) * R;
  const size_t chunk = N / R;

  std::vector<float> horig((size_t)R * N), hdata((size_t)R * N);
  bench::fill(horig.data(), horig.size(), 9);

  // Reference: the elementwise sum across ranks, in double. Every rank must end with this.
  std::vector<float> want(N);
  for (size_t i = 0; i < N; ++i) {
    double acc = 0.0;
    for (int r = 0; r < R; ++r) acc += horig[(size_t)r * N + i];
    want[i] = (float)acc;
  }

  float *dorig, *ddata;
  CUDA_CHECK(cudaMalloc((void**)&dorig, horig.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&ddata, horig.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dorig, horig.data(), horig.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<bench::Row> rows;
  const double payload = (double)N * sizeof(float);

  auto run = [&](const char* name, double per_rank_bytes, int steps, auto&& launch) {
    CUDA_CHECK(cudaMemcpy(ddata, dorig, horig.size() * sizeof(float), cudaMemcpyDeviceToDevice));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel([&] {
      CUDA_CHECK(cudaMemcpy(ddata, dorig, horig.size() * sizeof(float),
                            cudaMemcpyDeviceToDevice));
      launch();
    }, dev, 20, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdata.data(), ddata, horig.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // Every rank must hold the complete sum — not just rank 0. A reduce-scatter that never
    // gathers passes a check that only looks at one rank.
    double worst = 0;
    for (int rk = 0; rk < R; ++rk)
      worst = std::max(worst, bench::max_rel_err(&hdata[(size_t)rk * N], want.data(), N));
    r.err = worst;
    r.checksum = bench::checksum_of(hdata);
    r.bytes = per_rank_bytes * R;
    r.flops = (double)(R - 1) * N;
    char buf[56];
    std::snprintf(buf, sizeof buf, "%.2fxN per rank, %d step%s", per_rank_bytes / payload,
                  steps, steps == 1 ? "" : "s");
    r.note = buf;
    rows.push_back(r);
  };

  std::printf("problem   : %d ranks x %zu floats (%.1f MB per rank), chunk %zu\n", R, N,
              payload / 1048576.0, chunk);
  bench::header(dev);

  run("1 naive gather-and-sum", R * payload, 1, [&] {
    KERNEL_LAUNCH(allreduce_naive, dim3(GRID), dim3(BLOCK), 0, dorig, ddata, R, N);
  });
  run("2 ring RS + AG", 2.0 * (R - 1) / R * payload, 2 * (R - 1), [&] {
    for (int s = 0; s < R - 1; ++s)
      KERNEL_LAUNCH(ring_reduce_scatter, dim3(GRID), dim3(BLOCK), 0, ddata, R, N, chunk, s);
    for (int s = 0; s < R - 1; ++s)
      KERNEL_LAUNCH(ring_all_gather, dim3(GRID), dim3(BLOCK), 0, ddata, R, N, chunk, s);
  });
  run("3 direct RS + AG", 2.0 * (R - 1) / R * payload, 2, [&] {
    KERNEL_LAUNCH(direct_reduce_scatter, dim3(GRID), dim3(BLOCK), 0, dorig, ddata, R, N, chunk);
    KERNEL_LAUNCH(direct_all_gather, dim3(GRID), dim3(BLOCK), 0, ddata, R, N, chunk);
  });

  const double tol = 1e-5;
  bench::rows_out(rows, dev, tol);

  // ---- the cost model, which is the part you can act on --------------------------------
  //
  // The standard alpha-beta model: time = steps x latency + bytes_per_rank / bandwidth.
  //
  //     algorithm          steps        bytes moved per rank
  //     naive gather         1          R x n
  //     ring RS + AG      2(R-1)        2(R-1)/R x n
  //     direct RS + AG       2          2(R-1)/R x n
  //
  // Note what that says and does not say. Ring and direct move *exactly the same bytes*; they
  // differ only in how many round trips they serialize. So under this model the direct
  // algorithm is never slower, and for a small tensor — where the whole cost is latency — it
  // is dramatically faster. Only the naive version is beaten on bandwidth, and it is beaten
  // by a factor that grows with R.
  std::printf("\nAll-reduce cost model — time = steps x latency + bytes per rank / bandwidth\n");
  std::printf("(ring and direct move identical bytes; they differ only in serialized steps)\n\n");
  struct Link { const char* name; double gbps; double lat_us; };
  const Link links[] = {
      {"NVLink 4 (H100, intra-node)", 450.0, 3.0},
      {"PCIe 5 x16",                   55.0, 8.0},
      {"InfiniBand NDR 400G",          50.0, 5.0},
      {"100 GbE",                      12.0, 30.0},
  };
  std::printf("  %-30s %11s %11s %11s %11s\n", "link", "1 MB ring", "1 MB direct",
              "1 GB ring", "1 GB direct");
  for (const Link& L : links) {
    auto t = [&](double bytes, int steps) {
      return 1e6 * (2.0 * (R - 1) / R * bytes / (L.gbps * 1e9)) + steps * L.lat_us;  // us
    };
    std::printf("  %-30s %9.1f us %9.1f us %9.1f us %9.1f us\n", L.name,
                t(1e6, 2 * (R - 1)), t(1e6, 2), t(1e9, 2 * (R - 1)), t(1e9, 2));
  }
  std::printf("\n  What fraction of the ring's time is pure latency:\n");
  for (const Link& L : links) {
    auto frac = [&](double bytes) {
      const double lat = 2.0 * (R - 1) * L.lat_us;
      const double bw = 1e6 * (2.0 * (R - 1) / R * bytes / (L.gbps * 1e9));
      return 100.0 * lat / (lat + bw);
    };
    std::printf("    %-30s %5.1f%% at 1 MB, %5.2f%% at 1 GB\n", L.name, frac(1e6), frac(1e9));
  }

  std::printf(
      "\n  So why is the ring the default? Because this model has one link per rank and real\n"
      "  clusters do not. The ring visits each link exactly once per step, so it can be\n"
      "  embedded on a hierarchical network — GPUs within a node, then nodes within a rack —\n"
      "  without any link carrying more than its share. The direct algorithm needs every rank\n"
      "  talking to R-1 partners at once, which is fine across NVLink where every pair has a\n"
      "  fast path, and congests the moment the traffic has to cross a smaller number of\n"
      "  inter-node links.\n"
      "\n  The practical rule, and what NCCL implements: latency-optimal algorithms (direct,\n"
      "  or a recursive-halving tree at 2*log2(R) steps) for small tensors and within a node;\n"
      "  bandwidth-optimal rings for large tensors across nodes. The switch happens by message\n"
      "  size, and the table above is why.\n");

  std::printf("\n  For an 8B model in bf16 the gradient is 16 GB, all-reduced every step.\n");
  for (const Link& L : links) {
    const double sec = 2.0 * (R - 1) / R * 16e9 / (L.gbps * 1e9);
    std::printf("    %-30s %6.2f s per step of pure communication\n", L.name, sec);
  }
  std::printf("\n  That number is why gradient bucketing overlaps the all-reduce with the\n"
              "  backward pass, why ZeRO shards the optimizer instead of replicating it, and\n"
              "  why the parallelism plan is chosen by the interconnect and not by the model.\n");

  CUDA_CHECK(cudaFree(dorig));
  CUDA_CHECK(cudaFree(ddata));
  return bench::verdict(rows, tol, &dev);
}
