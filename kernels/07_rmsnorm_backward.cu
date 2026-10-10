// 07_rmsnorm_backward.cu — the backward pass, and where training stops being reproducible.
//
//     nvcc -O3 -arch=native 07_rmsnorm_backward.cu -o build/07 && build/07
//     make check
//
// Every kernel before this one was inference. Training runs the same layers backwards, and the
// backward pass is not simply "the forward pass again":
//
//   * it costs about **2x the FLOPs** of the forward pass, because each input needs a gradient
//     and each parameter needs a gradient
//   * it has a **different memory pattern**: dx is elementwise per row, but dW is a *reduction
//     across the whole batch*, so every row of every sequence contributes to the same H floats
//   * it needs the forward pass's intermediates, which is why activation memory exists and why
//     checkpointing is a real trade rather than a free win
//
// The maths. With y_i = x_i · r · w_i and r = (mean(x²) + eps)^(-1/2) over H:
//
//     dW_i = Σ_rows dy_i · x_i · r                    <- a reduction over the batch
//     dX_i = r · (dy_i·w_i − x_i · r² · S / H)        <- elementwise, given S
//     where S = Σ_j dy_j · w_j · x_j                  <- a reduction over the row
//
// That dW reduction is the interesting part, and it is the subject of this file.
//
// The reproducibility problem
// ---------------------------
// dW sums a contribution from every row in the batch. The obvious implementation is one
// atomicAdd per element per row. It is fast, it is correct, and **it is not reproducible**:
// floating-point addition is not associative, so the answer depends on the order the blocks
// happen to finish in, which depends on how busy the GPU was. Two runs of the same training
// step on the same data give weights that differ in the last bits, those differences compound
// over thousands of steps, and "why does my loss curve not reproduce" has its answer.
//
// Variant 2 fixes it with a fixed-shape two-stage reduction: each block accumulates privately,
// then a second kernel sums the partials in a fixed order. It costs one extra kernel launch
// and G·H floats of scratch, and it makes the run bit-reproducible.
//
// This is not a hypothetical distinction — it is the same mechanism behind batch-invariance
// failures at serving time, where a request's logits depend on what else was in its batch.
// kernelbench checks it directly: it re-runs each kernel under a shuffled block order and
// compares exact checksums, so variant 1 is *required* to be order-dependent here and
// variants 2-4 are *required* not to be.
#include "common.cuh"

#if SHIM_BUILD
constexpr int BLOCK = 64;
constexpr int MAX_PER_THREAD = 4;
constexpr int GRID = 4;
#else
constexpr int BLOCK = 256;
constexpr int MAX_PER_THREAD = 16;    // hidden sizes up to 256 * 16 = 4096
constexpr int GRID = 256;             // blocks cooperating on the dW reduction
#endif
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down_sync(FULL, v, off);
  return v;
}

__device__ __forceinline__ float block_sum(float v, float* scratch) {
  const int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  v = warp_reduce_sum(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  const int nwarps = blockDim.x / warpSize;
  if (warp == 0) {
    v = (lane < nwarps) ? scratch[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  const float out = scratch[0];
  __syncthreads();
  return out;
}

// ---------------------------------------------------------------------------------------
// Variant 1: atomics for dW. Fast, correct, and not reproducible.
//
// One block per row, grid-strided over the batch. dX is written independently per row so it
// is deterministic; dW is not, because `atomicAdd` commits in whatever order blocks reach it.
//
// Note that nothing here is *wrong*. Every ordering produces a valid answer, and the spread
// between them is at the level of fp32 rounding. It only matters because a training run
// compounds it, and because "bit-identical given the same inputs" is the property that makes
// a regression bisectable.
// ---------------------------------------------------------------------------------------
__global__ void bwd_atomic(const float* __restrict__ x, const float* __restrict__ dy,
                           const float* __restrict__ w, float* __restrict__ dx,
                           float* __restrict__ dw, int R, int H, float eps) {
  SHARED(float, scratch, 32);
  const int tid = threadIdx.x;

  for (int row = blockIdx.x; row < R; row += gridDim.x) {
    const float* xr = x + (size_t)row * H;
    const float* gr = dy + (size_t)row * H;
    float* dxr = dx + (size_t)row * H;

    float ss = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) ss += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(ss, scratch) / H + eps);

    float s = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) s += gr[i] * w[i] * xr[i];
    const float S = block_sum(s, scratch);

    for (int i = tid; i < H; i += blockDim.x) {
      dxr[i] = r * (gr[i] * w[i] - xr[i] * r * r * S / H);
      atomicAdd(&dw[i], gr[i] * xr[i] * r);        // <- the non-reproducible line
    }
  }
}

// ---------------------------------------------------------------------------------------
// Variant 2: deterministic dW, in two stages.
//
// Each block keeps its own private dW accumulator in registers across all the rows it owns,
// writes it once to `partial[blockIdx.x][H]`, and a second kernel sums the G partials in a
// fixed order. Same arithmetic, same result every time, on any GPU, under any load.
//
// The cost is G·H floats of scratch (256 · 4096 · 4 B = 4 MB here) and one extra launch. That
// is what reproducible training costs, and it is cheap enough that the usual reason people do
// not have it is that nobody asked.
// ---------------------------------------------------------------------------------------
__global__ void bwd_partial(const float* __restrict__ x, const float* __restrict__ dy,
                            const float* __restrict__ w, float* __restrict__ dx,
                            float* __restrict__ partial, int R, int H, float eps) {
  SHARED(float, scratch, 32);
  const int tid = threadIdx.x;

  float acc[MAX_PER_THREAD];
#pragma unroll
  for (int k = 0; k < MAX_PER_THREAD; ++k) acc[k] = 0.0f;

  for (int row = blockIdx.x; row < R; row += gridDim.x) {
    const float* xr = x + (size_t)row * H;
    const float* gr = dy + (size_t)row * H;
    float* dxr = dx + (size_t)row * H;

    float ss = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) ss += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(ss, scratch) / H + eps);

    float s = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) s += gr[i] * w[i] * xr[i];
    const float S = block_sum(s, scratch);

    int k = 0;
    for (int i = tid; i < H; i += blockDim.x, ++k) {
      dxr[i] = r * (gr[i] * w[i] - xr[i] * r * r * S / H);
      acc[k] += gr[i] * xr[i] * r;                 // private, so no ordering to depend on
    }
  }

  int k = 0;
  for (int i = tid; i < H; i += blockDim.x, ++k) partial[(size_t)blockIdx.x * H + i] = acc[k];
}

// Stage two. A fixed loop over a fixed number of partials, in index order: the same sum on
// every run, on every machine.
__global__ void reduce_partials(const float* __restrict__ partial, float* __restrict__ dw,
                                int G, int H) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < H; i += gridDim.x * blockDim.x) {
    float acc = 0.0f;
    for (int g = 0; g < G; ++g) acc += partial[(size_t)g * H + i];
    dw[i] = acc;
  }
}

// ---------------------------------------------------------------------------------------
// Variant 3: the same, with the row held in registers and 128-bit accesses.
//
// The backward pass reads x and dy and writes dx — three trips over the activations, against
// the forward pass's two. Keeping x in registers across the two reductions saves re-reading
// it, exactly as in the forward kernel, and float4 does what it did in 01_copy.cu.
// ---------------------------------------------------------------------------------------
__global__ void bwd_partial_vec4(const float4* __restrict__ x, const float4* __restrict__ dy,
                                 const float4* __restrict__ w, float4* __restrict__ dx,
                                 float* __restrict__ partial, int R, int H4, float eps) {
  SHARED(float, scratch, 32);
  const int tid = threadIdx.x;
  const int H = 4 * H4;
  constexpr int V = MAX_PER_THREAD / 4 + 1;

  float4 accw[V];
#pragma unroll
  for (int k = 0; k < V; ++k) accw[k] = make_float4(0.f, 0.f, 0.f, 0.f);

  for (int row = blockIdx.x; row < R; row += gridDim.x) {
    const float4* xr = x + (size_t)row * H4;
    const float4* gr = dy + (size_t)row * H4;
    float4* dxr = dx + (size_t)row * H4;

    float4 xv[V], gv[V];
    float ss = 0.0f;
    int k = 0;
    for (int i = tid; i < H4; i += blockDim.x, ++k) {
      float4 t = xr[i];
      xv[k] = t;
      ss += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
    }
    const float r = rsqrtf(block_sum(ss, scratch) / H + eps);

    float s = 0.0f;
    k = 0;
    for (int i = tid; i < H4; i += blockDim.x, ++k) {
      float4 g = gr[i], wv = w[i], t = xv[k];
      gv[k] = g;
      s += g.x * wv.x * t.x + g.y * wv.y * t.y + g.z * wv.z * t.z + g.w * wv.w * t.w;
    }
    const float S = block_sum(s, scratch);
    const float c = r * r * S / H;

    k = 0;
    for (int i = tid; i < H4; i += blockDim.x, ++k) {
      float4 t = xv[k], g = gv[k], wv = w[i];
      dxr[i] = make_float4(r * (g.x * wv.x - t.x * c), r * (g.y * wv.y - t.y * c),
                           r * (g.z * wv.z - t.z * c), r * (g.w * wv.w - t.w * c));
      accw[k].x += g.x * t.x * r;
      accw[k].y += g.y * t.y * r;
      accw[k].z += g.z * t.z * r;
      accw[k].w += g.w * t.w * r;
    }
  }

  int k = 0;
  for (int i = tid; i < H4; i += blockDim.x, ++k) {
    float* p = partial + (size_t)blockIdx.x * H + 4 * i;
    p[0] = accw[k].x; p[1] = accw[k].y; p[2] = accw[k].z; p[3] = accw[k].w;
  }
}

// ---------------------------------------------------------------------------------------
// Variant 4: reuse `rstd` saved by the forward pass instead of recomputing it.
//
// This is the activation-checkpointing trade in miniature. The forward pass already computed
// r for every row; saving it costs R floats (32 KB for 8192 rows) and removes an entire
// reduction pass over x from the backward. Recomputing costs one pass over the activations —
// which, on a memory-bound kernel, is the whole cost.
//
// Scaled up, that is the same decision as gradient checkpointing: store an intermediate, or
// recompute it. The answer depends on which side of the roofline you are on, and for
// normalization statistics — tiny to store, a full pass to recompute — storing wins easily.
// ---------------------------------------------------------------------------------------
__global__ void bwd_saved_rstd(const float4* __restrict__ x, const float4* __restrict__ dy,
                               const float4* __restrict__ w, const float* __restrict__ rstd,
                               float4* __restrict__ dx, float* __restrict__ partial, int R,
                               int H4) {
  SHARED(float, scratch, 32);
  const int tid = threadIdx.x;
  const int H = 4 * H4;
  constexpr int V = MAX_PER_THREAD / 4 + 1;

  float4 accw[V];
#pragma unroll
  for (int k = 0; k < V; ++k) accw[k] = make_float4(0.f, 0.f, 0.f, 0.f);

  for (int row = blockIdx.x; row < R; row += gridDim.x) {
    const float4* xr = x + (size_t)row * H4;
    const float4* gr = dy + (size_t)row * H4;
    float4* dxr = dx + (size_t)row * H4;
    const float r = rstd[row];                   // <- one load instead of a reduction

    float4 xv[V], gv[V];
    float s = 0.0f;
    int k = 0;
    for (int i = tid; i < H4; i += blockDim.x, ++k) {
      float4 t = xr[i], g = gr[i], wv = w[i];
      xv[k] = t;
      gv[k] = g;
      s += g.x * wv.x * t.x + g.y * wv.y * t.y + g.z * wv.z * t.z + g.w * wv.w * t.w;
    }
    const float S = block_sum(s, scratch);
    const float c = r * r * S / H;

    k = 0;
    for (int i = tid; i < H4; i += blockDim.x, ++k) {
      float4 t = xv[k], g = gv[k], wv = w[i];
      dxr[i] = make_float4(r * (g.x * wv.x - t.x * c), r * (g.y * wv.y - t.y * c),
                           r * (g.z * wv.z - t.z * c), r * (g.w * wv.w - t.w * c));
      accw[k].x += g.x * t.x * r;
      accw[k].y += g.y * t.y * r;
      accw[k].z += g.z * t.z * r;
      accw[k].w += g.w * t.w * r;
    }
  }

  int k = 0;
  for (int i = tid; i < H4; i += blockDim.x, ++k) {
    float* p = partial + (size_t)blockIdx.x * H + 4 * i;
    p[0] = accw[k].x; p[1] = accw[k].y; p[2] = accw[k].z; p[3] = accw[k].w;
  }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  int R = 8, H = 256;
#else
  int R = argc > 1 ? std::atoi(argv[1]) : 8192;
  int H = argc > 2 ? std::atoi(argv[2]) : 4096;
#endif
  (void)argc; (void)argv;
  if (H % (4 * BLOCK) != 0) H = ((H / (4 * BLOCK)) + 1) * 4 * BLOCK;
  if (H / BLOCK > MAX_PER_THREAD) {
    std::printf("hidden size %d needs %d registers per thread, budget %d\n", H, H / BLOCK,
                MAX_PER_THREAD);
    return 1;
  }
  const float eps = 1e-6f;
  const size_t NEL = (size_t)R * H;
  const int G = GRID < R ? GRID : R;

  std::vector<float> hx(NEL), hdy(NEL), hw(H), hdx(NEL), hdw(H), hrstd(R);
  bench::fill(hx.data(), NEL, 1);
  bench::fill(hdy.data(), NEL, 2);
  bench::fill(hw.data(), H, 3);
  for (int i = 0; i < H; ++i) hw[i] = 1.0f + 0.1f * hw[i];

  // Reference in double, plus the rstd the forward pass would have saved.
  std::vector<float> want_dx(NEL), want_dw(H, 0.0f);
  {
    std::vector<double> acc(H, 0.0);
    for (int r = 0; r < R; ++r) {
      double ss = 0.0;
      for (int i = 0; i < H; ++i) ss += (double)hx[(size_t)r * H + i] * hx[(size_t)r * H + i];
      const double rr = 1.0 / std::sqrt(ss / H + eps);
      hrstd[r] = (float)rr;
      double S = 0.0;
      for (int i = 0; i < H; ++i)
        S += (double)hdy[(size_t)r * H + i] * hw[i] * hx[(size_t)r * H + i];
      for (int i = 0; i < H; ++i) {
        double xi = hx[(size_t)r * H + i], gi = hdy[(size_t)r * H + i];
        want_dx[(size_t)r * H + i] = (float)(rr * (gi * hw[i] - xi * rr * rr * S / H));
        acc[i] += gi * xi * rr;
      }
    }
    for (int i = 0; i < H; ++i) want_dw[i] = (float)acc[i];
  }

  float *dX, *dDY, *dW, *dDX, *dDW, *dPart, *dRstd;
  CUDA_CHECK(cudaMalloc((void**)&dX, NEL * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dDY, NEL * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dW, (size_t)H * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dDX, NEL * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dDW, (size_t)H * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dPart, (size_t)G * H * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dRstd, (size_t)R * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dX, hx.data(), NEL * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dDY, hdy.data(), NEL * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW, hw.data(), (size_t)H * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dRstd, hrstd.data(), (size_t)R * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<bench::Row> rows;
  const double B = sizeof(float);
  // Compulsory traffic: read x, dy, w; write dx; and the dW reduction's partials.
  const double traffic = 3.0 * NEL * B + 2.0 * H * B;

  auto run = [&](const char* name, double extra, double flops, auto&& launch) {
    CUDA_CHECK(cudaMemset(dDX, 0, NEL * sizeof(float)));
    CUDA_CHECK(cudaMemset(dDW, 0, (size_t)H * sizeof(float)));
    CUDA_CHECK(cudaMemset(dPart, 0, (size_t)G * H * sizeof(float)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev, 20, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdx.data(), dDX, NEL * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdw.data(), dDW, (size_t)H * sizeof(float), cudaMemcpyDeviceToHost));
    r.err = std::max(bench::max_rel_err(hdx.data(), want_dx.data(), NEL),
                     bench::max_rel_err(hdw.data(), want_dw.data(), H));
    // Both outputs go into the checksum: a kernel that gets dX right and dW wrong is a kernel
    // whose model trains to a different place, and it must not be able to hide.
    r.checksum = bench::checksum_of(hdx) * 1099511628211ull ^ bench::checksum_of(hdw);
    r.bytes = traffic + extra;
    r.flops = flops;
    rows.push_back(r);
  };

  std::printf("problem   : %d rows x %d hidden, %d blocks cooperating on dW\n", R, H, G);
  std::printf("scratch   : deterministic dW needs %d x %d floats = %.2f MB of partials\n", G, H,
              (double)G * H * 4 / 1048576.0);
  bench::header(dev);

  // ~9 FLOPs per element: two reductions and the dX expression.
  const double flops = 9.0 * NEL;
  run("1 atomic dW (not reproducible)", 0.0, flops, [&] {
    KERNEL_LAUNCH(bwd_atomic, dim3(G), dim3(BLOCK), 0, dX, dDY, dW, dDX, dDW, R, H, eps);
  });
  run("2 deterministic dW", 2.0 * G * H * B, flops, [&] {
    KERNEL_LAUNCH(bwd_partial, dim3(G), dim3(BLOCK), 0, dX, dDY, dW, dDX, dPart, R, H, eps);
    KERNEL_LAUNCH(reduce_partials, dim3(64), dim3(BLOCK), 0, dPart, dDW, G, H);
  });
  run("3 + float4, x in registers", 2.0 * G * H * B, flops, [&] {
    KERNEL_LAUNCH(bwd_partial_vec4, dim3(G), dim3(BLOCK), 0, (const float4*)dX,
                  (const float4*)dDY, (const float4*)dW, (float4*)dDX, dPart, R, H / 4, eps);
    KERNEL_LAUNCH(reduce_partials, dim3(64), dim3(BLOCK), 0, dPart, dDW, G, H);
  });
  // Saves a whole reduction pass over x, at the cost of R floats carried from the forward.
  run("4 + rstd saved from forward", 2.0 * G * H * B + R * B, 7.0 * NEL, [&] {
    KERNEL_LAUNCH(bwd_saved_rstd, dim3(G), dim3(BLOCK), 0, (const float4*)dX,
                  (const float4*)dDY, (const float4*)dW, dRstd, (float4*)dDX, dPart, R, H / 4);
    KERNEL_LAUNCH(reduce_partials, dim3(64), dim3(BLOCK), 0, dPart, dDW, G, H);
  });

  const double tol = 2e-4;   // dW sums R terms in fp32; the reference sums them in double
  bench::rows_out(rows, dev, tol);

  std::printf("\nVariant 1's dW is a valid answer computed in an arbitrary order. Variants 2-4\n"
              "produce the same answer every time, for %.2f MB of scratch and one extra launch.\n"
              "kernelbench verifies exactly that: it re-runs each variant under a shuffled\n"
              "block order and requires variant 1 to change and the others not to.\n",
              (double)G * H * 4 / 1048576.0);

  for (void* p : {(void*)dX, (void*)dDY, (void*)dW, (void*)dDX, (void*)dDW, (void*)dPart,
                  (void*)dRstd})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
