// 09_fp8_scaling.cu — FP8 training numerics: scaling granularity, saturation, and the two
// ways an fp8 run silently stops learning.
//
//     nvcc -O3 -arch=native 09_fp8_scaling.cu -o build/09 && build/09
//     make check
//
// FP8 halves the bytes again from bf16, which by the argument in 05_dequant_gemv.cu is the
// only thing that makes a memory-bound kernel faster, and it doubles tensor-core throughput
// on Hopper and Blackwell. It is also the first format where **the scale factor is part of
// the algorithm** rather than an implementation detail, because the dynamic range is so small
// that an unscaled tensor will not fit in it.
//
// The two formats, and why there are two:
//
//     e4m3   4 exponent bits, 3 mantissa   max ±448      ~2 decimal digits   -> forward pass
//     e5m2   5 exponent bits, 2 mantissa   max ±57344    ~1.5 digits         -> gradients
//
// Activations and weights are narrow in range and want the precision, so they get e4m3.
// Gradients span many orders of magnitude — and the small ones matter — so they get the extra
// exponent bit and give up a mantissa bit for it. That split is not a convention, it is the
// whole reason the two formats exist.
//
// The scale is what makes it work: store `x/s` in fp8 and remember `s`, choosing s so the
// largest magnitude in the group lands near 448. Everything below is about the granularity of
// that choice, and about what happens when it is wrong.
//
// Two failure modes, both silent:
//
//   * **saturation** — the scale is too small, large values clamp to ±448, and gradients from
//     the biggest activations are truncated. Loss keeps decreasing, slightly wrong.
//   * **underflow** — the scale is too large, small values round to zero, and a whole
//     channel's gradient disappears. This is what loss scaling exists to prevent, and it is
//     why fp16 training needed a scaler while bf16 did not.
//
// Neither raises an error. Both are visible only if you count them, which is what the
// saturation counters in variant 4 are for.
//
// The fp8 conversion here is done in software, so the same source runs under nvcc and under
// the CPU shim. Real kernels use `__nv_fp8_e4m3` and hardware conversion instructions; the
// rounding behaviour implemented below is the same one those instructions perform.
#include "common.cuh"

#if SHIM_BUILD
constexpr int BLOCK = 64;
constexpr int GRID = 8;
#else
constexpr int BLOCK = 256;
constexpr int GRID = 512;
#endif
constexpr int GROUP = 128;             // elements sharing one scale, in the block-scaled variant
constexpr unsigned FULL = 0xffffffffu;

constexpr float E4M3_MAX = 448.0f;
constexpr float E5M2_MAX = 57344.0f;

// Round a float to the nearest value representable in fp8, and return it as a float.
//
// This is quantize-then-dequantize in one step, which is what a "simulated fp8" pass does and
// what lets the error be measured against fp32 on the same hardware. The mechanics: find the
// binade, derive the spacing of representable values inside it from the mantissa width, and
// round to a multiple of that spacing. Values below the smallest normal share the minimum
// exponent, which is what makes subnormals work and what stops small gradients from vanishing
// one binade earlier than they otherwise would.
__host__ __device__ __forceinline__ float fp8_round(float x, int mant_bits, float max_val,
                                                    int min_exp) {
  if (!(x == x)) return x;                       // NaN through
  const float s = (x < 0.0f) ? -1.0f : 1.0f;
  float a = fabsf(x);
  if (a == 0.0f) return 0.0f;
  if (a >= max_val) return s * max_val;          // saturating cast, as fp8 GEMMs perform
  int e;
  frexpf(a, &e);                                 // a = m * 2^e, m in [0.5, 1)
  e -= 1;                                        // a = m' * 2^e, m' in [1, 2)
  if (e < min_exp) e = min_exp;                  // subnormal: spacing stops shrinking
  const float step = ldexpf(1.0f, e - mant_bits);
  float q = rintf(a / step) * step;              // round to nearest, ties to even
  if (q > max_val) q = max_val;
  return s * q;
}
__host__ __device__ __forceinline__ float to_e4m3(float x) {
  return fp8_round(x, 3, E4M3_MAX, -6);
}
__host__ __device__ __forceinline__ float to_e5m2(float x) {
  return fp8_round(x, 2, E5M2_MAX, -14);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(FULL, v, off));
  return v;
}
__device__ __forceinline__ float block_max(float v, float* scratch) {
  const int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  v = warp_max(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  const int nwarps = blockDim.x / warpSize;
  if (warp == 0) {
    v = (lane < nwarps) ? scratch[lane] : 0.0f;
    v = warp_max(v);
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  const float out = scratch[0];
  __syncthreads();
  return out;
}

// ---------------------------------------------------------------------------------------
// Variant 1: one scale for the whole tensor.
//
// Cheapest to store — a single float — and the least accurate, because one outlier anywhere
// sets the scale for every element. Transformer activations reliably contain outlier channels
// two orders of magnitude above the rest, so this is exactly the case where per-tensor
// scaling collapses: the scale is set by the outlier, and everything else quantizes into the
// bottom few representable values.
// ---------------------------------------------------------------------------------------
__global__ void q_per_tensor(const float* __restrict__ x, float* __restrict__ y, float scale,
                             size_t n) {
  const size_t s = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += s)
    y[i] = to_e4m3(x[i] * scale) / scale;
}

// ---------------------------------------------------------------------------------------
// Variant 2: one scale per row (per token). One block per row computes its own amax and uses
// it, so an outlier in one token no longer degrades every other token.
//
// This is the granularity most fp8 inference stacks use for activations, because a row is the
// natural unit — it is what a GEMM's M dimension indexes, so the dequantize folds into the
// epilogue for free.
// ---------------------------------------------------------------------------------------
__global__ void q_per_row(const float* __restrict__ x, float* __restrict__ y,
                          float* __restrict__ scales, int H) {
  SHARED(float, scratch, 32);
  const int row = blockIdx.x;
  const float* xr = x + (size_t)row * H;
  float* yr = y + (size_t)row * H;

  float amax = 0.0f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) amax = fmaxf(amax, fabsf(xr[i]));
  amax = block_max(amax, scratch);
  // Scale so the row's largest magnitude lands exactly at the format's maximum. An empty or
  // all-zero row would divide by zero, so it keeps a scale of 1 and quantizes to zeros.
  const float scale = (amax > 0.0f) ? (E4M3_MAX / amax) : 1.0f;
  if (threadIdx.x == 0) scales[row] = scale;
  for (int i = threadIdx.x; i < H; i += blockDim.x) yr[i] = to_e4m3(xr[i] * scale) / scale;
}

// ---------------------------------------------------------------------------------------
// Variant 3: one scale per 128-element block, the granularity DeepSeek-V3 used to make fp8
// training work end to end.
//
// The finer the group, the closer every element is to its own scale, and the less an outlier
// costs its neighbours. The overhead is one fp32 scale per 128 fp8 values — 4 bytes per 128,
// about 3% — which is the same trade as the group-wise weight quantization in
// 05_dequant_gemv.cu, made for the same reason.
//
// The reason this matters more for training than for inference: an inference quantizer can
// look at the whole calibration set and pick scales offline. A training quantizer sees each
// tensor once, at whatever magnitude this step produced, and has to be right immediately.
// ---------------------------------------------------------------------------------------
__global__ void q_per_block(const float* __restrict__ x, float* __restrict__ y,
                            float* __restrict__ scales, size_t n) {
  SHARED(float, scratch, 32);
  const int ngroups = (int)((n + GROUP - 1) / GROUP);
  for (int g = blockIdx.x; g < ngroups; g += gridDim.x) {
    const size_t base = (size_t)g * GROUP;
    const int len = (int)((base + GROUP <= n) ? GROUP : (n - base));

    float amax = 0.0f;
    for (int i = threadIdx.x; i < len; i += blockDim.x) amax = fmaxf(amax, fabsf(x[base + i]));
    amax = block_max(amax, scratch);
    const float scale = (amax > 0.0f) ? (E4M3_MAX / amax) : 1.0f;
    if (threadIdx.x == 0) scales[g] = scale;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
      y[base + i] = to_e4m3(x[base + i] * scale) / scale;
  }
}

// ---------------------------------------------------------------------------------------
// Variant 4: delayed scaling with a saturation counter — what Transformer Engine actually
// does, and the failure it is built around.
//
// Computing an amax requires a full pass over the tensor before you can quantize it, which
// means two passes instead of one on a kernel that exists to move fewer bytes. So production
// fp8 uses the amax from *previous* steps: keep a short history, take the max, use it now.
// One pass, and the scale is right as long as the tensor's magnitude does not move much
// between steps.
//
// When it does move — a loss spike, a warmup transition, a new sequence length — the stale
// scale is too small and values saturate at ±448. Hence the counter: it is the only way to
// find out, and a run where it climbs is a run that is quietly training on clipped gradients.
// The count is an integer atomic, so it is exact regardless of the order blocks arrive in.
// ---------------------------------------------------------------------------------------
__global__ void q_delayed(const float* __restrict__ x, float* __restrict__ y, float scale,
                          unsigned* __restrict__ n_saturated, unsigned* __restrict__ n_zeroed,
                          size_t n) {
  const size_t s = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += s) {
    const float scaled = x[i] * scale;
    const float q = to_e4m3(scaled);
    y[i] = q / scale;
    if (fabsf(scaled) > E4M3_MAX) atomicAdd(n_saturated, 1u);
    else if (q == 0.0f && x[i] != 0.0f) atomicAdd(n_zeroed, 1u);
  }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  int R = 8, H = 256;
#else
  int R = argc > 1 ? std::atoi(argv[1]) : 4096;
  int H = argc > 2 ? std::atoi(argv[2]) : 4096;
#endif
  (void)argc; (void)argv;
  const size_t n = (size_t)R * H;

  std::vector<float> hx(n), hy(n);
  bench::fill(hx.data(), n, 5);
  // Transformer activations are not uniform. A handful of channels reliably sit two orders of
  // magnitude above the rest, and those outliers are the entire reason scaling granularity is
  // a subject at all — with well-behaved data every scheme below scores the same.
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < H; c += 512) hx[(size_t)r * H + c] *= 120.0f;

  float amax = 0.0f;
  for (size_t i = 0; i < n; ++i) amax = std::max(amax, std::fabs(hx[i]));
  const float tensor_scale = E4M3_MAX / amax;
  // Delayed scaling sees the previous step's amax. Simulate a tensor that grew 3x since then.
  const float stale_scale = E4M3_MAX / (amax / 3.0f);

  const int ngroups = (int)((n + GROUP - 1) / GROUP);
  float *dx, *dy, *dscale_row, *dscale_grp;
  unsigned *dsat, *dzero;
  CUDA_CHECK(cudaMalloc((void**)&dx, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dy, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dscale_row, (size_t)R * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dscale_grp, (size_t)ngroups * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dsat, sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc((void**)&dzero, sizeof(unsigned)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  // Host references: each kernel is checked against the same scheme computed on the CPU, so
  // this measures the kernel and not the quantizer's accuracy. Accuracy is reported
  // separately below, where it belongs.
  std::vector<float> ref_tensor(n), ref_row(n), ref_block(n), ref_delayed(n);
  for (size_t i = 0; i < n; ++i) ref_tensor[i] = to_e4m3(hx[i] * tensor_scale) / tensor_scale;
  for (int r = 0; r < R; ++r) {
    float m = 0.0f;
    for (int i = 0; i < H; ++i) m = std::max(m, std::fabs(hx[(size_t)r * H + i]));
    const float s = m > 0 ? E4M3_MAX / m : 1.0f;
    for (int i = 0; i < H; ++i)
      ref_row[(size_t)r * H + i] = to_e4m3(hx[(size_t)r * H + i] * s) / s;
  }
  for (int g = 0; g < ngroups; ++g) {
    const size_t base = (size_t)g * GROUP;
    const int len = (int)std::min<size_t>(GROUP, n - base);
    float m = 0.0f;
    for (int i = 0; i < len; ++i) m = std::max(m, std::fabs(hx[base + i]));
    const float s = m > 0 ? E4M3_MAX / m : 1.0f;
    for (int i = 0; i < len; ++i) ref_block[base + i] = to_e4m3(hx[base + i] * s) / s;
  }
  for (size_t i = 0; i < n; ++i) ref_delayed[i] = to_e4m3(hx[i] * stale_scale) / stale_scale;

  std::vector<bench::Row> rows;
  const double traffic = 2.0 * n * sizeof(float);

  auto run = [&](const char* name, const float* want, auto&& launch) {
    CUDA_CHECK(cudaMemset(dy, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(dsat, 0, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(dzero, 0, sizeof(unsigned)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev, 20, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    r.err = bench::max_rel_err(hy.data(), want, n);
    r.checksum = bench::checksum_of(hy);
    r.bytes = traffic;
    r.flops = 6.0 * n;
    char buf[52];
    std::snprintf(buf, sizeof buf, "L2 vs fp32: %.3e", bench::rel_l2(hy.data(), hx.data(), n));
    r.note = buf;
    rows.push_back(r);
  };

  std::printf("problem   : %d x %d activations, %d outlier channels at ~120x\n", R, H,
              (H + 511) / 512);
  std::printf("formats   : e4m3 max %.0f (fwd), e5m2 max %.0f (grads)\n", E4M3_MAX, E5M2_MAX);
  bench::header(dev);

  run("1 per-tensor scale", ref_tensor.data(), [&] {
    KERNEL_LAUNCH(q_per_tensor, dim3(GRID), dim3(BLOCK), 0, dx, dy, tensor_scale, n);
  });
  run("2 per-row (per-token) scale", ref_row.data(), [&] {
    KERNEL_LAUNCH(q_per_row, dim3(R), dim3(BLOCK), 0, dx, dy, dscale_row, H);
  });
  run("3 per-128-block scale", ref_block.data(), [&] {
    KERNEL_LAUNCH(q_per_block, dim3(GRID), dim3(BLOCK), 0, dx, dy, dscale_grp, n);
  });
  run("4 delayed (stale) scale", ref_delayed.data(), [&] {
    KERNEL_LAUNCH(q_delayed, dim3(GRID), dim3(BLOCK), 0, dx, dy, stale_scale, dsat, dzero, n);
  });

  // The kernels above are checked against their own schemes, so the tolerance is about float
  // reproducibility, not about quantization error.
  const double tol = 1e-6;
  bench::rows_out(rows, dev, tol);

  unsigned sat = 0, zeroed = 0;
  CUDA_CHECK(cudaMemcpy(&sat, dsat, sizeof(unsigned), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&zeroed, dzero, sizeof(unsigned), cudaMemcpyDeviceToHost));

  std::printf("\nAccuracy — the `L2 vs fp32` column above, restated:\n"
              "  finer scaling groups cost 4 bytes each and buy back the precision that one\n"
              "  outlier channel would otherwise take from every element sharing its scale.\n");
  std::printf("\nDelayed scaling, with the tensor 3x larger than the amax history expected:\n"
              "  %u of %zu elements saturated at +-448 (%.2f%%)\n"
              "  %u flushed to zero\n"
              "  Neither raises an error. A run where the first number climbs is training on\n"
              "  clipped gradients, and the only symptom is a slightly worse model.\n",
              sat, n, 100.0 * sat / (double)n, zeroed);

  // ---- the other failure: gradient underflow, and why loss scaling exists ---------------
  std::printf("\nWhy gradients get e5m2 and a loss scale — smallest representable magnitudes:\n");
  struct Fmt { const char* name; float smallest; };
  const Fmt fmts[] = {{"fp32", 1.4e-45f}, {"bf16", 9.2e-41f}, {"fp16", 6.0e-8f},
                      {"e5m2", ldexpf(1.0f, -16)}, {"e4m3", ldexpf(1.0f, -9)}};
  for (const Fmt& f : fmts) std::printf("  %-6s %.2e\n", f.name, f.smallest);
  const float tiny = 3e-8f;
  std::printf("\n  A gradient of %.0e:\n", tiny);
  std::printf("    e4m3, unscaled            -> %.3e  %s\n", to_e4m3(tiny),
              to_e4m3(tiny) == 0.0f ? "GONE" : "survives");
  std::printf("    e5m2, unscaled            -> %.3e  %s\n", to_e5m2(tiny),
              to_e5m2(tiny) == 0.0f ? "GONE" : "survives");
  std::printf("    e5m2, loss scale 2^15     -> %.3e  survives, and the optimizer divides\n"
              "                                   it back out before the update\n",
              to_e5m2(tiny * 32768.0f) / 32768.0f);
  std::printf("\n  bf16 keeps fp32's 8 exponent bits, which is why bf16 training needed no\n"
              "  loss scaler and fp16 training did. fp8 gives the range back up, so the\n"
              "  scaler returns — now per-tensor, per-row or per-block rather than global.\n");

  for (void* p : {(void*)dx, (void*)dy, (void*)dscale_row, (void*)dscale_grp, (void*)dsat,
                  (void*)dzero})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
