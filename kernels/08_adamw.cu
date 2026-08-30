// 08_adamw.cu — the optimizer step, and the memory budget that decides what you can train.
//
//     nvcc -O3 -arch=native 08_adamw.cu -o build/08 && build/08
//     make check
//
// AdamW, per parameter:
//
//     m = β₁·m + (1−β₁)·g
//     v = β₂·v + (1−β₂)·g²
//     p = p − lr·( m̂ / (√v̂ + ε) + λ·p )        with m̂ = m/(1−β₁ᵗ), v̂ = v/(1−β₂ᵗ)
//
// About 11 FLOPs against 28 bytes of traffic: **0.4 FLOP/byte**, an order of magnitude below
// even the memory-bound kernels in this directory. The optimizer step is pure bandwidth, and
// the only thing that makes it faster is moving fewer bytes.
//
// Which is exactly what a framework that composes primitives does not do. `m.mul_(b1).add_(g,
// alpha=1-b1)` and its four siblings are five separate kernels, each making a full round trip
// to HBM for an operation with no arithmetic to hide it. Variant 1 is that; variant 2 fuses
// it; the difference is a factor of ~1.7 in traffic on an operation that runs once per step
// over every parameter in the model.
//
// The other half of this file is the memory table at the end, which is the thing people
// actually need. A 7B model in fp32 AdamW needs 112 GB of optimizer state and gradients before
// a single activation exists — which is why nobody trains that way, and why ZeRO, 8-bit
// optimizer states, and bf16 master weights are not micro-optimizations but the difference
// between fitting and not.
//
// Prerequisite: 04_rmsnorm.cu, for the fusion argument this file applies at scale.
#include "common.cuh"

#if SHIM_BUILD
constexpr int BLOCK = 64;
constexpr int GRID = 8;
#else
constexpr int BLOCK = 256;
constexpr int GRID = 1024;
#endif

// bf16 as a plain 16-bit container, so the same source works under nvcc and under the CPU
// shim. Production kernels use __nv_bfloat16 and get hardware conversion instructions; the
// arithmetic and the rounding behaviour are what matter here.
//
// bf16 is fp32 with 16 mantissa bits removed: same 8 exponent bits, so the same *range* and
// no need for loss scaling, but only ~3 decimal digits of precision. That is why it replaced
// fp16 for training, and why the master weights below still have to be fp32.
typedef unsigned short bf16;

__host__ __device__ __forceinline__ unsigned f2u(float f) {
  union { float f; unsigned u; } c;
  c.f = f;
  return c.u;
}
__host__ __device__ __forceinline__ float u2f(unsigned u) {
  union { float f; unsigned u; } c;
  c.u = u;
  return c.f;
}
// Round-to-nearest-even, not truncation. Truncating introduces a systematic downward bias
// that a million optimizer steps will happily integrate into your weights.
__host__ __device__ __forceinline__ bf16 to_bf16(float f) {
  unsigned u = f2u(f);
  return (bf16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__host__ __device__ __forceinline__ float from_bf16(bf16 h) { return u2f((unsigned)h << 16); }

// ---------------------------------------------------------------------------------------
// Variant 1: five kernels, as a framework composing primitives would produce.
//
// Each one reads its operands from HBM and writes its result back, for two or three FLOPs.
// Nothing here is badly written; every kernel is optimally coalesced. The cost is entirely in
// there being five of them.
// ---------------------------------------------------------------------------------------
__global__ void step_m(float* __restrict__ m, const float* __restrict__ g, float b1, size_t n) {
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += s)
    m[i] = b1 * m[i] + (1.0f - b1) * g[i];
}
__global__ void step_v(float* __restrict__ v, const float* __restrict__ g, float b2, size_t n) {
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += s)
    v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
}
__global__ void step_decay(float* __restrict__ p, float lr, float wd, size_t n) {
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += s)
    p[i] -= lr * wd * p[i];
}
__global__ void step_update(float* __restrict__ p, const float* __restrict__ m,
                            const float* __restrict__ v, float lr, float c1, float c2,
                            float eps, size_t n) {
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += s)
    p[i] -= lr * (m[i] * c1) / (sqrtf(v[i] * c2) + eps);
}

// ---------------------------------------------------------------------------------------
// Variant 2: one fused kernel. Identical arithmetic, one pass.
//
// p, g, m and v are each read once and p, m, v written once: 28 bytes per parameter against
// the unfused version's 48. On an 8B model that is 224 GB of traffic per step instead of 384.
// ---------------------------------------------------------------------------------------
__global__ void adamw_fused(float* __restrict__ p, const float* __restrict__ g,
                            float* __restrict__ m, float* __restrict__ v, float lr, float b1,
                            float b2, float eps, float wd, float c1, float c2, size_t n) {
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += s) {
    const float gi = g[i];
    const float mi = b1 * m[i] + (1.0f - b1) * gi;
    const float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    // Decoupled weight decay — the W in AdamW. Applied to p directly rather than folded into
    // the gradient, so it does not get scaled by the adaptive denominator. Folding it in
    // instead gives you Adam+L2, which is a different algorithm with a different optimum.
    p[i] = p[i] - lr * ((mi * c1) / (sqrtf(vi * c2) + eps) + wd * p[i]);
  }
}

// ---------------------------------------------------------------------------------------
// Variant 3: fused and vectorized. Same bytes, a quarter of the instructions, and four
// independent loads in flight per thread — the argument from 01_copy.cu, applied to the one
// kernel that touches every parameter in the model on every step.
// ---------------------------------------------------------------------------------------
__global__ void adamw_fused_vec4(float4* __restrict__ p, const float4* __restrict__ g,
                                 float4* __restrict__ m, float4* __restrict__ v, float lr,
                                 float b1, float b2, float eps, float wd, float c1, float c2,
                                 size_t n4) {
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += s) {
    float4 pv = p[i], gv = g[i], mv = m[i], vv = v[i];
    float pa[4] = {pv.x, pv.y, pv.z, pv.w};
    float ga[4] = {gv.x, gv.y, gv.z, gv.w};
    float ma[4] = {mv.x, mv.y, mv.z, mv.w};
    float va[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      ma[k] = b1 * ma[k] + (1.0f - b1) * ga[k];
      va[k] = b2 * va[k] + (1.0f - b2) * ga[k] * ga[k];
      pa[k] = pa[k] - lr * ((ma[k] * c1) / (sqrtf(va[k] * c2) + eps) + wd * pa[k]);
    }
    m[i] = make_float4(ma[0], ma[1], ma[2], ma[3]);
    v[i] = make_float4(va[0], va[1], va[2], va[3]);
    p[i] = make_float4(pa[0], pa[1], pa[2], pa[3]);
  }
}

// ---------------------------------------------------------------------------------------
// Variant 4: bf16 gradients and a bf16 copy of the weights, with an fp32 master copy.
//
// This is how mixed-precision training actually works, and the reason the master copy exists
// is worth stating precisely. A typical update is lr·(m̂/√v̂) ≈ 1e-3 · 1 = 1e-3 against a
// weight of order 1. In bf16, with ~3 decimal digits, 1.0 + 0.001 rounds straight back to 1.0
// — **the update is silently discarded**, and the model stops learning while every metric
// looks fine. The fp32 master accumulates the updates; the bf16 copy is what the forward pass
// reads, because that is where the bandwidth and the tensor cores are.
//
// The accounting is the surprise: this does not save optimizer memory. p_bf16(2) + g_bf16(2)
// + master(4) + m(4) + v(4) = 16 B/param, the same as plain fp32 AdamW. What it buys is
// halved *activation* memory, halved gradient *communication*, and tensor cores in the
// forward and backward passes. The table at the end of this file spells that out.
// ---------------------------------------------------------------------------------------
__global__ void adamw_master(bf16* __restrict__ p_bf16, float* __restrict__ p_master,
                             const bf16* __restrict__ g_bf16, float* __restrict__ m,
                             float* __restrict__ v, float lr, float b1, float b2, float eps,
                             float wd, float c1, float c2, size_t n) {
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += s) {
    const float gi = from_bf16(g_bf16[i]);
    const float mi = b1 * m[i] + (1.0f - b1) * gi;
    const float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    // The update lands in fp32, every time...
    const float pi = p_master[i] - lr * ((mi * c1) / (sqrtf(vi * c2) + eps) + wd * p_master[i]);
    p_master[i] = pi;
    // ...and only the rounded view of it goes to the forward pass.
    p_bf16[i] = to_bf16(pi);
  }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  size_t n = 1 << 12;
#else
  size_t n = argc > 1 ? (size_t)std::atoll(argv[1]) : (1u << 24);   // 16M parameters
#endif
  (void)argc; (void)argv;
  n &= ~(size_t)3;

  const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.1f;
  const int t = 100;                                    // step number, for bias correction
  const float c1 = 1.0f / (1.0f - std::pow(b1, (float)t));
  const float c2 = 1.0f / (1.0f - std::pow(b2, (float)t));

  std::vector<float> hp(n), hg(n), hm(n), hv(n), out(n);
  bench::fill(hp.data(), n, 1);
  bench::fill(hg.data(), n, 2);
  bench::fill(hm.data(), n, 3);
  bench::fill(hv.data(), n, 4);
  for (size_t i = 0; i < n; ++i) {
    hg[i] *= 0.01f;                        // gradients are small
    hv[i] = std::fabs(hv[i]) * 1e-4f;      // v is a running mean of squares: non-negative
    hm[i] *= 0.01f;
  }

  // Reference in double.
  std::vector<float> want(n), want_master(n);
  for (size_t i = 0; i < n; ++i) {
    double g = hg[i];
    double m = b1 * (double)hm[i] + (1.0 - b1) * g;
    double v = b2 * (double)hv[i] + (1.0 - b2) * g * g;
    double p = (double)hp[i] - lr * ((m * c1) / (std::sqrt(v * c2) + eps) + wd * (double)hp[i]);
    want[i] = (float)p;
    want_master[i] = (float)p;
  }
  // Variant 4 sees bf16 gradients, so its answer legitimately differs. Give it its own
  // reference rather than widening the tolerance until everything passes.
  std::vector<float> want_bf(n);
  for (size_t i = 0; i < n; ++i) {
    double g = from_bf16(to_bf16(hg[i]));
    double m = b1 * (double)hm[i] + (1.0 - b1) * g;
    double v = b2 * (double)hv[i] + (1.0 - b2) * g * g;
    want_bf[i] = (float)((double)hp[i] -
                         lr * ((m * c1) / (std::sqrt(v * c2) + eps) + wd * (double)hp[i]));
  }

  float *dp, *dg, *dm, *dv, *dmaster;
  bf16 *dpbf, *dgbf;
  CUDA_CHECK(cudaMalloc((void**)&dp, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dg, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dm, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dv, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dmaster, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dpbf, n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc((void**)&dgbf, n * sizeof(bf16)));
  CUDA_CHECK(cudaMemcpy(dg, hg.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  {
    std::vector<bf16> gb(n);
    for (size_t i = 0; i < n; ++i) gb[i] = to_bf16(hg[i]);
    CUDA_CHECK(cudaMemcpy(dgbf, gb.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
  }

  const double B = sizeof(float);
  std::vector<bench::Row> rows;

  auto reset = [&] {
    CUDA_CHECK(cudaMemcpy(dp, hp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmaster, hp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dm, hm.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, hv.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  };

  auto run = [&](const char* name, double traffic, const float* want_p, const float* src,
                 auto&& launch) {
    reset();
    bench::Row r;
    r.name = name;
    // Warmup would re-apply the step to already-updated weights, so each rep resets first;
    // that reset is outside the timed region because it is not part of the optimizer.
    r.st = bench::time_kernel([&] { reset(); launch(); }, dev, 20, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(), src, n * sizeof(float), cudaMemcpyDeviceToHost));
    r.err = bench::max_rel_err(out.data(), want_p, n);
    r.checksum = bench::checksum_of(out);
    r.bytes = traffic;
    r.flops = 11.0 * n;
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.0f B/param", traffic / n);
    r.note = buf;
    rows.push_back(r);
  };

  std::printf("problem   : %zu parameters, AdamW step %d\n", n, t);
  bench::header(dev);

  // read m,g write m (12) + read v,g write v (12) + read p write p (8) + read p,m,v write p (16)
  run("1 five kernels (unfused)", 48.0 * n, want.data(), dp, [&] {
    KERNEL_LAUNCH(step_m, dim3(GRID), dim3(BLOCK), 0, dm, dg, b1, n);
    KERNEL_LAUNCH(step_v, dim3(GRID), dim3(BLOCK), 0, dv, dg, b2, n);
    KERNEL_LAUNCH(step_decay, dim3(GRID), dim3(BLOCK), 0, dp, lr, wd, n);
    KERNEL_LAUNCH(step_update, dim3(GRID), dim3(BLOCK), 0, dp, dm, dv, lr, c1, c2, eps, n);
  });
  // read p,g,m,v (16) + write p,m,v (12)
  run("2 fused", 28.0 * n, want.data(), dp, [&] {
    KERNEL_LAUNCH(adamw_fused, dim3(GRID), dim3(BLOCK), 0, dp, dg, dm, dv, lr, b1, b2, eps, wd,
                  c1, c2, n);
  });
  run("3 fused + float4", 28.0 * n, want.data(), dp, [&] {
    KERNEL_LAUNCH(adamw_fused_vec4, dim3(GRID), dim3(BLOCK), 0, (float4*)dp, (const float4*)dg,
                  (float4*)dm, (float4*)dv, lr, b1, b2, eps, wd, c1, c2, n / 4);
  });
  // read g_bf16(2), master(4), m(4), v(4) + write master(4), m(4), v(4), p_bf16(2)
  run("4 bf16 + fp32 master", 28.0 * n, want_bf.data(), dmaster, [&] {
    KERNEL_LAUNCH(adamw_master, dim3(GRID), dim3(BLOCK), 0, dpbf, dmaster, dgbf, dm, dv, lr,
                  b1, b2, eps, wd, c1, c2, n);
  });

  const double tol = 1e-5;
  bench::rows_out(rows, dev, tol);

  // ---- what a training run actually costs -------------------------------------------
  // The reason this file exists in a serving repo: the same roofline reasoning decides what
  // you can fine-tune, and the answer is almost never "compute".
  std::printf("\nWhere the memory goes, per billion parameters:\n\n");
  std::printf("  %-34s %8s %8s %8s %8s %8s\n", "recipe", "weights", "grads", "m", "v", "total");
  struct Recipe { const char* name; double w, g, m, v; };
  const Recipe recipes[] = {
      {"fp32 weights, fp32 AdamW",         4, 4, 4, 4},
      {"bf16 weights + fp32 master",       2 + 4, 2, 4, 4},
      {"bf16 + 8-bit optimizer states",    2 + 4, 2, 1, 1},
      {"bf16 + master, ZeRO-2 (8 GPUs)",   2 + 4.0 / 8, 2.0 / 8, 4.0 / 8, 4.0 / 8},
      {"bf16 + master, ZeRO-3 (8 GPUs)",   (2 + 4.0) / 8, 2.0 / 8, 4.0 / 8, 4.0 / 8},
      {"LoRA r=16 (0.1% trainable)",       2, 0.002, 0.004, 0.004},
  };
  for (const Recipe& r : recipes) {
    double tot = r.w + r.g + r.m + r.v;
    std::printf("  %-34s %7.2f %8.2f %8.2f %8.2f %7.2f GB\n", r.name, r.w, r.g, r.m, r.v, tot);
  }
  std::printf("\n  A 7B model, fp32 AdamW: %.0f GB of state before a single activation exists,\n"
              "  on an 80 GB card. That is the whole reason ZeRO, 8-bit states and LoRA exist —\n"
              "  and none of them is about making the arithmetic faster.\n", 7 * 16.0);
  std::printf("\n  The optimizer step itself is %.2f FLOP/byte: memory-bound by two orders of\n"
              "  magnitude, so fusing the five kernels into one is worth %.1fx and tuning the\n"
              "  arithmetic is worth nothing.\n", 11.0 / 28.0, 48.0 / 28.0);

  for (void* p : {(void*)dp, (void*)dg, (void*)dm, (void*)dv, (void*)dmaster, (void*)dpbf,
                  (void*)dgbf})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
