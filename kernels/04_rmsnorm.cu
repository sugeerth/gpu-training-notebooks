// 04_rmsnorm.cu — the first kernel here that a real LLM actually runs, and the clearest
// demonstration of why fusion is the dominant optimization for everything except the GEMMs.
//
//     nvcc -O3 -arch=native 04_rmsnorm.cu -o build/04_rmsnorm && build/04_rmsnorm
//     make check
//
//     y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
//
// Two FLOPs of useful arithmetic per element, against 8 bytes of traffic. Arithmetic intensity
// 0.25 FLOP/byte, against a ridge point of ~100. RMSNorm will never be compute-bound on any
// GPU that will ever be built. There is nothing to make faster *inside* this kernel.
//
// So the only lever is the byte count, and that is not a property of the kernel — it is a
// property of how many kernels there are. A transformer block does
//
//     h  = x + attn_out          <- elementwise add:  reads 2, writes 1
//     hn = rmsnorm(h) * w        <- reads 1 (twice, if two-pass), writes 1
//
// as separate launches, and every intermediate makes a full round trip to HBM even though the
// next kernel reads it back immediately. Fusing them does not make the arithmetic faster. It
// deletes the round trips. That is the entire optimization, and on a decode step — where the
// whole model is memory-bound — it is worth more than anything done to the math.
//
// The variants below move 1.5x, 1.0x, 1.0x and 2.0x the minimum bytes — and the last one is
// doing two operations' work for its 2.0x. Watch the GB/s column: every variant runs at close
// to the same bandwidth. They differ in *how many bytes they need*, not in how fast they move
// them. That distinction is the whole subject.
#include "common.cuh"

#if SHIM_BUILD
constexpr int BLOCK = 64;
constexpr int MAX_PER_THREAD = 4;
#else
constexpr int BLOCK = 256;
constexpr int MAX_PER_THREAD = 16;   // supports hidden sizes up to 256 * 16 = 4096
#endif
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down_sync(FULL, v, off);
  return v;
}

// Block-wide sum, broadcast to every thread. Two shuffle reductions and one barrier — the
// pattern from 02_reduce.cu, plus a broadcast, because here every thread needs the result,
// not just lane 0.
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
  int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  v = warp_reduce_sum(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  int nwarps = blockDim.x / warpSize;
  v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : 0.0f;
  if (warp == 0) v = warp_reduce_sum(v);
  if (threadIdx.x == 0) scratch[0] = v;
  __syncthreads();
  return scratch[0];
}

// ---------------------------------------------------------------------------------------
// Variant 1: two passes, two kernels. The shape you get for free from a framework that
// composes primitives — `x.pow(2).mean(-1)` then `x * rsqrt(...) * w`.
//
// x is read from HBM twice: once to compute the statistic, once to apply it. There is no way
// to avoid that with two kernels, because the second launch cannot see the first's registers.
// ---------------------------------------------------------------------------------------
__global__ void rms_pass1_sumsq(const float* __restrict__ x, float* __restrict__ ss, int H) {
  SHARED(float, scratch, 32);
  const float* row = x + (size_t)blockIdx.x * H;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) acc += row[i] * row[i];
  acc = block_reduce_sum(acc, scratch);
  if (threadIdx.x == 0) ss[blockIdx.x] = acc;
}

__global__ void rms_pass2_apply(const float* __restrict__ x, const float* __restrict__ w,
                                const float* __restrict__ ss, float* __restrict__ y, int H,
                                float eps) {
  const size_t r = blockIdx.x;
  const float scale = rsqrtf(ss[r] / H + eps);
  const float* row = x + r * H;
  float* out = y + r * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) out[i] = row[i] * scale * w[i];
}

// ---------------------------------------------------------------------------------------
// Variant 2: one pass. One block owns one row, and holds the row *in registers* across the
// reduction, so the second read never touches memory at all.
//
// This is the trick that makes fusion possible for normalization: the row is small enough
// (4096 floats = 16 per thread at 256 threads) to live in the register file while the block
// computes a statistic over it. It is also the constraint that decides the block size —
// registers per thread is a hard budget, and spilling one array to local memory puts the
// traffic straight back into HBM where it started, silently.
// ---------------------------------------------------------------------------------------
__global__ void rms_one_pass(const float* __restrict__ x, const float* __restrict__ w,
                             float* __restrict__ y, int H, float eps) {
  SHARED(float, scratch, 32);
  const float* row = x + (size_t)blockIdx.x * H;
  float* out = y + (size_t)blockIdx.x * H;

  float v[MAX_PER_THREAD];
  float acc = 0.0f;
  int n = 0;
  for (int i = threadIdx.x; i < H; i += blockDim.x, ++n) {
    v[n] = row[i];             // read once, keep
    acc += v[n] * v[n];
  }
  const float scale = rsqrtf(block_reduce_sum(acc, scratch) / H + eps);
  n = 0;
  for (int i = threadIdx.x; i < H; i += blockDim.x, ++n) out[i] = v[n] * scale * w[i];
}

// ---------------------------------------------------------------------------------------
// Variant 3: the same, with 128-bit accesses. Identical byte count to variant 2 — this is
// purely the instruction-count and memory-level-parallelism argument from 01_copy.cu, applied
// to a kernel where it matters more, because there is no arithmetic to hide latency behind.
// ---------------------------------------------------------------------------------------
__global__ void rms_one_pass_vec4(const float4* __restrict__ x, const float4* __restrict__ w,
                                  float4* __restrict__ y, int H4, float eps) {
  SHARED(float, scratch, 32);
  const float4* row = x + (size_t)blockIdx.x * H4;
  float4* out = y + (size_t)blockIdx.x * H4;

  float4 v[MAX_PER_THREAD / 4 + 1];
  float acc = 0.0f;
  int n = 0;
  for (int i = threadIdx.x; i < H4; i += blockDim.x, ++n) {
    float4 t = row[i];
    v[n] = t;
    acc += t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
  }
  const float scale = rsqrtf(block_reduce_sum(acc, scratch) / (4 * H4) + eps);
  n = 0;
  for (int i = threadIdx.x; i < H4; i += blockDim.x, ++n) {
    float4 g = w[i], t = v[n];
    out[i] = make_float4(t.x * scale * g.x, t.y * scale * g.y, t.z * scale * g.z,
                         t.w * scale * g.w);
  }
}

// ---------------------------------------------------------------------------------------
// Variant 4: fused residual-add + RMSNorm. This is the kernel a serving engine actually calls
// (vLLM names it `fused_add_rms_norm`), and it is the point of the whole file.
//
//     residual = residual + x        <- kept, it is the next block's skip connection
//     y        = rmsnorm(residual) * w
//
// Counted in round trips over the hidden state, and giving the unfused version the benefit of
// variant 2's one-pass norm rather than variant 1's:
//
//     unfused:  add kernel  = read x, read residual, write residual   -> 3
//               norm kernel = read residual, write y                  -> 2   total 5
//     fused:    read x, read residual, write residual, write y        ->     total 4
//
// A 20% cut, for a kernel whose arithmetic did not change at all. That is a modest-sounding
// number until you note that it applies to two normalizations in every one of 80 layers, on
// every decode step, and that the same reasoning is what makes fused QKV, fused SwiGLU, and
// fused epilogues each worth their own kernel.
//
// Note the accumulation in fp32 even though a production version would carry bf16 data. The
// reduction is where normalization loses precision, and it is the one place a low-precision
// kernel must not economize: a bf16 sum of 8192 squares has about three decimal digits left.
// ---------------------------------------------------------------------------------------
__global__ void fused_add_rms_norm(float4* __restrict__ residual, const float4* __restrict__ x,
                                   const float4* __restrict__ w, float4* __restrict__ y, int H4,
                                   float eps) {
  SHARED(float, scratch, 32);
  float4* res = residual + (size_t)blockIdx.x * H4;
  const float4* xr = x + (size_t)blockIdx.x * H4;
  float4* out = y + (size_t)blockIdx.x * H4;

  float4 v[MAX_PER_THREAD / 4 + 1];
  float acc = 0.0f;
  int n = 0;
  for (int i = threadIdx.x; i < H4; i += blockDim.x, ++n) {
    float4 a = res[i], b = xr[i];
    float4 s = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    res[i] = s;                                  // the skip connection, written once
    v[n] = s;                                    // and kept, for the normalize below
    acc += s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;
  }
  const float scale = rsqrtf(block_reduce_sum(acc, scratch) / (4 * H4) + eps);
  n = 0;
  for (int i = threadIdx.x; i < H4; i += blockDim.x, ++n) {
    float4 g = w[i], t = v[n];
    out[i] = make_float4(t.x * scale * g.x, t.y * scale * g.y, t.z * scale * g.z,
                         t.w * scale * g.w);
  }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  int R = 4, H = 256;
#else
  int R = argc > 1 ? std::atoi(argv[1]) : 8192;    // "batch x sequence" rows
  int H = argc > 2 ? std::atoi(argv[2]) : 4096;    // hidden size
#endif
  (void)argc; (void)argv;
  if (H % (4 * BLOCK) != 0) H = ((H / (4 * BLOCK)) + 1) * 4 * BLOCK;
  if (H / BLOCK > MAX_PER_THREAD) {
    std::printf("hidden size %d needs %d registers per thread, budget is %d\n", H, H / BLOCK,
                MAX_PER_THREAD);
    return 1;
  }
  const float eps = 1e-6f;
  const size_t NEL = (size_t)R * H;

  std::vector<float> hx(NEL), hw(H), hres(NEL), hy(NEL);
  bench::fill(hx.data(), NEL, 1);
  bench::fill(hw.data(), H, 2);
  bench::fill(hres.data(), NEL, 3);
  for (int i = 0; i < H; ++i) hw[i] = 1.0f + 0.1f * hw[i];   // weights near 1, as trained

  // References in double.
  std::vector<float> want_plain(NEL), want_fused(NEL), want_res(NEL);
  for (int r = 0; r < R; ++r) {
    double ss = 0.0, ssf = 0.0;
    for (int i = 0; i < H; ++i) {
      double a = hx[(size_t)r * H + i];
      ss += a * a;
      double s = a + (double)hres[(size_t)r * H + i];
      ssf += s * s;
      want_res[(size_t)r * H + i] = (float)s;
    }
    double sc = 1.0 / std::sqrt(ss / H + eps), scf = 1.0 / std::sqrt(ssf / H + eps);
    for (int i = 0; i < H; ++i) {
      want_plain[(size_t)r * H + i] = (float)(hx[(size_t)r * H + i] * sc * hw[i]);
      want_fused[(size_t)r * H + i] = (float)(want_res[(size_t)r * H + i] * scf * hw[i]);
    }
  }

  float *dx, *dw, *dy, *dss, *dres;
  CUDA_CHECK(cudaMalloc((void**)&dx, NEL * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dw, (size_t)H * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dy, NEL * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dres, NEL * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dss, (size_t)R * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), NEL * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, hw.data(), (size_t)H * sizeof(float), cudaMemcpyHostToDevice));

  const double B = sizeof(float);
  const double min_traffic = 2.0 * NEL * B + H * B;   // read x once, write y once
  std::vector<bench::Row> rows;

  auto run = [&](const char* name, double traffic, const float* want, auto&& launch) {
    CUDA_CHECK(cudaMemset(dy, 0, NEL * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dres, hres.data(), NEL * sizeof(float), cudaMemcpyHostToDevice));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev, 30, 10);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, NEL * sizeof(float), cudaMemcpyDeviceToHost));
    r.err = bench::max_rel_err(hy.data(), want, NEL);
    r.bytes = traffic;
    r.flops = 4.0 * NEL;   // square, add, multiply, multiply
    char buf[48];
    std::snprintf(buf, sizeof buf, "%.2fx minimum traffic", traffic / min_traffic);
    r.note = buf;
    rows.push_back(r);
  };

  std::printf("problem   : %d rows x %d hidden, %d threads/block, %d floats/thread in registers\n",
              R, H, BLOCK, H / BLOCK);
  bench::header(dev);

  run("1 two-pass, two kernels", 3.0 * NEL * B + H * B + 2.0 * R * B, want_plain.data(), [&] {
    KERNEL_LAUNCH(rms_pass1_sumsq, dim3(R), dim3(BLOCK), 0, dx, dss, H);
    KERNEL_LAUNCH(rms_pass2_apply, dim3(R), dim3(BLOCK), 0, dx, dw, dss, dy, H, eps);
  });
  run("2 one-pass, registers", min_traffic, want_plain.data(), [&] {
    KERNEL_LAUNCH(rms_one_pass, dim3(R), dim3(BLOCK), 0, dx, dw, dy, H, eps);
  });
  run("3 one-pass, float4", min_traffic, want_plain.data(), [&] {
    KERNEL_LAUNCH(rms_one_pass_vec4, dim3(R), dim3(BLOCK), 0, (const float4*)dx,
                  (const float4*)dw, (float4*)dy, H / 4, eps);
  });
  // Does strictly more work than the others: also consumes a residual and rewrites it.
  run("4 fused add+rmsnorm (float4)", 4.0 * NEL * B + H * B, want_fused.data(), [&] {
    KERNEL_LAUNCH(fused_add_rms_norm, dim3(R), dim3(BLOCK), 0, (float4*)dres,
                  (const float4*)dx, (const float4*)dw, (float4*)dy, H / 4, eps);
  });

  // Independently check that variant 4 also left the correct residual behind — a fused kernel
  // with two outputs is exactly where a test that only looks at one of them goes green while
  // the model quietly diverges.
  std::vector<float> got_res(NEL);
  CUDA_CHECK(cudaMemcpy(got_res.data(), dres, NEL * sizeof(float), cudaMemcpyDeviceToHost));
  double res_err = bench::max_rel_err(got_res.data(), want_res.data(), NEL);

  const double tol = 1e-5;
  bench::rows_out(rows, dev, tol);
  std::printf("\nvariant 4's residual output: max rel err %.2e  %s\n", res_err,
              res_err <= tol ? "ok" : "WRONG");

  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dw));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dres));
  CUDA_CHECK(cudaFree(dss));
  return bench::verdict(rows, tol) || !(res_err <= tol);
}
