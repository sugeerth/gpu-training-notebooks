// 05_dequant_gemv.cu — the kernel that decides how fast an LLM generates tokens, and the
// clearest statement of why quantization is a *serving* technique rather than a memory one.
//
//     nvcc -O3 -arch=native 05_dequant_gemv.cu -o build/05_dequant_gemv && build/05_dequant_gemv
//     make check
//
// During decode there is one new token per sequence, so every weight matrix is applied to a
// vector, not a matrix. y = W x with W of shape [N, K]:
//
//     arithmetic : 2 N K FLOPs
//     traffic    : N K x (bytes per weight)     — x and y are negligible, W is everything
//     intensity  : 2 / bytes_per_weight  FLOP/byte
//
//        fp32 -> 0.5      fp16 -> 1.0      int8 -> 2.0      int4 -> 4.0
//
// Every one of those is one to two orders of magnitude below the ridge point of any modern
// GPU (~100 FLOP/byte). A GEMV is memory-bound no matter what you do to it, which has a blunt
// consequence:
//
//     decode time is proportional to the number of BYTES in the weights.
//     Nothing else about the arithmetic matters.
//
// So int4 weights are ~4x faster to decode than fp16 weights — not because int4 arithmetic is
// fast (the kernel below converts every weight back to fp32 and does fp32 math), but because
// there are a quarter as many bytes to fetch. The dequantization is free: it happens in
// registers, on data that has already been paid for, on a machine with nothing else to do.
//
// This is also why the same quantization does almost nothing for prefill. Prefill applies W to
// a matrix of hundreds of tokens, the intensity is hundreds of FLOP/byte, the kernel is
// compute-bound, and shrinking the weights buys only capacity. One technique, two completely
// different reasons, and they do not both apply at once.
//
// Prerequisite: 02_reduce.cu (the shuffle reduction reused here for the dot product).
#include "common.cuh"
#include <cstdint>

#if SHIM_BUILD
constexpr int BLOCK = 64;
#else
constexpr int BLOCK = 256;
#endif
constexpr int GROUP = 128;   // weights per quantization scale, along K
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_down_sync(FULL, v, off);
  return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
  int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  v = warp_reduce_sum(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  int nwarps = blockDim.x / warpSize;
  v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : 0.0f;
  if (warp == 0) v = warp_reduce_sum(v);
  return v;   // valid on thread 0
}

// ---------------------------------------------------------------------------------------
// Variant 1: fp32 baseline. One block per output row; the block strides across K, each thread
// accumulating a partial dot product, then one shuffle reduction.
//
// Note what x's access pattern is *not* doing here: every block reads all of x. That looks
// wasteful and is not, because x is K floats — 16 KB at K=4096 — and lands in L2 after the
// first block touches it. The weight row is the only thing coming from HBM, every time, for
// every row. In the accounting below x is not even counted.
// ---------------------------------------------------------------------------------------
__global__ void gemv_fp32(const float* __restrict__ W, const float* __restrict__ x,
                          float* __restrict__ y, int K) {
  SHARED(float, scratch, 32);
  const float* row = W + (size_t)blockIdx.x * K;
  float acc = 0.0f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) acc += row[k] * x[k];
  acc = block_reduce_sum(acc, scratch);
  if (threadIdx.x == 0) y[blockIdx.x] = acc;
}

// ---------------------------------------------------------------------------------------
// Variant 2: the same, with float4 weight loads. Pure instruction-count and memory-level
// parallelism, exactly as in 01_copy.cu — identical bytes, fewer instructions to move them.
// ---------------------------------------------------------------------------------------
__global__ void gemv_fp32_vec4(const float4* __restrict__ W, const float4* __restrict__ x,
                               float* __restrict__ y, int K4) {
  SHARED(float, scratch, 32);
  const float4* row = W + (size_t)blockIdx.x * K4;
  float acc = 0.0f;
  for (int k = threadIdx.x; k < K4; k += blockDim.x) {
    float4 wv = row[k], xv = x[k];
    acc += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
  }
  acc = block_reduce_sum(acc, scratch);
  if (threadIdx.x == 0) y[blockIdx.x] = acc;
}

// ---------------------------------------------------------------------------------------
// Variant 3: int8 weights, symmetric per-group scales.
//
//     w ~= q * s,   q in [-127, 127],   s = max|w in group| / 127
//
// One scale per GROUP=128 weights along K, so the scale table costs K/GROUP floats per row —
// 1/128th of the weights in fp32, about 3% overhead on top of the int8 payload. Per-*tensor*
// scaling has no overhead at all and is markedly less accurate, because one outlier channel
// then sets the scale for the entire matrix; per-group is the standard compromise and the
// reason GPTQ and AWQ both quote a group size.
//
// The multiply is still fp32. `(float)q[k] * s` is two instructions on data already in a
// register, and the kernel has ~50 cycles of nothing to do while the next weights arrive.
// ---------------------------------------------------------------------------------------
__global__ void gemv_int8(const int8_t* __restrict__ Wq, const float* __restrict__ scales,
                          const float* __restrict__ x, float* __restrict__ y, int K,
                          int groups) {
  SHARED(float, scratch, 32);
  const int8_t* row = Wq + (size_t)blockIdx.x * K;
  const float* rs = scales + (size_t)blockIdx.x * groups;
  float acc = 0.0f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    acc += (float)row[k] * rs[k / GROUP] * x[k];
  acc = block_reduce_sum(acc, scratch);
  if (threadIdx.x == 0) y[blockIdx.x] = acc;
}

// ---------------------------------------------------------------------------------------
// Variant 4: int4 weights, asymmetric per-group scale and zero point, two weights per byte.
//
//     w ~= (q - z) * s,   q in [0, 15]
//
// Asymmetric because 4 bits is few enough that wasting one of them on a sign matters: a
// group whose weights all happen to be positive gets 16 levels instead of 8. The zero point
// costs one more float per group, which at GROUP=128 is another 3%.
//
// The packing is the fiddly part and the part worth reading closely. Two weights share a byte:
// even k in the low nibble, odd k in the high nibble. So thread t handles byte t, meaning
// weights 2t and 2t+1 — the warp reads 32 consecutive bytes, one 32-byte sector, perfectly
// coalesced. It also means each thread touches x[2t] and x[2t+1], which is a strided access
// into x. That would be a problem if x came from HBM. It does not; see variant 1.
//
// Note the cast through `int` before subtracting the zero point. `(q - z)` on unsigned nibbles
// wraps instead of going negative, and the resulting kernel is wrong only for weights below
// the zero point — which is half of them, which somehow still looks plausible in a loss curve.
// ---------------------------------------------------------------------------------------
__global__ void gemv_int4(const uint8_t* __restrict__ Wq, const float* __restrict__ scales,
                          const float* __restrict__ zeros, const float* __restrict__ x,
                          float* __restrict__ y, int K, int groups) {
  SHARED(float, scratch, 32);
  const uint8_t* row = Wq + (size_t)blockIdx.x * (K / 2);
  const float* rs = scales + (size_t)blockIdx.x * groups;
  const float* rz = zeros + (size_t)blockIdx.x * groups;

  float acc = 0.0f;
  for (int b = threadIdx.x; b < K / 2; b += blockDim.x) {
    uint8_t packed = row[b];
    int k0 = 2 * b;                       // GROUP is even, so k0 and k0+1 share a group
    int g = k0 / GROUP;
    float s = rs[g], z = rz[g];
    float w0 = ((float)(int)(packed & 0x0F) - z) * s;
    float w1 = ((float)(int)(packed >> 4) - z) * s;
    acc += w0 * x[k0] + w1 * x[k0 + 1];
  }
  acc = block_reduce_sum(acc, scratch);
  if (threadIdx.x == 0) y[blockIdx.x] = acc;
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  int N = 8, K = 256;
#else
  int N = argc > 1 ? std::atoi(argv[1]) : 4096;   // output features
  int K = argc > 2 ? std::atoi(argv[2]) : 4096;   // input features
#endif
  (void)argc; (void)argv;
  K = (K / GROUP) * GROUP;
  const int groups = K / GROUP;
  const size_t NW = (size_t)N * K;

  std::vector<float> hW(NW), hx(K);
  bench::fill(hW.data(), NW, 7);
  bench::fill(hx.data(), K, 11);

  // ---- quantize on the host, exactly as the kernels will interpret it -------------------
  std::vector<int8_t> q8(NW);
  std::vector<float> s8((size_t)N * groups);
  std::vector<uint8_t> q4(NW / 2);
  std::vector<float> s4((size_t)N * groups), z4((size_t)N * groups);

  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < groups; ++g) {
      const float* w = &hW[(size_t)n * K + (size_t)g * GROUP];
      float amax = 0.0f, lo = w[0], hi = w[0];
      for (int i = 0; i < GROUP; ++i) {
        amax = std::max(amax, std::fabs(w[i]));
        lo = std::min(lo, w[i]);
        hi = std::max(hi, w[i]);
      }
      // int8, symmetric
      float s = amax > 0 ? amax / 127.0f : 1.0f;
      s8[(size_t)n * groups + g] = s;
      for (int i = 0; i < GROUP; ++i) {
        int qv = (int)std::lround(w[i] / s);
        q8[(size_t)n * K + (size_t)g * GROUP + i] = (int8_t)std::max(-127, std::min(127, qv));
      }
      // int4, asymmetric: map [lo, hi] onto [0, 15]
      float step = (hi - lo) / 15.0f;
      if (step <= 0) step = 1.0f;
      float zp = -lo / step;                       // the quantized value that means "zero"
      s4[(size_t)n * groups + g] = step;
      z4[(size_t)n * groups + g] = zp;
      for (int i = 0; i < GROUP; ++i) {
        int qv = (int)std::lround(w[i] / step + zp);
        qv = std::max(0, std::min(15, qv));
        size_t k = (size_t)g * GROUP + i;
        size_t byte = ((size_t)n * K + k) / 2;
        if (k % 2 == 0)
          q4[byte] = (uint8_t)((q4[byte] & 0xF0) | (uint8_t)qv);
        else
          q4[byte] = (uint8_t)((q4[byte] & 0x0F) | (uint8_t)(qv << 4));
      }
    }
  }

  // ---- three references, in double ------------------------------------------------------
  // Each kernel is checked against a reference that models *its own* numerics. Checking the
  // int4 kernel against the fp32 answer would conflate two different questions — "is the
  // kernel right" and "how much accuracy does 4-bit cost" — and the second one is much larger
  // than the first, so a broken kernel would hide inside the quantization error.
  std::vector<float> want32(N), want8(N), want4(N);
  for (int n = 0; n < N; ++n) {
    double a32 = 0, a8 = 0, a4 = 0;
    for (int k = 0; k < K; ++k) {
      int g = k / GROUP;
      a32 += (double)hW[(size_t)n * K + k] * hx[k];
      a8 += (double)q8[(size_t)n * K + k] * s8[(size_t)n * groups + g] * hx[k];
      size_t byte = ((size_t)n * K + k) / 2;
      int qv = (k % 2 == 0) ? (q4[byte] & 0x0F) : (q4[byte] >> 4);
      a4 += ((double)qv - z4[(size_t)n * groups + g]) * s4[(size_t)n * groups + g] * hx[k];
    }
    want32[n] = (float)a32;
    want8[n] = (float)a8;
    want4[n] = (float)a4;
  }

  float *dW, *dx, *dy, *ds8, *ds4, *dz4;
  int8_t* dq8;
  uint8_t* dq4;
  CUDA_CHECK(cudaMalloc((void**)&dW, NW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dx, (size_t)K * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dy, (size_t)N * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dq8, NW));
  CUDA_CHECK(cudaMalloc((void**)&dq4, NW / 2));
  CUDA_CHECK(cudaMalloc((void**)&ds8, s8.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&ds4, s4.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dz4, z4.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dW, hW.data(), NW * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dq8, q8.data(), NW, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dq4, q4.data(), NW / 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ds8, s8.data(), s8.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ds4, s4.data(), s4.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dz4, z4.data(), z4.size() * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<float> hy(N);
  std::vector<bench::Row> rows;
  const double base_bytes = (double)NW * sizeof(float);

  auto run = [&](const char* name, double traffic, const float* want, auto&& launch) {
    CUDA_CHECK(cudaMemset(dy, 0, (size_t)N * sizeof(float)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev, 30, 10);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
    r.err = bench::max_rel_err(hy.data(), want, N);
    r.bytes = traffic;
    r.flops = 2.0 * NW;
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.2f FLOP/byte, %.2fx fp32 traffic",
                  2.0 * NW / traffic, traffic / base_bytes);
    r.note = buf;
    rows.push_back(r);
  };

  const double scale_bytes = 2.0 * (double)N * groups * sizeof(float);   // scales + zeros
  std::printf("problem   : y = W x, W is %d x %d, group size %d (%d groups per row)\n", N, K,
              GROUP, groups);
  std::printf("weights   : fp32 %.1f MB   int8 %.1f MB   int4 %.1f MB (+ %.2f MB of scales)\n",
              base_bytes / 1048576.0, (double)NW / 1048576.0, (double)NW / 2 / 1048576.0,
              scale_bytes / 1048576.0);
  bench::header(dev);

  run("1 fp32", base_bytes, want32.data(), [&] {
    KERNEL_LAUNCH(gemv_fp32, dim3(N), dim3(BLOCK), 0, dW, dx, dy, K);
  });
  run("2 fp32, float4 loads", base_bytes, want32.data(), [&] {
    KERNEL_LAUNCH(gemv_fp32_vec4, dim3(N), dim3(BLOCK), 0, (const float4*)dW,
                  (const float4*)dx, dy, K / 4);
  });
  run("3 int8, group scales", (double)NW + scale_bytes / 2, want8.data(), [&] {
    KERNEL_LAUNCH(gemv_int8, dim3(N), dim3(BLOCK), 0, dq8, ds8, dx, dy, K, groups);
  });
  run("4 int4, scale + zero point", (double)NW / 2 + scale_bytes, want4.data(), [&] {
    KERNEL_LAUNCH(gemv_int4, dim3(N), dim3(BLOCK), 0, dq4, ds4, dz4, dx, dy, K, groups);
  });

  const double tol = 1e-4;
  bench::rows_out(rows, dev, tol);

  // The other half of the story, reported separately and never folded into the pass/fail:
  // what the quantization itself costs in accuracy. These numbers are not bugs.
  std::printf("\nquantization error vs the fp32 answer (this is the accuracy you are buying with,\n"
              "as relative L2 over the output vector — a per-element ratio would mostly measure\n"
              "cancellation in a random dot product, not the quantizer):\n");
  std::printf("  int8, per-group symmetric : %.3e\n", bench::rel_l2(want8.data(), want32.data(), N));
  std::printf("  int4, per-group asymmetric: %.3e\n", bench::rel_l2(want4.data(), want32.data(), N));
  std::printf("\nOn a memory-bound GEMV the time ratios should track the traffic ratios above,\n"
              "not the FLOP counts — which are identical for all four rows.\n");

  CUDA_CHECK(cudaFree(dW));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(dy));
  CUDA_CHECK(cudaFree(dq8));
  CUDA_CHECK(cudaFree(dq4));
  CUDA_CHECK(cudaFree(ds8));
  CUDA_CHECK(cudaFree(ds4));
  CUDA_CHECK(cudaFree(dz4));
  return bench::verdict(rows, tol);
}
