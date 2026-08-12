// 01_copy.cu — coalescing, or: why the same bytes cost 10x more depending on who reads them.
//
//     nvcc -O3 -arch=native 01_copy.cu -o build/01_copy && build/01_copy
//     make check                                # CPU, correctness only, no GPU needed
//
// Copying an array is the simplest possible kernel: zero arithmetic, `2n` floats of traffic,
// nothing to optimize. It is therefore the cleanest place to see the one rule that governs
// every memory-bound kernel — and *most* LLM inference kernels are memory-bound.
//
// The rule
// --------
// A GPU does not have loads. It has *transactions*. When a warp (32 threads on NVIDIA, 64 on
// AMD) executes one load instruction, the memory system collects the 32 addresses and issues
// the smallest set of 32-byte sectors that covers them. So:
//
//     addresses are consecutive floats  ->  32 x 4 B = 128 B = 4 sectors  -> 4 transactions
//     addresses are 4 KB apart          ->  32 distinct sectors           -> 32 transactions
//
// Same 128 bytes wanted, 8x the traffic moved, because a sector is the smallest thing DRAM
// will hand over. The bytes you asked for are not the bytes you paid for.
//
// The trap this kernel is built around: the *intuitive* decomposition — "thread 0 takes the
// first chunk, thread 1 takes the next chunk" — is precisely the bad one. It is how you would
// split work across CPU threads, where each core wants its own contiguous run to keep its own
// cache lines private. On a GPU the threads of a warp share an instruction, so they must
// share a cache line too. The decomposition that is right on a CPU is worst-case here.
#include "common.cuh"

// ---------------------------------------------------------------------------------------
// Variant 1: thread-per-chunk. The CPU decomposition. Thread t owns [t*C, (t+1)*C).
// Lane 0 reads element 0, lane 1 reads element C, lane 2 reads element 2C ... with C in the
// thousands, every lane in the warp lands in a different sector. 32 transactions per load.
// ---------------------------------------------------------------------------------------
__global__ void copy_thread_chunks(const float* __restrict__ in, float* __restrict__ out,
                                   size_t n) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = (size_t)gridDim.x * blockDim.x;
  size_t chunk = (n + total - 1) / total;
  size_t begin = tid * chunk;
  size_t end = begin + chunk < n ? begin + chunk : n;
  for (size_t i = begin; i < end; ++i) out[i] = in[i];
}

// ---------------------------------------------------------------------------------------
// Variant 2: the grid-stride loop. Lane t reads element base+t, so the warp's 32 addresses
// are 32 consecutive floats: one 128-byte line, 4 sectors, minimum possible traffic.
//
// The `i += stride` shape matters for a second reason. The grid is sized to the *machine*
// (a few waves of blocks per SM), not to the problem, so the same binary is efficient for
// n = 1e4 and n = 1e9, and the launch cost does not scale with n.
// ---------------------------------------------------------------------------------------
__global__ void copy_coalesced(const float* __restrict__ in, float* __restrict__ out,
                               size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    out[i] = in[i];
}

// ---------------------------------------------------------------------------------------
// Variant 3: 128-bit accesses. Coalescing is already perfect in variant 2 — the same bytes
// move — so why is this faster?
//
// Because a memory-bound kernel is not only limited by DRAM. It is also limited by how many
// load instructions the SM can issue and how many can be *in flight* at once. Each thread
// may have only so many outstanding misses; each SM only so many. Widening every access to
// 16 bytes quarters the instruction count and quarters the number of in-flight requests
// needed to cover the same latency. On most cards, scalar float copy sits around 70-80% of
// achievable bandwidth and float4 reaches 90%+.
//
// The cost: `in` and `out` must be 16-byte aligned, and n must be a multiple of 4. cudaMalloc
// guarantees 256-byte alignment; an offset into someone else's buffer does not.
// ---------------------------------------------------------------------------------------
__global__ void copy_vec4(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
    out[i] = in[i];
}

// ---------------------------------------------------------------------------------------
// Variant 4: 128-bit, four in flight. Little's Law applied to a memory system:
//
//     bytes in flight needed = bandwidth x latency
//
// An H100 moves ~3 TB/s with ~500 ns of HBM latency, so ~1.5 MB must be *outstanding* at
// every instant to keep the pipes full. A thread with one load in flight contributes 16
// bytes. You reach 1.5 MB either with ~100k concurrent threads, or with fewer threads each
// holding several independent loads.
//
// Which is why the two loops below are two loops. Issuing four loads before consuming any of
// them means the four latencies overlap. If the loads and stores were interleaved in one
// loop body, the compiler would still try to hoist them, but a dependency — or a register
// budget that forces a spill — turns four overlapped latencies into four serial ones.
// ---------------------------------------------------------------------------------------
__global__ void copy_vec4_unroll4(const float4* __restrict__ in, float4* __restrict__ out,
                                  size_t n4) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t base = (size_t)blockIdx.x * blockDim.x * 4 + threadIdx.x; base < n4;
       base += stride * 4) {
    float4 reg[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      size_t i = base + (size_t)u * blockDim.x;
      if (i < n4) reg[u] = in[i];            // all four issued...
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      size_t i = base + (size_t)u * blockDim.x;
      if (i < n4) out[i] = reg[u];           // ...before any is consumed
    }
  }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();

  // The emulator runs one OS thread per CUDA thread, so it gets a toy problem. A real GPU
  // needs an array several times larger than L2 or the measurement is of cache, not DRAM.
#if SHIM_BUILD
  size_t n = 1 << 14;
  int block = 64, grid = 8;
#else
  size_t n = argc > 1 ? (size_t)std::atoll(argv[1]) : (1u << 26);   // 256 MB in, 256 MB out
  int block = 256;
  int grid = dev.sms ? dev.sms * 32 : 1024;                          // ~32 blocks per SM
#endif
  (void)argc; (void)argv;
  n &= ~(size_t)3;  // float4 wants a multiple of 4

  std::vector<float> h_in(n), h_out(n);
  bench::fill(h_in.data(), n);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const double bytes = 2.0 * n * sizeof(float);   // read once, write once. Compulsory traffic.
  std::vector<bench::Row> rows;

  auto run = [&](const char* name, auto&& launch) {
    CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(float)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    r.err = bench::max_rel_err(h_out.data(), h_in.data(), n);
    r.bytes = bytes;
    rows.push_back(r);
  };

  bench::header(dev);
  run("1 thread-per-chunk (bad)", [&] {
    KERNEL_LAUNCH(copy_thread_chunks, dim3(grid), dim3(block), 0, d_in, d_out, n);
  });
  run("2 grid-stride, coalesced", [&] {
    KERNEL_LAUNCH(copy_coalesced, dim3(grid), dim3(block), 0, d_in, d_out, n);
  });
  run("3 float4 (128-bit)", [&] {
    KERNEL_LAUNCH(copy_vec4, dim3(grid), dim3(block), 0, (const float4*)d_in, (float4*)d_out,
                  n / 4);
  });
  run("4 float4 x4 in flight", [&] {
    KERNEL_LAUNCH(copy_vec4_unroll4, dim3(grid), dim3(block), 0, (const float4*)d_in,
                  (float4*)d_out, n / 4);
  });
  // A copy is exact: any nonzero error is a real bug, not accumulated rounding.
  const double tol = 0.0;
  bench::rows_out(rows, dev, tol);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return bench::verdict(rows, tol);
}
