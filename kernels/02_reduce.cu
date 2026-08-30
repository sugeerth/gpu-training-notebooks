// 02_reduce.cu — a sum, five ways: divergence, bank conflicts, and why shuffles replaced
// shared memory for anything that fits in a warp.
//
//     nvcc -O3 -arch=native 02_reduce.cu -o build/02_reduce && build/02_reduce
//     make check
//
// Summing an array is the second-simplest kernel and it is *not* embarrassingly parallel:
// every thread's result has to reach every other thread's result. That makes it the standard
// vehicle for the three mechanisms that decide the speed of anything cooperative — and
// cooperative is what softmax, layernorm, RMSNorm, and attention all are.
//
// The three mechanisms, in the order the variants below expose them:
//
//   1. Divergence. A warp has one program counter. If lanes take different sides of a branch,
//      the hardware runs both sides and masks off the inactive lanes. `if (tid % 2 == 0)` does
//      not run at half cost; it runs at full cost with half the lanes idle.
//
//   2. Shared-memory bank conflicts. Shared memory is 32 banks, 4 bytes wide, striped:
//      address a lives in bank (a/4) % 32. One bank serves one address per cycle. If the 32
//      lanes of a warp hit 32 different banks, the access is one cycle. If they all hit the
//      same bank at different addresses, it is 32 cycles, serialized. A stride-2 access
//      pattern uses 16 banks and costs 2x; stride-32 uses one bank and costs 32x.
//
//   3. The register file is bigger and faster than shared memory. `__shfl_down_sync` reads
//      another lane's *register* directly. No shared memory, no __syncthreads, no bank
//      conflicts. For the last 32 elements of any reduction this is strictly better, and for
//      whole-warp work it eliminates shared memory entirely.
//
// Every variant does the identical global-memory phase — a grid-stride accumulate into one
// register — so the only thing that differs between rows is the tree. Otherwise the biggest
// term (reading n floats from HBM) would swamp the effect being measured, which is exactly
// how microbenchmarks come to show "no difference".
#include "common.cuh"

#if SHIM_BUILD
constexpr int BLOCK = 64;
#else
constexpr int BLOCK = 256;
#endif
constexpr unsigned FULL = 0xffffffffu;

// Identical in every variant: stream the input into one register per thread. Coalesced,
// grid-stride, exactly the shape 01_copy.cu argued for.
__device__ __forceinline__ float grid_stride_acc(const float* __restrict__ in, size_t n) {
  float acc = 0.0f;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    acc += in[i];
  return acc;
}

// ---------------------------------------------------------------------------------------
// Variant 1: interleaved addressing, divergent. The version everyone writes first.
//
//   stride=1: lanes 0,2,4,...  active   -> 50% of each warp idle
//   stride=2: lanes 0,4,8,...  active   -> 75% idle
//   ...
//
// The `%` is also a genuine integer division on the critical path. And the shared-memory
// pattern s[tid] += s[tid+stride] with tid stepping by 2*stride means active lanes touch
// banks 2*stride apart: at stride=16 all active lanes share two banks.
// ---------------------------------------------------------------------------------------
__global__ void reduce_divergent(const float* __restrict__ in, float* __restrict__ out,
                                 size_t n) {
  SHARED(float, s, BLOCK);
  unsigned tid = threadIdx.x;
  s[tid] = grid_stride_acc(in, n);
  __syncthreads();

  for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
    __syncthreads();   // outside the branch: every thread must reach every barrier
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// ---------------------------------------------------------------------------------------
// Variant 2: interleaved addressing, non-divergent. Same tree, renumbered so that the
// *contiguous* lanes are the active ones — at stride=1, lanes 0..127 work and 128..255 idle,
// so four whole warps are inactive rather than half of every warp. Divergence: gone.
//
// The bank conflicts are not. Thread tid touches s[2*stride*tid], so at stride=1 the warp
// reads banks 0,2,4,...,62 mod 32 — every bank hit exactly twice, a 2-way conflict. At
// stride=16 the whole warp is on one bank: 32-way, fully serialized. This variant is the
// reason "I removed the divergence and it barely helped" is such a common experience.
// ---------------------------------------------------------------------------------------
__global__ void reduce_interleaved(const float* __restrict__ in, float* __restrict__ out,
                                   size_t n) {
  SHARED(float, s, BLOCK);
  unsigned tid = threadIdx.x;
  s[tid] = grid_stride_acc(in, n);
  __syncthreads();

  for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
    unsigned idx = 2 * stride * tid;
    if (idx + stride < blockDim.x) s[idx] += s[idx + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// ---------------------------------------------------------------------------------------
// Variant 3: sequential addressing. The tree runs the other way — halve the *live range*
// each step instead of doubling the stride.
//
//   s[tid] += s[tid + stride]   with tid < stride
//
// Now lane t always touches s[t] and s[t+stride]. Consecutive lanes, consecutive addresses,
// 32 distinct banks: no conflicts at any step. And the active threads are always the low
// ones, so whole warps retire instead of whole warps running at half occupancy.
//
// One dead branch survives: for stride < 32 the entire block is a single warp's worth of
// work, and each __syncthreads() is a block-wide barrier for 32 threads that are already
// implicitly in lockstep. Variant 4 deletes those.
// ---------------------------------------------------------------------------------------
__global__ void reduce_sequential(const float* __restrict__ in, float* __restrict__ out,
                                  size_t n) {
  SHARED(float, s, BLOCK);
  unsigned tid = threadIdx.x;
  s[tid] = grid_stride_acc(in, n);
  __syncthreads();

  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// ---------------------------------------------------------------------------------------
// Variant 4: warp shuffles. A lane reads another lane's register, in one instruction, with
// no memory involved at all.
//
//     v += __shfl_down_sync(FULL, v, 16);   // lane L gets lane L+16's v
//     v += __shfl_down_sync(FULL, v,  8);
//     ...                                    // 5 instructions, 32 -> 1, no barriers
//
// Shared memory is then needed only to get one number per warp to one place: BLOCK/32 values,
// reduced by a single warp. A 256-thread block goes from 8 barriers and 8 shared round-trips
// to 1 barrier and 8 floats of shared traffic.
//
// The mask is not decoration. `FULL` asserts that all 32 lanes are present at this
// instruction. On Volta and later, lanes can be at different instructions (independent thread
// scheduling), and a shuffle whose mask names a lane that is not there is undefined behaviour
// — a class of bug that appears only under divergence, only on some inputs.
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    v += __shfl_down_sync(FULL, v, offset);
  return v;   // lane 0 holds the total; other lanes hold partial sums
}

__global__ void reduce_shuffle(const float* __restrict__ in, float* __restrict__ out,
                               size_t n) {
  SHARED(float, warp_sums, 32);          // at most 1024 threads / 32 = 32 warps
  unsigned tid = threadIdx.x;
  unsigned lane = tid % warpSize, warp = tid / warpSize;

  float v = warp_reduce_sum(grid_stride_acc(in, n));
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();                       // the only barrier in the kernel

  if (warp == 0) {
    unsigned nwarps = blockDim.x / warpSize;
    v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) out[blockIdx.x] = v;
  }
}

// ---------------------------------------------------------------------------------------
// Variant 5: the shape you would actually ship. Everything from variant 4, plus:
//
//   * float4 loads, for the reason 01_copy.cu measured
//   * a single atomicAdd per block instead of a second kernel launch
//
// The atomic is the interesting trade. One atomic per *block* (not per thread) is a few
// hundred contended operations for the whole reduction — free. It costs determinism: float
// addition is not associative, so the result depends on the order blocks happen to finish,
// and two runs of this kernel on identical input can differ in the last bits.
//
// That is usually fine and occasionally catastrophic. It is the same non-determinism that
// makes an LLM's logits depend on batch composition, which is why serving stacks that promise
// reproducible output use fixed-split reductions instead of atomics.
// ---------------------------------------------------------------------------------------
__global__ void reduce_atomic_vec4(const float4* __restrict__ in, float* __restrict__ out,
                                   size_t n4) {
  SHARED(float, warp_sums, 32);
  unsigned tid = threadIdx.x;
  unsigned lane = tid % warpSize, warp = tid / warpSize;

  float acc = 0.0f;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + tid; i < n4; i += stride) {
    float4 x = in[i];
    acc += (x.x + x.y) + (x.z + x.w);    // pairwise: shorter dependency chain than a+b+c+d
  }

  float v = warp_reduce_sum(acc);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();

  if (warp == 0) {
    unsigned nwarps = blockDim.x / warpSize;
    v = warp_reduce_sum((lane < nwarps) ? warp_sums[lane] : 0.0f);
    if (lane == 0) atomicAdd(out, v);
  }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  size_t n = 1 << 14;
  int grid = 8;
#else
  size_t n = argc > 1 ? (size_t)std::atoll(argv[1]) : (1u << 26);
  int grid = dev.sms ? dev.sms * 8 : 512;
#endif
  (void)argc; (void)argv;
  n &= ~(size_t)3;

  std::vector<float> h_in(n);
  bench::fill(h_in.data(), n);
  // All-positive input. Summing values that straddle zero gives a total near zero, and then
  // "relative error" measures nothing but catastrophic cancellation in the reference.
  for (size_t i = 0; i < n; ++i) h_in[i] = std::fabs(h_in[i]) + 0.25f;

  // Reference in double with Kahan compensation: the yardstick must be better than anything
  // it is measuring, or the test reports the yardstick's error.
  double sum = 0.0, c = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double y = (double)h_in[i] - c;
    double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  const float want = (float)sum;

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_out, (size_t)grid * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<float> h_out(grid);
  std::vector<bench::Row> rows;

  auto run = [&](const char* name, int n_partials, auto&& launch) {
    CUDA_CHECK(cudaMemset(d_out, 0, (size_t)grid * sizeof(float)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, (size_t)grid * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // The grid partials are summed on the host so that all five variants are compared at the
    // same tree depth; on a real deployment this is a second kernel launch, not a memcpy.
    double got = 0.0;
    for (int i = 0; i < n_partials; ++i) got += h_out[i];
    r.err = bench::max_rel_err_scalar((float)got, want);
    r.checksum = bench::hash_bytes(h_out.data(), (size_t)n_partials * sizeof(float));
    r.bytes = (double)n * sizeof(float);
    r.flops = (double)n;                  // one add per element
    rows.push_back(r);
  };

  bench::header(dev);
  run("1 interleaved, divergent", grid, [&] {
    KERNEL_LAUNCH(reduce_divergent, dim3(grid), dim3(BLOCK), 0, d_in, d_out, n);
  });
  run("2 interleaved, no divergence", grid, [&] {
    KERNEL_LAUNCH(reduce_interleaved, dim3(grid), dim3(BLOCK), 0, d_in, d_out, n);
  });
  run("3 sequential (no conflicts)", grid, [&] {
    KERNEL_LAUNCH(reduce_sequential, dim3(grid), dim3(BLOCK), 0, d_in, d_out, n);
  });
  run("4 warp shuffle", grid, [&] {
    KERNEL_LAUNCH(reduce_shuffle, dim3(grid), dim3(BLOCK), 0, d_in, d_out, n);
  });
  run("5 shuffle + float4 + atomic", 1, [&] {
    KERNEL_LAUNCH(reduce_atomic_vec4, dim3(grid), dim3(BLOCK), 0, (const float4*)d_in, d_out,
                  n / 4);
  });
  // fp32 tree reduction over n elements accumulates ~log2(n) * eps of relative error.
  const double tol = 1e-5;
  bench::rows_out(rows, dev, tol);

  std::printf("\nreference (Kahan, double): %.6f\n", sum);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  return bench::verdict(rows, tol, &dev);
}
