// cuda_shim.hpp — run a .cu file on a CPU, with g++, with no CUDA installed.
//
// Why this exists
// ---------------
// A CUDA kernel you cannot run is a picture of a CUDA kernel. Most people reading this repo do
// not have an NVIDIA GPU in front of them, and CI certainly does not. So every kernel here
// compiles two ways from the *same source*:
//
//     nvcc -O3 02_reduce.cu -o reduce        # real GPU, real timings
//     g++  -O2 -x c++ 02_reduce.cu -o reduce # this file: correctness only, anywhere
//
// The shim emulates one block at a time, with one std::thread per CUDA thread, and real
// barriers. That makes it *semantically* faithful where it matters for learning:
//
//   * __syncthreads() is a real barrier, so a missing one can still deadlock or race
//   * warp shuffles really do move data between lanes, so an off-by-one in a reduction
//     tree produces a wrong answer here exactly as it would on a GPU
//   * shared memory is really shared between the threads of a block, and really is not
//     shared between blocks
//   * blocks run in an arbitrary (here: sequential) order, so code that assumes an order
//     is wrong here too
//
// What it deliberately does NOT emulate: performance. There is no memory hierarchy, no
// coalescing, no bank conflicts, no occupancy. A kernel that is 40x faster on a GPU is
// indistinguishable here. Timing numbers from a shim build are labelled as such and are
// never presented as GPU results.
//
// Two source-level concessions, both marked at every use site:
//   KERNEL_LAUNCH(fn, grid, block, shmem, ...)   instead of  fn<<<grid, block, shmem>>>(...)
//   SHARED(float, tile, 64)                      instead of  __shared__ float tile[64]
// `<<<>>>` is not C++, and `__shared__` is a declaration prefix that a macro cannot rewrite
// into a per-block allocation. Everything else is ordinary CUDA C++.
#pragma once

#ifdef __CUDACC__
// ---------------------------------------------------------------------------------------
// Real CUDA. The macros are thin.
// ---------------------------------------------------------------------------------------
#include <cuda_runtime.h>
#define KERNEL_LAUNCH(fn, grid, block, shmem, ...) fn<<<(grid), (block), (shmem)>>>(__VA_ARGS__)
#define SHARED(T, name, n) __shared__ T name[n]
#define SHIM_BUILD 0

#else
// ---------------------------------------------------------------------------------------
// CPU emulation.
// ---------------------------------------------------------------------------------------
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#define SHIM_BUILD 1
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

// Vector types. On a GPU these are not a convenience — a float4 load is one 128-bit
// instruction instead of four 32-bit ones, which is the difference between saturating a
// memory controller and not. Here they are only a convenience.
struct alignas(16) float4 { float x, y, z, w; };
struct alignas(8) float2 { float x, y; };
struct alignas(16) int4 { int x, y, z, w; };
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }
inline float2 make_float2(float x, float y) { return float2{x, y}; }

namespace shim {

// A reusable counting barrier. Generation counter so a thread that races ahead into the next
// barrier cannot be released by the previous one.
class Barrier {
 public:
  explicit Barrier(unsigned n) : n_(n), waiting_(0), gen_(0) {}
  void wait() {
    if (n_ <= 1) return;
    std::unique_lock<std::mutex> lk(mu_);
    unsigned g = gen_;
    if (++waiting_ == n_) {
      waiting_ = 0;
      ++gen_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return gen_ != g; });
    }
  }

 private:
  unsigned n_, waiting_, gen_;
  std::mutex mu_;
  std::condition_variable cv_;
};

constexpr unsigned kWarpSize = 32;

// Everything one emulated thread block needs: a block-wide barrier, one barrier per warp
// (for shuffles), the shared-memory arena, and a lane exchange buffer for shuffles.
struct BlockCtx {
  unsigned nthreads;
  Barrier bar;
  std::vector<Barrier*> warp_bars;
  std::map<int, void*> shared;
  std::mutex mu;
  std::vector<uint64_t> exch;

  explicit BlockCtx(unsigned nt) : nthreads(nt), bar(nt), exch(nt, 0) {
    unsigned nwarps = (nt + kWarpSize - 1) / kWarpSize;
    for (unsigned w = 0; w < nwarps; ++w) {
      unsigned in_warp = nt - w * kWarpSize;
      if (in_warp > kWarpSize) in_warp = kWarpSize;
      warp_bars.push_back(new Barrier(in_warp));
    }
  }
  ~BlockCtx() {
    for (auto* b : warp_bars) delete b;
    for (auto& kv : shared) std::free(kv.second);
  }
};

inline thread_local dim3 tls_threadIdx;
inline thread_local dim3 tls_blockIdx;
inline thread_local dim3 tls_blockDim;
inline thread_local dim3 tls_gridDim;
inline thread_local BlockCtx* tls_block = nullptr;
inline thread_local unsigned tls_flat = 0;  // flattened thread id within the block

inline unsigned lane_id() { return tls_flat % kWarpSize; }
inline unsigned warp_id() { return tls_flat / kWarpSize; }

inline void sync_threads() { tls_block->bar.wait(); }
inline void sync_warp() { tls_block->warp_bars[warp_id()]->wait(); }

// Shared memory: the first thread to arrive allocates, the barrier publishes the pointer.
// Like __syncthreads(), this must be reached by every thread in the block, which is why
// SHARED() belongs at the top of a kernel and never inside a divergent branch.
inline void* shared_alloc(int id, size_t bytes) {
  BlockCtx* c = tls_block;
  {
    std::lock_guard<std::mutex> lk(c->mu);
    if (c->shared.find(id) == c->shared.end()) c->shared[id] = std::calloc(1, bytes);
  }
  c->bar.wait();
  return c->shared[id];
}

// Shuffles. Write my value into the block-wide exchange slot, sync the warp, read the slot
// belonging to the source lane. Two warp barriers, because the read must not race the next
// shuffle's write.
//
// Every thread in the warp reaches both barriers even when its source lane is out of range,
// which is the same rule real hardware enforces: a shuffle inside divergent code, with a
// mask that does not match who actually arrives, hangs here exactly as it hangs there.
template <typename T>
inline T shfl_generic(T v, unsigned lane_in_width, bool valid, unsigned width) {
  BlockCtx* c = tls_block;
  uint64_t bits = 0;
  std::memcpy(&bits, &v, sizeof(T));
  c->exch[tls_flat] = bits;
  sync_warp();
  unsigned base = (tls_flat / width) * width;
  unsigned idx = base + lane_in_width;
  uint64_t got = (valid && idx < c->nthreads) ? c->exch[idx] : bits;
  sync_warp();
  T out;
  std::memcpy(&out, &got, sizeof(T));
  return out;
}

}  // namespace shim

#define threadIdx shim::tls_threadIdx
#define blockIdx shim::tls_blockIdx
#define blockDim shim::tls_blockDim
#define gridDim shim::tls_gridDim
#define __syncthreads() shim::sync_threads()
#define __syncwarp(...) shim::sync_warp()
#define warpSize 32

// shfl_down past the end of the (sub)warp returns the thread's own value — the property
// that lets a power-of-two reduction tree run without a bounds check in the kernel.
template <typename T>
inline T __shfl_down_sync(unsigned, T v, unsigned delta, unsigned width = 32) {
  unsigned t = (shim::lane_id() % width) + delta;
  return shim::shfl_generic(v, t, t < width, width);
}
template <typename T>
inline T __shfl_xor_sync(unsigned, T v, unsigned mask, unsigned width = 32) {
  unsigned t = (shim::lane_id() % width) ^ mask;
  return shim::shfl_generic(v, t, t < width, width);
}
template <typename T>
inline T __shfl_sync(unsigned, T v, unsigned src, unsigned width = 32) {
  return shim::shfl_generic(v, src % width, true, width);
}

// Atomics. A single global lock is glacial and completely correct, which is the right
// trade for a correctness emulator.
namespace shim {
inline std::mutex& atomic_mu() {
  static std::mutex m;
  return m;
}
}  // namespace shim
template <typename T>
inline T atomicAdd(T* addr, T val) {
  std::lock_guard<std::mutex> lk(shim::atomic_mu());
  T old = *addr;
  *addr = old + val;
  return old;
}
template <typename T>
inline T atomicMax(T* addr, T val) {
  std::lock_guard<std::mutex> lk(shim::atomic_mu());
  T old = *addr;
  if (val > old) *addr = val;
  return old;
}
template <typename T>
inline T atomicMin(T* addr, T val) {
  std::lock_guard<std::mutex> lk(shim::atomic_mu());
  T old = *addr;
  if (val < old) *addr = val;
  return old;
}

// CUDA gives device code integer `min`/`max` as builtins. `<algorithm>` supplies the
// std:: versions but only inside the std namespace, and only for matching argument types, so
// kernel source written against the builtins does not compile here without these.
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }

inline float __expf(float x) { return expf(x); }
inline float __logf(float x) { return logf(x); }
inline float __fdividef(float a, float b) { return a / b; }
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
inline float __frcp_rn(float x) { return 1.0f / x; }

// ---- the CUDA runtime API, as much of it as these programs use ----------------------
typedef int cudaError_t;
#define cudaSuccess 0
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
inline cudaError_t cudaMalloc(void** p, size_t n) { *p = std::malloc(n); return *p ? 0 : 1; }
template <typename T>
inline cudaError_t cudaMalloc(T** p, size_t n) { *p = (T*)std::malloc(n); return *p ? 0 : 1; }
inline cudaError_t cudaFree(void* p) { std::free(p); return 0; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
  std::memcpy(d, s, n);
  return 0;
}
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return 0; }
inline cudaError_t cudaDeviceSynchronize() { return 0; }
inline cudaError_t cudaGetLastError() { return 0; }
inline const char* cudaGetErrorString(cudaError_t) { return "ok"; }

// Launch: run each block to completion, threads within a block genuinely concurrent.
//
// Block *order* is deliberately not fixed. CUDA guarantees nothing about the order blocks run
// in, and a surprising amount of numerical code depends on it anyway — anything that
// accumulates with atomicAdd gets a different floating-point rounding depending on which
// block arrives first. On a GPU that shows up as a run-to-run difference in the last bits,
// which is the mechanism behind "why does my loss curve not reproduce" and behind logits that
// depend on what else was in the batch.
//
// Setting KB_SHIM_SHUFFLE=<seed> permutes the block order, so running a program twice with
// two seeds and comparing checksums detects that dependence *deterministically, on a CPU*.
// kernelbench uses exactly that as its determinism check.
namespace shim {

inline unsigned shuffle_seed() {
  const char* s = std::getenv("KB_SHIM_SHUFFLE");
  return s ? (unsigned)std::strtoul(s, nullptr, 10) : 0u;
}

template <class F, class... Args>
void launch(F fn, dim3 grid, dim3 block, size_t /*shmem*/, Args... args) {
  const unsigned nt = block.x * block.y * block.z;
  const size_t nblocks = (size_t)grid.x * grid.y * grid.z;

  std::vector<size_t> order(nblocks);
  for (size_t i = 0; i < nblocks; ++i) order[i] = i;
  if (unsigned seed = shuffle_seed()) {
    unsigned s = seed * 2654435761u + 1u;
    for (size_t i = nblocks; i > 1; --i) {      // Fisher-Yates, seeded and reproducible
      s = s * 1664525u + 1013904223u;
      std::swap(order[i - 1], order[s % i]);
    }
  }

  for (size_t idx : order) {
    const unsigned bx = (unsigned)(idx % grid.x);
    const unsigned by = (unsigned)((idx / grid.x) % grid.y);
    const unsigned bz = (unsigned)(idx / ((size_t)grid.x * grid.y));
    BlockCtx ctx(nt);
    std::vector<std::thread> threads;
    threads.reserve(nt);
    for (unsigned t = 0; t < nt; ++t) {
      threads.emplace_back([&, t]() {
        tls_block = &ctx;
        tls_flat = t;
        tls_blockDim = block;
        tls_gridDim = grid;
        tls_blockIdx = dim3(bx, by, bz);
        tls_threadIdx = dim3(t % block.x, (t / block.x) % block.y, t / (block.x * block.y));
        fn(args...);
      });
    }
    for (auto& th : threads) th.join();
  }
}
}  // namespace shim

#define KERNEL_LAUNCH(fn, grid, block, shmem, ...) \
  shim::launch(fn, (grid), (block), (shmem), ##__VA_ARGS__)
#define SHARED(T, name, n) T* name = (T*)shim::shared_alloc(__COUNTER__, sizeof(T) * (size_t)(n))

#endif  // __CUDACC__
