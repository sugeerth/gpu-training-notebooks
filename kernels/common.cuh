// common.cuh — the measurement half of every kernel in this directory.
//
// The kernels are the interesting part to write and the boring part to trust. Almost every
// "my kernel is 3x faster" claim that turns out to be wrong is wrong here, not in the kernel:
//
//   * timed with wall clock across an asynchronous launch, so it measured the launch
//   * timed on the first call, so it measured JIT and context setup
//   * timed with the input already sitting in L2 from the previous rep, so a memory-bound
//     kernel reported bandwidth the card does not have
//   * reported as a mean over a handful of reps, so one preemption moved the number
//   * reported without a denominator, so nobody could tell 40% of peak from 4%
//
// So: CUDA events, warmup, an L2 flush between reps, median and MAD rather than mean and
// stddev, and every result divided by the hardware's own ceiling. A kernel that reports
// 92% of achievable bandwidth is finished. One that reports 9% has a bug, and the number
// says so without anyone needing a reference implementation to compare against.
#pragma once
#include "cuda_shim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if !SHIM_BUILD
#include <cuda_runtime.h>
#endif

#define CUDA_CHECK(expr)                                                          \
  do {                                                                            \
    cudaError_t _e = (expr);                                                      \
    if (_e != cudaSuccess) {                                                      \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),    \
                   __FILE__, __LINE__);                                           \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

namespace bench {

// ---------------------------------------------------------------------------------------
// What the hardware can actually do. Every measurement below is reported as a fraction of
// one of these two numbers, because a kernel timing without a ceiling is not a result.
// ---------------------------------------------------------------------------------------
struct Device {
  std::string name = "CPU (cuda_shim emulation)";
  double peak_gbps = 0;      // theoretical HBM/GDDR bandwidth
  double peak_gflops = 0;    // fp32 FMA peak, no tensor cores
  size_t l2_bytes = 0;
  int sms = 0;
  bool real = false;

  // Peak is a number from a spec sheet; nothing reaches it. STREAM-like kernels land at
  // 80-90% on HBM parts, so that is the bar a memory-bound kernel is held to here.
  double achievable_gbps() const { return peak_gbps * 0.90; }
};

#if SHIM_BUILD
inline Device query_device() { return Device{}; }
#else
// fp32 FMA lanes per SM, by compute capability. (Tensor cores are a separate, much larger
// number and deliberately not folded in here — see 03_sgemm.cu.)
inline int cores_per_sm(int major, int minor) {
  switch (major * 10 + minor) {
    case 60: return 64;                       // P100
    case 61: case 62: return 128;             // Pascal
    case 70: case 72: case 75: return 64;     // Volta, Turing
    case 80: return 64;                       // A100
    case 86: case 87: case 89: return 128;    // Ampere consumer, Ada
    case 90: return 128;                      // Hopper
    case 100: case 120: return 128;           // Blackwell
    default: return 128;
  }
}

inline Device query_device() {
  Device d;
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
  d.name = p.name;
  d.sms = p.multiProcessorCount;
  d.l2_bytes = (size_t)p.l2CacheSize;
  // memoryClockRate is in kHz and quotes the DDR rate, hence the factor of 2.
  d.peak_gbps = 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
  d.peak_gflops = (double)p.multiProcessorCount * cores_per_sm(p.major, p.minor) * 2.0 *
                  (p.clockRate * 1e3) / 1e9;
  d.real = true;
  return d;
}
#endif

// ---------------------------------------------------------------------------------------
// The L2 flush. Without it, a kernel that reads 32 MB on a card with a 50 MB L2 reports the
// bandwidth of L2, not of HBM — often 3-5x the real number, and always in the direction
// that flatters you.
// ---------------------------------------------------------------------------------------
class L2Flusher {
 public:
  explicit L2Flusher(const Device& d) {
    bytes_ = d.l2_bytes ? d.l2_bytes * 3 : 0;
    if (bytes_) CUDA_CHECK(cudaMalloc(&buf_, bytes_));
  }
  ~L2Flusher() { if (buf_) cudaFree(buf_); }
  void flush() { if (buf_) CUDA_CHECK(cudaMemset(buf_, 0, bytes_)); }

 private:
  void* buf_ = nullptr;
  size_t bytes_ = 0;
};

// ---------------------------------------------------------------------------------------
// Robust statistics. Median and MAD rather than mean and stddev: GPU timing distributions
// are not Gaussian, they are a tight mode with a right tail from clock drift and
// preemption. A mean chases the tail; a median ignores it.
// ---------------------------------------------------------------------------------------
struct Stats {
  double median_ms = 0, mad_ms = 0, min_ms = 0, p95_ms = 0;
  int reps = 0;
  bool timed = false;  // false on a shim build: correctness only, no timing claim
};

inline Stats summarize(std::vector<double> ms) {
  Stats s;
  if (ms.empty()) return s;
  std::sort(ms.begin(), ms.end());
  s.reps = (int)ms.size();
  s.min_ms = ms.front();
  s.median_ms = ms[ms.size() / 2];
  s.p95_ms = ms[(size_t)(0.95 * (ms.size() - 1))];
  std::vector<double> dev;
  dev.reserve(ms.size());
  for (double v : ms) dev.push_back(std::fabs(v - s.median_ms));
  std::sort(dev.begin(), dev.end());
  s.mad_ms = dev[dev.size() / 2];
  s.timed = true;
  return s;
}

// Time a launch. `fn` must enqueue the kernel and nothing else — no allocation, no copies,
// no host-side work that would land inside the event window.
template <class Fn>
Stats time_kernel(Fn&& fn, const Device& dev, int reps = 50, int warmup = 10) {
#if SHIM_BUILD
  (void)dev; (void)reps; (void)warmup;
  fn();  // correctness run only; the shim has no memory hierarchy to measure
  return Stats{};
#else
  L2Flusher flusher(dev);
  for (int i = 0; i < warmup; ++i) fn();   // JIT, context, clock ramp
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> samples;
  samples.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    flusher.flush();                        // cold cache, every rep
    CUDA_CHECK(cudaEventRecord(start));
    fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back((double)ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return summarize(samples);
#endif
}

// ---------------------------------------------------------------------------------------
// Reporting. One row per variant: time, the two rates, and the fraction of the ceiling the
// kernel is actually bound by. `bytes` is HBM traffic the kernel *must* do (compulsory
// misses), `flops` the useful arithmetic — both computed from the problem, never from the
// implementation, so a worse implementation gets a worse number instead of a smaller
// denominator.
// ---------------------------------------------------------------------------------------
struct Row {
  std::string name;
  Stats st;
  double bytes = 0, flops = 0;
  double err = 0;
  std::string note;   // e.g. how much HBM traffic this variant needed, when that is the point
};

inline void header(const Device& d) {
  std::printf("device    : %s\n", d.name.c_str());
  if (d.real) {
    std::printf("peak      : %.0f GB/s memory, %.1f TFLOP/s fp32, %d SMs, %.1f MB L2\n",
                d.peak_gbps, d.peak_gflops / 1e3, d.sms, d.l2_bytes / 1048576.0);
    std::printf("ridge     : %.1f FLOP/byte  (below this a kernel is memory-bound)\n",
                d.peak_gflops * 1e9 / (d.peak_gbps * 1e9));
  } else {
    std::printf("peak      : n/a — this is the CPU emulation, correctness only.\n"
                "            Timings are omitted rather than reported as GPU numbers.\n");
  }
  std::printf("\n%-30s %10s %8s %10s %9s %9s  %s\n", "variant", "median ms", "±MAD",
              "GB/s", "GFLOP/s", "max err", "");
  std::printf("%s\n", std::string(92, '-').c_str());
}

inline void row(const Row& r, const Device& d, double tol) {
  const char* ok = (r.err <= tol) ? "ok" : "WRONG";
  if (!r.st.timed) {
    std::printf("%-30s %10s %8s %10s %9s %9.2e  %-5s %s\n", r.name.c_str(), "-", "-", "-", "-",
                r.err, ok, r.note.c_str());
    return;
  }
  double sec = r.st.median_ms / 1e3;
  double gbps = r.bytes / sec / 1e9;
  double gflops = r.flops / sec / 1e9;
  char note[80] = "";
  if (d.real) {
    // Report the binding ceiling, not the flattering one: whichever of the two rates is a
    // larger fraction of its own peak is the resource this kernel is actually up against.
    double fb = gbps / d.achievable_gbps(), fc = gflops / d.peak_gflops;
    std::snprintf(note, sizeof note, "%.0f%% of %s", 100 * std::max(fb, fc),
                  fb >= fc ? "achievable BW" : "fp32 peak");
  }
  std::printf("%-30s %10.4f %8.4f %10.1f %9.1f %9.2e  %-5s %s%s%s\n", r.name.c_str(),
              r.st.median_ms, r.st.mad_ms, gbps, gflops, r.err, ok, note,
              r.note.empty() ? "" : "  ", r.note.c_str());
}

inline void rows_out(const std::vector<Row>& rows, const Device& d, double tol) {
  for (const auto& r : rows) row(r, d, tol);
}

// ---------------------------------------------------------------------------------------
// Correctness. Every kernel checks itself against a CPU reference on every run, including
// the timed one, because a fast wrong kernel is the single easiest thing to ship.
// ---------------------------------------------------------------------------------------
inline double max_rel_err(const float* got, const float* want, size_t n) {
  double worst = 0;
  for (size_t i = 0; i < n; ++i) {
    double denom = std::fabs((double)want[i]);
    if (denom < 1e-3) denom = 1e-3;
    worst = std::max(worst, std::fabs((double)got[i] - (double)want[i]) / denom);
  }
  return worst;
}

// Relative L2 error, ||got - want|| / ||want||. The right metric when the two vectors are
// genuinely different computations (fp32 vs int4, say) rather than one computation done twice:
// a per-element relative error divides by individual entries, and a dot product of random
// signs has entries near zero that make the ratio blow up for reasons that are about
// cancellation in the reference, not about the thing being measured.
inline double rel_l2(const float* got, const float* want, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = (double)got[i] - (double)want[i];
    num += d * d;
    den += (double)want[i] * (double)want[i];
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

inline double max_rel_err_scalar(float got, float want) {
  double denom = std::fabs((double)want);
  if (denom < 1e-3) denom = 1e-3;
  return std::fabs((double)got - (double)want) / denom;
}

// Deterministic input. rand() differs across libc versions, which quietly makes a test
// unreproducible between machines; this does not.
inline void fill(float* p, size_t n, unsigned seed = 1) {
  unsigned s = seed * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    p[i] = ((float)(s >> 8) / 8388608.0f) - 1.0f;  // roughly [-1, 1)
  }
}

// Exit code carries the verdict so `make check` and CI can rely on it.
inline int verdict(const std::vector<Row>& rows, double tol) {
  int bad = 0;
  for (const auto& r : rows)
    if (!(r.err <= tol)) ++bad;   // written this way so a NaN error counts as a failure
  std::printf("\n%s: %d/%d variants within %.1e\n", bad ? "FAIL" : "PASS",
              (int)rows.size() - bad, (int)rows.size(), tol);
  return bad ? 1 : 0;
}

}  // namespace bench
