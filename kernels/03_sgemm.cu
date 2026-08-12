// 03_sgemm.cu — C = A·B, four ways. The compute-bound counterpart to 01_copy.cu, and the
// place where the memory hierarchy stops being trivia and starts being the whole design.
//
//     nvcc -O3 -arch=native 03_sgemm.cu -o build/03_sgemm && build/03_sgemm
//     make check
//
// Why a matrix multiply is the interesting case
// ---------------------------------------------
// A copy has arithmetic intensity 0 FLOP/byte: nothing to do but move bytes. A GEMM of size
// n has 2n³ FLOPs over 3n² floats of data, so its intensity is O(n) — it can be as
// compute-bound as you like, *if* you get the reuse. The whole job is reuse.
//
// It is worth being precise here, because the two levels of tiling fix two *different*
// bottlenecks and the difference is routinely blurred.
//
// **HBM intensity is set by the block tile.** A block computing a BM x BN tile of C must read
// (BM + BN) x K values to do 2·BM·BN·K FLOPs, so
//
//     intensity = 2·BM·BN / (4·(BM + BN))  =  BM/4  for a square tile
//
//     naive (no tile)         0.25 FLOP/byte
//     TILE = 16  (variant 3)  4    FLOP/byte
//     BM = BN = 64 (var. 4)   16   FLOP/byte
//
// Against an A100's fp32 ridge of ~9.6, only the last one crosses. Note what that implies:
// register blocking does *not* raise HBM intensity — variant 4 wins the HBM argument purely by
// having a larger block tile.
//
// **Shared-memory traffic is set by the thread tile.** Variant 3's inner loop reads two floats
// from shared per FMA. Variant 4's reads TM + TN = 8 floats to do TM x TN = 16 FMAs — four
// times less shared traffic per unit of arithmetic. Shared memory bandwidth is what binds a
// kernel that has already fixed its HBM problem, which is why both levels exist and why "just
// use shared memory" gets you to variant 3 and no further.
//
// A sobering corollary you can see in the table each card prints: an L4 has ~30 TFLOP/s fp32
// against 300 GB/s, a ridge of ~101 FLOP/byte. A square block tile would have to be 404 wide
// to cross it. On parts like that, fp32 SIMT GEMM simply cannot become compute-bound, and
// tensor cores are not an optimization but the only route to the machine's arithmetic.
//
// Prerequisite: 01_copy.cu (coalescing) and 02_reduce.cu (shared memory, banks).
#include "common.cuh"

// ---------------------------------------------------------------------------------------
// Variant 1: naive, with the thread mapping the wrong way round.
//
// Every element of C is one thread; the thread walks the whole K dimension. The only thing
// that distinguishes this from variant 2 is which of threadIdx.x / threadIdx.y indexes the
// *column* of C. Here threadIdx.x indexes the row, so the 32 lanes of a warp write C[0][c],
// C[1][c], C[2][c]... — addresses N floats apart. 32 sectors per store, 32 per load of B.
//
// Same FLOPs, same algorithm, same occupancy, typically 5-10x slower. It is worth internalising
// how invisible this is in the source: the two kernels differ by swapping two letters.
// ---------------------------------------------------------------------------------------
__global__ void sgemm_naive_uncoalesced(const float* __restrict__ A, const float* __restrict__ B,
                                        float* __restrict__ C, int M, int N, int K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;   // <- lane index drives the ROW
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------------------
// Variant 2: the same kernel with the mapping fixed. Lanes now walk the column dimension, so
// B[k*N + col] and C[row*N + col] are consecutive across the warp: 4 sectors, not 32.
//
// A[row*K + k] is now *identical* across the warp — a broadcast, which the hardware serves
// from one sector. Still 2n³ HBM loads in principle, but the L2 and L1 absorb most of the
// redundancy, which is why this is fast enough to be tempting and slow enough to be wrong.
// ---------------------------------------------------------------------------------------
__global__ void sgemm_naive_coalesced(const float* __restrict__ A, const float* __restrict__ B,
                                      float* __restrict__ C, int M, int N, int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;   // <- lane index drives the COLUMN
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------------------
// Variant 3: shared-memory tiling. The block cooperatively stages a TILE x TILE square of A
// and of B into shared memory, then every thread does TILE multiply-adds out of it.
//
// Each element staged is read TILE times from shared and once from HBM, so HBM traffic drops
// by a factor of TILE. The two __syncthreads() are both mandatory and are the two places this
// kernel is most often broken:
//
//   * after the stores, so nobody reads a tile before it is written
//   * after the inner loop, so nobody overwrites a tile another thread is still reading
//
// Drop the second one and the kernel still passes on small inputs, because with one warp per
// block there is no one to race. It fails at scale, intermittently. The shim runs real
// threads with real barriers, so it fails here too.
// ---------------------------------------------------------------------------------------
constexpr int TILE = 16;

__global__ void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
  SHARED(float, As, TILE * TILE);
  SHARED(float, Bs, TILE * TILE);

  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;

  float acc = 0.0f;
  for (int t = 0; t < K; t += TILE) {
    As[ty * TILE + tx] = (row < M && t + tx < K) ? A[row * K + (t + tx)] : 0.0f;
    Bs[ty * TILE + tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty * TILE + k] * Bs[k * TILE + tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------------------
// Variant 4: two-level tiling — shared memory *and* registers. This is the shape of every
// production SGEMM (cuBLAS, CUTLASS, Triton's matmul template all look like this).
//
// The block owns a BM x BN tile of C. Each of the 256 threads owns a TM x TN sub-tile of that,
// held entirely in registers. The inner loop is:
//
//     load TM values of A and TN values of B from shared   ->  TM + TN reads
//     do TM * TN fused multiply-adds                       ->  TM * TN FLOPs
//
// With TM = TN = 4 that is 8 shared reads for 16 FMAs, versus variant 3's 2 reads for 1 FMA.
// The ratio of arithmetic to shared traffic went from 0.5 to 2.0 — a 4x cut in shared-memory
// pressure. Separately, and independently, the block tile grew from 16 to 64, which is what
// takes HBM intensity from 4 to 16 FLOP/byte. Both changes are in this kernel; only the second
// one is about HBM.
//
// Two details that are easy to miss and expensive to omit:
//
//   * `As` is stored transposed (As[k][m], not As[m][k]). The inner loop then reads TM
//     *consecutive* floats of A for a fixed k, instead of striding by BK. Same data, no bank
//     conflicts.
//   * the global->shared loads are strided by the block size so that consecutive threads read
//     consecutive addresses — the coalescing rule from 01_copy.cu, applied to the staging
//     loop rather than to the math.
// ---------------------------------------------------------------------------------------
constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
constexpr int NTHREADS = (BM * BN) / (TM * TN);   // 64*64/16 = 256

__global__ void __launch_bounds__(NTHREADS)
sgemm_blocked(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
              int M, int N, int K) {
  SHARED(float, As, BK * BM);   // transposed: As[k * BM + m]
  SHARED(float, Bs, BK * BN);   //             Bs[k * BN + n]

  const int tid = threadIdx.x;
  const int cRow = blockIdx.y, cCol = blockIdx.x;

  // Where this thread's TM x TN output sub-tile sits inside the block's BM x BN tile.
  const int threadCol = tid % (BN / TN);   // 0..15
  const int threadRow = tid / (BN / TN);   // 0..15

  // Where this thread helps load from. innerCol is the fast-varying index in both cases, so
  // the 32 lanes of a warp read 32 consecutive floats.
  const int innerColA = tid % BK, innerRowA = tid / BK;          // BK=8  -> 32 rows per pass
  const int innerColB = tid % BN, innerRowB = tid / BN;          // BN=64 -> 4 rows per pass
  const int strideA = NTHREADS / BK;                             // 32
  const int strideB = NTHREADS / BN;                             // 4

  A += (size_t)cRow * BM * K;
  B += (size_t)cCol * BN;
  C += (size_t)cRow * BM * N + (size_t)cCol * BN;

  float acc[TM][TN] = {};
  float regM[TM], regN[TN];

  for (int bk = 0; bk < K; bk += BK) {
    for (int off = 0; off < BM; off += strideA)
      As[innerColA * BM + innerRowA + off] = A[(size_t)(innerRowA + off) * K + innerColA];
    for (int off = 0; off < BK; off += strideB)
      Bs[(innerRowB + off) * BN + innerColB] = B[(size_t)(innerRowB + off) * N + innerColB];
    __syncthreads();

    A += BK;
    B += (size_t)BK * N;

    for (int dot = 0; dot < BK; ++dot) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[dot * BM + threadRow * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) regN[j] = Bs[dot * BN + threadCol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      C[(size_t)(threadRow * TM + i) * N + threadCol * TN + j] = acc[i][j];
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  int n = 64;         // one 64x64 block tile: enough to exercise every code path
#else
  int n = argc > 1 ? std::atoi(argv[1]) : 2048;
#endif
  (void)argc; (void)argv;
  n = (n / BM) * BM;  // variant 4 has no bounds checks in its inner loop, by design
  if (n < BM) n = BM;
  const int M = n, N = n, K = n;

  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
  bench::fill(hA.data(), hA.size(), 1);
  bench::fill(hB.data(), hB.size(), 2);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc((void**)&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

  // A full CPU reference is O(n^3) and would dominate the runtime at n=2048, so check a fixed
  // pseudo-random sample of output entries in double precision instead. Fixed, not random:
  // a test that checks different entries on every run is a test whose failures do not
  // reproduce.
  const int NCHECK = (int)std::min<size_t>(2048, hC.size());
  std::vector<size_t> probe(NCHECK);
  std::vector<double> want(NCHECK);
  {
    unsigned s = 12345;
    for (int i = 0; i < NCHECK; ++i) {
      s = s * 1664525u + 1013904223u;
      probe[i] = (size_t)(((unsigned long long)s << 8 | i) % hC.size());
      size_t r = probe[i] / N, c = probe[i] % N;
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += (double)hA[r * K + k] * (double)hB[(size_t)k * N + c];
      want[i] = acc;
    }
  }

  const double flops = 2.0 * M * N * K;
  const double bytes = ((double)M * K + (double)K * N + (double)M * N) * sizeof(float);
  std::vector<bench::Row> rows;

  auto run = [&](const char* name, auto&& launch) {
    CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev, 20, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double worst = 0;
    for (int i = 0; i < NCHECK; ++i)
      worst = std::max(worst, bench::max_rel_err_scalar(hC[probe[i]], (float)want[i]));
    r.err = worst;
    r.flops = flops;
    r.bytes = bytes;
    rows.push_back(r);
  };

  std::printf("problem   : %d x %d x %d, arithmetic intensity %.1f FLOP/byte if fully reused\n",
              M, N, K, flops / bytes);
  bench::header(dev);

  dim3 nb(TILE, TILE);
  dim3 gx((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  run("1 naive, uncoalesced", [&] {
    KERNEL_LAUNCH(sgemm_naive_uncoalesced, dim3((M + TILE - 1) / TILE, (N + TILE - 1) / TILE),
                  nb, 0, dA, dB, dC, M, N, K);
  });
  run("2 naive, coalesced", [&] {
    KERNEL_LAUNCH(sgemm_naive_coalesced, gx, nb, 0, dA, dB, dC, M, N, K);
  });
  run("3 shared-memory tiling", [&] {
    KERNEL_LAUNCH(sgemm_tiled, gx, nb, 0, dA, dB, dC, M, N, K);
  });
  run("4 + register blocking 4x4", [&] {
    KERNEL_LAUNCH(sgemm_blocked, dim3(N / BN, M / BM), dim3(NTHREADS), 0, dA, dB, dC, M, N, K);
  });

  // fp32 accumulation over K terms with mixed signs: error grows like sqrt(K)*eps, and the
  // sampled entries are O(sqrt(K)) so there is real cancellation in the reference too.
  const double tol = 1e-4;
  bench::rows_out(rows, dev, tol);

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return bench::verdict(rows, tol);
}
