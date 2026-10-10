// 12_moe_dispatch.cu — fine-grained Mixture-of-Experts: routing, dispatch, and why MoE decode
// is *more* memory-bound than the dense model it replaced.
//
//     nvcc -O3 -arch=native 12_moe_dispatch.cu -o build/12 && build/12
//     make check
//
// A dense FFN applies one weight matrix to every token. An MoE layer has E experts, routes
// each token to its top-k, and applies only those. The headline is that a model with 256
// experts and k=8 has 32x the parameters for the same arithmetic per token — and the headline
// is true. The part that gets left out is what it does to *memory traffic*, which is what
// actually sets decode speed.
//
// The accounting, per token, at decode time:
//
//     dense:  read the one FFN weight matrix          -> d_model x d_ff x bytes
//     MoE:    read k different expert matrices        -> k x d_model x d_ff_expert x bytes
//
// With DeepSeek-V3's shape (d_ff_expert = d_ff/8, k = 8) those are equal per token — the
// arithmetic is unchanged and so is the traffic, for a single token. But a *batch* of tokens
// in a dense model shares one weight matrix, read once and reused across the whole batch. In
// an MoE, a batch of B tokens scatters across up to min(B·k, E) experts, and each one that
// receives even a single token costs a full weight-matrix read.
//
// So MoE trades parameter count against batch efficiency. At batch 1 it is a wash; at batch
// 64 a dense layer reads its weights once and an MoE layer reads most of the model. That is
// why MoE serving needs expert parallelism and large batches, and why an MoE that is cheap to
// train can be expensive to serve.
//
// The three variants below are the three ways to implement the dispatch, and the difference
// between the first and the others is a factor of E/k in arithmetic.
//
// Prerequisite: 05_dequant_gemv.cu, for the memory-bound GEMV argument this builds on.
#include "common.cuh"

#if SHIM_BUILD
constexpr int EXPERTS = 8;
constexpr int TOPK = 4;   // >2: see the note on commutativity in moe_gather_atomic
constexpr int DM = 32;        // d_model
constexpr int DFF = 32;       // per-expert hidden
constexpr int BLOCK = 64;
#else
constexpr int EXPERTS = 64;
constexpr int TOPK = 8;
constexpr int DM = 512;
constexpr int DFF = 256;
constexpr int BLOCK = 256;
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
// The router. Scores every token against every expert, takes the top-k, and softmaxes the
// selected scores into gate weights.
//
// Softmaxing *after* selection rather than before is the detail that matters: the gates then
// sum to 1 over the chosen experts, so the layer's output magnitude does not depend on how
// confident the router was. Softmax-then-select gives you gates summing to something less
// than 1 that varies per token, and a model that trains around it in ways nobody wants.
//
// The top-k here is a simple selection sort over E, which is right for E in the tens and
// wrong for E in the thousands. Nothing subtle happens in it, which is the point — the
// interesting behaviour is in what the dispatch does with the result.
// ---------------------------------------------------------------------------------------
__global__ void route(const float* __restrict__ x,        // [T][DM]
                      const float* __restrict__ Wg,       // [E][DM]
                      int* __restrict__ topk_idx,         // [T][TOPK]
                      float* __restrict__ topk_gate,      // [T][TOPK]
                      int* __restrict__ expert_count,     // [E]
                      int T, int E) {
  SHARED(float, scratch, 32);
  SHARED(float, logits, EXPERTS);
  SHARED(int, chosen, TOPK);
  SHARED(float, gates, TOPK);

  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    const float* xt = x + (size_t)t * DM;
    for (int e = 0; e < E; ++e) {
      float acc = 0.0f;
      for (int i = threadIdx.x; i < DM; i += blockDim.x) acc += xt[i] * Wg[(size_t)e * DM + i];
      acc = block_sum(acc, scratch);
      if (threadIdx.x == 0) logits[e] = acc;
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      float work[EXPERTS];
      for (int e = 0; e < E; ++e) work[e] = logits[e];
      for (int k = 0; k < TOPK; ++k) {
        int best = 0;
        for (int e = 1; e < E; ++e)
          if (work[e] > work[best]) best = e;
        chosen[k] = best;
        gates[k] = work[best];
        work[best] = -1e30f;                  // take it out of contention
      }
      // Softmax over the selected logits only — so the gates sum to 1 whatever the router's
      // absolute confidence was.
      float mx = gates[0];
      for (int k = 1; k < TOPK; ++k) mx = fmaxf(mx, gates[k]);
      float sum = 0.0f;
      for (int k = 0; k < TOPK; ++k) { gates[k] = __expf(gates[k] - mx); sum += gates[k]; }
      for (int k = 0; k < TOPK; ++k) {
        topk_gate[(size_t)t * TOPK + k] = gates[k] / sum;
        topk_idx[(size_t)t * TOPK + k] = chosen[k];
        atomicAdd(&expert_count[chosen[k]], 1);   // integer: exact in any order
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------
// Variant 1: dense — run every token through every expert and mask.
//
// This is what you get from a naive `torch.einsum` implementation, and it is the reason people
// report that "MoE is slower than dense". It does E/k times the necessary arithmetic — 8x at
// DeepSeek's k=8 out of 64, 32x at 256 — and throws away all but k of the results.
//
// It exists here because it is the unambiguous reference: no gather, no permutation, nothing
// to get wrong. Variants 2 and 3 must reproduce it exactly.
// ---------------------------------------------------------------------------------------
__global__ void moe_dense(const float* __restrict__ x, const float* __restrict__ W1,
                          const float* __restrict__ W2, const int* __restrict__ topk_idx,
                          const float* __restrict__ topk_gate, float* __restrict__ out,
                          int T, int E) {
  SHARED(float, scratch, 32);
  SHARED(float, hid, DFF);

  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    const float* xt = x + (size_t)t * DM;
    for (int i = threadIdx.x; i < DM; i += blockDim.x) out[(size_t)t * DM + i] = 0.0f;
    __syncthreads();

    for (int e = 0; e < E; ++e) {
      // Was this expert selected, and with what gate? Computed rather than looked up, so the
      // control flow is identical for every expert — which is the whole cost of this variant.
      float gate = 0.0f;
      for (int k = 0; k < TOPK; ++k)
        if (topk_idx[(size_t)t * TOPK + k] == e) gate = topk_gate[(size_t)t * TOPK + k];

      for (int j = threadIdx.x; j < DFF; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < DM; ++i) acc += xt[i] * W1[((size_t)e * DFF + j) * DM + i];
        hid[j] = acc > 0.0f ? acc : 0.0f;              // ReLU
      }
      __syncthreads();

      for (int i = threadIdx.x; i < DM; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < DFF; ++j) acc += hid[j] * W2[((size_t)e * DM + i) * DFF + j];
        out[(size_t)t * DM + i] += gate * acc;         // gate is 0 for E-k of the experts
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------------------
// Variant 2: gather-scatter — each token visits only its k experts.
//
// One block per (token, slot) pair, so the grid is T x TOPK and each block does exactly the
// work its token needs. E/k times less arithmetic than variant 1, for one indirection.
//
// The accumulation into `out` is where it gets interesting: k blocks write to the same token's
// output. Doing that with atomicAdd is the obvious implementation and makes the layer
// order-dependent — the same reproducibility hazard as the dW reduction in
// 07_rmsnorm_backward.cu, in a place nobody thinks to look for it. This variant does exactly
// that, deliberately, and kernelbench requires it to show up as order-dependent.
// ---------------------------------------------------------------------------------------
__global__ void moe_gather_atomic(const float* __restrict__ x, const float* __restrict__ W1,
                                  const float* __restrict__ W2, const int* __restrict__ topk_idx,
                                  const float* __restrict__ topk_gate, float* __restrict__ out,
                                  int T) {
  SHARED(float, hid, DFF);
  const int t = blockIdx.x, k = blockIdx.y;
  if (t >= T) return;

  const int e = topk_idx[(size_t)t * TOPK + k];
  const float gate = topk_gate[(size_t)t * TOPK + k];
  const float* xt = x + (size_t)t * DM;

  for (int j = threadIdx.x; j < DFF; j += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < DM; ++i) acc += xt[i] * W1[((size_t)e * DFF + j) * DM + i];
    hid[j] = acc > 0.0f ? acc : 0.0f;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < DM; i += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < DFF; ++j) acc += hid[j] * W2[((size_t)e * DM + i) * DFF + j];
    atomicAdd(&out[(size_t)t * DM + i], gate * acc);   // k blocks race for this address
  }
}

// ---------------------------------------------------------------------------------------
// Variant 3: the same work, combined deterministically.
//
// Each (token, slot) block writes to its *own* slot in a [T][TOPK][DM] buffer, and a second
// kernel sums the k slots in index order. Same arithmetic, same traffic plus T·k·DM floats of
// scratch, and a bit-identical answer on every run.
//
// This is also the shape that a production MoE kernel wants for a different reason: with the
// contributions laid out per slot, the combine step is a clean elementwise reduction that
// fuses with whatever comes next, instead of a scattered set of atomics into memory that the
// next kernel then has to re-read.
//
// A detail worth knowing, because it decides whether your determinism test can detect
// anything: floating-point addition **is** commutative — a+b and b+a are bit-identical. It is
// *associativity* that fails, so an order-dependent result needs at least three addends. A
// top-2 MoE combined with atomics is therefore reproducible by accident, and a test written
// against a top-2 configuration will report that atomics are safe. They are not; the same
// kernel at top-8 is not reproducible. This file uses k=4 even in its toy configuration for
// exactly that reason.
// ---------------------------------------------------------------------------------------
__global__ void moe_gather_slots(const float* __restrict__ x, const float* __restrict__ W1,
                                 const float* __restrict__ W2, const int* __restrict__ topk_idx,
                                 const float* __restrict__ topk_gate,
                                 float* __restrict__ slots, int T) {
  SHARED(float, hid, DFF);
  const int t = blockIdx.x, k = blockIdx.y;
  if (t >= T) return;

  const int e = topk_idx[(size_t)t * TOPK + k];
  const float gate = topk_gate[(size_t)t * TOPK + k];
  const float* xt = x + (size_t)t * DM;

  for (int j = threadIdx.x; j < DFF; j += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < DM; ++i) acc += xt[i] * W1[((size_t)e * DFF + j) * DM + i];
    hid[j] = acc > 0.0f ? acc : 0.0f;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < DM; i += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < DFF; ++j) acc += hid[j] * W2[((size_t)e * DM + i) * DFF + j];
    slots[(((size_t)t * TOPK) + k) * DM + i] = gate * acc;
  }
}

__global__ void moe_combine(const float* __restrict__ slots, float* __restrict__ out, int T) {
  const size_t total = (size_t)T * DM;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const size_t t = idx / DM, i = idx % DM;
    float acc = 0.0f;
    for (int k = 0; k < TOPK; ++k) acc += slots[((t * TOPK) + k) * DM + i];  // fixed order
    out[idx] = acc;
  }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  int T = 16;
#else
  int T = argc > 1 ? std::atoi(argv[1]) : 512;    // tokens in the batch
#endif
  (void)argc; (void)argv;
  const int E = EXPERTS;

  std::vector<float> hx((size_t)T * DM), hWg((size_t)E * DM), hW1((size_t)E * DFF * DM),
      hW2((size_t)E * DM * DFF), hout((size_t)T * DM);
  bench::fill(hx.data(), hx.size(), 1);
  bench::fill(hWg.data(), hWg.size(), 2);
  bench::fill(hW1.data(), hW1.size(), 3);
  bench::fill(hW2.data(), hW2.size(), 4);
  const float s1 = 1.0f / std::sqrt((float)DM), s2 = 1.0f / std::sqrt((float)DFF);
  for (auto& v : hW1) v *= s1;
  for (auto& v : hW2) v *= s2;

  // Routing on the host, so the reference and every variant see the same assignment. A router
  // that disagreed between reference and kernel would make every variant "wrong" for a reason
  // that has nothing to do with the dispatch.
  std::vector<int> hidx((size_t)T * TOPK);
  std::vector<float> hgate((size_t)T * TOPK);
  std::vector<int> hcount(E, 0);
  for (int t = 0; t < T; ++t) {
    std::vector<double> logit(E);
    for (int e = 0; e < E; ++e) {
      double acc = 0.0;
      for (int i = 0; i < DM; ++i) acc += (double)hx[(size_t)t * DM + i] * hWg[(size_t)e * DM + i];
      logit[e] = acc;
    }
    std::vector<double> work = logit;
    double mx = -1e300, sum = 0.0;
    std::vector<double> sel(TOPK);
    for (int k = 0; k < TOPK; ++k) {
      int best = 0;
      for (int e = 1; e < E; ++e)
        if (work[e] > work[best]) best = e;
      hidx[(size_t)t * TOPK + k] = best;
      sel[k] = work[best];
      work[best] = -1e30;
      mx = std::max(mx, sel[k]);
      hcount[best]++;
    }
    for (int k = 0; k < TOPK; ++k) sum += std::exp(sel[k] - mx);
    for (int k = 0; k < TOPK; ++k) hgate[(size_t)t * TOPK + k] = (float)(std::exp(sel[k] - mx) / sum);
  }

  // Reference output in double.
  std::vector<float> want((size_t)T * DM);
  for (int t = 0; t < T; ++t) {
    std::vector<double> acc(DM, 0.0);
    for (int k = 0; k < TOPK; ++k) {
      const int e = hidx[(size_t)t * TOPK + k];
      const double g = hgate[(size_t)t * TOPK + k];
      std::vector<double> hid(DFF);
      for (int j = 0; j < DFF; ++j) {
        double a = 0.0;
        for (int i = 0; i < DM; ++i)
          a += (double)hx[(size_t)t * DM + i] * hW1[((size_t)e * DFF + j) * DM + i];
        hid[j] = a > 0.0 ? a : 0.0;
      }
      for (int i = 0; i < DM; ++i) {
        double a = 0.0;
        for (int j = 0; j < DFF; ++j) a += hid[j] * hW2[((size_t)e * DM + i) * DFF + j];
        acc[i] += g * a;
      }
    }
    for (int i = 0; i < DM; ++i) want[(size_t)t * DM + i] = (float)acc[i];
  }

  float *dx, *dWg, *dW1, *dW2, *dout, *dgate, *dslots;
  int *didx, *dcount;
  CUDA_CHECK(cudaMalloc((void**)&dx, hx.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dWg, hWg.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dW1, hW1.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dW2, hW2.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dout, hout.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dgate, hgate.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dslots, (size_t)T * TOPK * DM * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&didx, hidx.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dcount, (size_t)E * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWg, hWg.data(), hWg.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW1, hW1.data(), hW1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW2, hW2.data(), hW2.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(didx, hidx.data(), hidx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dgate, hgate.data(), hgate.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Check the on-device router against the host one before anything else — a dispatch that is
  // perfect but routes differently produces a plausible, wrong model. This gates the exit
  // code: every variant below is fed the *host* routing, so a broken device router would
  // otherwise pass silently.
  bool router_ok = true;
  CUDA_CHECK(cudaMemset(dcount, 0, (size_t)E * sizeof(int)));
  {
    std::vector<int> gidx(hidx.size());
    std::vector<float> ggate(hgate.size());
    int *tmpi;
    float *tmpg;
    CUDA_CHECK(cudaMalloc((void**)&tmpi, hidx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&tmpg, hgate.size() * sizeof(float)));
    KERNEL_LAUNCH(route, dim3(64), dim3(BLOCK), 0, dx, dWg, tmpi, tmpg, dcount, T, E);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(gidx.data(), tmpi, gidx.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ggate.data(), tmpg, ggate.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int mism = 0;
    for (size_t i = 0; i < gidx.size(); ++i) mism += (gidx[i] != hidx[i]);
    std::printf("router    : %d/%zu assignments match the host reference%s\n",
                (int)gidx.size() - mism, gidx.size(), mism ? "   <- ROUTER WRONG" : "");
    router_ok = (mism == 0);
    CUDA_CHECK(cudaFree(tmpi));
    CUDA_CHECK(cudaFree(tmpg));
  }

  std::vector<bench::Row> rows;
  const double expert_bytes = 2.0 * DM * DFF * sizeof(float);   // W1 + W2 for one expert

  auto run = [&](const char* name, double traffic, double flops, const char* note,
                 auto&& launch) {
    CUDA_CHECK(cudaMemset(dout, 0, hout.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(dslots, 0, (size_t)T * TOPK * DM * sizeof(float)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel([&] {
      CUDA_CHECK(cudaMemset(dout, 0, hout.size() * sizeof(float)));
      launch();
    }, dev, 20, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hout.data(), dout, hout.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    r.err = bench::max_rel_err(hout.data(), want.data(), hout.size());
    r.checksum = bench::checksum_of(hout);
    r.bytes = traffic;
    r.flops = flops;
    r.note = note;
    rows.push_back(r);
  };

  // How many distinct experts the batch actually touched — the number that decides traffic.
  int touched = 0;
  for (int e = 0; e < E; ++e) touched += (hcount[e] > 0);

  std::printf("problem   : %d tokens, %d experts, top-%d, d_model %d, d_ff %d\n", T, E, TOPK,
              DM, DFF);
  std::printf("routing   : %d of %d experts received at least one token\n", touched, E);
  bench::header(dev);

  run("1 dense (every expert)", (double)E * expert_bytes, 4.0 * T * E * DM * DFF,
      "runs all E experts",
      [&] {
        KERNEL_LAUNCH(moe_dense, dim3(64), dim3(BLOCK), 0, dx, dW1, dW2, didx, dgate, dout, T, E);
      });
  run("2 gather + atomic combine", (double)touched * expert_bytes, 4.0 * T * TOPK * DM * DFF,
      "not reproducible", [&] {
        KERNEL_LAUNCH(moe_gather_atomic, dim3(T, TOPK), dim3(BLOCK), 0, dx, dW1, dW2, didx,
                      dgate, dout, T);
      });
  run("3 gather + slot combine", (double)touched * expert_bytes + 2.0 * T * TOPK * DM * 4,
      4.0 * T * TOPK * DM * DFF, "deterministic", [&] {
        KERNEL_LAUNCH(moe_gather_slots, dim3(T, TOPK), dim3(BLOCK), 0, dx, dW1, dW2, didx,
                      dgate, dslots, T);
        KERNEL_LAUNCH(moe_combine, dim3(64), dim3(BLOCK), 0, dslots, dout, T);
      });

  const double tol = 2e-4;
  bench::rows_out(rows, dev, tol);

  // ---- the part that decides whether you can serve it ----------------------------------
  std::printf("\nExpert weights touched, as the batch grows (E=%d, k=%d):\n\n", E, TOPK);
  std::printf("  %8s %12s %14s %16s\n", "batch", "experts hit", "weights read", "vs dense");
  for (int b : {1, 4, 16, 64, 256, 1024}) {
    // Expected distinct experts hit by b tokens each choosing k of E, uniformly:
    //   E * (1 - (1 - k/E)^b)
    const double hit = E * (1.0 - std::pow(1.0 - (double)TOPK / E, (double)b));
    const double moe_bytes = hit * expert_bytes;
    const double dense_bytes = 8.0 * DM * DFF * sizeof(float);   // dense d_ff = 8x an expert's
    std::printf("  %8d %12.1f %11.2f MB %14.2fx\n", b, hit, moe_bytes / 1048576.0,
                moe_bytes / dense_bytes);
  }
  std::printf("\n  A dense layer reads its weights once however large the batch is. An MoE\n"
              "  layer reads every expert any token in the batch selected — so its weight\n"
              "  traffic *grows with batch size* until every expert is hit, and only then\n"
              "  flattens. That is the opposite of the property batching exists to exploit.\n"
              "\n  Which is why MoE serving looks the way it does: expert parallelism to spread\n"
              "  the weights across devices so no single GPU reads them all, large batches so\n"
              "  the reads amortize over more tokens, and a shared expert that every token\n"
              "  visits so at least some of the traffic is guaranteed reused.\n");

  for (void* p : {(void*)dx, (void*)dWg, (void*)dW1, (void*)dW2, (void*)dout, (void*)dgate,
                  (void*)dslots, (void*)didx, (void*)dcount})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev) || !router_ok;
}
