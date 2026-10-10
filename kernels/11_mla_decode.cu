// 11_mla_decode.cu — Multi-head Latent Attention: the KV cache as a compressed latent, and
// the "absorption" trick that makes it fast instead of merely small.
//
//     nvcc -O3 -arch=native 11_mla_decode.cu -o build/11 && build/11
//     make check
//
// Every KV-cache reduction before this one threw information away. GQA shares one K/V head
// across several query heads; sliding windows forget old tokens; quantization rounds. MLA
// (DeepSeek-V2/V3) does something different: it stores a **low-rank projection** of K and V
// and reconstructs them on the fly.
//
//     cached per token:   c_t  = W_DKV · h_t          [d_c]     e.g. 512
//                         k_r  = RoPE(W_KR · h_t)     [d_r]     e.g. 64
//     reconstructed:      K_i  = W_UK^i · c_t         [d_h] per head
//                         V_i  = W_UV^i · c_t         [d_h] per head
//
// The cache holds d_c + d_r floats per token per layer, independent of head count. Against
// MHA's 2·n_h·d_h that is a large factor — the table this program prints works it out — and
// unlike GQA it is not a reduction in what the model can express, because each head still gets
// its own K and V, derived through its own projection.
//
// So far this is a memory result. The performance result is the interesting part.
//
// The absorption trick
// --------------------
// The naive way to use this cache is to decompress: build K_i and V_i for every head and every
// cached token, then run ordinary attention. That reconstructs n_h · d_h floats per token —
// *more* data than MHA would have cached — and hands back every byte MLA saved, plus a large
// GEMM per decode step.
//
// The fix is to never form K_i at all. The score is
//
//     q_i · K_i = q_i · (W_UK^i · c_t) = (W_UK^i)ᵀ · q_i · c_t = q'_i · c_t
//
// so projecting the *query* once, at the start of the step, lets the attention run directly
// against the cached latent. q'_i is d_c wide instead of d_h, and it is computed once per
// decode step rather than once per cached token. The same identity absorbs W_UV into the
// output projection on the other side.
//
// That is the whole idea, and it is checkable: variants 1 and 2 below must produce the same
// numbers to fp32 rounding, and this file fails if they do not.
//
// RoPE is why there is a second, uncompressed piece. Rotary embeddings mix position into K in
// a way that does not commute with the up-projection — you cannot absorb W_UK through a
// position-dependent rotation. DeepSeek's answer is to split the head: a compressed part with
// no positional encoding, and a small decoupled part (d_r = 64) that carries RoPE and is
// cached directly. The score is the sum of the two. Every MLA implementation has this seam in
// it, and it is the part people get wrong.
//
// Prerequisite: 06_flash_decode.cu, whose online-softmax structure this reuses unchanged.
#include "common.cuh"

#if SHIM_BUILD
constexpr int HEADS = 4;
constexpr int DH = 32;      // per-head dim
constexpr int DC = 64;      // compressed latent dim
constexpr int DR = 16;      // decoupled RoPE dim
constexpr int TILE = 32;
constexpr int BLOCK = 64;
#else
constexpr int HEADS = 32;
constexpr int DH = 128;
constexpr int DC = 512;
constexpr int DR = 64;
constexpr int TILE = 16;   // TILE x DC floats of shared memory: 16 x 512 x 4 = 32 KB
constexpr int BLOCK = 128;
#endif
constexpr float NEG = -1e30f;

// ---------------------------------------------------------------------------------------
// Variant 1: decompress, then run ordinary attention.
//
// For every cached token, reconstruct this head's K and V from the latent, then do what
// 06_flash_decode.cu does. Correct, simple, and it defeats the point: the inner loop now reads
// the latent *and* does a d_c x d_h matrix-vector per token per head, so the arithmetic per
// cached token went from ~2·d_h to ~2·d_c·d_h — a factor of d_c. On the decode path, where
// there is one query and nothing to amortize over, that is the whole cost.
//
// This is not a strawman. It is what you get if you implement MLA by writing an adapter that
// materializes K and V and hands them to an existing attention kernel, which is the obvious
// first implementation and the one that makes people conclude MLA is slow.
// ---------------------------------------------------------------------------------------
__global__ void mla_decompress(const float* __restrict__ q_nope,   // [H][DH]
                               const float* __restrict__ q_rope,   // [H][DR]
                               const float* __restrict__ kv_c,     // [S][DC]  the cache
                               const float* __restrict__ k_r,      // [S][DR]  the cache
                               const float* __restrict__ W_UK,     // [H][DH][DC]
                               const float* __restrict__ W_UV,     // [H][DH][DC]
                               float* __restrict__ out,            // [H][DH]
                               int S) {
  SHARED(float, sq, DH);
  SHARED(float, sqr, DR);
  SHARED(float, sc, TILE * DC);
  SHARED(float, sp, TILE);

  const int h = blockIdx.x, d = threadIdx.x;
  const float scale = rsqrtf((float)(DH + DR));

  for (int i = d; i < DH; i += blockDim.x) sq[i] = q_nope[(size_t)h * DH + i];
  for (int i = d; i < DR; i += blockDim.x) sqr[i] = q_rope[(size_t)h * DR + i];
  __syncthreads();

  float m = NEG, l = 0.0f;
  // Each thread owns output dims d, d+blockDim, ... — ceil(DH/BLOCK) of them. Sizing this
  // array to DH instead would burn 128 registers per thread and spill the lot to local
  // memory, which is HBM wearing a different name.
  constexpr int OWN_DH = DH / BLOCK + 1;
  float acc[OWN_DH];
  int nown = 0;
  for (int i = d; i < DH; i += blockDim.x) acc[nown++] = 0.0f;

  for (int j0 = 0; j0 < S; j0 += TILE) {
    const int n = (S - j0 < TILE) ? (S - j0) : TILE;
    for (int r = 0; r < n; ++r)
      for (int i = d; i < DC; i += blockDim.x) sc[r * DC + i] = kv_c[(size_t)(j0 + r) * DC + i];
    __syncthreads();

    if (d < n) {
      // Reconstruct K for this head and this token: DH x DC multiply-adds, per token.
      float s = 0.0f;
      for (int a = 0; a < DH; ++a) {
        float ka = 0.0f;
        for (int b = 0; b < DC; ++b) ka += W_UK[((size_t)h * DH + a) * DC + b] * sc[d * DC + b];
        s += sq[a] * ka;
      }
      // ...plus the decoupled RoPE part, which is cached directly and never compressed.
      for (int a = 0; a < DR; ++a) s += sqr[a] * k_r[(size_t)(j0 + d) * DR + a];
      sp[d] = s * scale;
    }
    __syncthreads();

    float mt = NEG;
    for (int i = 0; i < n; ++i) mt = fmaxf(mt, sp[i]);
    const float mnew = fmaxf(m, mt);
    const float corr = __expf(m - mnew);

    float lt = 0.0f;
    nown = 0;
    for (int i = d; i < DH; i += blockDim.x) acc[nown++] *= corr;
    for (int i = 0; i < n; ++i) {
      const float p = __expf(sp[i] - mnew);
      lt += p;
      nown = 0;
      for (int a = d; a < DH; a += blockDim.x) {
        // Reconstruct V, one output dim at a time.
        float va = 0.0f;
        for (int b = 0; b < DC; ++b) va += W_UV[((size_t)h * DH + a) * DC + b] * sc[i * DC + b];
        acc[nown++] += p * va;
      }
    }
    l = l * corr + lt;
    m = mnew;
    __syncthreads();
  }

  nown = 0;
  for (int i = d; i < DH; i += blockDim.x) out[(size_t)h * DH + i] = acc[nown++] / l;
}

// ---------------------------------------------------------------------------------------
// Variant 2: absorbed. Project the query once, attend in latent space, project the output once.
//
//   q'_i = (W_UK^i)ᵀ q_i        [DC]   — once per decode step, not once per cached token
//   score = q'_i · c_t + q_r · k_r
//   acc   = Σ p_t · c_t         [DC]   — accumulate in latent space
//   out_i = W_UV^i · acc        [DH]   — once per decode step
//
// The inner loop now touches only the cached latent, with 2·d_c FLOPs per token instead of
// 2·d_c·d_h. The two projections are d_h x d_c each, done once — negligible against S = 4096
// cached tokens.
//
// The output accumulator is d_c wide rather than d_h. At d_c = 512 that is 512 floats per
// head in registers or shared memory, which is the real implementation constraint and the
// reason production MLA kernels tile the latent dimension.
// ---------------------------------------------------------------------------------------
__global__ void mla_project_q(const float* __restrict__ q_nope,   // [H][DH]
                              const float* __restrict__ W_UK,     // [H][DH][DC]
                              float* __restrict__ q_latent,       // [H][DC]
                              int H) {
  const int h = blockIdx.x;
  for (int b = threadIdx.x; b < DC; b += blockDim.x) {
    float acc = 0.0f;
    for (int a = 0; a < DH; ++a) acc += q_nope[(size_t)h * DH + a] * W_UK[((size_t)h * DH + a) * DC + b];
    q_latent[(size_t)h * DC + b] = acc;
  }
}

__global__ void mla_absorbed(const float* __restrict__ q_latent,  // [H][DC]
                             const float* __restrict__ q_rope,    // [H][DR]
                             const float* __restrict__ kv_c,      // [S][DC]
                             const float* __restrict__ k_r,       // [S][DR]
                             float* __restrict__ acc_latent,      // [H][DC]
                             int S) {
  SHARED(float, sq, DC);
  SHARED(float, sqr, DR);
  SHARED(float, sc, TILE * DC);
  SHARED(float, sp, TILE);

  const int h = blockIdx.x, d = threadIdx.x;
  const float scale = rsqrtf((float)(DH + DR));

  for (int i = d; i < DC; i += blockDim.x) sq[i] = q_latent[(size_t)h * DC + i];
  for (int i = d; i < DR; i += blockDim.x) sqr[i] = q_rope[(size_t)h * DR + i];
  __syncthreads();

  constexpr int OWN = DC / BLOCK + 1;
  float acc[OWN];
  int nown = 0;
  for (int i = d; i < DC; i += blockDim.x) acc[nown++] = 0.0f;
  float m = NEG, l = 0.0f;

  for (int j0 = 0; j0 < S; j0 += TILE) {
    const int n = (S - j0 < TILE) ? (S - j0) : TILE;
    for (int r = 0; r < n; ++r)
      for (int i = d; i < DC; i += blockDim.x) sc[r * DC + i] = kv_c[(size_t)(j0 + r) * DC + i];
    __syncthreads();

    if (d < n) {
      float s = 0.0f;
      for (int b = 0; b < DC; ++b) s += sq[b] * sc[d * DC + b];        // latent-space score
      for (int a = 0; a < DR; ++a) s += sqr[a] * k_r[(size_t)(j0 + d) * DR + a];  // RoPE part
      sp[d] = s * scale;
    }
    __syncthreads();

    float mt = NEG;
    for (int i = 0; i < n; ++i) mt = fmaxf(mt, sp[i]);
    const float mnew = fmaxf(m, mt);
    const float corr = __expf(m - mnew);

    nown = 0;
    for (int i = d; i < DC; i += blockDim.x) acc[nown++] *= corr;
    float lt = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float p = __expf(sp[i] - mnew);
      lt += p;
      nown = 0;
      for (int b = d; b < DC; b += blockDim.x) acc[nown++] += p * sc[i * DC + b];
    }
    l = l * corr + lt;
    m = mnew;
    __syncthreads();
  }

  nown = 0;
  for (int b = d; b < DC; b += blockDim.x) acc_latent[(size_t)h * DC + b] = acc[nown++] / l;
}

// The other half of the absorption: W_UV applied once, to the accumulated latent.
__global__ void mla_project_out(const float* __restrict__ acc_latent,  // [H][DC]
                                const float* __restrict__ W_UV,        // [H][DH][DC]
                                float* __restrict__ out) {             // [H][DH]
  const int h = blockIdx.x;
  for (int a = threadIdx.x; a < DH; a += blockDim.x) {
    float acc = 0.0f;
    for (int b = 0; b < DC; ++b)
      acc += W_UV[((size_t)h * DH + a) * DC + b] * acc_latent[(size_t)h * DC + b];
    out[(size_t)h * DH + a] = acc;
  }
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  int S = 128;
#else
  int S = argc > 1 ? std::atoi(argv[1]) : 4096;
#endif
  (void)argc; (void)argv;
  S = (S / TILE) * TILE;
  const int H = HEADS;

  std::vector<float> hq((size_t)H * DH), hqr((size_t)H * DR), hc((size_t)S * DC),
      hkr((size_t)S * DR), hwuk((size_t)H * DH * DC), hwuv((size_t)H * DH * DC),
      hout((size_t)H * DH);
  bench::fill(hq.data(), hq.size(), 1);
  bench::fill(hqr.data(), hqr.size(), 2);
  bench::fill(hc.data(), hc.size(), 3);
  bench::fill(hkr.data(), hkr.size(), 4);
  bench::fill(hwuk.data(), hwuk.size(), 5);
  bench::fill(hwuv.data(), hwuv.size(), 6);
  // Projection weights are 1/sqrt(fan_in) scaled, or the reconstructed K blows up and the
  // softmax saturates — which would make every variant agree trivially on a one-hot answer.
  const float w_scale = 1.0f / std::sqrt((float)DC);
  for (auto& v : hwuk) v *= w_scale;
  for (auto& v : hwuv) v *= w_scale;

  // Reference in double, by the *definition* — decompress and do plain attention. The
  // absorbed kernel is an algebraic rearrangement of exactly this, so both must match it.
  std::vector<float> want((size_t)H * DH);
  {
    const double scale = 1.0 / std::sqrt((double)(DH + DR));
    std::vector<double> sc(S);
    for (int h = 0; h < H; ++h) {
      double mx = -1e300;
      for (int t = 0; t < S; ++t) {
        double s = 0.0;
        for (int a = 0; a < DH; ++a) {
          double ka = 0.0;
          for (int b = 0; b < DC; ++b)
            ka += (double)hwuk[((size_t)h * DH + a) * DC + b] * hc[(size_t)t * DC + b];
          s += (double)hq[(size_t)h * DH + a] * ka;
        }
        for (int a = 0; a < DR; ++a)
          s += (double)hqr[(size_t)h * DR + a] * hkr[(size_t)t * DR + a];
        sc[t] = s * scale;
        mx = std::max(mx, sc[t]);
      }
      double l = 0.0;
      for (int t = 0; t < S; ++t) { sc[t] = std::exp(sc[t] - mx); l += sc[t]; }
      for (int a = 0; a < DH; ++a) {
        double o = 0.0;
        for (int t = 0; t < S; ++t) {
          double va = 0.0;
          for (int b = 0; b < DC; ++b)
            va += (double)hwuv[((size_t)h * DH + a) * DC + b] * hc[(size_t)t * DC + b];
          o += sc[t] * va;
        }
        want[(size_t)h * DH + a] = (float)(o / l);
      }
    }
  }

  float *dq, *dqr, *dc, *dkr, *dwuk, *dwuv, *dout, *dqlat, *dacc;
  CUDA_CHECK(cudaMalloc((void**)&dq, hq.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dqr, hqr.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dc, hc.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dkr, hkr.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dwuk, hwuk.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dwuv, hwuv.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dout, hout.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dqlat, (size_t)H * DC * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dacc, (size_t)H * DC * sizeof(float)));
  auto up = [&](float* d, const std::vector<float>& h) {
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
  };
  up(dq, hq); up(dqr, hqr); up(dc, hc); up(dkr, hkr); up(dwuk, hwuk); up(dwuv, hwuv);

  // The cache is what both variants must read. Only variant 1 also streams the projection
  // weights through the inner loop.
  const double cache_bytes = (double)S * (DC + DR) * sizeof(float);
  const double weight_bytes = 2.0 * H * DH * DC * sizeof(float);
  std::vector<bench::Row> rows;

  auto run = [&](const char* name, double traffic, double flops, const char* note,
                 auto&& launch) {
    CUDA_CHECK(cudaMemset(dout, 0, hout.size() * sizeof(float)));
    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel(launch, dev, 20, 5);
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

  std::printf("problem   : %d heads, KV length %d, d_h=%d, d_c=%d, d_r=%d\n", H, S, DH, DC, DR);
  std::printf("cache     : MLA %.2f KB   vs MHA %.2f KB   (%.1fx smaller) for this layer\n",
              cache_bytes / 1024.0, 2.0 * S * H * DH * sizeof(float) / 1024.0,
              2.0 * H * DH / (double)(DC + DR));
  bench::header(dev);

  // Variant 1 reads the latent AND does a DH x DC reconstruction per token per head, twice.
  run("1 decompress, then attend", cache_bytes + weight_bytes,
      4.0 * H * S * DH * DC, "reconstructs K and V per token", [&] {
        KERNEL_LAUNCH(mla_decompress, dim3(H), dim3(BLOCK), 0, dq, dqr, dc, dkr, dwuk, dwuv,
                      dout, S);
      });
  // Variant 2 touches the latent only; the projections happen once, outside the S loop.
  run("2 absorbed (project q once)", cache_bytes + weight_bytes,
      4.0 * H * S * DC + 4.0 * H * DH * DC, "attends in latent space", [&] {
        KERNEL_LAUNCH(mla_project_q, dim3(H), dim3(BLOCK), 0, dq, dwuk, dqlat, H);
        KERNEL_LAUNCH(mla_absorbed, dim3(H), dim3(BLOCK), 0, dqlat, dqr, dc, dkr, dacc, S);
        KERNEL_LAUNCH(mla_project_out, dim3(H), dim3(BLOCK), 0, dacc, dwuv, dout);
      });

  const double tol = 2e-4;
  bench::rows_out(rows, dev, tol);

  std::printf("\nBoth variants compute the same function — the check above is against a\n"
              "decompress-and-attend reference in double, so the absorption identity is\n"
              "verified rather than asserted.\n");
  std::printf("\nArithmetic inside the S loop, per cached token per head:\n"
              "  decompressed : 4 x d_h x d_c = %d FLOPs\n"
              "  absorbed     : 4 x d_c       = %d FLOPs   (%.0fx less)\n",
              4 * DH * DC, 4 * DC, (double)DH);
  std::printf("\nKV cache per token per layer, fp16 — at DeepSeek-V3's configuration, not the\n"
              "toy sizes above:\n");
  struct Arch { const char* name; double bytes; };
  const Arch archs[] = {
      {"MHA  (32 heads x 128)",       2.0 * 32 * 128 * 2},
      {"GQA  (8 kv heads x 128)",     2.0 * 8 * 128 * 2},
      {"MQA  (1 kv head x 128)",      2.0 * 1 * 128 * 2},
      {"MLA  (d_c 512 + d_r 64)",     (512 + 64) * 2.0},
  };
  for (const Arch& a : archs)
    std::printf("  %-28s %7.0f B   %5.1fx MLA\n", a.name, a.bytes, a.bytes / ((512 + 64) * 2.0));
  std::printf("\n  Read that table carefully: MLA is NOT the smallest. MQA's single shared\n"
              "  K/V head is less than half MLA's cache. The point is what each one gives up\n"
              "  to get there — MQA collapses all 32 heads onto one K and one V, and pays for\n"
              "  it in quality; MLA keeps a distinct K and V per head, reconstructed through\n"
              "  that head's own projection, and still lands 3.6x below GQA.\n"
              "\n  So the honest comparison is MLA against GQA, which is the configuration it\n"
              "  actually displaced: a smaller cache and more expressivity, rather than the\n"
              "  usual trade of one for the other. And the absorption identity is what keeps\n"
              "  that from costing arithmetic — verified above against a decompress-and-attend\n"
              "  reference, so a broken absorption is a wrong answer and not a slower one.\n");

  for (void* p : {(void*)dq, (void*)dqr, (void*)dc, (void*)dkr, (void*)dwuk, (void*)dwuv,
                  (void*)dout, (void*)dqlat, (void*)dacc})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
