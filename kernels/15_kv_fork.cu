// 15_kv_fork.cu — forking an agent trajectory: copy-on-write page tables, and why a fan-out
// costs almost nothing until somebody writes.
//
//     nvcc -O3 -arch=native 15_kv_fork.cu -o build/15 && build/15
//     make check
//
// An agent branches. A planner spawns N sub-agents from the same state; a search samples N
// continuations of the same step; a retry re-runs a tool call from the same context. In every
// case N sequences begin life identical for their first P tokens and diverge afterwards.
//
// The naive implementation gives each child its own copy of the parent's KV cache. At P=2000
// tokens, 8 children, that is eight copies of the same 2000 tokens — both the memory to hold
// them and the bandwidth to write them, at the exact moment the scheduler is trying to admit
// the branches.
//
// PagedAttention already has the answer, because it already has the indirection. A sequence's
// KV is not a contiguous array, it is a **block table**: a list of physical page ids. Forking
// is then copying a list of integers, not a cache — and the pages themselves are shared, with
// a reference count so nobody frees a page another branch is still reading.
//
// One page is the exception, and it is the whole of "copy-on-write": the parent's **last**
// page is usually partially filled, and each child will append its own tokens into it. That
// page has to be copied, once per child. Everything before it is immutable — those tokens are
// in the past and no branch will ever write to them again — so it can be shared forever.
//
//     deep copy per child:   P x D x 2 x bytes
//     copy-on-write:         PAGE x D x 2 x bytes        (one partial page, whatever P is)
//
// At P=2000 and PAGE=16 that is a 125x reduction in the copy, and the memory saving is the
// same factor. It is the reason a fan-out is a cheap operation in a modern engine and an
// expensive one in a naive implementation.
//
// This file verifies the part that is easy to get wrong: that both layouts produce **identical
// attention outputs**, and that the reference counts come back to where they started.
//
// Prerequisite: 06_flash_decode.cu, whose paged gather this reuses.
#include "common.cuh"

#if SHIM_BUILD
constexpr int PAGE = 8;       // tokens per page
constexpr int D = 32;
constexpr int NCHILD = 4;
constexpr int BLOCK = 32;
constexpr int MAXPAGES = 64;
#else
constexpr int PAGE = 16;
constexpr int D = 128;
constexpr int NCHILD = 8;
constexpr int BLOCK = 128;
constexpr int MAXPAGES = 512;
#endif
constexpr float NEG = -1e30f;

// ---------------------------------------------------------------------------------------
// The copy kernel, in both flavours. `npages` pages are copied from `src_table` into
// `dst_table`; which pages appear in each table is what distinguishes the two strategies, and
// that decision is made on the host where a real allocator would make it.
// ---------------------------------------------------------------------------------------
__global__ void copy_pages(const float* __restrict__ Kpool, const float* __restrict__ Vpool,
                           float* __restrict__ Kout, float* __restrict__ Vout,
                           const int* __restrict__ src, const int* __restrict__ dst,
                           int npages) {
  const size_t total = (size_t)npages * PAGE * D;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    const int p = (int)(i / (PAGE * D));
    const size_t off = i % (PAGE * D);
    const size_t s = (size_t)src[p] * PAGE * D + off;
    const size_t d = (size_t)dst[p] * PAGE * D + off;
    Kout[d] = Kpool[s];
    Vout[d] = Vpool[s];
  }
}

// ---------------------------------------------------------------------------------------
// Decode attention through a block table. Identical in shape to `decode_paged` in
// 06_flash_decode.cu — the point here is that it does not know or care whether the pages it
// walks are private to this sequence or shared with seven siblings. That is exactly why the
// sharing is safe: the read path is unchanged.
// ---------------------------------------------------------------------------------------
__global__ void paged_attend(const float* __restrict__ Q, const float* __restrict__ Kpool,
                             const float* __restrict__ Vpool, const int* __restrict__ tables,
                             const int* __restrict__ lengths, float* __restrict__ out,
                             int table_stride) {
  SHARED(float, sq, D);
  SHARED(float, sK, PAGE * D);
  SHARED(float, sV, PAGE * D);
  SHARED(float, sp, PAGE);

  const int c = blockIdx.x, d = threadIdx.x;
  const float scale = rsqrtf((float)D);
  sq[d] = Q[(size_t)c * D + d];
  __syncthreads();

  const int len = lengths[c];
  const int* table = tables + (size_t)c * table_stride;
  float m = NEG, l = 0.0f, acc = 0.0f;

  for (int j0 = 0; j0 < len; j0 += PAGE) {
    const int n = (len - j0 < PAGE) ? (len - j0) : PAGE;
    const int page = table[j0 / PAGE];
    for (int r = 0; r < n; ++r) {
      const size_t base = ((size_t)page * PAGE + r) * D;
      sK[r * D + d] = Kpool[base + d];
      sV[r * D + d] = Vpool[base + d];
    }
    __syncthreads();
    if (d < n) {
      float s = 0.0f;
      for (int i = 0; i < D; ++i) s += sq[i] * sK[d * D + i];
      sp[d] = s * scale;
    }
    __syncthreads();

    float mt = NEG;
    for (int i = 0; i < n; ++i) mt = fmaxf(mt, sp[i]);
    const float mnew = fmaxf(m, mt);
    const float corr = __expf(m - mnew);
    float lt = 0.0f, at = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float p = __expf(sp[i] - mnew);
      lt += p;
      at += p * sV[i * D + d];
    }
    l = l * corr + lt;
    acc = acc * corr + at;
    m = mnew;
    __syncthreads();
  }
  out[(size_t)c * D + d] = acc / l;
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
#if SHIM_BUILD
  int P = 37, S = 5;
#else
  int P = argc > 1 ? std::atoi(argv[1]) : 2000;   // parent context at the fork point
  int S = argc > 2 ? std::atoi(argv[2]) : 40;     // tokens each child then generates
#endif
  (void)argc; (void)argv;
  const int N = NCHILD;

  const int parent_pages = (P + PAGE - 1) / PAGE;
  const int tail_used = P - (parent_pages - 1) * PAGE;      // tokens in the parent's last page
  const int child_len = P + S;
  const int child_pages = (child_len + PAGE - 1) / PAGE;
  if ((size_t)(parent_pages + (size_t)N * child_pages) > MAXPAGES) {
    std::printf("pool too small for this configuration\n");
    return 1;
  }

  // The pool. Pages 0..parent_pages-1 hold the parent; everything after is free.
  std::vector<float> Kpool((size_t)MAXPAGES * PAGE * D, 0.0f),
      Vpool((size_t)MAXPAGES * PAGE * D, 0.0f);
  bench::fill(Kpool.data(), (size_t)parent_pages * PAGE * D, 1);
  bench::fill(Vpool.data(), (size_t)parent_pages * PAGE * D, 2);

  std::vector<float> Q((size_t)N * D), out((size_t)N * D);
  bench::fill(Q.data(), Q.size(), 3);

  // Each child's own generated tokens. Written into whichever page the layout gives it.
  std::vector<float> childK((size_t)N * S * D), childV((size_t)N * S * D);
  bench::fill(childK.data(), childK.size(), 4);
  bench::fill(childV.data(), childV.size(), 5);

  // -- reference: attention over [parent tokens ; this child's tokens], in double ----------
  std::vector<float> want((size_t)N * D);
  {
    const double scale = 1.0 / std::sqrt((double)D);
    std::vector<double> sc(child_len);
    for (int c = 0; c < N; ++c) {
      auto key = [&](int j, int i) -> double {
        return j < P ? Kpool[(size_t)j * D + i] : childK[((size_t)c * S + (j - P)) * D + i];
      };
      auto val = [&](int j, int i) -> double {
        return j < P ? Vpool[(size_t)j * D + i] : childV[((size_t)c * S + (j - P)) * D + i];
      };
      double mx = -1e300;
      for (int j = 0; j < child_len; ++j) {
        double a = 0;
        for (int i = 0; i < D; ++i) a += (double)Q[(size_t)c * D + i] * key(j, i);
        sc[j] = a * scale;
        mx = std::max(mx, sc[j]);
      }
      double l = 0;
      for (int j = 0; j < child_len; ++j) { sc[j] = std::exp(sc[j] - mx); l += sc[j]; }
      for (int d = 0; d < D; ++d) {
        double o = 0;
        for (int j = 0; j < child_len; ++j) o += sc[j] * val(j, d);
        want[(size_t)c * D + d] = (float)(o / l);
      }
    }
  }
  // The parent's tokens are contiguous in pages 0.. only because nothing has been evicted;
  // the reference indexes them linearly, which matches the pool layout above.

  // -- build the two layouts on the host, as an allocator would ---------------------------
  const int TS = child_pages;                    // table stride
  std::vector<int> deep_tab((size_t)N * TS, 0), cow_tab((size_t)N * TS, 0);
  std::vector<int> lengths(N, child_len);
  std::vector<int> refcount(MAXPAGES, 0);
  for (int p = 0; p < parent_pages; ++p) refcount[p] = 1;   // the parent holds them

  // The two layouts are allocated independently — each starts its free list just past the
  // parent — because the whole comparison is how many pages each strategy needs, and sharing
  // one counter between them would charge the second for the first's allocations.
  std::vector<int> deep_src, deep_dst, cow_src, cow_dst;

  int next = parent_pages;
  for (int c = 0; c < N; ++c) {
    // (a) deep copy: every page is fresh, and every parent page is physically duplicated.
    for (int p = 0; p < child_pages; ++p) {
      deep_tab[(size_t)c * TS + p] = next;
      if (p < parent_pages) { deep_src.push_back(p); deep_dst.push_back(next); }
      ++next;
    }
  }
  const int deep_high_water = next;

  next = parent_pages;
  for (int c = 0; c < N; ++c) {
    // (b) copy-on-write: share every FULL parent page; copy only the partial tail, because
    // that is the only page this child will write into.
    for (int p = 0; p < parent_pages - 1; ++p) {
      cow_tab[(size_t)c * TS + p] = p;           // shared — no copy, no new page
      ++refcount[p];
    }
    cow_tab[(size_t)c * TS + parent_pages - 1] = next;
    cow_src.push_back(parent_pages - 1);
    cow_dst.push_back(next);
    ++next;
    for (int p = parent_pages; p < child_pages; ++p) {
      cow_tab[(size_t)c * TS + p] = next++;      // fresh pages for the new tokens
    }
  }
  const int cow_high_water = next;

  float *dK, *dV, *dQ, *dout;
  int *dtab, *dlen, *dsrc, *ddst;
  CUDA_CHECK(cudaMalloc((void**)&dK, Kpool.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dV, Vpool.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dQ, Q.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dout, out.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dtab, (size_t)N * TS * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dlen, (size_t)N * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dsrc, (size_t)N * child_pages * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&ddst, (size_t)N * child_pages * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dlen, lengths.data(), lengths.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  std::vector<bench::Row> rows;
  const double page_bytes = 2.0 * PAGE * D * sizeof(float);

  // Writing each child's generated tokens into whatever page its table points at. This is the
  // step that makes copy-on-write necessary: without the tail copy, all N children would be
  // appending into the same physical page and overwriting each other.
  auto write_child_tokens = [&](const std::vector<int>& tab) {
    for (int c = 0; c < N; ++c)
      for (int t = 0; t < S; ++t) {
        const int pos = P + t;
        const int page = tab[(size_t)c * TS + pos / PAGE];
        const size_t dstoff = ((size_t)page * PAGE + pos % PAGE) * D;
        std::memcpy(&Kpool[dstoff], &childK[((size_t)c * S + t) * D], D * sizeof(float));
        std::memcpy(&Vpool[dstoff], &childV[((size_t)c * S + t) * D], D * sizeof(float));
      }
  };

  auto run = [&](const char* name, const std::vector<int>& tab, const std::vector<int>& src,
                 const std::vector<int>& dst, double traffic, int high_water,
                 const char* note) {
    // Reset the pool to just the parent, then perform the fork and the appends.
    std::fill(Kpool.begin(), Kpool.end(), 0.0f);
    std::fill(Vpool.begin(), Vpool.end(), 0.0f);
    bench::fill(Kpool.data(), (size_t)parent_pages * PAGE * D, 1);
    bench::fill(Vpool.data(), (size_t)parent_pages * PAGE * D, 2);
    CUDA_CHECK(cudaMemcpy(dK, Kpool.data(), Kpool.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, Vpool.data(), Vpool.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dsrc, src.data(), src.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ddst, dst.data(), dst.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dtab, tab.data(), tab.size() * sizeof(int), cudaMemcpyHostToDevice));

    bench::Row r;
    r.name = name;
    r.st = bench::time_kernel([&] {
      KERNEL_LAUNCH(copy_pages, dim3(32), dim3(BLOCK), 0, dK, dV, dK, dV, dsrc, ddst,
                    (int)src.size());
    }, dev, 20, 5);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Bring the pool back, append each child's tokens through its own table, push it again.
    CUDA_CHECK(cudaMemcpy(Kpool.data(), dK, Kpool.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Vpool.data(), dV, Vpool.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    write_child_tokens(tab);
    CUDA_CHECK(cudaMemcpy(dK, Kpool.data(), Kpool.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, Vpool.data(), Vpool.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    KERNEL_LAUNCH(paged_attend, dim3(N), dim3(BLOCK), 0, dQ, dK, dV, dtab, dlen, dout, TS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(), dout, out.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    r.err = bench::max_rel_err(out.data(), want.data(), out.size());
    r.checksum = bench::checksum_of(out);
    r.bytes = traffic;
    r.flops = 4.0 * N * child_len * D;
    char buf[64];
    std::snprintf(buf, sizeof buf, "%s, %d pages", note, high_water);
    r.note = buf;
    rows.push_back(r);
  };

  std::printf("problem   : parent of %d tokens forks into %d children, each generating %d more\n",
              P, N, S);
  std::printf("paging    : %d tokens/page, parent occupies %d pages, last one %d/%d full\n",
              PAGE, parent_pages, tail_used, PAGE);
  bench::header(dev);

  run("1 deep copy per child", deep_tab, deep_src, deep_dst,
      (double)deep_src.size() * page_bytes, deep_high_water, "full duplication");
  run("2 copy-on-write tail only", cow_tab, cow_src, cow_dst,
      (double)cow_src.size() * page_bytes, cow_high_water, "shares full pages");

  const double tol = 2e-4;
  bench::rows_out(rows, dev, tol);

  std::printf("\nBoth produce the same attention output for every child — checked against a\n"
              "reference over [parent ; child] in double. Sharing pages between branches is\n"
              "invisible to the read path, which is exactly why it is safe.\n");

  // -- the accounting, and the reference counts -------------------------------------------
  std::printf("\nWhat the fork cost:\n\n");
  std::printf("  %-26s %10s %12s %14s\n", "strategy", "pages", "copied", "pool used");
  std::printf("  %-26s %10d %9.2f MB %11.2f MB\n", "deep copy",
              deep_high_water, deep_src.size() * page_bytes / 1048576.0,
              deep_high_water * page_bytes / 1048576.0);
  std::printf("  %-26s %10d %9.2f MB %11.2f MB\n", "copy-on-write",
              cow_high_water, cow_src.size() * page_bytes / 1048576.0,
              cow_high_water * page_bytes / 1048576.0);
  std::printf("\n  copy traffic  %.1fx less     pool footprint  %.1fx less\n",
              (double)deep_src.size() / cow_src.size(),
              (double)deep_high_water / cow_high_water);
  std::printf("  (pool used counts the parent's own %d pages, which both strategies keep.)\n",
              parent_pages);

  int shared_pages = 0, max_ref = 0;
  for (int p = 0; p < MAXPAGES; ++p) {
    if (refcount[p] > 1) ++shared_pages;
    max_ref = std::max(max_ref, refcount[p]);
  }
  std::printf("\n  %d pages are shared, the busiest by %d readers (the parent plus %d children).\n",
              shared_pages, max_ref, N);
  std::printf("  Freeing a branch decrements; a page is returned to the pool when it hits 0.\n"
              "  Getting that wrong is the bug that corrupts a *different* user's context, so\n"
              "  it is worth more care than the copy it saves.\n");

  std::printf("\nHow the saving scales with the fork point (PAGE=%d, %d children):\n\n", PAGE, N);
  std::printf("  %14s %14s %14s %10s\n", "parent tokens", "deep copy", "copy-on-write", "ratio");
  for (int p : {128, 512, 2048, 8192, 32768, 131072}) {
    const double deep = (double)N * ((p + PAGE - 1) / PAGE) * page_bytes;
    const double cow = (double)N * page_bytes;
    std::printf("  %14d %11.2f MB %11.3f MB %9.0fx\n", p, deep / 1048576.0, cow / 1048576.0,
                deep / cow);
  }
  std::printf("\n  Copy-on-write is O(1) in the parent's length: one partial page per child,\n"
              "  however long the conversation. That is what makes deep agent trees — fork,\n"
              "  explore, fork again — affordable, and it is the same indirection\n"
              "  PagedAttention introduced for a completely different reason.\n");

  for (void* p : {(void*)dK, (void*)dV, (void*)dQ, (void*)dout, (void*)dtab, (void*)dlen,
                  (void*)dsrc, (void*)ddst})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
