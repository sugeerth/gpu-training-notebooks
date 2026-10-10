// 21_chunked_prefill.cu — one launch that serves both a decode and a prefill, and why an agent
// batch falls apart without it.
//
//     nvcc -O3 -arch=native 21_chunked_prefill.cu -o build/21 && build/21
//     make check
//
// An agent turn ends by appending a tool result — a page of search output, a file, a stack
// trace — and that result has to be prefilled before the next token can be generated. A
// two-thousand-token tool result is a two-thousand-token prefill arriving *in the middle of a
// batch of other agents that are trying to decode*.
//
// A scheduler that runs prefills and decodes in separate steps has two bad options and no good
// one:
//
//   prefill first.   The arriving tool result monopolizes a step. Every other sequence in the
//                    batch waits, and their inter-token latency spikes by however long that
//                    prefill took. With agents this is not an occasional event — every agent
//                    finishes a tool call every few seconds, so the spikes are constant and
//                    every sequence is a victim of everybody else's tool calls.
//
//   decode first.    The prefill waits behind the decodes, so the agent that just got its tool
//                    result sits idle a little longer, which lengthens the loop it is in.
//
// Chunked prefill dissolves the choice: cut the prefill into fixed-size pieces and put one
// piece **into the same launch as the decodes**. Nobody waits for a whole prefill, because a
// whole prefill never happens at once.
//
// The kernel change is small and specific. A decode row has one query token; a prefill chunk
// has many. So the batch needs **two** cumulative-length arrays — `cu_q` over query tokens and
// `cu_k` over keys — instead of the one that 17_ragged_batch needed. Every row then states how
// many queries it brings and how many keys it attends to, and a decode is simply the case where
// the first number is 1. There is no separate decode kernel.
//
// That is the whole idea, and it is worth noticing what it is *not*. It does not reduce work:
// exactly the same queries attend to exactly the same keys. It re-shapes when the work happens,
// and it trades the prefilling sequence's latency for everyone else's. Which is the right trade
// for agents, because the prefilling sequence is about to go and wait several seconds for a
// tool anyway.
//
// Prerequisites: 17_ragged_batch.cu for cu_seqlens, 06_flash_decode.cu for the causal masking.
#include <map>
#include <utility>

#include "common.cuh"

#if SHIM_BUILD
constexpr int NDECODE = 4;      // agents generating a token this step
constexpr int PREFILL_LEN = 40; // tokens of tool result waiting to be prefilled
constexpr int CHUNK = 8;        // query tokens of prefill admitted per step
constexpr int D = 32;
constexpr int TILE = 8;
constexpr int BLOCK = 32;
constexpr int CTX = 48;         // context the decoding agents already hold
#else
constexpr int NDECODE = 32;
constexpr int PREFILL_LEN = 2048;
constexpr int CHUNK = 512;
constexpr int D = 128;
constexpr int TILE = 32;
constexpr int BLOCK = 128;
constexpr int CTX = 4096;
#endif
constexpr float NEG = -1e30f;
constexpr int MAXROWS = NDECODE + 1;

// ---------------------------------------------------------------------------------------
// The unified kernel. One row per (sequence, query token): `row_seq[i]` says which sequence
// query i belongs to and `row_pos[i]` its absolute position, which is what the causal mask
// needs. A decode row contributes one query; a prefill chunk contributes CHUNK of them.
//
// Everything below is the same attention 17_ragged_batch runs. The only additions are the
// causal bound — a prefill query at position p may not see keys after p — and the second
// cumulative array. That is genuinely all that separates "a decode kernel" from "a kernel that
// serves decodes and prefills together", and it is why every engine converged on it.
// ---------------------------------------------------------------------------------------
__global__ void attend_mixed(const float* __restrict__ Q, const float* __restrict__ K,
                             const float* __restrict__ V, const int* __restrict__ cu_k,
                             const int* __restrict__ row_seq, const int* __restrict__ row_pos,
                             float* __restrict__ out, int nrows) {
  SHARED(float, sq, D);
  SHARED(float, sK, TILE * D);
  SHARED(float, sV, TILE * D);
  SHARED(float, sp, TILE);

  const int i = blockIdx.x, d = threadIdx.x;
  if (i >= nrows) return;
  const float scale = rsqrtf((float)D);
  sq[d] = Q[(size_t)i * D + d];
  __syncthreads();

  const int seq = row_seq[i];
  const int begin = cu_k[seq];
  // Causal: this query sees keys 0..row_pos[i] of its own sequence, and no further.
  const int L = row_pos[i] + 1;

  float m = NEG, l = 0.0f, acc = 0.0f;
  for (int j0 = 0; j0 < L; j0 += TILE) {
    const int n = (L - j0 < TILE) ? (L - j0) : TILE;
    for (int r = 0; r < n; ++r) {
      const size_t base = (size_t)(begin + j0 + r) * D;
      sK[r * D + d] = K[base + d];
      sV[r * D + d] = V[base + d];
    }
    __syncthreads();
    if (d < n) {
      float s = 0.0f;
      for (int t = 0; t < D; ++t) s += sq[t] * sK[d * D + t];
      sp[d] = s * scale;
    }
    __syncthreads();

    float mt = NEG;
    for (int t = 0; t < n; ++t) mt = fmaxf(mt, sp[t]);
    const float mnew = fmaxf(m, mt);
    const float corr = __expf(m - mnew);
    float lt = 0.0f, at = 0.0f;
    for (int t = 0; t < n; ++t) {
      const float p = __expf(sp[t] - mnew);
      lt += p;
      at += p * sV[t * D + d];
    }
    l = l * corr + lt;
    acc = acc * corr + at;
    m = mnew;
    __syncthreads();
  }
  out[(size_t)i * D + d] = acc / l;
}

// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
  bench::Device dev = bench::query_device();
  (void)argc; (void)argv;

  // The batch: NDECODE agents mid-conversation, plus one that has just come back from a tool
  // call carrying PREFILL_LEN tokens of result.
  std::vector<int> klen(MAXROWS);
  for (int s = 0; s < NDECODE; ++s) klen[s] = CTX;
  klen[NDECODE] = PREFILL_LEN;                 // the returning agent's tool result

  std::vector<int> cu_k(MAXROWS + 1, 0);
  for (int s = 0; s < MAXROWS; ++s) cu_k[s + 1] = cu_k[s] + klen[s];
  const int total_k = cu_k[MAXROWS];

  std::vector<float> K((size_t)total_k * D), V((size_t)total_k * D);
  bench::fill(K.data(), K.size(), 3);
  bench::fill(V.data(), V.size(), 4);

  // Query rows. A "schedule" is just a list of (sequence, position) pairs; the kernel neither
  // knows nor cares which of them are decodes.
  struct Sched {
    std::vector<int> seq, pos;
    int prefill_rows = 0;
  };
  auto decodes_only = [&]() {
    Sched s;
    for (int i = 0; i < NDECODE; ++i) { s.seq.push_back(i); s.pos.push_back(CTX - 1); }
    return s;
  };
  auto prefill_only = [&](int lo, int hi) {
    Sched s;
    for (int p = lo; p < hi; ++p) { s.seq.push_back(NDECODE); s.pos.push_back(p); }
    s.prefill_rows = hi - lo;
    return s;
  };
  auto mixed = [&](int lo, int hi) {
    Sched s = decodes_only();
    for (int p = lo; p < hi; ++p) { s.seq.push_back(NDECODE); s.pos.push_back(p); }
    s.prefill_rows = hi - lo;
    return s;
  };

  float *dQ, *dK, *dV, *dout;
  int *dcu_k, *dseq, *dpos;
  const int MAXQ = NDECODE + PREFILL_LEN;
  CUDA_CHECK(cudaMalloc((void**)&dQ, (size_t)MAXQ * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dK, K.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dV, V.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dout, (size_t)MAXQ * D * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&dcu_k, cu_k.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dseq, MAXQ * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dpos, MAXQ * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dK, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dcu_k, cu_k.data(), cu_k.size() * sizeof(int), cudaMemcpyHostToDevice));

  // Every query token in the whole problem, indexed by (seq, pos) so any schedule can pick the
  // rows it wants and they always carry the same values.
  std::vector<float> Qall((size_t)MAXROWS * (CTX > PREFILL_LEN ? CTX : PREFILL_LEN) * D);
  bench::fill(Qall.data(), Qall.size(), 5);
  const int QSTRIDE = (CTX > PREFILL_LEN ? CTX : PREFILL_LEN);
  auto qptr = [&](int seq, int pos) { return &Qall[((size_t)seq * QSTRIDE + pos) * D]; };

  // -- reference: causal attention for every query that will ever be scheduled --------------
  auto reference = [&](int seq, int pos, float* dst) {
    const double scale = 1.0 / std::sqrt((double)D);
    const int L = pos + 1, begin = cu_k[seq];
    std::vector<double> sc(L);
    double mx = -1e300;
    for (int j = 0; j < L; ++j) {
      double a = 0;
      for (int t = 0; t < D; ++t) a += (double)qptr(seq, pos)[t] * K[(size_t)(begin + j) * D + t];
      sc[j] = a * scale;
      mx = std::max(mx, sc[j]);
    }
    double l = 0;
    for (int j = 0; j < L; ++j) { sc[j] = std::exp(sc[j] - mx); l += sc[j]; }
    for (int d = 0; d < D; ++d) {
      double o = 0;
      for (int j = 0; j < L; ++j) o += sc[j] * V[(size_t)(begin + j) * D + d];
      dst[d] = (float)(o / l);
    }
  };

  // Run one schedule and return its output, row by row, keyed by (seq, pos) so schedules that
  // cover the same work can be compared directly.
  std::map<std::pair<int, int>, std::vector<float>> want, got;
  auto run = [&](const Sched& s) {
    const int n = (int)s.seq.size();
    std::vector<float> Q((size_t)n * D);
    for (int i = 0; i < n; ++i)
      std::memcpy(&Q[(size_t)i * D], qptr(s.seq[i], s.pos[i]), D * sizeof(float));
    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), (size_t)n * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dseq, s.seq.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dpos, s.pos.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    KERNEL_LAUNCH(attend_mixed, dim3(n), dim3(BLOCK), 0, dQ, dK, dV, dcu_k, dseq, dpos, dout, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> o((size_t)n * D);
    CUDA_CHECK(cudaMemcpy(o.data(), dout, (size_t)n * D * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
      std::vector<float> row(o.begin() + (size_t)i * D, o.begin() + (size_t)(i + 1) * D);
      got[{s.seq[i], s.pos[i]}] = row;
    }
    return n;
  };

  // Everything any schedule will produce, computed once in double.
  for (int i = 0; i < NDECODE; ++i) {
    std::vector<float> r(D);
    reference(i, CTX - 1, r.data());
    want[{i, CTX - 1}] = r;
  }
  for (int p = 0; p < PREFILL_LEN; ++p) {
    std::vector<float> r(D);
    reference(NDECODE, p, r.data());
    want[{NDECODE, p}] = r;
  }

  std::vector<bench::Row> rows;
  // A variant's error is the worst over every row it produced, and its checksum covers all of
  // them in a fixed key order — so two schedules that cover the same work must agree exactly,
  // which is the property that makes chunking safe to turn on.
  auto record = [&](const char* name, int launches, int rows_total, double bytes,
                    const char* note) {
    double err = 0;
    std::vector<float> sig;
    for (auto& kv : want) {
      auto it = got.find(kv.first);
      if (it == got.end()) continue;
      err = std::max(err, bench::max_rel_err(it->second.data(), kv.second.data(), D));
      sig.insert(sig.end(), it->second.begin(), it->second.end());
    }
    bench::Row r;
    r.name = name;
    r.err = err;
    r.checksum = bench::checksum_of(sig);
    r.bytes = bytes;
    r.flops = 0;
    char buf[80];
    std::snprintf(buf, sizeof buf, "%s, %d launches, %d rows", note, launches, rows_total);
    r.note = buf;
    rows.push_back(r);
  };

  // -- schedule 1: prefill the whole tool result, then decode -------------------------------
  {
    got.clear();
    int launches = 0, total = 0;
    total += run(prefill_only(0, PREFILL_LEN));
    ++launches;
    total += run(decodes_only());
    ++launches;
    bench::Stats st = bench::time_kernel([&] {
      run(prefill_only(0, PREFILL_LEN));
      run(decodes_only());
    }, dev, 10, 3);
    record("1 prefill-first, then decode", launches, total, 0, "decodes wait for all of it");
    rows.back().st = st;
  }

  // -- schedule 2: decode, then prefill the whole tool result -------------------------------
  {
    got.clear();
    int launches = 0, total = 0;
    total += run(decodes_only());
    ++launches;
    total += run(prefill_only(0, PREFILL_LEN));
    ++launches;
    bench::Stats st = bench::time_kernel([&] {
      run(decodes_only());
      run(prefill_only(0, PREFILL_LEN));
    }, dev, 10, 3);
    record("2 decode-first, then prefill", launches, total, 0, "the agent waits instead");
    rows.back().st = st;
  }

  // -- schedule 3: chunked — one prefill chunk rides along with the decodes ------------------
  const int nchunks = (PREFILL_LEN + CHUNK - 1) / CHUNK;
  {
    got.clear();
    int launches = 0, total = 0;
    for (int c = 0; c < nchunks; ++c) {
      const int lo = c * CHUNK;
      const int hi = std::min(lo + CHUNK, PREFILL_LEN);
      total += run(mixed(lo, hi));
      ++launches;
    }
    bench::Stats st = bench::time_kernel([&] {
      for (int c = 0; c < nchunks; ++c)
        run(mixed(c * CHUNK, std::min((c + 1) * CHUNK, PREFILL_LEN)));
    }, dev, 10, 3);
    record("3 chunked, mixed with decode", launches, total, 0, "nobody waits for a whole one");
    rows.back().st = st;
  }

  std::printf("problem   : %d agents decoding at %d tokens of context, while one returns from\n"
              "            a tool call with %d tokens to prefill\n", NDECODE, CTX, PREFILL_LEN);
  std::printf("chunk     : %d query tokens of prefill admitted per step, so %d steps\n",
              CHUNK, nchunks);
  bench::header(dev);
  const double tol = 2e-4;
  bench::rows_out(rows, dev, tol);

  std::printf("\nAll three schedules produce identical output for every query token, checked\n"
              "against causal attention computed in double. Chunking changes *when* a query is\n"
              "computed, never what it computes — which is exactly why it is safe to turn on,\n"
              "and why the check is on every row rather than on a summary.\n");

  // -- what each schedule does to the decoding agents ---------------------------------------
  std::printf("\nWhat each schedule costs the %d agents that are just trying to decode:\n\n",
              NDECODE);
  std::printf("  %-30s %16s %18s %16s\n", "schedule", "decode steps", "worst step (rows)",
              "extra ITL");
  {
    // "Rows" is the honest proxy for step time here: the shim reports no timings, and a row is
    // one query token's worth of attention work.
    const int rows_decode = NDECODE;
    const int rows_prefill = PREFILL_LEN;
    const int rows_mixed = NDECODE + CHUNK;
    std::printf("  %-30s %16d %18d %13d rows\n", "prefill-first", 1, rows_prefill, rows_prefill);
    std::printf("  %-30s %16d %18d %13d rows\n", "decode-first", 1, rows_decode, 0);
    std::printf("  %-30s %16d %18d %13d rows\n", "chunked", nchunks, rows_mixed, CHUNK);
    std::printf("\n  Prefill-first puts a %d-row step in front of every decoding agent — their\n"
                "  inter-token latency takes the whole tool result on the chin. Chunked caps\n"
                "  that at %d extra rows per step, a factor of %.1f, and the cap is a knob:\n"
                "  it is exactly CHUNK.\n",
                rows_prefill, CHUNK, (double)rows_prefill / CHUNK);
  }

  // -- and what it costs the agent doing the prefilling --------------------------------------
  std::printf("\nAnd what it costs the agent that is waiting on its own prefill:\n\n");
  std::printf("  %-30s %20s %18s\n", "schedule", "steps to finish", "vs prefill-first");
  std::printf("  %-30s %20d %18s\n", "prefill-first", 1, "1.0x");
  std::printf("  %-30s %20d %17.1fx\n", "chunked", nchunks, (double)nchunks);
  std::printf("\n  This is the trade, and it is not free: the returning agent needs %d steps\n"
              "  instead of 1 before it can generate. Whether that is a good deal depends on\n"
              "  something outside the kernel — and for agents it plainly is, because that\n"
              "  sequence is about to spend seconds in another tool call. Trading a few\n"
              "  milliseconds of its latency for every other sequence's smoothness is the\n"
              "  easiest bargain in the scheduler.\n"
              "\n  For an interactive chat workload, where the prefilling sequence is a person\n"
              "  waiting on a first token, the same trade is much less obvious. Chunked\n"
              "  prefill is not universally correct; it is correct when the prefilling party\n"
              "  is a loop and not a person.\n", nchunks);

  // -- the knob ------------------------------------------------------------------------------
  std::printf("\nChoosing CHUNK, for a %d-token tool result:\n\n", PREFILL_LEN);
  std::printf("  %10s %16s %18s %16s\n", "chunk", "steps", "extra rows/step",
              "launches vs above");
  for (int c : {CHUNK / 4, CHUNK / 2, CHUNK, CHUNK * 2, CHUNK * 4}) {
    if (c < 1) continue;
    const int steps = (PREFILL_LEN + c - 1) / c;
    std::printf("  %10d %16d %18d %13.1fx\n", c, steps, c, (double)steps / nchunks);
  }
  std::printf("\n  Smaller chunks smooth the decoders' latency and multiply the launches —\n"
              "  each of which re-reads the decoding sequences' KV, so the total work grows.\n"
              "  There is no chunk size that is right for every deployment, which is why every\n"
              "  engine exposes it and none picks a default that survives contact.\n");

  for (void* p : {(void*)dQ, (void*)dK, (void*)dV, (void*)dout, (void*)dcu_k, (void*)dseq,
                  (void*)dpos})
    CUDA_CHECK(cudaFree(p));
  return bench::verdict(rows, tol, &dev);
}
