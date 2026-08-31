# kernels/

Twenty-two CUDA programs that build up, from a memory copy to the scheduling decisions an
agent server makes a thousand times a second. Each one is a single self-contained `.cu` file: several implementations of the
same function, a correctness check against a CPU reference, and a benchmark that reports every
result as a fraction of what the card can actually do.

They are meant to be read in order. Each one introduces the mechanism the next one relies on.

## Run them

**On Colab, or anything with a GPU:**

```bash
git clone https://github.com/sugeerth/gpu-training-notebooks
cd gpu-training-notebooks/kernels
make run
```

`nvcc -arch=native` picks the right target for whatever card you were assigned. Each program
queries the device, measures against its peak numbers, and prints a table.

**With no GPU at all** — a laptop, a CI runner, a Colab CPU runtime:

```bash
make check
```

This compiles the same `.cu` files with `g++` and runs them. See [the shim](#no-gpu-no-cuda-no-problem)
below for how, and for what it does and does not prove.

Other targets: `make 03_sgemm` builds and runs one kernel; `make list`; `make clean`;
`make ARCH=sm_75` for CUDA toolkits older than 11.5, which do not support `-arch=native`.

## The twenty-two

**Foundations** — the machine, one mechanism at a time.

| file | what it is about | the mechanism it introduces |
|---|---|---|
| [`01_copy.cu`](01_copy.cu) | copying an array, four ways | coalescing, transactions, 128-bit access, memory-level parallelism |
| [`02_reduce.cu`](02_reduce.cu) | summing an array, five ways | warp divergence, shared-memory bank conflicts, warp shuffles |
| [`03_sgemm.cu`](03_sgemm.cu) | `C = A·B`, four ways | arithmetic intensity, shared-memory tiling, register blocking |
| [`04_rmsnorm.cu`](04_rmsnorm.cu) | the first real LLM op | why fusion beats optimization for memory-bound work |
| [`05_dequant_gemv.cu`](05_dequant_gemv.cu) | decode-time matrix-vector, fp32 → int4 | why quantization is a *speed* technique at decode and a *capacity* one at prefill |
| [`06_flash_decode.cu`](06_flash_decode.cu) | decode attention | online softmax, FlashDecoding split-K, PagedAttention |

**Training** — what the backward pass, the optimizer and the wire add.

| file | what it is about | the mechanism it introduces |
|---|---|---|
| [`07_rmsnorm_backward.cu`](07_rmsnorm_backward.cu) | the backward pass of `04` | why a backward kernel needs a cross-row reduction the forward one never does — and how `atomicAdd` buys speed with reproducibility |
| [`08_adamw.cu`](08_adamw.cu) | the optimizer step | four states per parameter, one fused pass; why an optimizer is pure bandwidth |
| [`09_fp8_scaling.cu`](09_fp8_scaling.cu) | fp8 with scaling factors | e4m3 vs e5m2, per-tensor / per-row / per-block / delayed scaling, and where each one silently underflows |
| [`10_collectives.cu`](10_collectives.cu) | ring vs direct all-reduce | identical bytes, different serialized step counts — why rings survive anyway |

**Modern architecture** — what MLA and MoE change about the kernel.

| file | what it is about | the mechanism it introduces |
|---|---|---|
| [`11_mla_decode.cu`](11_mla_decode.cu) | DeepSeek-style latent attention | absorbing `W_UK` into the query so the cache holds `c` and not `K`, and the RoPE part that refuses to absorb |
| [`12_moe_dispatch.cu`](12_moe_dispatch.cu) | routing tokens to experts | why MoE weight traffic *grows* with batch size — the opposite of dense |

**Agents** — what changes when the caller is a loop, not a person.

| file | the property of the loop | what it costs, and what fixes it |
|---|---|---|
| [`13_prefix_attention.cu`](13_prefix_attention.cu) | branches share a long prefix | reading the prefix `N` times. Cascade attention reads it once and merges with the identity from `06` — the saving tends to `(P+S)/S` |
| [`14_logit_mask.cu`](14_logit_mask.cu) | output is JSON at temperature 0 | a grammar mask on nearly every token. A bitset is 1/32 the bytes — and the whole thing is 0.05% of a decode step, which is the lesson |
| [`15_kv_fork.cu`](15_kv_fork.cu) | trajectories fork | duplicating the parent's KV. Copy-on-write shares every full page and copies one partial one per child — O(1) in the parent's length |
| [`16_prefix_match.cu`](16_prefix_match.cu) | the same context arrives again and again | re-prefilling what the server already holds. Chained block hashes find the longest matching prefix in one probe — the cheapest code in the serving path, and the most valuable |
| [`17_ragged_batch.cu`](17_ragged_batch.cu) | every agent is on a different turn | padding a batch to its longest sequence. `cu_seqlens` fixes the wasted work; splitting fixes the critical path, and they are not the same fix |
| [`18_spec_verify.cu`](18_spec_verify.cu) | tool calls are highly predictable text | nothing — this one is a *win* agents get and prose does not, because acceptance enters as `α^(k+1)` and a schema fixes most of the tokens |
| [`19_batch_invariant.cu`](19_batch_invariant.cu) | runs are replayed, retried and evaluated | logits that change with the rest of the batch, because an adaptive split count changes the reduction tree. A fixed split costs occupancy and buys replay |
| [`20_grammar_advance.cu`](20_grammar_advance.cu) | the grammar advances on every token | a CPU round trip inside the decode loop. Keep the per-state bitsets on the device and the per-step traffic becomes a state id — then nothing at all |
| [`21_chunked_prefill.cu`](21_chunked_prefill.cu) | tool results arrive mid-batch | everyone else's inter-token latency. One launch serves decodes and a slice of prefill together; the cap on the spike *is* the chunk size |
| [`22_kv_evict.cu`](22_kv_evict.cu) | the pool fills with sequences that are idle | evicting one that wanted to run. The scheduler issued the tool calls, so it knows who is parked — and the prefix cache makes bringing them back cheap |

The agent track is not a separate topic; it is the first three tracks pointed at a different
caller. `13`, `17` and `21` are `06`'s online-softmax merge over three different partitions of
the keys. `15` is `06`'s block table used for a purpose PagedAttention was not invented for.
`18` and `20` are `14`'s masked argmax run more often and fed differently. `22` is `02`'s argmax
pointed at a scheduling decision. And `19` is `02`'s reduction with the split count taken away
from the scheduler — the sibling of `07`'s order-dependence, one varying with block *order* and
the other with reduction *shape*, only the first detectable by rerunning the same binary.
[Agent Workloads on the Metal](../Agent_Workloads_On_The_Metal.ipynb) runs all ten and costs the
loop around them.

Three of them chase the same cost through three answers. `14` measures a grammar mask and finds
0.05% of a decode step. `18` runs it `k+1` times against a draft model's weights and finds 0.5%.
`20` finds that the expensive part was never the mask at all — it was needing the CPU inside the
decode loop — and fixes it with a table. Each measurement is correct; each was measuring the
cheap half. That progression is the most useful thing in the directory.

`common.cuh` is the measurement harness they share — CUDA events, warmup, an L2 flush between
reps, median and MAD instead of mean and stddev, and every number divided by the hardware's
own ceiling. `cuda_shim.hpp` is the CPU emulator.

## What each program prints

```
device    : NVIDIA A100-SXM4-80GB
peak      : 2039 GB/s memory, 19.5 TFLOP/s fp32, 108 SMs, 40.0 MB L2
ridge     : 9.6 FLOP/byte  (below this a kernel is memory-bound)

variant                         median ms    ±MAD       GB/s   GFLOP/s   max err
--------------------------------------------------------------------------------
1 thread-per-chunk (bad)           4.9012   0.0140      109.6       0.0  0.00e+00  ok   6% of achievable BW
2 grid-stride, coalesced           0.7714   0.0031      696.2       0.0  0.00e+00  ok  38% of achievable BW
...
```

Two things about that table are deliberate:

- **The error column is not decoration.** Every variant is checked against a CPU reference on
  the same run that is being timed, and the program exits non-zero if any variant is wrong. A
  fast kernel that computes the wrong thing is the easiest thing in this field to ship.
- **The percentage is the point.** A time without a ceiling cannot distinguish a good kernel
  from a bad one on a fast card. 700 GB/s is respectable on a T4 and a bug on an H100.

## No GPU, no CUDA, no problem

`cuda_shim.hpp` compiles a `.cu` file as ordinary C++ and runs it with **one `std::thread` per
CUDA thread**, one block at a time. That makes it faithful in the ways that matter for reading
kernel code:

- `__syncthreads()` is a real barrier
- shared memory is really shared between the threads of a block, and really is not shared
  between blocks
- warp shuffles really move data between lanes, so an off-by-one in a reduction tree gives a
  wrong answer here exactly as it would on hardware
- blocks run in an unspecified order, so code that assumes an order is wrong here too

It emulates **nothing about performance** — no memory hierarchy, no coalescing, no bank
conflicts, no occupancy. A kernel that is 40x faster on a GPU is indistinguishable here. So a
shim build prints no timings at all, rather than printing numbers that could be mistaken for
GPU results.

Two source-level concessions, marked at every use site: `KERNEL_LAUNCH(fn, grid, block, shmem, ...)`
instead of `fn<<<grid, block, shmem>>>(...)`, because `<<<>>>` is not C++; and
`SHARED(float, tile, 64)` instead of `__shared__ float tile[64]`, because a macro cannot rewrite
a declaration prefix into a per-block allocation. Everything else is ordinary CUDA C++ that
`nvcc` compiles unchanged.

### It reproduces GPU non-determinism, on a CPU

Blocks run in an unspecified order on hardware, so a reduction that accumulates with
`atomicAdd` gives a slightly different answer each run — floating-point addition is not
associative. That is the hardest class of bug to pin down, because it only shows up
intermittently and only at scale.

Set `KB_SHIM_SHUFFLE=<seed>` and the shim permutes block order deterministically. An
order-dependent kernel then gives a different checksum per seed, in a second, on a laptop. The
harness gates this in both directions: a kernel declared order-dependent **must** vary, and
every other kernel **must not**.

### It catches real bugs

Running real threads with real barriers means the shim detects race conditions too.
`tools/verify_kernels.py` proves that rather than asserting it — it is a thin wrapper over the
`kernelbench` harness, which mutates each source (deleting a `__syncthreads()`, changing a
stride, dropping a component of a `float4`) and requires the kernel's own correctness check to
fail on every mutant.

```
$ python tools/verify_kernels.py
...
TOTAL                    22/22 build  99/99 correct  22/22 variants  77/77 determinism  71/71 mutation   100%
```

That runs in CI on every push, on a machine with no GPU. A test suite that has never failed is
a rumour.

## Where these sit in the repo

- [GPU Architecture & CUDA Kernels](../GPU_Architecture_And_CUDA_Kernels.ipynb) walks through
  these files and the hardware they are written for.
- [Measuring GPU Code Honestly](../Measuring_GPU_Code_Honestly.ipynb) is about the other half:
  why a benchmark number is usually wrong and what to do about it.
- [Attention Kernels From Scratch](../Attention_Kernels_From_Scratch.ipynb) develops the
  algorithms in `06_flash_decode.cu` in NumPy first. The Python there is the specification;
  the CUDA here is the machine.
- [The Hardware Roofline](../Hardware_Roofline_NVIDIA_vs_AMD.ipynb) is where the ridge point
  and the memory-bound/compute-bound split come from.
- [Modern GPU & Model Architecture](../Modern_GPU_And_Model_Architecture.ipynb) covers what
  Hopper and Blackwell changed, and the MLA and MoE kernels in `11`–`12`.
- [Training Kernels & Memory](../Training_Kernels_And_Memory.ipynb) runs `07`–`10` and works
  out what a training step costs in memory and on the wire.
- [Agent Workloads on the Metal](../Agent_Workloads_On_The_Metal.ipynb) runs `13`–`15` and
  costs the loop they sit inside — where prefill goes quadratic, and where the kernel your
  instinct says to optimize turns out to be 0.05% of the step.
- [`kernelbench/`](../kernelbench/README.md) is the eval harness itself, with no dependency on
  this repository. Point it at your own `.cu` files.
