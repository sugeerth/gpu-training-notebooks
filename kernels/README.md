# kernels/

Six CUDA programs that build up, from a memory copy to a paged decode-attention kernel. Each
one is a single self-contained `.cu` file: several implementations of the same function, a
correctness check against a CPU reference, and a benchmark that reports every result as a
fraction of what the card can actually do.

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

## The six

| file | what it is about | the mechanism it introduces |
|---|---|---|
| [`01_copy.cu`](01_copy.cu) | copying an array, four ways | coalescing, transactions, 128-bit access, memory-level parallelism |
| [`02_reduce.cu`](02_reduce.cu) | summing an array, five ways | warp divergence, shared-memory bank conflicts, warp shuffles |
| [`03_sgemm.cu`](03_sgemm.cu) | `C = A·B`, four ways | arithmetic intensity, shared-memory tiling, register blocking |
| [`04_rmsnorm.cu`](04_rmsnorm.cu) | the first real LLM op | why fusion beats optimization for memory-bound work |
| [`05_dequant_gemv.cu`](05_dequant_gemv.cu) | decode-time matrix-vector, fp32 → int4 | why quantization is a *speed* technique at decode and a *capacity* one at prefill |
| [`06_flash_decode.cu`](06_flash_decode.cu) | decode attention | online softmax, FlashDecoding split-K, PagedAttention |

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

### It catches real bugs

Running real threads with real barriers means the shim detects race conditions, which is the
class of bug a GPU exposes only intermittently and only at scale. `tools/verify_kernels.py`
proves this rather than asserting it: it injects one realistic bug into each kernel and
requires that the check fails.

```
$ python tools/verify_kernels.py
  01_copy.cu               caught  — grid-stride loop skips half its elements
  02_reduce.cu             caught  — shuffle reduction tree stops one level early
  03_sgemm.cu              caught  — missing __syncthreads() — write-after-read race on the shared tile
  04_rmsnorm.cu            caught  — sum of squares drops the last component of each float4
  05_dequant_gemv.cu       caught  — int4 unpacking indexes weights instead of bytes
  06_flash_decode.cu       caught  — split-K writes normalized partials, so the merge cannot reweight them
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
