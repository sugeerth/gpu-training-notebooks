# kernelbench

An eval harness for CUDA kernels that runs **with or without a GPU**.

```bash
python -m kernelbench eval kernels/ --html report.html
```

```
kernel                             build   correctness      variants   determinism      mutation    efficiency    score
-----------------------------------------------------------------------------------------------------------------------
01_copy                              1/1           5/5           1/1           4/4           2/2           -      100%
02_reduce                            1/1           6/6           1/1           5/5           2/2           -      100%
...
12_moe_dispatch                      1/1           4/4           1/1           3/3           3/3           -      100%
-----------------------------------------------------------------------------------------------------------------------
TOTAL                              12/12         57/57         12/12         45/45         33/33             -     100%

all dimensions pass; 33 injected bugs were caught, so the correctness checks are known to
be capable of failing
```

## Why the dimensions are separate

A kernel test that reports one green tick is answering one question. These are six different
questions, and they fail independently:

| dimension | what it establishes |
|---|---|
| **build** | compiles under every toolchain present — `nvcc` when there is one, and always `g++` against the CPU shim |
| **correctness** | every variant matches the program's own reference, computed in double, **on the same run that is being timed** |
| **variants** | the declared number of implementations is present, so one cannot quietly disappear during a refactor |
| **determinism** | results are bit-stable under shuffled block order — and the variants *declared* order-dependent really are, checked in both directions |
| **mutation** | a realistic bug is injected into each kernel and its own check must catch it |
| **sanity** | no variant reports more bandwidth or FLOP/s than the hardware physically has |
| **efficiency** | the best variant reaches the declared fraction of the **measured** hardware ceiling |

**Read the mutation column first.** It is the only dimension that measures the *test* rather
than the code. A suite where every kernel passes and no injected bug is caught has established
nothing at all, and it will score 0 here rather than 100.

## Two things it does that a test runner does not

### It runs CUDA on machines that have none

`shim/cuda_shim.hpp` compiles a `.cu` file as ordinary C++ and runs it with **one
`std::thread` per CUDA thread**, one block at a time, with real barriers. So:

- `__syncthreads()` is a real barrier — a missing one can still deadlock or produce a race
- shared memory is genuinely shared between a block's threads, and genuinely not shared
  between blocks
- warp shuffles really move data between lanes, so an off-by-one in a reduction tree is wrong
  here exactly as it is on hardware
- blocks run in an unspecified order, so code that assumes an order is wrong here too

It emulates **nothing about performance** — no memory hierarchy, no coalescing, no bank
conflicts, no occupancy — so a shim build reports no timings at all rather than numbers that
could be mistaken for GPU results.

Two source-level concessions, and only two: `KERNEL_LAUNCH(fn, grid, block, shmem, ...)`
instead of `<<<>>>`, which is not C++; and `SHARED(float, tile, 64)` instead of
`__shared__ float tile[64]`, because a macro cannot rewrite a declaration prefix into a
per-block allocation. Everything else is ordinary CUDA C++ that `nvcc` compiles unchanged.

### It reproduces GPU non-determinism on a CPU

CUDA guarantees nothing about the order blocks run in, and the shim honours that: with
`KB_SHIM_SHUFFLE=<seed>` set, it permutes the order. Anything accumulating through
`atomicAdd` gets different floating-point rounding, so running twice with two seeds and
comparing exact checksums detects order-dependence **deterministically, on a laptop**.

That is the mechanism behind "why does my loss curve not reproduce" and behind logits that
depend on batch composition, and it is normally only observable as flakiness on real hardware.
Here it is a dimension you can gate a build on:

```json
"nondeterministic_variants": ["1 atomic dW (not reproducible)"]
```

Declaring a variant asserts that it **is** order-dependent. The harness fails if it turns out
to be stable — so the declaration cannot rot into a rubber stamp, and a kernel that silently
*becomes* order-dependent fails too.

> One catch worth knowing, because it decides whether your test can detect anything at all:
> floating-point addition **is** commutative — `a+b` and `b+a` are bit-identical. It is
> *associativity* that fails, so an order-dependent result needs at least three addends. A
> top-2 MoE combined with atomics is reproducible by accident; the same kernel at top-8 is not.

## Adopting it in your project

1. Copy this directory into your repo.
2. Have each kernel program print its results and end with
   `return bench::verdict(rows, tol, &dev);` — that emits the `##KB##` line the harness reads.
   `shim/common.cuh` provides the timing, the reference comparison helpers and the reporting.
3. Write a `kernelbench.json` next to your `.cu` files.
4. Run it in CI.

A minimal suite:

```json
{
  "name": "my-kernels",
  "shim_include": "../kernelbench/shim",
  "determinism_seeds": [1, 5, 42, 1337],
  "cases": [
    {
      "source": "softmax.cu",
      "title": "Softmax — online vs two-pass",
      "topic": "attention",
      "expect_variants": 3,
      "min_efficiency": 0.55,
      "nondeterministic_variants": [],
      "mutations": [
        {
          "name": "online softmax forgets to rescale the accumulator",
          "find": "acc = acc * corr + at;",
          "replace": "acc = acc + at;"
        }
      ]
    }
  ]
}
```

A mutation's `find` must appear **exactly once**. If it stops appearing — because the code was
refactored — that is reported as a failure rather than skipped, because a mutation test whose
anchor has drifted is a test that has silently stopped testing anything. The suite also fails
if a `.cu` file on disk is not declared, so a new kernel cannot slip in unevaluated.

## Options

```
python -m kernelbench eval <dir>
  --html report.html         a self-contained visual scorecard
  --markdown report.md       a PR-ready table
  --json report.json         machine-readable
  --baseline base.json       fail on score regressions, and on the best variant getting slower
  --update-baseline          record this run as the new baseline
  --regress-pct 15           how much slower counts as a regression (default 15%)
  --fail-under 1.0           minimum overall score (default: everything must pass)
  --only 03_sgemm.cu         restrict to one kernel
  --skip mutation            skip a dimension
  --no-nvcc                  force the CPU shim even where nvcc exists
  --arch sm_75               nvcc -arch (default: native, needs CUDA >= 11.5)
```

The baseline gate catches the failure mode nothing else here does: a kernel that stays correct
and gets slower, one careless commit at a time.

## Also here

`bench.py` is the Python-side measurement harness — CUDA events rather than a host clock
around an async launch, warmup, an L2 flush between repetitions, median and MAD instead of
mean and stddev, a bootstrap interval that tells you how many digits you may quote, and
optional CUDA-graph capture. It **measures the machine's own achievable bandwidth and FLOP/s**
at construction rather than reading a spec sheet, so a roofline drawn against it puts a good
kernel near 100% instead of near the 80% that vendor peak makes look like a ceiling.

```python
from kernelbench.bench import Bench
b = Bench()
print(b.describe())
r = b.time(lambda: y.copy_(x), "copy", nbytes=2 * x.nbytes)
print(b.table([r]))
print(b.roofline(r))
```

Run it directly (`python -m kernelbench.bench`) for a report on whatever hardware it finds.
