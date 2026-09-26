# servingkit

The planning models from this repository's notebooks, as one importable library — plus a
composition layer that decides which of them apply to *your* workload and what the combination
costs.

```bash
pip install -e .                      # from the repo root
python -m servingkit plan --agent --turns 30 --fanout 4
```

```python
import servingkit as sk

w = sk.Workload.agent(turns=30, fanout=4, tool_result_tokens=2000, replayed=True)
s = sk.recommend(w)
print(sk.plan_report(s))
print(sk.ladder_table(s.ladder()))
```

## Why this exists

Forty-nine notebooks each carried their own copy of the arithmetic. The eight demo pages
carried a third copy, in JavaScript. `tools/audit_consistency.py` existed to notice when the
copies disagreed, and it was needed, because they did.

This package is the single copy. Every model in it was lifted **verbatim** from the notebook
that explains it, and `python -m servingkit check` pulls each function back out of its notebook
and compares the two over ~16,000 inputs. The notebooks remain where the reasoning lives; this
is where the numbers live.

## Two layers

### The models — plain functions over plain dicts

Nothing to construct, nothing to configure, no dependencies at all.

```python
sk.predict("H100 SXM", "Llama-3.1-8B", batch=32)["tpot_ms"]
sk.kv_total_bytes(sk.MODELS["Mistral-7B"], ctx=131072) / 1e9    # sliding window: stops growing
sk.training_plan(params_b=70, gpus=8, zero_stage=3)["total_gb"]
sk.agent_run(turns=40, tool_result_tokens=2000)["prefill_naive"]
sk.expected_tokens(alpha=0.9, k=5)                              # speculation
sk.cheapest("Llama-3.1-8B", max_tpot_ms=30)[0]["cpm"]           # $/M tokens, cheapest first
```

| module | what it answers | lifted from |
|---|---|---|
| `catalog` | what hardware and models exist, and what they cost | eight notebooks, formerly |
| `kv` | what a KV cache costs, by attention architecture | `LongContext_KV_Compression_Serving` |
| `step` | where one decode step's milliseconds go | `The_Optimization_Stack` |
| `serving` | can I serve this, how fast, at what price | `Serving_WhatIf_Console` |
| `training` | can I train it, and what crosses the wire | `Training_Kernels_And_Memory` |
| `agents` | where an agent loop's tokens and seconds go | `Agent_Workloads_On_The_Metal` |
| `spec` | what speculation buys, and where it costs you | `Speculative_Decoding_Advanced_Serving` |

### The composition layer — what makes it infrastructure

A **workload** declares what it is. A **lever** declares what it changes. A **stack** composes
them and tells you what the composition actually does.

The point is that a lever carries four declarations beyond its implementation:

```python
Lever("cascade attention", _noop,
      domain="attention_partition",        # two levers in one domain are alternatives
      spends=("bandwidth",),               # overlap PREDICTS antagonism
      requires=("shared_prefix",),         # absent → not applicable, not merely weak
      kernel="13_prefix_attention.cu")     # "read the code" has an address
```

From those four, four behaviours follow that a dict of functions cannot provide:

**1. It refuses what does not apply.** A chat workload is not offered cascade attention, because
it has no shared prefix to exploit. The rejection names the missing property:

```
not applied
  - prefix caching: workload lacks a required property (repeated_context)
  - cascade attention: workload lacks a required property (shared_prefix)
```

**2. It refuses what conflicts.** A model has exactly one weight format, so `int4 weights` and
`fp8 weights` share a `domain` and cannot both be pulled.

**3. It predicts antagonism instead of discovering it.** When two levers spend the same declared
resource, the interaction table says so *before* the measurement:

```
A                 B              A      B  expect  actual  synergy   why
int4 weights      speculation  1.95x  1.19x   2.32x   1.32x     0.57   both spend spare_compute
2x batch          speculation  1.48x  1.19x   1.77x   1.34x     0.76   both spend spare_compute
```

A synergy below 1 with **no** declared conflict means a lever is lying about what it spends.
That is the check the table exists for, and it earned its place immediately: on the first run,
`int4 weights` and `fp8 weights` both declared only `accuracy`, and two unexplained 0.57s
revealed that shrinking the weight read is what *removes* the memory-bound-ness that batching
and speculation are paid out of. Both declarations were wrong; the table found them.

**4. It orders by money.** `ladder()` adds levers one at a time and reports cost per run at each
rung, including the GPUs a lever rents — because a lever that halves your step time and doubles
your bill has not obviously helped.

```
configuration                      tok/s     prefill     wall         $   util     Δwall      Δ$  kernel
baseline                             272   1,046,250    2234s     3.477   97%
+ prefix caching                     272     115,060    2187s     0.682   97%       -2%    -82%  16_prefix_match.cu
+ ragged batching                    511     115,060    1194s     0.651   95%      -45%    -10%  17_ragged_batch.cu
+ cascade attention                1,492     115,060     452s     0.629   87%      -62%     -6%  13_prefix_attention.cu
```

## The preconditions are the product

The most useful thing here is not the speedups, it is the refusals. Two examples that the
package gets right and most advice gets wrong:

- **"Just raise the batch size"** has a precondition. Batching amortizes *one weight read* over
  more sequences, so it needs the weight read to be the thing that hurts. At 66k of context,
  attention is 17× the GEMM and doubling the batch doubles attention while buying nothing. So
  `2x batch` requires `weight_bound`, and a long-context agent is not offered it.
- **Speculation** has the same precondition and a sharper version of the same failure: it
  multiplies the attention term by `k+1`. Offered to the wrong workload it is not a weak
  optimization, it is a regression — and the first draft of `workload.py` gated it on
  `batch < 128` and offered it to exactly the workload it would have harmed.

`Stack.unavailable()` inverts the question: it lists the levers you *cannot* use and the
property each one needs. That list is what to engineer into existence, not what to turn on.

## Every lever names its kernel

The levers are not abstractions over vendor documentation. Each one points at a compilable
program in [`kernels/`](../kernels/) that implements it, checks itself against a
double-precision reference, and is scored by
[`kernelbench`](../kernelbench/README.md) on build, correctness, determinism and mutation
coverage. You can run the one a lever names without a GPU:

```python
r = sk.run_kernel(sk.LEVERS["cascade attention"].kernel)    # compiles with g++ via the shim
print(r.ok, r.table())
```

`r.timed` is `False` on a CPU, and the timings are absent rather than fabricated.

## Every number is a planning model

None of this is a measurement. The catalogs are spec sheets and list prices; `BW_EFF = 0.75` and
`FLOP_EFF = 0.60` are what a good kernel reaches, not what yours does. Substitute your own:

```python
sk.GPUS["my H100"] = sk.recalibrate("H100 SXM", measured_bw_tbs=2.6, measured_gemm_tf=620)
```

The shape of the answer does not move when you do that. The crossovers do, and the crossovers
are the part you were going to make a decision on.

## Running it as a service

The composable layer is also a JSON API and an end-to-end pipeline, both on the standard library
only — no FastAPI, no uvicorn. That constraint is the point: the package's claim is that it is
arithmetic over dicts, and a service needing three hundred megabytes of wheels to expose that
arithmetic would undermine the claim.

```bash
python -m servingkit serve --port 8000
curl -s localhost:8000/v1/plan -d '{"kind":"agent","turns":30,"fanout":4}' | jq .levers
```

```bash
python -m servingkit pipeline --agent --turns 30 --fanout 4 -o scorecard.json
```

```
  resolve    ok        0.06s
  plan       ok        0.00s
  interact   ok        0.00s
  verify     ok       25.11s      <- compiles and runs the kernels the plan actually names
  drift      ok        0.04s
```

Stage 4 is why this is a pipeline rather than a report: a plan recommending cascade attention is a
claim about `13_prefix_attention.cu`, so the pipeline compiles that file and checks it against its
own double-precision reference before the claim ships. Stage 3 is a gate — an antagonistic pair
with no declared shared resource fails the run, because a lever lying about what it spends
produces advice that costs money.

[`deploy/`](../deploy/README.md) has the Dockerfiles, a compose stack, and Kubernetes manifests
including a fan-out Job that spreads the twenty-two kernels across a cluster with
`completionMode: Indexed`.

## Checks

```bash
python -m servingkit check          # package vs notebooks, ~16,000 comparisons
python tools/verify_console.py      # demo-page JavaScript vs the same models
python -m kernelbench eval kernels/ # the kernels the levers point at
```

Two edges, and a triangle that closes transitively: `servingkit check` pins the package against
the notebooks, `verify_console.py` pins the demo pages' JavaScript against the notebooks, so the
package and the browser agree too. Changing a number means changing it in one place and watching
the checks fail.

`pytest tests/` runs the same comparison plus the invariants worth stating outright — that a
refusal always names the missing property, that no stack holds two levers from one domain, that
every kernel a lever names exists, and that batching and speculation keep their precondition.

## Adding a lever

A dozen lines, and everything picks it up — applicability, exclusivity, interaction prediction,
the ladder, the report, the CLI.

```python
import servingkit as sk

sk.register(sk.Lever(
    "my trick",
    lambda c: {**c, "kv_bytes": c["kv_bytes"] * 0.75},
    domain="kv_format",
    spends=("accuracy", "memory"),
    requires=("long_context",),
    kernel="09_fp8_scaling.cu",
    note="What it does and what it costs. This string is shown to whoever runs the plan.",
))
```

If your lever measures as antagonistic against something and you did not declare a shared
resource, `python -m servingkit check` fails and tells you which pair. That is working as
intended.
