# GPU Training Notebooks

Train an LLM, then actually serve it — for free. **39 self-contained notebooks** covering the full
lifecycle: distributed fine-tuning, DPO/GRPO alignment, LLM-as-judge evaluation, multimodal
training, and a complete **19-notebook serving track** (vLLM, quantization, speculative decoding,
observability, capacity planning, structured output, multi-LoRA, long context, MoE, RAG/agents,
production hardening, **NVIDIA vs AMD** hardware modeling, and **how the optimizations compose**).
Everything is sized for **Kaggle's free 2×T4** or **Colab's free T4**, with small open models
(Qwen2.5-0.5B class) — and **15 of the serving notebooks need no GPU at all**.

**Browse with one-click Colab links:** [sugeerth.github.io/gpu-training-notebooks](https://sugeerth.github.io/gpu-training-notebooks/)

## The learning path

### 1 · Start here (~30 min)

| Notebook | What you learn | Runs on |
|---|---|---|
| [Simple_MultiGPU_Training](Simple_MultiGPU_Training.ipynb) | The simplest possible distributed fine-tuning run | Kaggle 2×T4 |
| [Simple_MultiGPU_ActualTraining](Simple_MultiGPU_ActualTraining.ipynb) | Full-parameter training end-to-end, no LoRA | Kaggle 2×T4 |
| [Simple_MultiGPU_Benchmark](Simple_MultiGPU_Benchmark.ipynb) | 1 GPU vs 2 GPUs vs parallelism strategies, measured | Kaggle 2×T4 |

### 2 · Core fine-tuning

| Notebook | What you learn | Runs on |
|---|---|---|
| [LoRA_QLoRA_FineTuning](LoRA_QLoRA_FineTuning.ipynb) | LoRA vs QLoRA vs full fine-tuning, side by side | Kaggle 2×T4 |
| [Modern_Full_FineTuning_PostTraining](Modern_Full_FineTuning_PostTraining.ipynb) | Full-weight fine-tuning with the modern post-training stack | Kaggle 2×T4 |
| [Distributed_Training_DeepSpeed](Distributed_Training_DeepSpeed.ipynb) | DPO + LoRA accelerated by DeepSpeed and Ray | Kaggle 2×T4 |
| [Free_Distributed_SFT_DPO](Free_Distributed_SFT_DPO.ipynb) | The SFT → DPO pipeline entirely on free GPUs | Kaggle 2×T4 |
| [Colab_Pro_A100_Max_Training](Colab_Pro_A100_Max_Training.ipynb) | How big you can go on a single A100 40GB (7B) | Colab Pro A100 |

### 3 · Alignment

| Notebook | What you learn | Runs on |
|---|---|---|
| [PostTraining_Core_Understanding](PostTraining_Core_Understanding.ipynb) | Watch a raw LM become an assistant, stage by stage | Kaggle 2×T4 |
| [DPO_Training](DPO_Training.ipynb) | DPO from scratch — preferences without a reward model | Kaggle 2×T4 |
| [Simple_MultiGPU_DPO_RLHF](Simple_MultiGPU_DPO_RLHF.ipynb) | The full SFT → DPO → RLHF pipeline on a tiny model | Kaggle 2×T4 |
| [Simple_MultiGPU_Alignment_Showdown](Simple_MultiGPU_Alignment_Showdown.ipynb) | DPO vs KTO vs ORPO vs SimPO on the same data | Kaggle 2×T4 |
| [GRPO_Reasoning_Training](GRPO_Reasoning_Training.ipynb) | GRPO — teaching step-by-step reasoning with pure RL | Kaggle 2×T4 |

### 4 · Evaluation

| Notebook | What you learn | Runs on |
|---|---|---|
| [LLM_as_Judge_Evaluation](LLM_as_Judge_Evaluation.ipynb) | Score model outputs with a judge LLM, self-contained | Kaggle 2×T4 |
| [Multi_Trace_Agent_Evaluation](Multi_Trace_Agent_Evaluation.ipynb) | Evaluate agent reasoning across multiple traces, with retries | Kaggle 2×T4 |

### 5 · Beyond text

| Notebook | What you learn | Runs on |
|---|---|---|
| [Simple_MultiGPU_Multimodal](Simple_MultiGPU_Multimodal.ipynb) | Fine-tune a vision-language model on image+text | Kaggle 2×T4 |
| [Multimodal_LoRA_QLoRA_DPO](Multimodal_LoRA_QLoRA_DPO.ipynb) | One VLM fine-tuned three ways: LoRA, QLoRA, DPO | Kaggle 2×T4 |
| [Simple_MultiGPU_Diffusion](Simple_MultiGPU_Diffusion.ipynb) | Teach Stable Diffusion a new concept with LoRA | Kaggle 2×T4 |
| [Simple_MultiGPU_ImageClassification](Simple_MultiGPU_ImageClassification.ipynb) | Vision Transformer classification, distributed | Kaggle 2×T4 |
| [Simple_MultiGPU_Audio](Simple_MultiGPU_Audio.ipynb) | Fine-tune Whisper for speech recognition | Kaggle 2×T4 |

### 6 · The serving track (19 notebooks)

You trained it — now serve it. This track builds the modern serving stack from first principles to
the 2025 frontier: every mechanism is **measured**, **visualized**, or **simulated**, never asserted.
Six of the ten need **no GPU at all** — they run on Colab's free CPU runtime.

**Understand the machine** — what a serving engine is and why it's built that way:

| Notebook | What you learn | Runs on |
|---|---|---|
| [Serving_Fundamentals_KV_Cache_Batching](Serving_Fundamentals_KV_Cache_Batching.ipynb) | KV cache math, prefill vs decode, TTFT/TPOT, continuous batching — measured | Colab T4 |
| [vLLM_High_Throughput_Serving](vLLM_High_Throughput_Serving.ipynb) | PagedAttention, prefix caching, and a real OpenAI-compatible server under load | Colab T4 |
| [Serving_Internals_Visualized_D3](Serving_Internals_Visualized_D3.ipynb) | **Interactive D3**: KV explorer, animated batching, block pool, speculation strip | CPU ✨ |

**Make it faster** — the three optimizations that matter most:

| Notebook | What you learn | Runs on |
|---|---|---|
| [Quantized_Serving_Showdown](Quantized_Serving_Showdown.ipynb) | FP16 vs AWQ vs GPTQ vs NF4: memory, speed, accuracy, benchmarked head-to-head | Colab T4 |
| [Speculative_Decoding_Advanced_Serving](Speculative_Decoding_Advanced_Serving.ipynb) | Speculative + prompt-lookup decoding, EAGLE/MTP, and the disaggregation frontier | Colab T4 |
| [Structured_Output_Guided_Decoding](Structured_Output_Guided_Decoding.ipynb) | Build a token-masking FSM from scratch; make invalid JSON unrepresentable | CPU ✨ |

**Run it in production** — the parts nobody teaches:

| Notebook | What you learn | Runs on |
|---|---|---|
| [Serving_Logs_Observability](Serving_Logs_Observability.ipynb) | **Read vLLM's logs line by line**; percentiles from histogram buckets; an animated incident | CPU ✨ |
| [Serving_Benchmark_Capacity_Planning](Serving_Benchmark_Capacity_Planning.ipynb) | Open-loop load testing, the latency knee, goodput, and $ per million tokens | CPU ✨ |
| [Distributed_MultiReplica_Serving](Distributed_MultiReplica_Serving.ipynb) | TP vs replicas, the communication tax, and prefix-aware routing | CPU ✨ |
| [MultiLoRA_Serving_At_Scale](MultiLoRA_Serving_At_Scale.ipynb) | 50 tenants on one GPU — batched multi-LoRA, `max_loras`, and the economics | CPU ✨ |

**Pick the hardware** — the same model behaves differently on every GPU, across both vendors:

| Notebook | What you learn | Runs on |
|---|---|---|
| [Hardware_Roofline_NVIDIA_vs_AMD](Hardware_Roofline_NVIDIA_vs_AMD.ipynb) | The roofline derived for LLM inference: the batch size where decode stops being memory-bound, why VRAM decides your topology, and a **portable CUDA/ROCm microbenchmark** | CPU ✨ |
| [Portable_Kernels_Precision_Matrix](Portable_Kernels_Precision_Matrix.ipynb) | CUDA vs HIP vs Triton, the **precision × architecture support matrix**, and one Triton kernel that runs on both vendors | CPU ✨ |
| [Serving_WhatIf_Console](Serving_WhatIf_Console.ipynb) | Every equation consolidated into an **interactive what-if console**, with tornado sensitivity, a Pareto frontier, and honest error bars | CPU ✨ |

**Handle the hard workloads** — where the standard recipe stops working:

| Notebook | What you learn | Runs on |
|---|---|---|
| [LongContext_KV_Compression_Serving](LongContext_KV_Compression_Serving.ipynb) | The KV wall at 128k+, sliding-window/hybrid/MLA architectures, FP8 KV, and an **eviction simulator** (attention sinks, heavy hitters) that shows what each policy throws away | CPU ✨ |
| [MoE_Serving_Expert_Parallelism](MoE_Serving_Expert_Parallelism.ipynb) | Total vs active params, why **MoE decode is *more* memory-bound** than dense, all-to-all traffic, and the routing-imbalance straggler that sets your step time | CPU ✨ |
| [RAG_Agent_Serving_Patterns](RAG_Agent_Serving_Patterns.ipynb) | Quadratic agent prefill, the **prompt-layout rule** that decides your hit rate, the cache hierarchy, cascades, and semantic caching's sharp edge | CPU ✨ |
| [Production_Hardening_Reliability](Production_Hardening_Reliability.ipynb) | The **cancellation leak**, bounded queues vs 429s, per-tenant fairness, graceful drain, a failure taxonomy, and a **chaos drill** | CPU ✨ |

**Zoom all the way in, then all the way out** — the two notebooks that make the rest cohere:

| Notebook | What you learn | Runs on |
|---|---|---|
| [Anatomy_Of_A_Decode_Step](Anatomy_Of_A_Decode_Step.ipynb) | Where the milliseconds actually go in **one decode step** — GEMMs, attention, sampling, launch overhead — an interactive budget you can re-proportion, a map of **which optimization cuts which slice**, and a portable CUDA/ROCm profiler | CPU ✨ |
| [The_Optimization_Stack](The_Optimization_Stack.ipynb) | **Why gains don't multiply.** A *computed* interaction matrix (which optimizations fight, which unlock each other), bottleneck migration as you stack, the waterfall vs the brochure, and a search for the best stack under an accuracy-risk and effort budget | CPU ✨ |

> ✨ = no GPU required. The GPU-only sections in these notebooks are gated and skip cleanly on CPU.
> The hardware notebooks cover **T4 → B200** and **MI210 → MI355X**, and their GPU cells run on
> **CUDA *or* ROCm** unchanged.

## Running and verifying the notebooks

**One line to execute the GPU notebooks on Colab.** Open any Colab notebook, set
*Runtime → Change runtime type → T4 GPU*, and paste:

```python
!git clone -q https://github.com/sugeerth/gpu-training-notebooks.git && cd gpu-training-notebooks && pip install -q nbclient nbformat ipykernel && python tools/run_notebooks.py --set gpu
```

It executes every GPU-dependent notebook headlessly and writes `notebook_run_report.md`
(plus `.json`) containing per-notebook pass/fail, timings, the failing cell and traceback for
anything broken, and the measured output of every cell — ready to paste into an issue or PR.

```
python tools/run_notebooks.py --set gpu        # the 7 notebooks with live-GPU sections
python tools/run_notebooks.py --set cpu        # the 10 that need no GPU
python tools/run_notebooks.py --only vLLM_High_Throughput_Serving.ipynb
python tools/run_notebooks.py --keep-going     # report every failing cell, not just the first
python tools/run_notebooks.py --skip-installs  # deps already provisioned
```

Notebooks that deliberately refuse to run without a GPU are reported as **⏭️ needs GPU**, not as
failures — so a CPU run still tells you something useful.

**Static checks** (no execution, no GPU, no network) — nbformat validity, Python syntax of every
cell, nav-chain integrity, and link resolution:

```
python tools/validate_notebooks.py
```

Both run automatically on every push via
[`.github/workflows/validate-notebooks.yml`](.github/workflows/validate-notebooks.yml).

## Field notes (the mistakes these notebooks already made for you)

- **Keep every line of code in notebook cells.** Kaggle kernels cannot import local `.py` files —
  anything that matters is inlined, which is why the notebooks are self-contained.
- **Launch with `accelerate launch`, not `notebook_launcher`.** `notebook_launcher` forks after
  CUDA is initialized and dies with cryptic CUDA re-init errors on Kaggle — 11 of these notebooks
  use the `accelerate launch` pattern instead, none use `notebook_launcher`.

## Support files

`dpo_train.py` + `dpo_results.json` — a standalone-script variant of the DPO run and its saved
metrics, referenced by [PostTraining_Core_Understanding](PostTraining_Core_Understanding.ipynb).
`requirements.txt` covers local runs; on Kaggle/Colab each notebook installs what it needs in its
first cell.
