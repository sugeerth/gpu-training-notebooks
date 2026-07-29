# GPU Training Notebooks

Multi-GPU LLM training you can actually run for free. 20 self-contained notebooks that go from
"first distributed run" to DPO/GRPO alignment, LLM-as-judge evaluation, and multimodal training —
all sized for **Kaggle's free 2×T4** (or Colab), with small open models (Qwen2.5-0.5B class).

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
