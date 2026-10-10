"""Hardware and model catalogs — the only copy.

Before this module existed, these tables were pasted into eight notebooks and four demo
pages, and `tools/audit_consistency.py` existed to notice when the copies drifted. They are
here now, once, and everything else imports them.

Every number is a vendor specification or a list price, not a measurement. That distinction
matters enough that the fields are named for it: `bw` is the spec sheet's bandwidth, and the
`*_EFF` fractions below are what a good kernel actually reaches. Measure your own and
substitute them — `servingkit.recalibrate()` exists for exactly that.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- accelerators
#
# bw    : HBM bandwidth, TB/s, spec sheet
# tf16  : dense fp16/bf16 tensor-core TFLOP/s
# tf8   : dense fp8 TFLOP/s, or None where there is no fp8 hardware path
# usd   : on-demand list price per GPU-hour, mid-2025, rounded
# int4  : kernel-quality factor for weight-only int4 — how close the *available* kernels get
#         to the ideal. 1.0 would be perfect; the gap is a software fact, not a silicon one.
GPUS: dict[str, dict] = {
    "T4":        dict(vendor="NVIDIA", vram=16,  bw=0.32, tf16=65,   tf8=None, usd=0.35, int4=0.55),
    "L4":        dict(vendor="NVIDIA", vram=24,  bw=0.30, tf16=121,  tf8=242,  usd=0.70, int4=0.90),
    "A10G":      dict(vendor="NVIDIA", vram=24,  bw=0.60, tf16=125,  tf8=None, usd=1.00, int4=0.90),
    "A100 80GB": dict(vendor="NVIDIA", vram=80,  bw=2.04, tf16=312,  tf8=None, usd=2.20, int4=0.95),
    "H100 SXM":  dict(vendor="NVIDIA", vram=80,  bw=3.35, tf16=990,  tf8=1979, usd=3.50, int4=0.95),
    "H200 SXM":  dict(vendor="NVIDIA", vram=141, bw=4.80, tf16=990,  tf8=1979, usd=4.50, int4=0.95),
    "B200":      dict(vendor="NVIDIA", vram=192, bw=8.00, tf16=2250, tf8=4500, usd=7.00, int4=0.95),
    "MI210":     dict(vendor="AMD",    vram=64,  bw=1.60, tf16=181,  tf8=None, usd=1.20, int4=0.45),
    "MI250X":    dict(vendor="AMD",    vram=128, bw=3.20, tf16=383,  tf8=None, usd=2.20, int4=0.45),
    "MI300X":    dict(vendor="AMD",    vram=192, bw=5.30, tf16=1307, tf8=2615, usd=3.50, int4=0.70),
    "MI325X":    dict(vendor="AMD",    vram=256, bw=6.00, tf16=1307, tf8=2615, usd=4.20, int4=0.70),
    "MI355X":    dict(vendor="AMD",    vram=288, bw=8.00, tf16=2300, tf8=4600, usd=7.00, int4=0.75),
}

# --------------------------------------------------------------------------------- models
#
# params in billions; `kv_heads` and `head_dim` set the KV cache, which is the number that
# decides how many concurrent requests fit. `scheme` is what the attention actually stores:
#
#   full    every layer keeps every token          (MHA, GQA, MQA)
#   swa     every layer keeps a sliding window
#   hybrid  some layers global, the rest windowed
#   mla     one latent vector per layer per token  (DeepSeek)
MODELS: dict[str, dict] = {
    "Qwen2.5-0.5B":   dict(params=0.5,  layers=24, kv_heads=2,  head_dim=64,  hidden=896,
                           heads=14, vocab=152000, scheme="full"),
    "Mistral-7B":     dict(params=7.0,  layers=32, kv_heads=8,  head_dim=128, hidden=4096,
                           heads=32, vocab=32000, scheme="swa", window=4096),
    "Llama-3.1-8B":   dict(params=8.0,  layers=32, kv_heads=8,  head_dim=128, hidden=4096,
                           heads=32, vocab=128000, scheme="full"),
    "Gemma-2-27B":    dict(params=27.0, layers=46, kv_heads=16, head_dim=128, hidden=4608,
                           heads=32, vocab=256000, scheme="hybrid", window=4096,
                           global_every=2),
    "Qwen2.5-32B":    dict(params=32.0, layers=64, kv_heads=8,  head_dim=128, hidden=5120,
                           heads=40, vocab=152000, scheme="full"),
    "Llama-3.1-70B":  dict(params=70.0, layers=80, kv_heads=8,  head_dim=128, hidden=8192,
                           heads=64, vocab=128000, scheme="full"),
    "DeepSeek-V3":    dict(params=671.0, layers=61, kv_heads=None, head_dim=None, hidden=7168,
                           heads=128, vocab=129280, scheme="mla", latent=512 + 64, active=37),
    "GPT-3-175B":     dict(params=175.0, layers=96, kv_heads=96, head_dim=128, hidden=12288,
                           heads=96, vocab=50257, scheme="full"),
}

PRECISION: dict[str, dict] = {
    "fp16": dict(bytes=2.0, needs_fp8=False, int4=False),
    "fp8":  dict(bytes=1.0, needs_fp8=True,  int4=False),
    "int4": dict(bytes=0.5, needs_fp8=False, int4=True),
}

# ------------------------------------------------------------------- achievable fractions
#
# These are the three numbers that turn a spec sheet into a prediction, and they are the three
# most worth replacing with your own. A good memory-bound kernel reaches ~75% of peak
# bandwidth; a good GEMM reaches ~60% of peak FLOP/s; an allocator that leaves no headroom
# will OOM, so 90% of VRAM is the usable ceiling.
BW_EFF = 0.75
FLOP_EFF = 0.60
MEM_UTIL = 0.90

# A KV pool too small to hold this many concurrent requests is not a deployment, it is a
# demo. `predict()` shards until the pool clears this bar.
MIN_CONCURRENCY = 32

# Fixed per-decode-step cost of an efficient engine: graph replay, scheduling, sampling glue.
# On a small model at low batch this single number is the bottleneck, which is the whole
# reason it is here rather than assumed to be zero.
ENGINE_OVERHEAD_MS = 0.8


def recalibrate(gpu: str, *, measured_bw_tbs: float | None = None,
                measured_gemm_tf: float | None = None) -> dict:
    """Return a copy of a GPU entry with spec numbers replaced by measured ones.

    Use it when you have run `kernels/01_copy` and `kernels/03_sgemm` on the card you
    actually have:

        >>> import servingkit as sk
        >>> real = sk.recalibrate("H100 SXM", measured_bw_tbs=2.6, measured_gemm_tf=620)
        >>> sk.GPUS["my H100"] = real

    Predictions made against a spec sheet are a planning exercise. Predictions made against
    your own measurements are an estimate. The difference is usually 20-30%.
    """
    g = dict(GPUS[gpu])
    if measured_bw_tbs is not None:
        g["bw"] = measured_bw_tbs / BW_EFF       # store as if it were a spec number
    if measured_gemm_tf is not None:
        g["tf16"] = measured_gemm_tf / FLOP_EFF
    g["measured"] = True
    return g


__all__ = ["GPUS", "MODELS", "PRECISION", "BW_EFF", "FLOP_EFF", "MEM_UTIL",
           "MIN_CONCURRENCY", "ENGINE_OVERHEAD_MS", "recalibrate"]
