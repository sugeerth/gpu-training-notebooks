"""The decode step, broken into the things that consume it.

`evaluate()` is the finest-grained model in the package: it costs one decode step as six
independent terms and reports which one is largest. Everything composable is built on it,
because an optimization is exactly "a transformation of the config that makes one of those
terms smaller" — and the reason gains do not multiply is that shrinking the largest term
promotes a different one.

Lifted verbatim from `The_Optimization_Stack.ipynb`.
"""
from __future__ import annotations

# Per-layer kernel count in an unfused decode step: the norms, projections, attention,
# residuals and MLP. It matters only for launch overhead, and only when graphs are off.
KERNELS_PER_LAYER = 14

BW_EFF, FLOP_EFF = 0.75, 0.60

# The step model needs a couple of fields the serving catalog does not carry, so it keeps its
# own view of the hardware. `link` is the interconnect bandwidth used for tensor-parallel
# all-reduces; `launch_us` is per-kernel launch latency.
STEP_HW: dict[str, dict] = {
    "T4":        dict(bw=0.32e12, tf=65e12,   link=None,  launch_us=2.2, vram=16),
    "A100 80GB": dict(bw=2.04e12, tf=312e12,  link=300e9, launch_us=1.5, vram=80),
    "H100 SXM":  dict(bw=3.35e12, tf=990e12,  link=900e9, launch_us=1.2, vram=80),
    "MI300X":    dict(bw=5.30e12, tf=1307e12, link=800e9, launch_us=1.5, vram=192),
}

STEP_MODELS: dict[str, dict] = {
    "Qwen2.5-0.5B":  dict(params=0.5e9, layers=24, hidden=896,  kv_heads=2, head_dim=64,
                          vocab=152000),
    "Llama-3.1-8B":  dict(params=8e9,   layers=32, hidden=4096, kv_heads=8, head_dim=128,
                          vocab=128000),
    "Llama-3.1-70B": dict(params=70e9,  layers=80, hidden=8192, kv_heads=8, head_dim=128,
                          vocab=128000),
}


def base_config(model: str = "Llama-3.1-8B", gpu: str = "H100 SXM", batch: int = 32,
                ctx: int = 2048) -> dict:
    """A plain fp16 decode configuration, with every optimization off.

    This is the thing levers transform. Keeping it a plain dict rather than a dataclass is
    deliberate: a lever is `dict -> dict`, which means anyone can write one in three lines
    without importing anything from here.
    """
    return dict(model=model, gpu=gpu, batch=batch, ctx=ctx,
                weight_bytes=2.0, kv_bytes=2.0, tp=1, graphs=True,
                spec_k=0, spec_alpha=0.0, kernel_eff=1.0, attn_ctx_frac=1.0)


def evaluate(cfg: dict) -> dict:
    """Cost one decode step. Returns the parts, the total, and what is binding.

    The six parts are independent and additive, except `GEMMs`, which is a max of its memory
    and compute times — that max is where "memory-bound" comes from, and `spare_compute`
    reports how much of the card's math capacity the losing side leaves idle. Every
    optimization that trades compute for bandwidth is spending exactly that.
    """
    m, g = STEP_MODELS[cfg["model"]], STEP_HW[cfg["gpu"]]
    bw = g["bw"] * BW_EFF * cfg["tp"]
    flops = g["tf"] * FLOP_EFF * cfg["tp"] * cfg["kernel_eff"]
    batch, k = cfg["batch"], cfg["spec_k"]

    # A verify pass covers (k+1) token positions per sequence: same weight read, more compute.
    positions = batch * (k + 1)
    gemm_mem = (m["params"] * cfg["weight_bytes"]) / bw
    gemm_cmp = (2 * m["params"] * positions) / flops
    gemm = max(gemm_mem, gemm_cmp)

    kv_per_tok = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * cfg["kv_bytes"]
    attention = (kv_per_tok * cfg["ctx"] * cfg["attn_ctx_frac"] * positions) / bw
    sampling = (positions * m["vocab"] * 4 * 3) / bw
    comm = 0.0
    if cfg["tp"] > 1 and g["link"]:
        comm = (2 * m["hidden"] * positions * 2 * (2 * (cfg["tp"] - 1) / cfg["tp"])
                * m["layers"] / g["link"])
    n_kernels = KERNELS_PER_LAYER * m["layers"]
    launch = (g["launch_us"] * 1e-6) * (3 if cfg["graphs"] else n_kernels)

    # A draft model costs a fraction of a target pass, k times over.
    draft = (k * 0.06 * gemm_mem) if k else 0.0

    parts = {"GEMMs": gemm, "attention": attention, "communication": comm,
             "sampling": sampling, "launch": launch, "draft": draft}
    step_s = sum(parts.values())

    # Tokens emitted per step: 1 normally; with speculation, the truncated geometric mean.
    a = cfg["spec_alpha"]
    tokens_per_seq = ((1 - a ** (k + 1)) / (1 - a)) if k and a < 1 else 1.0
    tps = batch * tokens_per_seq / step_s

    return {"parts": parts, "step_ms": step_s * 1000, "tokens_per_s": tps,
            "tpot_ms": step_s * 1000 / tokens_per_seq,
            "bottleneck": max(parts, key=parts.get),
            "gemm_bound": "compute" if gemm_cmp > gemm_mem else "memory",
            "spare_compute": 1 - min(1.0, gemm_cmp / max(gemm_mem, 1e-12))}


__all__ = ["base_config", "evaluate", "STEP_HW", "STEP_MODELS", "KERNELS_PER_LAYER"]
