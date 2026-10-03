"""What a KV cache costs, by attention architecture.

The single most useful number in serving, and the one most often computed wrong — usually by
forgetting that a sliding-window layer never stores more than its window, or that MLA stores
one latent vector per layer rather than K and V per head.

Lifted verbatim from `LongContext_KV_Compression_Serving.ipynb`, which remains the place the
reasoning is explained. `tests/test_servingkit.py` asserts the two have not drifted.
"""
from __future__ import annotations

from .catalog import MEM_UTIL, MODELS


def kv_bytes_per_token(m: dict, kv_bytes: int = 2) -> float:
    """Bytes of KV cache one token occupies, across all layers.

    `kv_bytes` is the element size: 2 for fp16, 1 for fp8 KV.
    """
    if m["scheme"] == "mla":
        return m["layers"] * m["latent"] * kv_bytes
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * kv_bytes


def kv_total_bytes(m: dict, ctx: int, kv_bytes: int = 2) -> float:
    """Bytes of KV cache one sequence of `ctx` tokens occupies.

    Not simply `ctx * kv_bytes_per_token` — that is only true for `scheme="full"`. A
    sliding-window layer stores `min(ctx, window)` tokens however long the sequence is, which
    is the entire point of sliding-window attention and the reason Mistral's cache stops
    growing at 4k.
    """
    per_layer = 2 * (m.get("kv_heads") or 0) * (m.get("head_dim") or 0) * kv_bytes
    if m["scheme"] == "full":
        return per_layer * m["layers"] * ctx
    if m["scheme"] == "mla":
        return m["layers"] * m["latent"] * kv_bytes * ctx
    if m["scheme"] == "swa":
        return per_layer * m["layers"] * min(ctx, m["window"])
    if m["scheme"] == "hybrid":
        n_global = m["layers"] // m["global_every"]
        n_local = m["layers"] - n_global
        return per_layer * (n_global * ctx + n_local * min(ctx, m["window"]))
    raise ValueError(f"unknown attention scheme {m['scheme']!r}")


def capacity(model: str | dict, ctx: int, vram_gb: float, *, kv_bytes: int = 2,
             weight_bytes: float = 2.0, util: float = MEM_UTIL) -> dict:
    """How many concurrent `ctx`-token sequences fit, after the weights are in.

    Returns the pool size, the concurrency, and which of the two is binding — because
    "how many users fit" and "does the model fit" are different questions with different
    fixes, and conflating them is how a deployment ends up with a 70B model and room for
    three conversations.
    """
    m = MODELS[model] if isinstance(model, str) else model
    weights_gb = m["params"] * weight_bytes
    budget_gb = vram_gb * util
    pool_gb = budget_gb - weights_gb
    per_seq = kv_total_bytes(m, ctx, kv_bytes)
    if pool_gb <= 0:
        return dict(weights_gb=weights_gb, pool_gb=0.0, per_seq_gb=per_seq / 1e9,
                    concurrency=0, binding="weights do not fit")
    conc = int(pool_gb * 1e9 // per_seq)
    return dict(weights_gb=weights_gb, pool_gb=pool_gb, per_seq_gb=per_seq / 1e9,
                concurrency=conc,
                binding="weights" if weights_gb > pool_gb else "KV pool")


__all__ = ["kv_bytes_per_token", "kv_total_bytes", "capacity"]
