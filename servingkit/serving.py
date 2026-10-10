"""Deployment-level prediction: throughput, latency, capacity and price.

`evaluate()` in `step.py` costs one step on one GPU. `predict()` answers the question a person
actually has — *can I serve this, how fast, and what does it cost per million tokens* — which
means it also has to decide how many GPUs the thing needs, and refuse configurations that
cannot run.

Returning an `{"error": ...}` dict rather than raising is deliberate: "this does not fit" is a
legitimate answer to a planning question, and the sweeps in `stack.py` need to distinguish
*infeasible* from *slow*.

Lifted verbatim from `Serving_WhatIf_Console.ipynb`.
"""
from __future__ import annotations

from .catalog import (BW_EFF, ENGINE_OVERHEAD_MS, FLOP_EFF, GPUS, MEM_UTIL, MIN_CONCURRENCY,
                      MODELS, PRECISION)


def predict(gpu_name: str, model_name: str, precision: str = "fp16", batch: int = 32,
            ctx: int = 2048, prefix_hit: float = 0.0, spec_alpha: float = 0.0,
            spec_k: int = 4, spec_cost: float = 0.05, tp: int | None = None,
            kv_fp8: bool = False, overhead_ms: float = ENGINE_OVERHEAD_MS) -> dict:
    """Predict serving performance for one configuration.

    Returns a dict with an "error" key if the configuration cannot run at all.
    """
    g, m, p = GPUS[gpu_name], MODELS[model_name], PRECISION[precision]
    if p["needs_fp8"] and not g["tf8"]:
        return {"error": f"{gpu_name} has no FP8 hardware - would be emulated"}
    if m.get("scheme") == "mla" or m.get("kv_heads") is None:
        return {"error": f"{model_name} needs the MLA path; use servingkit.kv for its cache"}

    weights_gb = m["params"] * p["bytes"]
    kv_bytes = 1 if kv_fp8 else 2
    kv_per_tok = 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * kv_bytes
    # A deployment needs weights AND a KV pool big enough to be worth running:
    kv_needed_gb = kv_per_tok * ctx * MIN_CONCURRENCY / 1e9

    # --- topology: shard until weights + a USABLE KV pool fit ------------------------------
    # This is why long context forces wide tensor parallelism: the pool requirement grows with
    # ctx while the weights do not, so past some context length you are buying GPUs for cache.
    if tp is None:
        tp = 1
        while tp < 16 and (weights_gb + kv_needed_gb) / tp > g["vram"] * MEM_UTIL:
            tp *= 2
    if (weights_gb + kv_needed_gb) / tp > g["vram"] * MEM_UTIL:
        return {"error": f"needs >16 GPUs to hold weights + a {MIN_CONCURRENCY}-request "
                         f"KV pool at {ctx} ctx"}
    tp_eff = 1.0 / (1.0 / tp + 0.06 * (tp - 1) / tp) if tp > 1 else 1.0

    # --- KV pool ---------------------------------------------------------------------------
    pool_gb = g["vram"] * tp * MEM_UTIL - weights_gb - 1.5 * tp
    max_conc = max(0, int(pool_gb * 1e9 // (kv_per_tok * ctx)))
    eff_batch = min(batch, max_conc) if max_conc else 0
    if eff_batch == 0:
        return {"error": f"KV pool cannot hold even one {ctx}-token request"}

    # --- decode: the roofline PLUS the engine's fixed per-step cost -------------------------
    bw = g["bw"] * 1e12 * BW_EFF * tp * tp_eff
    peak = (g["tf8"] if precision == "fp8" else g["tf16"]) * 1e12 * FLOP_EFF * tp * tp_eff
    kernel = g["int4"] if p["int4"] else 1.0
    mem_step_s = (m["params"] * 1e9 * p["bytes"]) / bw / kernel
    comp_step_s = (2 * m["params"] * 1e9 * eff_batch) / peak
    # One decode step serves the whole batch; overhead is per step, not per request.
    step_s = max(mem_step_s, comp_step_s) + overhead_ms / 1000
    decode_tps = eff_batch / step_s
    bound = ("overhead" if overhead_ms / 1000 > max(mem_step_s, comp_step_s)
             else "memory" if mem_step_s > comp_step_s else "compute")

    # --- speculation -----------------------------------------------------------------------
    spec_mult = 1.0
    if spec_alpha > 0:
        a, k = spec_alpha, spec_k
        spec_mult = ((1 - a ** (k + 1)) / (1 - a)) / (k * spec_cost + 1)
        # Speculation spends spare compute. A compute-bound step has none to spend, so most of
        # the theoretical win does not arrive — which is why this is gated on `bound`.
        spec_mult = max(1.0, spec_mult * (1.0 if bound == "memory" else 0.4))
    decode_tps *= spec_mult

    per_req_tps = decode_tps / eff_batch
    tpot_ms = 1000 / max(per_req_tps, 1e-9)

    # --- prefill ---------------------------------------------------------------------------
    prefill_tps = peak / (2 * m["params"] * 1e9)
    ttft_s = ctx * (1 - prefix_hit) / max(prefill_tps, 1e-9)

    gpus_used = tp
    cost_hr = g["usd"] * gpus_used
    cpm = (cost_hr / 3600) / max(decode_tps, 1e-9) * 1e6

    return {"gpu": gpu_name, "vendor": g["vendor"], "model": model_name, "precision": precision,
            "tp": tp, "gpus": gpus_used, "batch": eff_batch, "max_concurrency": max_conc,
            "decode_tps": decode_tps, "per_req_tps": per_req_tps, "tpot_ms": tpot_ms,
            "ttft_ms": ttft_s * 1000, "prefill_tps": prefill_tps, "bound": bound,
            "spec_mult": spec_mult, "kv_pool_gb": pool_gb, "usd_hr": cost_hr, "cpm": cpm}


def cheapest(model_name: str, *, batch: int = 32, ctx: int = 2048,
             min_tpot_ms: float | None = None, **kw) -> list[dict]:
    """Every feasible (GPU, precision) for a model, cheapest per million tokens first.

    The point of sorting by `cpm` rather than by throughput: the fastest configuration is
    almost never the cheapest one, and which you want depends on whether you are latency-bound
    or margin-bound. Pass `min_tpot_ms` to keep only the ones fast enough to be usable.
    """
    out = []
    for gpu in GPUS:
        for prec in PRECISION:
            r = predict(gpu, model_name, prec, batch=batch, ctx=ctx, **kw)
            if "error" in r:
                continue
            if min_tpot_ms is not None and r["tpot_ms"] > min_tpot_ms:
                continue
            out.append(r)
    return sorted(out, key=lambda r: r["cpm"])


__all__ = ["predict", "cheapest"]
