"""Compose levers over a workload, and be told what the composition costs.

A `Stack` is a workload plus an ordered list of levers. It does four things a bare list of
functions cannot:

  1. **Refuses what does not apply.** A lever whose `requires` the workload lacks is rejected
     with the missing property named, not silently applied to no effect.
  2. **Refuses what conflicts.** Two levers in the same domain are alternatives; adding the
     second replaces the first and says so.
  3. **Explains sub-multiplicative gains.** When two levers spend the same declared resource,
     the interaction report names the resource. The antagonism stops being a surprise.
  4. **Orders by money.** `ladder()` adds levers one at a time, cheapest-effort first, and
     reports cost per million tokens at each rung — because the question is rarely "what is
     fastest" and usually "what do I do on Monday".

The whole point is that adding a lever is a dozen lines in `levers.py` and everything here
picks it up: the applicability check, the exclusivity rule, the interaction prediction, the
ladder and the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .levers import LEVERS, Lever, shared_resources
from .serving import predict
from .step import base_config, evaluate
from .workload import Workload


@dataclass
class Rejection:
    """A lever that was asked for and not applied, with the reason."""
    lever: str
    reason: str
    detail: str = ""


@dataclass
class Stack:
    """A workload, the levers pulled on it, and what that composition does."""

    workload: Workload
    gpu: str = "H100 SXM"
    names: list[str] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    def __post_init__(self) -> None:
        # The workload carries the accelerator, because its properties depend on it. A Stack
        # built for a different card re-homes the workload rather than letting the two disagree
        # — a stack planning an H100 over a workload that thinks it is on a T4 would derive its
        # applicable levers from one card and its costs from the other.
        if self.workload.gpu != self.gpu:
            self.workload = replace(self.workload, gpu=self.gpu)

    # ------------------------------------------------------------------ construction
    def with_levers(self, *names: str) -> "Stack":
        """Add levers, dropping the ones that cannot apply and recording why."""
        s = Stack(self.workload, self.gpu, list(self.names), list(self.rejected))
        props = self.workload.properties
        for n in names:
            if n not in LEVERS:
                s.rejected.append(Rejection(n, "no such lever"))
                continue
            lv = LEVERS[n]
            missing = lv.missing(props)
            if missing:
                s.rejected.append(Rejection(
                    n, "workload lacks a required property", ", ".join(missing)))
                continue
            clash = next((m for m in s.names if LEVERS[m].domain == lv.domain), None)
            if clash:
                s.rejected.append(Rejection(
                    n, f"conflicts with {clash}", f"both set {lv.domain}"))
                continue
            s.names.append(n)
        return s

    def all_applicable(self) -> "Stack":
        """Every lever this workload can use, with domain conflicts resolved by declaration
        order. Useful as an upper bound, not as a recommendation."""
        return self.with_levers(*LEVERS)

    # ------------------------------------------------------------------------ costing
    @property
    def levers(self) -> list[Lever]:
        return [LEVERS[n] for n in self.names]

    def config(self) -> dict:
        cfg = base_config(model=_step_model(self.workload.model), gpu=_step_gpu(self.gpu),
                          batch=self.workload.batch, ctx=self.workload.ctx)
        for lv in self.levers:
            cfg = lv.apply(cfg)
        return cfg

    def step(self) -> dict:
        return evaluate(self.config())

    def factors(self) -> dict:
        """The composed effect of every lever on the terms `evaluate()` does not model.

        Multiplicative, because these are independent scalings of different quantities — and
        the one additive field, `host_sync_ms`, is additive because it is a time and not a
        ratio. Composing them here rather than inside `evaluate()` keeps the step model single
        and canonical.
        """
        f = dict(prefill=1.0, idle=1.0, attention=1.0, sampling=1.0, memory=1.0, gpu=1.0,
                 host_sync_ms=0.0)
        for lv in self.levers:
            g = lv.factors(self.workload)
            for k in ("prefill", "idle", "attention", "sampling", "memory", "gpu"):
                f[k] *= g[k]
            f["host_sync_ms"] += g["host_sync_ms"]
        return f

    def plan(self) -> dict:
        """Everything about this stack: the step, the run, the money, the binding constraint."""
        base_cfg = base_config(model=_step_model(self.workload.model), gpu=_step_gpu(self.gpu),
                               batch=self.workload.batch, ctx=self.workload.ctx)
        base_st = evaluate(base_cfg)
        st = evaluate(self.config())
        f = self.factors()
        w = self.workload

        # Re-cost the step with the levers' declared term factors folded in. `HOST_SYNC_MS` is
        # what an unoptimized grammar costs on the critical path; a lever with a negative
        # `host_sync_ms` removes it, and cannot remove more than exists.
        HOST_SYNC_MS = 1.2 if "structured_output" in w.properties else 0.0
        parts = dict(st["parts"])
        parts["attention"] *= f["attention"]
        parts["sampling"] *= f["sampling"]
        parts["host sync"] = max(0.0, HOST_SYNC_MS + f["host_sync_ms"]) / 1000.0
        step_s = sum(parts.values())

        cfg = self.config()
        a, k = cfg["spec_alpha"], cfg["spec_k"]
        per_seq = ((1 - a ** (k + 1)) / (1 - a)) if k and a < 1 else 1.0
        # The *config's* batch, not the workload's: a lever that doubles the batch has doubled
        # the number of sequences the step serves, and reading `w.batch` here made batching look
        # like a 2x slowdown instead of a wash.
        eff_batch = cfg["batch"]
        tokens_per_s = eff_batch * per_seq / step_s
        tpot_ms = step_s * 1000 / per_seq

        # Prefix caching sets prefill=0.0, meaning "only the tokens never seen before", which is
        # what agent_run's ideal branch computes. Anything between is a hit rate, so express it
        # as one rather than scaling a total — the two are not the same arithmetic.
        hit = w.prefix_hit_rate
        if f["prefill"] == 0.0:
            hit = max(hit, 0.95)      # a working prefix cache, not a perfect one
        run = w.run(**({"prefix_hit_rate": hit} if w.kind == "agent" else {}))

        tool_s = run.get("tool_s", 0.0) * f["idle"]
        decode_s = run["decode_tokens"] / max(tokens_per_s / max(eff_batch, 1), 1e-9)
        prefill_s = run["prefill_actual"] / 20000.0
        wall_s = prefill_s + decode_s + tool_s

        token_cost = (run["prefill_actual"] / 1e6 * w.price_in_per_mtok
                      + run["decode_tokens"] / 1e6 * w.price_out_per_mtok)
        # What the hardware costs for as long as this run holds a slot. A lever that halves the
        # wall clock and doubles the GPU count has not obviously helped, and a planner that
        # reports only latency cannot say so — which is what `gpu_factor` is for.
        from .catalog import GPUS
        gpus = f["gpu"]
        gpu_cost = GPUS[self.gpu]["usd"] * gpus * (wall_s / 3600.0) / max(eff_batch, 1)

        # Capacity: how many of these fit at once, after the weights.
        from .kv import kv_bytes_per_token
        from .catalog import MODELS, MEM_UTIL
        m = MODELS[w.model]
        kv_gb = kv_bytes_per_token(m) * run["final_context"] * f["memory"] / 1e9
        pool_gb = GPUS[self.gpu]["vram"] * gpus * MEM_UTIL - m["params"] * 2
        seats = pool_gb / kv_gb if kv_gb > 0 else float("inf")

        return dict(
            levers=list(self.names), rejected=list(self.rejected),
            step_ms=step_s * 1000, tokens_per_s=tokens_per_s, tpot_ms=tpot_ms,
            bottleneck=max(parts, key=parts.get), gemm_bound=st["gemm_bound"],
            spare_compute=st["spare_compute"], parts=parts,
            speedup=tokens_per_s / (base_cfg["batch"] / (sum(base_st["parts"].values())
                                                         + HOST_SYNC_MS / 1000.0)),
            batch=eff_batch,
            prefill_tokens=run["prefill_actual"], decode_tokens=run["decode_tokens"],
            prefill_s=prefill_s, decode_s=decode_s, tool_s=tool_s, wall_s=wall_s,
            slot_util=(prefill_s + decode_s) / wall_s if wall_s else 0.0,
            token_cost_usd=token_cost, gpu_cost_usd=gpu_cost,
            cost_usd=token_cost, total_cost_usd=token_cost + gpu_cost,
            gpus=gpus, kv_gb=kv_gb, concurrent=seats, effective_hit_rate=hit,
        )

    def deployment(self, precision: str = "fp16") -> dict:
        """The same stack costed as a deployment: GPUs needed, concurrency, $/M tokens."""
        cfg = self.config()
        return predict(self.gpu, self.workload.model, precision,
                       batch=self.workload.batch, ctx=self.workload.ctx,
                       prefix_hit=self.plan()["effective_hit_rate"],
                       spec_alpha=cfg["spec_alpha"], spec_k=cfg["spec_k"] or 4,
                       tp=cfg["tp"], kv_fp8=cfg["kv_bytes"] < 2.0)

    # ------------------------------------------------------------------ interactions
    def interactions(self) -> list[dict]:
        """Every pair in this stack: expected gain, actual gain, and the reason for the gap.

        `predicted_conflict` comes from the levers' own declarations — the resources they both
        spend — so the explanation is available before the measurement, and a measured
        antagonism with no shared resource is a signal that a declaration is wrong.
        """
        cfg0 = base_config(model=_step_model(self.workload.model), gpu=_step_gpu(self.gpu),
                           batch=self.workload.batch, ctx=self.workload.ctx)
        tps0 = evaluate(cfg0)["tokens_per_s"]

        def gain(ns):
            c = cfg0
            for n in ns:
                c = LEVERS[n].apply(c)
            return evaluate(c)["tokens_per_s"] / tps0

        out = []
        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                ga, gb, gab = gain([a]), gain([b]), gain([a, b])
                expected = ga * gb
                shared = shared_resources(a, b)
                out.append(dict(a=a, b=b, solo_a=ga, solo_b=gb, expected=expected,
                                actual=gab,
                                synergy=gab / expected if expected else float("nan"),
                                predicted_conflict=sorted(shared)))
        return sorted(out, key=lambda d: d["synergy"])

    # ------------------------------------------------------------------------ ladder
    def ladder(self, order: list[str] | None = None) -> list[dict]:
        """Add levers one at a time and report the running state at each rung.

        The order is the lesson, so it is explicit. With none given, the levers are sorted the
        way a team would reach for them: the ones that need no kernel work first, then the
        cheap kernel swaps, then the ones that cost accuracy.
        """
        names = order or _default_order(self.names)
        rows = [dict(label="baseline", **_row(Stack(self.workload, self.gpu).plan()))]
        prev = rows[0]
        s = Stack(self.workload, self.gpu)
        for n in names:
            s2 = s.with_levers(n)
            if n not in s2.names:
                continue                      # not applicable or conflicting; ladder skips it
            s = s2
            p = s.plan()
            row = dict(label=f"+ {n}", **_row(p))
            row["d_wall"] = p["wall_s"] / prev["wall_s"] - 1 if prev["wall_s"] else 0.0
            row["d_cost"] = p["cost_usd"] / prev["cost_usd"] - 1 if prev["cost_usd"] else 0.0
            row["kernel"] = LEVERS[n].kernel
            rows.append(row)
            prev = row
        return rows

    def offered(self) -> list[Lever]:
        """Levers this workload could use and this stack has not pulled."""
        props = self.workload.properties
        taken_domains = {LEVERS[n].domain for n in self.names}
        return [lv for n, lv in LEVERS.items()
                if n not in self.names and lv.applies_to(props)
                and lv.domain not in taken_domains]

    def unavailable(self) -> list[tuple[Lever, list[str]]]:
        """Levers that do not apply, and the property that would unlock each one.

        This is the most useful output for someone deciding what to *build* rather than what to
        turn on: it says which properties of a workload are worth engineering into existence.
        """
        props = self.workload.properties
        out = []
        for lv in LEVERS.values():
            missing = lv.missing(props)
            if missing:
                out.append((lv, missing))
        return out


# --------------------------------------------------------------------------------- helpers

def _step_gpu(name: str) -> str:
    """Map a catalog accelerator onto the nearest one the step budget knows about."""
    from .step import STEP_HW
    return name if name in STEP_HW else "H100 SXM"


def _step_model(name: str) -> str:
    """Map a catalog model onto the nearest one the step budget knows about."""
    from .step import STEP_MODELS
    if name in STEP_MODELS:
        return name
    from .catalog import MODELS
    want = MODELS[name]["params"]
    return min(STEP_MODELS, key=lambda n: abs(STEP_MODELS[n]["params"] / 1e9 - want))


def _row(p: dict) -> dict:
    return dict(tokens_per_s=p["tokens_per_s"], tpot_ms=p["tpot_ms"],
                prefill_tokens=p["prefill_tokens"], wall_s=p["wall_s"],
                cost_usd=p["total_cost_usd"], token_cost=p["token_cost_usd"],
                gpu_cost=p["gpu_cost_usd"], concurrent=p["concurrent"], gpus=p["gpus"],
                slot_util=p["slot_util"], bottleneck=p["bottleneck"],
                d_wall=0.0, d_cost=0.0, kernel=None)


# Effort ranking: config changes, then kernel swaps that cost nothing, then the ones that
# spend accuracy. Ties keep the registry's order, which is itself pedagogical.
_EFFORT = {
    "prefix caching": 0, "CUDA graphs": 0, "2x batch": 0, "ragged batching": 1,
    "bitset logit mask": 1, "chunked prefill": 2, "cascade attention": 2,
    "copy-on-write forking": 2, "agent-aware eviction": 2, "grammar on device": 3,
    "speculation": 3, "batch-invariant reduction": 3, "fp8 weights": 4, "fp8 KV cache": 4,
    "TP=2": 5, "int4 weights": 6, "KV eviction": 7,
}


def _default_order(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: (_EFFORT.get(n, 9), list(LEVERS).index(n)))


def recommend(workload: Workload, gpu: str = "H100 SXM") -> Stack:
    """The stack this workload should probably be running.

    Takes every applicable lever except the ones that spend accuracy, because that is a
    product decision rather than an engineering one and nothing here can make it for you.
    """
    s = Stack(workload, gpu)
    safe = [n for n, lv in LEVERS.items() if "accuracy" not in lv.spends]
    return s.with_levers(*_default_order(safe))


__all__ = ["Stack", "Rejection", "recommend"]
