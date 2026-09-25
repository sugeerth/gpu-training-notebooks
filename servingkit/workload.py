"""A workload declares what it is, and that decides which levers are even on the table.

This is the piece the repository was missing. Every notebook implicitly assumed a workload —
chat in most of them, an agent loop in one — and the reader had to work out which advice
transferred. A `Workload` says so explicitly: it carries its shape, and derives the set of
properties that levers test against.

Three constructors cover almost everything people serve:

    Workload.chat(...)    a person types, the model answers, context grows slowly
    Workload.agent(...)   a loop re-sends everything, emits a tool call, waits on a tool
    Workload.batch(...)   no one is waiting; throughput is the only objective

The properties are *derived*, not declared, so they cannot drift from the numbers. A chat
workload with a 200k context gets `long_context` automatically; an agent with `fanout=1` does
not get `shared_prefix`, because it has no branches to share one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agents import agent_run, padding_waste
from .catalog import MODELS
from .kv import kv_bytes_per_token


@dataclass
class Workload:
    """What you are serving, in the terms that decide which optimizations apply."""

    kind: str                      # "chat" | "agent" | "batch"
    model: str = "Llama-3.1-8B"
    ctx: int = 4096                # typical context at steady state
    batch: int = 32                # concurrent sequences
    output_tokens: int = 400       # tokens generated per request or per turn

    # agent shape — ignored for chat and batch
    turns: int = 1
    system_tokens: int = 0
    tool_def_tokens: int = 0
    tool_result_tokens: int = 0
    tool_latency_s: float = 0.0
    fanout: int = 1
    prefix_hit_rate: float = 0.0

    # promises the deployment makes
    structured_output: bool = False
    replayed: bool = False
    interactive: bool = True       # is a person waiting on the first token?

    # prices, for the money column
    price_in_per_mtok: float = 3.0
    price_out_per_mtok: float = 15.0

    # ----------------------------------------------------------------- constructors
    @classmethod
    def chat(cls, model: str = "Llama-3.1-8B", ctx: int = 4096, batch: int = 32,
             output_tokens: int = 400, **kw) -> "Workload":
        return cls(kind="chat", model=model, ctx=ctx, batch=batch,
                   output_tokens=output_tokens, turns=1, interactive=True, **kw)

    @classmethod
    def agent(cls, model: str = "Llama-3.1-8B", turns: int = 20, system_tokens: int = 2000,
              tool_def_tokens: int = 1500, tool_result_tokens: int = 800,
              output_tokens: int = 150, tool_latency_s: float = 2.0, fanout: int = 1,
              prefix_hit_rate: float = 0.0, batch: int = 32, **kw) -> "Workload":
        """An agent loop. `ctx` is derived from the turn count, not passed."""
        base = system_tokens + tool_def_tokens + 200
        ctx = base + (turns - 1) * (output_tokens + tool_result_tokens)
        kw.setdefault("structured_output", True)   # an agent's output is a tool call
        kw.setdefault("interactive", False)        # the caller is a loop
        return cls(kind="agent", model=model, ctx=ctx, batch=batch,
                   output_tokens=output_tokens, turns=turns, system_tokens=system_tokens,
                   tool_def_tokens=tool_def_tokens, tool_result_tokens=tool_result_tokens,
                   tool_latency_s=tool_latency_s, fanout=fanout,
                   prefix_hit_rate=prefix_hit_rate, **kw)

    @classmethod
    def batch(cls, model: str = "Llama-3.1-8B", ctx: int = 8192, batch: int = 256,
              output_tokens: int = 1000, **kw) -> "Workload":
        """Offline: nobody is waiting, so latency levers are worth nothing and batch is free."""
        return cls(kind="batch", model=model, ctx=ctx, batch=batch,
                   output_tokens=output_tokens, turns=1, interactive=False, **kw)

    # ------------------------------------------------------------------- derivation
    @property
    def properties(self) -> set[str]:
        """The property set levers are tested against. Derived, so it cannot drift."""
        p: set[str] = set()
        m = MODELS[self.model]

        # KV dominating memory is what "long context" actually means, so measure it rather
        # than picking a token threshold: weights vs the cache for one batch.
        kv_gb = kv_bytes_per_token(m) * self.ctx * self.batch / 1e9
        if kv_gb > m["params"] * 2 * 0.5:
            p.add("long_context")

        if self.structured_output:
            p.add("structured_output")
        if self.replayed:
            p.add("replayed")

        if self.kind == "agent":
            p |= {"repeated_context", "idle_holds", "predictable_output"}
            if self.turns > 2:
                p.add("ragged_batch")           # a batch is a mix of turn numbers
            if self.tool_result_tokens >= 256:
                p.add("mid_batch_prefill")      # results arrive big enough to stall a step
            if self.fanout > 1:
                p |= {"shared_prefix", "forks"}
        elif self.kind == "batch":
            p.add("ragged_batch")
            if self.turns > 1:
                p.add("repeated_context")

        # Spare compute is a property of the *step*, not of the workload's intent, so it is
        # measured rather than guessed from the batch size. A batch of 32 at 4k context leaves
        # most of the card's math idle; the same batch at 64k does not, because attention has
        # taken over. Deriving this from `batch < 128` got that backwards, and the symptom was
        # that speculation was offered to a long-context agent it would have slowed down.
        if self.spare_compute() > 0.25:
            p.add("spare_compute")

        # Speculation needs more than idle math: it amortizes ONE weight read over k+1 token
        # positions, and it multiplies the attention term by k+1 in the process. So it pays
        # only where the weight read dominates. At 66k of context attention is 17x the GEMM
        # and speculation makes the step slower — which the first version of this file got
        # wrong, because it gated speculation on `batch < 128` and offered it to exactly the
        # workload it would have harmed.
        if self.bottleneck() == "GEMMs":
            p.add("weight_bound")
        return p

    def step(self, gpu: str = "H100 SXM") -> dict:
        """This workload's decode step, before any lever touches it."""
        from .stack import _step_model
        from .step import base_config, evaluate
        return evaluate(base_config(model=_step_model(self.model), gpu=gpu,
                                    batch=self.batch, ctx=self.ctx))

    def spare_compute(self, gpu: str = "H100 SXM") -> float:
        """Fraction of the card's math capacity the GEMMs leave idle."""
        return self.step(gpu)["spare_compute"]

    def bottleneck(self, gpu: str = "H100 SXM") -> str:
        """The largest term of the decode step. The only honest basis for advice."""
        return self.step(gpu)["bottleneck"]

    # ------------------------------------------------------------------------ costs
    def run(self, **over) -> dict:
        """Cost this workload's whole run. Agents get the loop model; others get one pass."""
        if self.kind == "agent":
            kw = dict(turns=self.turns, system_tokens=self.system_tokens,
                      tool_def_tokens=self.tool_def_tokens, user_tokens=200,
                      response_tokens=self.output_tokens,
                      tool_result_tokens=self.tool_result_tokens,
                      tool_latency_s=self.tool_latency_s,
                      prefix_hit_rate=self.prefix_hit_rate, fanout=self.fanout,
                      kv_bytes_per_token=kv_bytes_per_token(MODELS[self.model]),
                      price_in_per_mtok=self.price_in_per_mtok,
                      price_out_per_mtok=self.price_out_per_mtok)
            kw.update(over)
            return agent_run(**kw)
        # chat and batch: one prefill of ctx, then output_tokens of decode.
        prefill = self.ctx * (1 - self.prefix_hit_rate)
        cost = (prefill / 1e6 * self.price_in_per_mtok
                + self.output_tokens / 1e6 * self.price_out_per_mtok)
        return dict(prefill_actual=prefill, prefill_naive=float(self.ctx),
                    prefill_ideal=prefill, decode_tokens=float(self.output_tokens),
                    final_context=float(self.ctx), tool_s=0.0, cost_usd=cost,
                    cost_usd_naive=cost)

    def batch_lengths(self) -> list[int]:
        """Plausible context lengths across the batch — what raggedness is measured on."""
        if self.kind != "agent" or self.turns <= 1:
            return [self.ctx] * self.batch
        base = self.system_tokens + self.tool_def_tokens + 200
        growth = self.output_tokens + self.tool_result_tokens
        # One sequence per turn number, cycled up to the batch size.
        return [base + (t % self.turns) * growth for t in range(self.batch)]

    def raggedness(self) -> dict:
        return padding_waste(self.batch_lengths())

    def describe(self) -> str:
        props = ", ".join(sorted(self.properties)) or "none"
        return (f"{self.kind}: {self.model}, ctx {self.ctx:,}, batch {self.batch}"
                f"\n  properties: {props}")


__all__ = ["Workload"]
