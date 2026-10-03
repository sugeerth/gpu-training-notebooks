"""A lever is an optimization you can declare, compose, and be told the cost of.

The repository's earlier framing was a dict of `name -> (config -> config)` functions. That
composes, but it cannot answer the two questions people actually have:

  * **does this apply to me?** Cascade attention is worth a lot to a planner spawning
    sub-agents and nothing at all to a single-turn chat. A function cannot say so; a lever
    that declares `requires=("shared_prefix",)` can, and the stack simply will not offer it.

  * **why did my two 2x wins give me 2.6x?** Because both spent the same scarce resource. A
    lever that declares `spends=("spare_compute",)` lets the interaction be *predicted* before
    it is measured, which turns a surprise into arithmetic.

So a lever carries four declarations beyond its transformation:

    domain    what it changes. Two levers in the same domain are alternatives, not additions
              — a model has exactly one weight format.
    spends    the scarce resources it consumes. Overlap predicts antagonism.
    requires  the workload properties it exploits. Absent → not applicable, not merely weak.
    kernel    the file in `kernels/` that implements it, so "read the code" has an address.

Adding a lever is a dozen lines and needs nothing from this module but the dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .step import base_config, evaluate

# The scarce resources levers compete for. Naming them is what makes antagonism predictable
# instead of surprising: two levers that spend the same one cannot both be paid.
RESOURCES = {
    "spare_compute": "math capacity a memory-bound step leaves idle",
    "memory": "HBM: weights, KV pool, activations",
    "bandwidth": "HBM bandwidth per step",
    "host_cpu": "CPU work on the critical path between forward pass and sampler",
    "interconnect": "NVLink/PCIe between ranks",
    "accuracy": "output quality — the resource people forget is scarce",
    "determinism": "the ability to reproduce a run exactly",
}

# Workload properties a lever can require. A workload declares which it has; see `workload.py`.
PROPERTIES = {
    "repeated_context": "the same leading tokens arrive again and again",
    "shared_prefix": "several sequences in flight share a long prefix",
    "structured_output": "output is constrained by a grammar or schema",
    "predictable_output": "output is largely determined before the model decides",
    "forks": "trajectories branch from a common state",
    "ragged_batch": "concurrent sequences differ widely in length",
    "idle_holds": "sequences hold their KV while producing nothing",
    "mid_batch_prefill": "long prefills arrive while others are decoding",
    "replayed": "runs are re-run, retried or compared against a recorded trace",
    "long_context": "context is long enough that KV dominates memory",
    "spare_compute": "the step is memory-bound, so math capacity is idle",
    "weight_bound": "the weight read is the largest term — what speculation amortizes",
}


@dataclass(frozen=True)
class Lever:
    """One optimization, declared rather than merely implemented."""

    name: str
    apply: Callable[[dict], dict]
    domain: str
    spends: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    kernel: str | None = None
    note: str = ""

    # Not every lever is a config change. `apply` covers the ones that alter a field
    # `evaluate()` already reads; the factors below cover the ones that change a *term* of the
    # step, or something outside the step entirely.
    #
    # Keeping these as declarations rather than as edits to `evaluate()` is deliberate. There
    # is exactly one copy of `evaluate()` and it is the notebook's, pinned by a test. A lever
    # that wants to halve attention traffic says so here, where it can be read, argued with and
    # checked, instead of adding a branch to the shared model.
    prefill_factor: float = 1.0     # multiplies prefill tokens (0.0 = only ever-unseen tokens)
    idle_factor: float = 1.0        # multiplies time a slot is held while producing nothing
    attention_factor: float = 1.0   # multiplies the attention term of the step
    sampling_factor: float = 1.0    # multiplies the sampling term
    memory_factor: float = 1.0      # multiplies KV bytes held per sequence
    gpu_factor: float = 1.0         # multiplies GPUs rented — the part that shows up on a bill
    host_sync_ms: float = 0.0       # adds (or removes, if negative) host round trip per step
    # Some effects depend on the workload, not on a constant. A lever may supply a hook that
    # returns its factors given the workload; it wins over the constants above.
    dynamic: Callable[[object], dict] | None = None

    def factors(self, workload=None) -> dict:
        f = dict(prefill=self.prefill_factor, idle=self.idle_factor,
                 attention=self.attention_factor, sampling=self.sampling_factor,
                 memory=self.memory_factor, gpu=self.gpu_factor,
                 host_sync_ms=self.host_sync_ms)
        if self.dynamic is not None and workload is not None:
            f.update(self.dynamic(workload))
        return f

    def applies_to(self, properties) -> bool:
        return all(r in properties for r in self.requires)

    def missing(self, properties) -> list[str]:
        return [r for r in self.requires if r not in properties]

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name


def _set(**kw) -> Callable[[dict], dict]:
    """A lever body that overwrites config fields."""
    def f(c: dict) -> dict:
        c = dict(c)
        c.update(kw)
        return c
    return f


def _scale(**kw) -> Callable[[dict], dict]:
    """A lever body that multiplies config fields."""
    def f(c: dict) -> dict:
        c = dict(c)
        for k, v in kw.items():
            c[k] = c[k] * v
        return c
    return f


def _noop(c: dict) -> dict:
    return dict(c)


# --------------------------------------------------------------------------------------------
# The registry.
#
# Grouped the way a serving team reaches for them. The agent levers are last not because they
# matter least but because they are the ones that only apply to a loop — and every one of them
# is an earlier kernel in this repository pointed at a different caller.
# --------------------------------------------------------------------------------------------
LEVERS: dict[str, Lever] = {}


def register(lever: Lever) -> Lever:
    if lever.domain and lever.spends:
        for r in lever.spends:
            if r not in RESOURCES:
                raise ValueError(f"{lever.name}: unknown resource {r!r}")
    for p in lever.requires:
        if p not in PROPERTIES:
            raise ValueError(f"{lever.name}: unknown property {p!r}")
    LEVERS[lever.name] = lever
    return lever


# ---- precision and memory -----------------------------------------------------------------
register(Lever(
    "int4 weights", _set(weight_bytes=0.5, kernel_eff=0.85), domain="weight_format",
    spends=("accuracy", "spare_compute"), kernel="05_dequant_gemv.cu",
    note="A quarter of the weight traffic, at a kernel-maturity discount that is a software "
         "fact rather than a silicon one. At decode this is a speed technique; at prefill it "
         "is only a capacity one.\n\n"
         "It spends `spare_compute` for a reason worth stating: shrinking the weight read is "
         "what *removes* the memory-bound-ness, and memory-bound-ness is the resource "
         "batching and speculation convert into tokens. The first version of this file omitted "
         "that, and the interaction table caught it — a 0.57 synergy with speculation and no "
         "declared conflict, which is precisely the signal that table exists to produce."))

register(Lever(
    "fp8 weights", _set(weight_bytes=1.0), domain="weight_format",
    spends=("accuracy", "spare_compute"), kernel="09_fp8_scaling.cu",
    note="Half the weight traffic with hardware support, so no kernel discount — but scaling "
         "granularity decides whether it is lossless or silently underflowing. Spends "
         "`spare_compute` for the same reason int4 does: it consumes the memory-bound-ness "
         "that batching and speculation are paid out of."))

register(Lever(
    "fp8 KV cache", _set(kv_bytes=1.0), domain="kv_format",
    spends=("accuracy", "memory"), requires=("long_context",), kernel="09_fp8_scaling.cu",
    note="Halves the cache, which doubles concurrency. Worth most exactly when KV dominates "
         "memory, which is what `long_context` asserts."))

register(Lever(
    "KV eviction", _set(attn_ctx_frac=0.25), domain="kv_extent",
    spends=("accuracy", "memory"), requires=("long_context",),
    note="Attend to a quarter of the context. Cheap in bandwidth, expensive in the thing you "
         "cannot measure from a throughput number."))

# ---- throughput and topology ---------------------------------------------------------------
register(Lever(
    "2x batch", _scale(batch=2), domain="batch",
    spends=("spare_compute", "memory"), requires=("spare_compute", "weight_bound"),
    note="Amortizes one weight read over twice as many sequences — so, exactly like "
         "speculation, it needs the weight read to be the thing that hurts. It requires "
         "`weight_bound` for the same reason and it is worth being blunt about: on a "
         "long-context step, attention scales with the batch and batching buys nothing while "
         "costing memory. 'Just raise the batch size' is advice with a precondition."))

register(Lever(
    "speculation", _set(spec_k=4, spec_alpha=0.75), domain="tokens_per_step",
    spends=("spare_compute",), requires=("spare_compute", "weight_bound"),
    kernel="18_spec_verify.cu",
    note="Amortizes one weight read over k+1 positions, so it needs the weight read to be the "
         "thing that hurts. It also multiplies the attention term by k+1, which is why "
         "`weight_bound` is a requirement and not a nicety: at long context speculation makes "
         "the step slower, and offering it there would be advice that costs money."))

register(Lever(
    "TP=2", _set(tp=2), domain="topology",
    spends=("interconnect",), kernel="10_collectives.cu", gpu_factor=2.0,
    note="Twice the bandwidth and twice the price, minus an all-reduce per layer. The price is "
         "what `gpu_factor` is for: a lever that halves your step time and doubles your bill "
         "has not obviously helped, and a planner that only reports latency cannot say so."))

register(Lever(
    "CUDA graphs", _set(graphs=True), domain="launch",
    note="Collapses hundreds of launches into three. Free, and already on in the baseline — "
         "it is here so that turning it *off* is expressible."))

# ---- agents: the caller is a loop ----------------------------------------------------------
register(Lever(
    "prefix caching", _noop, domain="prefill", prefill_factor=0.0,
    requires=("repeated_context",), kernel="16_prefix_match.cu",
    note="Not a kernel optimization at all — a hash lookup and a decision not to recompute. "
         "The largest single lever in agent serving, and the one with the best ratio of cost "
         "to value in the whole path."))

def _cascade(w) -> dict:
    # The prefix is everything up to the branch point; the suffix is what each branch has
    # generated since. Traffic goes from N*(P+S) to P + N*S, so the attention term scales by
    # the inverse of that ratio, bounded below by 1/N.
    from .agents import cascade_traffic
    suffix = max(w.output_tokens, 1)
    prefix = max(w.ctx - suffix, 1)
    c = cascade_traffic(prefix, suffix, max(w.fanout, 1), 1.0)
    return dict(attention=c["cascade_bytes"] / c["independent_bytes"])


register(Lever(
    "cascade attention", _noop, domain="attention_partition",
    spends=("bandwidth",), requires=("shared_prefix",), kernel="13_prefix_attention.cu",
    dynamic=_cascade,
    note="Read a shared prefix once per step instead of N times. The online-softmax merge "
         "from 06, pointed at a different partition of the keys. The saving tends to (P+S)/S "
         "as the fan-out grows, so its factor is the workload's own ratio, not a constant."))

def _ragged(w) -> dict:
    # Padding waste is measured on the batch's own length distribution, so a batch that
    # happens to be uniform gets no credit — which is correct, and is why this is dynamic.
    return dict(attention=1.0 - w.raggedness()["fraction"])


register(Lever(
    "ragged batching", _noop, domain="batch_layout",
    requires=("ragged_batch",), kernel="17_ragged_batch.cu", dynamic=_ragged,
    note="cu_seqlens: every sequence loops to its own length, so attention stops paying for "
         "padding. Removes the wasted work and leaves the imbalance — splitting is the "
         "separate fix, and the kernel shows they are not the same fix."))

register(Lever(
    "chunked prefill", _noop, domain="schedule", idle_factor=0.9,
    requires=("mid_batch_prefill",), kernel="21_chunked_prefill.cu",
    note="One launch serves decodes and a slice of prefill together, so nobody waits for a "
         "whole prefill. Trades the prefilling sequence's latency for everyone else's — a "
         "good bet only because the thing waiting is a loop and not a person."))

register(Lever(
    "bitset logit mask", _noop, domain="mask_format",
    requires=("structured_output",), kernel="14_logit_mask.cu", sampling_factor=0.97,
    note="1 bit per token instead of a float, so the sampling term loses the mask read. The "
         "factor is deliberately unimpressive: the kernel measures this at well under a "
         "percent of a step, and a lever that claimed more would be lying."))

register(Lever(
    "grammar on device", _noop, domain="grammar_placement",
    spends=("host_cpu",), requires=("structured_output",), kernel="20_grammar_advance.cu",
    host_sync_ms=-1.2,
    note="Per-state bitsets resident on the GPU, so the host leaves the decode loop. Note "
         "that this is the only lever in the registry whose effect is a *negative time* "
         "rather than a factor — because the cost it removes was never bandwidth. It was a "
         "device-host-device round trip that nothing else in the step could hide."))

def _cow(w) -> dict:
    # N branches share one parent instead of copying it, so the pool holds ~1 copy plus the
    # divergent tails rather than N copies. Nothing about the step changes at all.
    n = max(w.fanout, 1)
    return dict(memory=1.0 / n)


register(Lever(
    "copy-on-write forking", _noop, domain="fork",
    spends=("memory",), requires=("forks",), kernel="15_kv_fork.cu", dynamic=_cow,
    note="Share every full page, copy one partial page per child: O(1) in the parent's "
         "length. Changes no term of the step and every term of the capacity, which is why it "
         "shows up in concurrency rather than in tokens per second."))

register(Lever(
    "agent-aware eviction", _noop, domain="eviction", idle_factor=0.75, memory_factor=0.7,
    requires=("idle_holds", "repeated_context"), kernel="22_kv_evict.cu",
    note="Evict the sequences that are parked in a tool call, because the scheduler issued "
         "those calls and knows. Cheap to undo only because the prefix cache exists."))

register(Lever(
    "batch-invariant reduction", _noop, domain="reduction",
    spends=("spare_compute",), requires=("replayed",), kernel="19_batch_invariant.cu",
    sampling_factor=1.05,
    note="Fix the split count so logits do not depend on who else is in the batch. It costs a "
         "few percent of the reduction and buys the ability to replay a trajectory at all — "
         "the only lever here whose return is not a number."))


# Domains are exclusive by construction: a model has one weight format, a cache has one
# element size. The stack enforces it so a search cannot propose an incoherent config.
def exclusive_pairs() -> list[set[str]]:
    by_domain: dict[str, list[str]] = {}
    for name, lv in LEVERS.items():
        by_domain.setdefault(lv.domain, []).append(name)
    out = []
    for names in by_domain.values():
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out.append({a, b})
    return out


def shared_resources(a: str, b: str) -> set[str]:
    """Resources two levers both spend — the mechanical reason their gains will not multiply."""
    return set(LEVERS[a].spends) & set(LEVERS[b].spends)


def applicable(properties, names=None) -> list[Lever]:
    """Every lever whose preconditions this workload satisfies."""
    pool = names or list(LEVERS)
    return [LEVERS[n] for n in pool if LEVERS[n].applies_to(properties)]


def solo_gain(name: str, cfg: dict | None = None) -> float:
    """Throughput multiple from one lever alone, on the decode step."""
    cfg = cfg or base_config()
    lv = LEVERS[name]
    return (evaluate(lv.apply(cfg))["tokens_per_s"] / evaluate(cfg)["tokens_per_s"])


__all__ = ["Lever", "LEVERS", "RESOURCES", "PROPERTIES", "register", "exclusive_pairs",
           "shared_resources", "applicable", "solo_gain"]
