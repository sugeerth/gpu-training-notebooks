"""Where an agent run's tokens, seconds, memory and money go.

An agent turn re-sends its whole context, generates a short structured output, then holds its
KV cache for seconds while a tool runs. `agent_run()` costs that loop exactly; the quadratic
prefill it exposes is the single largest lever in agent serving and is not a kernel problem.

Lifted verbatim from `Agent_Workloads_On_The_Metal.ipynb`. `demo/agent-loop.html` reproduces it
in JavaScript and `tools/verify_console.py` checks the two agree over 3,600 loop shapes.
"""
from __future__ import annotations

import math


def agent_run(turns=20, system_tokens=2000, tool_def_tokens=1500, user_tokens=200,
              response_tokens=150, tool_result_tokens=800, tool_latency_s=2.0,
              prefix_hit_rate=0.95, fanout=1, prefill_tok_per_s=20000.0,
              decode_tok_per_s=60.0, kv_bytes_per_token=131072.0, pool_gb=40.0,
              price_in_per_mtok=3.0, price_out_per_mtok=15.0):
    """Where an agent run's tokens, time, memory and money actually go.

    Everything is exact arithmetic on the shape of the loop. The rates are the planning
    assumptions — measure your own and substitute them.
    """
    base = system_tokens + tool_def_tokens + user_tokens   # context before the first turn
    growth = response_tokens + tool_result_tokens          # added by each completed turn

    # Prefill, three ways.
    #   naive  — re-prefill the whole context every turn: quadratic in `turns`
    #   ideal  — only ever prefill tokens the server has never seen: linear
    #   actual — a cache that misses some fraction of the time, which is the real world
    naive = 0.0
    ideal = 0.0
    actual = 0.0
    contexts = []
    for i in range(turns):
        ctx = base + i * growth
        contexts.append(ctx)
        new = ctx if i == 0 else growth        # tokens this turn has never sent before
        cacheable = ctx - new
        naive += ctx
        ideal += new
        actual += new + (1.0 - prefix_hit_rate) * cacheable

    final_ctx = base + (turns - 1) * growth if turns else base
    decode_tokens = turns * response_tokens * fanout

    prefill_s = actual / prefill_tok_per_s
    decode_s = decode_tokens / decode_tok_per_s
    tool_s = turns * tool_latency_s
    wall_s = prefill_s + decode_s + tool_s

    # The KV a single agent holds at the end of the run, times its branches.
    kv_gb = final_ctx * kv_bytes_per_token * fanout / 1e9
    seats = pool_gb / kv_gb if kv_gb > 0 else float("inf")

    # A slot is "working" only while the GPU is doing something for it. During a tool call it
    # holds its KV and produces nothing.
    busy_s = prefill_s + decode_s
    slot_util = busy_s / wall_s if wall_s > 0 else 0.0

    cost = actual / 1e6 * price_in_per_mtok + decode_tokens / 1e6 * price_out_per_mtok
    cost_naive = naive / 1e6 * price_in_per_mtok + decode_tokens / 1e6 * price_out_per_mtok

    return dict(
        base_tokens=base, growth_per_turn=growth, final_context=final_ctx,
        prefill_naive=naive, prefill_ideal=ideal, prefill_actual=actual,
        decode_tokens=decode_tokens,
        prefill_s=prefill_s, decode_s=decode_s, tool_s=tool_s, wall_s=wall_s,
        kv_gb=kv_gb, concurrent_agents=seats, slot_util=slot_util,
        cost_usd=cost, cost_usd_naive=cost_naive,
        prefill_share=prefill_s / wall_s if wall_s else 0.0,
        decode_share=decode_s / wall_s if wall_s else 0.0,
        tool_share=tool_s / wall_s if wall_s else 0.0,
    )


def tool_gap_policy(context_tokens=20000, kv_bytes_per_token=131072.0, tool_latency_s=2.0,
                    prefill_tok_per_s=20000.0, pcie_gb_per_s=25.0, pool_gb=40.0):
    kv_gb = context_tokens * kv_bytes_per_token / 1e9
    return dict(
        kv_gb=kv_gb,
        # Holding costs nothing in time and everything in memory: the slot is unavailable for
        # the whole gap, so express it as GB-seconds of pool tied up.
        hold_gb_seconds=kv_gb * tool_latency_s,
        # Evicting frees the memory immediately and costs a full re-prefill on return.
        evict_extra_s=context_tokens / prefill_tok_per_s,
        # Offloading frees device memory for the gap, at two PCIe transfers.
        offload_extra_s=2.0 * kv_gb / pcie_gb_per_s,
        # How much of the gap the offload round trip eats — if it exceeds the gap, offloading
        # cannot even finish before the tool returns.
        pool_gb=pool_gb,
    )


def cascade_traffic(prefix_tokens: int, suffix_tokens: int, branches: int,
                    kv_bytes_per_token: float = 131072.0) -> dict:
    """KV traffic per decode step for N branches sharing a prefix, read two ways.

    `independent` runs every branch as its own sequence and streams the shared prefix N times.
    `cascade` attends to the prefix once and merges with the online-softmax identity — the same
    identity FlashDecoding uses to split one sequence across SMs, pointed at a different
    partition of the keys. Implemented in `kernels/13_prefix_attention.cu`.

    The ratio tends to `(P+S)/S` as the fan-out grows: the prefix becomes free and only the
    divergent suffix costs anything.
    """
    independent = branches * (prefix_tokens + suffix_tokens) * kv_bytes_per_token
    cascade = (prefix_tokens + branches * suffix_tokens) * kv_bytes_per_token
    return dict(independent_bytes=independent, cascade_bytes=cascade,
                ratio=independent / cascade if cascade else float("inf"),
                limit=(prefix_tokens + suffix_tokens) / suffix_tokens if suffix_tokens else
                float("inf"))


def prefill_saving(turns: int = 20, **kw) -> dict:
    """What a prefix cache is worth on an N-turn run, as a ratio against both bounds.

    Two ratios, because they answer different questions. `vs_naive` is what caching buys you
    over recomputing everything — it grows with turn count, because one curve is quadratic and
    the other is not. `vs_ideal` is how far your hit rate is from the floor, and it is the one
    to engineer against: a 95% hit rate costs 1.5x the ideal, because the 5% it misses is 5% of
    the whole context rather than of the new tokens.

    The lookup that produces the hit is `kernels/16_prefix_match.cu`.
    """
    a = agent_run(turns=turns, **kw)
    return dict(naive=a["prefill_naive"], ideal=a["prefill_ideal"], actual=a["prefill_actual"],
                vs_naive=a["prefill_naive"] / a["prefill_actual"],
                vs_ideal=a["prefill_actual"] / a["prefill_ideal"])


def restore_cost(context_tokens: int, hit_rate: float) -> float:
    """Tokens that must be re-prefilled when an evicted sequence comes back.

    `(1 - hit_rate) * tokens`, not `tokens` — and that is the whole reason eviction is a
    routine reclaim for an agent platform and a crisis for a chat one. At a 95% hit rate an
    evicted agent costs a twentieth of its context to restore. See
    `kernels/22_kv_evict.cu`, whose scoring divides by exactly this.
    """
    return (1.0 - hit_rate) * context_tokens


def padding_waste(context_lengths) -> dict:
    """What padding a ragged batch to its longest sequence throws away.

    An agent batch is a mix of *turn numbers*, so its spread comes from how many turns each run
    has taken — which has no natural ceiling the way message length does. Note that the
    percentage saturates between a third and a half for any spread wide enough to matter, chat
    included; what separates the workloads is the absolute token count, so both are returned.
    Fixed by `cu_seqlens` in `kernels/17_ragged_batch.cu` — which removes the wasted work and
    leaves the imbalance, a separate problem with a separate fix.
    """
    lens = list(context_lengths)
    if not lens:
        return dict(padded=0, needed=0, wasted=0, fraction=0.0)
    lmax, total = max(lens), sum(lens)
    padded = lmax * len(lens)
    return dict(padded=padded, needed=total, wasted=padded - total,
                fraction=1.0 - total / padded if padded else 0.0,
                spread=lmax / min(lens) if min(lens) else float("inf"))


__all__ = ["agent_run", "tool_gap_policy", "cascade_traffic", "prefill_saving",
           "restore_cost", "padding_waste"]
