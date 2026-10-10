"""Turn a plan into something a person reads without a debugger.

Four renderers, all returning plain strings so they work in a terminal, a notebook, a commit
message or a Slack paste. Nothing here computes: if a number appears in a report it came from
`stack.py`, so a reader can always find where it was decided.
"""
from __future__ import annotations

from .levers import LEVERS, RESOURCES


def _money(x: float) -> str:
    return f"${x:,.3f}" if x < 10 else f"${x:,.2f}"


def plan_report(stack) -> str:
    """The whole answer for one stack: what applies, what it costs, what is binding."""
    p = stack.plan()
    w = stack.workload
    out = [f"servingkit plan — {w.kind} on {stack.gpu}",
           "=" * 68,
           w.describe(),
           ""]

    if stack.names:
        out.append("levers pulled")
        for n in stack.names:
            lv = LEVERS[n]
            k = f"  [{lv.kernel}]" if lv.kernel else ""
            out.append(f"  + {n}{k}")
    else:
        out.append("levers pulled: none (baseline)")

    if p["rejected"]:
        out += ["", "not applied"]
        for r in p["rejected"]:
            detail = f" ({r.detail})" if r.detail else ""
            out.append(f"  - {r.lever}: {r.reason}{detail}")

    out += ["",
            "the step",
            f"  {p['step_ms']:.3f} ms · {p['tokens_per_s']:,.0f} tok/s · "
            f"{p['tpot_ms']:.1f} ms/token",
            f"  bottleneck: {p['bottleneck']} ({p['gemm_bound']}-bound GEMM), "
            f"{p['spare_compute']:.0%} of math capacity idle",
            f"  vs baseline: {p['speedup']:.2f}x",
            "",
            "the run",
            f"  prefill {p['prefill_tokens']:,.0f} tok · decode {p['decode_tokens']:,.0f} tok",
            f"  wall {p['wall_s']:.1f}s = prefill {p['prefill_s']:.1f}s + "
            f"decode {p['decode_s']:.1f}s + tools {p['tool_s']:.1f}s",
            f"  slot utilization {p['slot_util']:.0%}"
            + ("  — the rest is a KV cache with a reservation" if p["tool_s"] > 0 else ""),
            f"  cost {_money(p['cost_usd'])} per run"]

    offered = stack.offered()
    if offered:
        out += ["", "still on the table"]
        for lv in offered:
            k = f"  [{lv.kernel}]" if lv.kernel else ""
            out.append(f"  ? {lv.name}{k}")

    unav = [(lv, m) for lv, m in stack.unavailable() if lv.name not in stack.names]
    if unav:
        out += ["", "would need a different workload"]
        for lv, missing in unav[:8]:
            out.append(f"  x {lv.name}: needs {', '.join(missing)}")
        out.append("    (that list is what to engineer into existence, not what to turn on)")
    return "\n".join(out)


def lever_table(properties=None) -> str:
    """Every lever, what it spends, what it needs, and which kernel implements it."""
    rows = []
    for name, lv in LEVERS.items():
        mark = ""
        if properties is not None:
            mark = "  " if lv.applies_to(properties) else " x"
        rows.append((mark, name, lv.domain, ", ".join(lv.spends) or "-",
                     ", ".join(lv.requires) or "any", lv.kernel or "-"))
    w = [max(len(r[i]) for r in rows) for i in range(6)]
    head = (f"{'':<{w[0]}} {'lever':<{w[1]}}  {'domain':<{w[2]}}  {'spends':<{w[3]}}  "
            f"{'requires':<{w[4]}}  kernel")
    out = [head, "-" * len(head)]
    for r in rows:
        out.append(f"{r[0]:<{w[0]}} {r[1]:<{w[1]}}  {r[2]:<{w[2]}}  {r[3]:<{w[3]}}  "
                   f"{r[4]:<{w[4]}}  {r[5]}")
    if properties is not None:
        out.append("")
        out.append("x = not applicable to this workload. Not 'weak' — inapplicable: the "
                   "property it exploits is absent.")
    return "\n".join(out)


def ladder_table(rows) -> str:
    """The lever ladder: one rung per lever, with the running cost and wall clock."""
    head = (f"{'configuration':<30}{'tok/s':>10}{'prefill':>12}{'wall':>9}{'$':>10}"
            f"{'util':>7}   {'Δwall':>7} {'Δ$':>7}  kernel")
    out = [head, "-" * len(head)]
    for r in rows:
        dw = "" if not r["d_wall"] else f"{100 * r['d_wall']:+.0f}%"
        dc = "" if not r["d_cost"] else f"{100 * r['d_cost']:+.0f}%"
        k = r.get("kernel") or ""
        out.append(f"{r['label']:<30}{r['tokens_per_s']:>10,.0f}"
                   f"{r['prefill_tokens']:>12,.0f}{r['wall_s']:>8.0f}s"
                   f"{r['cost_usd']:>10.3f}{r['slot_util']:>6.0%}   {dw:>7} {dc:>7}  {k}")
    return "\n".join(out)


def interaction_table(pairs) -> str:
    """Pairwise gains, with the shared resource that explains each shortfall."""
    if not pairs:
        return "no pairs to compare (fewer than two levers change the decode step)"
    head = (f"{'A':<26}{'B':<26}{'A':>7}{'B':>7}{'expect':>8}{'actual':>8}{'synergy':>9}"
            f"   why")
    out = [head, "-" * len(head)]
    for c in pairs:
        why = ""
        if c["synergy"] < 0.9:
            why = ("both spend " + ", ".join(c["predicted_conflict"])
                   if c["predicted_conflict"] else "ANTAGONISTIC, no declared conflict")
        elif c["synergy"] > 1.1:
            why = "synergistic"
        out.append(f"{c['a']:<26}{c['b']:<26}{c['solo_a']:>6.2f}x{c['solo_b']:>6.2f}x"
                   f"{c['expected']:>7.2f}x{c['actual']:>7.2f}x{c['synergy']:>9.2f}   {why}")
    out.append("")
    out.append("A synergy below 1 with no declared conflict means a lever's `spends` is wrong.")
    out.append("That is the check this table exists for: the declaration predicts, the")
    out.append("measurement confirms, and a disagreement is a bug in the model.")
    return "\n".join(out)


def resource_legend() -> str:
    w = max(len(k) for k in RESOURCES)
    return "\n".join(f"  {k:<{w}}  {v}" for k, v in RESOURCES.items())


__all__ = ["plan_report", "lever_table", "ladder_table", "interaction_table", "resource_legend"]
