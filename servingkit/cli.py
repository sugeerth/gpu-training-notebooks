"""`python -m servingkit` — the package without writing any Python.

    servingkit levers                          what exists, what each one spends
    servingkit plan --agent --turns 30         the recommended stack for a workload
    servingkit ladder --agent --turns 30       one rung per lever, ordered by effort
    servingkit interactions --chat             which pairs fight, and over what
    servingkit cheapest Llama-3.1-8B           every feasible deployment, by $/M tokens
    servingkit kernel 13                       build and run one kernel, no GPU needed
    servingkit check                           does the package still match the notebooks

Everything prints plain text on purpose: it pastes into a commit message, a ticket or a Slack
thread without losing anything.
"""
from __future__ import annotations

import argparse
import sys

from . import report
from .levers import LEVERS
from .serving import cheapest
from .stack import Stack, recommend
from .workload import Workload


def _workload_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--agent", action="store_true", help="an agent loop (the default)")
    g.add_argument("--chat", action="store_true", help="a person typing")
    g.add_argument("--batch", action="store_true", help="offline: nobody is waiting")
    p.add_argument("--model", default="Llama-3.1-8B")
    p.add_argument("--gpu", default="H100 SXM")
    p.add_argument("--turns", type=int, default=20)
    p.add_argument("--ctx", type=int, default=None, help="chat/batch only; agents derive it")
    p.add_argument("--concurrency", type=int, default=32, help="sequences in flight")
    p.add_argument("--tool-tokens", type=int, default=800)
    p.add_argument("--tool-latency", type=float, default=2.0)
    p.add_argument("--fanout", type=int, default=1)
    p.add_argument("--hit-rate", type=float, default=0.0)
    p.add_argument("--replayed", action="store_true", help="runs are re-run or compared")
    p.add_argument("--structured", action="store_true", help="output follows a grammar")


def _build(a: argparse.Namespace) -> Workload:
    if a.chat:
        return Workload.chat(model=a.model, ctx=a.ctx or 4096, batch=a.concurrency,
                             replayed=a.replayed, structured_output=a.structured,
                             prefix_hit_rate=a.hit_rate)
    if a.batch:
        return Workload.batch(model=a.model, ctx=a.ctx or 8192, batch=a.concurrency,
                              replayed=a.replayed, structured_output=a.structured)
    return Workload.agent(model=a.model, turns=a.turns, batch=a.concurrency,
                          tool_result_tokens=a.tool_tokens, tool_latency_s=a.tool_latency,
                          fanout=a.fanout, prefix_hit_rate=a.hit_rate, replayed=a.replayed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="servingkit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("levers", help="every lever, and whether it applies")
    _workload_args(p)
    p.add_argument("--all", action="store_true", help="ignore the workload; list everything")

    for name, helptext in (("plan", "the recommended stack, costed"),
                           ("ladder", "one rung per lever, ordered by effort"),
                           ("interactions", "which pairs fight, and over what")):
        p = sub.add_parser(name, help=helptext)
        _workload_args(p)
        p.add_argument("--lever", action="append", default=None,
                       help="pull exactly these instead of the recommendation")

    p = sub.add_parser("cheapest", help="every feasible deployment, by $/M tokens")
    p.add_argument("model")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--max-tpot-ms", type=float, default=None)

    p = sub.add_parser("kernel", help="build and run one kernel from kernels/")
    p.add_argument("name", help='"13", "13_prefix_attention", or a filename')
    p.add_argument("--quiet", action="store_true", help="the summary only")

    sub.add_parser("check", help="does the package still agree with the notebooks")

    a = ap.parse_args(argv)

    if a.cmd == "levers":
        props = None if a.all else _build(a).properties
        print(report.lever_table(props))
        print()
        print("resources a lever can spend:")
        print(report.resource_legend())
        return 0

    if a.cmd in ("plan", "ladder", "interactions"):
        w = _build(a)
        s = Stack(w, a.gpu).with_levers(*a.lever) if a.lever else recommend(w, a.gpu)
        if a.cmd == "plan":
            print(report.plan_report(s))
        elif a.cmd == "ladder":
            print(w.describe())
            print()
            print(report.ladder_table(s.ladder()))
        else:
            print(w.describe())
            print()
            print(report.interaction_table(s.interactions()))
        return 0

    if a.cmd == "cheapest":
        rows = cheapest(a.model, batch=a.concurrency, ctx=a.ctx, min_tpot_ms=a.max_tpot_ms)
        if not rows:
            print("nothing feasible at those settings")
            return 1
        head = (f"{'gpu':<12}{'prec':>6}{'tp':>4}{'batch':>7}{'tok/s':>10}{'tpot ms':>10}"
                f"{'$/Mtok':>10}   bound")
        print(head)
        print("-" * len(head))
        for r in rows[:20]:
            print(f"{r['gpu']:<12}{r['precision']:>6}{r['tp']:>4}{r['batch']:>7}"
                  f"{r['decode_tps']:>10,.0f}{r['tpot_ms']:>10.1f}{r['cpm']:>10.3f}"
                  f"   {r['bound']}")
        return 0

    if a.cmd == "kernel":
        from .kernels import run_kernel
        r = run_kernel(a.name, quiet=a.quiet)
        print(f"{r.name} on {r.device}")
        print(r.table())
        if not r.timed:
            print("\n(no timings: this is the CPU shim, which emulates correctness and "
                  "nothing about performance)")
        return 0 if r.ok else 1

    if a.cmd == "check":
        from .selfcheck import run_checks
        return run_checks()

    return 2


if __name__ == "__main__":
    sys.exit(main())
