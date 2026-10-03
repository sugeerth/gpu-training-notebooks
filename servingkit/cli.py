"""`python -m servingkit` — the package without writing any Python.

    servingkit levers                          what exists, what each one spends
    servingkit plan --agent --turns 30         the recommended stack for a workload
    servingkit ladder --agent --turns 30       one rung per lever, ordered by effort
    servingkit interactions --chat             which pairs fight, and over what
    servingkit cheapest Llama-3.1-8B           every feasible deployment, by $/M tokens
    servingkit kernel 13                       build and run one kernel, no GPU needed
    servingkit agent --agent --turns 30 --max-tpot-ms 25
                                               let the control loop decide, and show its working
    servingkit check                           does the package still match the notebooks

Any command takes `--log-hook`, more than once if you like, to point the structured event stream
somewhere: `--log-hook text` for readable lines, `--log-hook file:/tmp/sk.jsonl`, `--log-hook
webhook:https://collector/ingest`. `$SERVINGKIT_LOG_HOOKS` does the same thing for a container.

Everything prints plain text on purpose: it pastes into a commit message, a ticket or a Slack
thread without losing anything.
"""
from __future__ import annotations

import argparse
import os
import sys

from . import report
from .events import BUS, configure
from .levers import LEVERS
from .serving import cheapest
from .stack import Stack, recommend
from .workload import Workload


def _objective_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-tpot-ms", type=float, default=None, help="per-token latency ceiling")
    p.add_argument("--max-wall-s", type=float, default=None, help="one run must finish inside")
    p.add_argument("--max-cost-usd", type=float, default=None, help="budget per run")
    p.add_argument("--min-tokens-per-s", type=float, default=None)
    p.add_argument("--minimize", default="total_cost_usd", help="a numeric key of the plan")
    p.add_argument("--maximize", default=None, help="maximize this instead of minimizing")
    p.add_argument("--spend-accuracy", action="store_true",
                   help="allow levers that cost output quality")
    p.add_argument("--no-gpu-moves", action="store_true", help="keep the card fixed")
    p.add_argument("--min-improvement", type=float, default=0.005,
                   help="deadband: reject changes smaller than this fraction")
    p.add_argument("--max-steps", type=int, default=12)


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
    # Global, and before the subcommand is chosen, because logging is not a property of any one
    # command. `text` on stderr keeps stdout clean for the thing being piped somewhere.
    ap.add_argument("--log-hook", action="append", default=None, metavar="SPEC",
                    help="a log hook: jsonl | text | file:PATH | ring:N | counter | "
                         "webhook:URL | plugin:mod:attr, optionally @level and #kind.prefix")
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

    p = sub.add_parser("agent", help="let the control loop tune the stack, and show its working")
    _workload_args(p)
    _objective_args(p)
    p.add_argument("--lever", action="append", default=None, help="start from these")
    p.add_argument("--json", action="store_true", help="the trace as JSON instead of a table")
    p.add_argument("--watch", action="store_true",
                   help="print every decision as it happens (a text log hook)")

    p = sub.add_parser("serve", help="run the JSON API")
    p.add_argument("--host", default="0.0.0.0")   # noqa: S104 - a container needs this
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))

    p = sub.add_parser("pipeline", help="spec -> plan -> verified kernels -> scorecard")
    _workload_args(p)
    p.add_argument("--lever", action="append", default=None)
    p.add_argument("-o", "--out", default=None, help="write the scorecard here")
    p.add_argument("--no-verify", action="store_true", help="skip compiling kernels")
    p.add_argument("--no-drift", action="store_true", help="skip the notebook comparison")
    p.add_argument("--all-kernels", action="store_true",
                   help="verify every kernel, not only the ones this plan names")
    p.add_argument("--shard", type=int, default=int(os.environ.get("SERVINGKIT_SHARD", 0)))
    p.add_argument("--shards", type=int, default=int(os.environ.get("SERVINGKIT_SHARDS", 1)))
    p.add_argument("--quiet", action="store_true", help="the scorecard only, no progress")
    p.add_argument("--tune", action="store_true",
                   help="let the control loop choose the plan instead of recommend()")
    _objective_args(p)

    p = sub.add_parser("merge", help="combine shard scorecards into one")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--out", required=True)

    a = ap.parse_args(argv)
    configure(a.log_hook if a.log_hook else None, default="")
    if getattr(a, "watch", False):
        from .events import TextHook
        BUS.subscribe(TextHook(min_level="info", prefixes=("agent.",)))

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

    if a.cmd == "agent":
        import json as _json

        from .agent import ControlLoop, Objective
        w = _build(a)
        obj = Objective(max_tpot_ms=a.max_tpot_ms, max_wall_s=a.max_wall_s,
                        max_cost_usd=a.max_cost_usd, min_tokens_per_s=a.min_tokens_per_s,
                        minimize=a.minimize, maximize=a.maximize,
                        allow_accuracy_loss=a.spend_accuracy,
                        allow_gpu_moves=not a.no_gpu_moves,
                        min_improvement=a.min_improvement)
        tr = ControlLoop(w, obj, gpu=a.gpu, levers=a.lever).run(max_steps=a.max_steps)
        if a.json:
            print(_json.dumps(tr.as_dict(), indent=2, allow_nan=False))
        else:
            print(w.describe())
            print()
            print(tr.table())
        # A findings list is a defect in `levers.py`, so it is an exit code and not a note.
        return 1 if tr.findings else 0

    if a.cmd == "check":
        from .selfcheck import run_checks
        return run_checks()

    if a.cmd == "serve":
        from .api import serve
        serve(a.host, a.port, log_hooks=",".join(a.log_hook) if a.log_hook else None)
        return 0

    if a.cmd == "pipeline":
        import json as _json
        from pathlib import Path

        from .pipeline import run, write
        w = _build(a)
        spec = dict(kind=w.kind, model=w.model, concurrency=w.batch, turns=w.turns,
                    tool_result_tokens=w.tool_result_tokens, tool_latency_s=w.tool_latency_s,
                    fanout=w.fanout, prefix_hit_rate=w.prefix_hit_rate, replayed=w.replayed,
                    structured_output=w.structured_output, gpu=a.gpu)
        if w.kind != "agent":
            spec["ctx"] = w.ctx
        if a.lever:
            spec["levers"] = a.lever

        def progress(st):
            mark = {"ok": "ok", "failed": "FAILED", "skipped": "--"}.get(st.status, st.status)
            extra = st.error or ""
            print(f"  {st.name:<10} {mark:<7} {st.seconds:6.2f}s  {extra}", file=sys.stderr)

        for k, v in (("max_tpot_ms", a.max_tpot_ms), ("max_wall_s", a.max_wall_s),
                     ("max_cost_usd", a.max_cost_usd),
                     ("min_tokens_per_s", a.min_tokens_per_s), ("maximize", a.maximize),
                     ("minimize", a.minimize), ("max_steps", a.max_steps),
                     ("allow_accuracy_loss", a.spend_accuracy),
                     ("allow_gpu_moves", not a.no_gpu_moves)):
            if v is not None:
                spec[k] = v
        doc = run(spec, verify=not a.no_verify, drift=not a.no_drift,
                  all_kernels=a.all_kernels, shard=a.shard, shards=a.shards,
                  tune=a.tune, on_stage=None if a.quiet else progress)
        if a.out:
            print(f"wrote {write(doc, Path(a.out))}", file=sys.stderr)
        else:
            print(_json.dumps(doc, indent=2, allow_nan=False))
        return 0 if doc["status"] == "ok" else 1

    if a.cmd == "merge":
        from pathlib import Path

        from .pipeline import merge, write
        doc = merge([Path(p) for p in a.inputs])
        print(f"merged {len(a.inputs)} shard(s) -> {write(doc, Path(a.out))}", file=sys.stderr)
        return 0 if doc["status"] == "ok" else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
