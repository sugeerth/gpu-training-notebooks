"""CLI: python -m kernelbench eval <dir>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import Evaluator
from .report import terminal, write
from .suite import Suite


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kernelbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("eval", help="score every kernel in a suite")
    ev.add_argument("path", help="directory containing kernelbench.json, or the file itself")
    ev.add_argument("--only", action="append", default=[], help="restrict to these sources")
    ev.add_argument("--skip", action="append", default=[],
                    help="dimensions to skip: mutation, determinism, efficiency")
    ev.add_argument("--no-nvcc", action="store_true", help="force the CPU shim build")
    ev.add_argument("--arch", default="native", help="nvcc -arch value (default: native)")
    ev.add_argument("--json", dest="json_path", help="write a machine-readable report here")
    ev.add_argument("--markdown", dest="md_path", help="write a PR-ready report here")
    ev.add_argument("--html", dest="html_path", help="write a visual scorecard here")
    ev.add_argument("--baseline", help="compare against a previous --json report")
    ev.add_argument("--update-baseline", action="store_true",
                    help="write this run to the --baseline path instead of comparing")
    ev.add_argument("--regress-pct", type=float, default=15.0,
                    help="flag a kernel whose best variant got this %% slower (default 15)")
    ev.add_argument("--fail-under", type=float, default=1.0,
                    help="exit non-zero if the overall score is below this (default 1.0)")

    args = ap.parse_args(argv)

    suite = Suite.load(args.path)
    problems = suite.validate()
    if problems:
        print(f"suite {suite.name} is not well-formed:")
        for p in problems:
            print("  -", p)
        return 2

    if args.only:
        keep = set(args.only)
        suite.cases = [c for c in suite.cases if c.source in keep or c.name in keep]
    if "determinism" in args.skip:
        suite.determinism_seeds = []
    for c in suite.cases:
        if "mutation" in args.skip:
            c.mutations = []
        if "efficiency" in args.skip:
            c.min_efficiency = 0.0

    print(f"kernelbench: {len(suite.cases)} kernels in {suite.root}\n")
    ev_ = Evaluator(suite, use_nvcc=False if args.no_nvcc else None, arch=args.arch)
    try:
        results = ev_.run()
    finally:
        ev_.cleanup()

    device = "unknown"
    for r in results:
        if r.kb:
            device = r.kb["device"]
            break

    print()
    print(terminal(results, suite.name, device))
    write(results, suite.name, device, args.json_path, args.md_path,
          args.html_path)

    regressed = _baseline(results, suite.name, device, args)

    overall = sum(r.score for r in results) / len(results) if results else 0.0
    if overall < args.fail_under:
        print(f"\noverall {overall:.0%} is below the --fail-under gate of {args.fail_under:.0%}")
        return 1
    if regressed:
        return 1
    return 0


def _baseline(results, suite_name, device, args) -> bool:
    """Compare against a stored run. Returns True if anything regressed.

    Two kinds of regression are worth failing a build over, and they are different: a *score*
    that fell means something broke, and a *time* that rose means something got slower while
    staying correct. The second one is invisible to every other check in this harness, and it
    is the one that accumulates one careless commit at a time.
    """
    if not args.baseline:
        return False
    from .report import to_json
    path = Path(args.baseline)
    if args.update_baseline or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_json(results, suite_name, device))
        print(f"\nbaseline written to {path}")
        return False

    old = json.loads(path.read_text())
    prev = {k["name"]: k for k in old.get("kernels", [])}
    problems = []
    for r in results:
        was = prev.get(r.case.name)
        if was is None:
            continue
        if r.score < was["score"] - 1e-9:
            problems.append(f"{r.case.name}: score {was['score']:.0%} -> {r.score:.0%}")
        # Times only compare on the same device; a different card is a different baseline.
        if old.get("device") != device:
            continue
        best_now = min((v["median_ms"] for v in (r.kb or {}).get("variants", [])
                        if v.get("timed")), default=0.0)
        best_was = min((v["median_ms"] for v in was.get("variants", []) if v.get("timed")),
                       default=0.0)
        if best_was > 0 and best_now > best_was * (1 + args.regress_pct / 100.0):
            problems.append(f"{r.case.name}: best variant {best_was:.4f} ms -> "
                            f"{best_now:.4f} ms ({100*(best_now/best_was-1):+.0f}%)")
    if problems:
        print(f"\n{len(problems)} regression(s) against {path}:")
        for p in problems:
            print("  -", p)
        return True
    print(f"\nno regressions against {path}")
    return False


if __name__ == "__main__":
    sys.exit(main())
