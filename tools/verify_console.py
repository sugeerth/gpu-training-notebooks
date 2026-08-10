#!/usr/bin/env python3
"""The demo console must agree with the notebook it was ported from.

`demo/serving-console.html` claims to use "the same equations as the notebooks".
This checks that claim the only way that means anything: it runs the notebook's
Python `predict()` and the console's JavaScript `predict()` over the same grid of
configurations and requires them to agree — including on which configurations are
infeasible.

The notebook is the source of truth. Both models are read from where they live, so
neither can be edited without this noticing.

    python tools/verify_console.py            # the standard grid
    python tools/verify_console.py --quick    # a smaller grid, for a fast local loop

Needs Node.js (preinstalled on GitHub runners) and nothing else.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "Serving_WhatIf_Console.ipynb"
CONSOLE = REPO / "demo" / "serving-console.html"

# Compared exactly - a disagreement here is a different decision, not a rounding difference.
EXACT = ("tp", "batch", "max_concurrency", "bound")
# Compared to 1e-9 relative: the same arithmetic in two languages, not two approximations.
NUMERIC = ("decode_tps", "tpot_ms", "ttft_ms", "spec_mult", "kv_pool_gb", "cpm")

JS_HARNESS = """
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
const out = cases.map(c => {
  const r = predict(c.gpu, c.model, c.prec, c.batch, c.ctx,
                    {kvFp8: c.kvFp8, specAlpha: c.specAlpha, prefixHit: c.prefixHit});
  if (r.error) return {error: true};
  const o = {};
  for (const k of %KEYS%) o[k] = r[k];
  return o;
});
process.stdout.write(JSON.stringify(out));
"""


def load_python_model() -> dict:
    """Execute the notebook cell that defines predict(), and hand back its namespace."""
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        if "def predict(" in src:
            ns: dict = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, str(NOTEBOOK), "exec"), ns)   # noqa: S102 - our own notebook
            return ns
    raise SystemExit(f"no cell defining predict() found in {NOTEBOOK.name}")


def load_js_model() -> str:
    """Slice the console's model out of the page, between its own markers."""
    html = CONSOLE.read_text(encoding="utf-8")
    m = re.search(r"/\* MODEL-START.*?\*/(.*?)/\* MODEL-END \*/", html, re.S)
    if not m:
        raise SystemExit(f"MODEL-START / MODEL-END markers not found in {CONSOLE.name}")
    return m.group(1)


def build_cases(quick: bool) -> list[dict]:
    ns = load_python_model()
    gpus, models, precs = list(ns["GPUS"]), list(ns["MODELS"]), list(ns["PRECISION"])
    batches = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
    ctxs = [512, 1024, 2048, 4096, 8192, 16384, 32768, 131072]
    if quick:
        batches, ctxs = [1, 32, 256], [1024, 16384]

    cases = [dict(gpu=g, model=m, prec=p, batch=b, ctx=c,
                  kvFp8=False, specAlpha=0.0, prefixHit=0.0)
             for g in gpus for m in models for p in precs for b in batches for c in ctxs]

    # ...then every flag combination over a deterministic sample, so the optional paths
    # (fp8 KV, speculation, prefix hits) are covered without a full cross product.
    rng = random.Random(11)
    base = list(cases)
    n = 20 if quick else 90
    for kv_fp8 in (False, True):
        for alpha in (0.0, 0.5, 0.75, 0.95):
            for hit in (0.0, 0.6, 0.95):
                for c in rng.sample(base, min(n, len(base))):
                    cases.append({**c, "kvFp8": kv_fp8, "specAlpha": alpha, "prefixHit": hit})
    return cases


def run_js(cases: list[dict]) -> list[dict]:
    keys = json.dumps(list(EXACT) + list(NUMERIC))
    script = load_js_model() + JS_HARNESS.replace("%KEYS%", keys)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.js"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(["node", str(path)], input=json.dumps(cases),
                                  capture_output=True, text=True, check=True)
        except FileNotFoundError:
            raise SystemExit("node is required to run the console's model (it is preinstalled on CI runners)")
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"the console's JavaScript model failed to run:\n{exc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller grid, for a fast local loop")
    args = ap.parse_args()

    ns = load_python_model()
    predict = ns["predict"]
    cases = build_cases(args.quick)
    js_results = run_js(cases)

    mismatches, feasible, infeasible = [], 0, 0
    for case, js in zip(cases, js_results):
        py = predict(case["gpu"], case["model"], case["prec"], batch=case["batch"],
                     ctx=case["ctx"], kv_fp8=case["kvFp8"], spec_alpha=case["specAlpha"],
                     prefix_hit=case["prefixHit"])
        py_bad, js_bad = "error" in py, bool(js.get("error"))
        if py_bad or js_bad:
            if py_bad != js_bad:
                mismatches.append((case, "feasibility",
                                   "infeasible" if py_bad else "feasible",
                                   "infeasible" if js_bad else "feasible"))
            else:
                infeasible += 1
            continue
        feasible += 1
        for k in EXACT:
            if py[k] != js[k]:
                mismatches.append((case, k, py[k], js[k]))
        for k in NUMERIC:
            a, b = py[k], js[k]
            if abs(a - b) > max(1e-9, abs(a) * 1e-9):
                mismatches.append((case, k, a, b))

    print(f"cases      : {len(cases):,}")
    print(f"  feasible : {feasible:,}")
    print(f"  rejected : {infeasible:,}  (both implementations agree these cannot run)")
    if mismatches:
        print(f"\n{len(mismatches)} disagreement(s) between the notebook and the console — first 12:")
        for case, key, a, b in mismatches[:12]:
            print(f"  {case['gpu']}/{case['model']}/{case['prec']} batch={case['batch']} "
                  f"ctx={case['ctx']} kv_fp8={case['kvFp8']} alpha={case['specAlpha']}")
            print(f"      {key}: notebook={a}  console={b}")
        return 1
    print("\nthe console reproduces the notebook exactly, to 1e-9 relative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
