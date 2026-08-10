#!/usr/bin/env python3
"""The demo pages must agree with the notebook they were ported from.

The pages under `demo/` claim to use "the same equations as the notebooks". This
checks that claim the only way that means anything — by running both implementations
and requiring them to agree:

  1. the hardware and model catalogs must be identical on every page that carries them
  2. `serving-console.html`'s `predict()` must match the notebook's `predict()` over a
     grid of configurations, including on which ones are infeasible
  3. `will-it-fit.html`'s `fit()` must match the notebook's memory arithmetic at TP=1
  4. `kv-cache.html` must match the long-context notebook's `kv_total_bytes()` for every
     architecture and context length
  5. `speculation.html` must match the speculative-decoding notebook's `expected_tokens()`
     and `speedup()`
  6. `batching.html`'s scheduler simulation must satisfy its invariants: every request
     served exactly once, no slot ever double-booked, continuous never worse than static

The notebook is the source of truth. Every model is read from where it lives, so none
can be edited without this noticing.

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
LONGCTX = REPO / "LongContext_KV_Compression_Serving.ipynb"
SPECNB = REPO / "Speculative_Decoding_Advanced_Serving.ipynb"
CONSOLE = REPO / "demo" / "serving-console.html"
FITPAGE = REPO / "demo" / "will-it-fit.html"
KVPAGE = REPO / "demo" / "kv-cache.html"
SPECPAGE = REPO / "demo" / "speculation.html"
BATCHPAGE = REPO / "demo" / "batching.html"
PAGES = (CONSOLE, FITPAGE, KVPAGE, SPECPAGE, BATCHPAGE)

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


def load_js_model(page: Path) -> str:
    """Slice a page's model out of its HTML, between the page's own markers."""
    html = page.read_text(encoding="utf-8")
    m = re.search(r"/\* MODEL-START.*?\*/(.*?)/\* MODEL-END \*/", html, re.S)
    if not m:
        raise SystemExit(f"MODEL-START / MODEL-END markers not found in {page.name}")
    return m.group(1)


def run_node(script: str, stdin: str = "") -> str:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.js"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(["node", str(path)], input=stdin,
                                  capture_output=True, text=True, check=True)
        except FileNotFoundError:
            raise SystemExit("node is required to run the demo models "
                             "(it is preinstalled on GitHub runners)")
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"a demo page's JavaScript failed to run:\n{exc.stderr[-2000:]}")
    return proc.stdout


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
    script = load_js_model(CONSOLE) + JS_HARNESS.replace("%KEYS%", keys)
    return json.loads(run_node(script, json.dumps(cases)))


CATALOG_HARNESS = """
const out = {};
for (const n of ["GPUS", "MODELS", "PRECISION"])
  if (eval("typeof " + n) !== "undefined") out[n] = eval(n);
process.stdout.write(JSON.stringify(out));
"""


def check_catalogs(ns: dict) -> list[str]:
    """Pages carry their own copy of the hardware tables. They must not drift."""
    want = {"GPUS": ns["GPUS"], "MODELS": ns["MODELS"], "PRECISION": ns["PRECISION"]}
    problems = []
    for page in PAGES:
        got = json.loads(run_node(load_js_model(page) + CATALOG_HARNESS))
        for table in want:
            if table not in got:
                continue          # not every tool needs every table
            for key in sorted(set(want[table]) | set(got[table])):
                a, b = want[table].get(key), got[table].get(key)
                if a is None or b is None:
                    problems.append(f"{page.name}: {table}['{key}'] is missing from "
                                    + ("the page" if b is None else "the notebook"))
                    continue
                for field in sorted(set(a) | set(b)):
                    if a.get(field) != b.get(field):
                        problems.append(f"{page.name}: {table}['{key}'].{field} "
                                        f"is {b.get(field)}, notebook says {a.get(field)}")
    return problems


FIT_HARNESS = """
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(cases.map(c => {
  const r = fit(c.gpu, c.model, c.prec, c.ctx);
  return {poolGb: r.poolGb, concurrent: r.concurrent, kvPerTok: r.kvPerTok};
})));
"""


def check_fit(ns: dict, cases: list[dict]) -> list[str]:
    """will-it-fit.html answers a subset of the same question: memory on one GPU.

    Compared against the notebook at TP=1. Where the notebook rejects a configuration
    outright (its topology rule demands room for 32 concurrent requests, which is a
    stricter bar than "can serve at all") there is nothing to compare, so those are
    skipped rather than forced into a false equivalence.
    """
    predict = ns["predict"]
    single = [c for c in cases if c["kvFp8"] is False]
    js = json.loads(run_node(load_js_model(FITPAGE) + FIT_HARNESS, json.dumps(single)))
    problems, compared = [], 0
    for case, got in zip(single, js):
        py = predict(case["gpu"], case["model"], case["prec"], batch=1,
                     ctx=case["ctx"], tp=1)
        if "error" in py:
            continue
        compared += 1
        pairs = (("kv_pool_gb", "poolGb", py["kv_pool_gb"], got["poolGb"]),
                 ("max_concurrency", "concurrent", py["max_concurrency"], got["concurrent"]))
        for py_name, js_name, a, b in pairs:
            if abs(a - b) > max(1e-9, abs(a) * 1e-9):
                problems.append(f"{case['gpu']}/{case['model']}/{case['prec']} "
                                f"ctx={case['ctx']}: {py_name}={a} but page's {js_name}={b}")
    return problems, compared


def load_defs(notebook: Path, names: tuple[str, ...]) -> dict:
    """Pull named function definitions out of a notebook and exec just those.

    Executing the whole cell would drag in matplotlib and draw figures; the functions
    are what the pages claim to reproduce, so the functions are what we lift.
    """
    nb = json.loads(notebook.read_text())
    ns: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for name in names:
            m = re.search(rf"^def {name}\(.*?(?=^\S|\Z)", src, re.S | re.M)
            if m and name not in ns:
                exec(compile(m.group(0), str(notebook), "exec"), ns)   # noqa: S102
    missing = [n for n in names if n not in ns]
    if missing:
        raise SystemExit(f"could not find {missing} in {notebook.name}")
    return ns


KV_HARNESS = """
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(cases.map(c =>
  ({total: kvTotalBytes(ARCH[c.arch], c.ctx, c.kvBytes),
    perTok: kvBytesPerToken(ARCH[c.arch], c.kvBytes)}))));
"""


def check_kv() -> tuple[list[str], int]:
    """kv-cache.html against the long-context notebook's own KV arithmetic."""
    ns = load_defs(LONGCTX, ("kv_bytes_per_token", "kv_total_bytes"))
    nb = json.loads(LONGCTX.read_text())
    models = None
    for cell in nb["cells"]:
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        if cell["cell_type"] == "code" and "MODELS = {" in src:
            m = re.search(r"^MODELS = \{.*?^\}", src, re.S | re.M)
            scope: dict = {}
            exec(compile(m.group(0), str(LONGCTX), "exec"), scope)   # noqa: S102
            models = scope["MODELS"]
            break
    if models is None:
        raise SystemExit("no MODELS table found in the long-context notebook")

    ctxs = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 524288, 1048576]
    cases = [{"arch": a, "ctx": c, "kvBytes": b}
             for a in models for c in ctxs for b in (1, 2)]
    js = json.loads(run_node(load_js_model(KVPAGE) + KV_HARNESS, json.dumps(cases)))
    problems = []
    for case, got in zip(cases, js):
        m = models[case["arch"]]
        want_total = ns["kv_total_bytes"](m, case["ctx"], case["kvBytes"])
        want_per = ns["kv_bytes_per_token"](m, case["kvBytes"])
        if got["total"] != want_total:
            problems.append(f"{case['arch']} @ {case['ctx']} tokens, {case['kvBytes']}B KV: "
                            f"page says {got['total']}, notebook says {want_total}")
        if got["perTok"] != want_per:
            problems.append(f"{case['arch']} per-token, {case['kvBytes']}B KV: "
                            f"page says {got['perTok']}, notebook says {want_per}")
    return problems, len(cases)


SPEC_HARNESS = """
const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(cases.map(c =>
  ({e: expectedTokens(c.alpha, c.k), gross: speedup(c.alpha, c.k, c.cost, "memory").gross}))));
"""


def check_spec() -> tuple[list[str], int]:
    """speculation.html against the speculative-decoding notebook's formulas."""
    ns = load_defs(SPECNB, ("expected_tokens", "speedup"))
    cases = [{"alpha": a / 100, "k": k, "cost": c / 100}
             for a in range(5, 100, 5) for k in range(1, 13) for c in (1, 5, 10, 20, 30)]
    js = json.loads(run_node(load_js_model(SPECPAGE) + SPEC_HARNESS, json.dumps(cases)))
    problems = []
    for case, got in zip(cases, js):
        want_e = ns["expected_tokens"](case["alpha"], case["k"])
        want_s = ns["speedup"](case["alpha"], case["k"], case["cost"])
        for label, a, b in (("E[tokens]", want_e, got["e"]), ("gross speedup", want_s, got["gross"])):
            if abs(a - b) > max(1e-9, abs(a) * 1e-9):
                problems.append(f"alpha={case['alpha']} k={case['k']} cost={case['cost']}: "
                                f"{label} notebook={a} page={b}")
    return problems, len(cases)


BATCH_HARNESS = """
const out = [];
for (const slots of [2, 4, 8, 16, 24])
  for (const spread of [0, 1, 2, 3])
    for (const seed of [1, 7, 99]) {
      const s = simulate({slots, spread, seed});
      const audit = (res) => {
        const seen = new Map();
        for (const row of res.timeline) {
          const live = row.filter(r => r !== -1);
          if (new Set(live).size !== live.length) return "a request occupied two slots at once";
          for (const r of live) seen.set(r, (seen.get(r) || 0) + 1);
          if (row.length !== slots) return "slot count changed mid-run";
        }
        if (res.done !== s.nReq) return `only ${res.done}/${s.nReq} requests finished`;
        for (let i = 0; i < s.nReq; i++)
          if ((seen.get(i) || 0) !== s.lengths[i]) return `request ${i} ran ${seen.get(i)} steps, needed ${s.lengths[i]}`;
        return null;
      };
      out.push({slots, spread, seed,
                staticErr: audit(s.static), contErr: audit(s.continuous),
                staticSteps: s.static.makespan, contSteps: s.continuous.makespan,
                staticUtil: s.static.utilisation, contUtil: s.continuous.utilisation});
    }
process.stdout.write(JSON.stringify(out));
"""


def check_batching() -> tuple[list[str], int]:
    """The scheduler simulation has no Python twin, so check its invariants instead."""
    runs = json.loads(run_node(load_js_model(BATCHPAGE) + BATCH_HARNESS))
    problems = []
    for r in runs:
        tag = f"slots={r['slots']} spread={r['spread']} seed={r['seed']}"
        for mode in ("static", "cont"):
            if r[f"{mode}Err"]:
                problems.append(f"{tag}: {mode} — {r[f'{mode}Err']}")
        if r["contSteps"] > r["staticSteps"]:
            problems.append(f"{tag}: continuous took {r['contSteps']} steps, static only "
                            f"{r['staticSteps']} — continuous can never be worse")
        if r["contUtil"] < r["staticUtil"] - 1e-9:
            problems.append(f"{tag}: continuous utilisation {r['contUtil']:.3f} below "
                            f"static's {r['staticUtil']:.3f}")
        if r["spread"] == 0 and r["contSteps"] != r["staticSteps"]:
            problems.append(f"{tag}: with identical lengths there is no straggler, so both "
                            f"schedulers must agree ({r['contSteps']} vs {r['staticSteps']})")
    return problems, len(runs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller grid, for a fast local loop")
    args = ap.parse_args()

    ns = load_python_model()
    predict = ns["predict"]
    cases = build_cases(args.quick)

    catalog_problems = check_catalogs(ns)
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

    fit_problems, fit_compared = check_fit(ns, cases)
    kv_problems, kv_n = check_kv()
    spec_problems, spec_n = check_spec()
    batch_problems, batch_n = check_batching()

    print(f"catalogs   : {'identical on every page' if not catalog_problems else str(len(catalog_problems)) + ' DRIFTED'}")
    print(f"console    : {len(cases):,} cases — {feasible:,} feasible, "
          f"{infeasible:,} rejected by both")
    print(f"will-it-fit: {fit_compared:,} cases compared at TP=1")
    print(f"kv-cache   : {kv_n:,} cases vs the long-context notebook")
    print(f"speculation: {spec_n:,} cases vs the speculative-decoding notebook")
    print(f"batching   : {batch_n:,} simulations, invariants checked")

    for label, problems in (("catalog", catalog_problems), ("will-it-fit", fit_problems),
                            ("kv-cache", kv_problems), ("speculation", spec_problems),
                            ("batching", batch_problems)):
        if problems:
            print(f"\n{len(problems)} {label} disagreement(s) — first 12:")
            for line in problems[:12]:
                print("  " + line)

    if mismatches:
        print(f"\n{len(mismatches)} disagreement(s) between the notebook and the console — first 12:")
        for case, key, a, b in mismatches[:12]:
            print(f"  {case['gpu']}/{case['model']}/{case['prec']} batch={case['batch']} "
                  f"ctx={case['ctx']} kv_fp8={case['kvFp8']} alpha={case['specAlpha']}")
            print(f"      {key}: notebook={a}  console={b}")
    if (mismatches or catalog_problems or fit_problems or kv_problems
            or spec_problems or batch_problems):
        return 1
    print("\nevery demo page reproduces the notebook exactly, to 1e-9 relative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
