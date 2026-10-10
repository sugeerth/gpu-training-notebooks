"""Does the package still agree with the notebooks it was lifted from?

This is the check that makes the package safe to be canonical. Every model in `servingkit` was
lifted verbatim from a notebook, and the notebook remains where the reasoning is explained — so
the two must not drift. `run_checks()` pulls each function's source back out of its notebook,
executes it in isolation, and requires it to agree with the package over a grid of inputs.

It is the same discipline `tools/verify_console.py` applies to the demo pages' JavaScript, and
it closes the loop: notebook, package and browser all now have to say the same thing, and the
only way to change a number is to change it in one place and watch three checks fail.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .kernels import repo_root


def load_from_notebook(notebook: Path, names: tuple[str, ...]) -> dict:
    """Pull named functions — and the module-level constants they close over — out of a notebook.

    Executing whole cells would drag in matplotlib and draw figures, so this parses each code
    cell with `ast` and re-executes only two kinds of top-level statement: the requested `def`s,
    and assignments to ALL-CAPS names. That second part is not optional: the notebooks' models
    close over their own `GPUS`, `MODELS` and `BW_EFF` tables, and lifting the function without
    them gives a `NameError` — which is how this function came to use `ast` instead of a regex.

    Constants are taken from the notebook, not from `servingkit.catalog`, deliberately. It means
    the comparison covers the catalog as well as the arithmetic: if a GPU's bandwidth differs
    between the two copies, the check fails on every configuration that touches it.
    """
    import ast

    nb = json.loads(notebook.read_text())
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        cells.append(src)

    wanted = set(names)
    pieces: list[str] = []
    found: set[str] = set()
    for src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines(keepends=True)
        for node in tree.body:
            take = False
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                take, label = True, node.name
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
                # Flatten tuple targets: `BW_EFF, FLOP_EFF = 0.75, 0.60` is one statement with
                # an ast.Tuple target, and missing it cost an afternoon of NameErrors.
                flat: list[str] = []
                for t in targets:
                    if isinstance(t, ast.Name):
                        flat.append(t.id)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        flat += [e.id for e in t.elts if isinstance(e, ast.Name)]
                caps = [nm for nm in flat if nm.isupper()]
                if caps:
                    take, label = True, ", ".join(caps)
            if take:
                seg = "".join(lines[node.lineno - 1:node.end_lineno])
                pieces.append(seg)
                found.update(label.split(", "))

    ns: dict = {"__name__": "notebook_lift"}
    exec("import math", ns)  # noqa: S102 - our own notebook

    # Piece by piece, tolerating the ones that do not stand alone. A notebook has constants that
    # reference helper functions further down the cell — `POLICIES = {"h2o": policy_h2o, ...}` —
    # and those are not what is being checked here. Skipping them is correct; skipping something
    # the wanted functions actually need shows up immediately as a failure to call them, which
    # the assertion below turns into a clear error rather than a mystery.
    skipped: list[str] = []
    for seg in pieces:
        try:
            exec(seg, ns)  # noqa: S102 - our own notebook
        except (NameError, ImportError, AttributeError) as exc:
            skipped.append(f"{seg.splitlines()[0][:50]} -> {exc}")

    missing = [n for n in names if n not in ns]
    if missing:
        raise SystemExit(f"could not find {missing} in {notebook.name} "
                         f"(found: {sorted(found)}; skipped: {skipped})")
    return ns


def _close(a, b, tol: float = 1e-9) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= max(tol, abs(a) * tol)
    return a == b


def _compare(label: str, got: dict, want: dict, problems: list[str]) -> int:
    keys = set(got) & set(want)
    for k in sorted(keys):
        if isinstance(got[k], dict):
            continue
        if not _close(got[k], want[k]):
            problems.append(f"{label}: {k} package={got[k]!r} notebook={want[k]!r}")
    return len(keys)


def check_agents(root: Path, problems: list[str]) -> int:
    from .agents import agent_run, tool_gap_policy
    ns = load_from_notebook(root / "Agent_Workloads_On_The_Metal.ipynb",
                            ("agent_run", "tool_gap_policy"))
    n = 0
    for turns in (2, 5, 20, 47, 80):
        for tool in (100, 800, 2000, 6000):
            for hit in (0.0, 0.8, 0.95, 0.99, 1.0):
                for fan in (1, 6):
                    kw = dict(turns=turns, tool_result_tokens=tool, prefix_hit_rate=hit,
                              fanout=fan)
                    n += _compare(f"agent_run{kw}", agent_run(**kw), ns["agent_run"](**kw),
                                  problems)
    for ctx in (4000, 20000, 200000):
        for lat in (0.0, 2.0, 30.0):
            kw = dict(context_tokens=ctx, tool_latency_s=lat)
            n += _compare(f"tool_gap_policy{kw}", tool_gap_policy(**kw),
                          ns["tool_gap_policy"](**kw), problems)
    return n


def check_training(root: Path, problems: list[str]) -> int:
    from .training import training_plan
    ns = load_from_notebook(root / "Training_Kernels_And_Memory.ipynb", ("training_plan",))
    n = 0
    for params in (1.24, 8.0, 70.6):
        for recipe in ("fp32", "bf16+master", "bf16+8bit", "lora-r16"):
            for zero in (0, 1, 2, 3):
                for ckpt in ("none", "selective", "full"):
                    for flash in (True, False):
                        kw = dict(params_b=params, recipe=recipe, zero_stage=zero,
                                  checkpointing=ckpt, flash_attention=flash, gpus=8)
                        n += _compare(f"training_plan({params}B/{recipe}/z{zero})",
                                      training_plan(**kw), ns["training_plan"](**kw), problems)
    return n


def check_serving(root: Path, problems: list[str]) -> int:
    from .serving import predict
    ns = load_from_notebook(root / "Serving_WhatIf_Console.ipynb", ("predict",))
    # The notebook's predict() closes over its own catalogs, so this compares the whole
    # pipeline — catalog included — not just the arithmetic.
    from .catalog import GPUS
    n = 0
    for gpu in list(GPUS)[:6]:
        for model in ("Qwen2.5-0.5B", "Llama-3.1-8B", "Llama-3.1-70B"):
            for prec in ("fp16", "fp8", "int4"):
                for batch in (1, 32, 256):
                    for ctx in (1024, 16384):
                        got = predict(gpu, model, prec, batch=batch, ctx=ctx)
                        want = ns["predict"](gpu, model, prec, batch=batch, ctx=ctx)
                        if ("error" in got) != ("error" in want):
                            problems.append(
                                f"predict({gpu}/{model}/{prec}/b{batch}/c{ctx}): feasibility "
                                f"differs — package={'error' in got}, "
                                f"notebook={'error' in want}")
                            continue
                        if "error" in got:
                            n += 1
                            continue
                        n += _compare(f"predict({gpu}/{model}/{prec}/b{batch}/c{ctx})",
                                      got, want, problems)
    return n


def check_kv(root: Path, problems: list[str]) -> int:
    from .kv import kv_bytes_per_token, kv_total_bytes
    ns = load_from_notebook(root / "LongContext_KV_Compression_Serving.ipynb",
                            ("kv_bytes_per_token", "kv_total_bytes"))
    from .catalog import MODELS
    n = 0
    for name, m in MODELS.items():
        for ctx in (1024, 8192, 131072, 1048576):
            for kvb in (1, 2):
                a, b = kv_total_bytes(m, ctx, kvb), ns["kv_total_bytes"](m, ctx, kvb)
                if not _close(a, b):
                    problems.append(f"kv_total_bytes({name},{ctx},{kvb}): {a} vs {b}")
                a, b = kv_bytes_per_token(m, kvb), ns["kv_bytes_per_token"](m, kvb)
                if not _close(a, b):
                    problems.append(f"kv_bytes_per_token({name},{kvb}): {a} vs {b}")
                n += 2
    return n


def check_spec(root: Path, problems: list[str]) -> int:
    from .spec import expected_tokens, speedup
    ns = load_from_notebook(root / "Speculative_Decoding_Advanced_Serving.ipynb",
                            ("expected_tokens", "speedup"))
    n = 0
    for alpha in [i / 20 for i in range(20)]:
        for k in range(1, 13):
            for c in (0.01, 0.05, 0.3):
                for f in ("expected_tokens", "speedup"):
                    got = (expected_tokens(alpha, k) if f == "expected_tokens"
                           else speedup(alpha, k, c))
                    want = (ns[f](alpha, k) if f == "expected_tokens" else ns[f](alpha, k, c))
                    if not _close(got, want):
                        problems.append(f"{f}({alpha},{k},{c}): {got} vs {want}")
                    n += 1
    return n


def check_step(root: Path, problems: list[str]) -> int:
    from .step import base_config, evaluate
    ns = load_from_notebook(root / "The_Optimization_Stack.ipynb", ("base_config", "evaluate"))
    n = 0
    from .step import STEP_MODELS
    for model in STEP_MODELS:
        for gpu in ("T4", "A100 80GB", "H100 SXM", "MI300X"):
            for batch in (1, 8, 32, 256):
                for ctx in (512, 2048, 32768):
                    cfg = base_config(model, gpu, batch, ctx)
                    n += _compare(f"evaluate({model}/{gpu}/b{batch}/c{ctx})",
                                  evaluate(cfg), ns["evaluate"](ns["base_config"](
                                      model, gpu, batch, ctx)), problems)
    return n


def check_lever_declarations(root: Path, problems: list[str]) -> int:
    """Every measured antagonism must be explained by a declaration.

    This is not a drift check — it is the package checking its own model of itself. A pair whose
    gains multiply to less than they should, with no shared resource declared, means a lever is
    lying about what it spends. Two were, on the first run, and this is how that was found.
    """
    from .stack import Stack
    from .workload import Workload
    n = 0
    for w in (Workload.chat(ctx=1024, batch=8), Workload.chat(ctx=2048, batch=32),
              Workload.chat(ctx=32768, batch=64)):
        s = Stack(w).all_applicable()
        for c in s.interactions():
            n += 1
            if c["synergy"] < 0.9 and not c["predicted_conflict"]:
                problems.append(
                    f"{c['a']} + {c['b']}: synergy {c['synergy']:.2f} with no shared resource "
                    f"declared — one of their `spends` is wrong")
    return n


CHECKS = [
    ("kv", check_kv),
    ("speculation", check_spec),
    ("decode step", check_step),
    ("serving", check_serving),
    ("training", check_training),
    ("agents", check_agents),
    ("lever declarations", check_lever_declarations),
]


def run_checks(root: Path | None = None) -> int:
    root = root or repo_root()
    problems: list[str] = []
    total = 0
    for label, fn in CHECKS:
        before = len(problems)
        n = fn(root, problems)
        total += n
        status = "ok" if len(problems) == before else f"{len(problems) - before} DISAGREE"
        print(f"{label:<20} {n:>7,} comparisons   {status}")
    print(f"{'':<20} {total:>7,} total")
    if problems:
        print(f"\n{len(problems)} disagreement(s) — first 15:")
        for line in problems[:15]:
            print("  " + line)
        return 1
    print("\nthe package, the notebooks and the levers' own declarations all agree")
    return 0


__all__ = ["run_checks", "load_from_notebook", "CHECKS"]
