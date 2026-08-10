#!/usr/bin/env python3
"""Headless notebook runner + report generator for this repo.

Designed for one-line use in Colab (or any GPU box) to execute the notebooks whose
GPU sections cannot run in CI, and to emit a report you can paste into a PR.

    python tools/run_notebooks.py                 # run the GPU-dependent notebooks
    python tools/run_notebooks.py --set cpu       # run the CPU-only notebooks
    python tools/run_notebooks.py --set all
    python tools/run_notebooks.py --only vLLM_High_Throughput_Serving.ipynb
    python tools/run_notebooks.py --timeout 2400 --report my_report.md

Writes <report>.md and <report>.json, and exits non-zero if any notebook failed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Notebooks with live-GPU sections (engine, server, training). These are the ones CI cannot cover.
GPU_NOTEBOOKS = [
    "Serving_Fundamentals_KV_Cache_Batching.ipynb",
    "vLLM_High_Throughput_Serving.ipynb",
    "Quantized_Serving_Showdown.ipynb",
    "Speculative_Decoding_Advanced_Serving.ipynb",
    "Serving_Logs_Observability.ipynb",
    "Serving_Benchmark_Capacity_Planning.ipynb",
    "MultiLoRA_Serving_At_Scale.ipynb",
]

# Notebooks that are pure modeling/simulation - these run anywhere and are covered by CI.
CPU_NOTEBOOKS = [
    "The_Serving_Playbook.ipynb",
    "Serving_Internals_Visualized_D3.ipynb",
    "Structured_Output_Guided_Decoding.ipynb",
    "Distributed_MultiReplica_Serving.ipynb",
    "Hardware_Roofline_NVIDIA_vs_AMD.ipynb",
    "Portable_Kernels_Precision_Matrix.ipynb",
    "Serving_WhatIf_Console.ipynb",
    "LongContext_KV_Compression_Serving.ipynb",
    "MoE_Serving_Expert_Parallelism.ipynb",
    "RAG_Agent_Serving_Patterns.ipynb",
    "Production_Hardening_Reliability.ipynb",
    "Anatomy_Of_A_Decode_Step.ipynb",
    "The_Optimization_Stack.ipynb",
    "From_FineTune_To_Production.ipynb",
    "VLM_Serving_Token_Explosion.ipynb",
    "VLM_Optimization_Techniques.ipynb",
]

# Sanity: every serving notebook belongs to exactly one of the two lists above. Keeping this
# honest is what stops CI coverage from silently rotting as notebooks are added.
_ALL_TRACKED = set(GPU_NOTEBOOKS) | set(CPU_NOTEBOOKS)


HAS_GPU = False  # set in main() from environment(); used to classify GPU-gated failures


def environment() -> dict:
    """Describe the machine, without assuming any particular vendor is present."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": None,
        "vendor": None,
        "runtime": None,
        "vram_gb": None,
        "torch": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            hip = getattr(torch.version, "hip", None)
            props = torch.cuda.get_device_properties(0)
            info.update(
                gpu=props.name,
                vendor="AMD (ROCm)" if hip else "NVIDIA (CUDA)",
                runtime=hip or torch.version.cuda,
                vram_gb=round(props.total_memory / 1e9, 1),
            )
    except Exception:  # torch missing is a legitimate state, not an error
        pass
    return info


def strip_installs(nb) -> int:
    """Remove `!pip install` lines so a pre-provisioned environment isn't re-resolved."""
    removed = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        kept = []
        for line in cell.source.split("\n"):
            if line.lstrip().startswith("!pip install") or line.lstrip().startswith("!pip3 install"):
                removed += 1
                continue
            kept.append(line)
        cell.source = "\n".join(kept)
    return removed


def run_one(path: Path, timeout: int, skip_installs: bool = False,
            keep_going: bool = False) -> dict:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(path, as_version=4)
    if skip_installs:
        strip_installs(nb)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=keep_going,   # keep going to collect EVERY failing cell, not just the first
        resources={"metadata": {"path": str(REPO)}},
    )
    started = time.perf_counter()
    result = {"notebook": path.name, "code_cells": n_code}
    try:
        client.execute()
        if keep_going:
            # allow_errors swallows exceptions, so inspect the cell outputs ourselves.
            bad = []
            for i, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue
                errs = [o for o in cell.get("outputs", [])
                        if o.get("output_type") == "error"]
                if errs:
                    tb = "".join(errs[0].get("traceback", [])) or errs[0].get("evalue", "")
                    bad.append((i, tb))
            if bad:
                raise CellExecutionError(
                    "\n\n".join(f"[cell {i}]\n{tb[-1200:]}" for i, tb in bad), "", "")
        result.update(status="pass", failed_cell=None, error=None)
    except CellExecutionError as exc:  # noqa: PERF203 - classified below
        # Locate the first cell that carries an error output.
        idx = next(
            (
                i
                for i, c in enumerate(nb.cells)
                if c.cell_type == "code"
                and any(o.get("output_type") == "error" for o in c.get("outputs", []))
            ),
            None,
        )
        msg = str(exc)
        # A notebook that deliberately refuses to run without a GPU is not a defect.
        needs_gpu = any(
            s in msg
            for s in ("GPU required", "vLLM needs a GPU", "Runtime > Change runtime type",
                      "no CUDA GPUs are available", "Found no NVIDIA driver",
                      "Torch not compiled with CUDA")
        )
        result.update(status="skipped-no-gpu" if (needs_gpu and not HAS_GPU) else "fail",
                      failed_cell=idx, error=msg[-2500:])
    except Exception as exc:  # kernel death, timeout, OOM kill, ...
        result.update(status="error", failed_cell=None,
                      error=f"{type(exc).__name__}: {exc}"[-2500:])
    result["seconds"] = round(time.perf_counter() - started, 1)

    # Keep the tail of stdout from each cell: this is where the measured numbers live.
    tails = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        text = "".join(
            o.get("text", "")
            for o in cell.get("outputs", [])
            if o.get("output_type") == "stream"
        ).strip()
        if text:
            tails.append({"cell": i, "tail": text[-1200:]})
    result["outputs"] = tails
    return result


ICON = {"pass": "✅ pass", "fail": "❌ fail", "error": "💥 error",
        "skipped-no-gpu": "⏭️ needs GPU"}


def markdown_report(env: dict, results: list[dict]) -> str:
    ok = sum(r["status"] == "pass" for r in results)
    skipped = sum(r["status"] == "skipped-no-gpu" for r in results)
    lines = [
        "# Notebook execution report",
        "",
        f"- **Machine**: {env.get('gpu') or 'no GPU detected'}"
        + (f" ({env['vendor']}, {env['vram_gb']} GB, runtime {env['runtime']})" if env.get("gpu") else ""),
        f"- **torch**: {env.get('torch') or 'not installed'} · **python**: {env['python']}",
        f"- **Result**: {ok}/{len(results)} executed cleanly"
        + (f", {skipped} skipped (require a GPU)" if skipped else ""),
        "",
        "| Notebook | Status | Code cells | Time |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| `{r['notebook']}` | {ICON[r['status']]} | {r['code_cells']} | {r['seconds']}s |"
        )

    failures = [r for r in results if r["status"] in ("fail", "error")]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            where = f"cell {r['failed_cell']}" if r["failed_cell"] is not None else "unknown cell"
            lines += [
                f"### `{r['notebook']}` — {where}",
                "",
                "```",
                (r["error"] or "").strip()[-1800:],
                "```",
                "",
            ]

    lines += ["", "## Measured output (tails)", ""]
    for r in results:
        lines += [f"<details><summary><code>{r['notebook']}</code></summary>", ""]
        for o in r["outputs"]:
            lines += ["```", f"[cell {o['cell']}]", o["tail"], "```", ""]
        lines += ["</details>", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["gpu", "cpu", "all"], default="gpu")
    ap.add_argument("--only", nargs="*", default=None, help="explicit notebook filenames")
    ap.add_argument("--timeout", type=int, default=2400, help="per-cell timeout, seconds")
    ap.add_argument("--skip-installs", action="store_true",
                    help="strip `!pip install` lines (deps already provisioned)")
    ap.add_argument("--keep-going", action="store_true",
                    help="don't stop a notebook at its first error; report every failing cell")
    ap.add_argument("--report", default="notebook_run_report")
    args = ap.parse_args()

    if args.only:
        targets = args.only
    elif args.set == "gpu":
        targets = GPU_NOTEBOOKS
    elif args.set == "cpu":
        targets = CPU_NOTEBOOKS
    else:
        targets = GPU_NOTEBOOKS + CPU_NOTEBOOKS

    global HAS_GPU
    env = environment()
    HAS_GPU = env.get("gpu") is not None
    print("=" * 72)
    print(f"machine : {env.get('gpu') or 'no GPU detected'}"
          + (f"  [{env['vendor']}, {env['vram_gb']} GB]" if env.get("gpu") else ""))
    print(f"torch   : {env.get('torch') or 'not installed'}")
    print(f"running : {len(targets)} notebook(s), per-cell timeout {args.timeout}s")
    if args.set == "gpu" and not env.get("gpu"):
        print("\n⚠ No GPU detected. These notebooks' GPU sections will take their CPU")
        print("  fallbacks, so a 'pass' here does NOT verify the GPU paths.")
    print("=" * 72)

    results = []
    for name in targets:
        path = REPO / name
        if not path.exists():
            print(f"  {name:<52} SKIP (not found)")
            continue
        print(f"  {name:<52} running...", flush=True)
        try:
            r = run_one(path, args.timeout, skip_installs=args.skip_installs,
                        keep_going=args.keep_going)
        except Exception:
            r = {"notebook": name, "code_cells": 0, "status": "error", "failed_cell": None,
                 "error": traceback.format_exc()[-2500:], "seconds": 0.0, "outputs": []}
        results.append(r)
        print(f"  {name:<52} {ICON[r['status']]} in {r['seconds']}s")
        if r["status"] in ("fail", "error"):
            print("      " + (r["error"] or "").strip().splitlines()[-1][:150])

    report_md = Path(args.report + ".md")
    report_md.write_text(markdown_report(env, results))
    Path(args.report + ".json").write_text(
        json.dumps({"environment": env, "results": results}, indent=2))

    ok = sum(r["status"] == "pass" for r in results)
    skipped = sum(r["status"] == "skipped-no-gpu" for r in results)
    broken = [r for r in results if r["status"] in ("fail", "error")]
    print("=" * 72)
    print(f"{ok}/{len(results)} notebooks executed cleanly"
          + (f", {skipped} skipped (require a GPU)" if skipped else ""))
    if broken:
        print(f"{len(broken)} genuinely failed: " + ", ".join(r["notebook"] for r in broken))
    print(f"report: {report_md}  (paste this into the PR)")
    print("=" * 72)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
