#!/usr/bin/env python3
"""Cross-notebook consistency audit.

Several notebooks independently declare the same physical constants — GPU bandwidth and
peak FLOPS, model layer/head geometry, prices. If those tables drift apart the track starts
contradicting itself, and a reader who compares two notebooks loses trust in both.

This extracts the shared tables from every notebook and cross-checks the overlapping keys.

    python tools/audit_consistency.py

Exits non-zero if any two notebooks disagree about the same fact.
"""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Table name -> (canonical field name -> aliases used across notebooks)
TRACKED = {
    "GPU": {
        "table_names": {"GPUS", "HW", "GPU"},
        "fields": {
            "bandwidth_tbs": {"bw_tbs", "bw"},
            "peak_fp16_tf": {"fp16_tf", "tf16", "tf"},
            "vram_gb": {"vram_gb", "vram"},
            "usd_hr": {"usd_hr", "usd"},
        },
    },
    "MODEL": {
        "table_names": {"MODELS", "ARCH", "MOE"},
        "fields": {
            "layers": {"layers"},
            "kv_heads": {"kv_heads", "kvh"},
            "head_dim": {"head_dim", "hd"},
            "hidden": {"hidden", "hidden_size"},
            "params_b": {"params", "params_b", "total"},
        },
    },
}

# Values legitimately expressed in different units across notebooks.
def normalise(kind: str, field: str, value):
    if value is None or isinstance(value, (str, bool)):
        return None
    if not isinstance(value, (int, float)):
        return None
    if kind == "GPU" and field == "bandwidth_tbs" and value > 100:
        return value / 1e12                      # some notebooks store bytes/s
    if kind == "GPU" and field == "peak_fp16_tf" and value > 1e6:
        return value / 1e12                      # some store FLOP/s
    if kind == "MODEL" and field == "params_b" and value > 1e6:
        return value / 1e9                       # some store raw parameter counts
    return value


def canonical_gpu(name: str) -> str:
    n = name.strip().lower().replace("nvidia ", "").replace("amd ", "")
    for suffix in (" sxm", " sxm5", " pcie"):
        n = n.replace(suffix, "")
    n = n.replace("40gb", "").replace("80gb", "80").replace("(gqa)", "")
    return " ".join(n.split())


def canonical_model(name: str) -> str:
    n = name.strip().lower()
    for junk in ("(gqa)", "(mha)", "(mla)", "(swa 4k)", "(hybrid)", "-instruct", "meta-llama/", "qwen/"):
        n = n.replace(junk, "")
    return " ".join(n.split())


COLUMN_SPEC_NAMES = {"COLS", "COLUMNS", "FIELDS"}


def extract_tables(nb_path: Path):
    """Statically evaluate top-level table literals whose names we track.

    Handles two shapes seen in this repo:
      MODELS = {"name": dict(...), ...}                     - dict of dicts
      GPUS = [("name", "NVIDIA", ..., 0.32, ...), ...]      - rows + a separate COLS tuple
        COLS = ("name", "vendor", ..., "bw_tbs", ...)
    """
    nb = json.loads(nb_path.read_text())
    found, columns = [], None
    pending_rows = []

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        src = src if isinstance(src, str) else "".join(src)
        code = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("!", "%")))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id in COLUMN_SPEC_NAMES:
                try:
                    cols = ast.literal_eval(node.value)
                    if isinstance(cols, (list, tuple)) and all(isinstance(c, str) for c in cols):
                        columns = list(cols)
                except Exception:
                    pass
                continue
            for kind, spec in TRACKED.items():
                if target.id not in spec["table_names"]:
                    continue
                table = safe_eval_table(node.value)
                if not table:
                    continue
                if any(isinstance(v, (list, tuple)) for v in table.values()):
                    pending_rows.append((kind, target.id, table))   # needs COLS, may come later
                else:
                    found.append((kind, target.id, table))

    # Zip positional rows against the column spec once we've seen the whole notebook.
    for kind, name, table in pending_rows:
        if not columns:
            continue
        zipped = {}
        for key, row in table.items():
            if isinstance(row, (list, tuple)) and len(row) == len(columns):
                zipped[key] = dict(zip(columns, row))
        if zipped:
            found.append((kind, name, zipped))
    return found


def safe_eval_table(node):
    """Evaluate a dict literal of dicts, tolerating dict(...) calls and simple arithmetic."""
    def ev(n):
        try:
            return ast.literal_eval(n)
        except Exception:
            pass
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "dict":
            out = {}
            for kw in n.keywords:
                v = ev(kw.value)
                if v is not _SENTINEL:
                    out[kw.arg] = v
            return out
        if isinstance(n, ast.BinOp):  # e.g. 512 + 64
            l, r = ev(n.left), ev(n.right)
            if isinstance(l, (int, float)) and isinstance(r, (int, float)):
                if isinstance(n.op, ast.Add): return l + r
                if isinstance(n.op, ast.Mult): return l * r
                if isinstance(n.op, ast.Sub): return l - r
        return _SENTINEL

    _SENTINEL = object()
    if not isinstance(node, ast.Dict):
        # also accept a list of tuples (the GPUS = [(...), ...] shape)
        if isinstance(node, (ast.List, ast.Tuple)):
            rows = [ev(e) for e in node.elts]
            rows = [r for r in rows if isinstance(r, (list, tuple))]
            return {r[0]: r for r in rows if r and isinstance(r[0], str)} or None
        return None
    out = {}
    for k, v in zip(node.keys, node.values):
        key = ev(k) if k is not None else None
        val = ev(v)
        if isinstance(key, str) and val is not _SENTINEL:
            out[key] = val
    return out or None


def main() -> int:
    # fact -> {(notebook, raw_key): value}
    facts: dict[tuple, dict] = defaultdict(dict)

    for nb_path in sorted(REPO.glob("*.ipynb")):
        for kind, table_name, table in extract_tables(nb_path):
            spec = TRACKED[kind]
            for raw_key, row in table.items():
                if not isinstance(row, dict):
                    continue
                key = canonical_gpu(raw_key) if kind == "GPU" else canonical_model(raw_key)
                for field, aliases in spec["fields"].items():
                    for alias in aliases:
                        if alias in row:
                            v = normalise(kind, field, row[alias])
                            if v is not None:
                                facts[(kind, key, field)][(nb_path.name, table_name)] = v
                            break

    disagreements = []
    checked = 0
    for (kind, key, field), sources in sorted(facts.items()):
        if len(sources) < 2:
            continue
        checked += 1
        values = list(sources.values())
        lo, hi = min(values), max(values)
        if lo == 0:
            continue
        if (hi - lo) / abs(lo) > 0.02:          # 2% tolerance for rounding
            disagreements.append((kind, key, field, sources))

    print(f"cross-checked {checked} facts shared by 2+ notebooks")
    if not disagreements:
        print("\nno contradictions found")
        return 0

    print(f"\n{len(disagreements)} CONTRADICTION(S):\n")
    for kind, key, field, sources in disagreements:
        print(f"  {kind} '{key}' · {field}")
        for (nb, table), v in sorted(sources.items()):
            print(f"      {v:<12} {nb} [{table}]")
        print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
