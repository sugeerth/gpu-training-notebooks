#!/usr/bin/env python3
"""Static checks for every notebook in the repo. No execution, no GPU, no network.

Run locally or in CI:

    python tools/validate_notebooks.py

Checks:
  1. every .ipynb is valid nbformat v4
  2. every code cell parses as Python (cell magics excluded)
  3. the prev/next nav chain is complete and consistently numbered
  4. every notebook / README / index.html link points at a file that exists
  5. no notebook refers to another one by position number
  6. every internal link on the demo site resolves
"""
from __future__ import annotations

import ast
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
MAGIC_CELL_PREFIXES = ("%%writefile", "%%bash", "%%capture", "%%html", "%%javascript")


def cell_source(cell) -> str:
    src = cell["source"]
    return src if isinstance(src, str) else "".join(src)


def main() -> int:
    import nbformat

    notebooks = sorted(p for p in REPO.glob("*.ipynb"))
    if not notebooks:
        print("no notebooks found")
        return 1

    failures: list[str] = []
    n_cells = 0

    # --- 1 & 2: nbformat validity and Python syntax -------------------------------------
    for path in notebooks:
        try:
            nbformat.validate(nbformat.reads(path.read_text(), as_version=4))
        except Exception as exc:
            failures.append(f"{path.name}: invalid nbformat - {exc}")
            continue
        nb = json.loads(path.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = cell_source(cell)
            if src.lstrip().startswith(MAGIC_CELL_PREFIXES):
                continue  # cell magics are not Python at the top level
            n_cells += 1
            code = "\n".join(
                l for l in src.split("\n") if not l.lstrip().startswith(("!", "%"))
            )
            try:
                ast.parse(code)
            except SyntaxError as exc:
                failures.append(f"{path.name} cell {i}: {exc}")

    # --- 3: nav chain ------------------------------------------------------------------
    nav = {}
    for path in notebooks:
        nb = json.loads(path.read_text())
        first = cell_source(nb["cells"][0])
        if "<!--nav-->" not in first:
            failures.append(f"{path.name}: missing nav cell")
            continue
        m = re.search(r"\*\*(\d+)/(\d+)\*\*", first)
        if not m:
            failures.append(f"{path.name}: nav cell has no N/M marker")
            continue
        idx, total = int(m.group(1)), int(m.group(2))
        if idx in nav:
            failures.append(f"duplicate nav index {idx}: {nav[idx]} and {path.name}")
        nav[idx] = path.name
        if total != len(notebooks):
            failures.append(
                f"{path.name}: nav says /{total} but the repo has {len(notebooks)} notebooks"
            )
    missing = set(range(1, len(notebooks) + 1)) - set(nav)
    if missing:
        failures.append(f"nav chain has gaps at positions: {sorted(missing)}")

    # --- 4: links ----------------------------------------------------------------------
    present = {p.name for p in REPO.glob("*.ipynb")}
    link_re = re.compile(r"(?:blob/main/|\]\(\./|\]\()([A-Za-z0-9_.\-]+\.ipynb)")
    for src_path in [REPO / "README.md", REPO / "index.html"] + notebooks:
        if not src_path.exists():
            continue
        if src_path.suffix == ".ipynb":
            nb = json.loads(src_path.read_text())
            text = "\n".join(cell_source(c) for c in nb["cells"])
        else:
            text = src_path.read_text()
        for m in link_re.finditer(text):
            if m.group(1) not in present:
                failures.append(f"{src_path.name}: broken link to {m.group(1)}")

    # --- 4b: the demo site's own links -------------------------------------------------
    demo = REPO / "demo"
    if demo.exists():
        demo_pages = sorted(demo.glob("*.html"))
        href_re = re.compile(r'href="(?!https?:|#|mailto:)([^"]+)"')
        for path in demo_pages:
            for m in href_re.finditer(path.read_text(encoding="utf-8")):
                target = (demo / m.group(1)).resolve()
                if not target.exists():
                    failures.append(f"demo/{path.name}: broken link to {m.group(1)}")
        # every tool must be reachable from the hub, or it may as well not exist
        hub = demo / "index.html"
        if hub.exists():
            hub_text = hub.read_text(encoding="utf-8")
            for path in demo_pages:
                if path.name != "index.html" and path.name not in hub_text:
                    failures.append(f"demo/{path.name} is not linked from demo/index.html")

    # --- 5: no cross-references by position number -------------------------------------
    # A notebook's number changes every time one is inserted ahead of it, so "notebook 26"
    # rots silently. Three overlapping numbering schemes had accumulated before this check
    # existed. Refer to notebooks by name and link instead. The 1-20 training range is
    # allowed: those are stable and are only ever cited as a block.
    ref_re = re.compile(r"(?:[Nn]otebooks?\s+|\bnb\s*)(\d+)")
    for path in notebooks:
        nb = json.loads(path.read_text())
        for i, cell in enumerate(nb["cells"]):
            for m in ref_re.finditer(cell_source(cell)):
                if int(m.group(1)) <= 20:
                    continue
                failures.append(
                    f"{path.name} cell {i}: refers to '{m.group(0)}' by position - "
                    "link the notebook by name instead"
                )

    print(f"notebooks : {len(notebooks)}")
    print(f"code cells: {n_cells} parsed")
    print(f"nav chain : 1..{len(nav)}")
    if failures:
        print(f"\n{len(failures)} problem(s):")
        for f in failures:
            print("  -", f)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
