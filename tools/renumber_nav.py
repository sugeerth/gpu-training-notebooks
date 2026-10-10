#!/usr/bin/env python3
"""Rewrite every notebook's nav line, deriving the order from the chain already in them.

The previous version of this script carried a hardcoded ORDER list in a scratch directory, which
is exactly the wrong place for the repository's table of contents: the scratch directory does not
survive a container, and the list drifted from the notebooks every time one was added.

This reads the order out of the notebooks themselves — each one already states its position —
so the only input is where to insert anything new.

    python tools/renumber_nav.py                       # renumber in place
    python tools/renumber_nav.py --first Start_Here...ipynb   # move one to the front
    python tools/renumber_nav.py --check               # report, change nothing
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NAV = re.compile(r"^<!--nav-->.*$", re.M)
POS = re.compile(r"\*\*(\d+)/(\d+)\*\*")
TITLE = re.compile(r"^#\s+(.+?)\s*$", re.M)


def notebooks() -> list[Path]:
    return sorted(p for p in REPO.glob("*.ipynb"))


def read(p: Path) -> dict:
    return json.loads(p.read_text())


def cell_text(cell: dict) -> str:
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def nav_position(nb: dict) -> int | None:
    for cell in nb["cells"][:2]:
        if cell["cell_type"] != "markdown":
            continue
        m = POS.search(cell_text(cell))
        if m:
            return int(m.group(1))
    return None


def short_title(nb: dict, path: Path) -> str:
    """The notebook's H1, trimmed at the first colon or dash — short enough for a nav line."""
    for cell in nb["cells"][:6]:
        if cell["cell_type"] != "markdown":
            continue
        m = TITLE.search(cell_text(cell))
        if m:
            t = m.group(1)
            for sep in (":", " — ", " - "):
                if sep in t:
                    t = t.split(sep)[0]
                    break
            return t.strip()
    return path.stem.replace("_", " ")


def build_order(first: list[str]) -> list[Path]:
    """Existing chain order, with `first` pulled to the front in the order given."""
    nbs = notebooks()
    pos = {p: (nav_position(read(p)) or 10_000) for p in nbs}
    rest = sorted((p for p in nbs if p.name not in first), key=lambda p: (pos[p], p.name))
    head = [REPO / n for n in first if (REPO / n).exists()]
    missing = [n for n in first if not (REPO / n).exists()]
    if missing:
        raise SystemExit(f"--first names a notebook that does not exist: {missing}")
    return head + rest


def nav_line(i: int, total: int, order: list[Path], titles: dict[Path, str]) -> str:
    parts = [f"<!--nav--> [🗺 Learning path](README.md) · **{i + 1}/{total}**"]
    if i > 0:
        p = order[i - 1]
        parts.append(f"◀ [{titles[p]}](./{p.name})")
    if i < total - 1:
        p = order[i + 1]
        parts.append(f"[{titles[p]}](./{p.name}) ▶")
    return " · ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", action="append", default=[],
                    help="notebook filename to place at the front (repeatable)")
    ap.add_argument("--check", action="store_true", help="report only")
    a = ap.parse_args(argv)

    order = build_order(a.first)
    total = len(order)
    titles = {p: short_title(read(p), p) for p in order}

    changed = []
    for i, p in enumerate(order):
        nb = read(p)
        want = nav_line(i, total, order, titles)
        cells = nb["cells"]
        # The nav lives in the first markdown cell that already has one, or a new leading cell.
        idx = next((j for j, c in enumerate(cells[:2])
                    if c["cell_type"] == "markdown" and "<!--nav-->" in cell_text(c)), None)
        if idx is None:
            if a.check:
                changed.append(f"{p.name}: has no nav line")
                continue
            cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": want})
            p.write_text(json.dumps(nb, indent=1))
            changed.append(f"{p.name}: nav added at {i + 1}/{total}")
            continue
        cur = cell_text(cells[idx])
        new = NAV.sub(want.replace("\\", "\\\\"), cur, count=1)
        if new == cur:
            continue
        if a.check:
            changed.append(f"{p.name}: {i + 1}/{total}")
            continue
        cells[idx]["source"] = new
        p.write_text(json.dumps(nb, indent=1))
        changed.append(f"{p.name}: {i + 1}/{total}")

    verb = "would change" if a.check else "renumbered"
    print(f"{total} notebooks, {verb} {len(changed)}")
    for line in changed[:12]:
        print("  " + line)
    if len(changed) > 12:
        print(f"  ... and {len(changed) - 12} more")
    return 1 if (a.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
