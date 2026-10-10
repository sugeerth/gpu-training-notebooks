"""Suite specification — what it means to evaluate a directory of CUDA kernels.

A suite is a `kernelbench.json` sitting next to the `.cu` files. It declares, per kernel, the
things a harness cannot infer:

  * how many variants the program is supposed to report (so one silently disappearing fails)
  * which variants are *expected* to be order-dependent (checked in both directions)
  * what bugs must be caught (mutation testing)
  * what fraction of the hardware ceiling counts as acceptable, on a real GPU

Everything else — errors, checksums, timings, the device's peak numbers — comes from the
program itself, on the `##KB##` line that `bench::verdict()` emits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Mutation:
    """A realistic bug that the kernel's own correctness check must catch.

    `find` must appear exactly once in the source. If it stops appearing — because the code
    was refactored — that is reported as a failure rather than skipped, because a mutation
    test whose anchor has drifted is a test that has silently stopped testing anything.
    """
    name: str
    find: str
    replace: str


@dataclass
class Case:
    source: str
    title: str = ""
    topic: str = ""
    expect_variants: int = 0
    # Variants whose result legitimately depends on the order blocks happen to run in — an
    # atomicAdd reduction, say. Declaring one asserts that it IS order-dependent; the harness
    # fails if it turns out to be stable, so this cannot rot into a rubber stamp.
    nondeterministic_variants: list[str] = field(default_factory=list)
    mutations: list[Mutation] = field(default_factory=list)
    # GPU only: the minimum fraction of the binding ceiling the best variant should reach.
    min_efficiency: float = 0.0
    timeout_s: int = 300

    @property
    def name(self) -> str:
        return Path(self.source).stem


@dataclass
class Suite:
    name: str
    root: Path
    shim_include: str = "../kernelbench/shim"
    determinism_seeds: list[int] = field(default_factory=lambda: [1, 5, 42, 1337])
    cases: list[Case] = field(default_factory=list)

    @property
    def include_dir(self) -> Path:
        return (self.root / self.shim_include).resolve()

    @staticmethod
    def load(path: str | Path) -> "Suite":
        p = Path(path).resolve()
        if p.is_dir():
            p = p / "kernelbench.json"
        spec = json.loads(p.read_text())
        root = p.parent
        cases = []
        for c in spec.get("cases", []):
            muts = [Mutation(**m) for m in c.pop("mutations", [])]
            cases.append(Case(mutations=muts, **c))
        return Suite(
            name=spec.get("name", root.name),
            root=root,
            shim_include=spec.get("shim_include", "../kernelbench/shim"),
            determinism_seeds=spec.get("determinism_seeds", [1, 5, 42, 1337]),
            cases=cases,
        )

    def validate(self) -> list[str]:
        """Problems with the suite itself, as opposed to with the kernels."""
        problems = []
        if not self.include_dir.exists():
            problems.append(f"shim_include {self.include_dir} does not exist")
        declared = {c.source for c in self.cases}
        on_disk = {p.name for p in self.root.glob("*.cu")}
        for missing in sorted(declared - on_disk):
            problems.append(f"{missing} is declared in the suite but not on disk")
        for unlisted in sorted(on_disk - declared):
            # An unlisted kernel is a hole in the eval, not a harmless extra file.
            problems.append(f"{unlisted} is on disk but not declared in the suite")
        return problems
