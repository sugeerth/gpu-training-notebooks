"""Build and run a kernel from `kernels/`, and get its results as data.

Every kernel in this repository prints a human table and then one machine-readable line:

    ##KB## {"schema":1,"device":"...","variants":[{"name":...,"err":...,"checksum":...}]}

`run_kernel()` parses that line, so a lever's claim can be checked against the program that
implements it rather than quoted from a comment. This is the bridge between the planning models
in this package and the code that does the work: `LEVERS["cascade attention"].kernel` names a
file, and this module runs it.

No GPU needed — the shim compiles the same `.cu` with `g++` and reports correctness only, which
is why `timed` comes back False on a CPU and the timings are absent rather than fabricated.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Find the repository root by looking for `kernels/Makefile`, upward from `start`."""
    p = (start or Path(__file__)).resolve()
    for cand in [p, *p.parents]:
        if (cand / "kernels" / "Makefile").exists():
            return cand
    raise FileNotFoundError("no kernels/Makefile found above " + str(p))


@dataclass
class KernelResult:
    """One kernel's run: its variants, and whether the numbers are real."""

    name: str
    device: str
    real_gpu: bool
    tol: float
    variants: list[dict] = field(default_factory=list)
    stdout: str = ""
    returncode: int = 0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and all(v.get("ok") for v in self.variants)

    @property
    def timed(self) -> bool:
        """True only when the timings mean something. On the CPU shim they do not exist."""
        return self.real_gpu and any(v.get("timed") for v in self.variants)

    def best(self, metric: str = "gbps") -> dict | None:
        timed = [v for v in self.variants if v.get("timed")]
        return max(timed, key=lambda v: v.get(metric, 0)) if timed else None

    def table(self) -> str:
        w = max((len(v["name"]) for v in self.variants), default=8)
        head = f"{'variant':<{w}}  {'max err':>10}  {'ok':>4}"
        if self.timed:
            head += f"  {'ms':>9}  {'GB/s':>9}"
        lines = [head, "-" * len(head)]
        for v in self.variants:
            row = f"{v['name']:<{w}}  {v['err']:>10.2e}  {'ok' if v['ok'] else 'FAIL':>4}"
            if self.timed:
                row += f"  {v.get('median_ms', 0):>9.4f}  {v.get('gbps', 0):>9.1f}"
            lines.append(row)
        return "\n".join(lines)


_KB = re.compile(r"^##KB##\s*(\{.*\})\s*$", re.M)


def available_kernels(root: Path | None = None) -> list[str]:
    """Every kernel in the repository, in reading order."""
    k = (root or repo_root()) / "kernels"
    return sorted(p.stem for p in k.glob("*.cu"))


def run_kernel(name: str, *, root: Path | None = None, quiet: bool = True,
               timeout: int = 900) -> KernelResult:
    """Build and run one kernel. `name` may be "13", "13_prefix_attention" or the filename.

        >>> import servingkit as sk
        >>> r = sk.run_kernel("13")
        >>> r.ok, [v["name"] for v in r.variants]
    """
    root = root or repo_root()
    kdir = root / "kernels"
    stem = name[:-3] if name.endswith(".cu") else name
    if not (kdir / f"{stem}.cu").exists():
        matches = [s for s in available_kernels(root) if s.startswith(stem.split("_")[0])]
        if not matches:
            raise FileNotFoundError(f"no kernel matching {name!r} in {kdir}")
        stem = matches[0]

    proc = subprocess.run(["make", "--no-print-directory", stem], cwd=str(kdir),
                          capture_output=True, text=True, timeout=timeout)
    out = proc.stdout or ""
    m = _KB.search(out)
    if not m:
        return KernelResult(stem, "unknown", False, 0.0, [], out, proc.returncode or 1)
    d = json.loads(m.group(1))
    if not quiet:
        print("\n".join(l for l in out.splitlines() if not l.startswith("##KB##")))
    return KernelResult(stem, d["device"], bool(d["real_gpu"]), d["tol"], d["variants"],
                        out, proc.returncode)


def kernel_for(lever: str) -> str | None:
    """The kernel that implements a lever, if one does."""
    from .levers import LEVERS
    return LEVERS[lever].kernel


__all__ = ["KernelResult", "run_kernel", "available_kernels", "kernel_for", "repo_root"]
