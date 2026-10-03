"""The eval engine: build, run, mutate, shuffle, score.

Six dimensions, each of which can fail independently and each of which is reported separately
rather than collapsed into one green tick:

  build        the kernel compiles — with nvcc if present, and always with g++ via the shim
  correctness  every variant agrees with the program's own CPU reference, and it exits 0
  variants     the number of variants matches what the suite declares
  determinism  checksums are stable across shuffled block orders — and the variants declared
               order-dependent really are, which is checked in the same pass
  mutation     every declared bug is caught by the kernel's own check
  efficiency   on a real GPU, the best variant reaches a declared fraction of the ceiling

The mutation score is the one that matters most, because it is the only dimension that
measures the *test*, rather than the code. A suite where every kernel passes and no mutation
is caught is a suite that proves nothing, and it will score 0 here rather than 100.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .suite import Case, Mutation, Suite

KB_LINE = re.compile(r"^##KB## (.*)$", re.M)


@dataclass
class Dimension:
    name: str
    passed: int = 0
    total: int = 0
    detail: list[str] = field(default_factory=list)
    skipped: str = ""       # non-empty means "not applicable here", not "passed"

    @property
    def score(self) -> float | None:
        if self.skipped or self.total == 0:
            return None
        return self.passed / self.total

    def ok(self) -> bool:
        return bool(self.skipped) or self.passed == self.total


@dataclass
class CaseResult:
    case: Case
    dims: dict[str, Dimension] = field(default_factory=dict)
    kb: dict | None = None
    build_error: str = ""
    seconds: float = 0.0

    def ok(self) -> bool:
        return all(d.ok() for d in self.dims.values())

    @property
    def score(self) -> float:
        vals = [d.score for d in self.dims.values() if d.score is not None]
        return sum(vals) / len(vals) if vals else 0.0


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 300,
         env: dict | None = None) -> tuple[int, str]:
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd, cwd=None if cwd is None else str(cwd), capture_output=True,
                           text=True, timeout=timeout, env=e)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def parse_kb(output: str) -> dict | None:
    """Pull the machine-readable line out of a program's output."""
    m = KB_LINE.search(output)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


class Evaluator:
    def __init__(self, suite: Suite, use_nvcc: bool | None = None, arch: str = "native",
                 quiet: bool = False):
        self.suite = suite
        self.quiet = quiet
        self.arch = arch
        self.nvcc = shutil.which("nvcc") if use_nvcc is not False else None
        self.cxx = shutil.which("g++") or shutil.which("clang++")
        self.tmp = Path(tempfile.mkdtemp(prefix="kernelbench-"))

    # -- compilation ---------------------------------------------------------------------
    def _cxx_cmd(self, src: Path, out: Path) -> list[str]:
        return [self.cxx, "-std=c++17", "-O2", f"-I{self.suite.include_dir}",
                f"-I{self.suite.root}", "-pthread", "-Wno-unknown-pragmas", "-x", "c++",
                str(src), "-o", str(out)]

    def _nvcc_cmd(self, src: Path, out: Path) -> list[str]:
        return [self.nvcc, "-std=c++17", "-O3", f"-I{self.suite.include_dir}",
                f"-I{self.suite.root}", f"-arch={self.arch}", str(src), "-o", str(out)]

    def build(self, src: Path, out: Path, text: str | None = None,
              with_nvcc: bool = False) -> tuple[bool, str]:
        """Compile one kernel. `text` replaces the source, for mutation testing."""
        target = src
        tmp_src = None
        if text is not None:
            # Written into the harness's own temp directory rather than beside the original.
            # The include path already covers the suite root, so relative includes still
            # resolve — and an interrupted run cannot leave a stray .cu in someone's source
            # tree, which the suite's own "undeclared kernel" check would then flag.
            fd = tempfile.NamedTemporaryFile("w", suffix=".cu", dir=str(self.tmp),
                                             delete=False)
            fd.write(text)
            fd.close()
            tmp_src = target = Path(fd.name)
        try:
            cmd = self._nvcc_cmd(target, out) if with_nvcc else self._cxx_cmd(target, out)
            rc, log = _run(cmd, timeout=600)
            return rc == 0, log[-3000:]
        finally:
            if tmp_src is not None:
                tmp_src.unlink(missing_ok=True)

    # -- the dimensions ------------------------------------------------------------------
    def evaluate_case(self, case: Case) -> CaseResult:
        res = CaseResult(case=case)
        t0 = time.time()
        src = self.suite.root / case.source
        binary = self.tmp / case.name

        # --- build --------------------------------------------------------------------
        d = Dimension("build")
        ok, log = self.build(src, binary)
        d.total += 1
        d.passed += int(ok)
        if not ok:
            d.detail.append(f"g++ (shim): FAILED\n{log}")
            res.build_error = log
            res.dims["build"] = d
            res.seconds = time.time() - t0
            return res
        if self.nvcc:
            nok, nlog = self.build(src, self.tmp / (case.name + "-gpu"), with_nvcc=True)
            d.total += 1
            d.passed += int(nok)
            if not nok:
                d.detail.append(f"nvcc -arch={self.arch}: FAILED\n{nlog}")
        else:
            d.detail.append("nvcc not present — shim build only")
        res.dims["build"] = d

        # --- correctness --------------------------------------------------------------
        rc, out = _run([str(binary)], timeout=case.timeout_s)
        kb = parse_kb(out)
        res.kb = kb
        c = Dimension("correctness")
        if kb is None:
            c.total, c.passed = 1, 0
            c.detail.append("no ##KB## line — the program must call bench::verdict(rows, tol, &dev)")
        else:
            variants = kb["variants"]
            c.total = len(variants) + 1
            c.passed = sum(1 for v in variants if v["ok"]) + int(rc == 0)
            for v in variants:
                if not v["ok"]:
                    c.detail.append(f"{v['name']}: err {v['err']:.3e} > tol {kb['tol']:.1e}")
            if rc != 0:
                c.detail.append(f"exit code {rc}")
        res.dims["correctness"] = c

        # --- variant count ------------------------------------------------------------
        v = Dimension("variants")
        if case.expect_variants:
            v.total = 1
            got = len(kb["variants"]) if kb else 0
            v.passed = int(got == case.expect_variants)
            if not v.passed:
                v.detail.append(f"expected {case.expect_variants} variants, program reported {got}")
        else:
            v.skipped = "no expect_variants declared"
        res.dims["variants"] = v

        # --- determinism ---------------------------------------------------------------
        res.dims["determinism"] = self._determinism(case, binary, kb)

        # --- mutation ------------------------------------------------------------------
        res.dims["mutation"] = self._mutation(case, src)

        # --- sanity --------------------------------------------------------------------
        res.dims["sanity"] = self._sanity(kb)

        # --- efficiency ----------------------------------------------------------------
        res.dims["efficiency"] = self._efficiency(case, kb)

        res.seconds = time.time() - t0
        return res

    def _determinism(self, case: Case, binary: Path, kb: dict | None) -> Dimension:
        """Re-run under several shuffled block orders and compare exact checksums.

        CUDA promises nothing about the order blocks run in, and the shim honours that: with
        KB_SHIM_SHUFFLE set it permutes the order. Anything accumulating through atomics gets
        different floating-point rounding, so this reproduces — on a CPU, reproducibly — the
        run-to-run drift that makes a training loss curve or a batch of logits fail to
        reproduce on real hardware.
        """
        d = Dimension("determinism")
        if kb is None:
            d.skipped = "no results to compare"
            return d
        names = [v["name"] for v in kb["variants"]]
        base = {v["name"]: v["checksum"] for v in kb["variants"]}
        varies = {n: False for n in names}
        for seed in self.suite.determinism_seeds:
            rc, out = _run([str(binary)], timeout=case.timeout_s,
                           env={"KB_SHIM_SHUFFLE": str(seed)})
            k2 = parse_kb(out)
            if k2 is None:
                d.total += 1
                d.detail.append(f"seed {seed}: no ##KB## line (exit {rc})")
                continue
            for v2 in k2["variants"]:
                if base.get(v2["name"]) != v2["checksum"]:
                    varies[v2["name"]] = True

        declared = set(case.nondeterministic_variants)
        for n in names:
            d.total += 1
            expected_to_vary = n in declared
            if varies[n] == expected_to_vary:
                d.passed += 1
            elif varies[n]:
                d.detail.append(
                    f"{n}: result changes with block order, and the suite does not say it "
                    f"should — an undeclared reproducibility hazard")
            else:
                d.detail.append(
                    f"{n}: declared order-dependent but the result is stable — either the "
                    f"kernel was fixed and the declaration is stale, or the shuffle is not "
                    f"reaching it")
        for stale in sorted(declared - set(names)):
            d.total += 1
            d.detail.append(f"'{stale}' is declared nondeterministic but is not a variant")
        return d

    def _mutation(self, case: Case, src: Path) -> Dimension:
        """Inject each declared bug and require the kernel's own check to catch it."""
        d = Dimension("mutation")
        if not case.mutations:
            d.skipped = "no mutations declared"
            return d
        text = src.read_text()
        for m in case.mutations:
            d.total += 1
            n = text.count(m.find)
            if n != 1:
                d.detail.append(
                    f"'{m.name}': anchor appears {n} times, expected exactly 1 — the mutation "
                    f"test has stopped testing anything")
                continue
            out_bin = self.tmp / f"{case.name}-mut-{d.total}"
            ok, log = self.build(src, out_bin, text=text.replace(m.find, m.replace, 1))
            if not ok:
                d.detail.append(f"'{m.name}': mutant does not compile\n{log[-600:]}")
                continue
            rc, out = _run([str(out_bin)], timeout=case.timeout_s)
            if rc != 0:
                d.passed += 1
            else:
                d.detail.append(f"'{m.name}': NOT caught — this kernel's check proves nothing "
                                f"about that class of bug")
        return d

    def _sanity(self, kb: dict | None) -> Dimension:
        """Results that the hardware cannot have produced.

        A kernel reporting more bandwidth than the memory controller has, or more FLOP/s than
        the SMs can issue, has not discovered anything — it has mistimed something. This is
        the single cheapest check in the harness and it catches the three most common
        benchmarking errors at once: timing an async launch, leaving the input in L2, and
        dividing by a byte count the kernel did not actually move.
        """
        d = Dimension("sanity")
        if kb is None or not kb.get("real_gpu"):
            d.skipped = "no GPU — nothing to exceed"
            return d
        peak_bw, peak_fl = kb.get("peak_gbps", 0), kb.get("peak_gflops", 0)
        for v in kb["variants"]:
            if not v.get("timed"):
                continue
            d.total += 1
            bad = []
            if peak_bw and v["gbps"] > peak_bw * 1.02:
                bad.append(f"{v['gbps']:.0f} GB/s exceeds the card's {peak_bw:.0f} GB/s")
            if peak_fl and v["gflops"] > peak_fl * 1.02:
                bad.append(f"{v['gflops']:.0f} GFLOP/s exceeds the card's {peak_fl:.0f}")
            if v["median_ms"] <= 0:
                bad.append("median time is zero")
            if bad:
                d.detail.append(f"{v['name']}: " + "; ".join(bad))
            else:
                d.passed += 1
        return d

    def _efficiency(self, case: Case, kb: dict | None) -> Dimension:
        d = Dimension("efficiency")
        if kb is None or not kb.get("real_gpu"):
            d.skipped = "no GPU — the shim emulates correctness, not performance"
            return d
        if not case.min_efficiency:
            d.skipped = "no min_efficiency declared"
            return d
        peak_bw = kb.get("peak_gbps", 0) * 0.90     # achievable, not sticker
        peak_fl = kb.get("peak_gflops", 0)
        best = 0.0
        for v in kb["variants"]:
            if not v.get("timed"):
                continue
            fb = v["gbps"] / peak_bw if peak_bw else 0.0
            fc = v["gflops"] / peak_fl if peak_fl else 0.0
            best = max(best, fb, fc)
        d.total = 1
        d.passed = int(best >= case.min_efficiency)
        if not d.passed:
            d.detail.append(f"best variant reached {best:.0%} of the binding ceiling, "
                            f"suite requires {case.min_efficiency:.0%}")
        else:
            d.detail.append(f"best variant reached {best:.0%} of the binding ceiling")
        return d

    # -- driver ---------------------------------------------------------------------------
    def run(self) -> list[CaseResult]:
        results = []
        for case in self.suite.cases:
            r = self.evaluate_case(case)
            results.append(r)
            if not self.quiet:
                marks = " ".join(
                    f"{n[:4]}:{'-' if d.skipped else ('ok' if d.ok() else 'FAIL')}"
                    for n, d in r.dims.items())
                print(f"  {case.source:<26} {marks}  ({r.seconds:.1f}s)")
        return results

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)
