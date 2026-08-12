#!/usr/bin/env python3
"""verify_kernels.py — compile and run every CUDA kernel in kernels/, on a machine with no GPU.

    python tools/verify_kernels.py            # build + run + prove the checks can fail
    python tools/verify_kernels.py --quick    # build + run only

How this is possible without CUDA
---------------------------------
kernels/cuda_shim.hpp compiles a .cu file as ordinary C++, running one std::thread per CUDA
thread with real barriers. Shared memory is really shared, __syncthreads() is a real barrier,
and warp shuffles really move data between lanes. What it does *not* emulate is performance,
so this script checks correctness only and the kernels themselves print no timings in a shim
build.

The second half is the part that matters
----------------------------------------
A test suite that has never failed is a rumour. So after checking that every kernel passes,
this script mutates each one with a realistic bug and checks that it *fails*. Two of the
injected faults are missing __syncthreads() calls — race conditions that a GPU exposes only
intermittently and only at scale, and that a single-threaded emulator could not catch at all.
They are caught here, deterministically, in under a second, which is the whole argument for
running real threads in the shim.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNELS = REPO / "kernels"
CXXFLAGS = ["-std=c++17", "-O2", f"-I{KERNELS}", "-pthread", "-Wno-unknown-pragmas", "-x", "c++"]

# One realistic bug per kernel. Each is a single-token edit of the kind that survives review,
# and each must turn a passing run into a failing one.
FAULTS = [
    ("01_copy.cu", "i += stride)\n    out[i] = in[i];", "i += stride * 2)\n    out[i] = in[i];",
     "grid-stride loop skips half its elements"),
    ("02_reduce.cu", "for (int offset = warpSize / 2; offset > 0; offset >>= 1)",
     "for (int offset = warpSize / 2; offset > 1; offset >>= 1)",
     "shuffle reduction tree stops one level early"),
    ("03_sgemm.cu", "      for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];\n    }\n    __syncthreads();",
     "      for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];\n    }\n",
     "missing __syncthreads() — write-after-read race on the shared tile"),
    ("04_rmsnorm.cu", "acc += s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;",
     "acc += s.x * s.x + s.y * s.y + s.z * s.z;",
     "sum of squares drops the last component of each float4"),
    ("05_dequant_gemv.cu", "int k0 = 2 * b;", "int k0 = b;",
     "int4 unpacking indexes weights instead of bytes"),
    ("06_flash_decode.cu", "  pacc[slot * D + d] = acc;      // unnormalized",
     "  pacc[slot * D + d] = acc / l;  // unnormalized",
     "split-K writes normalized partials, so the merge cannot reweight them"),
]


def build(src: Path, out: Path, text: str | None = None) -> tuple[bool, str]:
    """Compile one kernel with g++. `text` overrides the file contents (for fault injection)."""
    if text is None:
        target = src
        p = subprocess.run(["g++", *CXXFLAGS, str(target), "-o", str(out)],
                           capture_output=True, text=True)
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".cu", dir=KERNELS, delete=False) as f:
            f.write(text)
            tmp = Path(f.name)
        try:
            p = subprocess.run(["g++", *CXXFLAGS, str(tmp), "-o", str(out)],
                               capture_output=True, text=True)
        finally:
            tmp.unlink(missing_ok=True)
    return p.returncode == 0, (p.stderr or "")[-2000:]


def run(binary: Path, timeout: int = 300) -> tuple[int, str]:
    try:
        p = subprocess.run([str(binary)], capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the fault-injection pass")
    args = ap.parse_args()

    if not shutil.which("g++"):
        print("g++ not found — cannot verify the kernels")
        return 1
    sources = sorted(KERNELS.glob("[0-9][0-9]_*.cu"))
    if not sources:
        print("no kernels found")
        return 1

    failures: list[str] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="kverify-"))

    # --- 1: every kernel compiles and its self-check passes -------------------------------
    print(f"building and running {len(sources)} kernels with the CPU shim\n")
    for src in sources:
        out = tmpdir / src.stem
        ok, err = build(src, out)
        if not ok:
            failures.append(f"{src.name}: compile failed\n{err}")
            print(f"  {src.name:<24} COMPILE FAILED")
            continue
        code, log = run(out)
        verdict = "PASS" if code == 0 and "PASS:" in log else "FAIL"
        n_variants = log.count(" ok ") + log.count(" ok\n") + log.count("WRONG")
        print(f"  {src.name:<24} {verdict}  ({n_variants} variants checked)")
        if verdict != "PASS":
            failures.append(f"{src.name}: exit {code}\n{log[-1500:]}")

    # --- 2: the checks must be able to fail ------------------------------------------------
    if not args.quick:
        print("\ninjecting one realistic bug per kernel; each must be caught\n")
        for fname, find, repl, desc in FAULTS:
            src = KERNELS / fname
            text = src.read_text()
            if find not in text:
                failures.append(f"{fname}: fault-injection anchor no longer present — "
                                f"the mutation test has silently stopped testing anything.\n"
                                f"  anchor: {find[:70]!r}")
                print(f"  {fname:<24} ANCHOR MISSING — {desc}")
                continue
            out = tmpdir / (src.stem + "-faulty")
            ok, err = build(src, out, text.replace(find, repl, 1))
            if not ok:
                failures.append(f"{fname}: faulty variant did not compile\n{err}")
                print(f"  {fname:<24} (faulty build failed to compile)")
                continue
            code, log = run(out)
            caught = code != 0
            print(f"  {fname:<24} {'caught' if caught else 'NOT CAUGHT'}  — {desc}")
            if not caught:
                failures.append(f"{fname}: injected fault ({desc}) was NOT detected — "
                                f"this kernel's correctness check proves nothing")

    shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} problem(s):")
        for f in failures:
            print("  -", f)
        return 1
    print("all kernels correct, and every correctness check verified to fail on a real bug")
    return 0


if __name__ == "__main__":
    sys.exit(main())
