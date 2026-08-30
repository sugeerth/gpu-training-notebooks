#!/usr/bin/env python3
"""CI gate: score every CUDA kernel in kernels/ with the kernelbench eval harness.

    python tools/verify_kernels.py                    # full eval, must score 100%
    python tools/verify_kernels.py --quick            # skip mutation testing
    python -m kernelbench eval kernels/ --markdown r.md   # the harness directly

This is a thin wrapper. The harness lives in `kernelbench/` and knows nothing about this
repository — it takes a directory of `.cu` files plus a `kernelbench.json` and scores them on
build, correctness, variant count, determinism, mutation coverage and efficiency. Any project
can copy that directory and get the same evaluation of its own kernels; see
`kernelbench/README.md`.

No GPU is required. `kernelbench/shim/cuda_shim.hpp` compiles each `.cu` file as ordinary C++
and runs one std::thread per CUDA thread, with real barriers and real warp shuffles.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kernelbench.__main__ import main  # noqa: E402


if __name__ == "__main__":
    argv = ["eval", str(REPO / "kernels")]
    if "--quick" in sys.argv:
        argv += ["--skip", "mutation"]
    argv += [a for a in sys.argv[1:] if a != "--quick"]
    sys.exit(main(argv))
