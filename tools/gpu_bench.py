#!/usr/bin/env python3
"""Backwards-compatible entry point for the measurement harness.

The implementation moved to `kernelbench/bench.py` when the harness was generalized into a
drop-in eval package. This shim keeps `from tools.gpu_bench import Bench` and
`python tools/gpu_bench.py` working, and keeps the notebooks' import path stable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernelbench.bench import *          # noqa: F401,F403
from kernelbench.bench import Bench, Device, Result, SPEC_PEAKS, _self_test   # noqa: F401

if __name__ == "__main__":
    raise SystemExit(_self_test())
