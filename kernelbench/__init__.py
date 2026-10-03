"""kernelbench — an eval harness for CUDA kernels that runs with or without a GPU.

    python -m kernelbench eval kernels/

Point it at a directory containing `.cu` files and a `kernelbench.json`, and it scores every
kernel on six independent dimensions: build, correctness, variant count, determinism under
shuffled block order, mutation coverage, and efficiency against the measured hardware ceiling.

Without a GPU it still checks everything except efficiency, by compiling each `.cu` file as
ordinary C++ against `kernelbench/shim/cuda_shim.hpp`, which runs one std::thread per CUDA
thread with real barriers and real warp shuffles.

Drop-in for any project: copy this directory, write a `kernelbench.json`, have each program
end with `bench::verdict(rows, tol, &dev)`.
"""
from .suite import Suite, Case, Mutation          # noqa: F401
from .evaluate import Evaluator, CaseResult       # noqa: F401
from . import report                              # noqa: F401

__version__ = "0.1.0"
__all__ = ["Suite", "Case", "Mutation", "Evaluator", "CaseResult", "report"]
