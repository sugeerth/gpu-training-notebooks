"""servingkit — the models behind the notebooks, as one importable library.

This repository's notebooks each used to carry their own copy of the arithmetic, and the demo
pages carried a third in JavaScript. `tools/audit_consistency.py` existed to notice when the
copies drifted. This package is the single copy; the notebooks import it, the demo pages are
checked against it, and the tests assert the notebooks and the package have not diverged.

Two layers. The **models** are plain functions over plain dicts — import one and use it:

    >>> import servingkit as sk
    >>> sk.predict("H100 SXM", "Llama-3.1-8B", batch=32)["tpot_ms"]
    >>> sk.training_plan(params_b=70, gpus=8, zero_stage=3)["total_gb"]
    >>> sk.agent_run(turns=40, tool_result_tokens=2000)["prefill_naive"]

The **composable layer** is what makes it infrastructure. Declare a workload, and the levers
that apply to it are the ones you are offered:

    >>> w = sk.Workload.agent(turns=30, fanout=4, tool_result_tokens=2000)
    >>> s = sk.recommend(w)
    >>> for row in s.ladder(): print(row["label"], round(row["cost_usd"], 3))

A lever declares what it changes, the scarce resource it spends, the workload property it
needs, and the kernel in `kernels/` that implements it. From those four declarations the stack
derives applicability, exclusivity, and a *prediction* of which pairs will fight — so a
sub-multiplicative gain arrives with the reason attached instead of as a surprise.

Every number here is a planning model, not a measurement. `recalibrate()` takes your own.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .agents import (agent_run, cascade_traffic, padding_waste, prefill_saving, restore_cost,
                     tool_gap_policy)
from .catalog import (BW_EFF, ENGINE_OVERHEAD_MS, FLOP_EFF, GPUS, MEM_UTIL, MIN_CONCURRENCY,
                      MODELS, PRECISION, recalibrate)
from .kernels import KernelResult, available_kernels, run_kernel
from .kv import capacity, kv_bytes_per_token, kv_total_bytes
from .levers import (LEVERS, PROPERTIES, RESOURCES, Lever, applicable, exclusive_pairs,
                     register, shared_resources, solo_gain)
from .report import plan_report, ladder_table, lever_table, interaction_table
from .serving import cheapest, predict
from .spec import best_k, expected_tokens, speedup
from .stack import Rejection, Stack, recommend
from .step import base_config, evaluate
from .training import training_plan
from .workload import Workload

__all__ = [
    "__version__",
    # catalogs
    "GPUS", "MODELS", "PRECISION", "BW_EFF", "FLOP_EFF", "MEM_UTIL", "MIN_CONCURRENCY",
    "ENGINE_OVERHEAD_MS", "recalibrate",
    # models
    "kv_bytes_per_token", "kv_total_bytes", "capacity",
    "base_config", "evaluate", "predict", "cheapest",
    "training_plan",
    "agent_run", "tool_gap_policy", "cascade_traffic", "prefill_saving", "restore_cost",
    "padding_waste",
    "expected_tokens", "speedup", "best_k",
    # composition
    "Workload", "Lever", "LEVERS", "RESOURCES", "PROPERTIES", "register", "applicable",
    "exclusive_pairs", "shared_resources", "solo_gain",
    "Stack", "Rejection", "recommend",
    # kernels + reporting
    "run_kernel", "available_kernels", "KernelResult",
    "plan_report", "ladder_table", "lever_table", "interaction_table",
]
