"""End to end: a workload spec goes in, a verified scorecard comes out.

The repository has had the pieces for a while — a planning model, twenty-two kernels, an eval
harness, a drift check — and no single command that runs them in order and emits one artifact.
This is that command.

    python -m servingkit pipeline --agent --turns 30 --fanout 4 -o scorecard.json

Six stages, each recorded with its own timing and status so a failure names itself:

    1. resolve     the spec becomes a Workload; its properties are derived
    2. plan        the recommended stack, the ladder, the money
    3. interact    pairwise gains, and whether any antagonism is undeclared
    4. verify      the kernels the plan actually names are compiled and run
    5. drift       the package is compared against the notebooks it was lifted from
    6. emit        one JSON document, which is what the console reads

Stage 4 is the point of doing this as a pipeline rather than a report. A plan that recommends
cascade attention is a claim about `13_prefix_attention.cu`, and this compiles that file and
checks it against its own double-precision reference before the claim ships. The scorecard
records which kernels were verified and on what device, so a reader can tell a plan backed by a
real GPU run from one backed by a CPU emulation — and the answer here is always the emulation
until somebody runs it on hardware.

`--fan-out` prints one shard per kernel instead of running them, which is how the Kubernetes Job
in `deploy/k8s/` spreads verification across a cluster: each pod runs `--shard i --shards n` and
writes a partial scorecard, and `--merge` combines them.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .kernels import available_kernels, repo_root, run_kernel
from .levers import LEVERS
from .report import interaction_table, ladder_table, plan_report
from .stack import Stack, recommend
from .workload import Workload


@dataclass
class Stage:
    name: str
    status: str = "pending"
    seconds: float = 0.0
    detail: dict = field(default_factory=dict)
    error: str | None = None


def _env() -> dict:
    """Enough about where this ran to make the numbers interpretable later."""
    return dict(
        host=socket.gethostname(), python=platform.python_version(),
        platform=platform.platform(), servingkit=__version__,
        # A pod records its own identity so a merged scorecard can say which shard produced what.
        pod=os.environ.get("HOSTNAME"), node=os.environ.get("NODE_NAME"),
        shard=os.environ.get("SERVINGKIT_SHARD"), shards=os.environ.get("SERVINGKIT_SHARDS"),
        git_sha=os.environ.get("GIT_SHA"),
    )


def kernels_for(stack: Stack) -> list[str]:
    """The kernels this plan's levers actually name, deduplicated, in reading order."""
    named = {LEVERS[n].kernel for n in stack.names if LEVERS[n].kernel}
    return sorted(named)


def shard_of(items: list[str], shard: int, shards: int) -> list[str]:
    """Deal items round-robin, so shards get comparable work even when costs differ widely.

    Contiguous slicing would put `01_copy` and `02_reduce` in one pod and the four expensive
    attention kernels in another. Round-robin is not perfect balancing, but it is one line and
    it is much better than the obvious thing.
    """
    if shards <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % shards == shard]


def run(spec: dict, *, verify: bool = True, drift: bool = True, all_kernels: bool = False,
        shard: int = 0, shards: int = 1, root: Path | None = None,
        on_stage=None) -> dict:
    """Run the pipeline and return the scorecard.

    `on_stage(Stage)` is called as each stage finishes, so a long run can report progress
    without this function knowing anything about how it is being displayed.
    """
    root = root or repo_root()
    t0 = time.time()
    stages: list[Stage] = []

    def stage(name: str):
        st = Stage(name)
        stages.append(st)
        return st

    def done(st: Stage, started: float, **detail):
        st.seconds = time.time() - started
        st.status = "ok" if st.error is None else "failed"
        st.detail.update(detail)
        if on_stage:
            on_stage(st)

    # ---------------------------------------------------------------- 1. resolve
    st = stage("resolve")
    t = time.time()
    from .api import stack_from_spec, workload_from_spec
    try:
        w = workload_from_spec(spec)
        stack = stack_from_spec(spec)
    except Exception as exc:  # noqa: BLE001 - the spec came from outside
        st.error = f"{type(exc).__name__}: {exc}"
        done(st, t)
        return _emit(spec, stages, None, None, None, None, t0)
    done(st, t, kind=w.kind, model=w.model, ctx=w.ctx, batch=w.batch,
         properties=sorted(w.properties), bottleneck=w.bottleneck(stack.gpu))

    # ------------------------------------------------------------------- 2. plan
    st = stage("plan")
    t = time.time()
    plan = stack.plan()
    rows = stack.ladder()
    done(st, t, levers=stack.names, wall_s=plan["wall_s"],
         total_cost_usd=plan["total_cost_usd"], bottleneck=plan["bottleneck"],
         rungs=len(rows))

    # --------------------------------------------------------------- 3. interact
    st = stage("interact")
    t = time.time()
    pairs = stack.interactions()
    undeclared = [f"{c['a']} + {c['b']}" for c in pairs
                  if c["synergy"] < 0.9 and not c["predicted_conflict"]]
    if undeclared:
        # This is a failure, not a warning. An unexplained antagonism means a lever is lying
        # about what it spends, and a plan built on a lying lever is advice that costs money.
        st.error = f"{len(undeclared)} antagonistic pair(s) with no declared conflict"
    done(st, t, pairs=len(pairs), undeclared=undeclared,
         worst=min((c["synergy"] for c in pairs), default=None))

    # ----------------------------------------------------------------- 4. verify
    st = stage("verify")
    t = time.time()
    results: list[dict] = []
    if verify:
        wanted = [k + ".cu" for k in available_kernels()] if all_kernels else kernels_for(stack)
        mine = shard_of(wanted, shard, shards)
        for k in mine:
            try:
                r = run_kernel(k, root=root)
                results.append(dict(kernel=r.name, device=r.device, real_gpu=r.real_gpu,
                                    ok=r.ok, timed=r.timed,
                                    variants=[dict(name=v["name"], err=v["err"], ok=v["ok"])
                                              for v in r.variants]))
            except Exception as exc:  # noqa: BLE001 - a build failure is data, not a crash
                results.append(dict(kernel=k, ok=False, error=f"{type(exc).__name__}: {exc}"))
        failed = [r["kernel"] for r in results if not r.get("ok")]
        if failed:
            st.error = f"kernels failed: {', '.join(failed)}"
        done(st, t, requested=len(wanted), ran=len(mine), shard=shard, shards=shards,
             failed=failed,
             real_gpu=any(r.get("real_gpu") for r in results))
    else:
        st.status = "skipped"
        done(st, t, ran=0)
        st.status = "skipped"

    # ------------------------------------------------------------------ 5. drift
    st = stage("drift")
    t = time.time()
    if drift:
        from . import selfcheck
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = selfcheck.run_checks(root)
        summary = buf.getvalue().strip().splitlines()
        if rc != 0:
            st.error = "the package no longer agrees with the notebooks"
        done(st, t, returncode=rc, lines=summary[-3:] if summary else [])
    else:
        st.status = "skipped"
        done(st, t)
        st.status = "skipped"

    return _emit(spec, stages, plan, rows, pairs, results, t0, stack=stack)


def _emit(spec, stages, plan, rows, pairs, kernels, t0, stack=None) -> dict:
    failed = [s.name for s in stages if s.status == "failed"]
    doc = dict(
        schema=1,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        seconds=round(time.time() - t0, 3),
        status="failed" if failed else "ok",
        failed_stages=failed,
        environment=_env(),
        spec=spec,
        stages=[dict(name=s.name, status=s.status, seconds=round(s.seconds, 3),
                     error=s.error, **s.detail) for s in stages],
    )
    if plan is not None:
        from .api import _jsonable
        doc["plan"] = _jsonable(plan)
        doc["ladder"] = _jsonable(rows)
        doc["interactions"] = _jsonable(pairs)
        doc["kernels"] = kernels
        doc["text"] = dict(plan=plan_report(stack), ladder=ladder_table(rows),
                           interactions=interaction_table(pairs))
        # Every claim in this document that rests on a timing rests on an emulation unless this
        # says otherwise. Stating it in the artifact rather than only in prose means a downstream
        # reader cannot miss it.
        doc["provenance"] = dict(
            planning_model="arithmetic over vendor specifications and list prices, not measured",
            kernels_device=(kernels[0]["device"] if kernels else None),
            kernels_timed=any(k.get("timed") for k in (kernels or [])),
            caveat=("Kernel correctness is verified. Timings are absent unless kernels_timed is "
                    "true, which requires a real GPU; the CPU shim emulates nothing about "
                    "performance."),
        )
    return doc


def merge(paths: list[Path]) -> dict:
    """Combine shard scorecards into one. Used by the Kubernetes Job's completion step."""
    docs = [json.loads(p.read_text()) for p in paths]
    if not docs:
        raise ValueError("nothing to merge")
    base = dict(docs[0])
    base["shards"] = len(docs)
    base["kernels"] = [k for d in docs for k in (d.get("kernels") or [])]
    base["environment"] = dict(base["environment"], merged_from=[
        dict(pod=d["environment"].get("pod"), shard=d["environment"].get("shard"),
             seconds=d.get("seconds")) for d in docs])
    failed = sorted({s for d in docs for s in d.get("failed_stages", [])})
    base["failed_stages"] = failed
    base["status"] = "failed" if failed else "ok"
    base["seconds"] = round(max(d.get("seconds", 0) for d in docs), 3)
    if base.get("provenance"):
        base["provenance"]["kernels_timed"] = any(k.get("timed") for k in base["kernels"])
    return base


def write(doc: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False))
    return path


__all__ = ["run", "merge", "write", "kernels_for", "shard_of", "Stage"]
