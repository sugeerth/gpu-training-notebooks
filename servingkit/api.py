"""A JSON service over the composable layer, on the standard library only.

No FastAPI, no uvicorn, no pydantic. That is a deliberate constraint rather than austerity: the
package's whole claim is that it is arithmetic over dicts, and a service that needs three
hundred megabytes of wheels to expose that arithmetic would undermine the claim. The container
this runs in is `python:3.12-slim` plus this repository, and it starts in well under a second.

    python -m servingkit serve --port 8000
    curl -s localhost:8000/v1/plan -d '{"kind":"agent","turns":30,"fanout":4}' | jq .

Routes
------
    GET  /healthz                 liveness — no work, no imports, no allocation
    GET  /readyz                  readiness — the models answer, and agree with themselves
    GET  /v1/catalog              GPUs, models, precisions, resources, properties
    GET  /v1/levers               every lever and its four declarations
    POST /v1/workload             derive a workload's properties, bottleneck, raggedness
    POST /v1/plan                 the recommended stack, costed
    POST /v1/ladder               one rung per lever, ordered by effort, costed in money
    POST /v1/interactions         pairwise gains with the predicted conflict
    POST /v1/cheapest             feasible deployments, cheapest per million tokens first
    GET  /v1/kernels              what kernels exist and which lever names each
    POST /v1/kernels/{name}/run   compile and run one, no GPU required
    GET  /v1/scorecard            the last pipeline run, if one has been written
    GET  /metrics                 Prometheus text format

Every POST body is a workload spec, the same shape the frontend and the CLI use, so there is one
vocabulary across the whole system:

    {"kind": "agent", "model": "Llama-3.1-8B", "turns": 30, "fanout": 4,
     "tool_result_tokens": 2000, "replayed": true, "gpu": "H100 SXM"}
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import report
from .catalog import GPUS, MODELS, PRECISION
from .kernels import available_kernels, repo_root, run_kernel
from .levers import LEVERS, PROPERTIES, RESOURCES
from .serving import cheapest
from .stack import Stack, recommend
from .workload import Workload

STARTED = time.time()
_COUNTS: dict[str, int] = {}
_LATENCY: dict[str, float] = {}
_LOCK = threading.Lock()

# Where the pipeline writes its scorecard. The API only ever reads it, so a pipeline pod and an
# API pod can share a volume without either needing to know about the other.
SCORECARD = Path(os.environ.get("SERVINGKIT_SCORECARD", "/var/lib/servingkit/scorecard.json"))

# A kernel run compiles C++, which takes seconds and is not something to let an anonymous caller
# trigger without limit. One at a time, and only when explicitly enabled.
KERNELS_ENABLED = os.environ.get("SERVINGKIT_ENABLE_KERNELS", "1") not in ("0", "false", "")
_KERNEL_LOCK = threading.Semaphore(int(os.environ.get("SERVINGKIT_KERNEL_CONCURRENCY", "1")))


# --------------------------------------------------------------------------- spec -> objects
def workload_from_spec(spec: dict) -> Workload:
    """Build a Workload from a JSON spec, ignoring keys it does not own.

    Unknown keys are dropped rather than rejected. A frontend that sends `gpu` alongside the
    workload fields should not get a 400 for it, and a frontend written against a later version
    of this API should degrade rather than break.
    """
    kind = (spec.get("kind") or "agent").lower()
    common = dict(
        model=spec.get("model", "Llama-3.1-8B"),
        gpu=spec.get("gpu", "H100 SXM"),
        batch=int(spec.get("concurrency", spec.get("batch", 32))),
        replayed=bool(spec.get("replayed", False)),
        prefix_hit_rate=float(spec.get("prefix_hit_rate", 0.0)),
    )
    if "price_in_per_mtok" in spec:
        common["price_in_per_mtok"] = float(spec["price_in_per_mtok"])
    if "price_out_per_mtok" in spec:
        common["price_out_per_mtok"] = float(spec["price_out_per_mtok"])
    if "structured_output" in spec:
        common["structured_output"] = bool(spec["structured_output"])

    if kind == "chat":
        return Workload.chat(ctx=int(spec.get("ctx", 4096)),
                             output_tokens=int(spec.get("output_tokens", 400)), **common)
    if kind == "batch":
        return Workload.batch(ctx=int(spec.get("ctx", 8192)),
                              output_tokens=int(spec.get("output_tokens", 1000)), **common)
    return Workload.agent(
        turns=int(spec.get("turns", 20)),
        system_tokens=int(spec.get("system_tokens", 2000)),
        tool_def_tokens=int(spec.get("tool_def_tokens", 1500)),
        tool_result_tokens=int(spec.get("tool_result_tokens", 800)),
        output_tokens=int(spec.get("output_tokens", 150)),
        tool_latency_s=float(spec.get("tool_latency_s", 2.0)),
        fanout=int(spec.get("fanout", 1)),
        **common)


def stack_from_spec(spec: dict) -> Stack:
    w = workload_from_spec(spec)
    gpu = spec.get("gpu", "H100 SXM")
    if gpu not in GPUS:
        raise ValueError(f"unknown gpu {gpu!r}")
    names = spec.get("levers")
    return Stack(w, gpu).with_levers(*names) if names else recommend(w, gpu)


def _jsonable(x):
    """Make a plan safely JSON-encodable, including its Rejection dataclasses and infinities."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "__dataclass_fields__"):
        return {k: _jsonable(getattr(x, k)) for k in x.__dataclass_fields__}
    if isinstance(x, float):
        # JSON has no Infinity. `concurrent` is legitimately infinite for a zero-KV workload, and
        # silently emitting `Infinity` produces a body that every strict parser rejects.
        if x != x:
            return None
        if x in (float("inf"), float("-inf")):
            return None
        return x
    return x


# ------------------------------------------------------------------------------- handlers
def h_catalog(_spec: dict) -> dict:
    return dict(
        gpus={k: dict(v) for k, v in GPUS.items()},
        models={k: dict(v) for k, v in MODELS.items()},
        precisions={k: dict(v) for k, v in PRECISION.items()},
        resources=dict(RESOURCES),
        properties=dict(PROPERTIES),
        kinds=["chat", "agent", "batch"],
    )


def h_levers(spec: dict) -> dict:
    props = workload_from_spec(spec).properties if spec else None
    out = []
    for name, lv in LEVERS.items():
        out.append(dict(name=name, domain=lv.domain, spends=list(lv.spends),
                        requires=list(lv.requires), kernel=lv.kernel, note=lv.note,
                        applies=None if props is None else lv.applies_to(props),
                        missing=[] if props is None else lv.missing(props)))
    return dict(levers=out, properties=sorted(props) if props else None)


def h_workload(spec: dict) -> dict:
    w = workload_from_spec(spec)
    gpu = spec.get("gpu", "H100 SXM")
    return dict(kind=w.kind, model=w.model, ctx=w.ctx, batch=w.batch,
                properties=sorted(w.properties), bottleneck=w.bottleneck(gpu),
                spare_compute=w.spare_compute(gpu), raggedness=_jsonable(w.raggedness()),
                describe=w.describe())


def h_plan(spec: dict) -> dict:
    s = stack_from_spec(spec)
    return dict(plan=_jsonable(s.plan()), levers=s.names,
                offered=[lv.name for lv in s.offered()],
                unavailable=[dict(lever=lv.name, missing=m, kernel=lv.kernel)
                             for lv, m in s.unavailable() if lv.name not in s.names],
                text=report.plan_report(s))


def h_ladder(spec: dict) -> dict:
    s = stack_from_spec(spec)
    rows = s.ladder()
    return dict(rows=_jsonable(rows), text=report.ladder_table(rows))


def h_interactions(spec: dict) -> dict:
    s = stack_from_spec(spec)
    pairs = s.interactions()
    return dict(pairs=_jsonable(pairs), text=report.interaction_table(pairs))


def h_cheapest(spec: dict) -> dict:
    rows = cheapest(spec.get("model", "Llama-3.1-8B"),
                    batch=int(spec.get("concurrency", spec.get("batch", 32))),
                    ctx=int(spec.get("ctx", 2048)),
                    min_tpot_ms=spec.get("max_tpot_ms"))
    return dict(rows=_jsonable(rows[:int(spec.get("limit", 25))]), total=len(rows))


def h_kernels(_spec: dict) -> dict:
    by_kernel: dict[str, list[str]] = {}
    for name, lv in LEVERS.items():
        if lv.kernel:
            by_kernel.setdefault(lv.kernel, []).append(name)
    return dict(kernels=[dict(name=k, levers=by_kernel.get(k + ".cu", []))
                         for k in available_kernels()],
                enabled=KERNELS_ENABLED)


def h_kernel_run(name: str) -> dict:
    if not KERNELS_ENABLED:
        raise PermissionError("kernel runs are disabled (SERVINGKIT_ENABLE_KERNELS=0)")
    if not _KERNEL_LOCK.acquire(blocking=False):
        raise BlockingIOError("a kernel is already compiling; try again shortly")
    try:
        r = run_kernel(name)
        return dict(name=r.name, device=r.device, real_gpu=r.real_gpu, ok=r.ok, timed=r.timed,
                    tol=r.tol, variants=_jsonable(r.variants), table=r.table())
    finally:
        _KERNEL_LOCK.release()


def h_scorecard(_spec: dict) -> dict:
    if not SCORECARD.exists():
        raise FileNotFoundError(f"no scorecard at {SCORECARD}; run the pipeline first")
    return json.loads(SCORECARD.read_text())


POST_ROUTES = {
    "/v1/workload": h_workload,
    "/v1/plan": h_plan,
    "/v1/ladder": h_ladder,
    "/v1/interactions": h_interactions,
    "/v1/cheapest": h_cheapest,
    "/v1/levers": h_levers,
}
GET_ROUTES = {
    "/v1/catalog": h_catalog,
    "/v1/levers": h_levers,
    "/v1/kernels": h_kernels,
    "/v1/scorecard": h_scorecard,
}


# --------------------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    server_version = "servingkit"
    sys_version = ""

    def log_message(self, fmt, *args):  # structured, one JSON object per line
        print(json.dumps(dict(ts=time.time(), addr=self.client_address[0],
                              msg=fmt % args)), flush=True)

    # ------------------------------------------------------------------ plumbing
    def _send(self, code: int, body: dict | str, ctype: str = "application/json") -> None:
        if isinstance(body, dict):
            payload = json.dumps(body, indent=2, allow_nan=False).encode()
        else:
            payload = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        # The console is served from a different origin than the API in every deployment shape
        # here — an Artifact, a static bucket, a sidecar — so CORS is not optional.
        self.send_header("Access-Control-Allow-Origin", os.environ.get("SERVINGKIT_CORS", "*"))
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"body is not JSON: {exc}") from exc
        if not isinstance(spec, dict):
            raise ValueError("body must be a JSON object")
        return spec

    def _record(self, route: str, started: float) -> None:
        with _LOCK:
            _COUNTS[route] = _COUNTS.get(route, 0) + 1
            _LATENCY[route] = _LATENCY.get(route, 0.0) + (time.time() - started)

    def _dispatch(self, route: str, fn, spec: dict) -> None:
        started = time.time()
        try:
            self._send(200, fn(spec))
        except PermissionError as exc:
            self._send(403, dict(error=str(exc)))
        except BlockingIOError as exc:
            self._send(429, dict(error=str(exc)))
        except FileNotFoundError as exc:
            self._send(404, dict(error=str(exc)))
        except (KeyError, ValueError) as exc:
            # A bad model name arrives as a KeyError from the catalog; that is the caller's
            # mistake, not the server's, so it is a 400 with the offending key named.
            self._send(400, dict(error=f"{type(exc).__name__}: {exc}"))
        except Exception as exc:  # noqa: BLE001 - the last line of defence must not crash
            self._send(500, dict(error=str(exc), type=type(exc).__name__,
                                 traceback=traceback.format_exc().splitlines()[-4:]))
        finally:
            self._record(route, started)

    # ------------------------------------------------------------------- methods
    def do_OPTIONS(self):  # noqa: N802
        self._send(204, "")

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        path = u.path.rstrip("/") or "/"

        if path == "/healthz":
            return self._send(200, dict(status="ok", uptime_s=round(time.time() - STARTED, 3)))
        if path == "/readyz":
            return self._dispatch(path, lambda _s: self._ready(), {})
        if path == "/metrics":
            return self._send(200, self._prometheus(), "text/plain")
        if path in ("/", "/v1"):
            return self._send(200, dict(
                service="servingkit", version=_version(),
                routes=sorted(set(list(GET_ROUTES) + list(POST_ROUTES)
                                  + ["/healthz", "/readyz", "/metrics",
                                     "/v1/kernels/{name}/run"]))))
        if path.startswith("/v1/kernels/") and path.endswith("/run"):
            name = path[len("/v1/kernels/"):-len("/run")]
            return self._dispatch("/v1/kernels/run", lambda _s: h_kernel_run(name), {})
        fn = GET_ROUTES.get(path)
        if not fn:
            return self._send(404, dict(error=f"no route {path}"))
        # Query parameters become a spec, so every POST route is also reachable by GET for
        # anything a browser address bar can express.
        spec = {k: (v[0] if len(v) == 1 else v) for k, v in parse_qs(u.query).items()}
        return self._dispatch(path, fn, spec)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path.startswith("/v1/kernels/") and path.endswith("/run"):
            name = path[len("/v1/kernels/"):-len("/run")]
            return self._dispatch("/v1/kernels/run", lambda _s: h_kernel_run(name), {})
        fn = POST_ROUTES.get(path)
        if not fn:
            return self._send(404, dict(error=f"no route {path}"))
        try:
            spec = self._body()
        except ValueError as exc:
            return self._send(400, dict(error=str(exc)))
        return self._dispatch(path, fn, spec)

    # ------------------------------------------------------------------- helpers
    def _ready(self) -> dict:
        """Readiness means the models answer and agree with themselves — not merely that the
        process is up. A pod whose arithmetic is broken should not receive traffic."""
        w = Workload.agent(turns=5)
        plan = recommend(w).plan()
        bad = [f"{c['a']}+{c['b']}"
               for c in Stack(Workload.chat(ctx=2048, batch=32)).all_applicable().interactions()
               if c["synergy"] < 0.9 and not c["predicted_conflict"]]
        if bad:
            raise RuntimeError(f"lever declarations disagree with measurement: {bad}")
        return dict(status="ready", levers=len(LEVERS), kernels=len(available_kernels()),
                    sample_cost_usd=round(plan["total_cost_usd"], 6),
                    repo=str(repo_root()))

    def _prometheus(self) -> str:
        lines = ["# HELP servingkit_requests_total Requests by route.",
                 "# TYPE servingkit_requests_total counter"]
        with _LOCK:
            counts, lat = dict(_COUNTS), dict(_LATENCY)
        for route, n in sorted(counts.items()):
            lines.append(f'servingkit_requests_total{{route="{route}"}} {n}')
        lines += ["# HELP servingkit_request_seconds_sum Cumulative handler time by route.",
                  "# TYPE servingkit_request_seconds_sum counter"]
        for route, t in sorted(lat.items()):
            lines.append(f'servingkit_request_seconds_sum{{route="{route}"}} {t:.6f}')
        lines += ["# HELP servingkit_uptime_seconds Process uptime.",
                  "# TYPE servingkit_uptime_seconds gauge",
                  f"servingkit_uptime_seconds {time.time() - STARTED:.3f}",
                  "# HELP servingkit_levers Levers in the registry.",
                  "# TYPE servingkit_levers gauge",
                  f"servingkit_levers {len(LEVERS)}"]
        return "\n".join(lines) + "\n"


def _version() -> str:
    from . import __version__
    return __version__


def serve(host: str = "0.0.0.0", port: int = 8000) -> None:  # noqa: S104 - a container needs this
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    print(json.dumps(dict(event="listening", host=host, port=port, version=_version(),
                          levers=len(LEVERS), kernels_enabled=KERNELS_ENABLED)), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


__all__ = ["serve", "Handler", "workload_from_spec", "stack_from_spec"]
