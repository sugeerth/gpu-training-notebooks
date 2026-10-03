"""The API and the pipeline, exercised in-process.

These run without a container: the handler is a `BaseHTTPRequestHandler`, so a real server on a
free port is a couple of lines and much more honest than mocking the transport. The pipeline is
run with kernel verification off, because compiling twenty-two C++ programs belongs in the
kernels CI job rather than in a unit test.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import servingkit as sk
from servingkit import api, pipeline


@pytest.fixture(scope="module")
def base_url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def get(url, expect=200):
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        assert e.code == expect, f"{url} -> {e.code}"
        return e.code, json.loads(e.read() or b"{}")


def post(url, body, expect=200):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        assert e.code == expect, f"{url} -> {e.code}"
        return e.code, json.loads(e.read() or b"{}")


AGENT = {"kind": "agent", "turns": 30, "fanout": 4, "tool_result_tokens": 2000,
         "replayed": True, "gpu": "H100 SXM"}


def test_health_and_ready(base_url):
    assert get(f"{base_url}/healthz")[1]["status"] == "ok"
    code, d = get(f"{base_url}/readyz")
    assert code == 200 and d["status"] == "ready" and d["levers"] >= 17


def test_catalog_is_the_package(base_url):
    _, d = get(f"{base_url}/v1/catalog")
    assert set(d["gpus"]) == set(sk.GPUS)
    assert set(d["models"]) == set(sk.MODELS)
    assert set(d["resources"]) == set(sk.RESOURCES)


def test_plan_matches_the_library(base_url):
    _, d = post(f"{base_url}/v1/plan", AGENT)
    w = api.workload_from_spec(AGENT)
    want = sk.recommend(w, AGENT["gpu"])
    assert d["levers"] == want.names
    assert d["plan"]["total_cost_usd"] == pytest.approx(want.plan()["total_cost_usd"])
    # The refusals are the useful half of the answer, so they must survive serialization.
    assert d["unavailable"] and all(u["missing"] for u in d["unavailable"])


def test_bodies_are_strict_json_with_no_infinities(base_url):
    """`concurrent` is legitimately infinite for a zero-KV workload; `Infinity` is not JSON."""
    for spec in (AGENT, {"kind": "chat", "ctx": 512, "concurrency": 1}):
        _, d = post(f"{base_url}/v1/ladder", spec)
        json.dumps(d, allow_nan=False)      # raises if anything leaked through


def test_error_paths(base_url):
    assert post(f"{base_url}/v1/plan", {"model": "nope"}, expect=400)[0] == 400
    assert post(f"{base_url}/v1/plan", {"gpu": "nope"}, expect=400)[0] == 400
    assert get(f"{base_url}/v1/nope", expect=404)[0] == 404


def test_unknown_keys_are_ignored_not_rejected(base_url):
    """A console written against a later API must degrade rather than break."""
    code, _ = post(f"{base_url}/v1/plan", dict(AGENT, some_future_field=1, another="x"))
    assert code == 200


def test_metrics_are_prometheus_text(base_url):
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as r:
        body = r.read().decode()
    assert r.headers["Content-Type"].startswith("text/plain")
    assert "servingkit_requests_total" in body and "servingkit_levers" in body


def test_pipeline_runs_and_emits_a_valid_scorecard():
    doc = pipeline.run(AGENT, verify=False, drift=False)
    assert doc["status"] == "ok"
    assert [s["name"] for s in doc["stages"]] == \
        ["resolve", "plan", "interact", "verify", "drift"]
    json.dumps(doc, allow_nan=False)
    # The provenance block must state plainly that nothing here was timed on real hardware.
    assert doc["provenance"]["kernels_timed"] is False


def test_pipeline_names_the_kernels_its_plan_relies_on():
    doc = pipeline.run(AGENT, verify=False, drift=False)
    named = {sk.LEVERS[n].kernel for n in doc["plan"]["levers"] if sk.LEVERS[n].kernel}
    assert named, "an agent plan should rest on at least one kernel"
    root = sk.kernels.repo_root()
    for k in named:
        assert (root / "kernels" / k).exists()


def test_pipeline_fails_when_a_lever_misdeclares_itself(monkeypatch):
    """The interact stage is a gate, not a warning.

    `Lever` is frozen, so this swaps in a replaced copy rather than mutating one — which is the
    dataclass doing its job: a lever's declarations are not something code should be able to
    edit in place while a plan is being computed against them.
    """
    import dataclasses
    liar = dataclasses.replace(sk.LEVERS["speculation"], spends=())
    monkeypatch.setitem(sk.LEVERS, "speculation", liar)
    doc = pipeline.run({"kind": "chat", "ctx": 1024, "concurrency": 8,
                        "levers": ["int4 weights", "speculation"]}, verify=False, drift=False)
    interact = next(s for s in doc["stages"] if s["name"] == "interact")
    assert interact["status"] == "failed" and interact["undeclared"]
    assert doc["status"] == "failed"


def test_shards_partition_the_work_exactly_once():
    items = [f"{i:02d}_k.cu" for i in range(22)]
    for n in (1, 3, 6, 7):
        got = [x for s in range(n) for x in pipeline.shard_of(items, s, n)]
        assert sorted(got) == sorted(items), f"{n} shards lost or duplicated work"
        sizes = [len(pipeline.shard_of(items, s, n)) for s in range(n)]
        assert max(sizes) - min(sizes) <= 1, f"{n} shards are unbalanced: {sizes}"


def test_merge_combines_shards():
    a = pipeline.run(AGENT, verify=False, drift=False)
    b = pipeline.run(AGENT, verify=False, drift=False)
    a["kernels"] = [{"kernel": "13_prefix_attention", "ok": True, "timed": False}]
    b["kernels"] = [{"kernel": "14_logit_mask", "ok": True, "timed": False}]
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        pa, pb = Path(d) / "a.json", Path(d) / "b.json"
        pipeline.write(a, pa)
        pipeline.write(b, pb)
        m = pipeline.merge([pa, pb])
    assert m["shards"] == 2 and len(m["kernels"]) == 2 and m["status"] == "ok"


# ------------------------------------------------------- the agent, over HTTP and in-pipeline
def test_agent_route_returns_a_defensible_trace(base_url):
    """The service's answer must carry the reasoning, not only the conclusion."""
    _, d = post(f"{base_url}/v1/agent", {**AGENT, "max_tpot_ms": 25, "max_steps": 8})
    tr = d["trace"]
    assert tr["outcome"] in ("satisfied", "converged", "budget", "infeasible")
    assert tr["decisions"], "a trace with no decisions explains nothing"
    for step in tr["decisions"]:
        # Every step says what it predicted, what it measured, and whether it was kept. A step
        # missing any of the three is a step nobody can audit.
        assert step["predicted"] and step["actual"] and step["verdict"]
        assert isinstance(step["accepted"], bool)
    assert d["levers"] == tr["final"]["levers"]
    assert "outcome:" in d["text"]


def test_agent_route_respects_the_slo_it_is_given(base_url):
    _, d = post(f"{base_url}/v1/agent", {**AGENT, "max_tpot_ms": 25, "max_steps": 10})
    tr = d["trace"]
    if tr["outcome"] != "infeasible":
        assert tr["final"]["plan"]["tpot_ms"] <= 25 + 1e-6
    else:
        assert tr["violations"] and tr["engineer"]


def test_agent_route_is_bounded(base_url):
    """A route that compiles nothing can still be made to do arbitrary work. It must not be."""
    _, d = post(f"{base_url}/v1/agent", {**AGENT, "max_steps": 9999})
    assert len(d["trace"]["decisions"]) <= 50


def test_logs_route_serves_the_ring_and_its_counts(base_url):
    post(f"{base_url}/v1/plan", AGENT)
    _, d = get(f"{base_url}/v1/logs?limit=50")
    kinds = {e["kind"] for e in d["events"]}
    assert "http.request" in kinds, kinds
    assert d["counts"].get("http.request", 0) >= 1
    assert len(d["events"]) <= d["capacity"]
    # A request event names its route and outcome, which is the minimum for an access log.
    req = [e for e in d["events"] if e["kind"] == "http.request"][-1]
    assert req["data"]["route"] and req["data"]["status"] and "ms" in req["data"]


def test_logs_can_be_filtered_by_kind_and_run(base_url):
    post(f"{base_url}/v1/agent", {**AGENT, "max_steps": 4})
    _, d = get(f"{base_url}/v1/logs?kind=agent.&limit=200")
    assert d["events"] and all(e["kind"].startswith("agent.") for e in d["events"])
    run = d["events"][-1]["run"]
    _, one = get(f"{base_url}/v1/logs?run={run}&limit=200")
    assert one["events"] and all(e["run"] == run for e in one["events"])


def test_loghealth_reports_the_hooks(base_url):
    _, d = get(f"{base_url}/v1/loghealth")
    assert {"hooks", "events", "muted", "dropped"} <= set(d)
    assert any(h["name"] == "ring" for h in d["hooks"])
    assert d["events"] >= 1 and d["dropped"] == 0


def test_metrics_include_the_event_stream(base_url):
    get(f"{base_url}/v1/catalog")
    import urllib.request as u
    with u.urlopen(f"{base_url}/metrics", timeout=10) as r:
        text = r.read().decode()
    assert "servingkit_events_total{" in text
    assert "servingkit_log_hooks_muted 0" in text
    assert "servingkit_log_events_dropped_total 0" in text


def test_pipeline_tune_stage_puts_its_reasoning_in_the_scorecard():
    doc = pipeline.run({**AGENT, "max_tpot_ms": 25, "max_steps": 8},
                       verify=False, drift=False, tune=True)
    tune = next(s for s in doc["stages"] if s["name"] == "tune")
    assert tune["status"] == "ok" and tune["steps"] >= 1
    assert doc["tuning"]["decisions"], "the trace must survive into the artifact"
    # The plan that got costed is the one the loop arrived at, not recommend()'s guess.
    assert doc["plan"]["levers"] == doc["tuning"]["final"]["levers"] == tune["levers"]


def test_pipeline_tune_stage_fails_on_a_lever_that_lies():
    """The same gate as `interact`, reached by the loop instead of by enumeration."""
    liar = sk.Lever("2x batch again", lambda c: {**c, "batch": c["batch"] * 2},
                    domain="batch_size_2", spends=(), requires=("weight_bound",),
                    note="mis-declared on purpose")
    sk.LEVERS[liar.name] = liar
    try:
        doc = pipeline.run({"kind": "chat", "ctx": 2048, "concurrency": 32,
                            "maximize": "tokens_per_s", "allow_gpu_moves": False,
                            "levers": ["2x batch"], "max_steps": 6},
                           verify=False, drift=False, tune=True)
    finally:
        del sk.LEVERS[liar.name]
    tune = next(s for s in doc["stages"] if s["name"] == "tune")
    assert tune["status"] == "failed" and tune["findings"]
    assert doc["status"] == "failed"
