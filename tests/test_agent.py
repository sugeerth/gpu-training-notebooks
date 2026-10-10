"""The control loop and the log hooks.

Two things are worth testing here and they are not the obvious ones.

The loop's *conclusions* are barely worth asserting: they are arithmetic over a planning model,
and a test that pins "cascade attention comes second" pins a catalog price rather than a
behaviour. What is worth asserting is that the loop cannot lie to itself — that a reverted action
is really reverted, that a mis-declared lever is caught, that an SLO is never traded for a cost
saving, and that the deadband keeps it from shipping a change worth nothing.

For the hooks, the property worth testing is the failure mode. A log sink that raises must not
take the run down, and a sink that *blocks* must not take the latency with it. Both are tested
with sinks that deliberately misbehave, because a hook system that has only ever been run against
a working collector has not been tested at all.
"""
from __future__ import annotations

import io
import json
import threading
import time

import pytest

import servingkit as sk
from servingkit import agent as ag
from servingkit import events as ev
from servingkit.levers import LEVERS, Lever

AGENT = dict(turns=30, fanout=4, tool_result_tokens=2000, replayed=True)


@pytest.fixture
def bus():
    """A private bus, so a test's events cannot reach the package-wide one or vice versa."""
    b = ev.EventBus()
    yield b
    b.clear()


# --------------------------------------------------------------------------- the objective
def test_constraints_dominate_the_objective():
    """A cheaper plan that breaks the SLO must never beat a dearer one that meets it."""
    obj = ag.Objective(max_tpot_ms=20, minimize="total_cost_usd")
    meets = dict(tpot_ms=19.0, wall_s=1, total_cost_usd=10.0, tokens_per_s=100)
    breaks = dict(tpot_ms=25.0, wall_s=1, total_cost_usd=1.0, tokens_per_s=100)
    assert obj.score(meets) < obj.score(breaks)
    assert obj.satisfied(meets) and not obj.satisfied(breaks)
    assert obj.violations(breaks)[0]["constraint"] == "tpot_ms"


def test_the_deadband_rejects_a_change_worth_nothing():
    """The first version of the loop pulled a lever for a 0.004% gain. It should not."""
    obj = ag.Objective(minimize="total_cost_usd", min_improvement=0.005)
    before = dict(tpot_ms=1, wall_s=1, tokens_per_s=1, total_cost_usd=1.0)
    assert not obj.worth_it(before, {**before, "total_cost_usd": 0.999})   # 0.1%
    assert obj.worth_it(before, {**before, "total_cost_usd": 0.98})        # 2%


def test_fixing_an_slo_breach_overrides_the_deadband():
    obj = ag.Objective(max_tpot_ms=10, min_improvement=0.5)
    before = dict(tpot_ms=11.0, wall_s=1, tokens_per_s=1, total_cost_usd=1.0)
    after = dict(before, tpot_ms=9.9)          # no cost change at all, but now legal
    assert obj.worth_it(before, after)


# -------------------------------------------------------------------------------- the loop
def test_every_accepted_action_improves_the_objective(bus):
    w = sk.Workload.agent(**AGENT)
    tr = ag.ControlLoop(w, ag.Objective(max_tpot_ms=25), bus=bus).run(max_steps=12)
    obj = ag.Objective(max_tpot_ms=25)
    prev = tr.start_plan
    for d in tr.accepted:
        assert obj.score(d.actual) < obj.score(prev), f"{d.action} did not improve anything"
        prev = d.actual
    assert tr.outcome in ("satisfied", "converged", "budget", "infeasible")


def test_a_rejected_action_leaves_no_trace_in_the_stack(bus):
    """A revert must revert. If the stack keeps a rejected lever the whole loop is a lie.

    Batching under a *latency* objective is the clean case: doubling the batch raises throughput,
    which is what its config gain predicts, and raises per-token latency, which is what the
    measurement finds — so the loop proposes it, measures it, and puts it back.
    """
    w = sk.Workload.chat(model="Llama-3.1-8B", ctx=2048, batch=8)
    loop = ag.ControlLoop(w, ag.Objective(minimize="tpot_ms"), bus=bus)
    tr = loop.run(max_steps=8)
    rejected = [d for d in tr.decisions if not d.accepted]
    assert rejected, f"expected a rejection, got {[d.action for d in tr.decisions]}"
    assert any(d.action == "add 2x batch" for d in rejected)
    for d in rejected:
        if d.verb == "add":
            assert d.action.removeprefix("add ") not in loop.stack.names
        if d.verb == "move":
            assert d.action.removeprefix("move to ") != loop.gpu
        # The decision records the state *after* settling, so a rejected step must record the
        # state it reverted to and not the one it was trying out.
        assert d.gpu_after == loop.gpu or any(later.accepted for later in
                                              tr.decisions[tr.decisions.index(d) + 1:])
    # And a rejected action is struck off, so the loop cannot spend its budget oscillating.
    assert loop.refused


def test_moves_are_only_offered_to_cards_the_step_model_knows(bus):
    """A card the step budget does not model is evaluated as a different card entirely.

    `_step_gpu()` maps anything outside `STEP_HW` onto the nearest card it knows, so offering
    "move to B200" meant predicting 2.4x from B200's bandwidth and then measuring an unchanged
    H100. The loop dutifully reverted it, and the rejection said nothing about a B200.
    """
    from servingkit.step import STEP_HW
    loop = ag.ControlLoop(sk.Workload.agent(**AGENT), ag.Objective(maximize="tokens_per_s"),
                          bus=bus)
    moves = {a.gpu for a in loop.candidates() if a.verb == "move"}
    assert moves == set(STEP_HW) - {loop.gpu}
    assert "B200" not in moves and "B200" in sk.GPUS


def test_an_infeasible_objective_says_what_to_build(bus):
    tr = ag.ControlLoop(sk.Workload.chat(ctx=4096, batch=32),
                        ag.Objective(max_tpot_ms=0.5), bus=bus).run(max_steps=6)
    assert tr.outcome == "infeasible"
    assert tr.violations and tr.violations[0]["constraint"] == "tpot_ms"
    # The useful half: which workload properties would unlock more levers.
    assert tr.engineer and all(e["property"] in sk.PROPERTIES for e in tr.engineer)
    assert all(lv in LEVERS for e in tr.engineer for lv in e["levers"])


def test_accuracy_is_measured_on_the_objective_not_on_throughput(bus):
    """A cost run whose cost predictions are out must not report a perfect score.

    This is the check that the first version failed: it scored every step on tokens per second,
    so a 40%-out cost prediction still read 1.00.
    """
    loop = ag.ControlLoop(sk.Workload.agent(**AGENT), ag.Objective(minimize="total_cost_usd"),
                          bus=bus)
    before = loop.observe()
    pred = dict(before, total_cost_usd=before["total_cost_usd"] / 2)
    got = dict(before, total_cost_usd=before["total_cost_usd"] * 0.75)
    ratio, axis = loop._accuracy(before, pred, got)
    assert axis == "total_cost_usd"
    assert ratio == pytest.approx(0.5, abs=1e-6)      # half the predicted saving arrived


def test_the_loop_never_spends_accuracy_unless_allowed(bus):
    w = sk.Workload.agent(**AGENT)
    tr = ag.ControlLoop(w, ag.Objective(maximize="tokens_per_s"), bus=bus).run(max_steps=14)
    for name in tr.final_levers:
        assert "accuracy" not in LEVERS[name].spends, f"{name} spends accuracy uninvited"
    tr2 = ag.ControlLoop(w, ag.Objective(maximize="tokens_per_s", allow_accuracy_loss=True),
                         bus=bus).run(max_steps=14)
    assert tr2.final_plan["tokens_per_s"] >= tr.final_plan["tokens_per_s"]


def test_a_stack_never_holds_two_levers_from_one_domain(bus):
    tr = ag.ControlLoop(sk.Workload.agent(**AGENT),
                        ag.Objective(maximize="tokens_per_s", allow_accuracy_loss=True),
                        bus=bus).run(max_steps=20)
    domains = [LEVERS[n].domain for n in tr.final_levers]
    assert len(domains) == len(set(domains))


def test_a_mis_declared_lever_is_caught(bus, monkeypatch):
    """The finding path, against a lever that really does lie about what it spends.

    Two levers that each double the batch, in different domains, declaring nothing. Batching is
    paid out of spare compute, so the second one cannot pay what the first already spent — and
    because neither declares `spare_compute`, the loop has no explanation for the shortfall and
    must say so. Registering the liar in the test rather than shipping one is the point: the
    check has to be demonstrated against a real violation, not asserted about.
    """
    liar = Lever("2x batch again", lambda c: {**c, "batch": c["batch"] * 2},
                 domain="batch_size_2", spends=(), requires=("weight_bound",),
                 note="a deliberately mis-declared lever, for the test below")
    monkeypatch.setitem(LEVERS, liar.name, liar)
    w = sk.Workload.chat(ctx=2048, batch=32)
    assert "weight_bound" in w.properties, "this test needs a batchable workload"
    loop = ag.ControlLoop(w, ag.Objective(maximize="tokens_per_s", allow_gpu_moves=False),
                          levers=["2x batch"], bus=bus)
    tr = loop.run(max_steps=6)
    assert any(d.verdict == "short-undeclared" for d in tr.decisions), \
        f"verdicts were {[d.verdict for d in tr.decisions]}"
    assert tr.findings and "no declared shared resource" in tr.findings[0]


def test_a_declared_conflict_is_not_a_finding(bus, monkeypatch):
    """The same lever, honest about its resource, produces no finding. Same gap, named."""
    honest = Lever("2x batch again", lambda c: {**c, "batch": c["batch"] * 2},
                   domain="batch_size_2", spends=("spare_compute",),
                   requires=("weight_bound",), note="declares what it spends")
    monkeypatch.setitem(LEVERS, honest.name, honest)
    tr = ag.ControlLoop(sk.Workload.chat(ctx=2048, batch=32),
                        ag.Objective(maximize="tokens_per_s", allow_gpu_moves=False),
                        levers=["2x batch"], bus=bus).run(max_steps=6)
    assert not tr.findings
    assert not any(d.verdict == "short-undeclared" for d in tr.decisions)


def test_the_evaluation_budget_is_respected(bus):
    loop = ag.ControlLoop(sk.Workload.agent(**AGENT), ag.Objective(maximize="tokens_per_s"),
                          bus=bus, max_evaluations=8)
    tr = loop.run(max_steps=20)
    # A bound, with slack for the two evaluations a single pricing can cost and the closing
    # observe(). Without enforcement inside the ranking this reached 16 against a budget of 8.
    assert tr.evaluations <= 8 + 4, tr.evaluations
    assert len(tr.decisions) < 20


def test_the_trace_is_json_and_finite(bus):
    tr = ag.ControlLoop(sk.Workload.agent(**AGENT), ag.Objective(max_tpot_ms=25),
                        bus=bus).run(max_steps=8)
    blob = json.dumps(tr.as_dict(), allow_nan=False)      # raises on Infinity or NaN
    back = json.loads(blob)
    assert back["final"]["levers"] == tr.final_levers
    assert back["decisions"][0]["predicted"] and back["decisions"][0]["actual"]


# ------------------------------------------------------------------------------- the bus
def test_events_are_emitted_for_every_phase(bus):
    seen = []
    bus.subscribe(ev.CallableHook(seen.append))
    ag.ControlLoop(sk.Workload.agent(**AGENT), ag.Objective(max_tpot_ms=25),
                   bus=bus).run(max_steps=6)
    kinds = {e.kind for e in seen}
    assert {"agent.start", "agent.observe", "agent.decide", "agent.accept",
            "agent.done"} <= kinds
    # One run id across the whole run, and monotonic sequence numbers, so a consumer can detect
    # a gap rather than having to trust the transport.
    runs = {e.run for e in seen}
    assert len(runs) == 1 and next(iter(runs)).startswith("agent-")
    assert [e.seq for e in seen] == sorted(e.seq for e in seen)


def test_a_hook_that_raises_does_not_fail_the_run(bus):
    def explode(_ev):
        raise RuntimeError("the collector is down")

    bus.subscribe(ev.CallableHook(explode, name="broken"))
    ok = []
    bus.subscribe(ev.CallableHook(ok.append, name="fine"))
    tr = ag.ControlLoop(sk.Workload.agent(**AGENT), ag.Objective(max_tpot_ms=25),
                        bus=bus).run(max_steps=4)
    assert tr.outcome  # the run completed
    assert ok, "a working hook must still receive events when another one is broken"
    h = bus.health()
    assert h["errors"]["broken"] >= bus.MUTE_AFTER
    assert "broken" in h["muted"], "a hook that keeps failing must be muted, not retried forever"


def test_a_slow_hook_does_not_block_the_emitter(bus):
    """The failure mode that matters: a sink that hangs must drop, not wait."""
    release = threading.Event()

    class Molasses(ev.Hook):
        name = "molasses"

        def write(self, _ev):
            release.wait(30)

    inner = Molasses()
    hook = bus.subscribe(ev.Async(inner, maxsize=2))
    t0 = time.time()
    for i in range(200):
        bus.emit("test.flood", i=i)
    elapsed = time.time() - t0
    release.set()
    assert elapsed < 1.0, f"emitting took {elapsed:.2f}s: the queue is blocking"
    assert hook.dropped > 0, "a full queue must drop and count, not block"
    assert bus.health()["dropped"] == hook.dropped


def test_hook_specs_cover_the_documented_forms(tmp_path):
    assert isinstance(ev.from_spec("jsonl"), ev.JsonlHook)
    assert isinstance(ev.from_spec("text"), ev.TextHook)
    assert isinstance(ev.from_spec("counter"), ev.CounterHook)
    assert ev.from_spec("ring:64").events.maxlen == 64
    assert isinstance(ev.from_spec(f"file:{tmp_path / 'a.jsonl'}"), ev.FileHook)
    w = ev.from_spec("webhook:https://example.invalid/ingest")
    assert isinstance(w, ev.Async) and isinstance(w.inner, ev.WebhookHook)
    assert ev.from_spec("jsonl@warn").min_level == "warn"
    assert ev.from_spec("jsonl#agent.+http.").prefixes == ("agent.", "http.")
    with pytest.raises(ValueError):
        ev.from_spec("carrier-pigeon")


def test_a_bad_hook_spec_degrades_instead_of_refusing_to_start(bus):
    """One misspelled sink must not stop a service from starting."""
    hooks = ev.configure("counter,nonsense:whatever", bus=bus)
    assert len(hooks) == 1 and isinstance(hooks[0], ev.CounterHook)


def test_configuring_the_same_hook_twice_installs_it_once(bus):
    """A CLI and a server that both read the environment must not double every line."""
    first = ev.configure("counter,ring:8", bus=bus)
    again = ev.configure("counter,ring:8", bus=bus)
    assert len(first) == 2 and again == []
    assert len(bus.hooks) == 2
    assert len(ev.configure("counter", bus=bus, force=True)) == 1


def test_level_and_prefix_filters_select(bus):
    warns, agents = [], []
    bus.subscribe(ev.CallableHook(warns.append, name="w", min_level="warn"))
    bus.subscribe(ev.CallableHook(agents.append, name="a", prefixes=("agent.",)))
    bus.emit("agent.observe", level="info")
    bus.emit("http.request", level="error")
    assert [e.kind for e in warns] == ["http.request"]
    assert [e.kind for e in agents] == ["agent.observe"]


def test_a_plugin_hook_is_resolved_by_import(bus, monkeypatch):
    """The extension point with no registry: `module:attr`, imported."""
    import types
    got = []
    mod = types.ModuleType("sk_test_plugin")
    mod.sink = got.append
    monkeypatch.setitem(__import__("sys").modules, "sk_test_plugin", mod)
    bus.subscribe(ev.from_spec("plugin:sk_test_plugin:sink"))
    bus.emit("test.plugin", value=1)
    assert [e.data["value"] for e in got] == [1]


def test_the_ring_serves_recent_events_and_forgets_the_rest(bus):
    ring = bus.subscribe(ev.RingHook(capacity=5))
    for i in range(20):
        bus.emit("test.ring", i=i)
    recent = ring.recent(n=10)
    assert len(recent) == 5 and [e["data"]["i"] for e in recent] == [15, 16, 17, 18, 19]
    assert ring.recent(n=10, kind="nothing.") == []


def test_a_payload_field_named_kind_does_not_break_the_call(bus):
    """A logging call must never raise on the shape of its own payload.

    `kind` is an ordinary word — a workload has one, a pipeline stage reports one — and before
    `kind` was made positional-only every stage of the pipeline raised `TypeError` the moment it
    tried to log itself.
    """
    got = []
    bus.subscribe(ev.CallableHook(got.append))
    bus.emit("pipeline.stage.ok", kind="agent", level="info", stage="resolve")
    assert got[-1].kind == "pipeline.stage.ok"
    assert got[-1].data["kind"] == "agent" and got[-1].level == "info"


def test_events_serialize_to_one_json_object_per_line(bus):
    buf = io.StringIO()
    bus.subscribe(ev.JsonlHook(buf))
    bus.emit("test.line", a=1, b="two")
    (line,) = buf.getvalue().strip().splitlines()
    d = json.loads(line)
    assert d["kind"] == "test.line" and d["data"] == dict(a=1, b="two")
    assert d["seq"] >= 1 and d["iso"].endswith("Z")
