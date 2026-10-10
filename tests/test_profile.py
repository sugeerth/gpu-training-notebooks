"""The profile: costing a workload at volume, and attributing what the levers saved.

The numbers here turn into a procurement decision, so what is worth pinning is not any
particular dollar figure — those move with a price list — but the handful of claims the report
makes *about* its own arithmetic. Each test below corresponds to a way the first version of this
module was wrong.
"""
from __future__ import annotations

import json
import math

import pytest

import servingkit as sk
from servingkit import profile as pf
from servingkit.stack import Stack

AGENT = dict(turns=30, tool_result_tokens=2000, tool_latency_s=3.0, fanout=1, batch=32,
             replayed=True)


@pytest.fixture(scope="module")
def prof():
    return pf.profile(sk.Workload.agent(**AGENT), runs_per_month=35_000, gpu="H100 SXM",
                      optimize="api")


# ------------------------------------------------------------------- the batch that fits
def test_the_batch_is_clamped_to_the_seats_that_exist(prof):
    """Pricing a batch the KV pool cannot hold prices a deployment that OOMs."""
    b = prof.baseline
    assert b["seats"] < b["requested_batch"], "this workload is meant to be memory-limited"
    assert b["batch"] <= math.floor(b["seats"]) + 1e-9
    assert b["clamped"] is True
    assert any("does not fit" in n for n in prof.notes)


def test_the_clamp_re_costs_rather_than_dividing(prof):
    """The step is re-evaluated at the feasible batch, so throughput is not simply scaled."""
    w = sk.Workload.agent(**AGENT)
    asked = Stack(w, "H100 SXM").plan()
    got = prof.baseline
    # A step at batch 6 is not a step at batch 32 divided by five: the weight read is amortized
    # over fewer sequences, so per-token latency goes up rather than staying put.
    assert got["batch"] < asked["batch"]
    assert got["tpot_ms"] != pytest.approx(asked["tpot_ms"], rel=1e-6)


# ----------------------------------------------------------- what a caller waits for
def test_releasing_a_slot_does_not_shorten_the_tool_call(prof):
    """`idle_factor` frees the slot; it does not make somebody else's API return faster.

    Reporting the run's wall clock with the tool wait already discounted made agent-aware
    eviction look like a latency win, which it is not. It is a cost win.
    """
    b, t = prof.baseline, prof.tuned
    assert t["tool_s"] == pytest.approx(b["tool_s"]), "the tool wait is not a lever's to shrink"
    assert t["tool_s"] == pytest.approx(30 * 3.0)
    assert t["slot_s"] <= t["wall_s"] + 1e-9, "a slot cannot be held longer than the run"
    assert t["wall_s"] == pytest.approx(t["prefill_s"] + t["decode_s"] + t["tool_s"])


def test_the_fleet_is_sized_on_held_seconds_not_on_the_callers_wait(prof):
    t = prof.tuned
    expected = t["batch"] * 3600.0 / t["slot_s"]
    assert t["per_replica_per_hour"] == pytest.approx(expected)


# ------------------------------------------------------------------------ the two bills
def test_the_two_bills_are_never_added(prof):
    """Token price and GPU rent are alternatives. Their sum answers nobody's question."""
    d = prof.as_dict()
    blob = json.dumps(d)
    assert "total_cost" not in blob, "the profile must not carry the summed figure"
    for side in (d["baseline"], d["tuned"]):
        assert side["api_month"] > 0 and side["host_month"] > 0
        assert "combined" not in side


def test_the_host_bill_is_whole_cards(prof):
    for side in (prof.baseline, prof.tuned):
        assert side["gpus"] == math.ceil(side["exact_replicas"]) * side["gpus_per_replica"]
        assert side["host_month"] == pytest.approx(
            side["gpus"] * pf.HOURS_PER_MONTH * side["gpu_hour_price"])
        assert side["exact_gpus"] <= side["gpus"] + 1e-9


# --------------------------------------------------------------------------- the chooser
@pytest.mark.parametrize("bill", ["api", "host"])
def test_the_chooser_never_keeps_a_lever_that_raises_the_bill(bill):
    w = sk.Workload.agent(**AGENT)
    chosen = pf.select_for_bill(w, "H100 SXM", 35_000, bill=bill)
    base = pf.cost_at_volume(Stack(w, "H100 SXM"), 35_000)
    got = pf.cost_at_volume(chosen, 35_000)
    key = "api_month" if bill == "api" else "host_month"
    assert got[key] <= base[key] + 1e-9
    other = "host_month" if bill == "api" else "api_month"
    assert got[other] <= base[other] + 1e-9, "a tie-break must not spend the other bill"


def test_the_chooser_drops_a_lever_that_widens_the_replica():
    """`TP=2` lowers per-run cost and doubles the cards. Every per-run chooser pulls it.

    This is the whole reason the module exists: the control loop minimizing `total_cost_usd`
    picks TP=2 because a second card halves the slot-seconds a run holds, and the invoice then
    arrives for two whole cards serving a fifth of a card's traffic.
    """
    from servingkit.agent import Objective
    w = sk.Workload.agent(**AGENT)
    per_run = pf.profile(w, runs_per_month=35_000, gpu="H100 SXM", sensitivity=False,
                         objective=Objective(max_tpot_ms=40, minimize="total_cost_usd",
                                             allow_gpu_moves=False))
    assert "TP=2" in per_run.levers
    assert per_run.tuned["host_month"] > per_run.baseline["host_month"]

    by_bill = pf.profile(w, runs_per_month=35_000, gpu="H100 SXM", sensitivity=False,
                         optimize="host")
    assert "TP=2" not in by_bill.levers
    assert by_bill.tuned["host_month"] <= by_bill.baseline["host_month"]


# ------------------------------------------------------------------------ attribution
def test_marginal_attribution_telescopes_to_the_total(prof):
    """Each row is what was left to take, so the rows must sum to what was taken."""
    total = sum(a["api_saved"] for a in prof.attribution)
    assert total == pytest.approx(prof.api_saved_per_month, rel=1e-9)


def test_solo_and_marginal_are_both_reported(prof):
    """A lever's solo saving can exceed its marginal one; hiding either misleads."""
    assert prof.attribution
    for a in prof.attribution:
        assert "solo_api_saved" in a and "api_saved" in a
        assert a["solo_api_saved"] >= a["api_saved"] - 1e-6


# ------------------------------------------------------------------------ sensitivity
def test_the_hit_rate_sweep_actually_sweeps(prof):
    """The first version printed six identical rows: the lever's own floor overrode the sweep."""
    rows = prof.sensitivity
    assert len(rows) >= 5
    tokens = [r["prefill_tokens"] for r in rows]
    bills = [r["api_month"] for r in rows]
    assert len(set(tokens)) == len(tokens), f"the sweep is flat: {tokens}"
    assert tokens == sorted(tokens, reverse=True), "more cache hits cannot mean more prefill"
    assert bills == sorted(bills, reverse=True)
    # And the headline number is the one the lever assumes, not the best case.
    at95 = next(r for r in rows if r["hit_rate"] == 0.95)
    assert at95["api_month"] == pytest.approx(prof.tuned["api_month"], rel=1e-6)


# ------------------------------------------------------------------------------ shape
def test_the_profile_is_json_and_finite(prof):
    blob = json.dumps(prof.as_dict(), allow_nan=False)      # raises on Infinity or NaN
    back = json.loads(blob)
    assert back["saved"]["api_per_month"] == pytest.approx(prof.api_saved_per_month)
    assert 0.0 <= back["saved"]["api_fraction"] <= 1.0


def test_the_report_states_what_it_is(prof):
    text = pf.profile_report(prof)
    assert "planning model" in text, "a cost report must say it is not a measurement"
    assert "alternatives, not addends" in text
    assert "kernels/16_prefix_match.cu" in text


def test_volume_scales_the_api_bill_linearly_and_the_fleet_in_steps():
    w = sk.Workload.agent(**AGENT)
    small = pf.profile(w, runs_per_month=10_000, gpu="H100 SXM", optimize="api",
                       sensitivity=False)
    big = pf.profile(w, runs_per_month=100_000, gpu="H100 SXM", optimize="api",
                     sensitivity=False)
    assert big.tuned["api_month"] == pytest.approx(small.tuned["api_month"] * 10, rel=1e-6)
    # Hardware does not: it is bought in whole cards, so it climbs in steps and never falls.
    assert big.tuned["gpus"] >= small.tuned["gpus"]
    assert float(big.tuned["gpus"]).is_integer()
