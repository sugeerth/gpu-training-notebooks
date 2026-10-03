"""pytest wrapper around `python -m servingkit check`, plus the properties worth pinning.

The substantive check lives in `servingkit/selfcheck.py` because it is useful outside a test
runner — `python -m servingkit check` is something you run after editing a notebook. This file
makes it reachable from pytest and adds the invariants that are cheap to state and easy to break.
"""
from __future__ import annotations

import pytest

import servingkit as sk
from servingkit import selfcheck


def test_package_matches_notebooks():
    """~16,000 comparisons: every model against the notebook it was lifted from."""
    assert selfcheck.run_checks() == 0


def test_every_antagonism_is_declared():
    """A pair that measures sub-multiplicative must share a declared resource.

    This caught two wrong declarations the first time it ran: `int4 weights` and `fp8 weights`
    claimed to spend only `accuracy`, when shrinking the weight read is what removes the
    memory-bound-ness that batching and speculation are paid out of.
    """
    for w in (sk.Workload.chat(ctx=1024, batch=8),
              sk.Workload.chat(ctx=2048, batch=32),
              sk.Workload.chat(ctx=32768, batch=64)):
        for c in sk.Stack(w).all_applicable().interactions():
            if c["synergy"] < 0.9:
                assert c["predicted_conflict"], (
                    f"{c['a']} + {c['b']} measure {c['synergy']:.2f} with nothing declared")


def test_levers_declare_known_resources_and_properties():
    for name, lv in sk.LEVERS.items():
        for r in lv.spends:
            assert r in sk.RESOURCES, f"{name} spends unknown resource {r!r}"
        for p in lv.requires:
            assert p in sk.PROPERTIES, f"{name} requires unknown property {p!r}"


def test_a_stack_never_holds_two_levers_from_one_domain():
    w = sk.Workload.agent(turns=30, fanout=4, replayed=True)
    s = sk.Stack(w).with_levers(*sk.LEVERS)
    domains = [sk.LEVERS[n].domain for n in s.names]
    assert len(domains) == len(set(domains)), "a domain was taken twice"


def test_inapplicable_levers_are_refused_with_a_reason():
    chat = sk.Workload.chat(ctx=2048, batch=32)
    s = sk.Stack(chat).with_levers("cascade attention", "prefix caching")
    assert s.names == []
    assert {r.lever for r in s.rejected} == {"cascade attention", "prefix caching"}
    assert all(r.detail for r in s.rejected), "a refusal must name the missing property"


@pytest.mark.parametrize("lever", [n for n, lv in sk.LEVERS.items() if lv.kernel])
def test_every_named_kernel_exists(lever):
    root = sk.kernels.repo_root()
    assert (root / "kernels" / sk.LEVERS[lever].kernel).exists()


def test_batching_and_speculation_need_the_weight_read_to_dominate():
    """The precondition that most advice omits, pinned so it cannot be dropped again."""
    long_ctx = sk.Workload.agent(turns=30, tool_result_tokens=2000)
    assert long_ctx.bottleneck() != "GEMMs"
    for lv in ("2x batch", "speculation"):
        assert not sk.LEVERS[lv].applies_to(long_ctx.properties)
    short = sk.Workload.chat(ctx=1024, batch=8)
    assert short.bottleneck() == "GEMMs"
    for lv in ("2x batch", "speculation"):
        assert sk.LEVERS[lv].applies_to(short.properties)


def test_a_ladder_is_monotonically_reported_and_never_empty():
    w = sk.Workload.agent(turns=20, fanout=2, replayed=True)
    rows = sk.recommend(w).ladder()
    assert rows and rows[0]["label"] == "baseline"
    assert all(r["wall_s"] > 0 and r["cost_usd"] >= 0 for r in rows)
