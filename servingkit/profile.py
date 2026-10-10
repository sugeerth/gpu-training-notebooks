"""What one workload costs, where the cost goes, and what the plan actually saves.

`plan()` and `ladder()` cost a single run. Nobody buys a single run. This converts a plan into
the two numbers a decision gets made on — **money per month** and **accelerators needed** — and
then says where that money goes and which lever removed which part of it.

    python -m servingkit profile --agent --turns 30 --tool-tokens 2000 \\
        --runs-per-month 35000 --max-tpot-ms 40

Three things it does that the ladder cannot:

**It separates the two ways you pay.** `plan()` reports `total_cost_usd` as token cost plus GPU
cost, and those are alternatives rather than addends: buy the tokens from a vendor and you do not
rent the GPU; rent the GPU and you do not pay per token. Summing them answers no one's question.
So the profile carries two columns — *as an API bill* and *self-hosted* — and the saving is
different in each, because the levers that save tokens are not the levers that save GPU-hours.
Prefix caching cuts the API bill by four fifths and barely moves the hardware. Everything that
shrinks the step cuts the hardware and does nothing to the token count.

**It uses the batch that fits.** A plan can cost a batch of 32 while reporting that 6 sequences
fit in the KV pool, because the step model is handed a batch and the capacity is computed
separately. At fleet scale that gap is the whole answer: you cannot run the batch you priced, so
the throughput you priced is not available, so the fleet is five times the size. The profile
clamps the batch to the seats that exist and says so — and that makes the memory levers legible,
because shrinking the cache is what buys the batch back.

**It attributes.** Each lever gets the money it removed, in the order the ladder applies them,
alongside what it would have saved alone. Those two numbers differ whenever levers overlap, and
the gap is the honest health warning on any "this lever saves $X" claim.

Everything here is arithmetic over the planning model. The catalogs are spec sheets, `BW_EFF`
and `FLOP_EFF` are what a good kernel reaches rather than what yours does, and `recalibrate()` is
where your own measurements go. The one thing on this page that was measured is which kernels
compile and agree with their own double-precision reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .catalog import GPUS
from .events import BUS
from .levers import LEVERS
from .stack import Stack, _default_order, recommend
from .workload import Workload

# Hours a GPU is rented per month, if you rent it continuously. 730 is the usual convention
# (8760/12) and it is what every cloud price list is quoted against.
HOURS_PER_MONTH = 730.0


@dataclass
class Profile:
    """One workload, costed at volume, baseline against tuned."""

    workload: str
    gpu: str
    runs_per_month: float
    levers: list[str] = field(default_factory=list)
    baseline: dict = field(default_factory=dict)
    tuned: dict = field(default_factory=dict)
    attribution: list[dict] = field(default_factory=list)
    kernels: list[dict] = field(default_factory=list)
    sensitivity: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ savings
    @property
    def api_saved_per_month(self) -> float:
        return self.baseline["api_month"] - self.tuned["api_month"]

    @property
    def host_saved_per_month(self) -> float:
        return self.baseline["host_month"] - self.tuned["host_month"]

    def as_dict(self) -> dict:
        from .api import _jsonable
        return _jsonable(dict(
            workload=self.workload, gpu=self.gpu, runs_per_month=self.runs_per_month,
            levers=self.levers, baseline=self.baseline, tuned=self.tuned,
            attribution=self.attribution, kernels=self.kernels,
            sensitivity=self.sensitivity, notes=self.notes,
            saved=dict(api_per_month=self.api_saved_per_month,
                       host_per_month=self.host_saved_per_month,
                       api_fraction=_frac(self.baseline["api_month"], self.tuned["api_month"]),
                       host_fraction=_frac(self.baseline["host_month"],
                                           self.tuned["host_month"]))))


def _frac(before: float, after: float) -> float:
    return (before - after) / before if before else 0.0


# --------------------------------------------------------------------------------- costing
def cost_at_volume(stack: Stack, runs_per_month: float) -> dict:
    """Cost one configuration at a monthly volume, in both payment models.

    The fleet arithmetic is the part worth reading. A replica serves `batch` runs at once and
    each run holds its slot for `wall_s`, so a replica retires `batch * 3600 / wall_s` runs an
    hour; the fleet is the arrival rate divided by that, rounded up, times the accelerators a
    replica needs. Rounding up matters: a fleet is bought in whole cards, and the difference
    between 2.1 and 3 is a third of the bill.
    """
    p = stack.plan()
    w = stack.workload

    # The batch you can actually run, not the batch you asked for. `concurrent` is how many of
    # these sequences fit in the KV pool once the weights are resident; pricing a batch larger
    # than that prices a deployment that OOMs on its first full minute.
    seats = p["concurrent"]
    requested = int(p["batch"])
    feasible = requested if seats >= requested else int(math.floor(seats))
    clamped = feasible < requested

    if feasible >= 1 and clamped:
        # Re-cost at the batch that fits. Step time changes with the batch, so the throughput,
        # the wall clock and the money all move; taking the old numbers and dividing would be a
        # different and wrong answer.
        from dataclasses import replace
        s2 = Stack(replace(w, batch=feasible), stack.gpu).with_levers(*stack.names)
        p = s2.plan()

    # Two different seconds, and conflating them flatters the wrong levers. `plan()` scales the
    # tool wait by `idle_factor`, which models the *slot* being released while a tool runs — not
    # the tool returning sooner. The person waiting still waits the whole 3 seconds; the fleet
    # only pays for the part it holds. So the run breakdown reports the wait in full and the
    # fleet arithmetic uses the held part, and agent-aware eviction stops looking like it makes
    # an agent faster when what it does is make it cheaper.
    raw_tool_s = (w.turns * w.tool_latency_s) if w.kind == "agent" else 0.0
    slot_s = p["wall_s"]                       # prefill + decode + the tool time still held
    wall_s = p["prefill_s"] + p["decode_s"] + raw_tool_s
    eff_batch = max(int(p["batch"]), 1)
    gpus_per_replica = p["gpus"]

    runs_per_hour = runs_per_month / HOURS_PER_MONTH
    per_replica_per_hour = eff_batch * 3600.0 / slot_s if slot_s else float("inf")
    exact_replicas = runs_per_hour / per_replica_per_hour if per_replica_per_hour else 0.0
    replicas = math.ceil(exact_replicas)
    gpus = replicas * gpus_per_replica
    # The fractional fleet, kept beside the rounded one. You buy whole cards, so the bill uses
    # the ceiling — but at a small fleet the ceiling quantizes every lever's effect to zero, and
    # a report where eight levers all saved exactly nothing is a report nobody can act on.
    exact_gpus = exact_replicas * gpus_per_replica
    gpu_hour_price = GPUS[stack.gpu]["usd"]

    # Two bills, never added together. See the module docstring.
    api_month = (p["prefill_tokens"] / 1e6 * w.price_in_per_mtok
                 + p["decode_tokens"] / 1e6 * w.price_out_per_mtok) * runs_per_month
    host_month = gpus * HOURS_PER_MONTH * gpu_hour_price

    return dict(
        levers=list(stack.names),
        step_ms=p["step_ms"], tpot_ms=p["tpot_ms"], tokens_per_s=p["tokens_per_s"],
        parts_ms={k: v * 1000 for k, v in p["parts"].items()},
        wall_s=wall_s, slot_s=slot_s, prefill_s=p["prefill_s"], decode_s=p["decode_s"],
        tool_s=raw_tool_s, tool_held_s=p["tool_s"], slot_util=p["slot_util"],
        prefill_tokens=p["prefill_tokens"], decode_tokens=p["decode_tokens"],
        kv_gb=p["kv_gb"], seats=seats, requested_batch=requested, batch=eff_batch,
        clamped=clamped,
        api_run=p["token_cost_usd"], api_month=api_month,
        gpus_per_replica=gpus_per_replica, replicas=replicas, gpus=gpus,
        exact_replicas=exact_replicas, exact_gpus=exact_gpus,
        exact_host_month=exact_gpus * HOURS_PER_MONTH * gpu_hour_price,
        gpu_hours_month=gpus * HOURS_PER_MONTH, gpu_hour_price=gpu_hour_price,
        host_month=host_month,
        host_per_run=host_month / runs_per_month if runs_per_month else 0.0,
        runs_per_hour=runs_per_hour, per_replica_per_hour=per_replica_per_hour,
    )


def select_for_bill(workload: Workload, gpu: str, runs_per_month: float,
                    bill: str = "api") -> Stack:
    """Keep only the levers that lower the bill you actually receive.

    Every other chooser here optimizes a **per-run** cost, and that is why they all pull `TP=2`:
    a second card halves the step, which halves the slot-seconds a run holds, which lowers the
    per-run GPU cost. The invoice disagrees. A fleet is bought in whole cards, so a replica that
    needs a fifth of a card still bills for the two that `TP=2` requires, and the per-run
    arithmetic cannot see that because the ceiling happens at the fleet, not at the run.

    So this one walks the same ladder and asks the only question the finance team asks: did the
    monthly number go down. Ties break on the *other* bill first and on latency last — a lever
    that is free on the bill you chose is not free if it puts a second card on the bill you
    didn't. Breaking straight to latency pulled `TP=2` back in on the API column, where it
    changes no token and buys an accelerator.
    """
    def key(st: Stack) -> tuple[float, float, float]:
        c = cost_at_volume(st, runs_per_month)
        chosen, other = ((c["api_month"], c["host_month"]) if bill == "api"
                         else (c["host_month"], c["api_month"]))
        return (chosen, other, c["tpot_ms"])

    props = workload.properties
    order = _default_order([n for n, lv in LEVERS.items()
                            if "accuracy" not in lv.spends and lv.applies_to(props)])
    held = Stack(workload, gpu)
    best = key(held)
    for n in order:
        trial = Stack(workload, gpu).with_levers(*held.names, n)
        if n not in trial.names:
            continue
        k = key(trial)
        if k < best:
            held, best = trial, k
    return held


# -------------------------------------------------------------------------------- profile
def profile(workload: Workload, *, runs_per_month: float = 10_000.0, gpu: str | None = None,
            levers: list[str] | None = None, objective=None, max_steps: int = 12,
            sensitivity: bool = True, optimize: str | None = None) -> Profile:
    """Cost a workload at volume, before and after, and attribute the difference.

    With an `objective` the control loop chooses the levers and the profile reports what the
    loop arrived at; without one it takes `recommend()`, or exactly the levers you name.
    """
    gpu = gpu or workload.gpu
    base = Stack(workload, gpu)

    if levers is not None:
        tuned_stack = base.with_levers(*levers)
        chosen_by = "the levers you named"
    elif optimize in ("api", "host"):
        tuned_stack = select_for_bill(workload, gpu, runs_per_month, bill=optimize)
        chosen_by = (f"the monthly {'API' if optimize == 'api' else 'hardware'} bill "
                     f"at {runs_per_month:,.0f} runs/month")
    elif objective is not None:
        from .agent import ControlLoop
        loop = ControlLoop(workload, objective, gpu=gpu)
        trace = loop.run(max_steps=max_steps)
        tuned_stack, gpu = loop.stack, loop.gpu
        chosen_by = f"the control loop ({trace.outcome}, {len(trace.accepted)} actions kept)"
    else:
        tuned_stack = recommend(workload, gpu)
        chosen_by = "recommend()"

    with BUS.run(prefix="profile"):
        BUS.emit("profile.start", workload=workload.kind, model=workload.model, gpu=gpu,
                 runs_per_month=runs_per_month, chosen_by=chosen_by)
        before = cost_at_volume(Stack(workload, gpu), runs_per_month)
        after = cost_at_volume(tuned_stack, runs_per_month)

        prof = Profile(workload=workload.describe().replace("\n", " "), gpu=gpu,
                       runs_per_month=runs_per_month, levers=list(tuned_stack.names),
                       baseline=before, tuned=after)
        prof.attribution = _attribute(workload, gpu, tuned_stack.names, runs_per_month, before)
        prof.kernels = [dict(lever=n, kernel=LEVERS[n].kernel) for n in tuned_stack.names]
        prof.notes = _notes(before, after, workload, chosen_by)
        if sensitivity:
            prof.sensitivity = _sensitivity(workload, gpu, tuned_stack.names, runs_per_month)
        BUS.emit("profile.done", levers=prof.levers,
                 api_saved=prof.api_saved_per_month, host_saved=prof.host_saved_per_month,
                 gpus_before=before["gpus"], gpus_after=after["gpus"])
    return prof


def _attribute(w: Workload, gpu: str, names: list[str], runs: float, before: dict) -> list[dict]:
    """What each lever removed, in the order a team would reach for them — and alone.

    Marginal attribution depends on the order, which is why the solo column is here beside it.
    A lever with a large solo saving and a small marginal one is not weak; it is late, and
    something earlier already took what it was going to take. Reporting only the marginal number
    makes the fourth lever look worthless, and reporting only the solo number makes any four of
    them look like they add up to more than the stack delivers.
    """
    out, prev, held = [], before, []
    for n in _default_order(names):
        s = Stack(w, gpu).with_levers(*held, n)
        if n not in s.names:
            continue
        held = list(s.names)
        now = cost_at_volume(s, runs)
        solo = cost_at_volume(Stack(w, gpu).with_levers(n), runs)
        out.append(dict(
            lever=n, kernel=LEVERS[n].kernel,
            api_saved=prev["api_month"] - now["api_month"],
            host_saved=prev["exact_host_month"] - now["exact_host_month"],
            gpus_removed=prev["exact_gpus"] - now["exact_gpus"],
            solo_api_saved=before["api_month"] - solo["api_month"],
            solo_host_saved=before["exact_host_month"] - solo["exact_host_month"],
            wall_s=now["wall_s"], batch=now["batch"], gpus=now["gpus"],
        ))
        prev = now
    return out


def _sensitivity(w: Workload, gpu: str, names: list[str], runs: float) -> list[dict]:
    """The prefix hit rate, swept — because the biggest lever rests entirely on it.

    Prefix caching is worth more than everything else in an agent profile put together, and what
    it is worth is linear in a number nobody measures before deploying. A miss is a miss on the
    *whole* context rather than on the new tokens, so 99% and 90% are not nearly the same number.
    This is the one assumption worth seeing moved.
    """
    from dataclasses import replace
    # Sweeping the workload's hit rate with `prefix caching` pulled changes nothing, because the
    # lever's own `prefill_factor = 0.0` means "only the tokens never seen before" and `plan()`
    # floors that at a 0.95 hit rate. The first version of this sweep did exactly that and
    # printed six identical rows. The question worth asking is what a cache that achieves *h*
    # costs, so the sweep drops the lever and sets the rate directly — and the lever's own row
    # is the 95% one.
    others = [n for n in names if n != "prefix caching"]
    rows = []
    for hit in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        w2 = replace(w, prefix_hit_rate=hit)
        c = cost_at_volume(Stack(w2, gpu).with_levers(*others), runs)
        rows.append(dict(hit_rate=hit, api_month=c["api_month"], host_month=c["host_month"],
                         gpus=c["gpus"], exact_gpus=c["exact_gpus"],
                         prefill_tokens=c["prefill_tokens"]))
    return rows


def _notes(before: dict, after: dict, w: Workload, chosen_by: str) -> list[str]:
    notes = [f"levers chosen by {chosen_by}"]
    if before["clamped"] or after["clamped"]:
        notes.append(
            f"the KV pool holds {before['seats']:.1f} of these sequences at "
            f"{w.ctx:,} tokens, so the batch was clamped from {before['requested_batch']} to "
            f"{before['batch']} at baseline and {after['batch']} tuned — the plan would "
            f"otherwise have priced a batch that does not fit")
    if after["tool_s"] > 0.35 * after["wall_s"]:
        notes.append(
            f"{after['tool_s'] / after['wall_s'] * 100:.0f}% of the tuned wall clock is waiting "
            f"on tools, and a held slot costs the same whether it is computing or waiting")
    if after["host_month"] > before["host_month"] + 1e-9:
        # The comparison that matters is the one on the invoice. A stack can need *less* card
        # than the baseline and still cost more of it, because a fleet is bought whole: a
        # two-card replica serving a fifth of a card's worth of traffic bills for two cards.
        notes.append(
            f"the tuned stack bills for MORE hardware than the baseline "
            f"({before['gpus']:.0f} cards -> {after['gpus']:.0f}, "
            f"{money(before['host_month'])} -> {money(after['host_month'])}) even though it "
            f"needs less of one ({before['exact_gpus']:.2f} -> {after['exact_gpus']:.2f}): at "
            f"this volume a lever that widens a replica is paid for in whole cards")
    if after["gpus"] == before["gpus"] and after["gpus"]:
        notes.append(
            "the fleet did not shrink: the levers bought latency and tokens, not hardware, "
            "at this volume")
    return notes


# ---------------------------------------------------------------------------------- report
def profile_report(p: Profile) -> str:
    b, t = p.baseline, p.tuned
    L = []
    L.append(p.workload)
    L.append(f"{p.runs_per_month:,.0f} runs per month on {p.gpu}")
    L.append("")

    L.append("WHERE ONE STEP GOES                      baseline         tuned")
    L.append("-" * 66)
    keys = sorted(set(b["parts_ms"]) | set(t["parts_ms"]),
                  key=lambda k: -t["parts_ms"].get(k, 0))
    for k in keys:
        bb, tt = b["parts_ms"].get(k, 0.0), t["parts_ms"].get(k, 0.0)
        if bb < 1e-4 and tt < 1e-4:
            continue
        L.append(f"  {k:<36}{bb:>10.3f}ms{tt:>12.3f}ms")
    L.append(f"  {'step':<36}{b['step_ms']:>10.3f}ms{t['step_ms']:>12.3f}ms")
    L.append(f"  {'per output token':<36}{b['tpot_ms']:>10.1f}ms{t['tpot_ms']:>12.1f}ms")
    L.append("")

    L.append("WHERE ONE RUN GOES                       baseline         tuned")
    L.append("-" * 66)
    for label, key in (("prefill", "prefill_s"), ("decode", "decode_s"),
                       ("waiting on tools", "tool_s")):
        L.append(f"  {label:<36}{b[key]:>11.1f}s{tt_fmt(t[key]):>13}")
    L.append(f"  {'wall clock, what a caller waits':<36}{b['wall_s']:>11.1f}s"
             f"{tt_fmt(t['wall_s']):>13}")
    L.append(f"  {'slot seconds, what the fleet pays':<36}{b['slot_s']:>11.1f}s"
             f"{tt_fmt(t['slot_s']):>13}")
    L.append(f"  {'prefill tokens':<36}{b['prefill_tokens']:>12,.0f}{t['prefill_tokens']:>13,.0f}")
    L.append(f"  {'batch that fits':<36}{b['batch']:>12,}{t['batch']:>13,}"
             + (f"   (of {b['requested_batch']} asked for)" if b["clamped"] else ""))
    L.append("")

    L.append("WHAT IT COSTS A MONTH                    baseline         tuned         saved")
    L.append("-" * 80)
    L.append(f"  {'as an API bill':<36}{money(b['api_month']):>12}{money(t['api_month']):>14}"
             f"{money(p.api_saved_per_month):>14}"
             f"   {_frac(b['api_month'], t['api_month']) * 100:>5.0f}%")
    L.append(f"  {'self-hosted: cards needed':<36}{b['exact_gpus']:>12,.2f}"
             f"{t['exact_gpus']:>14,.2f}{b['exact_gpus'] - t['exact_gpus']:>14,.2f}")
    L.append(f"  {'self-hosted: cards bought (whole)':<36}{b['gpus']:>12,.0f}"
             f"{t['gpus']:>14,.0f}{b['gpus'] - t['gpus']:>+14,.0f}")
    L.append(f"  {'self-hosted: rent':<36}{money(b['host_month']):>12}"
             f"{money(t['host_month']):>14}{money(p.host_saved_per_month):>14}"
             f"   {_frac(b['host_month'], t['host_month']) * 100:>5.0f}%")
    L.append("")
    L.append("  The two rows are alternatives, not addends: you buy the tokens or you rent the")
    L.append("  cards. Which saving is yours depends on which bill you actually receive.")
    L.append("")

    if p.attribution:
        L.append("WHAT EACH LEVER REMOVED                  API/mo      host/mo    GPUs"
                 "    alone (API)")
        L.append("-" * 80)
        for a in p.attribution:
            L.append(f"  {a['lever']:<36}{money(a['api_saved']):>10}"
                     f"{money(a['host_saved']):>13}{a['gpus_removed']:>8,.0f}"
                     f"{money(a['solo_api_saved']):>15}")
        L.append("")
        L.append("  In ladder order, so each row is what was left to take. The last column is")
        L.append("  what the lever saves on its own — a gap between the two is overlap, not")
        L.append("  weakness.")
        L.append("")

    if p.sensitivity:
        L.append("IF THE PREFIX CACHE MISSES MORE OFTEN")
        L.append("-" * 66)
        L.append(f"  {'hit rate':<14}{'prefill tokens':>18}{'API/mo':>14}{'GPUs':>8}")
        for s in p.sensitivity:
            mark = "  <- what the lever assumes" if abs(s["hit_rate"] - 0.95) < 1e-9 else ""
            L.append(f"  {s['hit_rate'] * 100:>6.0f}%       {s['prefill_tokens']:>18,.0f}"
                     f"{money(s['api_month']):>14}{s['exact_gpus']:>8,.2f}{mark}")
        L.append("")

    if p.kernels:
        named = [k for k in p.kernels if k["kernel"]]
        L.append("THE CODE BEHIND IT")
        L.append("-" * 66)
        for k in named:
            L.append(f"  {k['lever']:<36}kernels/{k['kernel']}")
        decisions = [k["lever"] for k in p.kernels if not k["kernel"]]
        if decisions:
            L.append(f"  {'(no kernel — a scheduling decision)':<36}{', '.join(decisions)}")
        L.append("")

    if p.notes:
        L.append("WORTH KNOWING")
        L.append("-" * 66)
        for n in p.notes:
            L.append(f"  - {n}")
        L.append("")

    L.append("Every number above is a planning model over vendor specifications and list")
    L.append("prices. None of it is a measurement of your deployment.")
    return "\n".join(L)


def tt_fmt(x: float) -> str:
    return f"{x:,.1f}s"


def money(x: float) -> str:
    if abs(x) >= 100_000:
        return f"${x / 1000:,.0f}k"
    if abs(x) >= 100:
        return f"${x:,.0f}"
    return f"${x:,.2f}"


__all__ = ["profile", "Profile", "profile_report", "cost_at_volume", "HOURS_PER_MONTH"]
