"""A control loop that tunes a serving stack against an objective, and can be argued with.

Everything else in this package answers a question you asked. This asks its own: given a
workload, a service-level objective and a budget of measurements, it decides what to change,
predicts what the change will do, makes it, measures, and **keeps or reverts** on the result.

    loop = ControlLoop(Workload.agent(turns=30, fanout=4),
                       Objective(max_tpot_ms=25, minimize="total_cost_usd"))
    trace = loop.run(max_steps=10)
    print(trace.table())

Six phases per step, which is why this is a loop and not a sort:

    observe    measure the current stack: latency, wall clock, money, what binds
    propose    enumerate the legal actions — add, swap, drop, move card
    predict    say what each will do, from the levers' *declarations* and a solo measurement
    act        apply the best-predicted one
    verify     measure the composition, and compare against the prediction
    settle     keep it, or revert it and record why

The verify phase is the part that makes this worth building rather than a greedy loop over
`ladder()`. The prediction and the measurement are genuinely different computations:

  * the **prediction** is the lever's solo gain — measured once against the bare workload —
    composed multiplicatively, which is what "these are independent optimizations" means;
  * the **measurement** is the joint evaluation of the whole stack.

When they disagree, one of three things is true, and the loop distinguishes them:

  1. the levers declared a shared resource — the antagonism was predicted, and is priced in;
  2. they declared none — then a declaration is **wrong**, and the loop records a finding
     rather than quietly accepting a worse number. This is the same gate `pipeline.py` applies,
     except found by a loop that was trying to do something else;
  3. the action made the objective *worse* — it is reverted, and never proposed again.

Case 2 is the one that earns the design. A lever that lies about what it spends produces advice
that costs money, and the only way to catch it is to compare a composition against a prediction
made before the composition existed.

Every phase emits an event on the bus in `events.py`, so a run is observable from outside
without this module knowing anything about who is watching:

    {"kind":"agent.decide","data":{"step":3,"action":"add cascade attention",
     "predicted_tps":1492.4,"reason":"best predicted objective"}}
    {"kind":"agent.verify","data":{"step":3,"predicted":1492.4,"actual":1490.9,
     "verdict":"met","accepted":true}}

Measurements are counted and bounded. An agent that evaluates every subset is not an agent, it
is a brute-force search with better branding — so candidates are *ranked by prediction*, which
is cheap, and only the leader is measured, which is not. `Trace.evaluations` reports what the
search cost, and `Trace.prediction_accuracy` reports whether the ranking was worth trusting.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .catalog import GPUS
from .events import BUS, EventBus
from .levers import LEVERS, shared_resources
from .stack import Stack, recommend
from .step import base_config, evaluate
from .workload import Workload

# How far under its prediction an action may measure before the gap is treated as a finding
# rather than as noise. 0.9 is the same threshold `interactions()` uses, deliberately: one
# number, one meaning, in both places.
TOLERANCE = 0.9


# ------------------------------------------------------------------------------ objective
@dataclass(frozen=True)
class Objective:
    """What the loop is for. Constraints first, then the thing to improve.

    The ordering is lexicographic and it is the whole ethics of the loop: an action that
    improves cost while breaking the latency SLO is not an improvement, it is a different
    product. So violations are compared first, and the objective only breaks ties among
    configurations that satisfy the constraints — or, when none does, among the ones that
    violate them least.
    """

    max_tpot_ms: float | None = None      # per-token latency ceiling
    max_wall_s: float | None = None       # one run must finish inside this
    max_cost_usd: float | None = None     # money per run, tokens plus the slot
    min_tokens_per_s: float | None = None
    minimize: str = "total_cost_usd"      # any numeric key of `Stack.plan()`
    maximize: str | None = None           # or maximize one instead
    allow_accuracy_loss: bool = False     # may the loop spend the `accuracy` resource?
    allow_gpu_moves: bool = True          # may it change the card?
    # A deadband, and not an ornamental one. Without it the loop accepts any improvement its
    # arithmetic can represent, and the first run of this file duly pulled a lever for a gain of
    # 0.004% — a deploy, a review and a rollback risk in exchange for nothing. An action has to
    # be worth doing, not merely non-negative.
    min_improvement: float = 0.005        # 0.5% of the objective, relative

    def worth_it(self, before: dict, after: dict) -> bool:
        """Is this change big enough to be worth making? Constraints override the deadband."""
        vb, va = self.score(before)[0], self.score(after)[0]
        if va < vb - 1e-12:
            return True                   # it fixes an SLO breach: size does not matter
        if va > vb + 1e-12:
            return False
        b, a = self.value(before), self.value(after)
        return (b - a) > abs(b) * self.min_improvement

    def violations(self, plan: dict) -> list[dict]:
        """Every constraint this plan breaks, with how badly, as a ratio."""
        out = []
        for key, limit, worse in (("tpot_ms", self.max_tpot_ms, "above"),
                                  ("wall_s", self.max_wall_s, "above"),
                                  ("total_cost_usd", self.max_cost_usd, "above"),
                                  ("tokens_per_s", self.min_tokens_per_s, "below")):
            if limit is None:
                continue
            actual = float(plan[key])
            bad = actual > limit if worse == "above" else actual < limit
            if bad:
                over = (actual / limit - 1) if worse == "above" else (limit / actual - 1)
                out.append(dict(constraint=key, limit=limit, actual=actual,
                                over=over, direction=worse))
        return out

    def value(self, plan: dict) -> float:
        """The scalar being improved, always as a thing to make smaller."""
        if self.maximize:
            v = float(plan[self.maximize])
            return -v if v else float("inf")
        return float(plan[self.minimize])

    def score(self, plan: dict) -> tuple[float, float]:
        """`(total violation, objective)`. Compared as a tuple, so constraints dominate."""
        return (sum(v["over"] for v in self.violations(plan)), self.value(plan))

    def satisfied(self, plan: dict) -> bool:
        return not self.violations(plan)

    def describe(self) -> str:
        cons = [f"{k} <= {v}" for k, v in (("tpot_ms", self.max_tpot_ms),
                                           ("wall_s", self.max_wall_s),
                                           ("cost_usd", self.max_cost_usd)) if v is not None]
        if self.min_tokens_per_s:
            cons.append(f"tokens_per_s >= {self.min_tokens_per_s}")
        goal = f"maximize {self.maximize}" if self.maximize else f"minimize {self.minimize}"
        return f"{goal}" + (f" subject to {', '.join(cons)}" if cons else " (unconstrained)")


# --------------------------------------------------------------------------------- actions
@dataclass(frozen=True)
class Action:
    """One legal change to the stack. Four verbs, and nothing else is permitted.

    `drop` is the one people leave out, and it is the reason a loop beats a ladder: a lever that
    paid when it was added can stop paying once a later lever removes the bottleneck it was
    exploiting. Without a drop, the stack only ever accumulates.
    """

    verb: str                       # "add" | "swap" | "drop" | "move"
    lever: str | None = None
    out: str | None = None          # the lever a swap displaces
    gpu: str | None = None

    def label(self) -> str:
        if self.verb == "add":
            return f"add {self.lever}"
        if self.verb == "swap":
            return f"swap {self.lever} for {self.out}"
        if self.verb == "drop":
            return f"drop {self.lever}"
        return f"move to {self.gpu}"

    def key(self) -> tuple:
        return (self.verb, self.lever, self.out, self.gpu)


@dataclass
class Decision:
    """One step of the loop, with the prediction that justified it and the verdict on it."""

    step: int
    action: str
    verb: str
    reason: str
    predicted: dict = field(default_factory=dict)
    actual: dict = field(default_factory=dict)
    verdict: str = ""               # met | mispredicted X | short-* | not worth it | worse
    accuracy: float = 0.0           # of the predicted improvement, the fraction that arrived
    accepted: bool = False
    finding: str | None = None
    considered: int = 0
    seconds: float = 0.0
    levers_after: list[str] = field(default_factory=list)
    gpu_after: str | None = None


@dataclass
class Trace:
    """The whole run: what it decided, what it cost, and how good its predictions were."""

    run_id: str
    objective: str
    workload: str
    start_gpu: str
    start_levers: list[str] = field(default_factory=list)
    start_plan: dict = field(default_factory=dict)
    decisions: list[Decision] = field(default_factory=list)
    final_levers: list[str] = field(default_factory=list)
    final_gpu: str = ""
    final_plan: dict = field(default_factory=dict)
    outcome: str = ""               # converged | satisfied | budget | infeasible
    violations: list[dict] = field(default_factory=list)
    engineer: list[dict] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    evaluations: int = 0
    seconds: float = 0.0

    # ------------------------------------------------------------------ reporting
    @property
    def accepted(self) -> list[Decision]:
        return [d for d in self.decisions if d.accepted]

    @property
    def prediction_accuracy(self) -> float | None:
        """Median share of the predicted improvement that arrived, on the objective's own axis.

        Reported rather than asserted. A loop whose predictions are systematically optimistic
        still converges — it just wastes measurements on candidates it ranked too highly, and
        this is the number that says so. Measuring it on the objective rather than on throughput
        is the point: a cost run whose throughput predictions are perfect and whose cost
        predictions are 40% out should not report 1.00.
        """
        rs = sorted(d.accuracy for d in self.decisions if d.accuracy)
        return rs[len(rs) // 2] if rs else None

    def as_dict(self) -> dict:
        from .api import _jsonable
        return _jsonable(dict(
            run_id=self.run_id, objective=self.objective, workload=self.workload,
            outcome=self.outcome, seconds=round(self.seconds, 3),
            evaluations=self.evaluations,
            prediction_accuracy=self.prediction_accuracy,
            start=dict(gpu=self.start_gpu, levers=self.start_levers, plan=self.start_plan),
            final=dict(gpu=self.final_gpu, levers=self.final_levers, plan=self.final_plan),
            violations=self.violations, engineer=self.engineer, findings=self.findings,
            decisions=[d.__dict__ for d in self.decisions],
        ))

    def table(self) -> str:
        head = (f"{'#':>2}  {'action':<34}{'verdict':<22}{'tok/s':>9}{'$':>9}"
                f"{'tpot':>8}{'acc':>7}  kept")
        lines = [self.objective, "", head, "-" * len(head)]
        p0 = self.start_plan
        lines.append(f"{'0':>2}  {'baseline':<34}{'':<22}{p0.get('tokens_per_s', 0):>9,.0f}"
                     f"{p0.get('total_cost_usd', 0):>9.4f}{p0.get('tpot_ms', 0):>8.1f}")
        for d in self.decisions:
            a = d.actual
            lines.append(f"{d.step:>2}  {d.action:<34}{d.verdict:<22}"
                         f"{a.get('tokens_per_s', 0):>9,.0f}{a.get('total_cost_usd', 0):>9.4f}"
                         f"{a.get('tpot_ms', 0):>8.1f}{d.accuracy:>7.2f}"
                         f"  {'yes' if d.accepted else 'NO'}")
        lines.append("")
        lines.append(f"outcome: {self.outcome}  "
                     f"({len(self.accepted)} of {len(self.decisions)} kept, "
                     f"{self.evaluations} evaluations, {self.seconds:.2f}s)")
        acc = self.prediction_accuracy
        if acc is not None:
            lines.append(f"prediction accuracy: {acc:.2f} of the predicted improvement "
                         f"arrived (median, on the objective's own axis)")
        if self.violations:
            lines.append("")
            lines.append("unmet after every action available:")
            for v in self.violations:
                lines.append(f"  {v['constraint']}: {v['actual']:,.2f} vs "
                             f"{v['limit']:,.2f} ({v['over'] * 100:.0f}% over)")
        if self.engineer:
            lines.append("")
            lines.append("what to engineer into existence, to unlock more:")
            for e in self.engineer[:6]:
                lines.append(f"  {e['property']:<20} would unlock {', '.join(e['levers'])}")
        if self.findings:
            lines.append("")
            lines.append("findings — a lever's declarations did not match the measurement:")
            for f in self.findings:
                lines.append(f"  {f}")
        return "\n".join(lines)


# ------------------------------------------------------------------------------ the loop
class ControlLoop:
    """Observe, propose, predict, act, verify, settle — until nothing is left to try."""

    def __init__(self, workload: Workload, objective: Objective | None = None, *,
                 gpu: str | None = None, levers: list[str] | None = None,
                 bus: EventBus | None = None, tolerance: float = TOLERANCE,
                 max_evaluations: int = 400):
        self.objective = objective or Objective()
        self.bus = bus or BUS
        self.tolerance = tolerance
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.gpu = gpu or workload.gpu
        # Start from the bare workload, not from `recommend()`, unless asked. The loop's job is
        # to arrive somewhere; starting it at the answer would make the trace a formality.
        self.stack = (Stack(workload, self.gpu).with_levers(*levers) if levers
                      else Stack(workload, self.gpu))
        self.refused: set[tuple] = set()      # actions measured and rejected: never re-propose
        self.findings: list[str] = []
        self._solo: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------- observe
    def observe(self) -> dict:
        self.evaluations += 1
        return self.stack.plan()

    def _config_gain(self, lever: str, gpu: str) -> float:
        """This lever's throughput gain from its *config change alone*, on this card.

        Measured once against the bare config and cached. Config changes — `tp`, `batch`,
        `kv_bytes`, the int4 kernel factor — are the ones whose composition is not obvious, and
        this is the number `predict()` assumes composes. That assumption is the thing `verify()`
        tests, and it is where every declaration bug this package has had actually lived.
        """
        key = (lever, gpu)
        if key not in self._solo:
            self.evaluations += 1
            w = self.stack.workload
            from .stack import _step_gpu, _step_model
            c0 = base_config(model=_step_model(w.model), gpu=_step_gpu(gpu),
                             batch=w.batch, ctx=w.ctx)
            t0 = evaluate(c0)["tokens_per_s"]
            self._solo[key] = evaluate(LEVERS[lever].apply(c0))["tokens_per_s"] / t0
        return self._solo[key]

    @staticmethod
    def _apply_factors(parts: dict, f: dict, sign: float) -> dict:
        """Scale a step's terms by a lever's declared factors. `sign` is +1 add, -1 remove."""
        out = {}
        for k, v in parts.items():
            if k == "attention":
                out[k] = v * (f["attention"] if sign > 0 else 1.0 / max(f["attention"], 1e-9))
            elif k == "sampling":
                out[k] = v * (f["sampling"] if sign > 0 else 1.0 / max(f["sampling"], 1e-9))
            elif k == "host sync":
                out[k] = max(0.0, v + sign * f["host_sync_ms"] / 1000.0)
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------- propose
    def candidates(self) -> list[Action]:
        """Every legal action, before any of them is priced."""
        out: list[Action] = []
        held = set(self.stack.names)
        obj = self.objective

        def allowed(name: str) -> bool:
            return obj.allow_accuracy_loss or "accuracy" not in LEVERS[name].spends

        for lv in self.stack.offered():
            if allowed(lv.name):
                out.append(Action("add", lever=lv.name))

        # A swap is how the loop reaches a lever whose domain is already occupied — int4 in
        # place of fp8, one attention partition for another. Without it, the first lever pulled
        # in a domain is permanent, which is an artefact of arrival order, not of merit.
        props = self.stack.workload.properties
        for name, lv in LEVERS.items():
            if name in held or not allowed(name) or not lv.applies_to(props):
                continue
            rival = next((h for h in self.stack.names if LEVERS[h].domain == lv.domain), None)
            if rival:
                out.append(Action("swap", lever=name, out=rival))

        for name in self.stack.names:
            out.append(Action("drop", lever=name))

        if obj.allow_gpu_moves:
            # Only cards the step budget actually models. `_step_gpu()` maps anything else onto
            # the nearest one it knows, so a move to a card outside that set evaluates as a
            # different card entirely: the loop proposed "move to B200", predicted 2.4x from the
            # bandwidth ratio, measured an unchanged H100, and reverted it — a rejection that
            # said nothing about a B200. Better to not offer a decision the model cannot make.
            from .step import STEP_HW
            for g in GPUS:
                if g != self.gpu and g in STEP_HW:
                    out.append(Action("move", gpu=g))

        return [a for a in out if a.key() not in self.refused]

    # ------------------------------------------------------------------- predict
    def predict(self, action: Action, plan: dict) -> dict:
        """What this action will do, without evaluating the composition.

        The prediction composes at the level of the step's *terms*, not of its total, and that
        distinction took a wrong version to find. Composing solo throughput gains
        multiplicatively is optimistic for **any** two levers that shrink parts of an additive
        budget — halve attention and halve sampling and the total does not quarter — so a
        multiplicative rule reports a shortfall on perfectly honest levers. That is Amdahl's
        law, not a lying declaration, and a loop that cannot tell them apart produces a finding
        on every second step and trains its reader to ignore findings.

        So each verb predicts from the source that is actually independent of the joint
        measurement:

        * **add / swap** — the lever's declared factors applied to the terms of the step *as it
          now stands* (Amdahl-correct), multiplied by the lever's solo **config** gain. The
          assumption being tested is the one worth testing: that a lever's effect on `tp`,
          `batch` or `kv_bytes` composes with whatever the stack already did. Every declaration
          bug this package has had lived exactly there.
        * **drop** — the same arithmetic inverted. A lever whose removal is predicted to cost
          nothing has stopped paying, which is how the loop prunes.
        * **move** — from the *catalog*, not the step model: a memory-bound step scales with HBM
          bandwidth, a compute-bound one with FLOP/s. Genuinely independent, and wrong exactly
          when the move changes which term binds — the interesting case, which `verify()` names.
          Only cards `step.STEP_HW` models are offered; see `candidates()`.

        The step is only half of it. A lever's `prefill_factor` and `idle_factor` touch no step
        term at all: they change how many tokens the run pays for and how long a slot is held,
        which is where the *money* is. Predicting only the step made the loop blind to prefix
        caching — the single largest cost lever in the package, an 82% saving — because its
        effect on a decode step is exactly nothing. A cost objective that cannot see the cost
        lever is not a cost objective, so the prediction covers the whole run.
        """
        gpu, target = self.gpu, (action.gpu or self.gpu)
        w = self.stack.workload
        fp = fi = fg = 1.0            # prefill tokens, idle time, GPUs held
        if action.verb in ("add", "swap", "drop"):
            sign = -1.0 if action.verb == "drop" else 1.0
            parts = plan["parts"]
            step_s = sum(parts.values())
            f = LEVERS[action.lever].factors(w)
            after = self._apply_factors(parts, f, sign)
            if action.verb == "swap":
                # A swap is a removal and an addition, and both sides' factors must be undone
                # and applied — otherwise the prediction credits the incoming lever with the
                # outgoing one's effect and every swap looks like a free win.
                g2 = LEVERS[action.out].factors(w)
                after = self._apply_factors(after, g2, -1.0)
                fp, fi, fg = _ratio(f, g2, "prefill"), _ratio(f, g2, "idle"), _ratio(f, g2, "gpu")
            else:
                fp, fi, fg = (_pow(f["prefill"], sign), _pow(f["idle"], sign),
                              _pow(f["gpu"], sign))
            gf = step_s / max(sum(after.values()), 1e-12)
            gc = self._config_gain(action.lever, gpu)
            if action.verb == "swap":
                gc /= max(self._config_gain(action.out, gpu), 1e-9)
            elif action.verb == "drop":
                gc = 1.0 / max(gc, 1e-9)
            g = gf * gc
            rivals = [h for h in self.stack.names if h != action.lever]
            shared = sorted(set().union(*[shared_resources(action.lever, h) for h in rivals])
                            if rivals else set())
            rule = "declared factors on the current terms x solo config gain"
        else:
            src, dst = GPUS[gpu], GPUS[target]
            # `gemm_bound` is the string "compute" or "memory", not a boolean, so a bare truth
            # test takes the compute branch for *both* values and the bandwidth rule never runs —
            # which is the wrong rule for almost every decode step, since almost every decode
            # step is memory-bound. The console's cross-check is what surfaced it: the two
            # implementations picked different cards, and the disagreement was real in both.
            compute_bound = plan["gemm_bound"] == "compute"
            g = (dst["tf16"] / src["tf16"]) if compute_bound else (dst["bw"] / src["bw"])
            shared = []
            rule = ("FLOP/s ratio (compute-bound)" if compute_bound
                    else "HBM bandwidth ratio (memory-bound)")

        pred_tps = plan["tokens_per_s"] * g
        # The run, not just the step. Prefill tokens scale by the declared factor; decode shrinks
        # with throughput; tool waiting shrinks only if a lever overlaps it.
        prefill_tokens = plan["prefill_tokens"] * fp
        pred_wall = plan["prefill_s"] * fp + plan["decode_s"] / max(g, 1e-9) + plan["tool_s"] * fi
        pred_token_cost = (prefill_tokens / 1e6 * w.price_in_per_mtok
                           + plan["decode_tokens"] / 1e6 * w.price_out_per_mtok)
        # Money moves two ways at once on a card change, and on TP: the hourly rate or the
        # number of cards changes, and so does how long the slot is held. Predicting only the
        # rate would rank every cheap card first.
        pred_gpus = plan["gpus"] * fg
        pred_gpu_cost = (GPUS[target]["usd"] * pred_gpus * (pred_wall / 3600.0)
                         / max(plan["batch"], 1))
        pred = dict(plan)
        pred.update(tokens_per_s=pred_tps,
                    tpot_ms=plan["tpot_ms"] / max(g, 1e-9),
                    wall_s=pred_wall, prefill_tokens=prefill_tokens, gpus=pred_gpus,
                    token_cost_usd=pred_token_cost, gpu_cost_usd=pred_gpu_cost,
                    total_cost_usd=pred_token_cost + pred_gpu_cost,
                    gain=g, rule=rule, predicted_conflict=shared)
        return pred

    # ----------------------------------------------------------------------- act
    def _with(self, action: Action) -> tuple[Stack, str]:
        """The stack this action would produce. Pure: it does not mutate the loop."""
        w, gpu, names = self.stack.workload, self.gpu, list(self.stack.names)
        if action.verb == "add":
            names.append(action.lever)
        elif action.verb == "swap":
            names = [action.lever if n == action.out else n for n in names]
        elif action.verb == "drop":
            names = [n for n in names if n != action.lever]
        else:
            gpu = action.gpu
        return Stack(w, gpu).with_levers(*names), gpu

    # -------------------------------------------------------------------- verify
    def _accuracy(self, before: dict, pred: dict, got: dict) -> tuple[float, str]:
        """How much of the predicted improvement actually arrived, on the objective's own axis.

        Checking throughput would be the obvious thing and it is the wrong thing: a run
        minimizing cost can predict a 3.48 -> 0.44 saving, measure 0.63, and be scored "met"
        because tokens per second came out exactly as predicted. The prediction that matters is
        the prediction of the quantity being optimized, so that is what is scored — and on the
        very first run this turned a spurious "met" into an honest 0.66.
        """
        obj = self.objective
        want = obj.value(before) - obj.value(pred)
        if abs(want) > abs(obj.value(before)) * 1e-6:
            return (obj.value(before) - obj.value(got)) / want, obj.maximize or obj.minimize
        # Nothing material was predicted on the objective — an SLO-driven move, typically — so
        # fall back to the axis the action was actually pulling on.
        if pred["tokens_per_s"]:
            return got["tokens_per_s"] / pred["tokens_per_s"], "tokens_per_s"
        return 1.0, "none"

    def _verdict(self, pred: dict, got: dict, before: dict) -> tuple[str, str | None, float]:
        """Compare the measurement against the prediction, and name the disagreement."""
        ratio, axis = self._accuracy(before, pred, got)
        if not self.objective.worth_it(before, got):
            # Two different rejections, and conflating them would misreport the loop's own
            # behaviour: a card that measures *slower* than the one it replaces is a mistake,
            # while one that measures the same is a deploy nobody should schedule.
            sb, sa = self.objective.score(before), self.objective.score(got)
            return ("worse" if sa > sb else "not worth it"), None, ratio

        # Two axes, two meanings, and only one of them is evidence about a lever's declarations.
        #
        # The *step* prediction is term-exact by construction — the declared factors, applied to
        # the terms as they stand — so a shortfall there is a real disagreement about how two
        # levers compose, and with no declared shared resource it is a defect in `levers.py`.
        #
        # The *objective* prediction goes through the whole run model, including a prefill
        # estimate that is deliberately a clamp rather than a re-derivation. A shortfall there
        # says the loop's own arithmetic is coarse, not that a lever lied — and reporting it as a
        # lever defect would be a confident accusation built on the wrong evidence.
        step_ratio = got["tokens_per_s"] / pred["tokens_per_s"] if pred["tokens_per_s"] else 1.0
        if step_ratio < self.tolerance:
            if pred["predicted_conflict"]:
                return "short-as-declared", None, ratio
            return "short-undeclared", (
                f"predicted {pred['tokens_per_s']:,.0f} tok/s, measured "
                f"{got['tokens_per_s']:,.0f} ({step_ratio:.2f}x) with no declared shared "
                f"resource"), ratio
        if ratio < self.tolerance:
            return f"mispredicted {axis}", None, ratio
        return "met", None, ratio

    # ---------------------------------------------------------------------- step
    def step(self, n: int) -> Decision | None:
        """One iteration. Returns None when no legal action is left to try."""
        t0 = time.time()
        before = self.observe()
        self.bus.emit("agent.observe", step=n, levers=list(self.stack.names), gpu=self.gpu,
                      tokens_per_s=before["tokens_per_s"], tpot_ms=before["tpot_ms"],
                      cost_usd=before["total_cost_usd"], bottleneck=before["bottleneck"],
                      satisfied=self.objective.satisfied(before))

        cands = self.candidates()
        if not cands:
            self.bus.emit("agent.exhausted", step=n, reason="no legal action left")
            return None
        if self.evaluations >= self.max_evaluations:
            self.bus.emit("agent.exhausted", level="warn", step=n,
                          reason="evaluation budget spent", evaluations=self.evaluations)
            return None

        # Rank on the prediction — cheap — and measure only the leader. This is the line that
        # separates a control loop from an exhaustive search over 2^17 subsets.
        #
        # Pricing is not free either: a candidate whose solo config gain has not been cached
        # costs an evaluation to rank. So the budget is enforced *inside* the ranking and not
        # only at the top of the step — otherwise a single step with twenty-two candidates
        # overshoots a budget of eight before the loop ever looks at it, which is what the first
        # version did. Running out mid-ranking degrades to "choose the best of what we could
        # afford to price", which is the right behaviour for a bounded search.
        priced, truncated = [], False
        for a in cands:
            if priced and self.evaluations >= self.max_evaluations:
                truncated = True
                break
            p = self.predict(a, before)
            priced.append((self.objective.score(p), a, p))
        if truncated:
            self.bus.emit("agent.budget", level="warn", step=n, priced=len(priced),
                          candidates=len(cands), evaluations=self.evaluations,
                          reason="evaluation budget spent while pricing candidates")
        ranked = sorted(priced, key=lambda t: t[0])
        _, action, pred = ranked[0]
        if not self.objective.worth_it(before, pred):
            self.bus.emit("agent.converged", step=n, considered=len(priced),
                          best_candidate=action.label(),
                          reason=(f"no action is predicted to improve the objective by "
                                  f"{self.objective.min_improvement:.1%}"))
            return None

        self.bus.emit("agent.decide", step=n, action=action.label(), verb=action.verb,
                      considered=len(priced), rule=pred["rule"],
                      predicted_tokens_per_s=pred["tokens_per_s"],
                      predicted_cost_usd=pred["total_cost_usd"],
                      predicted_conflict=pred["predicted_conflict"],
                      runner_up=ranked[1][1].label() if len(ranked) > 1 else None)

        cand_stack, cand_gpu = self._with(action)
        self.evaluations += 1
        got = cand_stack.plan()
        verdict, finding, ratio = self._verdict(pred, got, before)

        d = Decision(step=n, action=action.label(), verb=action.verb,
                     reason=f"best predicted objective of {len(cands)} candidates",
                     predicted=_slim(pred), actual=_slim(got), verdict=verdict,
                     accuracy=ratio, considered=len(priced), seconds=time.time() - t0)
        if finding:
            d.finding = f"{action.label()}: {finding}"
            self.findings.append(d.finding)
            self.bus.emit("agent.finding", level="warn", step=n, action=action.label(),
                          detail=finding)

        if verdict in ("worse", "not worth it"):
            # Reverted, and struck off. Re-proposing an action that has already measured badly
            # is how a greedy loop spends a whole budget oscillating between two states.
            self.refused.add(action.key())
            d.accepted = False
            self.bus.emit("agent.revert", level="warn", step=n, action=action.label(),
                          predicted_tokens_per_s=pred["tokens_per_s"],
                          actual_tokens_per_s=got["tokens_per_s"], verdict=verdict,
                          reason=("measured worse than the state it replaced" if verdict == "worse"
                                  else "measured no better than the deadband requires"))
        else:
            self.stack, self.gpu = cand_stack, cand_gpu
            d.accepted = True
            # A change of state invalidates nothing about solo gains on this card, but a change
            # of *card* does: every cached gain was measured against a different baseline.
            if action.verb == "move":
                self._solo.clear()
            self.bus.emit("agent.accept", step=n, action=action.label(), verdict=verdict,
                          accuracy=round(ratio, 4),
                          levers=list(self.stack.names), gpu=self.gpu,
                          tokens_per_s=got["tokens_per_s"], cost_usd=got["total_cost_usd"],
                          tpot_ms=got["tpot_ms"], satisfied=self.objective.satisfied(got))
        d.levers_after = list(self.stack.names)
        d.gpu_after = self.gpu
        return d

    # ----------------------------------------------------------------------- run
    def run(self, max_steps: int = 12) -> Trace:
        """Loop until convergence, exhaustion or the step budget, and return the trace."""
        t0 = time.time()
        w = self.stack.workload
        with self.bus.run(prefix="agent") as rid:
            start = self.observe()
            tr = Trace(run_id=rid, objective=self.objective.describe(),
                       workload=w.describe().replace("\n", " "), start_gpu=self.gpu,
                       start_levers=list(self.stack.names), start_plan=_slim(start))
            self.bus.emit("agent.start", objective=self.objective.describe(),
                          workload=w.kind, model=w.model, gpu=self.gpu,
                          properties=sorted(w.properties), max_steps=max_steps)

            n = 0
            while n < max_steps:
                n += 1
                d = self.step(n)
                if d is None:
                    tr.outcome = "converged"
                    break
                tr.decisions.append(d)
            else:
                tr.outcome = "budget"

            final = self.observe()
            tr.final_levers = list(self.stack.names)
            tr.final_gpu = self.gpu
            tr.final_plan = _slim(final)
            tr.violations = self.objective.violations(final)
            tr.findings = list(self.findings)
            tr.evaluations = self.evaluations
            tr.seconds = time.time() - t0
            if tr.violations:
                # Converged and still short of the SLO is not a failure of the loop, it is a
                # statement about the workload: no combination of available levers gets there.
                # What is left is to change the workload, so say which properties would help.
                tr.outcome = "infeasible"
                tr.engineer = self.engineer()
            elif tr.outcome == "converged" and self.objective.satisfied(final):
                tr.outcome = "satisfied"

            self.bus.emit("agent.done",
                          level="warn" if tr.outcome == "infeasible" else "info",
                          outcome=tr.outcome, steps=len(tr.decisions),
                          kept=len(tr.accepted), evaluations=tr.evaluations,
                          levers=tr.final_levers, gpu=tr.final_gpu,
                          tokens_per_s=final["tokens_per_s"],
                          cost_usd=final["total_cost_usd"], tpot_ms=final["tpot_ms"],
                          speedup=final["tokens_per_s"] / max(start["tokens_per_s"], 1e-9),
                          findings=len(tr.findings), seconds=round(tr.seconds, 3))
        return tr

    # ------------------------------------------------------------------ escalate
    def engineer(self) -> list[dict]:
        """The properties worth building, grouped by what each would unlock.

        This is the loop's last move and its most useful one. When no action reaches the
        objective, the answer is not a smaller number, it is a different workload — a prefix
        worth caching, a batch worth making ragged — and `Stack.unavailable()` already knows
        which property gates which lever.
        """
        by_prop: dict[str, list[str]] = {}
        for lv, missing in self.stack.unavailable():
            for m in missing:
                by_prop.setdefault(m, []).append(lv.name)
        return [dict(property=p, levers=sorted(ls), count=len(ls))
                for p, ls in sorted(by_prop.items(), key=lambda kv: -len(kv[1]))]


def _pow(x: float, sign: float) -> float:
    """A factor applied (`sign=+1`) or undone (`sign=-1`), with the plan's own floor.

    `prefix caching` declares `prefill_factor = 0.0`, meaning "only the tokens never seen
    before", and a zero has no inverse. `Stack.plan()` already resolves that by treating a
    working prefix cache as a 0.95 hit rate rather than a perfect one, so the floor here is that
    same 0.05 — the same clamp, in the same units, not a fudge factor chosen to make division
    work.
    """
    x = max(x, 0.05)
    return x if sign > 0 else 1.0 / x


def _ratio(incoming: dict, outgoing: dict, key: str) -> float:
    """The net factor of a swap: what arrives, divided by what leaves."""
    return max(incoming[key], 0.05) / max(outgoing[key], 0.05)


def _slim(plan: dict) -> dict:
    """The fields a trace needs. A full plan per decision would be mostly duplicated config."""
    keep = ("tokens_per_s", "tpot_ms", "step_ms", "wall_s", "total_cost_usd", "token_cost_usd",
            "gpu_cost_usd", "bottleneck", "gpus", "concurrent", "batch", "spare_compute",
            "gemm_bound", "gain", "rule", "predicted_conflict")
    return {k: plan[k] for k in keep if k in plan}


def tune(workload: Workload, objective: Objective | None = None, *, max_steps: int = 12,
         **kw) -> Trace:
    """One call, for the common case."""
    return ControlLoop(workload, objective, **kw).run(max_steps=max_steps)


__all__ = ["ControlLoop", "Objective", "Action", "Decision", "Trace", "tune", "TOLERANCE"]
