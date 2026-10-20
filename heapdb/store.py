"""HeapDB: the store, the top-k attribution query, feedback and persistence.

Scoring model
-------------
For an effect ``E`` observed for entity ``x`` at time ``T`` and a candidate
event ``e`` of ``x`` (time ``t``, weight ``w``, features ``F``, type ``y``)::

    score(e | E) = kernel_E(T - t) * w * noisy_or_{f in F}(strength_x(f, E)) * rules_E(e)

* *timing*: does the lag fit how this effect usually lags its causes?
* *dose*: ``w`` is the exposure magnitude (portion size, deploy size, amount).
  Propensity is **not** in the weight; it lives in the links.
* *plausibility*: entity-level link strengths (registered prior + shrunk
  global evidence + the entity's own evidence; see :mod:`heapdb.links`).
* *rules*: product of the multipliers (each in ``[0, 1]``) of the effect's
  plausibility rules that match the event. ``cause_types`` remains the
  type-level hard filter.

Top-k query (threshold-algorithm style)
---------------------------------------
1. Only segments overlapping ``[T - support, T - min_lag]`` can contribute.
2. For each such segment compute an upper bound on the score of *any* event
   in it, from its summaries and the *current* links (so it stays valid as
   links are learned)::

       bound = kernel.max_over(lag range of segment)
             * max over distinct (type, feature set) keys of the segment of
                   max_weight(key) * noisy_or(strengths of key's features)
                                   * rule_bound(type)

   Each factor upper-bounds the matching factor of every event with that
   key, so the product bounds every event in the segment. If the segment
   tracks too many distinct keys we fall back to the (looser, still valid)
   union bound ``max_weight * noisy_or(union of features) * max rule_bound``.
   Note that replacing noisy-OR with the max single link would *not* be an
   upper bound: noisy-OR >= max link.
3. Visit segments in descending bound order, scoring their events into a
   bounded min-heap of size ``k``. As soon as the heap is full and its
   smallest score is ``>=`` the next segment's bound, no remaining segment
   can improve the answer and we stop. Everything not visited is "pruned".

The result is exactly the brute-force top-k (up to ties); see
``tests/test_topk.py``, which checks this on random data.
"""

from __future__ import annotations

import bisect
import heapq
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping

from .events import Event, as_feature_set, validate_weight
from .kernels import Kernel, kernel_from_dict
from .links import CREDIT_MODES, BetaCounts, LinkTable, _fmt, responsibilities
from .segments import DEFAULT_MAX_SETS, EntityLog, Segment
from .stats import beta_interval

# Relative slack applied to segment bounds before comparing them with real
# scores. Mathematically bound >= score; this guards against last-bit floating
# point differences (e.g. noisy-OR over a union versus a subset in a different
# order), so we never prune an event we should have kept.
_BOUND_SLACK = 1e-9

SNAPSHOT_VERSION = 3

#: Default background (leak) score of an effect for :meth:`HeapDB.observe`.
DEFAULT_LEAK = 0.05


# ---------------------------------------------------------------------------
# Effect registration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A soft plausibility rule: matching events get ``score *= multiplier``.

    ``multiplier`` is in ``[0, 1]`` (0 = hard block). Values above 1 are
    rejected because they would break the segment upper bounds.

    ``predicate`` is any ``Callable[[Event], bool]``. Since it is opaque,
    the query engine's bound treats it as "may not match" (factor 1). A rule
    with ``predicate=None`` matches *every* event of ``types`` (or every
    event, if ``types`` is None); such type-level rules do tighten bounds
    and, unlike predicate rules, survive a snapshot round-trip.
    """

    name: str
    predicate: Callable[[Event], bool] | None
    multiplier: float
    types: frozenset[str] | None = None

    def applies_to_type(self, type_: str) -> bool:
        return self.types is None or type_ in self.types

    def matches(self, event: Event) -> bool:
        if not self.applies_to_type(event.type):
            return False
        return self.predicate is None or bool(self.predicate(event))


@dataclass
class EffectSpec:
    """Everything HeapDB knows about one effect type."""

    name: str
    kernel: Kernel
    links: LinkTable
    #: If set, only events of these types are considered as causes.
    cause_types: frozenset[str] | None = None
    #: Plausibility rules, applied in order (their multipliers multiply).
    rules: list[Rule] = field(default_factory=list)
    #: Credit assignment for confirmations: "proportional" (EM) or "uniform".
    credit: str = "proportional"
    #: Names of predicate rules found in a snapshot that must be re-registered.
    pending_rules: list[str] = field(default_factory=list)
    #: v0.4 observational learning: background score of the effect, i.e. the
    #: "happened for no event in the window" alternative in :meth:`HeapDB.observe`.
    leak: float = DEFAULT_LEAK
    #: v0.4: negative evidence per feature added by :meth:`HeapDB.settle` for an
    #: event whose window passed without the effect.
    w_neg: float = 1.0
    #: v0.4: False for one-shot effects (churn, first conversion): once an
    #: entity has had the effect, :meth:`HeapDB.settle` does not add negative
    #: evidence for its later events (they could not have caused it again).
    repeatable: bool = True
    #: v0.4: observational bookkeeping (occurrences, credited/rejected events,
    #: settle cursors). See :class:`ObservationState`.
    obs: "ObservationState" = field(default_factory=lambda: ObservationState())

    def rule_multiplier(self, event: Event) -> float:
        """Product of the multipliers of the rules matching ``event``."""
        m = 1.0
        for r in self.rules:
            if m == 0.0:
                break
            if r.matches(event):
                m *= r.multiplier
        return m

    def rule_bound(self, type_: str) -> float:
        """Upper bound of the rule product for any event of ``type_``."""
        b = 1.0
        for r in self.rules:
            if r.predicate is None and r.applies_to_type(type_):
                b *= r.multiplier
        return b

    def type_allowed(self, type_: str) -> bool:
        return self.cause_types is None or type_ in self.cause_types


@dataclass
class ObservationState:
    """Per-effect state of observational learning (v0.4).

    * ``occurrences``: entity -> sorted times at which the effect was observed
      with :meth:`HeapDB.observe` (no labelled cause).
    * ``credited``: event id -> total soft responsibility the event received
      from observed occurrences. Credited events are never rejected by
      :meth:`HeapDB.settle` for this effect.
    * ``rejected``: event id -> negative evidence weight added by ``settle``.
      If an occurrence is observed later in the event's window, ``observe``
      undoes that rejection before crediting (order independence).
    * ``cursors``: entity -> event-time watermark; every event with
      ``time <= cursor`` has been settled. ``late``: entity -> ids of events
      inserted at or before the cursor (settled on the next call).
    * ``stats``: running counters (see :meth:`HeapDB.observation_stats`).
    """

    occurrences: dict[str, list[float]] = field(default_factory=dict)
    credited: dict[int, float] = field(default_factory=dict)
    rejected: dict[int, float] = field(default_factory=dict)
    cursors: dict[str, float] = field(default_factory=dict)
    late: dict[str, list[int]] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=lambda: dict(_EMPTY_OBS_STATS))

    def to_dict(self) -> dict:
        return {
            "occurrences": {e: ts for e, ts in sorted(self.occurrences.items())},
            "credited": {str(k): v for k, v in sorted(self.credited.items())},
            "rejected": {str(k): v for k, v in sorted(self.rejected.items())},
            "cursors": dict(sorted(self.cursors.items())),
            "late": {e: ids for e, ids in sorted(self.late.items()) if ids},
            "stats": dict(self.stats),
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> "ObservationState":
        st = cls()
        if not d:
            return st
        st.occurrences = {e: sorted(float(t) for t in ts) for e, ts in d.get("occurrences", {}).items()}
        st.credited = {int(k): float(v) for k, v in d.get("credited", {}).items()}
        st.rejected = {int(k): float(v) for k, v in d.get("rejected", {}).items()}
        st.cursors = {e: float(t) for e, t in d.get("cursors", {}).items()}
        st.late = {e: [int(i) for i in ids] for e, ids in d.get("late", {}).items()}
        st.stats.update({k: float(v) for k, v in d.get("stats", {}).items()})
        return st


_EMPTY_OBS_STATS = {
    "occurrences": 0.0,          # observe() calls
    "leak_responsibility": 0.0,  # sum over occurrences of leak / (leak + sum scores)
    "credited_mass": 0.0,        # sum over occurrences of 1 - leak responsibility
    "events_settled": 0.0,       # events whose window was examined by settle()
    "events_rejected": 0.0,      # ... that received negative evidence
    "rejected_weight": 0.0,      # total negative evidence weight (per event)
    "skipped_credited": 0.0,     # ... skipped because they had soft credit
    "skipped_observed": 0.0,     # ... skipped: an occurrence fell in their window
    "skipped_terminal": 0.0,     # ... skipped: after a one-shot effect occurred
    "rejections_undone": 0.0,    # rejections reversed by a late observe()
}


@dataclass(frozen=True)
class EventCredit:
    """Soft credit one candidate event received from :meth:`HeapDB.observe`."""

    event_id: int
    score: float
    responsibility: float           # r_e = score / (leak + sum of scores)
    features: Mapping[str, float]   # feature -> alpha added (r_e * share_f)

    def __str__(self) -> str:
        fs = ", ".join(f"{f}+{a:.3f}" for f, a in sorted(self.features.items(),
                                                           key=lambda kv: -kv[1]))
        return f"#{self.event_id} score={self.score:.4f} r={self.responsibility:.3f} [{fs}]"


@dataclass(frozen=True)
class ObserveReport:
    """What :meth:`HeapDB.observe` did for one effect occurrence."""

    entity: str
    effect: str
    at: float
    leak: float
    leak_responsibility: float      # leak / (leak + sum of scores)
    credits: tuple[EventCredit, ...]  # sorted by responsibility, desc

    @property
    def total_credit(self) -> float:
        """Sum of the events' responsibilities (= 1 - leak responsibility)."""
        return sum(c.responsibility for c in self.credits)

    def __str__(self) -> str:
        lines = [f"Observed '{self.effect}' for {self.entity} at t={self.at:g}: "
                 f"{len(self.credits)} candidate(s), leak {self.leak:g} took "
                 f"{self.leak_responsibility:.1%}"]
        lines += [f"  {c}" for c in self.credits]
        return "\n".join(lines)


@dataclass(frozen=True)
class SettleReport:
    """What :meth:`HeapDB.settle` did (summed over effects and entities)."""

    until: float
    events_settled: int
    events_rejected: int
    rejected_weight: float
    skipped_credited: int
    skipped_observed: int  # effect observed in the window, or after a one-shot effect
    per_effect: Mapping[str, int]  # effect -> events rejected

    def __str__(self) -> str:
        return (f"Settled until t={self.until:g}: {self.events_settled} event(s) examined, "
                f"{self.events_rejected} rejected (weight {self.rejected_weight:.2f}), "
                f"{self.skipped_credited} skipped (had soft credit), "
                f"{self.skipped_observed} skipped (effect observed in window)")


@dataclass(frozen=True)
class LinkRow:
    """One feature in a :class:`LinkReport`."""

    feature: str
    strength: float          # posterior mean alpha / (alpha + beta)
    alpha: float
    beta: float
    n: float                 # effective evidence (excludes the prior)
    lo: float                # credible interval lower bound
    hi: float                # credible interval upper bound
    lift: float              # strength / baseline


@dataclass(frozen=True)
class LinkReport:
    """Features ranked by learned link strength for one effect.

    These are **associations under the model** (noisy-OR links learned from
    feedback and/or observation), *not* causal effects: features that merely
    co-occur with true causes (confounders, selection effects) can rank high.
    """

    effect: str
    entity: str | None
    baseline: float
    level: float
    rows: tuple[LinkRow, ...]

    def __iter__(self) -> Iterator[LinkRow]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def row(self, feature: str) -> LinkRow | None:
        for r in self.rows:
            if r.feature == feature:
                return r
        return None

    def __str__(self) -> str:
        who = self.entity if self.entity is not None else "(global)"
        pct = f"{self.level:.0%}"
        lines = [f"Links of '{self.effect}' for {who} (baseline {self.baseline:.3f}; "
                 f"associations under the model, not causal effects):",
                 f"  {'feature':<36} {'mean':>6} {pct + ' CI':>15} {'lift':>6} "
                 f"{'n':>8} {'alpha':>8} {'beta':>8}"]
        for r in self.rows:
            lines.append(f"  {r.feature:<36} {r.strength:>6.3f} "
                         f"[{r.lo:.3f}, {r.hi:.3f}] {r.lift:>6.2f} {r.n:>8.1f} "
                         f"{r.alpha:>8.2f} {r.beta:>8.2f}")
        if not self.rows:
            lines.append("  (no features with enough support)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureContribution:
    """How one feature of a candidate event contributes to its score."""

    feature: str
    strength: float  # posterior mean alpha / (alpha + beta)
    alpha: float
    beta: float

    def __str__(self) -> str:
        return f"{self.feature}={self.strength:.2f} (a={_fmt(self.alpha)}, b={_fmt(self.beta)})"


@dataclass(frozen=True)
class AppliedRule:
    name: str
    multiplier: float

    def __str__(self) -> str:
        return f"{self.name} x{self.multiplier:g}"


@dataclass(frozen=True)
class JointCause:
    """A set of features reported as one cause (e.g. ``beans+onion``).

    ``strength`` is what scoring uses (noisy-OR of the members' per-feature
    strengths). ``joint`` is the joint Beta link learned for exactly this
    set, if it has been judged by feedback. ``inseparable`` is True while
    the members have always co-occurred for this entity.
    """

    features: tuple[str, ...]
    strength: float
    joint: BetaCounts | None
    inseparable: bool

    @property
    def name(self) -> str:
        return "+".join(self.features)

    def __str__(self) -> str:
        out = f"{self.name}={self.strength:.2f}"
        if self.joint is not None:
            out += (f" (joint link {self.joint.strength:.2f}, "
                    f"a={_fmt(self.joint.alpha)}, b={_fmt(self.joint.beta)})")
        if self.inseparable:
            out += " inseparable"
        return out


@dataclass(frozen=True)
class Explanation:
    """Why an event got its score: ``kernel * weight * link * rules``."""

    dt: float                 # seconds between cause and effect
    kernel_value: float       # timing plausibility in [0, 1]
    weight: float             # event weight (dose / exposure)
    link_strength: float      # noisy-OR of feature strengths
    features: tuple[FeatureContribution, ...]  # sorted by strength, desc
    warnings: tuple[str, ...] = ()
    rules: tuple[AppliedRule, ...] = ()
    joint_causes: tuple[JointCause, ...] = ()
    entity: str = ""
    #: True for :meth:`HeapDB.feature_risk` explanations: the candidate has
    #: not happened yet, so there is no lag and ``kernel_value`` is 1.
    prospective: bool = False

    @property
    def rule_multiplier(self) -> float:
        m = 1.0
        for r in self.rules:
            m *= r.multiplier
        return m

    @property
    def score(self) -> float:
        return self.kernel_value * self.weight * self.link_strength * self.rule_multiplier

    @property
    def inseparable(self) -> bool:
        """True if some features of this event are an inseparable joint cause."""
        return any(j.inseparable for j in self.joint_causes)

    def __str__(self) -> str:
        joint = {f for j in self.joint_causes if j.inseparable for f in j.features}
        parts = [str(j) for j in self.joint_causes if j.inseparable]
        parts += [str(c) for c in self.features if c.feature not in joint]
        if self.prospective:
            out = f"no lag (pre-event) -> weight {self.weight:.3g} x link {self.link_strength:.3f}"
        else:
            out = (f"lag {self.dt / 3600.0:.2f}h -> kernel {self.kernel_value:.3f} "
                   f"x weight {self.weight:.3g} x link {self.link_strength:.3f}")
        if self.rules:
            out += " x rules " + ", ".join(str(r) for r in self.rules)
        out += f" [noisy-OR of {', '.join(parts) or '-'}]"
        for j in self.joint_causes:
            if not j.inseparable:
                out += f"\n      joint link (now separable): {j}"
        for w in self.warnings:
            out += f"\n      warning: {w}"
        return out


@dataclass(frozen=True)
class Cause:
    """One ranked candidate cause."""

    rank: int
    event: Event
    score: float
    explanation: Explanation | None = None

    @property
    def entity(self) -> str:
        return self.event.entity


@dataclass
class QueryStats:
    """Work counters, so you can see the pruning happen."""

    segments_in_window: int = 0   # segments overlapping the kernel support
    segments_scanned: int = 0     # segments whose events were scored
    segments_pruned: int = 0      # skipped thanks to the bound (or zero bound)
    events_scanned: int = 0       # events actually scored
    union_fallbacks: int = 0      # segments bounded with the loose union bound

    def add(self, other: "QueryStats") -> None:
        self.segments_in_window += other.segments_in_window
        self.segments_scanned += other.segments_scanned
        self.segments_pruned += other.segments_pruned
        self.events_scanned += other.events_scanned
        self.union_fallbacks += other.union_fallbacks

    def __str__(self) -> str:
        return (f"segments: {self.segments_scanned} scanned / "
                f"{self.segments_pruned} pruned / {self.segments_in_window} in window; "
                f"events scored: {self.events_scanned}")


@dataclass
class QueryResult:
    """Ranked causes (best first) plus execution statistics. Iterable.

    For group queries ``entity`` is the comma-joined list and ``entities``
    holds the individual entities.
    """

    entity: str
    effect: str
    at: float
    causes: list[Cause] = field(default_factory=list)
    stats: QueryStats = field(default_factory=QueryStats)
    entities: tuple[str, ...] = ()

    def __iter__(self) -> Iterator[Cause]:
        return iter(self.causes)

    def __len__(self) -> int:
        return len(self.causes)

    def __getitem__(self, i: int) -> Cause:
        return self.causes[i]

    @property
    def top(self) -> Cause | None:
        """The single most likely cause ("pop the top"), or None."""
        return self.causes[0] if self.causes else None

    @property
    def is_group(self) -> bool:
        return len(self.entities) > 1


@dataclass(frozen=True)
class Candidate:
    """Something that has *not* happened yet and could cause an effect.

    E.g. an open pull request: ``Candidate("PR-812", {"migration",
    "review:fast"}, type="deploy", attrs={"lines_changed": 340},
    entity="payments")``. ``weight=None`` derives the dose from ``attrs``
    with the type's registered weight function (else 1.0). ``entity=None``
    uses the ``entity`` passed to :meth:`HeapDB.rank_candidates`.
    """

    id: Any
    features: frozenset[str] = field(default_factory=frozenset)
    weight: float | None = None
    type: str | None = None
    attrs: Mapping[str, Any] | None = field(default=None, hash=False, compare=False)
    entity: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", as_feature_set(self.features))

    @classmethod
    def from_any(cls, c: "Candidate | Mapping[str, Any]", index: int) -> "Candidate":
        if isinstance(c, Candidate):
            return c
        if isinstance(c, Mapping):
            unknown = set(c) - {"id", "features", "weight", "type", "attrs", "entity"}
            if unknown:
                raise ValueError(f"unknown candidate keys {sorted(unknown)}")
            return cls(id=c.get("id", index), features=c.get("features") or frozenset(),
                       weight=c.get("weight"), type=c.get("type"), attrs=c.get("attrs"),
                       entity=c.get("entity"))
        raise TypeError(f"candidate must be a Candidate or a dict, got {type(c).__name__}")


@dataclass(frozen=True)
class RankedCandidate:
    """One row of :meth:`HeapDB.rank_candidates` (best first)."""

    rank: int
    candidate: Candidate
    score: float
    entity: str | None
    explanation: Explanation | None = None


@dataclass(frozen=True)
class RiskResult:
    """Result of a ``RISK OF ...`` statement (see :meth:`HeapDB.feature_risk`)."""

    effect: str
    entity: str | None
    features: frozenset[str]
    type: str | None
    score: float
    explanation: Explanation | None = None

    def __str__(self) -> str:
        who = self.entity if self.entity is not None else "(global)"
        typ = f" type={self.type}" if self.type else ""
        out = (f"Risk of '{self.effect}' for {who}{typ} "
               f"[{','.join(sorted(self.features))}]: {self.score:.4f}")
        if self.explanation is not None:
            out += f"\n      {self.explanation}"
        return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRecord:
    entity: str
    event_id: int
    at: float
    score: float      # score shown at feedback time, before links were updated
    confirmed: bool

    @property
    def predicted(self) -> float:
        """Scores are not probabilities (weights can exceed 1); clip to [0, 1]."""
        return min(1.0, max(0.0, self.score))


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_predicted: float | None
    observed_rate: float | None


@dataclass(frozen=True)
class CalibrationReport:
    """Reliability table + Brier score for one effect.

    ``brier`` is the mean of ``(clip(score) - outcome)**2`` over all
    judged events; None if there is no feedback yet.
    """

    effect: str
    n: int
    brier: float | None
    bins: tuple[CalibrationBin, ...]

    def __str__(self) -> str:
        lines = [f"Calibration of '{self.effect}': n={self.n}, "
                 f"Brier={'-' if self.brier is None else f'{self.brier:.4f}'}"]
        lines.append("  bin           count  mean_pred  observed")
        for b in self.bins:
            if b.count == 0:
                continue
            lines.append(f"  [{b.lo:.2f}, {b.hi:.2f})  {b.count:>5}  "
                         f"{b.mean_predicted:>9.3f}  {b.observed_rate:>8.3f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class FeedbackReport:
    """What a feedback call did: per-feature credit of the confirmed event."""

    confirmed: int | None
    rejected: tuple[int, ...]
    credit: Mapping[str, float]


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------


class HeapDB:
    """An append-only event store answering "what most likely caused this?".

    Args:
        bucket_seconds: Width of the time buckets each entity's log is split
            into. Smaller buckets -> tighter bounds and more pruning, but more
            segments to bound. A few times smaller than your kernels' support
            is a good start.
        max_sets_per_segment: How many distinct (type, feature set) keys a
            segment tracks for tight bounds before falling back to the union
            bound. 0 always uses the union bound (v0.1 behaviour).
    """

    def __init__(self, bucket_seconds: float = 900.0,
                 max_sets_per_segment: int = DEFAULT_MAX_SETS) -> None:
        if bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be > 0")
        if max_sets_per_segment < 0:
            raise ValueError("max_sets_per_segment must be >= 0")
        self.bucket_seconds = float(bucket_seconds)
        self.max_sets_per_segment = int(max_sets_per_segment)
        self._logs: dict[str, EntityLog] = {}
        self._events: dict[int, Event] = {}
        self._effects: dict[str, EffectSpec] = {}
        self._event_types: dict[str, Callable[[Mapping[str, Any]], float] | None] = {}
        self._calibration: dict[str, list[CalibrationRecord]] = {}
        #: Event types with weight functions named in a loaded snapshot that
        #: have not been re-registered yet.
        self.pending_event_types: set[str] = set()
        self._next_id = 1

    # -- schema -------------------------------------------------------------

    def register_effect(
        self,
        name: str,
        kernel: Kernel,
        priors: Mapping[str, tuple[float, float]] | None = None,
        default_prior: tuple[float, float] = (1.0, 4.0),
        cause_types: Iterable[str] | None = None,
        credit: str = "proportional",
        global_pseudocount: float = 5.0,
        hierarchical: bool = True,
        leak: float = DEFAULT_LEAK,
        w_neg: float = 1.0,
        repeatable: bool = True,
    ) -> EffectSpec:
        """Declare an effect type, its lag kernel and its link priors.

        ``priors`` maps feature -> (alpha, beta). Features not listed use
        ``default_prior`` (default Beta(1, 4): weakly sceptical, mean 0.2).
        ``cause_types`` optionally restricts which event types may be causes.
        ``credit`` picks the confirmation update ("proportional" EM credit,
        or v0.1 "uniform"). ``global_pseudocount`` caps how much an entity
        borrows from evidence pooled across entities; ``hierarchical=False``
        disables borrowing for this effect.

        Observational learning (v0.4): ``leak`` is the effect's background
        score in :meth:`observe` -- the weight of "it happened for no event in
        the window" (> 0; larger = more sceptical, slower learning).
        ``w_neg`` is the negative evidence per feature that :meth:`settle`
        adds for an event whose window passed without the effect (>= 0).
        ``repeatable=False`` marks a one-shot effect (churn, first
        conversion): ``settle`` then ignores an entity's events from its first
        observed occurrence on, since they could not have caused it again.
        """
        if credit not in CREDIT_MODES:
            raise ValueError(f"credit must be one of {CREDIT_MODES}, got {credit!r}")
        leak = _check_leak(leak)
        w_neg = _check_w_neg(w_neg)
        spec = EffectSpec(
            name=name,
            kernel=kernel,
            links=LinkTable(default_prior=default_prior, priors=priors,
                            global_pseudocount=global_pseudocount,
                            hierarchical=hierarchical),
            cause_types=frozenset(cause_types) if cause_types else None,
            credit=credit,
            leak=leak,
            w_neg=w_neg,
            repeatable=bool(repeatable),
        )
        self._effects[name] = spec
        return spec

    def effect(self, name: str) -> EffectSpec:
        try:
            return self._effects[name]
        except KeyError:
            raise KeyError(f"unknown effect {name!r}; call register_effect first") from None

    @property
    def effects(self) -> list[str]:
        return sorted(self._effects)

    def register_event_type(
        self, type: str, weight: Callable[[Mapping[str, Any]], float] | None = None
    ) -> None:
        """Declare an event type, optionally deriving weight from ``attrs``.

        ``weight(attrs) -> float`` is called on insert when no explicit
        weight is given, e.g. ``lambda a: a.get("portion", 1.0)`` for meals
        or ``lambda a: math.log1p(a["lines_changed"])`` for deploys. The
        result must be > 0. Functions are not persisted: re-register after
        :meth:`load` (see ``pending_event_types``).
        """
        self._event_types[type] = weight
        self.pending_event_types.discard(type)

    def add_rule(
        self,
        effect: str,
        name: str,
        predicate: Callable[[Event], bool] | None,
        multiplier: float,
        types: Iterable[str] | None = None,
    ) -> Rule:
        """Add a soft plausibility rule to ``effect``.

        Matching events (``predicate(event)`` is true and, if ``types`` is
        given, the event type is in it) get ``score *= multiplier``.
        ``multiplier`` must be in ``[0, 1]``; 0 is a hard block. Rules apply
        in score and are listed in explanations. Predicate rules must be
        re-registered after :meth:`load` (see ``EffectSpec.pending_rules``).
        Re-adding a rule with an existing name replaces it.
        """
        spec = self.effect(effect)
        m = float(multiplier)
        if not 0.0 <= m <= 1.0 or math.isnan(m):
            raise ValueError(f"rule multiplier must be in [0, 1] (bounds rely on it), got {multiplier!r}")
        if predicate is not None and not callable(predicate):
            raise TypeError("predicate must be callable or None")
        rule = Rule(name, predicate, m, frozenset(types) if types is not None else None)
        spec.rules = [r for r in spec.rules if r.name != name] + [rule]
        if name in spec.pending_rules:
            spec.pending_rules.remove(name)
        return rule

    def remove_rule(self, effect: str, name: str) -> None:
        spec = self.effect(effect)
        spec.rules = [r for r in spec.rules if r.name != name]

    # -- writes -------------------------------------------------------------

    def insert(
        self,
        entity: str,
        time: float,
        type: str,
        features: Iterable[str] | str | None = None,
        weight: float | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> Event:
        """Append an event and return it (with its assigned id).

        ``weight`` is the dose / exposure (> 0). If omitted it is derived
        from ``attrs`` by the type's registered weight function, else 1.0.
        """
        attrs = dict(attrs or {})
        if weight is None:
            fn = self._event_types.get(type)
            weight = fn(attrs) if fn is not None else 1.0
        ev = Event(
            id=self._next_id,
            entity=entity,
            time=float(time),
            type=type,
            features=as_feature_set(features),
            weight=validate_weight(weight),
            attrs=attrs,
        )
        self._next_id += 1
        self._append(ev)
        return ev

    def _append(self, ev: Event) -> None:
        log = self._logs.get(ev.entity)
        if log is None:
            log = self._logs[ev.entity] = EntityLog(self.bucket_seconds,
                                                    self.max_sets_per_segment)
        log.add(ev)
        self._events[ev.id] = ev
        # Observational learning: an event arriving behind a settle cursor is
        # queued so the next settle() still examines it.
        for spec in self._effects.values():
            cur = spec.obs.cursors.get(ev.entity)
            if cur is not None and ev.time <= cur:
                spec.obs.late.setdefault(ev.entity, []).append(ev.id)

    # -- reads --------------------------------------------------------------

    def get(self, event_id: int) -> Event:
        return self._events[event_id]

    def events(self, entity: str | None = None) -> Iterator[Event]:
        if entity is None:
            yield from self._events.values()
        elif entity in self._logs:
            yield from self._logs[entity].events()

    @property
    def entities(self) -> list[str]:
        return sorted(self._logs)

    def entity_log(self, entity: str) -> EntityLog | None:
        return self._logs.get(entity)

    def link(self, effect: str, feature: str, entity: str | None = None) -> BetaCounts:
        """Effective link belief for ``feature`` (global view if no entity)."""
        return self.effect(effect).links.get(feature, entity)

    def _applied_rules(self, spec: EffectSpec, event: Event) -> list[Rule]:
        return [r for r in spec.rules if r.matches(event)]

    def score(self, event: Event, effect: str, at: float) -> float:
        """Score of a single event for an effect at time ``at``."""
        spec = self.effect(effect)
        if not spec.type_allowed(event.type):
            return 0.0
        dt = at - event.time
        if dt < 0:  # temporal sanity, whatever the kernel says
            return 0.0
        k = spec.kernel.value(dt)
        if k == 0.0:
            return 0.0
        s = k * event.weight * spec.links.noisy_or(event.features, event.entity)
        for r in spec.rules:
            if s == 0.0:
                break
            if r.matches(event):
                s *= r.multiplier
        return s

    def explain(self, event: Event, effect: str, at: float) -> Explanation:
        """Break an event's score into timing, weight, links and rules."""
        spec = self.effect(effect)
        dt = at - event.time
        contribs = []
        for f in event.features:
            c = spec.links.get(f, event.entity)
            contribs.append(FeatureContribution(f, c.strength, c.alpha, c.beta))
        contribs.sort(key=lambda c: (-c.strength, c.feature))
        groups = self._inseparable_groups(event)
        return Explanation(
            dt=dt,
            kernel_value=spec.kernel.value(dt) if dt >= 0 else 0.0,
            weight=event.weight,
            link_strength=spec.links.noisy_or(event.features, event.entity),
            features=tuple(contribs),
            warnings=tuple(self._confounding_warnings(event, spec, groups)),
            rules=tuple(AppliedRule(r.name, r.multiplier)
                        for r in self._applied_rules(spec, event)),
            joint_causes=tuple(self._joint_causes(event, spec, groups)),
            entity=event.entity,
        )

    # -- co-occurrence / joint causes ----------------------------------------

    def _inseparable_groups(self, event: Event) -> list[tuple[frozenset[str], int]]:
        """Groups (size >= 2) of the event's features that have always
        co-occurred for its entity, with the number of events they share.

        Features whose posting lists (entity-wide inverted index) are
        *identical* form one group.
        """
        log = self._logs.get(event.entity)
        if log is None or len(event.features) < 2:
            return []
        groups: dict[tuple[int, ...], list[str]] = {}
        for f in sorted(event.features):
            groups.setdefault(tuple(log.event_ids_with_feature(f)), []).append(f)
        return [(frozenset(fs), len(ids)) for ids, fs in groups.items() if len(fs) >= 2]

    def _has_feedback(self, spec: EffectSpec, entity: str, features: Iterable[str]) -> bool:
        own = spec.links.entity_counts(entity)
        return any(f in own for f in features)

    def _joint_causes(
        self, event: Event, spec: EffectSpec, groups: list[tuple[frozenset[str], int]]
    ) -> list[JointCause]:
        out: list[JointCause] = []
        seen: set[frozenset[str]] = set()
        for fs, _n in groups:
            joint = spec.links.joint(event.entity, fs)
            if joint is None and not self._has_feedback(spec, event.entity, fs):
                continue
            seen.add(fs)
            out.append(JointCause(tuple(sorted(fs)),
                                  spec.links.noisy_or(fs, event.entity), joint, True))
        # Joint links learned earlier for subsets that are now separable.
        for fs in spec.links.joint_sets(event.entity):
            if fs in seen or not fs <= event.features:
                continue
            out.append(JointCause(tuple(sorted(fs)), spec.links.noisy_or(fs, event.entity),
                                  spec.links.joint(event.entity, fs), False))
        out.sort(key=lambda j: (-j.strength, j.features))
        return out

    def _confounding_warnings(
        self, event: Event, spec: EffectSpec,
        groups: list[tuple[frozenset[str], int]], min_support: int = 2,
    ) -> list[str]:
        """Flag groups of this event's features that *always* appear together
        and have seen feedback: their learned strengths are ambiguous."""
        warnings = []
        for fs, n in groups:
            if n < min_support or not self._has_feedback(spec, event.entity, fs):
                continue
            names = ", ".join(f"'{f}'" for f in sorted(fs))
            warnings.append(
                f"{names} always co-occur ({n} events): feedback cannot separate "
                f"their effects yet; reported as joint cause {'+'.join(sorted(fs))}"
            )
        return warnings

    # -- the main query -----------------------------------------------------

    def _window(self, spec: EffectSpec, at: float, within: float | None) -> tuple[float, float]:
        """Event-time interval that can possibly contribute to the effect."""
        lookback = spec.kernel.support if within is None else min(within, spec.kernel.support)
        return at - lookback, at - max(spec.kernel.min_lag, 0.0)

    def segment_bound(
        self,
        spec: EffectSpec,
        seg: Segment,
        entity: str,
        at: float,
        t_lo: float,
        t_hi: float,
        use_sets: bool = True,
        _cache: dict[frozenset[str], float] | None = None,
    ) -> tuple[float, bool]:
        """Upper bound on the score of any event of ``seg``; and whether the
        loose union bound was used. See the module docstring."""
        if spec.cause_types is not None and not (seg.types & spec.cause_types):
            return 0.0, False
        dt_lo = at - min(seg.max_time, t_hi)
        dt_hi = at - max(seg.min_time, t_lo)
        kb = spec.kernel.max_over(dt_lo, dt_hi) if dt_lo <= dt_hi else 0.0
        if kb <= 0.0:
            return 0.0, False
        links = spec.links
        if use_sets and seg.sets is not None:
            cache = {} if _cache is None else _cache
            best = 0.0
            for (type_, feats), w in seg.sets.items():
                if not spec.type_allowed(type_):
                    continue
                rb = spec.rule_bound(type_)
                if w * rb <= best:
                    continue  # cannot beat the current best even with link = 1
                lk = cache.get(feats)
                if lk is None:
                    lk = cache[feats] = links.noisy_or(feats, entity)
                best = max(best, w * lk * rb)
            return kb * best, False
        rb = max((spec.rule_bound(t) for t in seg.types if spec.type_allowed(t)), default=0.0)
        return kb * seg.max_weight * rb * links.noisy_or(seg.feature_counts, entity), True

    def top_causes(
        self,
        entity: str,
        effect: str,
        at: float,
        k: int = 3,
        explain: bool = False,
        within: float | None = None,
        _use_sets: bool = True,
    ) -> QueryResult:
        """Return the ``k`` most likely causes of ``effect`` for ``entity`` at ``at``.

        Args:
            within: Optional extra cap (seconds) on lookback, applied on top of
                the kernel's support window.
            explain: Attach an :class:`Explanation` to each cause.
        """
        if k <= 0:
            raise ValueError("k must be >= 1")
        spec = self.effect(effect)
        result = QueryResult(entity=entity, effect=effect, at=at, entities=(entity,))
        stats = result.stats
        log = self._logs.get(entity)
        if log is None:
            return result

        t_lo, t_hi = self._window(spec, at, within)
        segments = log.segments_between(t_lo, t_hi)
        stats.segments_in_window = len(segments)

        # Step 1: bound every segment from its summaries + current links.
        cache: dict[frozenset[str], float] = {}
        bounded = []
        for seg in segments:
            bound, union = self.segment_bound(spec, seg, entity, at, t_lo, t_hi,
                                              use_sets=_use_sets, _cache=cache)
            stats.union_fallbacks += union
            bounded.append((bound * (1.0 + _BOUND_SLACK), seg))

        # Step 2: visit segments best-bound-first (ties by start: deterministic).
        bounded.sort(key=lambda bs: (-bs[0], bs[1].start))

        # Bounded min-heap of (score, event_id, event). Smallest at [0], so
        # it is the k-th best so far and the one to evict. Ties are broken by
        # event id: the later event wins, as in the brute-force reference.
        heap: list[tuple[float, int, Event]] = []
        for i, (bound, seg) in enumerate(bounded):
            # Threshold check: nothing in this or any later segment can beat
            # (or tie) our current k-th best, so stop.
            if bound <= 0.0 or (len(heap) == k and heap[0][0] >= bound):
                stats.segments_pruned += len(bounded) - i
                break
            stats.segments_scanned += 1
            for ev in seg.events:
                if not (t_lo <= ev.time <= t_hi):
                    continue
                stats.events_scanned += 1
                s = self.score(ev, effect, at)
                if s <= 0.0:
                    continue
                item = (s, ev.id, ev)
                if len(heap) < k:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)

        ranked = sorted(heap, key=lambda it: (-it[0], -it[1]))
        result.causes = [
            Cause(rank=i + 1, event=ev, score=s,
                  explanation=self.explain(ev, effect, at) if explain else None)
            for i, (s, _, ev) in enumerate(ranked)
        ]
        return result

    def top_causes_group(
        self,
        entities: Iterable[str],
        effect: str,
        at: float,
        k: int = 3,
        explain: bool = False,
        within: float | None = None,
    ) -> QueryResult:
        """Top-k causes of one effect across several entities.

        Runs the per-entity top-k and merges the sorted lists with a k-way
        merge (``heapq.merge``). Exact with respect to brute force over the
        union of the entities' events. Only a single effect is allowed:
        scores of different effects live on different scales.

        Caveat: scores are only comparable *across entities* to the extent
        they are calibrated (see :meth:`calibration_report`). Entities with
        more feedback have sharper links, and weights (doses) may be on
        different scales per entity; treat cross-entity ranking as a
        triage aid until calibration says otherwise.
        """
        if not isinstance(effect, str):
            effs = list(effect)  # type: ignore[arg-type]
            if len(set(effs)) != 1:
                raise ValueError("top_causes_group takes a single effect; "
                                 f"mixing effects {sorted(set(effs))} is not supported")
            effect = effs[0]
        ents = list(dict.fromkeys(entities))
        if not ents:
            raise ValueError("need at least one entity")
        per = [self.top_causes(e, effect, at, k=k, explain=explain, within=within) for e in ents]
        result = QueryResult(entity=",".join(ents), effect=effect, at=at, entities=tuple(ents))
        for r in per:
            result.stats.add(r.stats)
        merged = heapq.merge(*(r.causes for r in per),
                             key=lambda c: (-c.score, -c.event.id))
        result.causes = [
            Cause(rank=i + 1, event=c.event, score=c.score, explanation=c.explanation)
            for i, c in zip(range(k), merged)
        ]
        return result

    def top_causes_naive(
        self, entity: str, effect: str, at: float, k: int = 3, within: float | None = None
    ) -> list[tuple[float, Event]]:
        """Reference brute force: score everything, sort, cut. For testing."""
        spec = self.effect(effect)
        t_lo, t_hi = self._window(spec, at, within)
        scored = [
            (self.score(ev, effect, at), ev)
            for ev in self.events(entity)
            if t_lo <= ev.time <= t_hi
        ]
        scored = [(s, ev) for s, ev in scored if s > 0.0]
        scored.sort(key=lambda se: (-se[0], -se[1].id))
        return scored[:k]

    def most_likely_cause(self, entity: str, effect: str, at: float) -> Cause | None:
        """Pop the top: the single most likely cause right now."""
        return self.top_causes(entity, effect, at, k=1, explain=True).top

    # -- prospective risk: candidates that have not happened yet ------------

    def _transient_event(self, entity: str | None, type: str | None,
                         features: Iterable[str] | str | None, weight: float | None,
                         attrs: Mapping[str, Any] | None) -> Event:
        """An Event that is *not* stored (id 0, time NaN), used to evaluate
        rules and co-occurrence for a candidate."""
        attrs = dict(attrs or {})
        if weight is None:
            fn = self._event_types.get(type) if type is not None else None
            weight = fn(attrs) if fn is not None else 1.0
        return Event(id=0, entity=entity if entity is not None else "", time=math.nan,
                     type=type if type is not None else "",
                     features=as_feature_set(features), weight=validate_weight(weight),
                     attrs=attrs)

    def _risk(self, spec: EffectSpec, ev: Event, entity: str | None, type_given: bool) -> float:
        if type_given and not spec.type_allowed(ev.type):
            return 0.0
        s = ev.weight * spec.links.noisy_or(ev.features, entity)
        for r in spec.rules:
            if s == 0.0:
                break
            if r.matches(ev):
                s *= r.multiplier
        return s

    def _risk_explanation(self, spec: EffectSpec, ev: Event, entity: str | None,
                          type_given: bool) -> Explanation:
        contribs = []
        for f in ev.features:
            c = spec.links.get(f, entity)
            contribs.append(FeatureContribution(f, c.strength, c.alpha, c.beta))
        contribs.sort(key=lambda c: (-c.strength, c.feature))
        rules = [AppliedRule(r.name, r.multiplier) for r in self._applied_rules(spec, ev)]
        warnings: list[str] = []
        joint: list[JointCause] = []
        if type_given and not spec.type_allowed(ev.type):
            rules.insert(0, AppliedRule(f"type {ev.type!r} not in cause_types", 0.0))
        if entity is not None:
            # Historical co-occurrence of the candidate's features for this
            # entity (the candidate itself is not in the log).
            groups = [(fs, n) for fs, n in self._inseparable_groups(ev) if n > 0]
            warnings = self._confounding_warnings(ev, spec, groups)
            joint = self._joint_causes(ev, spec, groups)
        return Explanation(
            dt=0.0, kernel_value=1.0, weight=ev.weight,
            link_strength=spec.links.noisy_or(ev.features, entity),
            features=tuple(contribs), warnings=tuple(warnings), rules=tuple(rules),
            joint_causes=tuple(joint), entity=ev.entity, prospective=True,
        )

    def feature_risk(
        self,
        effect: str,
        features: Iterable[str] | str,
        weight: float | None = 1.0,
        entity: str | None = None,
        type: str | None = None,
        attrs: Mapping[str, Any] | None = None,
        explain: bool = False,
    ) -> float | tuple[float, Explanation]:
        """Risk that a candidate which has **not happened yet** causes ``effect``.

        This is the *priority* half of attribution: ``top_causes`` asks
        "which past event caused this?", ``feature_risk`` asks "if this
        happens (merges, ships, is eaten), how likely is it to cause that?".
        Typical use: rank open pull requests by the risk of an incident
        *before* they merge, from the links learned from earlier
        confirmations (reverts) and rejections::

            risk = weight * noisy_OR_f( strength_entity(f, effect) ) * rules(event)

        It is ``score`` without the kernel term (there is no lag yet), so for
        a stored event ``e``: ``score(e, effect, at) ==
        kernel(at - e.time) * feature_risk(effect, e.features, e.weight,
        e.entity, e.type, e.attrs)`` (for rules that do not look at time).

        Args:
            features: The candidate's features (set or comma-separated string).
            weight: Dose (> 0). ``None`` derives it from ``attrs`` with the
                type's registered weight function (else 1.0).
            entity: Whose links to use (entity layer + shrunk global evidence
                + registered priors). ``None`` uses the global (population) view.
            type: Event type. If given, ``cause_types`` applies (a disallowed
                type scores 0) and type-scoped rules can match. If omitted,
                the type filter is skipped and only untyped rules can match.
            attrs: Attributes visible to rule predicates (and the weight fn).
            explain: Also return an :class:`Explanation` (``prospective=True``,
                ``kernel_value == 1``) with per-feature links, applied rules and
                joint causes from the entity's history.

        Rules are evaluated against a transient, unstored event (``id=0``,
        ``time=nan``): predicates that look at ``time`` will not match.

        Returns:
            The risk score, or ``(score, explanation)`` if ``explain``.
        """
        spec = self.effect(effect)
        ev = self._transient_event(entity, type, features, weight, attrs)
        s = self._risk(spec, ev, entity, type is not None)
        if explain:
            return s, self._risk_explanation(spec, ev, entity, type is not None)
        return s

    def rank_candidates(
        self,
        effect: str,
        candidates: Iterable[Candidate | Mapping[str, Any]],
        k: int = 10,
        entity: str | None = None,
        explain: bool = False,
    ) -> list[RankedCandidate]:
        """Top-``k`` not-yet-happened candidates by :meth:`feature_risk`.

        ``candidates`` are :class:`Candidate` objects or dicts with the same
        keys (``id``, ``features``, ``weight``, ``type``, ``attrs``,
        ``entity``). A candidate's own ``entity`` overrides the ``entity``
        argument, so one call can rank PRs across several services (with the
        same calibration caveat as :meth:`top_causes_group`). Candidates
        scoring 0 (hard-blocked by a rule or type filter) are dropped, as in
        ``top_causes``. Uses a bounded min-heap of size ``k`` over a single
        pass; ties keep the earlier candidate. Explanations are built only for
        the returned rows.
        """
        if k <= 0:
            raise ValueError("k must be >= 1")
        spec = self.effect(effect)
        # Min-heap of (score, -index, ...): the root is the current k-th best;
        # on ties the earlier candidate (smaller index) ranks higher.
        heap: list[tuple[float, int, Candidate, str | None, Event]] = []
        for i, raw in enumerate(candidates):
            c = Candidate.from_any(raw, i)
            ent = c.entity if c.entity is not None else entity
            ev = self._transient_event(ent, c.type, c.features, c.weight, c.attrs)
            s = self._risk(spec, ev, ent, c.type is not None)
            if s <= 0.0:
                continue
            item = (s, -i, c, ent, ev)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
        ranked = sorted(heap, key=lambda it: (-it[0], -it[1]))
        return [
            RankedCandidate(
                rank=r + 1, candidate=c, score=s, entity=ent,
                explanation=(self._risk_explanation(spec, ev, ent, c.type is not None)
                             if explain else None))
            for r, (s, _, c, ent, ev) in enumerate(ranked)
        ]

    # -- feedback -----------------------------------------------------------

    def feedback(
        self,
        entity: str,
        effect_type: str,
        at_time: float,
        confirmed_event_id: int | None = None,
        rejected_ids: Iterable[int] = (),
        credit: str | None = None,
    ) -> FeedbackReport:
        """Teach the database which candidate really was the cause.

        * Confirmed event: proportional (EM) credit by default -- each
          feature gets ``alpha += r_f`` and ``beta += 1 - r_f`` with
          ``r_f = s_f / noisy_or(F)``; ``credit="uniform"`` (or the effect's
          ``credit`` setting) gives every feature ``alpha += 1``.
        * Rejected events: every feature gets ``beta += 1``.

        Updates go to the entity's links and the global layer. Inseparable
        feature groups of the judged events get a joint link. The scores the
        judged events had *before* the update are recorded for calibration.
        A cause must belong to ``entity`` and precede ``at_time``.
        """
        spec = self.effect(effect_type)
        mode = credit or spec.credit
        if mode not in CREDIT_MODES:
            raise ValueError(f"credit must be one of {CREDIT_MODES}, got {mode!r}")

        def checked(eid: int) -> Event:
            ev = self._events.get(int(eid))
            if ev is None:
                raise KeyError(f"unknown event id {eid}")
            if ev.entity != entity:
                raise ValueError(f"event {eid} belongs to {ev.entity!r}, not {entity!r}")
            if ev.time > at_time:
                raise ValueError(f"event {eid} happened after the effect")
            return ev

        # Validate everything before mutating anything.
        confirmed = checked(confirmed_event_id) if confirmed_event_id is not None else None
        rejected = [checked(eid) for eid in rejected_ids]
        if confirmed is not None and any(r.id == confirmed.id for r in rejected):
            raise ValueError(f"event {confirmed.id} cannot be both confirmed and rejected")

        # Calibration: the scores as they were shown, before learning.
        records = self._calibration.setdefault(effect_type, [])
        for ev, ok in ([(confirmed, True)] if confirmed else []) + [(r, False) for r in rejected]:
            records.append(CalibrationRecord(entity, ev.id, float(at_time),
                                             self.score(ev, effect_type, at_time), ok))

        links = spec.links
        credit_map: dict[str, float] = {}
        if confirmed is not None:
            groups = self._inseparable_groups(confirmed)
            p_all = links.noisy_or(confirmed.features, entity)
            group_p = {fs: links.noisy_or(fs, entity) for fs, _ in groups}
            credit_map = links.confirm(confirmed.features, entity=entity, credit=mode)
            for fs, _ in groups:
                r = 1.0 if mode == "uniform" or p_all <= 0 else min(1.0, group_p[fs] / p_all)
                links.update_joint(entity, fs, r, 1.0 - r)
        for ev in rejected:
            for fs, _ in self._inseparable_groups(ev):
                links.update_joint(entity, fs, 0.0, 1.0)
            links.reject(ev.features, entity=entity)
        return FeedbackReport(confirmed.id if confirmed else None,
                              tuple(r.id for r in rejected), credit_map)

    # -- observational learning (v0.4) ----------------------------------------

    def _window_events(self, spec: EffectSpec, entity: str, at: float) -> list[Event]:
        log = self._logs.get(entity)
        if log is None:
            return []
        t_lo, t_hi = self._window(spec, at, None)
        return [ev for seg in log.segments_between(t_lo, t_hi) for ev in seg.events
                if t_lo <= ev.time <= t_hi]

    def _shares(self, spec: EffectSpec, ev: Event) -> dict[str, float]:
        """Per-feature share of an event's credit (mirrors :meth:`feedback`)."""
        if spec.credit == "uniform":
            return {f: 1.0 for f in ev.features}
        return responsibilities(spec.links.strengths(sorted(ev.features), ev.entity))

    def observe(self, entity: str, effect: str, at: float,
                leak: float | None = None) -> ObserveReport:
        """Record an occurrence of ``effect`` with **no labelled cause**.

        Observational (implicit) learning by one soft-EM step. Every candidate
        event ``e`` in the kernel window is scored exactly as in
        :meth:`top_causes` (kernel x dose x noisy-OR links x rules) and gets
        the responsibility::

            r_e = score_e / (leak + sum_e' score_e')

        where ``leak`` (default: the effect's ``leak``) is the background
        rate: "it happened for no event in the window". Each feature of ``e``
        then gets proportional credit scaled by ``r_e`` --
        ``alpha_f += r_e * share_f`` and ``beta_f += r_e * (1 - share_f)``,
        with ``share_f = s_f / noisyOR(F)`` as in :meth:`feedback` (``1`` for
        ``credit="uniform"``) -- on both the entity and the global layer.
        Scores and shares are computed before any update.

        A large leak means nothing is learned (``leak -> inf`` gives
        ``r_e -> 0``). Credited events are remembered and are never rejected
        by :meth:`settle` for this effect; if ``settle(partial=True)`` had
        already added negative evidence to one, it is undone first.

        This learns **associations under the model**, not causal effects:
        a feature that co-occurs with a true cause still receives some soft
        credit (see the README section "Observational learning").
        """
        spec = self.effect(effect)
        lk = spec.leak if leak is None else _check_leak(leak)
        at = float(at)
        obs = spec.obs
        scored = []
        for ev in self._window_events(spec, entity, at):
            s = self.score(ev, effect, at)
            if s > 0.0:
                scored.append((ev, s, self._shares(spec, ev)))
        total = lk + sum(s for _, s, _ in scored)
        if math.isinf(lk):  # infinite background: nothing is learned
            scored = []
        credits = []
        links = spec.links
        for ev, s, shares in scored:
            r = s / total
            undo = obs.rejected.pop(ev.id, 0.0)
            if undo:
                for f in ev.features:
                    links.update(f, 0.0, -undo, entity)
                obs.stats["rejections_undone"] += 1
            added = {}
            for f in sorted(ev.features):
                sh = shares[f]
                links.update(f, r * sh, r * (1.0 - sh), entity)
                added[f] = r * sh
            obs.credited[ev.id] = obs.credited.get(ev.id, 0.0) + r
            credits.append(EventCredit(ev.id, s, r, added))
        credits.sort(key=lambda c: (-c.responsibility, -c.event_id))
        bisect.insort(obs.occurrences.setdefault(entity, []), at)
        leak_r = 1.0 if math.isinf(lk) else lk / total
        obs.stats["occurrences"] += 1
        obs.stats["leak_responsibility"] += leak_r
        obs.stats["credited_mass"] += 1.0 - leak_r
        return ObserveReport(entity, effect, at, lk, leak_r, tuple(credits))

    def _occurred_in_window(self, spec: EffectSpec, ev: Event) -> bool:
        occ = spec.obs.occurrences.get(ev.entity)
        if not occ:
            return False
        lo = ev.time + max(spec.kernel.min_lag, 0.0)
        hi = ev.time + spec.kernel.support
        i = bisect.bisect_left(occ, lo)
        return i < len(occ) and occ[i] <= hi

    def settle(self, until: float, effects: Iterable[str] | str | None = None,
               entity: str | None = None, partial: bool = False) -> SettleReport:
        """Add negative evidence for events whose window passed without the effect.

        For every event ``e`` (of an allowed cause type) whose kernel support
        window ``[t_e + min_lag, t_e + support]`` has fully elapsed by
        ``until`` and in which ``effect`` was never observed for ``e``'s entity
        (via :meth:`observe`), each feature gets ``beta_f += w_neg * rules(e)``
        (the effect's ``w_neg``, scaled by the event's rule multiplier; events
        blocked by a rule are skipped). Events that received soft credit from
        :meth:`observe` are never rejected for that effect. For a one-shot
        effect (``repeatable=False``) events at or after the entity's first
        occurrence are skipped too.

        Incremental: a per-(entity, effect) cursor records the event-time
        watermark already settled, so each call only looks at events newer
        than the cursor (plus events inserted late, behind it). Calling it
        twice with the same ``until`` is a no-op.

        With ``partial=True``, events whose window has only *partly* elapsed
        also get negative evidence, scaled by the fraction of the kernel's
        mass already covered (``w_neg * mass_fraction(until - t_e)``), topped
        up on later calls; a later :meth:`observe` in the window undoes it.

        Args:
            effects: one effect name, several, or None for all effects.
            entity: restrict to one entity (default: all).
        """
        until = float(until)
        if effects is None:
            names = self.effects
        elif isinstance(effects, str):
            names = [effects]
        else:
            names = list(effects)
        ents = [entity] if entity is not None else list(self._logs)
        n_settled = n_rej = n_cred = n_obs = 0
        w_total = 0.0
        per: dict[str, int] = {}
        for name in names:
            spec = self.effect(name)
            obs = spec.obs
            per[name] = 0
            for ent in ents:
                log = self._logs.get(ent)
                if log is None:
                    continue
                cur = obs.cursors.get(ent, -math.inf)
                new_cur = max(cur, until - spec.kernel.support)
                todo: list[tuple[Event, float]] = []
                late = obs.late.pop(ent, [])
                for eid in late:
                    ev = self._events.get(eid)
                    if ev is not None:
                        todo.append((ev, 1.0))
                if new_cur > cur:
                    lo = cur if math.isfinite(cur) else -math.inf
                    segs = (log.segments() if not math.isfinite(lo)
                            else log.segments_between(lo, new_cur))
                    for seg in segs:
                        for ev in seg.events:
                            if lo < ev.time <= new_cur:
                                todo.append((ev, 1.0))
                if partial:
                    for seg in log.segments_between(new_cur, until):
                        for ev in seg.events:
                            if new_cur < ev.time <= until:
                                todo.append((ev, spec.kernel.mass_fraction(until - ev.time)))
                obs.cursors[ent] = new_cur
                for ev, frac in todo:
                    if not spec.type_allowed(ev.type):
                        continue
                    m = spec.rule_multiplier(ev)
                    if m <= 0.0:
                        continue
                    if frac >= 1.0:
                        n_settled += 1
                        obs.stats["events_settled"] += 1
                    if ev.id in obs.credited:
                        if frac >= 1.0:
                            n_cred += 1
                            obs.stats["skipped_credited"] += 1
                        continue
                    if not spec.repeatable:
                        occ = obs.occurrences.get(ev.entity)
                        if occ and ev.time >= occ[0]:
                            if frac >= 1.0:
                                n_obs += 1
                                obs.stats["skipped_terminal"] += 1
                            continue
                    if self._occurred_in_window(spec, ev):
                        if frac >= 1.0:
                            n_obs += 1
                            obs.stats["skipped_observed"] += 1
                        continue
                    target = spec.w_neg * m * frac
                    w = target - obs.rejected.get(ev.id, 0.0)
                    if w <= 1e-12:
                        continue
                    for f in ev.features:
                        spec.links.update(f, 0.0, w, ev.entity)
                    obs.rejected[ev.id] = target
                    w_total += w
                    obs.stats["rejected_weight"] += w
                    if frac >= 1.0:
                        n_rej += 1
                        per[name] += 1
                        obs.stats["events_rejected"] += 1
        return SettleReport(until, n_settled, n_rej, w_total, n_cred, n_obs, per)

    def observation_stats(self, effect: str) -> dict[str, float]:
        """Running counters of observational learning for ``effect``.

        Besides the raw counters (see ``ObservationState``) it reports
        ``mean_leak_responsibility``: the average share of an observed
        occurrence attributed to the background. Close to 1 means the events
        in the window rarely explain the effect (consider a smaller ``leak``,
        better features or a different kernel); close to 0 means almost every
        occurrence is explained by some event.
        """
        st = dict(self.effect(effect).obs.stats)
        n = st["occurrences"]
        st["mean_leak_responsibility"] = st["leak_responsibility"] / n if n else math.nan
        return st

    def link_report(
        self,
        effect: str,
        entity: str | None = None,
        min_support: float = 1.0,
        level: float = 0.9,
        baseline: float | str = "mean",
        features: Iterable[str] | None = None,
    ) -> LinkReport:
        """Features ranked by mean link strength for ``effect``.

        Each row has the Beta posterior mean, ``alpha``, ``beta``, the
        effective evidence ``n`` (own + borrowed evidence, excluding the
        prior), an equal-tailed ``level`` credible interval (default 90%)
        and ``lift = strength / baseline``. ``baseline`` is ``"mean"`` (mean
        strength of the reported features, default), ``"prior"`` (the
        effect's default prior mean) or a number.

        Only features with ``n >= min_support`` are reported. ``entity=None``
        gives the global (population) view.

        **These are associations under the model, not causal effects.**
        With observational learning, a feature that merely co-occurs with a
        real cause (a confounder: e.g. power users both use feature X and
        convert more) also gets credit, and selection, novelty and survivor
        effects are not modelled. Use the report to choose what to test
        (e.g. an A/B experiment), not as proof.
        """
        spec = self.effect(effect)
        links = spec.links
        if features is not None:
            feats = set(features)
        else:
            feats = set(links._priors) | set(links._global)
            if entity is not None:
                feats |= set(links.entity_counts(entity))
        rows = []
        for f in sorted(feats):
            b = links.get(f, entity)
            n = b.observations
            if n < min_support:
                continue
            lo, hi = beta_interval(b.alpha, b.beta, level)
            rows.append([f, b.strength, b.alpha, b.beta, n, lo, hi])
        if isinstance(baseline, str):
            if baseline == "mean":
                base = sum(r[1] for r in rows) / len(rows) if rows else math.nan
            elif baseline == "prior":
                a, bb = links.default_prior
                base = a / (a + bb)
            else:
                raise ValueError("baseline must be 'mean', 'prior' or a number")
        else:
            base = float(baseline)
        out = tuple(
            LinkRow(f, s, a, bb, n, lo, hi, s / base if base and base > 0 else math.nan)
            for f, s, a, bb, n, lo, hi in sorted(rows, key=lambda r: (-r[1], r[0]))
        )
        return LinkReport(effect, entity, base, level, out)

    # -- calibration --------------------------------------------------------

    def calibration_records(self, effect: str) -> list[CalibrationRecord]:
        self.effect(effect)
        return list(self._calibration.get(effect, ()))

    def calibration_report(self, effect: str, bins: int = 10) -> CalibrationReport:
        """Mean predicted score vs observed confirm rate per score bin.

        Scores are clipped to [0, 1] (a score is kernel x dose x link x
        rules, so heavy doses can exceed 1). Bins are equal-width on [0, 1].
        No automatic recalibration is done yet (roadmap).
        """
        if bins < 1:
            raise ValueError("bins must be >= 1")
        recs = self.calibration_records(effect)
        acc = [[0, 0.0, 0] for _ in range(bins)]  # count, sum_pred, confirmed
        sq = 0.0
        for r in recs:
            p = r.predicted
            i = min(bins - 1, int(p * bins))
            acc[i][0] += 1
            acc[i][1] += p
            acc[i][2] += r.confirmed
            sq += (p - r.confirmed) ** 2
        out = tuple(
            CalibrationBin(i / bins, (i + 1) / bins, n,
                           s / n if n else None, c / n if n else None)
            for i, (n, s, c) in enumerate(acc)
        )
        return CalibrationReport(effect, len(recs), sq / len(recs) if recs else None, out)

    # -- privacy ------------------------------------------------------------

    def forget_entity(self, entity: str, scrub_global: bool = False) -> dict[str, int]:
        """Remove everything stored about ``entity``.

        Drops its events, segments, entity-level and joint links, and
        calibration records. The global link layer keeps the entity's past
        contributions unless ``scrub_global=True``, in which case they are
        subtracted exactly (the entity layer records them). Returns counts
        of what was removed.
        """
        log = self._logs.pop(entity, None)
        n_events = n_segments = 0
        if log is not None:
            n_segments = log.num_segments
            for ev in log.events():
                self._events.pop(ev.id, None)
                n_events += 1
        n_links = sum(spec.links.forget_entity(entity, scrub_global=scrub_global)
                      for spec in self._effects.values())
        for spec in self._effects.values():
            obs = spec.obs
            obs.occurrences.pop(entity, None)
            obs.cursors.pop(entity, None)
            obs.late.pop(entity, None)
            if log is not None:
                for ev in log.events():
                    obs.credited.pop(ev.id, None)
                    obs.rejected.pop(ev.id, None)
        n_cal = 0
        for eff, recs in self._calibration.items():
            keep = [r for r in recs if r.entity != entity]
            n_cal += len(recs) - len(keep)
            self._calibration[eff] = keep
        return {"events": n_events, "segments": n_segments,
                "links": n_links, "calibration_records": n_cal}

    # -- persistence --------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a JSONL snapshot.

        One meta line, one line per effect (kernel, links incl. global and
        joint layers, rules metadata), one line per event, one per
        calibration record. Python callables (predicate rules, weight
        functions) are *not* serialised: only their names are kept, and they
        must be re-registered after :meth:`load`. Type-level rules (no
        predicate) are restored automatically.
        """
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "kind": "meta", "version": SNAPSHOT_VERSION,
                "bucket_seconds": self.bucket_seconds,
                "max_sets_per_segment": self.max_sets_per_segment,
                "next_id": self._next_id,
                "event_types": {t: fn is not None for t, fn in sorted(self._event_types.items())},
            }) + "\n")
            for spec in self._effects.values():
                fh.write(json.dumps({
                    "kind": "effect",
                    "name": spec.name,
                    "kernel": spec.kernel.to_dict(),
                    "links": spec.links.to_dict(),
                    "cause_types": sorted(spec.cause_types) if spec.cause_types else None,
                    "credit": spec.credit,
                    "rules": [{"name": r.name, "multiplier": r.multiplier,
                               "types": sorted(r.types) if r.types is not None else None,
                               "predicate": r.predicate is not None}
                              for r in spec.rules],
                    "pending_rules": spec.pending_rules,
                    "leak": spec.leak,
                    "w_neg": spec.w_neg,
                    "repeatable": spec.repeatable,
                    "observations": spec.obs.to_dict(),
                }) + "\n")
            for ev in sorted(self._events.values(), key=lambda e: e.id):
                fh.write(json.dumps({"kind": "event", **ev.to_dict()}) + "\n")
            for eff, recs in sorted(self._calibration.items()):
                for r in recs:
                    fh.write(json.dumps({"kind": "calibration", "effect": eff,
                                         "entity": r.entity, "event_id": r.event_id,
                                         "at": r.at, "score": r.score,
                                         "confirmed": r.confirmed}) + "\n")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "HeapDB":
        """Rebuild a database (segments, indexes, links, calibration) from a
        snapshot. Reads v1, v2 and v3 snapshots."""
        db: HeapDB | None = None
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                kind = rec.get("kind")
                if kind == "meta":
                    db = cls(bucket_seconds=rec["bucket_seconds"],
                             max_sets_per_segment=rec.get("max_sets_per_segment",
                                                          DEFAULT_MAX_SETS))
                    db._next_id = int(rec["next_id"])
                    for t, has_fn in rec.get("event_types", {}).items():
                        db._event_types[t] = None
                        if has_fn:
                            db.pending_event_types.add(t)
                    continue
                if db is None:
                    raise ValueError(f"{path}:{lineno}: snapshot must start with a meta line")
                if kind == "effect":
                    ct = rec.get("cause_types")
                    spec = EffectSpec(
                        name=rec["name"],
                        kernel=kernel_from_dict(rec["kernel"]),
                        links=LinkTable.from_dict(rec["links"]),
                        cause_types=frozenset(ct) if ct else None,
                        credit=rec.get("credit", "proportional"),
                        pending_rules=list(rec.get("pending_rules", [])),
                        leak=float(rec.get("leak", DEFAULT_LEAK)),
                        w_neg=float(rec.get("w_neg", 1.0)),
                        repeatable=bool(rec.get("repeatable", True)),
                        obs=ObservationState.from_dict(rec.get("observations")),
                    )
                    for r in rec.get("rules", []):
                        if r["predicate"]:
                            spec.pending_rules.append(r["name"])
                        else:
                            types = r.get("types")
                            spec.rules.append(Rule(r["name"], None, float(r["multiplier"]),
                                                   frozenset(types) if types is not None else None))
                    db._effects[rec["name"]] = spec
                elif kind == "event":
                    ev = Event.from_dict(rec)
                    db._append(ev)
                    db._next_id = max(db._next_id, ev.id + 1)
                elif kind == "calibration":
                    db._calibration.setdefault(rec["effect"], []).append(CalibrationRecord(
                        rec["entity"], int(rec["event_id"]), float(rec["at"]),
                        float(rec["score"]), bool(rec["confirmed"])))
                else:
                    raise ValueError(f"{path}:{lineno}: unknown record kind {kind!r}")
        if db is None:
            raise ValueError(f"{path}: empty snapshot")
        return db

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        return (f"HeapDB(events={len(self)}, entities={len(self._logs)}, "
                f"effects={self.effects}, bucket_seconds={self.bucket_seconds:g})")


def _check_leak(leak: float) -> float:
    lk = float(leak)
    if not lk > 0.0:  # also rejects NaN
        raise ValueError(f"leak must be > 0 (inf disables observational learning), got {leak!r}")
    return lk


def _check_w_neg(w: float) -> float:
    w = float(w)
    if not (w >= 0.0 and math.isfinite(w)):
        raise ValueError(f"w_neg must be a finite number >= 0, got {w!r}")
    return w


def half_life_to_lambda(half_life: float) -> float:
    """Convenience: decay rate for a given half-life (seconds)."""
    return math.log(2.0) / half_life
