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

import heapq
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping

from .events import Event, as_feature_set, validate_weight
from .kernels import Kernel, kernel_from_dict
from .links import CREDIT_MODES, BetaCounts, LinkTable, _fmt
from .segments import DEFAULT_MAX_SETS, EntityLog, Segment

# Relative slack applied to segment bounds before comparing them with real
# scores. Mathematically bound >= score; this guards against last-bit floating
# point differences (e.g. noisy-OR over a union versus a subset in a different
# order), so we never prune an event we should have kept.
_BOUND_SLACK = 1e-9

SNAPSHOT_VERSION = 2


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

    def rule_bound(self, type_: str) -> float:
        """Upper bound of the rule product for any event of ``type_``."""
        b = 1.0
        for r in self.rules:
            if r.predicate is None and r.applies_to_type(type_):
                b *= r.multiplier
        return b

    def type_allowed(self, type_: str) -> bool:
        return self.cause_types is None or type_ in self.cause_types


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
        hours = self.dt / 3600.0
        joint = {f for j in self.joint_causes if j.inseparable for f in j.features}
        parts = [str(j) for j in self.joint_causes if j.inseparable]
        parts += [str(c) for c in self.features if c.feature not in joint]
        out = (f"lag {hours:.2f}h -> kernel {self.kernel_value:.3f} "
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
    ) -> EffectSpec:
        """Declare an effect type, its lag kernel and its link priors.

        ``priors`` maps feature -> (alpha, beta). Features not listed use
        ``default_prior`` (default Beta(1, 4): weakly sceptical, mean 0.2).
        ``cause_types`` optionally restricts which event types may be causes.
        ``credit`` picks the confirmation update ("proportional" EM credit,
        or v0.1 "uniform"). ``global_pseudocount`` caps how much an entity
        borrows from evidence pooled across entities; ``hierarchical=False``
        disables borrowing for this effect.
        """
        if credit not in CREDIT_MODES:
            raise ValueError(f"credit must be one of {CREDIT_MODES}, got {credit!r}")
        spec = EffectSpec(
            name=name,
            kernel=kernel,
            links=LinkTable(default_prior=default_prior, priors=priors,
                            global_pseudocount=global_pseudocount,
                            hierarchical=hierarchical),
            cause_types=frozenset(cause_types) if cause_types else None,
            credit=credit,
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
        snapshot. Reads v1 and v2 snapshots."""
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


def half_life_to_lambda(half_life: float) -> float:
    """Convenience: decay rate for a given half-life (seconds)."""
    return math.log(2.0) / half_life
