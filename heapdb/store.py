"""HeapDB: the store, the top-k attribution query, feedback and persistence.

Scoring model
-------------
For an effect ``E`` observed for entity ``x`` at time ``T`` and a candidate
event ``e`` (time ``t``, weight ``w``, features ``F``)::

    score(e | E) = kernel_E(T - t)  *  w  *  noisy_or_{f in F}( strength(f, E) )

i.e. *timing* (does the lag fit how this effect usually lags its causes?) x
*magnitude* x *learned plausibility* of the event's features.

Top-k query (threshold-algorithm style)
---------------------------------------
1. Only segments overlapping ``[T - support, T - min_lag]`` can contribute.
2. For each such segment compute an upper bound on the score of *any* event
   in it, from its summaries alone::

       bound = kernel.max_over(lag range of segment)
             * segment.max_weight
             * noisy_or(strength(f) for f in union of segment features)

   Each factor upper-bounds the matching factor of every event in the
   segment (noisy-OR is monotone in its feature set), so the product does.
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
from typing import Iterable, Iterator, Mapping

from .events import Event, as_feature_set
from .kernels import Kernel, kernel_from_dict
from .links import LinkTable
from .segments import EntityLog

# Relative slack applied to segment bounds before comparing them with real
# scores. Mathematically bound >= score; this guards against the last-bit
# floating point differences between computing noisy-OR over a feature
# *union* versus a subset, so we never prune an event we should have kept.
_BOUND_SLACK = 1e-9


# ---------------------------------------------------------------------------
# Effect registration
# ---------------------------------------------------------------------------


@dataclass
class EffectSpec:
    """Everything HeapDB knows about one effect type."""

    name: str
    kernel: Kernel
    links: LinkTable
    #: If set, only events of these types are considered as causes.
    cause_types: frozenset[str] | None = None


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
        return f"{self.feature}={self.strength:.2f} (a={self.alpha:g}, b={self.beta:g})"


@dataclass(frozen=True)
class Explanation:
    """Why an event got the score it got. ``score == kernel * weight * link``."""

    dt: float                 # seconds between cause and effect
    kernel_value: float       # timing plausibility in [0, 1]
    weight: float             # event weight
    link_strength: float      # noisy-OR of feature strengths
    features: tuple[FeatureContribution, ...]  # sorted by strength, desc
    warnings: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        return self.kernel_value * self.weight * self.link_strength

    def __str__(self) -> str:
        hours = self.dt / 3600.0
        feats = ", ".join(str(c) for c in self.features) or "-"
        out = (f"lag {hours:.2f}h -> kernel {self.kernel_value:.3f} "
               f"x weight {self.weight:g} x link {self.link_strength:.3f} "
               f"[noisy-OR of {feats}]")
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


@dataclass
class QueryStats:
    """Work counters, so you can see the pruning happen."""

    segments_in_window: int = 0   # segments overlapping the kernel support
    segments_scanned: int = 0     # segments whose events were scored
    segments_pruned: int = 0      # skipped thanks to the bound (or zero bound)
    events_scanned: int = 0       # events actually scored

    def __str__(self) -> str:
        return (f"segments: {self.segments_scanned} scanned / "
                f"{self.segments_pruned} pruned / {self.segments_in_window} in window; "
                f"events scored: {self.events_scanned}")


@dataclass
class QueryResult:
    """Ranked causes (best first) plus execution statistics. Iterable."""

    entity: str
    effect: str
    at: float
    causes: list[Cause] = field(default_factory=list)
    stats: QueryStats = field(default_factory=QueryStats)

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
    """

    def __init__(self, bucket_seconds: float = 900.0) -> None:
        if bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be > 0")
        self.bucket_seconds = float(bucket_seconds)
        self._logs: dict[str, EntityLog] = {}
        self._events: dict[int, Event] = {}
        self._effects: dict[str, EffectSpec] = {}
        self._next_id = 1

    # -- schema -------------------------------------------------------------

    def register_effect(
        self,
        name: str,
        kernel: Kernel,
        priors: Mapping[str, tuple[float, float]] | None = None,
        default_prior: tuple[float, float] = (1.0, 4.0),
        cause_types: Iterable[str] | None = None,
    ) -> EffectSpec:
        """Declare an effect type, its lag kernel and its link priors.

        ``priors`` maps feature -> (alpha, beta). Features not listed use
        ``default_prior`` (default Beta(1, 4): weakly sceptical, mean 0.2).
        ``cause_types`` optionally restricts which event types may be causes.
        """
        spec = EffectSpec(
            name=name,
            kernel=kernel,
            links=LinkTable(default_prior=default_prior, priors=priors),
            cause_types=frozenset(cause_types) if cause_types else None,
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

    # -- writes -------------------------------------------------------------

    def insert(
        self,
        entity: str,
        time: float,
        type: str,
        features: Iterable[str] | str | None = None,
        weight: float = 1.0,
    ) -> Event:
        """Append an event and return it (with its assigned id)."""
        ev = Event(
            id=self._next_id,
            entity=entity,
            time=float(time),
            type=type,
            features=as_feature_set(features),
            weight=float(weight),
        )
        self._next_id += 1
        self._append(ev)
        return ev

    def _append(self, ev: Event) -> None:
        log = self._logs.get(ev.entity)
        if log is None:
            log = self._logs[ev.entity] = EntityLog(self.bucket_seconds)
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

    def entity_log(self, entity: str) -> EntityLog | None:
        return self._logs.get(entity)

    def score(self, event: Event, effect: str, at: float) -> float:
        """Score of a single event for an effect at time ``at``."""
        spec = self.effect(effect)
        if spec.cause_types is not None and event.type not in spec.cause_types:
            return 0.0
        k = spec.kernel.value(at - event.time)
        if k == 0.0 or event.weight == 0.0:
            return 0.0
        return k * event.weight * spec.links.noisy_or(event.features)

    def explain(self, event: Event, effect: str, at: float) -> Explanation:
        """Break an event's score into timing, weight and per-feature links."""
        spec = self.effect(effect)
        dt = at - event.time
        contribs = []
        for f in event.features:
            c = spec.links.get(f)
            contribs.append(FeatureContribution(f, c.strength, c.alpha, c.beta))
        contribs.sort(key=lambda c: (-c.strength, c.feature))
        return Explanation(
            dt=dt,
            kernel_value=spec.kernel.value(dt),
            weight=event.weight,
            link_strength=spec.links.noisy_or(event.features),
            features=tuple(contribs),
            warnings=tuple(self._confounding_warnings(event, effect)),
        )

    def _confounding_warnings(
        self, event: Event, effect: str, min_support: int = 2
    ) -> list[str]:
        """Flag groups of this event's features that *always* appear together.

        If ``coffee`` and ``dairy`` have never been consumed apart, feedback
        can't tell them apart: confirming the latte raises both links equally.
        That is a confounder, and the honest thing is to say so.

        Deliberately simple: features whose posting lists (from the segment
        inverted indexes) are *identical* form one group. We only warn when
        the group has seen feedback, since that is when learned strengths are
        being attributed ambiguously.
        """
        log = self._logs.get(event.entity)
        if log is None or len(event.features) < 2:
            return []
        links = self.effect(effect).links
        groups: dict[frozenset[int], list[str]] = {}
        for f in sorted(event.features):
            groups.setdefault(frozenset(log.event_ids_with_feature(f)), []).append(f)
        warnings = []
        for ids, feats in groups.items():
            if len(feats) < 2 or len(ids) < min_support:
                continue
            if not any(links.get(f).pos + links.get(f).neg > 0 for f in feats):
                continue
            names = ", ".join(f"'{f}'" for f in feats)
            warnings.append(
                f"{names} always co-occur ({len(ids)} events): "
                f"feedback cannot separate their effects yet"
            )
        return warnings

    # -- the main query -----------------------------------------------------

    def _window(self, spec: EffectSpec, at: float, within: float | None) -> tuple[float, float]:
        """Event-time interval that can possibly contribute to the effect."""
        lookback = spec.kernel.support if within is None else min(within, spec.kernel.support)
        return at - lookback, at - spec.kernel.min_lag

    def top_causes(
        self,
        entity: str,
        effect: str,
        at: float,
        k: int = 3,
        explain: bool = False,
        within: float | None = None,
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
        result = QueryResult(entity=entity, effect=effect, at=at)
        stats = result.stats
        log = self._logs.get(entity)
        if log is None:
            return result

        t_lo, t_hi = self._window(spec, at, within)
        segments = log.segments_between(t_lo, t_hi)
        stats.segments_in_window = len(segments)

        # Step 1: bound every segment from its summaries alone.
        bounded = []
        for seg in segments:
            if spec.cause_types is not None and not (seg.types & spec.cause_types):
                bound = 0.0
            else:
                # Lags covered by this segment's events (clipped to window).
                dt_lo = at - min(seg.max_time, t_hi)
                dt_hi = at - max(seg.min_time, t_lo)
                kb = spec.kernel.max_over(dt_lo, dt_hi) if dt_lo <= dt_hi else 0.0
                bound = kb * seg.max_weight
                if bound > 0.0:
                    bound *= spec.links.noisy_or(seg.feature_counts)
            bounded.append((bound * (1.0 + _BOUND_SLACK), seg))

        # Step 2: visit segments best-bound-first. (Sort by -bound, then by
        # segment start so ties are deterministic.)
        bounded.sort(key=lambda bs: (-bs[0], bs[1].start))

        # Bounded min-heap of (score, event_id, event). Smallest at [0], so
        # it is the k-th best so far and the one to evict. Ties are broken by
        # event id: the later event wins, which is also what the brute-force
        # reference does.
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
            Cause(
                rank=i + 1,
                event=ev,
                score=s,
                explanation=self.explain(ev, effect, at) if explain else None,
            )
            for i, (s, _, ev) in enumerate(ranked)
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
    ) -> None:
        """Teach the database which candidate really was the cause.

        Each feature of the confirmed event gets ``alpha += 1``; each feature
        of every rejected event gets ``beta += 1``. ``entity`` and
        ``at_time`` are validated against the events (a cause must belong to
        the entity and precede the effect) but are otherwise informational.
        """
        spec = self.effect(effect_type)

        def checked(eid: int) -> Event:
            ev = self._events.get(int(eid))
            if ev is None:
                raise KeyError(f"unknown event id {eid}")
            if ev.entity != entity:
                raise ValueError(f"event {eid} belongs to {ev.entity!r}, not {entity!r}")
            if ev.time > at_time:
                raise ValueError(f"event {eid} happened after the effect")
            return ev

        if confirmed_event_id is not None:
            spec.links.confirm(checked(confirmed_event_id).features)
        for eid in rejected_ids:
            spec.links.reject(checked(eid).features)

    # -- persistence --------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a JSONL snapshot: one meta line, one line per effect, one per event."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "meta", "version": 1,
                                 "bucket_seconds": self.bucket_seconds,
                                 "next_id": self._next_id}) + "\n")
            for spec in self._effects.values():
                fh.write(json.dumps({
                    "kind": "effect",
                    "name": spec.name,
                    "kernel": spec.kernel.to_dict(),
                    "links": spec.links.to_dict(),
                    "cause_types": sorted(spec.cause_types) if spec.cause_types else None,
                }) + "\n")
            for ev in sorted(self._events.values(), key=lambda e: e.id):
                fh.write(json.dumps({"kind": "event", **ev.to_dict()}) + "\n")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "HeapDB":
        """Rebuild a database (segments, indexes, links) from a snapshot."""
        db: HeapDB | None = None
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                kind = rec.get("kind")
                if kind == "meta":
                    db = cls(bucket_seconds=rec["bucket_seconds"])
                    db._next_id = int(rec["next_id"])
                    continue
                if db is None:
                    raise ValueError(f"{path}:{lineno}: snapshot must start with a meta line")
                if kind == "effect":
                    ct = rec.get("cause_types")
                    db._effects[rec["name"]] = EffectSpec(
                        name=rec["name"],
                        kernel=kernel_from_dict(rec["kernel"]),
                        links=LinkTable.from_dict(rec["links"]),
                        cause_types=frozenset(ct) if ct else None,
                    )
                elif kind == "event":
                    ev = Event.from_dict(rec)
                    db._append(ev)
                    db._next_id = max(db._next_id, ev.id + 1)
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
