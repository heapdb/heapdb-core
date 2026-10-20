"""Time-bucketed segments: the unit of storage *and* of pruning.

Each entity's log is cut into fixed-width time buckets. A :class:`Segment`
holds the events of one bucket plus small summaries that let the query engine
bound *every* event in the segment without reading any of them:

* ``min_time`` / ``max_time`` -- the actual event-time range (tighter than the
  bucket edges), which gives a lag range and therefore a kernel upper bound;
* ``max_weight`` -- the largest event weight;
* ``feature_counts`` -- the union of features (with multiplicities), which
  gives a noisy-OR upper bound on link strength;
* ``types`` -- the event types present, so whole segments can be skipped when
  an effect only accepts certain cause types;
* ``index`` -- an inverted index ``feature -> [event ids]`` for feature
  lookups (used by the confounding check and ``events_with_feature``).

Summaries only ever grow (the log is append-only), so maintaining them on
insert is O(#features).
"""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from typing import Iterator

from .events import Event


class Segment:
    """All events of one entity whose time falls in ``[start, start + width)``."""

    __slots__ = ("start", "width", "events", "min_time", "max_time",
                 "max_weight", "feature_counts", "types", "index")

    def __init__(self, start: float, width: float) -> None:
        self.start = start
        self.width = width
        self.events: list[Event] = []
        self.min_time = float("inf")
        self.max_time = float("-inf")
        self.max_weight = 0.0
        self.feature_counts: Counter[str] = Counter()
        self.types: set[str] = set()
        self.index: dict[str, list[int]] = defaultdict(list)

    @property
    def end(self) -> float:
        return self.start + self.width

    def add(self, event: Event) -> None:
        self.events.append(event)
        self.min_time = min(self.min_time, event.time)
        self.max_time = max(self.max_time, event.time)
        self.max_weight = max(self.max_weight, event.weight)
        self.feature_counts.update(event.features)
        self.types.add(event.type)
        for f in event.features:
            self.index[f].append(event.id)

    @property
    def features(self) -> set[str]:
        """Union of features of all events in the segment."""
        return set(self.feature_counts)

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return (f"Segment(start={self.start:g}, n={len(self)}, "
                f"max_w={self.max_weight:g}, feats={len(self.feature_counts)})")


class EntityLog:
    """The segmented log of a single entity.

    Segments live in a dict keyed by bucket number, plus a sorted list of
    bucket numbers so a time range maps to segments with two bisections.
    Out-of-order inserts are fine: they just land in an older bucket.
    """

    def __init__(self, bucket_seconds: float) -> None:
        self.bucket_seconds = float(bucket_seconds)
        self._segments: dict[int, Segment] = {}
        self._keys: list[int] = []  # sorted bucket numbers

    def _bucket(self, t: float) -> int:
        return int(t // self.bucket_seconds)

    def add(self, event: Event) -> None:
        b = self._bucket(event.time)
        seg = self._segments.get(b)
        if seg is None:
            seg = Segment(b * self.bucket_seconds, self.bucket_seconds)
            self._segments[b] = seg
            bisect.insort(self._keys, b)
        seg.add(event)

    def segments_between(self, t_lo: float, t_hi: float) -> list[Segment]:
        """Segments whose bucket overlaps ``[t_lo, t_hi]``."""
        if t_lo > t_hi:
            return []
        i = bisect.bisect_left(self._keys, self._bucket(t_lo))
        j = bisect.bisect_right(self._keys, self._bucket(t_hi))
        return [self._segments[k] for k in self._keys[i:j]]

    def segments(self) -> Iterator[Segment]:
        for k in self._keys:
            yield self._segments[k]

    def events(self) -> Iterator[Event]:
        for seg in self.segments():
            yield from seg.events

    def event_ids_with_feature(self, feature: str) -> list[int]:
        out: list[int] = []
        for seg in self.segments():
            out.extend(seg.index.get(feature, ()))
        return out

    def __len__(self) -> int:
        return sum(len(s) for s in self._segments.values())

    @property
    def num_segments(self) -> int:
        return len(self._segments)
