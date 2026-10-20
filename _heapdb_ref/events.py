"""Core event record.

An :class:`Event` is an immutable fact appended to the log: "entity X had an
event of type T at time t, with these features and this weight".

Events are the *candidate causes* that HeapDB ranks. Features are the unit of
learning: link strengths are learned per (feature, effect_type) pair, so an
event's causal plausibility is an aggregate over its features.

Weight semantics (v0.2): ``weight`` is the **dose / exposure**, i.e. the
magnitude of the event -- portion size of a meal, size of a deploy, amount of
a transaction. It defaults to 1.0 and must be > 0. *Propensity* ("how likely
is this kind of thing to cause that effect") does **not** belong in the
weight: it lives in the learned links. Weights can be derived from ``attrs``
by registering an event type with a weight function, see
:meth:`heapdb.HeapDB.register_event_type`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Event:
    """A single append-only event.

    Attributes:
        id: Monotonic integer id assigned by the store (unique per database).
        entity: Partition key, e.g. a user id, a service name, a device.
        time: Event time in seconds (any consistent epoch; floats allowed).
        type: Free-form event type, e.g. ``"meal"`` or ``"deploy"``.
        features: Set of feature tokens, e.g. ``{"beans", "onion"}``.
        weight: Dose / exposure magnitude (> 0). Scores scale linearly with
            it. Defaults to 1.0. Propensity belongs in links, not here.
        attrs: Optional free-form attributes (JSON-serialisable values), e.g.
            ``{"portion": 1.5}`` or ``{"lines_changed": 420}``. Used by
            event-type weight functions and plausibility rules. Read-only.
    """

    id: int
    entity: str
    time: float
    type: str
    features: frozenset[str] = field(default_factory=frozenset)
    weight: float = 1.0
    # Excluded from the hash (mappings are unhashable) but part of equality.
    attrs: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        # Normalise features to a frozenset so events are hashable and safe to
        # share between the log, segments and query results.
        if not isinstance(self.features, frozenset):
            object.__setattr__(self, "features", frozenset(self.features))
        if not isinstance(self.attrs, MappingProxyType):
            object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs or {})))
        validate_weight(self.weight)

    # -- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity": self.entity,
            "time": self.time,
            "type": self.type,
            "features": sorted(self.features),
            "weight": self.weight,
            **({"attrs": dict(self.attrs)} if self.attrs else {}),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        return cls(
            id=int(d["id"]),
            entity=str(d["entity"]),
            time=float(d["time"]),
            type=str(d["type"]),
            features=frozenset(d.get("features", ())),
            weight=float(d.get("weight", 1.0)),
            attrs=d.get("attrs") or {},
        )

    def short(self) -> str:
        """Compact human-readable form used by examples and the CLI."""
        feats = ",".join(sorted(self.features))
        return f"#{self.id} {self.type}@{format_time(self.time)} [{feats}] w={self.weight:.3g}"


def validate_weight(weight: float) -> float:
    """Weights are doses: finite and strictly positive."""
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"event weight must be a finite number > 0, got {weight!r}")
    return float(weight)


def format_time(t: float) -> str:
    """Show small (simulated) times as plain seconds, epoch-like times as UTC."""
    if abs(t) < 1e8:  # < ~3 years: clearly not a real Unix timestamp
        return f"{t:g}"
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def as_feature_set(features: Iterable[str] | str | None) -> frozenset[str]:
    """Accept a set/list of features, a comma-separated string, or None."""
    if features is None:
        return frozenset()
    if isinstance(features, str):
        return frozenset(f.strip() for f in features.split(",") if f.strip())
    return frozenset(features)
