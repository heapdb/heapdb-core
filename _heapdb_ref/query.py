"""A tiny query language for HeapDB.

Grammar (keywords are case-insensitive; ``[...]`` is optional)::

    TOP <k> CAUSES OF <effect> FOR <entity>[,<entity>...] [AT <time>]
        [WITHIN <duration>] [EXPLAIN]

    CALIBRATION OF <effect> [BINS <n>]

    RISK OF <effect> FOR <entity> FEATURES <f>[,<f>...] [WEIGHT <w>] [TYPE <t>]
        [ATTRS k=v[,k=v...]] [EXPLAIN]

    FEEDBACK <effect> FOR <entity> AT <time> CONFIRMED <event_id>
             [REJECTED <id>[,<id>...]]

    INSERT <type> FOR <entity> [AT <time>] [FEATURES <f>[,<f>...]] [WEIGHT <w>]

    OBSERVE <effect> FOR <entity> [AT <time>] [LEAK <x>]

    SETTLE [<effect>] UNTIL <time> [FOR <entity>]

    LINKS OF <effect> [FOR <entity>] [MIN <n>]

``OBSERVE`` records an occurrence of an effect with no labelled cause and
learns from it observationally (soft EM credit, see
:meth:`~heapdb.store.HeapDB.observe`). ``SETTLE`` adds negative evidence for
events whose kernel window passed without the effect
(:meth:`~heapdb.store.HeapDB.settle`). ``LINKS OF`` lists learned link
strengths with 90% credible intervals and lift
(:meth:`~heapdb.store.HeapDB.link_report`) -- associations under the model,
not causal effects.

``FOR a,b,c`` (several entities) runs a cross-entity group query
(:meth:`~heapdb.store.HeapDB.top_causes_group`); scores are only comparable
across entities once calibrated (see ``CALIBRATION OF``). ``CALIBRATION OF``
reports mean predicted score vs observed confirm rate per bin and the Brier
score. ``INSERT`` also accepts ``ATTRS k=v[,k=v...]`` (numeric values are
parsed as numbers); without ``WEIGHT`` the event type's registered weight
function, if any, derives the weight from the attributes.

``RISK OF`` scores a candidate that has *not happened yet* (e.g. an open pull
request) with :meth:`~heapdb.store.HeapDB.feature_risk`: dose x noisy-OR of
the entity's current links x rules, with no timing term. Without ``WEIGHT``
the dose is derived from ``ATTRS`` by the type's weight function (else 1.0).

``<time>`` is a number of seconds, ``now``, ``now-<duration>``, or an ISO-8601
datetime (``2024-05-01T13:30``). ``<duration>`` is a number with an optional
unit: ``s``, ``m``, ``h``, ``d`` (``90``, ``30m``, ``12h``, ``1.5d``).

The parser produces small dataclasses; :func:`execute` runs them against a
:class:`~heapdb.store.HeapDB`. ``now`` is resolved by the caller-supplied
clock so tests (and simulations) can pin it.
"""

from __future__ import annotations

import re
import shlex
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Union

from .events import format_time
from .store import (
    CalibrationReport,
    HeapDB,
    LinkReport,
    ObserveReport,
    QueryResult,
    RiskResult,
    SettleReport,
)

_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)([smhd]?)$", re.IGNORECASE)


class QueryError(ValueError):
    """Raised for anything the parser does not understand."""


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeExpr:
    """Either an absolute time or ``now - offset``; resolved at execution."""

    absolute: float | None = None
    now_offset: float = 0.0

    def resolve(self, now: float) -> float:
        return self.absolute if self.absolute is not None else now - self.now_offset


NOW = TimeExpr()


@dataclass(frozen=True)
class TopCausesQuery:
    k: int
    effect: str
    entity: str
    at: TimeExpr = NOW
    within: float | None = None
    explain: bool = False
    #: All entities named in ``FOR``; more than one means a group query.
    entities: tuple[str, ...] = ()

    @property
    def is_group(self) -> bool:
        return len(self.entities) > 1


@dataclass(frozen=True)
class CalibrationQuery:
    effect: str
    bins: int = 10


@dataclass(frozen=True)
class RiskQuery:
    effect: str
    entity: str
    features: frozenset[str]
    weight: float | None = None  # None: derived from attrs, else 1.0
    type: str | None = None
    attrs: tuple[tuple[str, object], ...] = ()
    explain: bool = False


@dataclass(frozen=True)
class FeedbackStmt:
    effect: str
    entity: str
    at: TimeExpr
    confirmed: int
    rejected: tuple[int, ...] = ()


@dataclass(frozen=True)
class InsertStmt:
    type: str
    entity: str
    at: TimeExpr = NOW
    features: frozenset[str] = field(default_factory=frozenset)
    weight: float | None = None  # None: derived from attrs, else 1.0
    attrs: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class ObserveStmt:
    effect: str
    entity: str
    at: TimeExpr = NOW
    leak: float | None = None


@dataclass(frozen=True)
class SettleStmt:
    until: TimeExpr
    effect: str | None = None
    entity: str | None = None


@dataclass(frozen=True)
class LinksQuery:
    effect: str
    entity: str | None = None
    min_support: float = 1.0


Statement = Union[TopCausesQuery, FeedbackStmt, InsertStmt, CalibrationQuery, RiskQuery,
                  ObserveStmt, SettleStmt, LinksQuery]


# ---------------------------------------------------------------------------
# Lexical helpers
# ---------------------------------------------------------------------------


def parse_duration(text: str) -> float:
    """``"12h"`` -> 43200.0. A bare number means seconds."""
    m = _DURATION_RE.match(text.strip())
    if not m:
        raise QueryError(f"bad duration {text!r} (expected e.g. 90, 30m, 12h, 2d)")
    return float(m.group(1)) * _UNITS[(m.group(2) or "s").lower()]


def parse_time(text: str) -> TimeExpr:
    t = text.strip()
    low = t.lower()
    if low == "now":
        return NOW
    if low.startswith("now-"):
        return TimeExpr(now_offset=parse_duration(t[4:]))
    try:
        return TimeExpr(absolute=float(t))
    except ValueError:
        pass
    try:
        return TimeExpr(absolute=datetime.fromisoformat(t).timestamp())
    except ValueError:
        raise QueryError(f"bad time {text!r} (expected seconds, now, now-2h or ISO-8601)") from None


class _Tokens:
    """A cursor over whitespace-separated tokens with keyword helpers."""

    def __init__(self, text: str) -> None:
        try:
            self.toks = shlex.split(text.strip().rstrip(";"))
        except ValueError as e:
            raise QueryError(str(e)) from None
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def peek_kw(self, *kws: str) -> bool:
        p = self.peek()
        return p is not None and p.upper() in kws

    def next(self, what: str) -> str:
        p = self.peek()
        if p is None:
            raise QueryError(f"unexpected end of query, expected {what}")
        self.i += 1
        return p

    def expect(self, kw: str) -> None:
        tok = self.next(kw)
        if tok.upper() != kw:
            raise QueryError(f"expected {kw}, got {tok!r}")

    def done(self) -> bool:
        return self.i >= len(self.toks)


def _int(tok: str, what: str) -> int:
    try:
        return int(tok.lstrip("#"))
    except ValueError:
        raise QueryError(f"{what} must be an integer, got {tok!r}") from None


def _id_list(tok: str) -> tuple[int, ...]:
    return tuple(_int(x, "event id") for x in tok.split(",") if x)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse(text: str) -> Statement:
    """Parse one statement. Raises :class:`QueryError` on bad input."""
    ts = _Tokens(text)
    head = ts.next("a statement").upper()
    if head == "TOP":
        stmt: Statement = _parse_top(ts)
    elif head == "FEEDBACK":
        stmt = _parse_feedback(ts)
    elif head == "INSERT":
        stmt = _parse_insert(ts)
    elif head == "CALIBRATION":
        stmt = _parse_calibration(ts)
    elif head == "RISK":
        stmt = _parse_risk(ts)
    elif head == "OBSERVE":
        stmt = _parse_observe(ts)
    elif head == "SETTLE":
        stmt = _parse_settle(ts)
    elif head == "LINKS":
        stmt = _parse_links(ts)
    else:
        raise QueryError(f"unknown statement {head!r} (expected TOP, RISK, FEEDBACK, "
                         f"INSERT, CALIBRATION, OBSERVE, SETTLE or LINKS)")
    if not ts.done():
        raise QueryError(f"unexpected trailing input: {' '.join(ts.toks[ts.i:])!r}")
    return stmt


def _parse_top(ts: _Tokens) -> TopCausesQuery:
    k = _int(ts.next("k"), "k")
    if k < 1:
        raise QueryError("k must be >= 1")
    if not ts.peek_kw("CAUSE", "CAUSES"):
        raise QueryError(f"expected CAUSES, got {ts.peek()!r}")
    ts.next("CAUSES")
    ts.expect("OF")
    effect = ts.next("effect name")
    ts.expect("FOR")
    entity_tok = ts.next("entity")
    entities = tuple(dict.fromkeys(e for e in entity_tok.split(",") if e))
    if not entities:
        raise QueryError(f"bad entity list {entity_tok!r}")
    entity = ",".join(entities)
    at, within, explain = NOW, None, False
    while not ts.done():
        kw = ts.next("clause").upper()
        if kw == "AT":
            at = parse_time(ts.next("time"))
        elif kw == "WITHIN":
            within = parse_duration(ts.next("duration"))
        elif kw == "EXPLAIN":
            explain = True
        else:
            raise QueryError(f"unexpected {kw!r} (expected AT, WITHIN or EXPLAIN)")
    return TopCausesQuery(k=k, effect=effect, entity=entity, at=at, within=within,
                          explain=explain, entities=entities)


def _parse_calibration(ts: _Tokens) -> CalibrationQuery:
    ts.expect("OF")
    effect = ts.next("effect name")
    bins = 10
    if ts.peek_kw("BINS"):
        ts.next("BINS")
        bins = _int(ts.next("bins"), "bins")
        if bins < 1:
            raise QueryError("BINS must be >= 1")
    return CalibrationQuery(effect, bins)


def _attr_value(v: str) -> object:
    for conv in (int, float):
        try:
            return conv(v)
        except ValueError:
            pass
    return v


def _parse_attrs(tok: str) -> tuple[tuple[str, object], ...]:
    out = []
    for part in tok.split(","):
        if not part:
            continue
        k, sep, v = part.partition("=")
        if not sep or not k:
            raise QueryError(f"bad attribute {part!r} (expected key=value)")
        out.append((k, _attr_value(v)))
    return tuple(out)


def _weight(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        raise QueryError("WEIGHT must be a number") from None


def _parse_risk(ts: _Tokens) -> RiskQuery:
    ts.expect("OF")
    effect = ts.next("effect name")
    ts.expect("FOR")
    entity = ts.next("entity")
    feats: frozenset[str] | None = None
    weight: float | None = None
    type_: str | None = None
    attrs: tuple[tuple[str, object], ...] = ()
    explain = False
    while not ts.done():
        kw = ts.next("clause").upper()
        if kw == "FEATURES":
            feats = frozenset(f for f in ts.next("features").split(",") if f)
        elif kw == "WEIGHT":
            weight = _weight(ts.next("weight"))
        elif kw == "TYPE":
            type_ = ts.next("event type")
        elif kw == "ATTRS":
            attrs = _parse_attrs(ts.next("attributes"))
        elif kw == "EXPLAIN":
            explain = True
        else:
            raise QueryError(f"unexpected {kw!r} (expected FEATURES, WEIGHT, TYPE, ATTRS or EXPLAIN)")
    if not feats:
        raise QueryError("RISK needs FEATURES <f>[,<f>...]")
    return RiskQuery(effect, entity, feats, weight, type_, attrs, explain)


def _parse_feedback(ts: _Tokens) -> FeedbackStmt:
    effect = ts.next("effect name")
    ts.expect("FOR")
    entity = ts.next("entity")
    ts.expect("AT")
    at = parse_time(ts.next("time"))
    ts.expect("CONFIRMED")
    confirmed = _int(ts.next("event id"), "event id")
    rejected: tuple[int, ...] = ()
    if ts.peek_kw("REJECTED"):
        ts.next("REJECTED")
        rejected = _id_list(ts.next("event ids"))
    return FeedbackStmt(effect, entity, at, confirmed, rejected)


def _parse_insert(ts: _Tokens) -> InsertStmt:
    type_ = ts.next("event type")
    ts.expect("FOR")
    entity = ts.next("entity")
    at, feats = NOW, frozenset()
    weight: float | None = None
    attrs: tuple[tuple[str, object], ...] = ()
    while not ts.done():
        kw = ts.next("clause").upper()
        if kw == "AT":
            at = parse_time(ts.next("time"))
        elif kw == "FEATURES":
            feats = frozenset(f for f in ts.next("features").split(",") if f)
        elif kw == "WEIGHT":
            weight = _weight(ts.next("weight"))
        elif kw == "ATTRS":
            attrs = _parse_attrs(ts.next("attributes"))
        else:
            raise QueryError(f"unexpected {kw!r} (expected AT, FEATURES, WEIGHT or ATTRS)")
    return InsertStmt(type_, entity, at, feats, weight, attrs)


def _number(tok: str, what: str) -> float:
    try:
        return float(tok)
    except ValueError:
        raise QueryError(f"{what} must be a number, got {tok!r}") from None


def _parse_observe(ts: _Tokens) -> ObserveStmt:
    effect = ts.next("effect name")
    ts.expect("FOR")
    entity = ts.next("entity")
    at: TimeExpr = NOW
    leak: float | None = None
    while not ts.done():
        kw = ts.next("clause").upper()
        if kw == "AT":
            at = parse_time(ts.next("time"))
        elif kw == "LEAK":
            leak = _number(ts.next("leak"), "LEAK")
            if not leak > 0:
                raise QueryError("LEAK must be > 0")
        else:
            raise QueryError(f"unexpected {kw!r} (expected AT or LEAK)")
    return ObserveStmt(effect, entity, at, leak)


def _parse_settle(ts: _Tokens) -> SettleStmt:
    effect: str | None = None
    if not ts.peek_kw("UNTIL"):
        effect = ts.next("effect name or UNTIL")
    ts.expect("UNTIL")
    until = parse_time(ts.next("time"))
    entity: str | None = None
    if ts.peek_kw("FOR"):
        ts.next("FOR")
        entity = ts.next("entity")
    return SettleStmt(until, effect, entity)


def _parse_links(ts: _Tokens) -> LinksQuery:
    ts.expect("OF")
    effect = ts.next("effect name")
    entity: str | None = None
    min_support = 1.0
    while not ts.done():
        kw = ts.next("clause").upper()
        if kw == "FOR":
            entity = ts.next("entity")
        elif kw == "MIN":
            min_support = _number(ts.next("min support"), "MIN")
            if min_support < 0:
                raise QueryError("MIN must be >= 0")
        else:
            raise QueryError(f"unexpected {kw!r} (expected FOR or MIN)")
    return LinksQuery(effect, entity, min_support)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute(
    db: HeapDB,
    stmt: Statement | str,
    clock: Callable[[], float] = _time.time,
) -> QueryResult | CalibrationReport | RiskResult | LinkReport | ObserveReport | SettleReport | str:
    """Run a statement. Queries return a QueryResult (or CalibrationReport,
    RiskResult, LinkReport); OBSERVE and SETTLE return their reports; other
    writes return a message."""
    if isinstance(stmt, str):
        stmt = parse(stmt)
    now = clock()
    if isinstance(stmt, TopCausesQuery):
        if stmt.is_group:
            return db.top_causes_group(stmt.entities, stmt.effect, stmt.at.resolve(now),
                                       k=stmt.k, explain=stmt.explain, within=stmt.within)
        return db.top_causes(stmt.entity, stmt.effect, stmt.at.resolve(now),
                             k=stmt.k, explain=stmt.explain, within=stmt.within)
    if isinstance(stmt, CalibrationQuery):
        return db.calibration_report(stmt.effect, bins=stmt.bins)
    if isinstance(stmt, RiskQuery):
        out = db.feature_risk(stmt.effect, stmt.features, weight=stmt.weight,
                              entity=stmt.entity, type=stmt.type, attrs=dict(stmt.attrs),
                              explain=stmt.explain)
        score, ex = out if isinstance(out, tuple) else (out, None)
        return RiskResult(stmt.effect, stmt.entity, stmt.features, stmt.type, score, ex)
    if isinstance(stmt, FeedbackStmt):
        db.feedback(stmt.entity, stmt.effect, stmt.at.resolve(now),
                    confirmed_event_id=stmt.confirmed, rejected_ids=stmt.rejected)
        return (f"ok: confirmed #{stmt.confirmed}"
                + (f", rejected {', '.join('#' + str(r) for r in stmt.rejected)}"
                   if stmt.rejected else ""))
    if isinstance(stmt, ObserveStmt):
        return db.observe(stmt.entity, stmt.effect, stmt.at.resolve(now), leak=stmt.leak)
    if isinstance(stmt, SettleStmt):
        return db.settle(stmt.until.resolve(now), effects=stmt.effect, entity=stmt.entity)
    if isinstance(stmt, LinksQuery):
        return db.link_report(stmt.effect, entity=stmt.entity, min_support=stmt.min_support)
    if isinstance(stmt, InsertStmt):
        ev = db.insert(stmt.entity, stmt.at.resolve(now), stmt.type, stmt.features,
                       stmt.weight, attrs=dict(stmt.attrs))
        return f"ok: inserted {ev.short()}"
    raise QueryError(f"cannot execute {stmt!r}")  # pragma: no cover


def format_result(res: QueryResult | CalibrationReport | RiskResult | LinkReport
                  | ObserveReport | SettleReport) -> str:
    """Pretty-print a QueryResult or any of the report types."""
    if isinstance(res, (CalibrationReport, RiskResult, LinkReport, ObserveReport, SettleReport)):
        return str(res)
    who = f"entities {res.entity}" if res.is_group else res.entity
    lines = [f"Top causes of '{res.effect}' for {who} at t={format_time(res.at)}:"]
    if not res.causes:
        lines.append("  (no candidate causes in window)")
    for c in res:
        ent = f"{c.entity}: " if res.is_group else ""
        lines.append(f"  {c.rank}. {ent}{c.event.short():<44} score={c.score:.4f}")
        if c.explanation is not None:
            lines.append(f"      {c.explanation}")
    lines.append(f"  [{res.stats}]")
    return "\n".join(lines)
