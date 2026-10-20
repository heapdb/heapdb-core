"""A tiny query language for HeapDB.

Grammar (keywords are case-insensitive; ``[...]`` is optional)::

    TOP <k> CAUSES OF <effect> FOR <entity> [AT <time>] [WITHIN <duration>] [EXPLAIN]

    FEEDBACK <effect> FOR <entity> AT <time> CONFIRMED <event_id>
             [REJECTED <id>[,<id>...]]

    INSERT <type> FOR <entity> [AT <time>] [FEATURES <f>[,<f>...]] [WEIGHT <w>]

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
from .store import HeapDB, QueryResult

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
    weight: float = 1.0


Statement = Union[TopCausesQuery, FeedbackStmt, InsertStmt]


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
    else:
        raise QueryError(f"unknown statement {head!r} (expected TOP, FEEDBACK or INSERT)")
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
    entity = ts.next("entity")
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
    return TopCausesQuery(k=k, effect=effect, entity=entity, at=at, within=within, explain=explain)


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
    at, feats, weight = NOW, frozenset(), 1.0
    while not ts.done():
        kw = ts.next("clause").upper()
        if kw == "AT":
            at = parse_time(ts.next("time"))
        elif kw == "FEATURES":
            feats = frozenset(f for f in ts.next("features").split(",") if f)
        elif kw == "WEIGHT":
            try:
                weight = float(ts.next("weight"))
            except ValueError:
                raise QueryError("WEIGHT must be a number") from None
        else:
            raise QueryError(f"unexpected {kw!r} (expected AT, FEATURES or WEIGHT)")
    return InsertStmt(type_, entity, at, feats, weight)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute(
    db: HeapDB,
    stmt: Statement | str,
    clock: Callable[[], float] = _time.time,
) -> QueryResult | str:
    """Run a statement. Queries return a QueryResult; writes return a message."""
    if isinstance(stmt, str):
        stmt = parse(stmt)
    now = clock()
    if isinstance(stmt, TopCausesQuery):
        return db.top_causes(stmt.entity, stmt.effect, stmt.at.resolve(now),
                             k=stmt.k, explain=stmt.explain, within=stmt.within)
    if isinstance(stmt, FeedbackStmt):
        db.feedback(stmt.entity, stmt.effect, stmt.at.resolve(now),
                    confirmed_event_id=stmt.confirmed, rejected_ids=stmt.rejected)
        return (f"ok: confirmed #{stmt.confirmed}"
                + (f", rejected {', '.join('#' + str(r) for r in stmt.rejected)}"
                   if stmt.rejected else ""))
    if isinstance(stmt, InsertStmt):
        ev = db.insert(stmt.entity, stmt.at.resolve(now), stmt.type, stmt.features, stmt.weight)
        return f"ok: inserted {ev.short()}"
    raise QueryError(f"cannot execute {stmt!r}")  # pragma: no cover


def format_result(res: QueryResult) -> str:
    """Pretty-print a QueryResult (used by the CLI and examples)."""
    lines = [f"Top causes of '{res.effect}' for {res.entity} at t={format_time(res.at)}:"]
    if not res.causes:
        lines.append("  (no candidate causes in window)")
    for c in res:
        lines.append(f"  {c.rank}. {c.event.short():<44} score={c.score:.4f}")
        if c.explanation is not None:
            lines.append(f"      {c.explanation}")
    lines.append(f"  [{res.stats}]")
    return "\n".join(lines)
