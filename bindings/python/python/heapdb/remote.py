"""Client for a HeapDB server (hosted at api.heapdb.com, or self-hosted).

    >>> import heapdb
    >>> db = heapdb.connect("heapdb://hdb_live_...@api.heapdb.com/db_...")   # doctest: +SKIP
    >>> db = heapdb.connect()            # reads HEAPDB_URL                  # doctest: +SKIP

``connect()`` returns a :class:`RemoteHeapDB` whose methods mirror
:class:`heapdb.HeapDB` and return results with the same attributes
(``Cause.rank``, ``.event``, ``.score``, ``.explanation``,
``.event.short()``, ...). Pure standard library (``urllib`` + ``json``).

Connection URLs:

* ``heapdb://<api_key>@<host>[:port]/<database_id>``: HTTPS (hosted).
* ``heapdb+http://<api_key>@<host>[:port]/<database_id>``: plain HTTP
  (a self-hosted server on localhost or a private network).

Weight functions and rule predicates are *declarative* (no code is sent):
see :func:`weight_attr`, :func:`weight_log1p`, :func:`weight_const` and
:func:`where`.

Retries: 429 ``rate_limited``, 502/503/504 and connection errors are
retried with exponential backoff, honouring ``Retry-After``. Quota errors
(``monthly_quota_exceeded``, ``events_quota_exceeded``,
``storage_quota_exceeded``) are never retried. A retried write whose first
attempt actually succeeded can be applied twice; give events an
``external_id`` to make inserts idempotent.
"""

from __future__ import annotations

import json
import math
import os
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ._core import Candidate, Event, QueryError

__all__ = [
    "AuthenticationError",
    "AppliedRule",
    "BetaCounts",
    "CalibrationBin",
    "CalibrationReport",
    "Cause",
    "ConflictError",
    "EventCredit",
    "Explanation",
    "FeatureContribution",
    "FeedbackReport",
    "HeapDBError",
    "InvalidArgumentError",
    "JointCause",
    "LinkReport",
    "LinkRow",
    "NotFoundError",
    "ObserveReport",
    "PayloadTooLargeError",
    "PermissionDeniedError",
    "QueryLanguageError",
    "QueryResult",
    "QueryStats",
    "QuotaExceededError",
    "RankedCandidate",
    "RateLimitError",
    "RemoteConnectionError",
    "RemoteHeapDB",
    "RiskResult",
    "ServiceUnavailableError",
    "SettleReport",
    "connect",
    "parse_url",
    "weight_attr",
    "weight_const",
    "weight_log1p",
    "where",
]

_VERSION = "0.6.0"
MAX_BATCH = 1000

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HeapDBError(Exception):
    """An error returned by the server (``code`` is stable, see docs/server.md)."""

    def __init__(self, message: str, *, code: str = "error", status: int | None = None,
                 retry_after: float | None = None, request_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.retry_after = retry_after
        self.request_id = request_id

    def __str__(self) -> str:
        s = f"{self.code}: {self.message}"
        if self.status is not None:
            s = f"[{self.status}] {s}"
        return s


class AuthenticationError(HeapDBError):
    """401: missing, unknown or expired API key."""


class PermissionDeniedError(HeapDBError):
    """403: missing scope, key of another database, or suspended database."""


class NotFoundError(HeapDBError, KeyError):
    """404: unknown database, effect, event or route (also a ``KeyError``,
    like the local API's unknown-effect errors)."""

    __str__ = HeapDBError.__str__


class InvalidArgumentError(HeapDBError, ValueError):
    """400: rejected arguments (also a ``ValueError``)."""


class QueryLanguageError(InvalidArgumentError, QueryError):
    """400 ``query_error``: the statement did not parse (also a ``heapdb.QueryError``)."""


class ConflictError(HeapDBError):
    """409."""


class PayloadTooLargeError(HeapDBError):
    """413: request or batch too large."""


class QuotaExceededError(HeapDBError):
    """A quota is used up: ``monthly_quota_exceeded`` (429), ``events_quota_exceeded``
    or ``storage_quota_exceeded`` (413). Never retried."""


class RateLimitError(HeapDBError):
    """429 ``rate_limited`` after all retries."""


class ServiceUnavailableError(HeapDBError):
    """5xx after all retries."""


class RemoteConnectionError(HeapDBError, ConnectionError):
    """The server could not be reached."""


_QUOTA_CODES = {"monthly_quota_exceeded", "events_quota_exceeded", "storage_quota_exceeded"}


def _error_for(status: int, code: str, message: str, retry_after: float | None,
               request_id: str | None) -> HeapDBError:
    kw = dict(code=code, status=status, retry_after=retry_after, request_id=request_id)
    if code in _QUOTA_CODES:
        return QuotaExceededError(message, **kw)
    if code == "query_error":
        return QueryLanguageError(message, **kw)
    cls: type[HeapDBError] = {
        400: InvalidArgumentError,
        401: AuthenticationError,
        403: PermissionDeniedError,
        404: NotFoundError,
        409: ConflictError,
        413: PayloadTooLargeError,
        429: RateLimitError,
    }.get(status, ServiceUnavailableError if status >= 500 else HeapDBError)
    return cls(message, **kw)


# ---------------------------------------------------------------------------
# Declarative weight functions and predicates
# ---------------------------------------------------------------------------


def weight_const(value: float) -> dict[str, Any]:
    """Every event of the type weighs ``value``."""
    return {"kind": "const", "value": float(value)}


def weight_attr(attr: str, default: float | None = None) -> dict[str, Any]:
    """Weight = ``attrs[attr]`` (``default`` when missing), like
    ``lambda a: a.get(attr, default)``."""
    d: dict[str, Any] = {"kind": "attr", "attr": attr}
    if default is not None:
        d["value"] = float(default)
    return d


def weight_log1p(attr: str, default: float | None = None) -> dict[str, Any]:
    """Weight = ``log1p(attrs[attr])``, like ``lambda a: math.log1p(a[attr])``."""
    d: dict[str, Any] = {"kind": "log1p", "attr": attr}
    if default is not None:
        d["value"] = float(default)
    return d


_OPS = {"==": "eq", "!=": "ne", "<": "lt", "<=": "lte", ">": "gt", ">=": "gte"}
_MISSING = object()


def where(attr: str, op: str, value: Any = None, *, default: Any = _MISSING) -> dict[str, Any]:
    """A rule predicate on an event attribute, e.g. ``where("amount", "<", 10, default=0)``
    for ``lambda e: e.attrs.get("amount", 0) < 10``.

    Operators: ``== != < <= > >=`` (or ``eq ne lt lte gt gte``), ``in``,
    ``not_in``, ``contains``, ``exists``, ``missing``.
    """
    d: dict[str, Any] = {"attr": attr, "op": _OPS.get(op, op)}
    if d["op"] not in ("exists", "missing"):
        d["value"] = list(value) if isinstance(value, (set, frozenset, tuple)) else value
    if default is not _MISSING:
        d["default"] = default
    return d


def _weight_spec(weight: Any) -> dict[str, Any] | None:
    if weight is None:
        return None
    if isinstance(weight, Mapping):
        return dict(weight)
    if isinstance(weight, (int, float)) and not isinstance(weight, bool):
        return weight_const(weight)
    raise TypeError("remote databases take declarative weight functions only: use "
                    "heapdb.remote.weight_attr / weight_log1p / weight_const (no code is uploaded)")


def _predicate_spec(predicate: Any) -> dict[str, Any] | None:
    if predicate is None:
        return None
    if isinstance(predicate, Mapping):
        return dict(predicate)
    raise TypeError("remote databases take declarative predicates only: use "
                    "heapdb.remote.where(attr, op, value) (no code is uploaded)")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _f(v: Any) -> float:
    if isinstance(v, str):
        return float({"Infinity": "inf", "-Infinity": "-inf"}.get(v, v))
    return float(v)


def _fo(v: Any) -> float | None:
    return None if v is None else _f(v)


def _num_out(x: float) -> Any:
    if math.isfinite(x):
        return x
    return "NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")


def _event(d: Mapping[str, Any]) -> Event:
    return Event(int(d["id"]), d["entity"], _f(d["time"]), d["type"], d.get("features") or (),
                 _f(d.get("weight", 1.0)), d.get("attrs") or None)


@dataclass(frozen=True)
class BetaCounts:
    prior_alpha: float
    prior_beta: float
    pos: float
    neg: float
    global_pos: float
    global_neg: float
    alpha: float
    beta: float
    strength: float

    @classmethod
    def _from(cls, d: Mapping[str, Any] | None) -> BetaCounts | None:
        if not d:
            return None
        return cls(*(_f(d.get(k, 0.0)) for k in ("prior_alpha", "prior_beta", "pos", "neg", "global_pos",
                                                  "global_neg", "alpha", "beta", "strength")))


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    strength: float
    alpha: float
    beta: float
    log_odds: float | None = None    # logistic only

    @property
    def odds_multiplier(self) -> float | None:
        return None if self.log_odds is None else math.exp(self.log_odds)


@dataclass(frozen=True)
class AppliedRule:
    name: str
    multiplier: float


@dataclass(frozen=True)
class JointCause:
    features: tuple[str, ...]
    strength: float
    joint: BetaCounts | None
    inseparable: bool

    @property
    def name(self) -> str:
        return "+".join(self.features)


@dataclass(frozen=True)
class Explanation:
    dt: float
    kernel_value: float
    weight: float
    link_strength: float
    features: tuple[FeatureContribution, ...]
    warnings: tuple[str, ...]
    rules: tuple[AppliedRule, ...]
    joint_causes: tuple[JointCause, ...]
    entity: str
    prospective: bool
    text: str = field(default="", repr=False, compare=False)
    base_log_odds: float | None = None    # logistic only

    @property
    def combine(self) -> str:
        return "noisy_or" if self.base_log_odds is None else "logistic"

    @property
    def log_odds(self) -> float | None:
        if self.base_log_odds is None:
            return None
        return self.base_log_odds + math.fsum(c.log_odds or 0.0 for c in self.features)

    @property
    def base_odds(self) -> float | None:
        return None if self.base_log_odds is None else math.exp(self.base_log_odds)

    @property
    def odds(self) -> float | None:
        z = self.log_odds
        return None if z is None else math.exp(z)

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
        return any(j.inseparable for j in self.joint_causes)

    def __str__(self) -> str:
        return self.text or repr(self)

    @classmethod
    def _from(cls, d: Mapping[str, Any] | None) -> Explanation | None:
        if not d:
            return None
        return cls(
            dt=_f(d["dt"]), kernel_value=_f(d["kernel_value"]), weight=_f(d["weight"]),
            link_strength=_f(d["link_strength"]),
            features=tuple(FeatureContribution(c["feature"], _f(c["strength"]), _f(c["alpha"]), _f(c["beta"]),
                                               _fo(c.get("log_odds")))
                           for c in d.get("features", ())),
            warnings=tuple(d.get("warnings", ())),
            rules=tuple(AppliedRule(r["name"], _f(r["multiplier"])) for r in d.get("rules", ())),
            joint_causes=tuple(JointCause(tuple(j["features"]), _f(j["strength"]), BetaCounts._from(j.get("joint")),
                                          bool(j.get("inseparable"))) for j in d.get("joint_causes", ())),
            entity=d.get("entity", ""), prospective=bool(d.get("prospective")), text=d.get("text", ""),
            base_log_odds=_fo(d.get("base_log_odds")),
        )


@dataclass(frozen=True)
class Cause:
    rank: int
    event: Event
    score: float
    explanation: Explanation | None = None

    @property
    def entity(self) -> str:
        return self.event.entity


@dataclass(frozen=True)
class QueryStats:
    segments_in_window: int = 0
    segments_scanned: int = 0
    segments_pruned: int = 0
    events_scanned: int = 0
    union_fallbacks: int = 0


@dataclass(frozen=True)
class QueryResult:
    entity: str
    effect: str
    at: float
    causes: list[Cause]
    stats: QueryStats
    entities: tuple[str, ...]
    text: str = field(default="", repr=False, compare=False)

    def __iter__(self) -> Iterator[Cause]:
        return iter(self.causes)

    def __len__(self) -> int:
        return len(self.causes)

    def __getitem__(self, i: int) -> Cause:
        return self.causes[i]

    @property
    def top(self) -> Cause | None:
        return self.causes[0] if self.causes else None

    @property
    def is_group(self) -> bool:
        return len(self.entities) > 1

    def __str__(self) -> str:
        return self.text or repr(self)

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> QueryResult:
        return cls(
            entity=d.get("entity", ""), effect=d.get("effect", ""), at=_f(d.get("at", 0.0)),
            causes=[Cause(int(c["rank"]), _event(c["event"]), _f(c["score"]), Explanation._from(c.get("explanation")))
                    for c in d["causes"]],
            stats=QueryStats(**{k: int(v) for k, v in d.get("stats", {}).items()
                                if k in QueryStats.__dataclass_fields__}),
            entities=tuple(d.get("entities", ())), text=d.get("text", ""),
        )


@dataclass(frozen=True)
class RiskResult:
    effect: str
    entity: str | None
    features: frozenset[str]
    type: str | None
    score: float
    explanation: Explanation | None = None

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> RiskResult:
        return cls(d.get("effect", ""), d.get("entity"), frozenset(d.get("features", ())), d.get("type"),
                   _f(d["score"]), Explanation._from(d.get("explanation")))


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    candidate: Candidate
    score: float
    entity: str | None
    explanation: Explanation | None = None


@dataclass(frozen=True)
class FeedbackReport:
    confirmed: int | None
    rejected: tuple[int, ...]
    credit: dict[str, float]


@dataclass(frozen=True)
class EventCredit:
    event_id: int
    score: float
    responsibility: float
    features: dict[str, float]


@dataclass(frozen=True)
class ObserveReport:
    entity: str
    effect: str
    at: float
    leak: float
    leak_responsibility: float
    credits: tuple[EventCredit, ...]
    text: str = field(default="", repr=False, compare=False)

    @property
    def total_credit(self) -> float:
        return math.fsum(c.responsibility for c in self.credits)

    def __str__(self) -> str:
        return self.text or repr(self)

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> ObserveReport:
        return cls(d["entity"], d["effect"], _f(d["at"]), _f(d["leak"]), _f(d["leak_responsibility"]),
                   tuple(EventCredit(int(c["event_id"]), _f(c["score"]), _f(c["responsibility"]),
                                     {k: _f(v) for k, v in c["features"].items()}) for c in d["credits"]),
                   d.get("text", ""))


@dataclass(frozen=True)
class SettleReport:
    until: float
    events_settled: int
    events_rejected: int
    rejected_weight: float
    skipped_credited: int
    skipped_observed: int
    per_effect: dict[str, int]
    text: str = field(default="", repr=False, compare=False)

    def __str__(self) -> str:
        return self.text or repr(self)

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> SettleReport:
        return cls(_f(d["until"]), int(d["events_settled"]), int(d["events_rejected"]), _f(d["rejected_weight"]),
                   int(d["skipped_credited"]), int(d["skipped_observed"]),
                   {k: int(v) for k, v in d.get("per_effect", {}).items()}, d.get("text", ""))


@dataclass(frozen=True)
class LinkRow:
    feature: str
    strength: float
    alpha: float
    beta: float
    n: float
    lo: float
    hi: float
    lift: float
    log_odds: float | None = None    # logistic only


@dataclass(frozen=True)
class LinkReport:
    effect: str
    entity: str | None
    baseline: float
    level: float
    rows: tuple[LinkRow, ...]
    text: str = field(default="", repr=False, compare=False)

    def __iter__(self) -> Iterator[LinkRow]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def row(self, feature: str) -> LinkRow | None:
        return next((r for r in self.rows if r.feature == feature), None)

    def __str__(self) -> str:
        return self.text or repr(self)

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> LinkReport:
        return cls(d["effect"], d.get("entity"), _f(d["baseline"]), _f(d["level"]),
                   tuple(LinkRow(r["feature"], *(_f(r[k]) for k in ("strength", "alpha", "beta", "n", "lo", "hi",
                                                                       "lift")), log_odds=_fo(r.get("log_odds")))
                         for r in d["rows"]),
                   d.get("text", ""))


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_predicted: float | None
    observed_rate: float | None


@dataclass(frozen=True)
class CalibrationReport:
    effect: str
    n: int
    brier: float | None
    bins: tuple[CalibrationBin, ...]
    text: str = field(default="", repr=False, compare=False)

    def __str__(self) -> str:
        return self.text or repr(self)

    @classmethod
    def _from(cls, d: Mapping[str, Any]) -> CalibrationReport:
        return cls(d["effect"], int(d["n"]), _fo(d.get("brier")),
                   tuple(CalibrationBin(_f(b["lo"]), _f(b["hi"]), int(b["count"]), _fo(b.get("mean_predicted")),
                                        _fo(b.get("observed_rate"))) for b in d["bins"]),
                   d.get("text", ""))


# ---------------------------------------------------------------------------
# Connection URL
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    base_url: str
    api_key: str
    database_id: str


def parse_url(url: str) -> tuple[str, str, str]:
    """``heapdb://key@host/db`` -> ``(base_url, api_key, database_id)``."""
    p = urllib.parse.urlsplit(url.strip())
    schemes = {"heapdb": "https", "heapdb+https": "https", "heapdb+http": "http"}
    if p.scheme not in schemes:
        raise ValueError("connection URL must look like heapdb://<api_key>@<host>/<database_id>")
    key = urllib.parse.unquote(p.username or "")
    if p.password:
        key = f"{key}:{urllib.parse.unquote(p.password)}"
    host = p.hostname or ""
    db = p.path.strip("/")
    if not key or not host or not db or "/" in db:
        raise ValueError("connection URL must look like heapdb://<api_key>@<host>/<database_id>")
    netloc = f"[{host}]" if ":" in host else host
    if p.port:
        netloc += f":{p.port}"
    return f"{schemes[p.scheme]}://{netloc}", key, db


def _resolve(url_or_env: str | None) -> str:
    if url_or_env is None:
        url_or_env = "HEAPDB_URL"
    if "://" in url_or_env:
        return url_or_env
    v = os.environ.get(url_or_env)
    if not v:
        raise ValueError(f"no connection URL: pass one or set {url_or_env}")
    return v


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def _features(features: Iterable[str] | str | None) -> list[str]:
    if features is None:
        return []
    if isinstance(features, str):
        return [f.strip() for f in features.split(",") if f.strip()]
    return sorted({str(f) for f in features})


def _kernel_spec(kernel: Any) -> dict[str, Any]:
    if isinstance(kernel, Mapping):
        d = dict(kernel)
        if "type" in d:
            return {"type": d["type"], "params": dict(d.get("params") or {})}
        kind = d.pop("kind")
        return {"type": kind, "params": d}
    to_dict = getattr(kernel, "to_dict", None)
    if to_dict is None:
        raise TypeError("kernel must be a heapdb Kernel or {'type': ..., 'params': {...}}")
    d = dict(to_dict())
    kind = d.pop("kind")
    return {"type": kind, "params": d}


def _pair(p: Any) -> list[float]:
    a, b = p
    return [float(a), float(b)]


class RemoteHeapDB:
    """A database on a HeapDB server; mirrors :class:`heapdb.HeapDB`."""

    def __init__(self, base_url: str, api_key: str, database_id: str, *, timeout: float = 30.0,
                 retries: int = 3, backoff: float = 0.5, max_backoff: float = 30.0,
                 user_agent: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.database_id = database_id
        self._key = api_key
        self.timeout = timeout
        self.retries = max(0, int(retries))
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.user_agent = user_agent or f"heapdb-python/{_VERSION}"

    def __repr__(self) -> str:
        return f"RemoteHeapDB({self.base_url}/v1/databases/{self.database_id}, key=hdb_***)"

    def __enter__(self) -> RemoteHeapDB:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Nothing to release (connections are per request); kept for symmetry."""

    # -- transport ------------------------------------------------------------

    def _request(self, method: str, path: str, body: Any = None, query: Mapping[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/v1/databases/{urllib.parse.quote(self.database_id, safe='')}{path}"
        if query:
            q = {k: v for k, v in query.items() if v is not None}
            if q:
                url += "?" + urllib.parse.urlencode(q)
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        headers = {"Authorization": f"Bearer {self._key}", "Accept": "application/json",
                   "User-Agent": self.user_agent}
        if data is not None:
            headers["Content-Type"] = "application/json"
        attempt = 0
        while True:
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                err = self._http_error(e)
                retryable = err.status in (502, 503, 504) or (err.status == 429 and err.code == "rate_limited")
                if not retryable or attempt >= self.retries:
                    raise err from None
                delay = err.retry_after if err.retry_after is not None else self._delay(attempt)
            except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
                if attempt >= self.retries:
                    reason = getattr(e, "reason", e)
                    raise RemoteConnectionError(f"cannot reach {self.base_url}: {reason}",
                                                code="connection_error") from None
                delay = self._delay(attempt)
            attempt += 1
            time.sleep(min(delay, max(self.max_backoff, 60.0)))

    def _delay(self, attempt: int) -> float:
        d = min(self.max_backoff, self.backoff * (2 ** attempt))
        return d * (0.5 + random.random() / 2)

    @staticmethod
    def _http_error(e: urllib.error.HTTPError) -> HeapDBError:
        raw = e.read() or b""
        code, message = "error", raw.decode("utf-8", "replace") or (e.reason or "")
        try:
            j = json.loads(raw)
            code = j["error"]["code"]
            message = j["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        ra = e.headers.get("Retry-After") if e.headers else None
        retry_after = None
        if ra:
            try:
                retry_after = float(ra)
            except ValueError:
                retry_after = None
        rid = e.headers.get("x-request-id") if e.headers else None
        return _error_for(e.code, code, message, retry_after, rid)

    def _post(self, path: str, body: Any) -> Any:
        return self._request("POST", path, body)

    # -- info ---------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        """``{database_id, plan, usage, limits}`` (limits: ``None`` = unlimited)."""
        return self._request("GET", "")

    # -- schema --------------------------------------------------------------------

    def register_effect(self, name: str, kernel: Any, priors: Mapping[str, tuple[float, float]] | None = None,
                        default_prior: tuple[float, float] = (1.0, 4.0), cause_types: Iterable[str] | None = None,
                        credit: str | None = None, global_pseudocount: float = 5.0, hierarchical: bool = True,
                        leak: float = 0.05, w_neg: float = 1.0, repeatable: bool = True,
                        combine: str = "noisy_or", learning_rate: float | None = None,
                        l2: float | None = None) -> None:
        """Declare (or replace) an effect. ``kernel``: a heapdb kernel object
        (``LogNormalKernel(...)``) or ``{"type", "params"}``. ``combine``:
        ``"noisy_or"`` (default) or ``"logistic"`` (``learning_rate``, ``l2``
        apply to logistic only). ``credit`` defaults to ``"proportional"``
        for noisy-OR and ``"uniform"`` for logistic."""
        body: dict[str, Any] = {
            "name": name, "kernel": _kernel_spec(kernel), "default_prior": _pair(default_prior),
            "global_pseudocount": float(global_pseudocount), "hierarchical": bool(hierarchical),
            "leak": _num_out(float(leak)), "w_neg": float(w_neg), "repeatable": bool(repeatable),
        }
        if credit is not None:
            body["credit"] = credit
        if combine != "noisy_or":
            body["combine"] = combine
        if learning_rate is not None:
            body["learning_rate"] = float(learning_rate)
        if l2 is not None:
            body["l2"] = float(l2)
        if priors:
            body["priors"] = {str(k): _pair(v) for k, v in priors.items()}
        if cause_types is not None:
            body["cause_types"] = sorted(cause_types) if not isinstance(cause_types, str) else [cause_types]
        self._post("/effects", body)

    def register_event_type(self, type: str, weight: Any = None) -> None:  # noqa: A002 - mirrors HeapDB
        """Declare an event type; ``weight``: :func:`weight_attr`, :func:`weight_log1p`,
        :func:`weight_const` (or a number). Callables are rejected (no code upload)."""
        body: dict[str, Any] = {"type": type}
        spec = _weight_spec(weight)
        if spec is not None:
            body["weight"] = spec
        self._post("/event-types", body)

    def add_rule(self, effect: str, name: str, predicate: Any, multiplier: float,
                 types: Iterable[str] | None = None) -> None:
        """Add (or replace) a rule; ``predicate``: ``None`` (type-level) or :func:`where`."""
        body: dict[str, Any] = {"effect": effect, "name": name, "multiplier": float(multiplier)}
        spec = _predicate_spec(predicate)
        if spec is not None:
            body["predicate"] = spec
        if types is not None:
            body["types"] = sorted(types) if not isinstance(types, str) else [types]
        self._post("/rules", body)

    # -- writes ------------------------------------------------------------------

    @staticmethod
    def _event_body(entity: str, time: float, type: str, features: Any = None, weight: float | None = None,
                    attrs: Mapping[str, Any] | None = None, external_id: str | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {"entity": str(entity), "time": float(time), "type": str(type),
                             "features": _features(features)}
        if weight is not None:
            d["weight"] = float(weight)
        if attrs:
            d["attrs"] = dict(attrs)
        if external_id is not None:
            d["external_id"] = str(external_id)
        return d

    def insert(self, entity: str, time: float, type: str, features: Iterable[str] | str | None = None,
               weight: float | None = None, attrs: Mapping[str, Any] | None = None, *,
               external_id: str | None = None) -> Event:
        """Append one event and return it. ``external_id`` makes the insert
        idempotent (a repeat returns the stored event)."""
        body = {"events": [self._event_body(entity, time, type, features, weight, attrs, external_id)],
                "return_events": True}
        r = self._post("/events", body)
        return _event(r["events"][0])

    def insert_many(self, events: Iterable[Mapping[str, Any] | Sequence[Any]], *,
                    batch_size: int = MAX_BATCH) -> list[int]:
        """Append many events (dicts with ``entity, time, type, features?, weight?,
        attrs?, external_id?`` or tuples in ``insert`` order), in batches of at
        most 1000. Returns the ids in input order."""
        size = max(1, min(int(batch_size), MAX_BATCH))
        ids: list[int] = []
        batch: list[dict[str, Any]] = []

        def flush() -> None:
            if batch:
                ids.extend(int(i) for i in self._post("/events", {"events": batch})["ids"])
                batch.clear()

        for e in events:
            if isinstance(e, Mapping):
                batch.append(self._event_body(e["entity"], e["time"], e["type"], e.get("features"), e.get("weight"),
                                              e.get("attrs"), e.get("external_id")))
            else:
                batch.append(self._event_body(*e))
            if len(batch) >= size:
                flush()
        flush()
        return ids

    def feedback(self, entity: str, effect_type: str, at_time: float, confirmed_event_id: int | None = None,
                 rejected_ids: Iterable[int] = (), credit: str | None = None) -> FeedbackReport:
        body: dict[str, Any] = {"entity": entity, "effect": effect_type, "at": float(at_time),
                                "rejected_ids": [int(i) for i in rejected_ids]}
        if confirmed_event_id is not None:
            body["confirmed_event_id"] = int(confirmed_event_id)
        if credit:
            body["credit"] = credit
        r = self._post("/feedback", body)
        return FeedbackReport(r.get("confirmed"), tuple(r.get("rejected", ())),
                              {k: _f(v) for k, v in r.get("credit", {}).items()})

    def observe(self, entity: str, effect: str, at: float, leak: float | None = None) -> ObserveReport:
        body: dict[str, Any] = {"entity": entity, "effect": effect, "at": float(at)}
        if leak is not None:
            body["leak"] = _num_out(float(leak))
        return ObserveReport._from(self._post("/observe", body))

    def settle(self, until: float, effects: Iterable[str] | str | None = None, entity: str | None = None,
               partial: bool = False) -> SettleReport:
        body: dict[str, Any] = {"until": float(until), "partial": bool(partial)}
        if effects is not None:
            body["effects"] = [effects] if isinstance(effects, str) else list(effects)
        if entity is not None:
            body["entity"] = entity
        return SettleReport._from(self._post("/settle", body))

    def forget_entity(self, entity: str, scrub_global: bool = False) -> dict[str, int]:
        """Remove everything about ``entity``; returns the removed counts."""
        path = "/entities/" + urllib.parse.quote(entity, safe="")
        r = self._request("DELETE", path, query={"scrub_global": "true" if scrub_global else "false"})
        return {k: int(v) for k, v in r["counts"].items()}

    # -- queries --------------------------------------------------------------------

    def top_causes(self, entity: str, effect: str, at: float, k: int = 3, explain: bool = False,
                   within: float | None = None) -> QueryResult:
        body: dict[str, Any] = {"entity": entity, "effect": effect, "at": float(at), "k": int(k),
                                "explain": bool(explain)}
        if within is not None:
            body["within"] = float(within)
        return QueryResult._from(self._post("/query/top-causes", body))

    def top_causes_group(self, entities: Iterable[str], effect: str, at: float, k: int = 3,
                         explain: bool = False, within: float | None = None) -> QueryResult:
        body: dict[str, Any] = {"entities": list(entities), "effect": effect, "at": float(at), "k": int(k),
                                "explain": bool(explain)}
        if within is not None:
            body["within"] = float(within)
        return QueryResult._from(self._post("/query/top-causes", body))

    def most_likely_cause(self, entity: str, effect: str, at: float) -> Cause | None:
        return self.top_causes(entity, effect, at, k=1, explain=True).top

    def feature_risk(self, effect: str, features: Iterable[str] | str, weight: float | None = 1.0,
                     entity: str | None = None, type: str | None = None,  # noqa: A002
                     attrs: Mapping[str, Any] | None = None, explain: bool = False) -> Any:
        """Risk of a not-yet-happened candidate; ``(score, Explanation)`` with ``explain``."""
        body: dict[str, Any] = {"effect": effect, "features": _features(features), "explain": bool(explain)}
        if weight is not None:
            body["weight"] = float(weight)
        if entity is not None:
            body["entity"] = entity
        if type is not None:
            body["type"] = type
        if attrs:
            body["attrs"] = dict(attrs)
        r = RiskResult._from(self._post("/query/risk", body))
        return (r.score, r.explanation) if explain else r.score

    def rank_candidates(self, effect: str, candidates: Iterable[Candidate | Mapping[str, Any]], k: int = 10,
                        entity: str | None = None, explain: bool = False) -> list[RankedCandidate]:
        cands = [Candidate.from_any(c, i) for i, c in enumerate(candidates)]
        wire = []
        for c in cands:
            d: dict[str, Any] = {"id": c.id if isinstance(c.id, (str, int, float, bool)) or c.id is None else str(c.id),
                                 "features": sorted(c.features)}
            if c.weight is not None:
                d["weight"] = float(c.weight)
            if c.type is not None:
                d["type"] = c.type
            if c.attrs:
                d["attrs"] = dict(c.attrs)
            if c.entity is not None:
                d["entity"] = c.entity
            wire.append(d)
        body: dict[str, Any] = {"effect": effect, "candidates": wire, "k": int(k), "explain": bool(explain)}
        if entity is not None:
            body["entity"] = entity
        r = self._post("/query/risk", body)
        return [RankedCandidate(int(x["rank"]), cands[int(x["index"])], _f(x["score"]), x.get("entity"),
                                Explanation._from(x.get("explanation"))) for x in r["candidates"]]

    def link_report(self, effect: str, entity: str | None = None, min_support: float = 1.0, level: float = 0.9,
                    baseline: float | str = "mean", features: Iterable[str] | None = None) -> LinkReport:
        q: dict[str, Any] = {"effect": effect, "entity": entity, "min_support": repr(float(min_support)),
                             "level": repr(float(level)),
                             "baseline": baseline if isinstance(baseline, str) else repr(float(baseline))}
        if features is not None:
            q["features"] = ",".join(features)
        return LinkReport._from(self._request("GET", "/links", query=q))

    def calibration_report(self, effect: str, bins: int = 10) -> CalibrationReport:
        return CalibrationReport._from(self._request("GET", "/calibration", query={"effect": effect,
                                                                                    "bins": int(bins)}))

    def execute(self, query: str, now: float | None = None) -> Any:
        """Run one query-language statement on the server (``now`` resolves
        ``now``/``now-2h``; default: server time). Returns the same result
        types as the methods (a ``str`` for INSERT / FEEDBACK)."""
        body: dict[str, Any] = {"query": query}
        if now is not None:
            body["now"] = float(now)
        r = self._post("/hql", body)
        kind, res = r["kind"], r["result"]
        return {
            "top_causes": QueryResult._from, "calibration": CalibrationReport._from, "risk": RiskResult._from,
            "links": LinkReport._from, "observe": ObserveReport._from, "settle": SettleReport._from,
        }.get(kind, lambda x: x)(res)


def connect(url_or_env: str | None = None, *, timeout: float = 30.0, retries: int = 3, backoff: float = 0.5,
            max_backoff: float = 30.0, user_agent: str | None = None) -> RemoteHeapDB:
    """Connect to a database on a HeapDB server.

    ``url_or_env``: a connection URL (``heapdb://<api_key>@<host>/<database_id>``,
    or ``heapdb+http://...`` for plain HTTP), or the name of an environment
    variable holding one; default ``HEAPDB_URL``.
    """
    base, key, db = parse_url(_resolve(url_or_env))
    return RemoteHeapDB(base, key, db, timeout=timeout, retries=retries, backoff=backoff,
                        max_backoff=max_backoff, user_agent=user_agent)
