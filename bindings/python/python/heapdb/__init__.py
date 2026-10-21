"""HeapDB: pop the most likely cause, right now.

Ranked, explainable attribution over an append-only event stream. The engine
is written in Rust (crate ``heapdb``); this package is its Python binding.

    >>> from heapdb import HeapDB, LogNormalKernel
    >>> db = HeapDB(bucket_seconds=900)
    >>> _ = db.register_effect("bloating", LogNormalKernel(median=7200, sigma=0.6),
    ...                        priors={"beans": (3, 1)})
    >>> _ = db.insert("me", 0, "meal", {"beans", "onion"}, weight=1.5)  # weight = dose
    >>> db.top_causes("me", "bloating", at=7200, k=1).top.event.features == {"beans", "onion"}
    True

A database on a HeapDB server (hosted or self-hosted) has the same API:

    >>> remote = heapdb.connect("heapdb://hdb_live_...@api.heapdb.com/db_...")  # doctest: +SKIP
"""

from ._core import (
    AppliedRule,
    BetaCounts,
    CalibrationBin,
    CalibrationRecord,
    CalibrationReport,
    Candidate,
    Cause,
    EffectSpec,
    EntityLog,
    Event,
    EventCredit,
    Explanation,
    ExponentialDecay,
    FeatureContribution,
    FeedbackReport,
    ForwardDecayHeap,
    GammaKernel,
    HeapDB,
    JointCause,
    Kernel,
    LinkReport,
    LinkRow,
    LinkTable,
    LogNormalKernel,
    ObservationState,
    ObserveReport,
    QueryError,
    QueryResult,
    QueryStats,
    RankedCandidate,
    RiskResult,
    Rule,
    Segment,
    SettleReport,
    WindowKernel,
    beta_interval,
    beta_ppf,
    betainc,
    execute,
    format_result,
    noisy_or,
    parse,
    responsibilities,
    spearman,
)

from .remote import HeapDBError, RemoteHeapDB, connect  # noqa: E402

__version__ = "0.5.0"

__all__ = [
    "AppliedRule",
    "BetaCounts",
    "CalibrationBin",
    "CalibrationRecord",
    "CalibrationReport",
    "Candidate",
    "Cause",
    "EffectSpec",
    "EntityLog",
    "Event",
    "EventCredit",
    "Explanation",
    "ExponentialDecay",
    "FeatureContribution",
    "FeedbackReport",
    "ForwardDecayHeap",
    "GammaKernel",
    "HeapDB",
    "HeapDBError",
    "JointCause",
    "Kernel",
    "LinkReport",
    "LinkRow",
    "LinkTable",
    "LogNormalKernel",
    "ObservationState",
    "ObserveReport",
    "QueryError",
    "QueryResult",
    "QueryStats",
    "RankedCandidate",
    "RemoteHeapDB",
    "RiskResult",
    "Rule",
    "Segment",
    "SettleReport",
    "WindowKernel",
    "beta_interval",
    "beta_ppf",
    "betainc",
    "connect",
    "execute",
    "format_result",
    "noisy_or",
    "parse",
    "responsibilities",
    "spearman",
]
