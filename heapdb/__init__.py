"""HeapDB: pop the most likely cause, right now.

A reference implementation of a database whose primary operation is ranked,
explainable attribution over an append-only event stream.

    >>> from heapdb import HeapDB, LogNormalKernel
    >>> db = HeapDB(bucket_seconds=900)
    >>> _ = db.register_effect("bloating", LogNormalKernel(median=7200, sigma=0.6),
    ...                        priors={"beans": (3, 1)})
    >>> _ = db.insert("me", 0, "meal", {"beans", "onion"}, weight=1.5)  # weight = dose
    >>> db.top_causes("me", "bloating", at=7200, k=1).top.event.features == {"beans", "onion"}
    True
"""

from .events import Event
from .forward_decay import ForwardDecayHeap
from .kernels import ExponentialDecay, GammaKernel, Kernel, LogNormalKernel, WindowKernel
from .links import BetaCounts, LinkTable, noisy_or, responsibilities
from .query import QueryError, execute, format_result, parse
from .segments import EntityLog, Segment
from .store import (
    AppliedRule,
    CalibrationBin,
    CalibrationRecord,
    CalibrationReport,
    Candidate,
    Cause,
    EffectSpec,
    EventCredit,
    Explanation,
    FeatureContribution,
    FeedbackReport,
    HeapDB,
    JointCause,
    LinkReport,
    LinkRow,
    ObservationState,
    ObserveReport,
    QueryResult,
    QueryStats,
    RankedCandidate,
    RiskResult,
    Rule,
    SettleReport,
)
from .stats import beta_interval, beta_ppf, betainc, spearman

__version__ = "0.4.0"

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
    "RiskResult",
    "Rule",
    "Segment",
    "SettleReport",
    "WindowKernel",
    "beta_interval",
    "beta_ppf",
    "betainc",
    "execute",
    "format_result",
    "noisy_or",
    "parse",
    "responsibilities",
    "spearman",
]
