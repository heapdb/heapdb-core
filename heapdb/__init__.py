"""HeapDB: pop the most likely cause, right now.

A reference implementation of a database whose primary operation is ranked,
explainable attribution over an append-only event stream.

    >>> from heapdb import HeapDB, LogNormalKernel
    >>> db = HeapDB(bucket_seconds=900)
    >>> _ = db.register_effect("bloating", LogNormalKernel(median=7200, sigma=0.6),
    ...                        priors={"beans": (3, 1)})
    >>> _ = db.insert("me", 0, "meal", {"beans", "onion"}, weight=1.5)
    >>> db.top_causes("me", "bloating", at=7200, k=1).top.event.features == {"beans", "onion"}
    True
"""

from .events import Event
from .forward_decay import ForwardDecayHeap
from .kernels import ExponentialDecay, GammaKernel, Kernel, LogNormalKernel, WindowKernel
from .links import BetaCounts, LinkTable, noisy_or
from .query import QueryError, execute, format_result, parse
from .segments import EntityLog, Segment
from .store import (
    Cause,
    EffectSpec,
    Explanation,
    FeatureContribution,
    HeapDB,
    QueryResult,
    QueryStats,
)

__version__ = "0.1.0"

__all__ = [
    "BetaCounts",
    "Cause",
    "EffectSpec",
    "EntityLog",
    "Event",
    "Explanation",
    "ExponentialDecay",
    "FeatureContribution",
    "ForwardDecayHeap",
    "GammaKernel",
    "HeapDB",
    "Kernel",
    "LinkTable",
    "LogNormalKernel",
    "QueryError",
    "QueryResult",
    "QueryStats",
    "Segment",
    "WindowKernel",
    "execute",
    "format_result",
    "noisy_or",
    "parse",
]
