"""ForwardDecayHeap: a priority queue of exponentially decaying scores.

The problem
-----------
We want "pop the item with the highest *current* score", where every item's
score decays over time: ``score(now) = w * exp(-lam * (now - t))``. Naively the
scores change every tick, so a heap would need re-keying constantly.

The trick (forward decay, Cormode et al. 2009)
----------------------------------------------
Rewrite the score in log space::

    log score(now) = log w + lam * t  -  lam * now
                     \\____ key ____/     \\_ same for every item _/

The ``key = log(w) + lam * t`` is fixed at insert time, and the term that
depends on ``now`` is *identical for every item*. Subtracting the same
constant never changes the order, so the heap ordering by ``key`` is valid at
every point in time. We only convert back to a real score when someone asks
(``score = exp(key - lam * now)``). Working in log space also avoids the
overflow you would hit with the textbook ``w * exp(lam * t)`` landmark form.

Deletion uses tombstones: ``remove(handle)`` just marks the handle dead, and
dead entries are discarded lazily when they surface at the top of the heap.
This keeps ``remove`` O(1) and ``pop`` amortised O(log n).

This structure is independent of :class:`heapdb.HeapDB`; it demonstrates the
ordering invariance that makes "pop the top right now" cheap for the special
case of a single exponential kernel (e.g. an always-up-to-date "most likely
suspect" per entity).
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Any, Generic, Iterator, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Popped(Generic[T]):
    item: T
    score: float  # score evaluated at the ``now`` passed to pop/peek
    handle: int


class ForwardDecayHeap(Generic[T]):
    """Max-heap of items whose scores decay as ``w * exp(-lam * age)``.

    Args:
        lam: Decay rate per second (``lam = ln 2 / half_life``).
    """

    def __init__(self, lam: float) -> None:
        if lam < 0:
            raise ValueError("lam must be >= 0")
        self.lam = float(lam)
        # heapq is a min-heap, so we store the *negated* key.
        self._heap: list[tuple[float, int, T]] = []
        self._dead: set[int] = set()       # tombstoned handles
        self._live: dict[int, float] = {}  # handle -> key, for live items
        self._ids = itertools.count()

    @classmethod
    def from_half_life(cls, half_life: float) -> "ForwardDecayHeap[Any]":
        return cls(math.log(2.0) / half_life)

    # -- core operations ----------------------------------------------------

    def key(self, weight: float, t: float) -> float:
        """Time-invariant priority: ``log(w) + lam * t``."""
        if weight <= 0:
            raise ValueError("weight must be > 0 (log of 0 is undefined)")
        return math.log(weight) + self.lam * t

    def score_at(self, key: float, now: float) -> float:
        """Convert a key back into a real score at time ``now``."""
        return math.exp(key - self.lam * now)

    def push(self, item: T, weight: float, t: float) -> int:
        """Insert ``item`` observed at time ``t`` with weight ``w``.

        Returns a handle that can be passed to :meth:`remove`.
        """
        k = self.key(weight, t)
        handle = next(self._ids)
        # Tuple order: (-key, handle, item). The handle breaks ties so items
        # themselves never need to be comparable.
        heapq.heappush(self._heap, (-k, handle, item))
        self._live[handle] = k
        return handle

    def remove(self, handle: int) -> bool:
        """Tombstone an entry. O(1); the entry is physically dropped later."""
        if handle in self._live:
            del self._live[handle]
            self._dead.add(handle)
            return True
        return False

    def _skip_dead(self) -> None:
        while self._heap and self._heap[0][1] in self._dead:
            _, handle, _ = heapq.heappop(self._heap)
            self._dead.discard(handle)

    def peek_top(self, now: float) -> Popped[T] | None:
        """Highest-scoring live item at time ``now`` (without removing it)."""
        self._skip_dead()
        if not self._heap:
            return None
        neg_k, handle, item = self._heap[0]
        return Popped(item, self.score_at(-neg_k, now), handle)

    def pop(self, now: float) -> Popped[T] | None:
        """Remove and return the highest-scoring live item at time ``now``."""
        self._skip_dead()
        if not self._heap:
            return None
        neg_k, handle, item = heapq.heappop(self._heap)
        del self._live[handle]
        return Popped(item, self.score_at(-neg_k, now), handle)

    def top_k(self, k: int, now: float) -> list[Popped[T]]:
        """The ``k`` best live items at ``now``, best first, without removing."""
        best = heapq.nsmallest(
            k, (e for e in self._heap if e[1] not in self._dead)
        )
        return [Popped(item, self.score_at(-neg_k, now), h) for neg_k, h, item in best]

    # -- housekeeping -------------------------------------------------------

    def compact(self) -> None:
        """Physically drop all tombstoned entries (O(n))."""
        self._heap = [e for e in self._heap if e[1] not in self._dead]
        heapq.heapify(self._heap)
        self._dead.clear()

    def __len__(self) -> int:
        return len(self._live)

    def __iter__(self) -> Iterator[T]:
        for _, handle, item in self._heap:
            if handle not in self._dead:
                yield item
