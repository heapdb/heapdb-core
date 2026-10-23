"""Dependency-free core of the bug-inducing-commit (BIC) evaluation.

Data model, the prequential (time-ordered, test-then-train) protocol,
ranking with an explicit tie rule, metrics, the paired bootstrap,
follow-the-leader model selection and bisection cost. Everything here is
pure Python so the leakage and metric tests run in CI without numpy.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Sequence

KS = (1, 3, 5, 10)


# -- data model -----------------------------------------------------------------

@dataclass(frozen=True)
class Cand:
    """One candidate commit for one bug, as every method sees it.

    ``num`` holds numeric signals (names in ``CaseView.num_names``), ``cat``
    binary/categorical features (``author:<name>``, ``dir:<path>``,
    ``kw:<word>`` and the bucketed numeric signals ``b:<signal>=<bucket>``).
    The bucket features are a fixed re-encoding of ``num``: every method
    receives the same information.
    """

    id: str
    time: float          # commit time (unix seconds)
    pos: int             # recency position among this bug's candidates, 0 = newest
    num: Sequence[float]
    cat: tuple[str, ...]


@dataclass(frozen=True)
class CaseView:
    """What a method may see when it ranks a bug: no label."""

    id: str
    project: str
    time: float           # failure / report time (the effect)
    cands: tuple[Cand, ...]
    num_names: tuple[str, ...]

    def col(self, name: str) -> list[float]:
        j = self.num_names.index(name)
        return [c.num[j] for c in self.cands]


@dataclass(frozen=True)
class Case:
    view: CaseView
    labels: frozenset[str]           # true BIC ids (may be outside the candidates)
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def id(self) -> str:
        return self.view.id

    @property
    def time(self) -> float:
        return self.view.time

    @property
    def in_scope(self) -> bool:
        return any(c.id in self.labels for c in self.view.cands)


def stream_order(cases: Iterable[Case]) -> list[Case]:
    """Chronological order of the effect, ties by id (deterministic)."""
    return sorted(cases, key=lambda c: (c.time, c.id))


# -- ranking ----------------------------------------------------------------------

def rank_of(view: CaseView, scores: Sequence[float], labels: frozenset[str],
            tie: str = "recency") -> int | None:
    """1-based rank of the best-ranked label among the candidates.

    Higher score = more suspicious. ``tie="recency"``: equal scores are
    ordered newest first (every method gets the same deterministic
    tie-break, which is also the ``recency`` baseline). ``tie="max"``: a
    label tied with others gets the worst rank of its tie group (FONTE's
    convention, used only to reproduce its published numbers).
    ``None``: no label is among the candidates (outside the window).
    """
    if len(scores) != len(view.cands):
        raise ValueError("one score per candidate")
    best: int | None = None
    for i, c in enumerate(view.cands):
        if c.id not in labels:
            continue
        s = scores[i]
        if math.isnan(s):
            raise ValueError("NaN score")
        if tie == "max":
            r = sum(1 for x in scores if x >= s)
        elif tie == "recency":
            r = 1 + sum(1 for j, x in enumerate(scores)
                        if x > s or (x == s and view.cands[j].pos < c.pos))
        else:
            raise ValueError(tie)
        best = r if best is None else min(best, r)
    return best


def order(view: CaseView, scores: Sequence[float]) -> list[int]:
    """Candidate indices, best first, ties newest first."""
    return sorted(range(len(view.cands)), key=lambda i: (-scores[i], view.cands[i].pos))


# -- metrics ------------------------------------------------------------------------

def rr(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def summarize(ranks: Sequence[int | None]) -> dict:
    """Acc@k, MRR, mean/median rank. A label outside the candidates counts
    as a miss (RR 0, not in Acc@k); mean/median rank are over in-scope bugs."""
    n = len(ranks)
    inr = [r for r in ranks if r is not None]
    out = {"n": n, "outside": n - len(inr)}
    for k in KS:
        out[f"Acc@{k}"] = sum(1 for r in inr if r <= k) / n if n else float("nan")
    out["MRR"] = sum(rr(r) for r in ranks) / n if n else float("nan")
    out["mean_rank"] = statistics.fmean(inr) if inr else float("nan")
    out["median_rank"] = statistics.median(inr) if inr else float("nan")
    return out


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 10_000,
                     seed: int = 7) -> tuple[float, float, float]:
    """(mean(a - b), 95% CI lo, hi): paired bootstrap over bugs."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples of equal, non-zero length")
    d = [x - y for x, y in zip(a, b)]
    m = len(d)
    try:
        import numpy as np
    except ImportError:          # pure-Python fallback (same estimator, different random stream)
        np = None
    if np is not None:
        # Vectorised, seeded, in chunks of 500 resamples so memory stays small
        # on the 15K-bug DPDK stream.
        rng_np = np.random.default_rng(seed)
        dv = np.asarray(d, dtype=np.float64)
        means = []
        for start in range(0, n, 500):
            k = min(500, n - start)
            means.append(dv[rng_np.integers(0, m, size=(k, m))].mean(axis=1))
        boots = np.sort(np.concatenate(means))
        return statistics.fmean(d), float(boots[int(0.025 * n)]), float(boots[min(n - 1, int(0.975 * n))])
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return statistics.fmean(d), boots[int(0.025 * n)], boots[min(n - 1, int(0.975 * n))]


# -- protocol ------------------------------------------------------------------------

class Method(Protocol):
    name: str

    def reset(self) -> None: ...

    def rank(self, view: CaseView) -> list[float]: ...

    def learn(self, view: CaseView, labels: frozenset[str]) -> None: ...


def prequential(cases: Sequence[Case], method: Method, *, learn: bool = True,
                tie: str = "recency", on_rank: Callable[[Case, list[float]], None] | None = None,
                ) -> list[int | None]:
    """Test-then-train over the chronological stream.

    For each bug: the method ranks the label-free view, the rank is
    recorded, and only then is the label revealed (``learn``). A method can
    therefore never see the label of the bug it ranks or of any later bug.
    """
    method.reset()
    ranks: list[int | None] = []
    for case in stream_order(cases):
        scores = list(method.rank(case.view))
        if on_rank is not None:
            on_rank(case, scores)
        ranks.append(rank_of(case.view, scores, case.labels, tie=tie))
        if learn:
            method.learn(case.view, case.labels)
    return ranks


def follow_the_leader(per_config: dict[str, Sequence[float]], warmup: int = 1) -> tuple[list[float], list[str]]:
    """Prequential model selection over a fixed grid.

    ``per_config[c][i]`` is the reciprocal rank config ``c`` got on bug ``i``
    in its own prequential run (which used only bugs ``< i``). For bug ``i``
    we use the config with the best mean RR on bugs ``< i`` (ties: grid
    order; before ``warmup`` bugs: the first config). Leak-free: the choice
    for bug ``i`` depends only on outcomes of earlier bugs.
    """
    names = list(per_config)
    n = len(per_config[names[0]])
    sums = {c: 0.0 for c in names}
    chosen_rr, chosen = [], []
    for i in range(n):
        if i < warmup:
            c = names[0]
        else:
            c = max(names, key=lambda k: (sums[k], -names.index(k)))
        chosen.append(c)
        chosen_rr.append(per_config[c][i])
        for k in names:
            sums[k] += per_config[k][i]
    return chosen_rr, chosen


# -- bisection cost (port of FONTE's experiment_utils, MIT, COINSE KAIST) ------------

def standard_bisection(n: int, bic_index: int) -> int:
    """Test runs ``git bisect`` needs over ``n`` commits ordered newest first
    (index 0 known bad); ``bic_index`` = position of the BIC."""
    bad, good, steps = 0, n, 0
    while good > bad + 1:
        steps += 1
        pivot = (bad + good) // 2
        if pivot > bic_index:
            good = pivot
        else:
            bad = pivot
    assert bad == bic_index
    return steps


def weighted_bisection(scores_newest_first: Sequence[float], bic_index: int) -> int:
    """FONTE's score-weighted bisection: pivots split the remaining score
    mass in half. Commits with score <= 0 are skipped (as in FONTE), so the
    BIC must have a positive score."""
    idx = [i for i, s in enumerate(scores_newest_first) if s > 0]
    s = [scores_newest_first[i] for i in idx]
    b = idx.index(bic_index)
    pre = [0.0]
    for x in s:
        pre.append(pre[-1] + x)
    bad, good, steps = 0, len(s), 0
    while good > bad + 1:
        steps += 1
        best, pivot = None, None
        for i in range(bad + 1, good):
            diff = abs((pre[i] - pre[bad]) - (pre[good] - pre[i]))
            if best is None or diff < best:
                best, pivot = diff, i
            else:
                break
        if pivot > b:
            good = pivot
        else:
            bad = pivot
    assert bad == b
    return steps
