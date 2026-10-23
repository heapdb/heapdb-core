"""Combine formulations tried for HeapDB's logistic mode, on the development
slices only (first 25% of each stream).

Pure-Python reference implementations of the candidate ways to combine an
event's feature evidence multiplicatively, run with the benchmark's protocol
(prequential, flat kernel, no dose, confirm the BIC / reject the other
candidates). This is how the engine's formulation was chosen; only the
development slice is used, so the choice does not look at the evaluation
part of the streams. The engine's own runs (``heapdb+logistic*`` in
``run.py``) are the reported results; the ``(c) online LR`` row with
``lr=0.1`` reproduces the engine's default logistic config exactly (the
test suite checks the engine against :class:`FTRL` on a small stream).

Formulations:

* noisy-OR: HeapDB's default (Beta links, EM credit), prior 0.001/10.
* (a) naive Bayes in odds form: ``logit P = b + sum_f (logit s_f - b)``
  with ``s_f`` the Beta link means and ``b`` the default prior's log-odds.
* (a)+(b): the same with a learned intercept ``b`` (Beta over all judged
  events) and feature links shrunk towards it (prior mean = the learned
  base rate, prior strength = the default prior's), with three credit rules
  for the confirmed event (count = every feature +1, EM = noisy-OR
  responsibilities, proportional to each feature's positive log-odds).
* (c) online logistic regression, FTRL-Proximal with L2 towards the seeds
  (intercept seeded at the prior's log-odds), one step per judged event
  (confirmed first) or one batched, order-independent step per bug.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from .core import Case, rank_of, rr

NOR_PRIOR = (0.001, 10.0)


class NaiveBayes:
    """(a) / (a)+(b), and noisy-OR (``kind="noisy_or"``) for reference."""

    def __init__(self, kind: str = "learned", credit: str = "count", shrink: bool = True,
                 prior: tuple[float, float] = NOR_PRIOR):
        self.kind, self.credit, self.shrink = kind, credit, shrink
        self.a0, self.b0 = prior
        self.pos: dict[str, float] = defaultdict(float)
        self.neg: dict[str, float] = defaultdict(float)
        self.P = 0.0
        self.N = 0.0

    def mean(self, f: str) -> float:
        return (self.a0 + self.pos[f]) / (self.a0 + self.b0 + self.pos[f] + self.neg[f])

    def base(self) -> float:
        if self.kind == "fixed":
            return math.log(self.a0) - math.log(self.b0)
        return math.log(self.a0 + self.P) - math.log(self.b0 + self.N)

    def w(self, f: str, bl: float) -> float:
        if self.kind == "fixed" or not self.shrink:
            a, b = self.a0 + self.pos[f], self.b0 + self.neg[f]
        else:
            k = self.a0 + self.b0
            base = 1.0 / (1.0 + math.exp(-bl))
            a, b = k * base + self.pos[f], k * (1.0 - base) + self.neg[f]
        return math.log(a) - math.log(b) - bl

    def score(self, fs: set[str]) -> float:
        if self.kind == "noisy_or":
            miss = 1.0
            for f in sorted(fs, key=self.mean):
                miss *= 1.0 - self.mean(f)
            return 1.0 - miss
        bl = self.base()
        return bl + sum(self.w(f, bl) for f in sorted(fs))

    def learn(self, cands, labels) -> None:
        for c in cands:
            fs = sorted(set(c.cat))
            if c.id not in labels:
                for f in fs:
                    self.neg[f] += 1.0
                self.N += 1.0
                continue
            if self.kind == "noisy_or" or self.credit == "em":
                s = [self.mean(f) for f in fs]
                miss = 1.0
                for x in s:
                    miss *= 1.0 - x
                p = 1.0 - miss
                r = [min(1.0, x / p) if p > 0 else 1.0 / len(fs) for x in s]
            elif self.credit == "logodds":
                bl = self.base()
                ws = [max(0.0, self.w(f, bl)) for f in fs]
                t = sum(ws)
                r = [x / t if t > 0 else 1.0 / len(fs) for x in ws]
            else:
                r = [1.0] * len(fs)
            for f, x in zip(fs, r):
                self.pos[f] += x
                self.neg[f] += 1.0 - x
            self.P += 1.0


class FTRL:
    """(c) FTRL-Proximal logistic regression on binary features, L2 towards
    the seeds; the intercept is seeded at the prior's log-odds (feature
    seeds are 0: no registered priors here). Mirrors the engine's
    ``crates/heapdb/src/logistic.rs``."""

    BIAS = "\x00bias"

    def __init__(self, lr: float = 0.1, l2: float = 1.0, prior: tuple[float, float] = (1.0, 4.0),
                 batch: bool = False, beta: float = 1.0):
        self.lr, self.l2, self.beta, self.batch = lr, l2, beta, batch
        self.seed = math.log(prior[0]) - math.log(prior[1])
        self.z: dict[str, float] = defaultdict(float)
        self.n: dict[str, float] = defaultdict(float)

    def delta(self, k: str) -> float:
        z = self.z.get(k, 0.0)
        if z == 0.0:
            return 0.0
        return -z / ((self.beta + math.sqrt(self.n.get(k, 0.0))) / self.lr + self.l2)

    def score(self, fs: set[str]) -> float:
        return self.seed + self.delta(self.BIAS) + sum(self.delta(f) for f in sorted(fs))

    def _p(self, fs) -> float:
        z = self.score(fs)
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))

    def _apply(self, k: str, g: float, g2: float) -> None:
        d = self.delta(k)
        n0 = self.n.get(k, 0.0)
        sigma = (math.sqrt(n0 + g2) - math.sqrt(n0)) / self.lr
        self.z[k] += g - sigma * d
        self.n[k] = n0 + g2

    def learn(self, cands, labels) -> None:
        pos = [c for c in cands if c.id in labels]
        neg = [c for c in cands if c.id not in labels]
        order = [(c, 1.0) for c in pos[:1]] + [(c, 0.0) for c in neg] + [(c, 1.0) for c in pos[1:]]
        if not self.batch:
            for c, y in order:
                fs = set(c.cat)
                g = self._p(fs) - y
                if g == 0.0:
                    continue
                for k in [self.BIAS, *sorted(fs)]:
                    self._apply(k, g, g * g)
            return
        # one order-independent step per bug: gradients at the pre-update
        # weights, n accumulates the per-event squared gradients
        G: dict[str, float] = defaultdict(float)
        G2: dict[str, float] = defaultdict(float)
        for c, y in order:
            fs = set(c.cat)
            g = self._p(fs) - y
            for k in [self.BIAS, *sorted(fs)]:
                G[k] += g
                G2[k] += g * g
        for k in G:
            self._apply(k, G[k], G2[k])


#: (row, factory); the first row is the reference.
VARIANTS = [
    ("noisy-OR, EM credit (HeapDB default)", lambda: NaiveBayes(kind="noisy_or")),
    ("(a) naive Bayes, base = default prior, count credit", lambda: NaiveBayes(kind="fixed")),
    ("(a)+(b) naive Bayes, learned base, count credit", lambda: NaiveBayes()),
    ("(a)+(b), EM credit", lambda: NaiveBayes(credit="em")),
    ("(a)+(b), credit by positive log-odds", lambda: NaiveBayes(credit="logodds")),
    ("(a)+(b), no shrinkage to the learned base", lambda: NaiveBayes(shrink=False)),
    ("(c) online LR, per event, lr 0.1 (= engine default)", lambda: FTRL(lr=0.1)),
    ("(c) online LR, per event, lr 0.5", lambda: FTRL(lr=0.5)),
    ("(c) online LR, one batched step per bug, lr 0.1", lambda: FTRL(lr=0.1, batch=True)),
    ("(c) online LR, one batched step per bug, lr 0.5", lambda: FTRL(lr=0.5, batch=True)),
    ("(c) online LR, intercept seeded at prior 0.001/10, lr 0.1", lambda: FTRL(lr=0.1, prior=NOR_PRIOR)),
    ("(c) online LR, intercept seeded at prior 0.001/10, lr 0.5", lambda: FTRL(lr=0.5, prior=NOR_PRIOR)),
]


def run_dev(stream: Sequence[Case], dev_n: int) -> dict[str, float]:
    """Dev-slice MRR of every variant (``stream`` in chronological order)."""
    out = {}
    for name, make in VARIANTS:
        m = make()
        tot = 0.0
        for c in stream[:dev_n]:
            v = c.view
            tot += rr(rank_of(v, [m.score(set(x.cat)) for x in v.cands], c.labels))
            m.learn(v.cands, c.labels)
        out[name] = tot / dev_n
    return out
