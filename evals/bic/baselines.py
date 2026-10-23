"""Baselines over the shared signal table (needs numpy, scikit-learn, lightgbm).

Fixed scorers (no learning): recency, file overlap, BM25, FONTE's vote and
max-aggregation (FL track only), and the released outputs of Bug2Commit and
FBL-BERT (re-scored on the same candidates). Learned: a tuned linear
combination (weights chosen prequentially), logistic regression and
LambdaMART, each trained prequentially on the bugs already seen.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np

from .core import Case, CaseView


class Fixed:
    """Score = a function of the view; never learns."""

    def __init__(self, name: str, fn, description: str = ""):
        self.name, self.fn, self.description = name, fn, description

    def reset(self) -> None:
        pass

    def rank(self, view: CaseView) -> list[float]:
        return [float(x) for x in self.fn(view)]

    def learn(self, view: CaseView, labels: frozenset[str]) -> None:
        pass


def col(view: CaseView, name: str) -> np.ndarray:
    return np.asarray(view.col(name), dtype=float)


def recency(view):
    return [-c.pos for c in view.cands]


def published(meta_of, key: str):
    """Released system output; candidates the system did not score get -inf."""
    def fn(view):
        d = meta_of[view.id][key]
        return [d.get(c.id, -1e18) for c in view.cands]
    return fn


# -- tuned linear combination ---------------------------------------------------------------

class LinearGrid:
    """Within-bug z-scored signals, weights from a fixed grid. Produces one
    rank per bug per weight vector; ``core.follow_the_leader`` then picks the
    weights prequentially."""

    def __init__(self, signals: list[tuple[str, float]], levels=(0.0, 0.5, 1.0)):
        self.signals = signals            # (num name, sign)
        self.W = np.asarray(list(itertools.product(levels, repeat=len(signals))), dtype=float).T
        self.W = self.W[:, self.W.sum(axis=0) > 0]   # drop the all-zero vector

    def names(self) -> list[str]:
        return [",".join(f"{n}={w:g}" for (n, _), w in zip(self.signals, self.W[:, j]))
                for j in range(self.W.shape[1])]

    def rr_matrix(self, cases: list[Case]) -> np.ndarray:
        """(n_bugs, n_configs) reciprocal ranks, ties newest first."""
        out = np.zeros((len(cases), self.W.shape[1]))
        for i, case in enumerate(cases):
            v = case.view
            Z = np.stack([sgn * col(v, n) for n, sgn in self.signals], axis=1)
            sd = Z.std(axis=0)
            Z = (Z - Z.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
            S = Z @ self.W                                     # (n_cands, n_cfg)
            pos = np.asarray([c.pos for c in v.cands])
            lab = [k for k, c in enumerate(v.cands) if c.id in case.labels]
            if not lab:
                continue
            best = None
            for k in lab:
                s = S[k]
                r = 1 + (S > s).sum(axis=0) + ((S == s) & (pos[:, None] < pos[k])).sum(axis=0)
                best = r if best is None else np.minimum(best, r)
            out[i] = 1.0 / best
        return out


# -- learning to rank ---------------------------------------------------------------------------

class CatEncoder:
    """One-hot of binary features seen at least ``min_count`` times in the
    training rows (vocabulary fitted on training bugs only)."""

    def __init__(self, min_count: int = 3):
        self.min_count = min_count
        self.vocab: dict[str, int] = {}

    def fit(self, cat_rows) -> "CatEncoder":
        cnt = Counter(x for row in cat_rows for x in row)
        self.vocab = {k: i for i, k in enumerate(sorted(k for k, n in cnt.items() if n >= self.min_count))}
        return self

    def transform(self, cat_rows):
        import scipy.sparse as sp
        rows, cols = [], []
        for i, row in enumerate(cat_rows):
            for x in row:
                j = self.vocab.get(x)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                             shape=(len(cat_rows), max(1, len(self.vocab))))


class LTR:
    """Prequential learning to rank: retrain on all bugs seen so far every
    ``retrain_every`` bugs; before the first fit, score 0 (= recency order).

    ``kind``: ``logreg`` (scikit-learn LogisticRegression, L2, ``C``) or
    ``lambdamart`` (LightGBM LGBMRanker, lambdarank). Features: every numeric
    signal (standardised) + one-hot binary features (``CatEncoder``).
    ``neg_per_bug``: keep at most that many random negatives per training
    bug (seeded); None keeps all.
    """

    def __init__(self, name: str, kind: str = "logreg", C: float = 1.0, retrain_every: int = 1,
                 neg_per_bug: int | None = None, seed: int = 0, n_estimators: int = 300):
        self.name, self.kind, self.C = name, kind, C
        self.retrain_every, self.neg_per_bug, self.seed = retrain_every, neg_per_bug, seed
        self.n_estimators = n_estimators

    def reset(self) -> None:
        self.rows_num: list[np.ndarray] = []
        self.rows_cat: list[list] = []
        self.rows_y: list[np.ndarray] = []
        self.groups: list[int] = []
        self.model = None
        self.enc = None
        self.mu = self.sd = None
        self.seen = 0
        self.rng = random.Random(self.seed)

    def _X(self, num: np.ndarray, cats):
        import scipy.sparse as sp
        Xn = (num - self.mu) / self.sd
        return sp.hstack([sp.csr_matrix(Xn), self.enc.transform(cats)], format="csr")

    def rank(self, view: CaseView) -> list[float]:
        if self.model is None:
            return [0.0] * len(view.cands)
        num = np.asarray([c.num for c in view.cands], dtype=float)
        X = self._X(num, [c.cat for c in view.cands])
        if self.kind == "logreg":
            return list(self.model.decision_function(X))
        return list(self.model.predict(X))

    def learn(self, view: CaseView, labels: frozenset[str]) -> None:
        idx = list(range(len(view.cands)))
        pos = [i for i in idx if view.cands[i].id in labels]
        neg = [i for i in idx if view.cands[i].id not in labels]
        if self.neg_per_bug is not None and len(neg) > self.neg_per_bug:
            neg = sorted(self.rng.sample(neg, self.neg_per_bug))
        keep = pos + neg
        self.rows_num.append(np.asarray([view.cands[i].num for i in keep], dtype=float))
        self.rows_cat.append([view.cands[i].cat for i in keep])
        self.rows_y.append(np.asarray([1] * len(pos) + [0] * len(neg)))
        self.groups.append(len(keep))
        self.seen += 1
        if self.seen % self.retrain_every == 0:
            self._fit()

    def _fit(self) -> None:
        y = np.concatenate(self.rows_y)
        if y.sum() < 2:
            return
        num = np.vstack(self.rows_num)
        cats = [r for rows in self.rows_cat for r in rows]
        self.mu = num.mean(axis=0)
        sd = num.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.enc = CatEncoder(min_count=3).fit(cats)
        X = self._X(num, cats)
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(C=self.C, solver="liblinear", max_iter=1000, random_state=self.seed)
            m.fit(X, y)
        else:
            import lightgbm as lgb
            m = lgb.LGBMRanker(objective="lambdarank", n_estimators=self.n_estimators, learning_rate=0.05,
                               num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, random_state=self.seed, n_jobs=2, verbose=-1,
                               deterministic=True, force_row_wise=True)
            m.fit(X.astype(np.float32), y, group=self.groups)
        self.model = m
