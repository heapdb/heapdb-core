"""HeapDB as a BIC ranker (pure Python + the ``heapdb`` package).

Mapping:

* every candidate commit is a **cause event** (type ``commit``) at its commit
  time, with the candidate's binary features as event features
  (``Cand.cat``: author, directories, message keywords and the fixed
  buckets of the shared signals, e.g. ``b:bm25_diff=1``,
  ``b:trace_src=1``, ``b:lag=<7d``);
* the failure (or the fix, for DPDK) is the **effect** ``bic`` at the
  failure time; ``top_causes`` ranks the candidates by
  ``kernel(lag) x dose x noisy-OR(feature links)``;
* each bug is its own entity, so what is learned lives in the global
  (pooled) layer of HeapDB's hierarchical Beta links, uncapped
  (``global_pseudocount`` large): with one bug per entity there is no
  per-entity evidence to borrow against;
* after ranking, the label is revealed through ``feedback``: the true BIC is
  confirmed (proportional / EM credit across its features), every other
  candidate is rejected (the label says none of them caused this bug).
  The entity is then forgotten (its events dropped; the global counts
  stay), which keeps memory flat.

Optional: ``dose`` = ``log1p(lines changed)`` scaled so 100 lines = 1, and an
exponential lag kernel instead of a flat window.

``combine="logistic"`` registers the effect with HeapDB's logistic combine
mode instead of noisy-OR: ``link = sigmoid(b + sum_f w_f)``, weights learned
online (FTRL-Proximal) from the same confirm/reject feedback. Those configs
use HeapDB's default prior Beta(1, 4) (it only seeds the intercept) and
vary the learning rate instead of the prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import heapdb

from .core import CaseView

DAY = 86400.0
YEAR = 365.0 * DAY
FLAT_SUPPORT = 60 * YEAR


@dataclass(frozen=True)
class HDBConfig:
    kernel: str = "flat"                    # "flat" | "exp1y" | "exp90d"
    dose: bool = False
    prior: tuple[float, float] = (0.001, 10.0)
    learn: bool = True
    combine: str = "noisy_or"               # "noisy_or" | "logistic"
    lr: float = 0.1                         # logistic learning rate

    @property
    def name(self) -> str:
        if self.combine == "logistic":
            return (f"combine=logistic,kernel={self.kernel},dose={'on' if self.dose else 'off'},"
                    f"lr={self.lr:g}")
        p = f"{self.prior[0]:g}/{self.prior[1]:g}"
        return f"kernel={self.kernel},dose={'on' if self.dose else 'off'},prior={p}"


def make_kernel(name: str):
    if name == "flat":
        return heapdb.WindowKernel(min_lag=0.0, max_lag=FLAT_SUPPORT)
    if name == "exp1y":
        return heapdb.ExponentialDecay(half_life=YEAR, horizon_half_lives=60.0)
    if name == "exp90d":
        return heapdb.ExponentialDecay(half_life=90 * DAY, horizon_half_lives=240.0)
    raise ValueError(name)


def dose_of(lines: float) -> float:
    return max(math.log1p(max(lines, 0.0) + 1.0) / math.log1p(101.0), 1e-3)


class HeapDBRanker:
    """``rank`` inserts the bug's candidates and runs ``top_causes``;
    ``learn`` gives feedback for that bug."""

    EFFECT = "bic"

    def __init__(self, cfg: HDBConfig, name: str = "heapdb", features=None):
        self.cfg = cfg
        self.name = name
        self.features = features or (lambda c: c.cat)
        self.db: heapdb.HeapDB | None = None
        self._ids: dict[str, dict[str, int]] = {}

    def reset(self) -> None:
        self.db = heapdb.HeapDB(bucket_seconds=30 * DAY)
        if self.cfg.combine == "logistic":
            # credit: the logistic default (uniform); it only feeds the Beta
            # evidence ledger, scoring uses the learned weights.
            self.db.register_effect(self.EFFECT, make_kernel(self.cfg.kernel),
                                    default_prior=self.cfg.prior, cause_types=["commit"],
                                    global_pseudocount=1e12, hierarchical=True,
                                    combine="logistic", learning_rate=self.cfg.lr)
        else:
            self.db.register_effect(self.EFFECT, make_kernel(self.cfg.kernel),
                                    default_prior=self.cfg.prior, cause_types=["commit"],
                                    credit="proportional", global_pseudocount=1e12, hierarchical=True)
        self._ids = {}

    def _insert(self, view: CaseView) -> dict[str, int]:
        if view.id in self._ids:
            return self._ids[view.id]
        j_lines = view.num_names.index("lines") if "lines" in view.num_names else None
        ids = {}
        for c in view.cands:
            w = dose_of(float(c.num[j_lines])) if (self.cfg.dose and j_lines is not None) else 1.0
            ev = self.db.insert(view.id, c.time, "commit", self.features(c), weight=w)
            ids[c.id] = ev.id
        self._ids[view.id] = ids
        return ids

    def rank(self, view: CaseView) -> list[float]:
        ids = self._insert(view)
        res = self.db.top_causes(view.id, self.EFFECT, at=view.time, k=len(view.cands))
        by_event = {cause.event.id: cause.score for cause in res}
        return [by_event.get(ids[c.id], 0.0) for c in view.cands]

    def learn(self, view: CaseView, labels: frozenset[str]) -> None:
        ids = self._insert(view)
        if self.cfg.learn:
            pos = [ids[c.id] for c in view.cands if c.id in labels]
            neg = [ids[c.id] for c in view.cands if c.id not in labels]
            if pos:
                self.db.feedback(view.id, self.EFFECT, view.time, confirmed_event_id=pos[0],
                                 rejected_ids=neg)
                for extra in pos[1:]:
                    self.db.feedback(view.id, self.EFFECT, view.time, confirmed_event_id=extra)
            elif neg:
                self.db.feedback(view.id, self.EFFECT, view.time, rejected_ids=neg)
        self.forget(view)

    def forget(self, view: CaseView) -> None:
        if view.id in self._ids:
            self.db.forget_entity(view.id, scrub_global=False)
            del self._ids[view.id]

    # -- inspection ------------------------------------------------------------------
    def links(self, min_support: float = 20.0) -> list[tuple[str, float, float]]:
        """(feature, strength, evidence n) from the global layer; under
        logistic, (feature, weight = log-odds multiplier, evidence n)."""
        rep = self.db.link_report(self.EFFECT, entity=None, min_support=min_support)
        if self.cfg.combine == "logistic":
            return [(r.feature, r.log_odds, r.n) for r in rep]
        return [(r.feature, r.strength, r.n) for r in rep]

    def base_log_odds(self) -> float | None:
        return self.db.effect(self.EFFECT).links.base_log_odds


#: Grid fixed in advance (prequential follow-the-leader picks from it).
GRID = [HDBConfig(kernel=k, dose=d, prior=p)
        for k in ("flat", "exp1y")
        for d in (False, True)
        for p in ((0.001, 10.0), (0.01, 10.0))]
DEFAULT = GRID[0]

#: Logistic combine mode: the same kernel x dose grid, with the learning rate
#: in place of the prior (HeapDB's default prior Beta(1, 4) seeds the
#: intercept). Fixed before the full runs, from the development slice.
LOGISTIC_PRIOR = (1.0, 4.0)
LGRID = [HDBConfig(kernel=k, dose=d, prior=LOGISTIC_PRIOR, combine="logistic", lr=lr)
         for k in ("flat", "exp1y")
         for d in (False, True)
         for lr in (0.1, 0.5)]
LDEFAULT = LGRID[0]
