"""Bug-inducing-commit evaluation (evals/bic): leakage, determinism, metrics.

These run on toy data (no download). The checks that matter for the
published numbers:

* no label of the bug being ranked, or of any later bug, is visible when a
  bug is ranked (by construction of ``prequential`` and checked as a black
  box: changing later labels never changes earlier ranks);
* runs are deterministic;
* rank, Acc@k, MRR, the paired bootstrap, follow-the-leader and bisection
  are computed correctly on hand-checked examples.
"""

import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.bic.core import (Cand, Case, CaseView, follow_the_leader, paired_bootstrap,  # noqa: E402
                            prequential, rank_of, rr, standard_bisection, stream_order,
                            summarize, weighted_bisection)
from evals.bic.hdb import GRID, HDBConfig, HeapDBRanker  # noqa: E402

DAY = 86400.0


def view(scores_pos=4, t=100 * DAY, cid="bug"):
    cands = tuple(Cand(id=f"c{i}", time=t - (i + 1) * DAY, pos=i, num=(float(i),), cat=())
                  for i in range(scores_pos))
    return CaseView(id=cid, project="p", time=t, cands=cands, num_names=("x",))


# -- metrics on a toy example ----------------------------------------------------------------

def test_rank_tie_rules():
    v = view()
    scores = [0.1, 0.5, 0.5, 0.9]          # c0 newest ... c3 oldest
    # recency tie rule: c3 (0.9) first, then c1 before c2 (c1 is newer)
    assert rank_of(v, scores, frozenset({"c2"})) == 3
    assert rank_of(v, scores, frozenset({"c1"})) == 2
    assert rank_of(v, scores, frozenset({"c3"})) == 1
    assert rank_of(v, scores, frozenset({"c0"})) == 4
    # max rule (FONTE): a tie takes the worst rank of its group
    assert rank_of(v, scores, frozenset({"c1"}), tie="max") == 3
    assert rank_of(v, scores, frozenset({"c2"}), tie="max") == 3
    # several labels: best one counts; label outside the candidates: None
    assert rank_of(v, scores, frozenset({"c0", "c3"})) == 1
    assert rank_of(v, scores, frozenset({"zz"})) is None


def test_summarize_hand_checked():
    s = summarize([1, 3, None, 12])
    assert s["n"] == 4 and s["outside"] == 1
    assert s["Acc@1"] == 0.25
    assert s["Acc@3"] == 0.5 and s["Acc@5"] == 0.5 and s["Acc@10"] == 0.5
    assert math.isclose(s["MRR"], (1 + 1 / 3 + 0 + 1 / 12) / 4)
    assert math.isclose(s["mean_rank"], 16 / 3)
    assert s["median_rank"] == 3
    assert rr(None) == 0.0 and rr(4) == 0.25


def test_paired_bootstrap():
    a = [1.0, 0.5, 0.25, 1.0, 0.0] * 20
    b = [0.5, 0.5, 0.25, 0.5, 0.0] * 20
    d, lo, hi = paired_bootstrap(a, b, n=2000, seed=7)
    assert math.isclose(d, 0.2)
    assert lo <= d <= hi and lo > 0
    assert paired_bootstrap(a, b, n=2000, seed=7) == (d, lo, hi)      # deterministic
    d0, lo0, hi0 = paired_bootstrap(a, a, n=500)
    assert d0 == lo0 == hi0 == 0.0


def test_follow_the_leader_uses_only_the_past():
    per = {"A": [1.0, 0.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0, 1.0]}
    rr_, chosen = follow_the_leader(per)
    # bug 0: first config (no history); bug 1: A led after bug 0; bug 2: tie 1-1 -> grid order A;
    # bug 3: B leads 2-1
    assert chosen == ["A", "A", "A", "B"]
    assert rr_ == [1.0, 0.0, 0.0, 1.0]
    # changing the outcome of the last bug cannot change earlier choices
    per2 = {"A": [1.0, 0.0, 0.0, 1.0], "B": [0.0, 1.0, 1.0, 0.0]}
    assert follow_the_leader(per2)[1][:4] == chosen[:4]


def test_bisection():
    # 8 commits newest first, BIC at index 5: pivots 4 (bad), 6 (good), 5 (bad) -> 3 steps
    assert standard_bisection(8, 5) == 3
    assert standard_bisection(1, 0) == 0
    # all mass on the BIC: weighted bisection isolates it quickly
    w = [0.001] * 8
    w[5] = 1.0
    assert weighted_bisection(w, 5) <= standard_bisection(8, 5)
    assert weighted_bisection([1.0] * 8, 5) == standard_bisection(8, 5)


# -- leakage --------------------------------------------------------------------------------

def make_stream(n_bugs=30, n_cands=12, seed=1):
    """Toy stream: the BIC tends to carry feature ``hot`` and to be recent."""
    rng = random.Random(seed)
    cases = []
    for b in range(n_bugs):
        t = (b + 1) * 10 * DAY
        bic = rng.randrange(n_cands)
        cands = []
        for i in range(n_cands):
            feats = {f"dir:d{rng.randrange(3)}", f"b:lines={rng.choice(['<=5', '<=20'])}"}
            if i == bic or rng.random() < 0.1:
                feats.add("hot")
            cands.append(Cand(id=f"b{b}c{i}", time=t - (i + 1) * 3600.0, pos=i,
                              num=(float(10 + i), float(rng.randrange(100))), cat=tuple(sorted(feats))))
        v = CaseView(id=f"bug{b:02d}", project=f"p{b % 3}", time=t, cands=tuple(cands),
                     num_names=("x", "lines"))
        cases.append(Case(view=v, labels=frozenset({f"b{b}c{bic}"})))
    return cases


class Spy:
    """Records which labels were revealed before each rank call."""

    name = "spy"

    def __init__(self):
        self.revealed = []
        self.log = []

    def reset(self):
        self.revealed = []
        self.log = []

    def rank(self, v):
        assert not hasattr(v, "labels")
        self.log.append((v.id, v.time, list(self.revealed)))
        return [0.0] * len(v.cands)

    def learn(self, v, labels):
        self.revealed.append((v.id, v.time, labels))


def test_prequential_reveals_only_past_labels():
    cases = make_stream()
    shuffled = cases[:]
    random.Random(3).shuffle(shuffled)
    spy = Spy()
    prequential(shuffled, spy)
    order = [c.id for c in stream_order(cases)]
    assert [x[0] for x in spy.log] == order                       # chronological
    for bug_id, t, revealed in spy.log:
        ids = [r[0] for r in revealed]
        assert bug_id not in ids                                   # never its own label
        assert all(rt < t or (rt == t and rid < bug_id) for rid, rt, _ in revealed)
        assert ids == order[:order.index(bug_id)]                  # exactly the earlier bugs


@pytest.mark.parametrize("cfg", [GRID[0], GRID[-1]])
def test_heapdb_future_invariance(cfg):
    """Black box: rewriting the labels of later bugs must not change the
    ranks of earlier bugs (HeapDB learns only from feedback already given)."""
    cases = stream_order(make_stream())
    base = prequential(cases, HeapDBRanker(cfg))
    k = 12
    rng = random.Random(9)
    altered = cases[:k] + [Case(view=c.view, labels=frozenset({rng.choice(c.view.cands).id}))
                           for c in cases[k:]]
    again = prequential(altered, HeapDBRanker(cfg))
    assert again[:k] == base[:k]


def test_heapdb_learns_and_is_deterministic():
    cases = stream_order(make_stream(n_bugs=40))
    a = prequential(cases, HeapDBRanker(GRID[0]))
    b = prequential(cases, HeapDBRanker(GRID[0]))
    assert a == b                                                 # deterministic
    nolearn = prequential(cases, HeapDBRanker(HDBConfig(learn=False)), learn=False)
    late = lambda r: sum(rr(x) for x in r[20:]) / len(r[20:])  # noqa: E731
    assert late(a) > late(nolearn)                                 # feedback helps on the toy


def test_heapdb_rank_does_not_mutate_learning_state():
    """Ranking a bug twice gives the same scores: ranking alone gives no feedback."""
    cases = stream_order(make_stream())
    m = HeapDBRanker(GRID[0])
    m.reset()
    for c in cases[:10]:
        m.rank(c.view)
        m.learn(c.view, c.labels)
    s1 = m.rank(cases[10].view)
    s2 = m.rank(cases[10].view)
    assert s1 == s2


def test_ltr_future_invariance_and_determinism():
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    from evals.bic.baselines import LTR
    cases = stream_order(make_stream())
    a = prequential(cases, LTR("lr", "logreg", retrain_every=1))
    b = prequential(cases, LTR("lr", "logreg", retrain_every=1))
    assert a == b
    k = 15
    altered = cases[:k] + [Case(view=c.view, labels=frozenset({c.view.cands[-1].id})) for c in cases[k:]]
    assert prequential(altered, LTR("lr", "logreg", retrain_every=1))[:k] == a[:k]


def test_bucket_edges():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from evals.bic.signals import LAG_EDGES, RANK_EDGES, bucket
    assert bucket(1, RANK_EDGES, ">100") == "1"
    assert bucket(4, RANK_EDGES, ">100") == "4-5"
    assert bucket(101, RANK_EDGES, ">100") == ">100"
    assert bucket(0.5, LAG_EDGES, ">=3y", strict=True) == "<1d"
    assert bucket(1.0, LAG_EDGES, ">=3y", strict=True) == "<7d"
    assert bucket(5000, LAG_EDGES, ">=3y", strict=True) == ">=3y"
