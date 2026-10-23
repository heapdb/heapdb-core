"""The logistic combine mode from Python: registration, explanations, link
inspection, persistence, the remote client, and the engine against the
pure-Python reference used to choose the formulation (evals/bic)."""

import math
import os
import random
import sys

import pytest

import heapdb
from heapdb import HeapDB, LogNormalKernel, WindowKernel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.bic.combine_dev import FTRL  # noqa: E402
from evals.bic.core import Cand, Case, CaseView, prequential  # noqa: E402
from evals.bic.hdb import LDEFAULT, LGRID, HeapDBRanker  # noqa: E402

H = 3600.0


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def bloat(**kw):
    db = HeapDB()
    db.register_effect("bloating", LogNormalKernel(median=2 * H, sigma=0.6), priors={"beans": (3, 1)},
                       combine="logistic", **kw)
    a = db.insert("me", 0, "meal", {"beans", "onion"})
    b = db.insert("me", 600, "meal", {"rice", "onion"})
    return db, a, b


def test_register_and_spec():
    db, _, _ = bloat()
    spec = db.effect("bloating")
    assert (spec.combine, spec.learning_rate, spec.l2, spec.credit) == ("logistic", 0.1, 1.0, "uniform")
    assert "combine='logistic'" in repr(spec)
    db.register_effect("x", WindowKernel(0, H), combine="logistic", learning_rate=0.5, l2=2.0, credit="proportional")
    s = db.effect("x")
    assert (s.learning_rate, s.l2, s.credit) == (0.5, 2.0, "proportional")
    db.register_effect("plain", WindowKernel(0, H))
    p = db.effect("plain")
    assert (p.combine, p.learning_rate, p.l2, p.credit) == ("noisy_or", None, None, "proportional")
    assert "combine" not in repr(p)
    for bad in (dict(combine="noisy-or"), dict(combine="logistic", learning_rate=0.0),
                dict(combine="logistic", l2=-1.0)):
        with pytest.raises(ValueError):
            db.register_effect("bad", WindowKernel(0, H), **bad)


def test_seeds_and_link_math():
    db, _, _ = bloat()
    links = db.effect("bloating").links
    assert links.combine == "logistic"
    assert links.base_log_odds == pytest.approx(math.log(1 / 4))        # default prior Beta(1, 4)
    assert links.weight("onion") == 0.0                                # at the default prior
    assert links.weight("beans") == pytest.approx(math.log(3) - math.log(1 / 4))
    assert links.link([]) == pytest.approx(0.2)
    assert links.link(["onion", "garlic"]) == pytest.approx(0.2)        # no evidence, no change
    assert links.link(["beans"]) == pytest.approx(0.75)                 # = the prior mean
    plain = HeapDB()
    plain.register_effect("e", WindowKernel(0, H))
    pl = plain.effect("e").links
    assert pl.combine == "noisy_or" and pl.base_log_odds is None and pl.weight("x") is None
    assert pl.link(["a", "b"]) == pytest.approx(pl.noisy_or(["a", "b"]))


def test_explanation_fields_and_feedback():
    db, a, b = bloat()
    rep = db.feedback("me", "bloating", 2 * H, confirmed_event_id=a.id, rejected_ids=[b.id])
    assert set(rep.credit.values()) == {1.0}                        # count credit on the Beta ledger
    w = db.effect("bloating").links.weight
    assert w("rice") < 0 < w("onion") < w("beans")
    ex = db.explain(a, "bloating", 2 * H)
    assert ex.combine == "logistic"
    z = ex.base_log_odds + sum(c.log_odds for c in ex.features)
    assert ex.log_odds == pytest.approx(z) and ex.link_strength == pytest.approx(sigmoid(z))
    assert ex.odds == pytest.approx(ex.base_odds * math.prod(c.odds_multiplier for c in ex.features))
    assert ex.score == pytest.approx(db.score(a, "bloating", 2 * H))
    assert "logistic: base odds" in str(ex)
    rows = {r.feature: r for r in db.link_report("bloating", min_support=0)}
    assert rows["rice"].log_odds == pytest.approx(w("rice"))
    # noisy-OR explanations are unchanged: no logistic fields
    plain = HeapDB()
    plain.register_effect("e", WindowKernel(0, 10 * H))
    e = plain.insert("me", 0, "t", {"x"})
    pe = plain.explain(e, "e", H)
    assert pe.combine == "noisy_or" and pe.base_log_odds is None and pe.log_odds is None
    assert pe.features[0].log_odds is None and pe.features[0].odds_multiplier is None


def test_snapshot_round_trip(tmp_path):
    db, a, b = bloat(learning_rate=0.5)
    db.feedback("me", "bloating", 2 * H, confirmed_event_id=a.id, rejected_ids=[b.id])
    db.observe("me", "bloating", 3 * H)
    p = tmp_path / "s.jsonl"
    db.save(p)
    db2 = HeapDB.load(p)
    assert db2.effect("bloating").combine == "logistic" and db2.effect("bloating").learning_rate == 0.5
    q = lambda d: [(c.event.id, c.score) for c in d.top_causes("me", "bloating", 2 * H, k=5)]  # noqa: E731
    assert q(db) == q(db2)
    assert db2.effect("bloating").links.weight("beans") == db.effect("bloating").links.weight("beans")


def _toy_stream(n_bugs=40, seed=3):
    """Bugs whose BIC tends to carry 'hot' and 'b:x=1', with distractors."""
    rng = random.Random(seed)
    cases = []
    for i in range(n_bugs):
        t = (i + 1) * 100 * 86400.0
        cands = []
        bic = rng.randrange(12)
        for j in range(12):
            cat = {f"kw:w{rng.randrange(6)}", f"b:x={rng.choice(['1', '2', '3'])}"}
            if j == bic and rng.random() < 0.7:
                cat |= {"hot", "b:x=1"}
            elif rng.random() < 0.15:
                cat.add("hot")
            cands.append(Cand(id=f"b{i}c{j}", time=t - (j + 1) * 86400.0, pos=j, num=(0.0,), cat=tuple(sorted(cat))))
        cases.append(Case(CaseView(id=f"bug{i}", project="p", time=t, cands=tuple(cands), num_names=("x",)),
                          frozenset({f"b{i}c{bic}"})))
    return cases


def test_engine_matches_the_reference_ftrl():
    """The engine's default logistic config is the reference FTRL of
    evals/bic/combine_dev.py: same log-odds on every candidate, every bug."""
    cases = _toy_stream()
    eng = HeapDBRanker(LDEFAULT)
    eng.reset()
    ref = FTRL(lr=LDEFAULT.lr, prior=LDEFAULT.prior)
    links = None
    for c in cases:
        links = eng.db.effect("bic").links
        for x in c.view.cands:
            z = ref.score(set(x.cat))
            assert links.link(x.cat) == pytest.approx(sigmoid(z), rel=1e-12, abs=1e-15)
        eng.rank(c.view)
        eng.learn(c.view, c.labels)
        ref.learn(c.view.cands, c.labels)
    assert links.weight("hot") > 0


def test_logistic_learns_the_toy_signal():
    cases = _toy_stream(n_bugs=60)
    ranks = prequential(cases, HeapDBRanker(LGRID[1]))       # lr 0.5
    late = [1 / r for r in ranks[30:]]
    assert sum(late) / len(late) > 0.4                        # random ranking: ~0.26


# -- remote client --------------------------------------------------------------------------

from test_remote import _close, _new_db, server  # noqa: E402,F401


def test_remote_parity_logistic(server):  # noqa: F811
    local, remote = HeapDB(bucket_seconds=900), _new_db(server)
    for d in (local, remote):
        d.register_effect("inc", WindowKernel(0, 6 * H), priors={"migration": (3, 1)}, combine="logistic",
                          learning_rate=0.5)
    rng = random.Random(5)
    evs = [dict(entity="svc", time=i * 300.0, type="deploy",
                features={rng.choice(["migration", "config", "flag"]), f"team:{rng.randrange(3)}"})
           for i in range(30)]
    for e in evs:
        local.insert(e["entity"], e["time"], e["type"], e["features"])
    remote.insert_many(evs)
    for at in (2 * H, 4 * H):
        top = local.top_causes("svc", "inc", at, k=3)
        args = ("svc", "inc", at, top[1].event.id, [top[0].event.id, top[2].event.id])
        fl, fr = local.feedback(*args), remote.feedback(*args)
        assert fl.credit == fr.credit
    local.observe("svc", "inc", 5 * H)
    remote.observe("svc", "inc", 5 * H)
    ql = local.top_causes("svc", "inc", 6 * H, k=5, explain=True)
    qr = remote.top_causes("svc", "inc", 6 * H, k=5, explain=True)
    assert [c.event.id for c in ql] == [c.event.id for c in qr]
    for cl, cr in zip(ql, qr):
        el, er = cl.explanation, cr.explanation
        assert _close(cl.score, cr.score) and er.combine == "logistic"
        assert _close(el.base_log_odds, er.base_log_odds) and _close(el.log_odds, er.log_odds)
        assert [(c.feature, c.log_odds) for c in el.features] == [(c.feature, c.log_odds) for c in er.features]
        assert _close(cr.explanation.features[0].odds_multiplier, el.features[0].odds_multiplier)
    lr = remote.link_report("inc", min_support=0)
    ll = local.link_report("inc", min_support=0)
    assert [(r.feature, r.log_odds) for r in ll] == [(r.feature, r.log_odds) for r in lr]
    # noisy-OR registration through the client sends no logistic fields
    remote.register_effect("plain", WindowKernel(0, H))
    assert remote.execute("LINKS OF plain") is not None
