"""v0.2: weights as dose, rules, group queries, calibration, forget_entity,
and snapshot round-trip of the new state."""

import math

import pytest

from heapdb import ExponentialDecay, HeapDB, LogNormalKernel, WindowKernel, format_result

H = 3600.0
M = 60.0


# -- weight = dose / exposure -----------------------------------------------------


def test_weight_from_attrs_and_validation():
    db = HeapDB()
    db.register_event_type("meal", weight=lambda a: a.get("portion", 1.0))
    db.register_event_type("deploy", weight=lambda a: math.log1p(a["lines_changed"]))
    assert db.insert("me", 0, "meal", {"x"}, attrs={"portion": 2.5}).weight == 2.5
    assert db.insert("me", 0, "meal", {"x"}).weight == 1.0
    assert db.insert("me", 0, "meal", {"x"}, weight=0.7, attrs={"portion": 9}).weight == 0.7
    d = db.insert("svc", 0, "deploy", {"y"}, attrs={"lines_changed": 99})
    assert d.weight == pytest.approx(math.log1p(99))
    assert d.attrs["lines_changed"] == 99
    assert db.insert("svc", 0, "unregistered").weight == 1.0
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            db.insert("me", 0, "meal", weight=bad)
    with pytest.raises(ValueError):
        db.insert("svc", 0, "deploy", attrs={"lines_changed": 0})  # log1p(0) = 0
    with pytest.raises(TypeError):
        d.attrs["x"] = 1  # read-only


# -- rules ------------------------------------------------------------------------------


def test_rules_soft_and_hard():
    db = HeapDB()
    db.register_effect("fraud", ExponentialDecay(half_life=H))
    db.add_rule("fraud", "small transaction", lambda e: e.attrs.get("amount", 0) < 10, 0.3)
    small = db.insert("acct", 0, "txn", {"card"}, attrs={"amount": 5})
    big = db.insert("acct", 0, "txn", {"card"}, attrs={"amount": 500})
    assert db.score(small, "fraud", 60) == pytest.approx(0.3 * db.score(big, "fraud", 60))
    ex = db.explain(small, "fraud", 60)
    assert [r.name for r in ex.rules] == ["small transaction"]
    assert ex.score == pytest.approx(db.score(small, "fraud", 60))
    assert db.explain(big, "fraud", 60).rules == ()

    db.add_rule("fraud", "block card", lambda e: "card" in e.features, 0.0)
    assert db.top_causes("acct", "fraud", at=60, k=5).causes == []
    db.remove_rule("fraud", "block card")
    assert len(db.top_causes("acct", "fraud", at=60, k=5)) == 2


def test_rule_multiplier_validation():
    db = HeapDB()
    db.register_effect("fx", WindowKernel(0, H))
    for bad in (1.01, -0.1, float("nan")):
        with pytest.raises(ValueError):
            db.add_rule("fx", "r", lambda e: True, bad)
    with pytest.raises(KeyError):
        db.add_rule("nope", "r", lambda e: True, 0.5)
    db.add_rule("fx", "r", lambda e: True, 0.5)
    db.add_rule("fx", "r", lambda e: True, 0.2)  # same name replaces
    assert [(r.name, r.multiplier) for r in db.effect("fx").rules] == [("r", 0.2)]


def test_type_level_rule_tightens_bound():
    db = HeapDB(bucket_seconds=600)
    db.register_effect("fx", ExponentialDecay(half_life=H))
    db.add_rule("fx", "cron is rarely it", None, 0.1, types={"cron"})
    for i in range(20):
        db.insert("s", i * 600 + 10, "cron", {"job"}, weight=1.0)
    spec = db.effect("fx")
    seg = next(db.entity_log("s").segments())
    bound, _ = db.segment_bound(spec, seg, "s", 12_000, 0, 12_000)
    assert bound == pytest.approx(0.1 * spec.kernel.max_over(12_000 - seg.max_time,
                                                             12_000 - seg.min_time)
                                  * spec.links.noisy_or({"job"}, "s"))


def test_cause_types_still_hard_filter():
    db = HeapDB()
    db.register_effect("fx", ExponentialDecay(half_life=H), cause_types={"deploy"})
    db.insert("s", 0, "cron", {"x"})
    assert db.top_causes("s", "fx", at=60, k=3).causes == []


def test_future_events_never_score():
    db = HeapDB()
    db.register_effect("fx", WindowKernel(0, H))
    ev = db.insert("s", 1000, "t", {"x"})
    assert db.score(ev, "fx", 999) == 0.0
    assert db.explain(ev, "fx", 999).kernel_value == 0.0


# -- group queries ---------------------------------------------------------------------


def test_group_query_merges_and_labels_entities():
    db = HeapDB()
    db.register_effect("latency", ExponentialDecay(half_life=20 * M), priors={"pool": (3, 2)})
    a = db.insert("checkout", 0, "config", {"pool"})
    b = db.insert("payments", 300, "deploy", {"ui"})
    c = db.insert("search", 600, "config", {"pool"})
    res = db.top_causes_group(["checkout", "payments", "search"], "latency",
                              at=1200, k=2, explain=True)
    assert [x.event.id for x in res] == [c.id, a.id]
    assert [x.rank for x in res] == [1, 2]
    assert [x.entity for x in res] == ["search", "checkout"]
    assert res.top.explanation.entity == "search"
    assert "1. search: #" in format_result(res)
    assert res.is_group and res.entities == ("checkout", "payments", "search")
    assert res.stats.segments_in_window == 3
    del b


def test_group_query_rejects_mixed_effects():
    db = HeapDB()
    db.register_effect("a", WindowKernel(0, H))
    db.register_effect("b", WindowKernel(0, H))
    with pytest.raises(ValueError):
        db.top_causes_group(["x", "y"], ["a", "b"], at=0)
    db.top_causes_group(["x"], ["a", "a"], at=0)  # one effect: fine
    with pytest.raises(ValueError):
        db.top_causes_group([], "a", at=0)


# -- calibration -----------------------------------------------------------------------


def test_calibration_report():
    db = HeapDB()
    db.register_effect("fx", WindowKernel(0, H), priors={"good": (9, 1), "bad": (1, 9)})
    for i in range(10):
        t = i * 10 * H
        good = db.insert("e", t, "x", {"good"})
        bad = db.insert("e", t, "x", {"bad"})
        # 'good' (score 0.9) is confirmed 9/10 times; 'bad' (0.1) rejected.
        if i < 9:
            db.feedback("e", "fx", t + 60, confirmed_event_id=good.id, rejected_ids=[bad.id])
        else:
            db.feedback("e", "fx", t + 60, rejected_ids=[good.id, bad.id])
    recs = db.calibration_records("fx")
    assert len(recs) == 20
    # Scores recorded as shown (before the update): first round exactly the priors.
    assert recs[0].score == pytest.approx(0.9) and recs[0].confirmed
    rep = db.calibration_report("fx", bins=10)
    assert rep.n == 20 and sum(b.count for b in rep.bins) == 20
    hi = [b for b in rep.bins if b.count and b.lo >= 0.5]
    lo = [b for b in rep.bins if b.count and b.hi <= 0.5]
    assert sum(b.observed_rate * b.count for b in hi) / sum(b.count for b in hi) == pytest.approx(0.9)
    assert all(b.observed_rate == 0 for b in lo)
    assert 0 < rep.brier < 0.1
    assert "Brier" in str(rep)
    assert db.calibration_report("fx", bins=1).bins[0].count == 20
    empty = HeapDB()
    empty.register_effect("fx", WindowKernel(0, H))
    assert empty.calibration_report("fx").brier is None


# -- forget_entity ------------------------------------------------------------------------


def _privacy_db(credit="proportional"):
    db = HeapDB()
    db.register_effect("bloating", LogNormalKernel(2 * H, 0.8), credit=credit)
    for who in ("alice", "bob"):
        for d in range(3):
            ev = db.insert(who, d * 24 * H, "meal", {"dairy", who})
            db.feedback(who, "bloating", d * 24 * H + 2 * H, confirmed_event_id=ev.id)
    return db


def test_forget_entity_keeps_global_by_default():
    db = _privacy_db()
    before = db.link("bloating", "dairy")
    out = db.forget_entity("alice")
    assert out["events"] == 3 and out["calibration_records"] == 3 and out["links"] == 2
    assert list(db.events("alice")) == [] and db.entity_log("alice") is None
    assert "alice" not in db.entities
    assert db.link("bloating", "dairy", "alice").pos == 0
    assert db.effect("bloating").links.joint_sets("alice") == []
    assert all(r.entity == "bob" for r in db.calibration_records("bloating"))
    assert db.link("bloating", "dairy") == before  # global untouched
    assert db.top_causes("alice", "bloating", at=2 * H, k=3).causes == []


def test_forget_entity_scrub_global_subtracts_exact_contributions():
    db = _privacy_db()
    links = db.effect("bloating").links
    contrib = links.entity_counts("alice")
    before = {f: db.link("bloating", f) for f in ("dairy", "bob")}
    db.forget_entity("alice", scrub_global=True)
    for f, b in before.items():
        a = db.link("bloating", f)
        p, n = contrib.get(f, (0.0, 0.0))
        assert a.global_pos == pytest.approx(b.global_pos - p)
        assert a.global_neg == pytest.approx(b.global_neg - n)
    assert "alice" not in dict(links.items())  # her private feature is gone entirely


def test_forget_entity_scrub_global_matches_replay_without_entity():
    """With uniform credit updates don't depend on current strengths, so the
    scrubbed global layer equals one that never saw alice. (With proportional
    credit, bob's own updates were computed with strengths partly borrowed
    from alice, so they legitimately differ; scrubbing removes exactly what
    alice contributed, see the test above.)"""
    db = _privacy_db(credit="uniform")
    ref = HeapDB()
    ref.register_effect("bloating", LogNormalKernel(2 * H, 0.8), credit="uniform")
    for d in range(3):
        ev = ref.insert("bob", d * 24 * H, "meal", {"dairy", "bob"})
        ref.feedback("bob", "bloating", d * 24 * H + 2 * H, confirmed_event_id=ev.id)
    db.forget_entity("alice", scrub_global=True)
    for f in ("dairy", "bob", "alice"):
        a, b = db.link("bloating", f), ref.link("bloating", f)
        assert (a.alpha, a.beta) == pytest.approx((b.alpha, b.beta))


# -- persistence -----------------------------------------------------------------------------


def test_snapshot_roundtrip_v02_state(tmp_path):
    db = _privacy_db()
    db.register_effect("fraud", ExponentialDecay(H), global_pseudocount=3,
                       hierarchical=False, credit="uniform")
    db.register_event_type("txn", weight=lambda a: a["amount"] / 100)
    db.add_rule("fraud", "small transaction", lambda e: e.attrs["amount"] < 10, 0.3)
    db.add_rule("fraud", "refunds", None, 0.5, types={"refund"})
    t = db.insert("acct", 0, "txn", {"card"}, attrs={"amount": 5})
    db.insert("acct", 10, "refund", {"card"})
    db.feedback("acct", "fraud", 60, confirmed_event_id=t.id)
    path = tmp_path / "snap.jsonl"
    db.save(path)

    db2 = HeapDB.load(path)
    assert db2.get(t.id).attrs == {"amount": 5}
    assert db2.pending_event_types == {"txn"}
    spec = db2.effect("fraud")
    assert spec.credit == "uniform" and spec.links.global_pseudocount == 3
    assert not spec.links.hierarchical
    assert [r.name for r in spec.rules] == ["refunds"]       # type-level rule restored
    assert spec.pending_rules == ["small transaction"]       # predicate rule: by name
    db2.add_rule("fraud", "small transaction", lambda e: e.attrs["amount"] < 10, 0.3)
    db2.register_event_type("txn", weight=lambda a: a["amount"] / 100)
    assert spec.pending_rules == [] and db2.pending_event_types == set()

    for eff, ent in (("bloating", "alice"), ("bloating", "carol"), ("fraud", "acct")):
        r1 = db.top_causes(ent, eff, at=3 * 24 * H, k=5, explain=True)
        r2 = db2.top_causes(ent, eff, at=3 * 24 * H, k=5, explain=True)
        assert [(c.event, c.score) for c in r1] == [(c.event, c.score) for c in r2]
        assert [str(c.explanation) for c in r1] == [str(c.explanation) for c in r2]
    # float counts, joint links, global layer, calibration
    assert db2.link("bloating", "dairy", "alice") == db.link("bloating", "dairy", "alice")
    assert db2.link("bloating", "dairy") == db.link("bloating", "dairy")
    assert db2.effect("bloating").links.joint_sets("alice") == \
        db.effect("bloating").links.joint_sets("alice")
    assert db2.calibration_records("bloating") == db.calibration_records("bloating")
    assert db2.calibration_report("fraud").brier == db.calibration_report("fraud").brier
