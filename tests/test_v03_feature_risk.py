"""v0.3: feature_risk / rank_candidates (pre-event risk) and RISK OF."""

import math
import random

import pytest

from heapdb import (
    Candidate,
    ExponentialDecay,
    GammaKernel,
    HeapDB,
    LogNormalKernel,
    RiskResult,
    WindowKernel,
    execute,
    format_result,
    parse,
)
from heapdb.cli import handle_line
from heapdb.query import QueryError, RiskQuery

H = 3600.0
M = 60.0


def _db(**kw) -> HeapDB:
    db = HeapDB(bucket_seconds=600)
    db.register_event_type("deploy", weight=lambda a: math.log1p(a["lines_changed"]))
    db.register_effect("incident", LogNormalKernel(median=3 * H, sigma=0.7),
                       priors={"migration": (3, 2), "docs_only": (1, 20)},
                       cause_types={"deploy"}, **kw)
    return db


def test_feature_risk_formula_and_explanation():
    db = _db()
    s = db.feature_risk("incident", {"migration", "config"}, weight=2.0, entity="svc")
    assert s == pytest.approx(2.0 * (1 - (1 - 0.6) * (1 - 0.2)))
    s2, ex = db.feature_risk("incident", "migration,config", weight=2.0, entity="svc",
                             explain=True)
    assert s2 == pytest.approx(s) and ex.score == pytest.approx(s)
    assert ex.prospective and ex.kernel_value == 1.0 and ex.dt == 0.0
    assert [c.feature for c in ex.features] == ["migration", "config"]
    assert "no lag" in str(ex)
    assert db.feature_risk("incident", set()) == 0.0


def test_weight_from_attrs_and_validation():
    db = _db()
    s = db.feature_risk("incident", {"migration"}, weight=None, type="deploy",
                        attrs={"lines_changed": 99})
    assert s == pytest.approx(math.log1p(99) * 0.6)
    assert db.feature_risk("incident", {"migration"}, weight=None) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        db.feature_risk("incident", {"migration"}, weight=0)
    with pytest.raises(KeyError):
        db.feature_risk("nope", {"migration"})


@pytest.mark.parametrize("seed", range(5))
def test_monotone_in_link_strength_and_dose(seed):
    rng = random.Random(seed)
    db = _db()
    feats = {"migration", "lockfile"}
    prev = db.feature_risk("incident", feats, entity="svc")
    # Confirmations can only raise strengths; risk must not drop.
    for _ in range(5):
        ev = db.insert("svc", rng.uniform(0, H), "deploy", feats, weight=1.0)
        db.feedback("svc", "incident", 2 * H, confirmed_event_id=ev.id)
        cur = db.feature_risk("incident", feats, entity="svc")
        assert cur >= prev - 1e-12
        prev = cur
    # Rejections can only lower them.
    for _ in range(5):
        ev = db.insert("svc", rng.uniform(0, H), "deploy", feats, weight=1.0)
        db.feedback("svc", "incident", 2 * H, rejected_ids=[ev.id])
        cur = db.feature_risk("incident", feats, entity="svc")
        assert cur <= prev + 1e-12
        prev = cur
    # Linear in dose, monotone in the feature set (noisy-OR).
    w = [rng.uniform(0.1, 5) for _ in range(6)]
    risks = [db.feature_risk("incident", feats, weight=x, entity="svc") for x in sorted(w)]
    assert risks == sorted(risks)
    assert db.feature_risk("incident", feats, weight=3, entity="svc") == pytest.approx(
        3 * db.feature_risk("incident", feats, weight=1, entity="svc"))
    assert (db.feature_risk("incident", feats | {"config"}, entity="svc")
            >= db.feature_risk("incident", feats, entity="svc"))


def test_rules_reduce_and_type_filter():
    db = _db()
    base = db.feature_risk("incident", {"migration", "docs_only"}, type="deploy")
    db.add_rule("incident", "fast review", lambda e: e.attrs.get("review_minutes", 99) < 5, 0.5)
    db.add_rule("incident", "docs only", lambda e: "docs_only" in e.features, 0.0)
    db.add_rule("incident", "deploys canaried", None, 0.8, types={"deploy"})
    s, ex = db.feature_risk("incident", {"migration"}, type="deploy",
                            attrs={"review_minutes": 2}, explain=True)
    assert s == pytest.approx(0.6 * 0.5 * 0.8)
    assert [r.name for r in ex.rules] == ["fast review", "deploys canaried"]
    assert db.feature_risk("incident", {"migration", "docs_only"}, type="deploy") == 0.0
    assert base > 0
    # Type-scoped rules only apply when the type is given.
    assert db.feature_risk("incident", {"migration"}) == pytest.approx(0.6)
    # cause_types applies when the type is given.
    s, ex = db.feature_risk("incident", {"migration"}, type="cron", explain=True)
    assert s == 0.0 and ex.score == 0.0 and "cause_types" in ex.rules[0].name


def test_entity_aware_and_hierarchical():
    db = _db()
    for i in range(6):
        ev = db.insert("payments", i * H, "deploy", {"lockfile"}, weight=1.0)
        db.feedback("payments", "incident", i * H + 3 * H, confirmed_event_id=ev.id)
    pay = db.feature_risk("incident", {"lockfile"}, entity="payments")
    new = db.feature_risk("incident", {"lockfile"}, entity="brand-new")
    glob = db.feature_risk("incident", {"lockfile"})
    prior = 0.2
    assert pay > new > prior          # borrows (shrunk) global evidence
    assert glob > prior
    assert new == pytest.approx(db.link("incident", "lockfile", "brand-new").strength)


@pytest.mark.parametrize("kernel", [
    ExponentialDecay(half_life=H), LogNormalKernel(median=2 * H, sigma=0.6),
    GammaKernel(shape=2.0, scale=H), WindowKernel(0, 6 * H),
])
@pytest.mark.parametrize("seed", range(4))
def test_property_equals_event_score_over_kernel(kernel, seed):
    """For an inserted event: score == kernel(dt) * feature_risk(same features)."""
    rng = random.Random(seed)
    vocab = ["a", "b", "c", "d", "e", "f"]
    db = HeapDB(bucket_seconds=900)
    db.register_event_type("deploy", weight=lambda a: math.log1p(a["n"]))
    db.register_effect("fx", kernel, priors={"a": (3, 1), "b": (1, 3)},
                       cause_types={"deploy", "flag"})
    db.add_rule("fx", "small", lambda e: e.attrs.get("n", 100) < 20, 0.3)
    db.add_rule("fx", "flags", None, 0.5, types={"flag"})
    db.add_rule("fx", "no f", lambda e: "f" in e.features, 0.0)
    events = []
    for i in range(60):
        typ = rng.choice(["deploy", "flag", "cron"])
        feats = set(rng.sample(vocab, rng.randint(1, 3)))
        attrs = {"n": rng.choice([5, 50, 500])}
        ent = rng.choice(["s1", "s2"])
        events.append(db.insert(ent, i * 300.0, typ, feats, attrs=attrs))
        if i % 7 == 6:  # interleave feedback so links are learned
            at = i * 300.0 + 60
            ev = rng.choice(events)
            if ev.entity == ent and ev.time <= at:
                db.feedback(ent, "fx", at, confirmed_event_id=ev.id)
    at = 60 * 300.0 + 30 * M
    for ev in events:
        k = kernel.value(at - ev.time)
        r = db.feature_risk("fx", ev.features, weight=ev.weight, entity=ev.entity,
                            type=ev.type, attrs=ev.attrs)
        assert db.score(ev, "fx", at) == pytest.approx(k * r, rel=1e-12, abs=1e-15)
        ex = db.explain(ev, "fx", at)
        r2, rex = db.feature_risk("fx", ev.features, weight=ev.weight, entity=ev.entity,
                                  type=ev.type, attrs=ev.attrs, explain=True)
        assert rex.link_strength == pytest.approx(ex.link_strength)
        assert rex.features == ex.features
        if db.effect("fx").type_allowed(ev.type):
            assert [r.name for r in rex.rules] == [r.name for r in ex.rules]
        if k > 0 and r > 0:
            assert db.score(ev, "fx", at) / k == pytest.approx(r)


def test_rank_candidates_topk_matches_sort():
    rng = random.Random(3)
    db = _db()
    db.add_rule("incident", "docs only", lambda e: "docs_only" in e.features, 0.0)
    vocab = ["migration", "config", "lockfile", "flag_flip", "docs_only", "dep_bump"]
    cands = []
    for i in range(80):
        feats = set(rng.sample(vocab, rng.randint(1, 3)))
        ent = rng.choice(["checkout", "payments", None])
        if rng.random() < 0.5:
            cands.append(Candidate(f"PR-{i}", feats, type="deploy",
                                   attrs={"lines_changed": rng.randint(1, 900)}, entity=ent))
        else:
            cands.append({"id": f"PR-{i}", "features": sorted(feats),
                          "weight": rng.uniform(0.5, 3), "entity": ent})
    for k in (1, 5, 10, 200):
        got = db.rank_candidates("incident", cands, k=k, entity="search", explain=True)
        ref = []
        for i, c in enumerate(cands):
            c = Candidate.from_any(c, i)
            ent = c.entity if c.entity is not None else "search"
            s = db.feature_risk("incident", c.features, weight=c.weight, entity=ent,
                                type=c.type, attrs=c.attrs)
            if s > 0:
                ref.append((-s, i, c.id))
        ref.sort()
        assert [g.candidate.id for g in got] == [cid for _, _, cid in ref[:k]]
        assert [g.rank for g in got] == list(range(1, len(got) + 1))
        for g in got:
            assert g.explanation is not None and g.explanation.score == pytest.approx(g.score)
            assert g.entity == (g.candidate.entity or "search")
            assert "docs_only" not in g.candidate.features
    # Ties: earlier candidate first.
    tie = db.rank_candidates("incident", [{"id": "a", "features": "config"},
                                          {"id": "b", "features": "config"}], k=1)
    assert tie[0].candidate.id == "a"
    with pytest.raises(ValueError):
        db.rank_candidates("incident", cands, k=0)
    with pytest.raises(ValueError):
        db.rank_candidates("incident", [{"features": "a", "bogus": 1}])
    with pytest.raises(TypeError):
        db.rank_candidates("incident", ["migration"])


def test_risk_explanation_joint_causes_from_history():
    db = _db()
    for i in range(3):
        ev = db.insert("svc", i * H, "deploy", {"lockfile", "dep_bump"}, weight=1.0)
        db.feedback("svc", "incident", i * H + 2 * H, confirmed_event_id=ev.id)
    _, ex = db.feature_risk("incident", {"lockfile", "dep_bump", "config"}, entity="svc",
                            type="deploy", explain=True)
    assert ex.inseparable
    assert [j.name for j in ex.joint_causes] == ["dep_bump+lockfile"]
    assert ex.warnings and "always co-occur" in ex.warnings[0]
    # Global view: no entity history, so no joint causes.
    _, gx = db.feature_risk("incident", {"lockfile", "dep_bump"}, explain=True)
    assert gx.joint_causes == () and gx.warnings == ()


# -- query language -----------------------------------------------------------------


def test_parse_risk():
    q = parse("RISK OF incident FOR payments FEATURES migration,review:fast WEIGHT 2.5 "
              "TYPE deploy ATTRS lines_changed=40 EXPLAIN")
    assert q == RiskQuery("incident", "payments", frozenset({"migration", "review:fast"}),
                          2.5, "deploy", (("lines_changed", 40),), True)
    q = parse("risk of incident for svc features a")
    assert (q.weight, q.type, q.explain) == (None, None, False)


@pytest.mark.parametrize("bad", [
    "RISK incident FOR s FEATURES a",
    "RISK OF incident FOR s",
    "RISK OF incident FOR s FEATURES a WEIGHT heavy",
    "RISK OF incident FOR s FEATURES a AT now",
    "RISK OF incident FEATURES a",
])
def test_parse_risk_errors(bad):
    with pytest.raises(QueryError):
        parse(bad)


def test_execute_risk_and_repl():
    db = _db()
    res = execute(db, "RISK OF incident FOR svc FEATURES migration WEIGHT 2 TYPE deploy EXPLAIN",
                  clock=lambda: 0.0)
    assert isinstance(res, RiskResult)
    assert res.score == pytest.approx(1.2) and res.explanation is not None
    text = format_result(res)
    assert "Risk of 'incident' for svc type=deploy [migration]: 1.2000" in text
    assert "no lag" in text
    res = execute(db, "RISK OF incident FOR svc FEATURES migration TYPE deploy "
                      "ATTRS lines_changed=99", clock=lambda: 0.0)
    assert res.score == pytest.approx(math.log1p(99) * 0.6)
    _, out = handle_line(db, "RISK OF incident FOR svc FEATURES migration", lambda: 0.0)
    assert out.startswith("Risk of 'incident'")
    _, help_text = handle_line(db, ".help", lambda: 0.0)
    assert "RISK OF" in help_text
