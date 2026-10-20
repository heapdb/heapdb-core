"""v0.2 learning: proportional (EM) credit, joint links, hierarchical priors."""

import pytest

from heapdb import HeapDB, LogNormalKernel, responsibilities
from heapdb.links import LinkTable

H = 3600.0
DAY = 24 * H


def make_db(**kw):
    db = HeapDB(bucket_seconds=900)
    db.register_effect("bloating", LogNormalKernel(median=2 * H, sigma=0.8),
                       priors={"beans": (4, 1)}, cause_types={"meal"}, **kw)
    return db


# -- proportional credit ------------------------------------------------------


def test_responsibilities_formula():
    r = responsibilities({"beans": 0.8, "onion": 0.2})
    p = 1 - 0.2 * 0.8
    assert r == pytest.approx({"beans": 0.8 / p, "onion": 0.2 / p})
    assert responsibilities({"a": 0.0, "b": 0.0}) == {"a": 0.5, "b": 0.5}
    assert responsibilities({"a": 1.0, "b": 1.0}) == {"a": 1.0, "b": 1.0}  # clamped
    assert responsibilities({}) == {}


def test_strong_feature_takes_most_credit():
    db = make_db()
    ev = db.insert("me", 0, "meal", {"beans", "onion"})
    rep = db.feedback("me", "bloating", 2 * H, confirmed_event_id=ev.id)
    p = 1 - (1 - 0.8) * (1 - 0.2)
    assert rep.credit["beans"] == pytest.approx(0.8 / p)
    assert rep.credit["onion"] == pytest.approx(0.2 / p)
    assert rep.credit["beans"] > 3 * rep.credit["onion"]
    beans, onion = db.link("bloating", "beans", "me"), db.link("bloating", "onion", "me")
    # alpha += r, beta += 1 - r: total evidence is still one observation each.
    assert (beans.alpha, beans.beta) == pytest.approx((4 + 0.8 / p, 1 + 1 - 0.8 / p))
    assert (onion.alpha, onion.beta) == pytest.approx((1 + 0.2 / p, 4 + 1 - 0.2 / p))


def test_rejection_is_full_beta_per_feature():
    db = make_db()
    ev = db.insert("me", 0, "meal", {"beans", "onion"})
    db.feedback("me", "bloating", 2 * H, rejected_ids=[ev.id])
    assert db.link("bloating", "beans", "me").beta == 2
    assert db.link("bloating", "onion", "me").beta == 5


def test_weak_rider_no_longer_converges_to_strong_feature():
    """beans (strong) and onion (weak) always arrive together and are always
    confirmed. Uniform credit drags onion up to beans; proportional does not."""
    out = {}
    for credit in ("uniform", "proportional"):
        db = make_db(credit=credit)
        for d in range(30):
            ev = db.insert("me", d * DAY, "meal", {"beans", "onion"})
            db.feedback("me", "bloating", d * DAY + 2 * H, confirmed_event_id=ev.id)
        out[credit] = (db.link("bloating", "beans", "me").strength,
                       db.link("bloating", "onion", "me").strength)
    ub, uo = out["uniform"]
    pb, po = out["proportional"]
    assert uo > 0.85 and ub - uo < 0.1   # uniform: the rider is dragged up to ~beans
    assert pb > 0.9                      # proportional: the strong feature still converges
    assert po < 0.4                  # the rider stays near its prior
    assert pb - po > 0.5


def test_credit_option_validation_and_override():
    with pytest.raises(ValueError):
        make_db(credit="bogus")
    db = make_db()
    ev = db.insert("me", 0, "meal", {"beans", "onion"})
    rep = db.feedback("me", "bloating", 2 * H, confirmed_event_id=ev.id, credit="uniform")
    assert rep.credit == {"beans": 1.0, "onion": 1.0}
    with pytest.raises(ValueError):
        db.feedback("me", "bloating", 2 * H, confirmed_event_id=ev.id, rejected_ids=[ev.id])


def test_feedback_is_atomic_on_validation_error():
    db = make_db()
    ev = db.insert("me", 0, "meal", {"beans"})
    with pytest.raises(KeyError):
        db.feedback("me", "bloating", 2 * H, confirmed_event_id=ev.id, rejected_ids=[999])
    assert db.link("bloating", "beans", "me").alpha == 4
    assert db.calibration_records("bloating") == []


# -- joint links ----------------------------------------------------------------


def test_joint_cause_for_inseparable_features():
    db = make_db()
    for d in range(3):
        latte = db.insert("me", d * DAY, "meal", {"coffee", "dairy"})
        db.feedback("me", "bloating", d * DAY + 2 * H, confirmed_event_id=latte.id)
    ex = db.explain(latte, "bloating", 2 * DAY + 2 * H)
    assert ex.inseparable
    (jc,) = ex.joint_causes
    assert jc.name == "coffee+dairy" and jc.inseparable
    assert jc.joint is not None
    assert jc.joint.pos == pytest.approx(3.0)  # the set was fully credited each time
    assert jc.strength == pytest.approx(ex.link_strength)
    assert "coffee+dairy=" in str(ex) and "joint link" in str(ex)
    assert any("joint cause coffee+dairy" in w for w in ex.warnings)

    # Seen apart once -> separable; the joint link is still reported.
    db.insert("me", 3 * DAY, "meal", {"coffee"})
    ex = db.explain(latte, "bloating", 2 * DAY + 2 * H)
    assert not ex.inseparable and ex.warnings == ()
    (jc,) = ex.joint_causes
    assert not jc.inseparable and jc.joint is not None
    assert "now separable" in str(ex)


def test_joint_links_are_bounded_to_judged_sets():
    db = make_db()
    for d in range(5):
        db.insert("me", d * DAY, "meal", {"a", "b", "c"})  # never judged
    links = db.effect("bloating").links
    assert links.joint_sets("me") == []
    ev = db.insert("me", 6 * DAY, "meal", {"x", "y"})
    db.feedback("me", "bloating", 6 * DAY + H, rejected_ids=[ev.id])
    assert links.joint_sets("me") == [frozenset({"x", "y"})]
    assert links.joint("me", {"x", "y"}).neg == 1
    # Joint link does not affect scoring: per-feature noisy-OR only.
    assert db.score(ev, "bloating", 6 * DAY + H) == pytest.approx(
        db.effect("bloating").kernel.value(H) * links.noisy_or({"x", "y"}, "me"))


# -- hierarchical priors ----------------------------------------------------------


def population_db(n_users=10, rounds=4, **kw):
    db = HeapDB(bucket_seconds=900)
    db.register_effect("bloating", LogNormalKernel(median=2 * H, sigma=0.8),
                       cause_types={"meal"}, **kw)  # dairy: default prior 0.2
    for u in range(n_users):
        for d in range(rounds):
            ev = db.insert(f"u{u}", d * DAY, "meal", {"dairy"})
            db.feedback(f"u{u}", "bloating", d * DAY + 2 * H, confirmed_event_id=ev.id)
    return db


def test_new_entity_inherits_global_tendency():
    db = population_db()
    new = db.link("bloating", "dairy", "newbie")
    assert new.pos == new.neg == 0
    # 40 global confirmations, shrunk to 5 pseudo-observations: (1+5)/(1+5+4).
    assert new.global_pos + new.global_neg == pytest.approx(5.0)
    assert new.strength == pytest.approx(6 / 10)
    assert new.strength > 0.2  # beats the registered prior
    # The global view itself is unshrunk.
    assert db.link("bloating", "dairy").strength == pytest.approx(41 / 45)


def test_global_cap_and_disable():
    db = population_db(global_pseudocount=2)
    assert db.link("bloating", "dairy", "newbie").strength == pytest.approx(3 / 7)
    db = population_db(hierarchical=False)
    assert db.link("bloating", "dairy", "newbie").strength == pytest.approx(0.2)
    assert db.link("bloating", "dairy").global_pos == 40  # still tracked


def test_leave_one_out_no_double_counting():
    db = population_db(n_users=1, rounds=10)
    c = db.link("bloating", "dairy", "u0")
    assert c.global_pos == c.global_neg == 0  # only its own evidence exists
    assert c.pos == 10


def test_entity_evidence_eventually_dominates():
    db = population_db()
    t = 100 * DAY
    strengths = []
    for i in range(40):
        ev = db.insert("tolerant", t + i * DAY, "meal", {"dairy"})
        db.feedback("tolerant", "bloating", t + i * DAY + 2 * H, rejected_ids=[ev.id])
        strengths.append(db.link("bloating", "dairy", "tolerant").strength)
    assert strengths[0] > 0.5           # population says dairy is bad...
    assert strengths[-1] < 0.15         # ...but this entity's evidence wins
    # Feedback updates both layers.
    assert db.link("bloating", "dairy").global_neg == 40
    assert db.link("bloating", "dairy", "tolerant").neg == 40


def test_ranking_for_new_entity_uses_global():
    db = population_db()
    # newbie: dairy vs an unseen feature with the same default prior.
    db.insert("newbie", 100 * DAY, "meal", {"apple"})
    dairy = db.insert("newbie", 100 * DAY + 60, "meal", {"dairy"})
    assert db.top_causes("newbie", "bloating", 100 * DAY + 2 * H, k=1).top.event.id == dairy.id


def test_linktable_v1_dict_still_loads():
    t = LinkTable.from_dict({"default_prior": [1, 4], "links": {"beans": [4, 1, 2, 3]}})
    assert t.get("beans").alpha == 6 and t.get("beans").beta == 4
    assert t.prior("beans") == (4, 1)
