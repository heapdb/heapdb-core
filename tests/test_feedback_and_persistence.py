import pytest

from heapdb import GammaKernel, HeapDB, LogNormalKernel, WindowKernel

H = 3600.0


def make_db():
    db = HeapDB(bucket_seconds=900)
    db.register_effect("bloating", LogNormalKernel(median=2 * H, sigma=0.8),
                       priors={"beans": (4, 1)}, cause_types={"meal"})
    return db


def test_feedback_updates_beta_counts():
    db = make_db()
    a = db.insert("me", 0, "meal", {"beans", "onion"})
    b = db.insert("me", H, "meal", {"dairy"})
    db.feedback("me", "bloating", 3 * H, confirmed_event_id=b.id, rejected_ids=[a.id])
    links = db.effect("bloating").links
    assert (links.get("dairy").alpha, links.get("dairy").beta) == (2, 4)
    assert (links.get("beans").alpha, links.get("beans").beta) == (4, 2)
    assert (links.get("onion").alpha, links.get("onion").beta) == (1, 5)


def test_feedback_changes_ranking():
    db = make_db()
    rounds = []
    for day in range(6):
        t0 = day * 24 * H
        beans = db.insert("me", t0, "meal", {"beans"}, 1.0)
        dairy = db.insert("me", t0 + 0.5 * H, "meal", {"dairy"}, 1.0)
        at = t0 + 2.5 * H
        rounds.append(db.top_causes("me", "bloating", at=at, k=2).top.event.id)
        db.feedback("me", "bloating", at, confirmed_event_id=dairy.id, rejected_ids=[beans.id])
    # Initially the prior blames beans; after feedback dairy wins.
    assert db.get(rounds[0]).features == {"beans"}
    assert db.get(rounds[-1]).features == {"dairy"}


def test_feedback_validation():
    db = make_db()
    e = db.insert("me", 100, "meal", {"x"})
    db.insert("you", 100, "meal", {"x"})
    with pytest.raises(KeyError):
        db.feedback("me", "bloating", 200, confirmed_event_id=999)
    with pytest.raises(ValueError):
        db.feedback("you", "bloating", 200, confirmed_event_id=e.id)  # wrong entity
    with pytest.raises(ValueError):
        db.feedback("me", "bloating", 50, confirmed_event_id=e.id)  # cause after effect
    with pytest.raises(KeyError):
        db.feedback("me", "nope", 200, confirmed_event_id=e.id)


def test_confounding_warning():
    db = make_db()
    for d in range(3):
        latte = db.insert("me", d * 24 * H, "meal", {"coffee", "dairy"})
        db.feedback("me", "bloating", d * 24 * H + 2 * H, confirmed_event_id=latte.id)
    ex = db.top_causes("me", "bloating", at=2 * 24 * H + 2 * H, k=1, explain=True).top.explanation
    assert any("'coffee', 'dairy'" in w for w in ex.warnings)
    # Once they are observed apart, the warning goes away.
    db.insert("me", 3 * 24 * H, "meal", {"coffee"})
    ex = db.explain(db.get(latte.id), "bloating", 2 * 24 * H + 2 * H)
    assert ex.warnings == ()


def test_snapshot_roundtrip(tmp_path):
    db = make_db()
    db.register_effect("headache", GammaKernel(2, 1800), default_prior=(1, 2))
    db.register_effect("instant", WindowKernel(0, 600))
    evs = [db.insert("me", i * 1000.0, "meal", {f"f{i % 3}", "beans"}, 1 + i % 2) for i in range(20)]
    db.feedback("me", "bloating", 20_000, confirmed_event_id=evs[-1].id, rejected_ids=[evs[-2].id])
    path = tmp_path / "snap.jsonl"
    db.save(path)

    db2 = HeapDB.load(path)
    assert len(db2) == len(db)
    assert db2.effects == db.effects
    for eff in db.effects:
        r1 = db.top_causes("me", eff, at=20_500, k=5)
        r2 = db2.top_causes("me", eff, at=20_500, k=5)
        assert [(c.event, c.score) for c in r1] == [(c.event, c.score) for c in r2]
    # New ids continue after the loaded ones.
    assert db2.insert("me", 1, "meal").id == evs[-1].id + 1
