"""Top-k with segment pruning must equal brute force, and must actually prune."""

import random

import pytest

from heapdb import ExponentialDecay, GammaKernel, HeapDB, LogNormalKernel, WindowKernel

FEATURES = [f"f{i}" for i in range(30)]


def random_db(seed: int, kernel, n_events=400, bucket=600.0, span=3 * 86400):
    rng = random.Random(seed)
    db = HeapDB(bucket_seconds=bucket)
    priors = {f: (rng.uniform(0.2, 5), rng.uniform(0.2, 5)) for f in rng.sample(FEATURES, 12)}
    db.register_effect("fx", kernel=kernel, priors=priors,
                       cause_types={"a", "b"} if seed % 2 else None)
    for _ in range(n_events):
        db.insert(
            entity=rng.choice(["e1", "e2"]),
            time=rng.uniform(0, span),
            type=rng.choice(["a", "b", "c"]),
            features=rng.sample(FEATURES, rng.randint(0, 4)),
            # Include zero weights and a few heavy hitters.
            weight=rng.choice([0.0, 0.3, 1.0, 1.0, 2.0, rng.uniform(0, 8)]),
        )
    return db, rng


KERNEL_FACTORIES = [
    lambda: ExponentialDecay(half_life=3600),
    lambda: WindowKernel(min_lag=600, max_lag=4 * 3600),
    lambda: LogNormalKernel(median=2 * 3600, sigma=0.7),
    lambda: GammaKernel(shape=2.5, scale=2400),
]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("kf", KERNEL_FACTORIES, ids=["exp", "window", "lognormal", "gamma"])
def test_topk_equals_brute_force(seed, kf):
    db, rng = random_db(seed, kf())
    for _ in range(15):
        at = rng.uniform(0, 3.2 * 86400)
        k = rng.choice([1, 2, 3, 5, 10])
        within = rng.choice([None, None, 3600.0, 7200.0])
        entity = rng.choice(["e1", "e2", "nobody"])
        got = db.top_causes(entity, "fx", at=at, k=k, within=within)
        want = db.top_causes_naive(entity, "fx", at=at, k=k, within=within)
        assert [round(s, 12) for s, _ in want] == [round(c.score, 12) for c in got]
        assert [e.id for _, e in want] == [c.event.id for c in got]


def test_topk_after_feedback_still_exact():
    db, rng = random_db(99, LogNormalKernel(median=3600, sigma=0.5))
    for _ in range(30):
        at = rng.uniform(0, 3 * 86400)
        res = db.top_causes("e1", "fx", at=at, k=3)
        if len(res) >= 2:
            db.feedback("e1", "fx", at, res[1].event.id, rejected_ids=[res[0].event.id])
        assert [c.event.id for c in db.top_causes("e1", "fx", at=at, k=3)] == \
               [e.id for _, e in db.top_causes_naive("e1", "fx", at=at, k=3)]


def test_pruning_actually_prunes():
    """One strong recent cause + lots of weak noise: most segments must be skipped."""
    db = HeapDB(bucket_seconds=300)
    db.register_effect("fx", ExponentialDecay(half_life=3600),
                       default_prior=(1, 50), priors={"culprit": (9, 1)})
    rng = random.Random(1)
    now = 100_000.0
    for _ in range(2000):  # weak noise over the last ~day
        db.insert("x", now - rng.uniform(0, 30_000), "noise", {f"n{rng.randint(0, 50)}"}, 0.2)
    culprit = db.insert("x", now - 600, "change", {"culprit"}, 1.0)

    res = db.top_causes("x", "fx", at=now, k=1)
    assert res.top.event.id == culprit.id
    s = res.stats
    assert s.segments_in_window > 50
    assert s.segments_scanned + s.segments_pruned == s.segments_in_window
    assert s.segments_pruned >= 0.9 * s.segments_in_window
    assert s.events_scanned < 100  # out of ~2000


def test_window_excludes_out_of_support_and_future():
    db = HeapDB(bucket_seconds=100)
    db.register_effect("fx", WindowKernel(0, 1000))
    old = db.insert("x", 0, "t", {"a"})
    ok = db.insert("x", 1500, "t", {"a"})
    db.insert("x", 2500, "t", {"a"})  # after the effect
    res = db.top_causes("x", "fx", at=2000, k=5)
    assert [c.event.id for c in res] == [ok.id]
    assert old.id not in [c.event.id for c in res]


def test_explanation_matches_score():
    db = HeapDB()
    db.register_effect("fx", LogNormalKernel(7200, 0.6), priors={"beans": (3, 1)})
    db.insert("me", 0, "meal", {"beans", "onion"}, 1.5)
    c = db.top_causes("me", "fx", at=7000, k=1, explain=True).top
    ex = c.explanation
    assert ex.score == pytest.approx(c.score)
    assert ex.dt == 7000
    assert ex.features[0].feature == "beans"
    assert ex.link_strength == pytest.approx(1 - (1 - 0.75) * (1 - 0.2))
