"""Top-k with segment pruning must equal brute force, and must actually prune."""

import random

import pytest

from heapdb import ExponentialDecay, GammaKernel, HeapDB, LogNormalKernel, WindowKernel

FEATURES = [f"f{i}" for i in range(30)]
ENTITIES = ["e1", "e2", "e3"]


def random_db(seed: int, kernel, n_events=400, bucket=600.0, span=3 * 86400, max_sets=64):
    rng = random.Random(seed)
    db = HeapDB(bucket_seconds=bucket, max_sets_per_segment=max_sets)
    priors = {f: (rng.uniform(0.2, 5), rng.uniform(0.2, 5)) for f in rng.sample(FEATURES, 12)}
    db.register_effect("fx", kernel=kernel, priors=priors,
                       cause_types={"a", "b"} if seed % 2 else None,
                       hierarchical=seed % 3 != 0)
    # Soft rules: a predicate rule, a type-level rule and (sometimes) a hard block.
    db.add_rule("fx", "small", lambda e: e.attrs.get("amount", 100) < 10, 0.3)
    db.add_rule("fx", "type-c-dampened", None, 0.5, types={"c"})
    if seed % 4 == 1:
        db.add_rule("fx", "block-f0", lambda e: "f0" in e.features, 0.0)
    for _ in range(n_events):
        db.insert(
            entity=rng.choice(ENTITIES[:2]),
            time=rng.uniform(0, span),
            type=rng.choice(["a", "b", "c"]),
            features=rng.sample(FEATURES, rng.randint(0, 4)),
            # Light doses and a few heavy hitters.
            weight=rng.choice([0.05, 0.3, 1.0, 1.0, 2.0, rng.uniform(0.01, 8)]),
            attrs={"amount": rng.choice([1, 5, 50, 500])},
        )
    return db, rng


KERNEL_FACTORIES = [
    lambda: ExponentialDecay(half_life=3600),
    lambda: WindowKernel(min_lag=600, max_lag=4 * 3600),
    lambda: LogNormalKernel(median=2 * 3600, sigma=0.7),
    lambda: GammaKernel(shape=2.5, scale=2400),
]


def assert_exact(db, entity, at, k, within=None):
    got = db.top_causes(entity, "fx", at=at, k=k, within=within)
    want = db.top_causes_naive(entity, "fx", at=at, k=k, within=within)
    assert [round(s, 12) for s, _ in want] == [round(c.score, 12) for c in got]
    assert [e.id for _, e in want] == [c.event.id for c in got]
    return got


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("kf", KERNEL_FACTORIES, ids=["exp", "window", "lognormal", "gamma"])
def test_topk_equals_brute_force(seed, kf):
    # Odd seeds use a tiny set cap so the union-bound fallback is exercised too.
    db, rng = random_db(seed, kf(), max_sets=64 if seed % 2 == 0 else 2)
    for _ in range(15):
        at = rng.uniform(0, 3.2 * 86400)
        k = rng.choice([1, 2, 3, 5, 10])
        within = rng.choice([None, None, 3600.0, 7200.0])
        assert_exact(db, rng.choice(ENTITIES), at, k, within)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("credit", ["proportional", "uniform"])
def test_topk_after_feedback_still_exact(seed, credit):
    """Bounds are computed from current links, so they stay valid as entity
    and global links move (including links borrowed from other entities)."""
    db, rng = random_db(seed, LogNormalKernel(median=3600, sigma=0.5))
    db.effect("fx").credit = credit
    for _ in range(40):
        at = rng.uniform(0, 3 * 86400)
        ent = rng.choice(ENTITIES[:2])
        res = db.top_causes(ent, "fx", at=at, k=3)
        if len(res) >= 2:
            # Confirm a weaker candidate, reject the top: maximal link churn.
            db.feedback(ent, "fx", at, res[1].event.id, rejected_ids=[res[0].event.id])
        for e in ENTITIES:
            assert_exact(db, e, at, 3)


@pytest.mark.parametrize("seed", range(8))
def test_group_query_equals_brute_force_over_union(seed):
    db, rng = random_db(seed, ExponentialDecay(half_life=3600))
    for _ in range(10):
        at = rng.uniform(0, 3 * 86400)
        k = rng.choice([1, 3, 7])
        got = db.top_causes_group(ENTITIES, "fx", at=at, k=k)
        want = sorted(
            (pair for e in ENTITIES for pair in db.top_causes_naive(e, "fx", at=at, k=10**6)),
            key=lambda se: (-se[0], -se[1].id),
        )[:k]
        assert [e.id for _, e in want] == [c.event.id for c in got]
        assert [c.rank for c in got] == list(range(1, len(got) + 1))


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


def sparse_strong_db(max_sets=64):
    """Each segment mixes many weak, heavy events with one light event that
    carries a strong feature. The union bound pairs the heavy weight with the
    strong link (which never occur together); the per-set bound does not."""
    rng = random.Random(3)
    db = HeapDB(bucket_seconds=600, max_sets_per_segment=max_sets)
    db.register_effect("fx", ExponentialDecay(half_life=6 * 3600),
                       default_prior=(1, 30),
                       priors={f"strong{i}": (8, 2) for i in range(5)})
    now = 200_000.0
    for b in range(150):
        t0 = now - (b + 1) * 600
        for _ in range(10):  # heavy doses of weak features
            db.insert("x", t0 + rng.uniform(0, 599), "noise",
                      {f"w{rng.randint(0, 20)}"}, weight=rng.uniform(2.0, 4.0))
        # light dose of a strong feature
        db.insert("x", t0 + rng.uniform(0, 599), "change",
                  {f"strong{rng.randint(0, 4)}"}, weight=0.1)
    culprit = db.insert("x", now - 300, "change", {"strong0", "w1"}, weight=1.0)
    return db, now, culprit


def test_set_bound_prunes_more_than_union_bound():
    """Record both pruning rates; the per-set bound must be strictly better
    while returning the identical (brute-force) answer."""
    db, now, culprit = sparse_strong_db()
    k = 3
    new = db.top_causes("x", "fx", at=now, k=k)
    old = db.top_causes("x", "fx", at=now, k=k, _use_sets=False)  # v0.1 union bound
    want = [e.id for _, e in db.top_causes_naive("x", "fx", at=now, k=k)]
    assert [c.event.id for c in new] == [c.event.id for c in old] == want
    assert new.top.event.id == culprit.id

    n = new.stats.segments_in_window
    old_rate = old.stats.segments_pruned / n
    new_rate = new.stats.segments_pruned / n
    print(f"\npruning on sparse-strong dataset ({n} segments in window, k={k}): "
          f"union bound {old.stats.segments_pruned}/{n} pruned ({old_rate:.0%}), "
          f"{old.stats.events_scanned} events scored; "
          f"per-set bound {new.stats.segments_pruned}/{n} pruned ({new_rate:.0%}), "
          f"{new.stats.events_scanned} events scored")
    assert new.stats.union_fallbacks == 0
    assert new_rate > old_rate + 0.3
    assert new.stats.events_scanned < old.stats.events_scanned / 3

    # With a set cap of 0 every segment falls back to the union bound.
    db0, _, _ = sparse_strong_db(max_sets=0)
    fallback = db0.top_causes("x", "fx", at=now, k=k)
    assert fallback.stats.union_fallbacks == fallback.stats.segments_in_window
    assert fallback.stats.segments_pruned == old.stats.segments_pruned


def test_segment_bound_is_true_upper_bound():
    """Directly: every event's score <= its segment's bound, set and union."""
    for seed in range(4):
        db, rng = random_db(seed, GammaKernel(shape=2.5, scale=2400), max_sets=3)
        spec = db.effect("fx")
        for _ in range(10):
            at = rng.uniform(0, 3 * 86400)
            for ent in ENTITIES[:2]:
                t_lo, t_hi = db._window(spec, at, None)
                for seg in db.entity_log(ent).segments_between(t_lo, t_hi):
                    for use_sets in (True, False):
                        bound, _ = db.segment_bound(spec, seg, ent, at, t_lo, t_hi, use_sets)
                        for ev in seg.events:
                            if t_lo <= ev.time <= t_hi:
                                assert db.score(ev, "fx", at) <= bound * (1 + 1e-9)


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
    db.add_rule("fx", "small portion", lambda e: e.attrs.get("portion", 1) < 0.5, 0.4)
    db.insert("me", 0, "meal", {"beans", "onion"}, 1.5, attrs={"portion": 0.3})
    c = db.top_causes("me", "fx", at=7000, k=1, explain=True).top
    ex = c.explanation
    assert ex.score == pytest.approx(c.score)
    assert ex.dt == 7000
    assert ex.entity == "me"
    assert ex.features[0].feature == "beans"
    assert ex.link_strength == pytest.approx(1 - (1 - 0.75) * (1 - 0.2))
    assert [(r.name, r.multiplier) for r in ex.rules] == [("small portion", 0.4)]
    assert "small portion x0.4" in str(ex)
