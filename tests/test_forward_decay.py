import math
import random

import pytest

from heapdb import ForwardDecayHeap


def naive_order(items, lam, now):
    """Brute force: compute every current score and sort."""
    scored = [(w * math.exp(-lam * (now - t)), h) for h, (w, t) in items.items()]
    scored.sort(key=lambda x: -x[0])
    return [h for _, h in scored]


def test_score_matches_definition():
    h = ForwardDecayHeap.from_half_life(100)
    h.push("a", 2.0, t=1000)
    top = h.peek_top(now=1100)
    assert top.score == pytest.approx(1.0)  # 2 * 0.5


@pytest.mark.parametrize("seed", range(5))
def test_ordering_is_time_invariant(seed):
    """The pop order must equal brute-force ordering at *any* query time."""
    rng = random.Random(seed)
    lam = math.log(2) / 300
    heap = ForwardDecayHeap(lam)
    items = {}
    for _ in range(200):
        w, t = rng.uniform(0.1, 10), rng.uniform(0, 10_000)
        items[heap.push(None, w, t)] = (w, t)

    expected = naive_order(items, lam, now=10_000)
    for now in (10_000, 20_000, 1e6):  # far future too: log-space keys don't overflow
        # Same ordering at every time; scores consistent with the definition.
        top = heap.top_k(len(items), now)
        assert [p.handle for p in top] == expected
        for p in top[:5]:
            w, t = items[p.handle]
            assert p.score == pytest.approx(w * math.exp(-lam * (now - t)), rel=1e-9, abs=1e-300)


def test_pop_and_tombstones():
    heap = ForwardDecayHeap.from_half_life(60)
    a = heap.push("a", 1.0, t=0)
    b = heap.push("b", 1.0, t=60)   # newer -> higher now
    c = heap.push("c", 5.0, t=0)    # heavier -> highest
    assert len(heap) == 3
    assert heap.remove(c)
    assert not heap.remove(c)       # already removed
    assert len(heap) == 2
    assert heap.peek_top(100).item == "b"
    assert heap.pop(100).item == "b"
    assert heap.pop(100).handle == a
    assert heap.pop(100) is None
    assert heap.peek_top(100) is None
    del b


def test_compact_and_iter():
    heap = ForwardDecayHeap(0.01)
    hs = [heap.push(i, 1 + i, t=0) for i in range(10)]
    for h in hs[::2]:
        heap.remove(h)
    heap.compact()
    assert sorted(heap) == [1, 3, 5, 7, 9]
    assert heap.peek_top(0).item == 9


def test_rejects_nonpositive_weight():
    with pytest.raises(ValueError):
        ForwardDecayHeap(0.1).push("x", 0.0, 0)
