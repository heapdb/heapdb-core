"""Snapshots written by the original pure-Python engine (v0.1 -> format v1,
v0.2 -> v2, v0.4 -> v3; see crates/heapdb/tests/fixtures) load into the Rust
engine and answer exactly as the v0.4 Python reference did after loading
them (expected answers recorded with that reference)."""

import json
import os

import pytest

from heapdb import HeapDB, format_result

FIX = os.path.join(os.path.dirname(__file__), "..", "crates", "heapdb", "tests", "fixtures")


@pytest.mark.parametrize("ver", [1, 2, 3])
def test_old_snapshots_load_and_answer_like_the_reference(ver, tmp_path):
    with open(os.path.join(FIX, f"snapshot_v{ver}.expected.json"), encoding="utf-8") as fh:
        exp = json.load(fh)
    db = HeapDB.load(os.path.join(FIX, f"snapshot_v{ver}.jsonl"))
    # Rewriting (as v3) and reloading must not change anything either.
    path = tmp_path / "rewritten.jsonl"
    db.save(path)
    for d in (db, HeapDB.load(path)):
        for a in exp["answers"]:
            r = d.top_causes(a["entity"], a["effect"], a["at"], k=5, explain=True)
            assert [c.event.id for c in r] == [c[0] for c in a["causes"]]
            assert [c.score for c in r] == pytest.approx([c[1] for c in a["causes"]], rel=1e-9)
            assert format_result(r) == a["text"]
        for key, (alpha, beta) in exp["links"].items():
            eff, feat, ent = key.split("|")
            b = d.link(eff, feat, ent or None)
            assert (b.alpha, b.beta) == pytest.approx((alpha, beta), rel=1e-9)
    assert sorted(db.pending_event_types) == exp["meta"]["pending_event_types"]
    for eff, rules in exp["meta"]["pending_rules"].items():
        assert db.effect(eff).pending_rules == rules
    assert db.insert("x", 0, "t").id == exp["meta"]["next_id"]
