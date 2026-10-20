"""Root-cause analysis: which change most likely caused the latency spike?

Run:  python examples/incident_rca.py

A service ("checkout") sees a stream of deploys, config changes, feature
flag flips and noisy cron jobs over a simulated week. At the end a p99
latency spike fires. We ask HeapDB for the top suspects, then show how
pruning lets it answer by reading only a few segments, and how feedback from
a postmortem sharpens future rankings. Finally the ForwardDecayHeap keeps a
continuously-updated "hottest recent change" without re-scoring anything.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heapdb import ExponentialDecay, ForwardDecayHeap, HeapDB, format_result  # noqa: E402

M = 60.0
H = 3600.0


def main() -> None:
    rng = random.Random(7)
    db = HeapDB(bucket_seconds=10 * M)

    # Latency regressions show up quickly after a change and fade over hours.
    db.register_effect(
        "latency_spike",
        kernel=ExponentialDecay(half_life=20 * M),
        default_prior=(1, 9),   # most changes are innocent
        priors={
            "db_pool_size": (3, 2),     # history says pool changes bite
            "cache_ttl": (2, 3),
            "orm_upgrade": (2, 3),
            "log_level": (1, 20),       # basically never
        },
        cause_types={"deploy", "config", "flag"},
    )

    # --- a week of background changes ------------------------------------
    kinds = {
        "deploy": ["orm_upgrade", "ui_tweak", "copy_change", "dep_bump", "refactor"],
        "config": ["cache_ttl", "log_level", "timeout", "db_pool_size"],
        "flag": ["new_checkout_flow", "promo_banner", "dark_mode"],
        "cron": ["nightly_report"],
    }
    t = 0.0
    week = 7 * 24 * H
    while t < week:
        t += rng.expovariate(1 / (45 * M))
        typ = rng.choice(list(kinds))
        feats = set(rng.sample(kinds[typ], k=min(2, len(kinds[typ]))))
        db.insert("checkout", t, typ, feats, weight=rng.choice([0.5, 1.0, 1.0, 2.0]))

    # --- the incident ------------------------------------------------------
    spike = week + 2 * H
    db.insert("checkout", spike - 95 * M, "deploy", {"ui_tweak", "copy_change"}, 1.0)
    bad = db.insert("checkout", spike - 25 * M, "config", {"db_pool_size", "timeout"}, 1.0)
    db.insert("checkout", spike - 12 * M, "config", {"log_level"}, 1.0)
    db.insert("checkout", spike - 8 * M, "flag", {"promo_banner"}, 1.0)
    db.insert("checkout", spike - 3 * M, "cron", {"nightly_report"}, 3.0)  # not a cause type

    log = db.entity_log("checkout")
    print(f"{len(db)} events in {log.num_segments} segments\n")

    res = db.top_causes("checkout", "latency_spike", at=spike, k=3, explain=True)
    print(format_result(res))

    naive = db.top_causes_naive("checkout", "latency_spike", at=spike, k=3)
    assert [e.id for _, e in naive] == [c.event.id for c in res], "pruned top-k != brute force"
    print(f"\n(brute force agrees; it scored {sum(1 for _ in db.events('checkout'))} events "
          f"vs {res.stats.events_scanned} with pruning)")

    # --- postmortem feedback ----------------------------------------------
    rejected = [c.event.id for c in res if c.event.id != bad.id]
    db.feedback("checkout", "latency_spike", spike, confirmed_event_id=bad.id,
                rejected_ids=rejected)
    print(f"\npostmortem: confirmed #{bad.id}, rejected {rejected}")
    pool = db.effect("latency_spike").links.get("db_pool_size")
    print(f"db_pool_size link now {pool.strength:.2f} (was 0.60)")

    # --- forward decay: a live 'hottest change' ticker ---------------------
    # With an exponential kernel, score = w * link * 2^(-dt/half_life), which
    # is exactly the forward-decay form. So pushing (weight * link strength)
    # gives a heap whose top IS the top cause at every future instant, with
    # no re-scoring as time passes.
    print("\nForwardDecayHeap (half-life 20m), keyed by weight x link strength:")
    spec = db.effect("latency_spike")
    fdh: ForwardDecayHeap[int] = ForwardDecayHeap.from_half_life(20 * M)
    for ev in db.events("checkout"):
        w = ev.weight * spec.links.noisy_or(ev.features)
        if ev.type in (spec.cause_types or ()) and w > 0:
            fdh.push(ev.id, w, ev.time)
    for now in (spike, spike + 30 * M, spike + 2 * H):
        top = fdh.peek_top(now)
        assert top is not None
        ref = db.most_likely_cause("checkout", "latency_spike", now)
        assert ref is not None and ref.event.id == top.item
        print(f"  at spike+{(now - spike) / M:>3.0f}m top = {db.get(top.item).short():<44} "
              f"score={top.score:.4f} (top_causes agrees: {ref.score:.4f})")


if __name__ == "__main__":
    main()
