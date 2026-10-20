"""Root-cause analysis: which change most likely caused the latency spike?

Run:  python examples/incident_rca.py

A service ("checkout") sees a stream of deploys, config changes, feature
flag flips and noisy cron jobs over a simulated week. At the end a p99
latency spike fires. We ask HeapDB for the top suspects, then show how
pruning lets it answer by reading only a few segments, and how feedback from
a postmortem sharpens future rankings. Finally the ForwardDecayHeap keeps a
continuously-updated "hottest recent change" without re-scoring anything.

What v0.2 adds to the story:

* Weight = dose: a deploy's weight is derived from its size,
  ``log1p(lines_changed) / log1p(100)`` (a 100-line deploy weighs 1.0); other
  change types default to 1.0.
* ``cause_types`` stays the hard, type-level filter (cron jobs are never
  causes), and soft plausibility *rules* down-weight unlikely suspects:
  config changes made in a ``dry_run`` are multiplied by 0.1, flag flips by
  0.5 (type-level rule). Applied rules appear in the explanation.
* A cross-service *group query*: the same spike also hit ``payments`` and
  ``search``; one ``top_causes_group`` call ranks changes across all three.
* Hierarchical priors: the postmortem feedback on checkout also updates the
  global layer, so a brand-new service inherits the lesson.
"""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heapdb import ExponentialDecay, ForwardDecayHeap, HeapDB, format_result  # noqa: E402

M = 60.0
H = 3600.0
EFFECT = "latency_spike"


def deploy_weight(attrs) -> float:
    """Dose of a deploy: grows with its size, 1.0 at 100 changed lines."""
    return math.log1p(attrs.get("lines_changed", 100)) / math.log1p(100)


def main() -> None:
    rng = random.Random(7)
    db = HeapDB(bucket_seconds=10 * M)
    db.register_event_type("deploy", weight=deploy_weight)

    # Latency regressions show up quickly after a change and fade over hours.
    db.register_effect(
        EFFECT,
        kernel=ExponentialDecay(half_life=20 * M),
        default_prior=(1, 9),   # most changes are innocent
        priors={
            "db_pool_size": (3, 2),     # history says pool changes bite
            "cache_ttl": (2, 3),
            "orm_upgrade": (2, 3),
            "log_level": (1, 20),       # basically never
        },
        cause_types={"deploy", "config", "flag"},  # hard filter: cron is never a cause
    )
    # Soft plausibility rules (multipliers in [0, 1]).
    db.add_rule(EFFECT, "dry-run config", lambda e: bool(e.attrs.get("dry_run")), 0.1)
    db.add_rule(EFFECT, "flags are gated", None, 0.5, types={"flag"})

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
        attrs = {"lines_changed": rng.choice([5, 40, 120, 900])} if typ == "deploy" else {}
        db.insert("checkout", t, typ, feats, attrs=attrs)

    # --- the incident ------------------------------------------------------
    spike = week + 2 * H
    db.insert("checkout", spike - 95 * M, "deploy", {"ui_tweak", "copy_change"},
              attrs={"lines_changed": 30})
    bad = db.insert("checkout", spike - 25 * M, "config", {"db_pool_size", "timeout"})
    dry = db.insert("checkout", spike - 15 * M, "config", {"db_pool_size"},
                    attrs={"dry_run": True})
    db.insert("checkout", spike - 12 * M, "config", {"log_level"})
    db.insert("checkout", spike - 8 * M, "flag", {"promo_banner"})
    db.insert("checkout", spike - 3 * M, "cron", {"nightly_report"}, weight=3.0)  # filtered

    log = db.entity_log("checkout")
    print(f"{len(db)} events in {log.num_segments} segments\n")

    res = db.top_causes("checkout", EFFECT, at=spike, k=3, explain=True)
    print(format_result(res))

    naive = db.top_causes_naive("checkout", EFFECT, at=spike, k=3)
    assert [e.id for _, e in naive] == [c.event.id for c in res], "pruned top-k != brute force"
    print(f"\n(brute force agrees; it scored {sum(1 for _ in db.events('checkout'))} events "
          f"vs {res.stats.events_scanned} with pruning)")

    # --- the same spike across services: a group query -----------------------
    db.insert("payments", spike - 40 * M, "deploy", {"orm_upgrade", "refactor"},
              attrs={"lines_changed": 2500})
    db.insert("payments", spike - 6 * M, "flag", {"new_checkout_flow"})
    db.insert("search", spike - 30 * M, "config", {"cache_ttl"})
    db.insert("search", spike - 50 * M, "deploy", {"copy_change"}, attrs={"lines_changed": 3})
    print("\nTOP 4 CAUSES OF latency_spike FOR checkout,payments,search")
    group = db.top_causes_group(["checkout", "payments", "search"], EFFECT, at=spike, k=4,
                                explain=True)
    print(format_result(group))
    print("(scores are comparable across services only once calibrated; "
          "see calibration_report)")

    # --- postmortem feedback ----------------------------------------------
    # The dry run never touched production: it is neither cause nor evidence
    # against db_pool_size, so it is left out of the feedback.
    rejected = [c.event.id for c in res if c.event.id not in (bad.id, dry.id)]
    rep = db.feedback("checkout", EFFECT, spike, confirmed_event_id=bad.id,
                      rejected_ids=rejected)
    print(f"\npostmortem: confirmed #{bad.id}, rejected {rejected}")
    print("  proportional credit: "
          + ", ".join(f"{f} {r:.2f}" for f, r in sorted(rep.credit.items())))
    pool = db.link(EFFECT, "db_pool_size", "checkout")
    print(f"  db_pool_size link for checkout now {pool.strength:.2f} (was 0.60)")
    fresh = db.link(EFFECT, "db_pool_size", "brand-new-service")
    print(f"  a brand-new service inherits {fresh.strength:.2f} via the global layer")

    # --- forward decay: a live 'hottest change' ticker ---------------------
    # With an exponential kernel, score = dose * link * rules * 2^(-dt/half_life),
    # which is exactly the forward-decay form. So pushing (dose x link x rules)
    # gives a heap whose top IS the top cause at every future instant, with no
    # re-scoring as time passes.
    print("\nForwardDecayHeap (half-life 20m), keyed by dose x link x rules:")
    spec = db.effect(EFFECT)
    fdh: ForwardDecayHeap[int] = ForwardDecayHeap.from_half_life(20 * M)
    for ev in db.events("checkout"):
        if not spec.type_allowed(ev.type):
            continue
        ex = db.explain(ev, EFFECT, ev.time)  # dt = 0: kernel factor is 1
        w = ex.weight * ex.link_strength * ex.rule_multiplier
        if w > 0:
            fdh.push(ev.id, w, ev.time)
    for now in (spike, spike + 30 * M, spike + 2 * H):
        top = fdh.peek_top(now)
        assert top is not None
        ref = db.most_likely_cause("checkout", EFFECT, now)
        assert ref is not None and ref.event.id == top.item
        print(f"  at spike+{(now - spike) / M:>3.0f}m top = {db.get(top.item).short():<44} "
              f"score={top.score:.4f} (top_causes agrees: {ref.score:.4f})")


if __name__ == "__main__":
    main()
