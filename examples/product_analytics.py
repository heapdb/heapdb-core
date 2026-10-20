"""Product analytics: which user actions and exposures go with which outcomes?

Run:  python examples/product_analytics.py

There is **no human feedback** here. Nobody says "this rage click was caused
by the new editor". HeapDB learns *observationally* (v0.4):

* ``observe(user, effect, at)``: the effect happened and the cause is
  unknown. Every event in the kernel window gets a soft share of the credit
  in proportion to its current score. The effect's ``leak`` (background
  rate) takes the share of "it happened for no event in the window".
* ``settle(until)``: events whose window has passed *without* the effect get
  negative evidence (``beta += w_neg``).
* ``link_report(effect)``: features ranked by learned link strength, with
  90% credible intervals and lift.

Synthetic, seeded data: 1,200 users over ~2 months. Each user is in a
feature-flag variant (``flag:new-editor=test|control``), visits pages, does
or skips onboarding, sometimes hits an exception, and some are "power users".

Hidden ground truth (per event, noisy-OR):

* rage_click (lag: minutes):  flag:new-editor=test 0.15, perf:slow 0.25,
  exception:TypeError 0.30
* converted (lag: ~1 day):    event:onboarding_completed 0.30, page:/pricing 0.06
* churned (lag: ~2 weeks):    event:onboarding_skipped 0.20, perf:slow 0.08,
  exception:TypeError 0.08

plus background rates. CONFOUNDER: power users convert much more often (a
hidden trait, 35% background conversion) and also use the export feature a
lot. They convert about a day after an active session. ``event:used_export``
has **no effect of its own**, but observational credit cannot know that.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time as _time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heapdb import ExponentialDecay, HeapDB, LogNormalKernel, spearman  # noqa: E402

M, H, D = 60.0, 3600.0, 86400.0
SEED = 4

TRUTH = {
    "rage_click": {"flag:new-editor=test": 0.15, "perf:slow": 0.25, "exception:TypeError": 0.30},
    "converted": {"event:onboarding_completed": 0.30, "page:/pricing": 0.06},
    "churned": {"event:onboarding_skipped": 0.20, "perf:slow": 0.08, "exception:TypeError": 0.08},
}
BACKGROUND = {"rage_click": 0.02, "converted": 0.04, "churned": 0.03}
POWER_CONVERT = 0.35
LAGS = {  # how the simulator draws lags (matches the registered kernels)
    "rage_click": lambda r: r.expovariate(math.log(2) / (2 * M)),
    "converted": lambda r: math.exp(r.gauss(math.log(D), 0.6)),
    "churned": lambda r: math.exp(r.gauss(math.log(14 * D), 0.4)),
}
CONFOUNDER = "event:used_export"


def make_db() -> HeapDB:
    db = HeapDB(bucket_seconds=6 * H)
    db.register_effect("rage_click", ExponentialDecay(half_life=2 * M), leak=0.05)
    # Conversion and churn are one-shot: once a user has converted (churned),
    # their later events get no negative evidence for it.
    db.register_effect("converted", LogNormalKernel(median=D, sigma=0.6), leak=0.05,
                       repeatable=False)
    db.register_effect("churned", LogNormalKernel(median=14 * D, sigma=0.4), leak=0.05,
                       repeatable=False)
    for eff in db.effects:
        # Leaving a page is not an action that causes anything: low weight.
        db.add_rule(eff, "pageleave-low", None, 0.2, types={"pageleave"})
    return db


def simulate(rng: random.Random, n_users: int = 1200):
    """Return (events, occurrences). events: (user, t, type, features)."""
    events, occ = [], []
    for u in range(n_users):
        user = f"user-{u:04d}"
        power = rng.random() < 0.25
        variant = "test" if rng.random() < 0.5 else "control"
        t0 = rng.uniform(0, 20 * D)
        evs = []
        onboard = "completed" if rng.random() < (0.85 if power else 0.55) else "skipped"
        evs.append((t0 + 5 * M, "action", {f"event:onboarding_{onboard}"}))
        for _ in range(rng.randint(8, 14) if power else rng.randint(3, 7)):
            ts = t0 + rng.uniform(0.1, 40) * D
            for j in range(rng.randint(3, 6)):
                t = ts + j * rng.uniform(1, 6) * M
                r = rng.random()
                if r < 0.35:
                    evs.append((t, "action", {"event:used_editor", f"flag:new-editor={variant}"}))
                elif r < 0.75:
                    page = rng.choice(["/home", "/docs", "/pricing", "/settings/billing", "/projects/:id"])
                    feats = {f"page:{page}"}
                    if page == "/settings/billing" and rng.random() < 0.4:
                        feats.add("perf:slow")
                    evs.append((t, "pageview", feats))
                elif r < 0.85:
                    evs.append((t, "pageleave", {"event:$pageleave"}))
                elif r < 0.95:
                    if rng.random() < (0.6 if power else 0.08):
                        evs.append((t, "action", {CONFOUNDER}))
                    else:
                        evs.append((t, "action", {"event:invited_teammate"}))
                elif rng.random() < 0.5:
                    evs.append((t, "exception", {"exception:TypeError"}))
        evs.sort(key=lambda e: e[0])
        events += [(user, t, typ, feats) for t, typ, feats in evs]
        # Generate outcomes from the hidden truth (noisy-OR per event).
        first: dict[str, float] = {}
        for t, _typ, feats in evs:
            for eff, table in TRUTH.items():
                for f in feats:
                    if rng.random() < table.get(f, 0.0):
                        when = t + LAGS[eff](rng)
                        if eff == "rage_click":
                            occ.append((when, user, eff))
                        else:
                            first[eff] = min(first.get(eff, math.inf), when)
        end = t0 + 45 * D
        for eff, p in BACKGROUND.items():
            if rng.random() < p:
                when = rng.uniform(t0, end)
                if eff == "rage_click":
                    occ.append((when, user, eff))
                else:
                    first[eff] = min(first.get(eff, math.inf), when)
        if power and rng.random() < POWER_CONVERT:
            # Power users convert for reasons of their own, typically about a
            # day after an active session -- in which they often export.
            t_act = rng.choice(evs)[0]
            first["converted"] = min(first.get("converted", math.inf), t_act + LAGS["converted"](rng))
        occ += [(t, user, eff) for eff, t in first.items()]
    occ.sort()
    return events, occ


def main() -> None:
    started = _time.perf_counter()
    rng = random.Random(SEED)
    events, occ = simulate(rng)
    db = make_db()
    for user, t, typ, feats in events:
        db.insert(user, t, typ, feats)
    for t, user, eff in occ:
        db.observe(user, eff, t)
    horizon = max(e[1] for e in events) + 60 * D
    settled = db.settle(horizon)
    print(f"{len(events):,} events for {len(db.entities):,} users; "
          f"{len(occ):,} outcomes observed (no labelled causes)")
    print(settled)

    for eff in ("rage_click", "converted", "churned"):
        rep = db.link_report(eff, min_support=20)
        st = db.observation_stats(eff)
        print()
        print("=" * 100)
        print(f"{eff}: {int(st['occurrences'])} occurrences, mean leak responsibility "
              f"{st['mean_leak_responsibility']:.2f}")
        print(rep)
        truth = TRUTH[eff]
        feats = [r.feature for r in rep]
        ranks = {f: feats.index(f) + 1 for f in truth if f in feats}
        nulls = [r.strength for r in rep if r.feature not in truth]
        pairs = [(rep.row(f).strength > n) for f in ranks for n in nulls]
        rho = spearman([truth.get(f, 0.0) for f in feats], [r.strength for r in rep])
        print("  true causes (rank): " + ", ".join(f"{f} p={p} (#{ranks.get(f, '-')})"
                                                  for f, p in truth.items()))
        print(f"  true cause above null feature in {sum(pairs)}/{len(pairs)} pairs; "
              f"Spearman over all features {rho:.2f} (nulls tie at 0)")

    # The confounder.
    conv = db.link_report("converted", min_support=20)
    row = conv.row(CONFOUNDER)
    rank = [r.feature for r in conv].index(CONFOUNDER) + 1
    print()
    print("CONFOUNDER CAVEAT")
    print(f"  {CONFOUNDER} ranks #{rank} for 'converted' with lift {row.lift:.2f} "
          f"(90% CI {row.lo:.3f}-{row.hi:.3f}),")
    print("  yet it has NO effect in the simulation: power users export a lot AND convert")
    print("  more for unrelated reasons. Observational credit sees 'export shortly before")
    print("  conversion' and cannot tell the two apart. link_report shows associations")
    print("  under the model, not causal effects: confirm with an experiment (A/B test")
    print("  or holdout) before acting on it.")

    # Why did this user churn?
    churners = sorted({u for t, u, e in occ if e == "churned"})
    when = {u: t for t, u, e in occ if e == "churned"}
    pick = None
    for u in churners:
        res = db.top_causes(u, "churned", when[u], k=3, explain=True)
        if res.top is not None and "event:onboarding_skipped" in res.top.event.features:
            pick = (u, res)
            break
    if pick is None and churners:
        u = churners[0]
        pick = (u, db.top_causes(u, "churned", when[u], k=3, explain=True))
    if pick is not None:
        u, res = pick
        print()
        print(f"WHY DID {u} CHURN? (EXPLAIN, t={when[u] / D:.1f}d)")
        for c in res:
            print(f"  {c.rank}. {c.event.type} [{', '.join(sorted(c.event.features))}] "
                  f"{(when[u] - c.event.time) / D:.1f}d before, score {c.score:.4f}")
            print(f"      {c.explanation}")
    print()
    print(f"(ran in {_time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
