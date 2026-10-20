"""AI-heavy code changes: attribute failures to merged PRs, learn from reverts,
and rank open PRs by risk *before* they merge.

Run:  python examples/code_changes.py

A deterministic (seeded) week of development on three services (checkout,
payments, search). About 150 PRs are merged and deployed. Each deploy has
features such as ``path:payments/retry``, ``migration``, ``lockfile``,
``dep_bump``, ``config``, ``flag_flip``, ``size:large``, ``review:fast``
(approved in < 5 min), ``review:none``, ``ai_authored``, ``tests:none`` and
``docs_only``. Its dose is its size: ``log1p(lines_changed)``, scaled so that
a 100-line PR weighs 1.0.

Hidden ground truth (HeapDB never sees it):

* ``migration`` + ``review:fast``          -> incidents and error spikes
* ``lockfile`` in **payments** only        -> incidents and broken main
* ``config`` without tests (``tests:none``) -> error spikes and incidents
* ``path:search/ranking`` + ``size:large`` -> error spikes
* ``review:none``, ``dep_bump``            -> broken main
* ``ai_authored`` has **no** effect of its own. AI-authored PRs are just more
  often large and fast-reviewed, so the feature is a confounder.
* ``docs_only`` never breaks anything.

Three effects, each with a lag kernel matching how the failure shows up:
``failing_main_check`` (minutes, exponential), ``error_spike`` (log-normal,
median 30 min) and ``incident`` (log-normal, median 3 h).

PRs ship on deploy trains (10:00, 12:00, 14:00, 16:00), so several PRs
share almost the same lag and only learned links can separate them. Each
hidden mechanism fires with probability ``1 - (1 - p) ** dose``, so bigger
diffs are riskier.

Feedback comes from reverts. When an effect fires, the culprit PR is
eventually reverted, so it is CONFIRMED. The other top-3 suspects that the
postmortem cleared are REJECTED. For background spikes with no culprit, all
top-3 suspects are rejected. Links are entity-aware: each service has its own
evidence plus shrunk evidence from the others (hierarchical priors).

One simulated week has only ~50 attributions, so the per-seed numbers are
noisy. The script also prints the average over 20 seeds.

Two halves of the use case:

1. Attribution (``top_causes``): which merged PR caused this incident?
2. Priority (``feature_risk`` / ``rank_candidates``): which *open* PRs deserve
   a careful human review before they merge?
"""

from __future__ import annotations

import math
import os
import random
import sys
import time as _time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heapdb import (  # noqa: E402
    Candidate,
    ExponentialDecay,
    HeapDB,
    LogNormalKernel,
    execute,
    format_result,
)

M = 60.0
H = 3600.0
D = 24 * H
WEEK = 7 * D
SEED = 41

SERVICES = {
    "checkout": ["cart", "pricing", "ui"],
    "payments": ["retry", "ledger", "gateway"],
    "search": ["ranking", "indexer", "api"],
}
EFFECTS = ("failing_main_check", "error_spike", "incident")
TRAINS = (10 * 3600.0, 12 * 3600.0, 14 * 3600.0, 16 * 3600.0)
BENIGN_TRAITS = [("refactor", 0.30), ("logging", 0.25), ("typing", 0.20),
                 ("i18n", 0.10), ("ui_copy", 0.15), ("touches:tests", 0.40)]
KEY_FEATURES = ["migration", "review:fast", "lockfile", "config", "tests:none",
                "size:large", "ai_authored", "review:none", "dep_bump", "flag_flip"]


# ---------------------------------------------------------------------------
# Hidden ground truth
# ---------------------------------------------------------------------------


def true_failure_probs(service: str, f: frozenset[str]) -> dict[str, float]:
    """P(this PR causes each effect). Noisy-OR of hidden mechanisms.

    ``ai_authored`` does not appear here: it has no causal effect.
    """
    if "docs_only" in f:
        return {e: 0.0 for e in EFFECTS}
    mech: dict[str, list[float]] = {e: [] for e in EFFECTS}
    if "migration" in f and "review:fast" in f:
        mech["incident"].append(0.70)
        mech["error_spike"].append(0.40)
    if "lockfile" in f and service == "payments":
        mech["incident"].append(0.60)
        mech["failing_main_check"].append(0.40)
    if "config" in f and "tests:none" in f:
        mech["error_spike"].append(0.60)
        mech["incident"].append(0.35)
    if "path:search/ranking" in f and "size:large" in f:
        mech["error_spike"].append(0.55)
    if "review:none" in f:
        mech["failing_main_check"].append(0.50)
    if "dep_bump" in f:
        mech["failing_main_check"].append(0.25)
    if "flag_flip" in f:
        mech["error_spike"].append(0.10)
    base = {"failing_main_check": 0.02, "error_spike": 0.02, "incident": 0.01}
    out = {}
    for e in EFFECTS:
        miss = 1.0 - base[e]
        for p in mech[e]:
            miss *= 1.0 - p
        out[e] = 1.0 - miss
    return out


def sample_lag(rng: random.Random, effect: str) -> float:
    """Draw a lag from the same shape HeapDB's kernel assumes."""
    if effect == "failing_main_check":
        return rng.expovariate(math.log(2) / (10 * M))
    if effect == "error_spike":
        return rng.lognormvariate(math.log(30 * M), 0.6)
    return rng.lognormvariate(math.log(3 * H), 0.6)


def random_pr(rng: random.Random, service: str) -> tuple[frozenset[str], int]:
    """Features and lines changed of a merged PR."""
    if rng.random() < 0.08:
        lines = rng.randint(3, 80)
        return frozenset({f"path:{service}/docs", "docs_only"}), lines
    f: set[str] = {f"path:{service}/{rng.choice(SERVICES[service])}"}
    ai = rng.random() < 0.5
    if ai:
        f.add("ai_authored")
    lines = int(rng.lognormvariate(math.log(140 if ai else 70), 0.9)) + 1
    if lines > 400:
        f.add("size:large")
    # AI-authored PRs get rubber-stamped more often: the confounder.
    r = rng.random()
    if r < (0.06 if ai else 0.04):
        f.add("review:none")
    elif r < (0.50 if ai else 0.15):
        f.add("review:fast")
    if rng.random() < 0.15:
        f.add("migration")
    # payments pins a lot of SDKs (gateway clients), so it bumps deps more.
    if rng.random() < (0.30 if service == "payments" else 0.15):
        f.add("dep_bump")
        if rng.random() < 0.7:
            f.add("lockfile")
    elif rng.random() < 0.06:
        f.add("lockfile")
    if rng.random() < 0.20:
        f.add("config")
    if rng.random() < 0.08:
        f.add("flag_flip")
    if rng.random() < (0.30 if ai else 0.20):
        f.add("tests:none")
    # Common, harmless traits: noise that makes innocent PRs look busy.
    for trait, p in BENIGN_TRAITS:
        if rng.random() < p:
            f.add(trait)
    return frozenset(f), lines


def simulate(seed: int = SEED):
    """Merged PRs (as deploys) and the effects they really caused.

    Returns ``(prs, effects)``; ``effects`` are ``(time, service, effect,
    culprit_index | None)`` sorted by time. ``None`` means background noise
    (a traffic blip), where no deploy is at fault.
    """
    rng = random.Random(seed)
    prs = []
    for day in range(7):
        weekend = day >= 5
        for service in SERVICES:
            n = rng.randint(1, 3) if weekend else rng.randint(8, 11)
            for _ in range(n):
                # PRs merge during the day and ship with the next deploy
                # train (10:00, 12:00, 14:00, 16:00), a few seconds apart.
                # PRs on the same train have the same lag, so only learned
                # links can tell them apart.
                train = day * D + rng.choice(TRAINS)
                t = train + rng.uniform(0, 120)
                feats, lines = random_pr(rng, service)
                prs.append((t, service, feats, lines))
    prs.sort(key=lambda p: p[0])
    effects = []
    for i, (t, service, feats, lines) in enumerate(prs):
        dose = deploy_dose({"lines_changed": lines})
        for eff, p in true_failure_probs(service, feats).items():
            # Bigger diffs are more exposure: each mechanism gets `dose` tries.
            if rng.random() < 1.0 - (1.0 - p) ** dose:
                effects.append((t + sample_lag(rng, eff), service, eff, i))
    for _ in range(4):  # background error spikes nobody caused
        effects.append((rng.uniform(D, WEEK), rng.choice(list(SERVICES)), "error_spike", None))
    effects = [e for e in effects if e[0] < WEEK + 6 * H]
    effects.sort(key=lambda e: e[0])
    return prs, effects


# ---------------------------------------------------------------------------
# HeapDB setup and the online loop
# ---------------------------------------------------------------------------


def deploy_dose(attrs) -> float:
    """``log1p(lines_changed)``, scaled so that a 100-line PR has dose 1.0."""
    return math.log1p(attrs["lines_changed"]) / math.log1p(100)


def build_db(credit: str = "proportional") -> HeapDB:
    db = HeapDB(bucket_seconds=15 * M)
    db.register_event_type("deploy", weight=deploy_dose)
    kernels = {
        "failing_main_check": ExponentialDecay(half_life=10 * M),
        "error_spike": LogNormalKernel(median=30 * M, sigma=0.6),
        "incident": LogNormalKernel(median=3 * H, sigma=0.6),
    }
    for eff, k in kernels.items():
        # Weak, sceptical default prior (mean 0.13, worth 2.3 observations) so
        # a few reverts move a link; services share evidence through the
        # global layer (up to 5 pseudo-observations, the default).
        db.register_effect(eff, k, default_prior=(0.3, 2.0), cause_types=["deploy"],
                           credit=credit)
        db.add_rule(eff, "docs-only change", lambda e: "docs_only" in e.features, 0.0)
    return db


def run(db: HeapDB, prs, effects, learn: bool = True, verbose: bool = False):
    """Insert PRs, then replay effects in time order: query, then revert feedback.

    All PRs are inserted up front. That is safe because scoring ignores
    events after the effect (negative lag scores 0), so every query only
    sees PRs merged before it.
    """
    ids = []
    for t, service, feats, lines in prs:
        ev = db.insert(service, t, "deploy", feats, attrs={"lines_changed": lines})
        ids.append(ev.id)
    # Some non-deploy noise: on-call handoffs are never causes (cause_types).
    for day in range(7):
        for service in SERVICES:
            db.insert(service, day * D + 9 * H, "oncall_handoff", {"rotation"})

    log = []  # (effect, time, service, culprit_id, rank or None)
    shown = 0
    for t, service, eff, culprit in effects:
        res = db.top_causes(service, eff, at=t, k=5, explain=verbose)
        ranked = [c.event.id for c in res]
        cid = ids[culprit] if culprit is not None else None
        rank = ranked.index(cid) + 1 if cid in ranked else None
        if cid is not None:
            log.append((eff, t, service, cid, rank))
        if verbose and eff == "incident" and cid is not None and shown < 3 and (
                shown == 0 or t > 2.2 * D * shown):
            shown += 1
            print(f"\n--- incident on {service} at day {t / D:.2f} "
                  f"(true culprit #{cid}, ranked {rank or '>5'}) ---")
            print(f"TOP 3 CAUSES OF incident FOR {service} AT {t:.0f} EXPLAIN")
            res3 = db.top_causes(service, eff, at=t, k=3, explain=True)
            print(format_result(res3))
        if not learn:
            continue
        # The revert: culprit confirmed, the other top-3 suspects cleared.
        cleared = [i for i in ranked[:3] if i != cid]
        if cid is None and not cleared:
            continue
        db.feedback(service, eff, t, confirmed_event_id=cid, rejected_ids=cleared)
    return log


def hit_rates(log, lo: float, hi: float, effect: str | None = None):
    rows = [r for r in log if lo <= r[1] < hi and (effect is None or r[0] == effect)]
    n = len(rows)
    if n == 0:
        return 0, float("nan"), float("nan")
    h1 = sum(1 for r in rows if r[4] == 1) / n
    h3 = sum(1 for r in rows if r[4] is not None and r[4] <= 3) / n
    return n, h1, h3


def print_seed_average(n_seeds: int) -> None:
    """One simulated week is small and noisy; average over other seeds."""
    acc = {k: [0, 0] for k in ("fb1", "fb2", "nofb2")}
    acc3 = {k: [0, 0] for k in acc}
    for seed in range(n_seeds):
        prs, effects = simulate(seed)
        fb = run(build_db(), prs, effects)
        nofb = run(build_db(), prs, effects, learn=False)
        for key, lg, lo, hi in (("fb1", fb, 0, WEEK / 2), ("fb2", fb, WEEK / 2, math.inf),
                                ("nofb2", nofb, WEEK / 2, math.inf)):
            rows = [r for r in lg if lo <= r[1] < hi]
            acc[key][0] += sum(r[4] == 1 for r in rows)
            acc3[key][0] += sum(r[4] is not None and r[4] <= 3 for r in rows)
            acc[key][1] += len(rows)
            acc3[key][1] += len(rows)
    f = {k: (acc[k][0] / acc[k][1], acc3[k][0] / acc3[k][1]) for k in acc}
    print(f"  averaged over {n_seeds} seeds ({acc['fb1'][1] + acc['fb2'][1]} attributions): "
          f"hit@1/hit@3 days 0-3.5 {f['fb1'][0]:.2f}/{f['fb1'][1]:.2f} -> "
          f"days 3.5-7 {f['fb2'][0]:.2f}/{f['fb2'][1]:.2f} "
          f"(no feedback: {f['nofb2'][0]:.2f}/{f['nofb2'][1]:.2f})")


def print_hits(title: str, log) -> None:
    print(f"\n{title}")
    print(f"  {'effect':<20} {'half':<12} {'n':>3}  {'hit@1':>6}  {'hit@3':>6}")
    half = WEEK / 2
    for eff in (*EFFECTS, None):
        for name, lo, hi in (("days 0-3.5", 0, half), ("days 3.5-7", half, math.inf)):
            n, h1, h3 = hit_rates(log, lo, hi, eff)
            print(f"  {eff or 'ALL':<20} {name:<12} {n:>3}  {h1:>6.2f}  {h3:>6.2f}")


# ---------------------------------------------------------------------------
# Pre-merge review queue
# ---------------------------------------------------------------------------

OPEN_PRS = [
    Candidate("PR-1201", {"path:payments/retry", "lockfile", "dep_bump", "ai_authored"},
              type="deploy", attrs={"lines_changed": 60}, entity="payments"),
    Candidate("PR-1202", {"path:checkout/pricing", "migration", "review:fast", "ai_authored",
                          "size:large"},
              type="deploy", attrs={"lines_changed": 820}, entity="checkout"),
    Candidate("PR-1203", {"path:search/docs", "docs_only", "ai_authored"},
              type="deploy", attrs={"lines_changed": 40}, entity="search"),
    Candidate("PR-1204", {"path:search/indexer", "config", "tests:none"},
              type="deploy", attrs={"lines_changed": 25}, entity="search"),
    Candidate("PR-1205", {"path:checkout/ui", "ai_authored", "size:large", "review:fast"},
              type="deploy", attrs={"lines_changed": 1400}, entity="checkout"),
    Candidate("PR-1206", {"path:search/ranking", "lockfile", "dep_bump"},
              type="deploy", attrs={"lines_changed": 90}, entity="search"),
    Candidate("PR-1207", {"path:payments/ledger", "migration"},
              type="deploy", attrs={"lines_changed": 210}, entity="payments"),
    Candidate("PR-1208", {"path:checkout/cart", "flag_flip"},
              type="deploy", attrs={"lines_changed": 8}, entity="checkout"),
    Candidate("PR-1209", {"path:payments/gateway", "review:none", "ai_authored"},
              type="deploy", attrs={"lines_changed": 150}, entity="payments"),
    Candidate("PR-1210", {"path:search/ranking", "size:large", "ai_authored"},
              type="deploy", attrs={"lines_changed": 650}, entity="search"),
]


def print_review_queue(db: HeapDB) -> None:
    print("\n=== Pre-merge review queue: 10 open PRs ranked by incident risk ===")
    print("(rank_candidates -> feature_risk = dose x noisy-OR of current links x rules; "
          "no timing term,\n the PR has not merged yet)")
    queue = db.rank_candidates("incident", OPEN_PRS, k=10, explain=True)
    for row in queue:
        c = row.candidate
        truth = true_failure_probs(row.entity, c.features)["incident"]
        print(f"  {row.rank:>2}. {c.id} {row.entity:<9} risk={row.score:.3f} "
              f"(hidden P(incident)={truth:.2f})  [{','.join(sorted(c.features))}]")
        if row.rank <= 3:
            print(f"      {row.explanation}")
    dropped = {c.id for c in OPEN_PRS} - {r.candidate.id for r in queue}
    print(f"  dropped (score 0, hard-blocked by a rule): {', '.join(sorted(dropped))}")
    print("  Reading it: links are per-feature (noisy-OR), so a harmful *combination*\n"
          "  (migration+review:fast, config+tests:none, lockfile in payments) is learned\n"
          "  only through its members. That is why PR-1205 (review:fast without a migration)\n"
          "  still ranks high. Add explicit cross features (e.g. 'migration&review:fast')\n"
          "  if interactions matter. After one week the per-service split is still small.")
    # The same thing through the query language, for one PR.
    q = ("RISK OF incident FOR payments FEATURES path:payments/retry,lockfile,dep_bump,"
         "ai_authored TYPE deploy ATTRS lines_changed=60 EXPLAIN")
    print(f"\n{q}")
    print(format_result(execute(db, q, clock=lambda: WEEK)))
    print("\nTop 3 open PRs by error_spike risk (a different effect, a different queue):")
    for row in db.rank_candidates("error_spike", OPEN_PRS, k=3):
        print(f"  {row.rank}. {row.candidate.id} {row.entity:<9} risk={row.score:.3f}")


# ---------------------------------------------------------------------------


def main() -> None:
    t0 = _time.perf_counter()
    prs, effects = simulate()
    n_ai = sum("ai_authored" in f for _, _, f, _ in prs)
    print(f"Simulated week: {len(prs)} merged PRs ({n_ai} AI-authored) across "
          f"{len(SERVICES)} services; effects: "
          + ", ".join(f"{sum(e[2] == x for e in effects)} {x}" for x in EFFECTS)
          + f" ({sum(e[3] is None for e in effects)} with no culprit)")

    db = build_db("proportional")
    log = run(db, prs, effects, learn=True, verbose=True)

    print_hits("Attribution accuracy with revert feedback (rank of the true culprit):", log)
    base_log = run(build_db(), prs, effects, learn=False)
    _, h1, h3 = hit_rates(base_log, WEEK / 2, math.inf)
    print(f"  baseline, days 3.5-7 with NO feedback (priors only): hit@1 {h1:.2f}, hit@3 {h3:.2f}")
    print_seed_average(20)

    print("\nLearned incident links (entity-aware; global view = population):")
    print(f"  {'feature':<14} {'global':>7} {'checkout':>9} {'payments':>9} {'search':>7}")
    for f in KEY_FEATURES:
        row = [db.link("incident", f).strength] + [
            db.link("incident", f, s).strength for s in SERVICES]
        print(f"  {f:<14} " + " ".join(f"{v:>{w}.3f}" for v, w in zip(row, (7, 9, 9, 7))))

    # Proportional vs uniform credit: does the confounder get blamed?
    udb = build_db("uniform")
    ulog = run(udb, prs, effects, learn=True)
    print("\nai_authored has NO causal effect (it only co-occurs with size:large and "
          "review:fast).")
    print(f"  {'effect':<20} {'credit':<13} {'ai_authored':>11} {'review:fast':>11} "
          f"{'migration':>9}  {'2nd-half hit@1':>14}")
    for eff in EFFECTS:
        for name, d, lg in (("proportional", db, log), ("uniform", udb, ulog)):
            vals = [d.link(eff, f).strength for f in ("ai_authored", "review:fast", "migration")]
            _, h1, _ = hit_rates(lg, WEEK / 2, math.inf, eff)
            print(f"  {eff:<20} {name:<13} {vals[0]:>11.3f} {vals[1]:>11.3f} "
                  f"{vals[2]:>9.3f}  {h1:>14.2f}")
    p_ai = db.link("incident", "ai_authored").strength
    u_ai = udb.link("incident", "ai_authored").strength
    print(f"  -> incident link of ai_authored: proportional {p_ai:.3f} vs uniform {u_ai:.3f}")

    print()
    print(db.calibration_report("incident", bins=5))
    print(db.calibration_report("error_spike", bins=5))

    # Cross-service group query: which recent change is the hottest error-spike
    # suspect anywhere, right now?
    at = 3 * D + 17 * H
    print(f"\nTOP 4 CAUSES OF error_spike FOR checkout,payments,search AT {at:.0f}")
    print(format_result(db.top_causes_group(list(SERVICES), "error_spike", at=at, k=4)))

    print_review_queue(db)
    print(f"\n(ran in {_time.perf_counter() - t0:.2f}s)")


if __name__ == "__main__":
    main()
