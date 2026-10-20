"""Ranking quality against a known ground truth, with and without feedback.

Uses the simulation of ``examples/code_changes.py`` unchanged (a week of
~150 PRs on three services, hidden causal mechanisms, three effects with
different lag kernels, revert feedback). For every effect that has a true
culprit PR, each method ranks all eligible candidates and we record the
rank of the culprit:

* ``recency``        -- most recent deploy first, among deploys in the
                        effect's kernel window that are not ``docs_only``
                        (the same candidate set and hard rule HeapDB uses).
                        No timing shape, no dose, no links.
* ``kernel_only``    -- ``kernel(dt) x dose x rules`` with every link equal
                        (the link factor is the same constant for every
                        event, so it does not affect the ranking). No learning.
* ``priors_only``    -- HeapDB scoring with the example's default prior
                        (Beta(0.3, 2) for every feature) and no feedback.
                        Differs from kernel_only only because noisy-OR of
                        equal links grows with the number of features.
* ``heapdb_proportional`` -- HeapDB with revert feedback, EM (proportional)
                        credit: the culprit is confirmed, the other top-3
                        suspects are rejected (exactly as in the example).
* ``heapdb_uniform`` -- the same with v0.1 uniform credit.

Metrics per seed and per half of the week (days 0-3.5 / 3.5-7): hit@1,
hit@3 and MRR (reciprocal rank; 0 if the culprit is not among the
candidates, e.g. when the lag exceeds the kernel support). Mean and
(sample) std over seeds of the per-seed values.
"""

from __future__ import annotations

import importlib.util
import math
import os
import statistics
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(_HERE, "..", "examples", "code_changes.py")

METHODS = ("recency", "kernel_only", "priors_only", "heapdb_proportional", "heapdb_uniform")


def load_example():
    spec = importlib.util.spec_from_file_location("heapdb_example_code_changes", _EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _insert_all(ex, db, prs) -> list[int]:
    ids = []
    for t, service, feats, lines in prs:
        ids.append(db.insert(service, t, "deploy", feats, attrs={"lines_changed": lines}).id)
    for day in range(7):  # same non-deploy noise as the example
        for service in ex.SERVICES:
            db.insert(service, day * ex.D + 9 * ex.H, "oncall_handoff", {"rotation"})
    return ids


def _rank(order: list[int], cid: int) -> int | None:
    try:
        return order.index(cid) + 1
    except ValueError:
        return None


def _baseline_order(db, service: str, eff: str, at: float, mode: str) -> list[int]:
    """Candidate ids for the non-learning baselines, best first."""
    spec = db.effect(eff)
    t_lo, t_hi = db._window(spec, at, None)
    rows = []
    for ev in db.events(service):
        if not (t_lo <= ev.time <= t_hi) or ev.time > at or not spec.type_allowed(ev.type):
            continue
        rule = 1.0
        for r in spec.rules:
            if r.matches(ev):
                rule *= r.multiplier
        if rule <= 0.0:
            continue
        if mode == "recency":
            key = ev.time
        else:  # kernel_only: constant link
            key = spec.kernel.value(at - ev.time) * ev.weight * rule
            if key <= 0.0:
                continue
        rows.append((key, ev.id))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    return [i for _, i in rows]


def run_method(ex, prs, effects, method: str) -> list[tuple[str, float, int | None]]:
    """Returns (effect, time, rank or None) for effects with a culprit."""
    credit = "uniform" if method == "heapdb_uniform" else "proportional"
    db = ex.build_db(credit)
    ids = _insert_all(ex, db, prs)
    learn = method in ("heapdb_proportional", "heapdb_uniform")
    out = []
    for t, service, eff, culprit in effects:
        cid = ids[culprit] if culprit is not None else None
        if method in ("recency", "kernel_only"):
            order = _baseline_order(db, service, eff, t, method)
        else:
            res = db.top_causes(service, eff, at=t, k=10_000)
            order = [c.event.id for c in res]
        if cid is not None:
            out.append((eff, t, _rank(order, cid)))
        if learn:
            # The revert, exactly as in examples/code_changes.py.
            cleared = [i for i in order[:3] if i != cid]
            if cid is None and not cleared:
                continue
            db.feedback(service, eff, t, confirmed_event_id=cid, rejected_ids=cleared)
    return out


def metrics(rows) -> dict[str, float] | None:
    if not rows:
        return None
    n = len(rows)
    return {
        "n": n,
        "hit1": sum(1 for r in rows if r[2] == 1) / n,
        "hit3": sum(1 for r in rows if r[2] is not None and r[2] <= 3) / n,
        "mrr": sum(1.0 / r[2] for r in rows if r[2] is not None) / n,
    }


def mean_std(xs: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "n_seeds": len(xs)}


def run_quality(n_seeds: int = 20, seed0: int = 0, progress=None) -> dict[str, Any]:
    ex = load_example()
    half = ex.WEEK / 2
    periods = {"days 0-3.5": (0.0, half), "days 3.5-7": (half, math.inf),
               "all": (0.0, math.inf)}
    per_seed: dict[str, dict[str, list[dict[str, float]]]] = {
        m: {p: [] for p in periods} for m in METHODS}
    per_effect: dict[str, dict[str, list[dict[str, float]]]] = {
        m: {e: [] for e in ex.EFFECTS} for m in METHODS}
    n_attr = []
    for seed in range(seed0, seed0 + n_seeds):
        prs, effects = ex.simulate(seed)
        n_attr.append(sum(1 for e in effects if e[3] is not None))
        for m in METHODS:
            rows = run_method(ex, prs, effects, m)
            for p, (lo, hi) in periods.items():
                r = metrics([x for x in rows if lo <= x[1] < hi])
                if r is not None:
                    per_seed[m][p].append(r)
            for e in ex.EFFECTS:
                r = metrics([x for x in rows if x[0] == e])
                if r is not None:
                    per_effect[m][e].append(r)
        if progress:
            progress(seed)

    def agg(lst):
        return {k: mean_std([d[k] for d in lst]) for k in ("hit1", "hit3", "mrr")} | {
            "attributions": sum(d["n"] for d in lst)}

    # How often does recency beat HeapDB-with-feedback on a seed (second half)?
    wins = {"recency_beats_heapdb_hit1": 0, "heapdb_beats_recency_hit1": 0, "ties": 0}
    for a, b in zip(per_seed["recency"]["days 3.5-7"], per_seed["heapdb_proportional"]["days 3.5-7"]):
        if a["hit1"] > b["hit1"]:
            wins["recency_beats_heapdb_hit1"] += 1
        elif a["hit1"] < b["hit1"]:
            wins["heapdb_beats_recency_hit1"] += 1
        else:
            wins["ties"] += 1
    return {
        "n_seeds": n_seeds,
        "seeds": [seed0, seed0 + n_seeds - 1],
        "attributions_per_seed_mean": statistics.fmean(n_attr),
        "by_period": {m: {p: agg(v) for p, v in per_seed[m].items()} for m in METHODS},
        "by_effect": {m: {e: agg(v) for e, v in per_effect[m].items()} for m in METHODS},
        "seed_wins_second_half": wins,
    }
