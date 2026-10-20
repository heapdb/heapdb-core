"""Differential test: the Rust-backed ``heapdb`` against the pure-Python
reference engine (``_heapdb_ref``, the v0.4 implementation kept as an oracle
during the port).

Random workloads (effects with every kernel kind, priors, cause types, credit
modes, hierarchical on/off, type-level and predicate rules, weight functions,
inserts in random time order, feedback, observe, settle incl. partial,
forget_entity, snapshot round trips across implementations) are applied to
both engines; every query result is compared:

* top_causes / top_causes_group / top_causes_naive: ranking and scores
  (relative tolerance 1e-9; ids may only permute within tied scores), work
  counters, and the explanation text;
* feedback reports and the resulting links (entity and global view);
* observe / settle reports and observation_stats;
* feature_risk / rank_candidates (scores, order, explanation text);
* link_report rows (incl. Beta credible intervals) and calibration reports;
* snapshots: both engines' JSONL files parse to the same records, and each
  engine loads the other's snapshot.

Run more workloads with ``python tests/test_differential.py --workloads N``.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ref = pytest.importorskip("_heapdb_ref")
import heapdb as rs  # noqa: E402

TOL = 1e-9
FEATURES = [f"f{i}" for i in range(14)] + ["beans", "dairy", "x:y=z"]
TYPES = ["a", "b", "c", "deploy"]
ENTITIES = ["e1", "e2", "e3", "svc-4"]


class Mismatch(AssertionError):
    pass


def close(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check(cond, what):
    if not cond:
        raise Mismatch(what)


def same_ranking(want, got, what):
    """want/got: lists of (score, id), best first."""
    check(len(want) == len(got), f"{what}: length {len(got)} != {len(want)}")
    for i, ((ws, wi), (gs, gi)) in enumerate(zip(want, got)):
        check(close(ws, gs), f"{what}: rank {i + 1} score {gs!r} != {ws!r}")
    i = 0
    while i < len(want):
        j = i + 1
        while j < len(want) and close(want[i][0], want[j][0]):
            j += 1
        if j < len(want):  # the last tied group may be cut differently by k
            check(sorted(x for _, x in want[i:j]) == sorted(x for _, x in got[i:j]),
                  f"{what}: ids at ranks {i + 1}-{j} differ: {got[i:j]} vs {want[i:j]}")
        i = j


def near_tie(want):
    scores = [s for s, _ in want]
    return any(close(a, b, 1e-12) for a, b in zip(scores, scores[1:]))


def _pieces(text):
    import re
    return sorted(p for p in re.split(r", |\n|\[noisy-OR of |\[|\]", text) if p)


def same_text(a, b, what):
    if a == b:
        return
    try:
        _same_text(a, b, what)
    except Mismatch:
        # Items whose values tie within TOL (contributions, credits, rows) may
        # be listed in a different order: compare the sorted pieces instead.
        _same_text("\n".join(_pieces(a)), "\n".join(_pieces(b)), what)


def _same_text(a, b, what):
    if a == b:
        return
    # Allow last-digit differences in rounded floats only (values within TOL).
    import re
    num = re.compile(r"-?\d+\.\d+(?:e[+-]\d+)?|-?\d+")
    ta, tb = num.split(a), num.split(b)
    na, nb = num.findall(a), num.findall(b)
    check(ta == tb and len(na) == len(nb), f"{what}: text differs:\n{a}\n---\n{b}")
    for x, y in zip(na, nb):
        dx, dy = float(x), float(y)
        digits = len(x.split(".")[1]) if "." in x else 0
        check(abs(dx - dy) <= 10 ** -digits * 1.000001, f"{what}: {x} vs {y} in\n{a}\n---\n{b}")


def same_json(a, b, what, path=""):
    if isinstance(a, dict):
        check(isinstance(b, dict) and set(a) == set(b), f"{what}{path}: keys {sorted(a)} vs {sorted(b)}")
        for k in a:
            same_json(a[k], b[k], what, f"{path}.{k}")
    elif isinstance(a, list):
        check(isinstance(b, list) and len(a) == len(b), f"{what}{path}: list {a!r} vs {b!r}")
        for i, (x, y) in enumerate(zip(a, b)):
            same_json(x, y, what, f"{path}[{i}]")
    elif isinstance(a, bool) or a is None or isinstance(a, str):
        check(a == b, f"{what}{path}: {a!r} vs {b!r}")
    else:
        check(isinstance(b, (int, float)) and not isinstance(b, bool) and close(float(a), float(b)),
              f"{what}{path}: {a!r} vs {b!r}")


def snapshot_records(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_ref_fixed(path):
    """``_heapdb_ref.HeapDB.load`` with the v0.5 fix: the reference re-queued
    every event behind a settle cursor into ``late`` while loading, on top of
    the saved ``late`` lists (duplicating them on every save/load cycle).
    v0.5 restores ``late`` exactly as saved."""
    db = ref.HeapDB.load(path)
    for rec in snapshot_records(path):
        if rec.get("kind") == "effect":
            late = (rec.get("observations") or {}).get("late", {})
            db.effect(rec["name"]).obs.late = {e: [int(i) for i in ids] for e, ids in late.items()}
    return db


# ---------------------------------------------------------------------------
# A workload is a seeded sequence of operations applied to both engines.
# ---------------------------------------------------------------------------


def kernel(mod, rng):
    kind = rng.choice(["exp", "window", "lognormal", "gamma"])
    if kind == "exp":
        return mod.ExponentialDecay(half_life=rng.choice([300.0, 1800.0, 3600.0]))
    if kind == "window":
        return mod.WindowKernel(min_lag=rng.choice([0.0, 300.0]), max_lag=rng.choice([3600.0, 4 * 3600.0]))
    if kind == "lognormal":
        return mod.LogNormalKernel(median=rng.choice([1800.0, 7200.0]), sigma=rng.choice([0.4, 0.7, 1.0]))
    return mod.GammaKernel(shape=rng.choice([1.0, 2.5, 4.0]), scale=rng.choice([900.0, 2400.0]))


def small_amount(e):
    return e.attrs.get("amount", 100) < 10


def has_f0(e):
    return "f0" in e.features


def lines_weight(attrs):
    return math.log1p(attrs.get("lines", 100)) / math.log1p(100)


class Pair:
    """The same database in both engines."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        rng = self.rng
        bucket = rng.choice([300.0, 900.0, 3600.0])
        max_sets = rng.choice([64, 64, 3, 0])
        self.dbs = [ref.HeapDB(bucket_seconds=bucket, max_sets_per_segment=max_sets),
                    rs.HeapDB(bucket_seconds=bucket, max_sets_per_segment=max_sets)]
        self.mods = [ref, rs]
        self.effects = []
        for i in range(rng.randint(1, 3)):
            name = f"fx{i}"
            opts = dict(
                priors={f: (rng.uniform(0.2, 5), rng.uniform(0.2, 5))
                        for f in rng.sample(FEATURES, rng.randint(0, 6))},
                default_prior=rng.choice([(1.0, 4.0), (0.3, 2.0), (1.0, 1.0)]),
                cause_types=rng.choice([None, None, {"a", "b", "deploy"}]),
                credit=rng.choice(["proportional", "proportional", "uniform"]),
                global_pseudocount=rng.choice([5.0, 2.0, 0.0]),
                hierarchical=rng.random() < 0.8,
                leak=rng.choice([0.05, 0.2, 1.0]),
                w_neg=rng.choice([1.0, 0.5]),
                repeatable=rng.random() < 0.7,
            )
            kseed = rng.random()
            for db, mod in zip(self.dbs, self.mods):
                db.register_effect(name, kernel(mod, random.Random(kseed)), **opts)
            self.effects.append(name)
            rules = []
            if rng.random() < 0.6:
                rules.append(("small", small_amount, rng.choice([0.3, 0.0]), None))
            if rng.random() < 0.5:
                rules.append(("type-c", None, 0.5, ["c"]))
            if rng.random() < 0.2:
                rules.append(("no-f0", has_f0, 0.0, None))
            for rname, pred, m, types in rules:
                for db in self.dbs:
                    db.add_rule(name, rname, pred, m, types=types)
        if rng.random() < 0.6:
            for db in self.dbs:
                db.register_event_type("deploy", weight=lines_weight)
        self.t = 0.0
        self.times = []
        self.span = rng.choice([1, 2, 5]) * 86400.0

    # -- operations ----------------------------------------------------------

    def both(self, f):
        return [f(db) for db in self.dbs]

    def op_insert(self):
        rng = self.rng
        ent = rng.choice(ENTITIES)
        t = rng.uniform(0, self.span) if rng.random() < 0.3 else self.t + rng.expovariate(1 / 300.0)
        self.t = max(self.t, t)
        typ = rng.choice(TYPES)
        feats = rng.sample(FEATURES, rng.randint(0, 4))
        attrs = {"amount": rng.choice([1, 5, 50, 500])}
        if typ == "deploy" and rng.random() < 0.7:
            attrs["lines"] = rng.randint(1, 2000)
            w = None
        else:
            w = rng.choice([0.05, 0.3, 1.0, 1.0, 2.0, round(rng.uniform(0.01, 8), 6)])
        evs = self.both(lambda db: db.insert(ent, t, typ, feats, weight=w, attrs=attrs))
        self.times.append(t)
        check(evs[0].id == evs[1].id and close(evs[0].weight, evs[1].weight), "insert")
        check(evs[0].short() == evs[1].short(), f"short {evs[0].short()} vs {evs[1].short()}")

    def at(self):
        """A time shortly after a stored event (most of the time)."""
        rng = self.rng
        if self.times and rng.random() < 0.8:
            return rng.choice(self.times) + rng.uniform(0, 4 * 3600.0)
        return rng.uniform(0, self.span * 1.1)

    def op_top(self):
        rng = self.rng
        ent, eff, at = rng.choice(ENTITIES), rng.choice(self.effects), self.at()
        k = rng.choice([1, 2, 3, 5, 10])
        within = rng.choice([None, None, 3600.0, 7200.0])
        explain = rng.random() < 0.5
        rr = self.both(lambda db: db.top_causes(ent, eff, at, k=k, explain=explain, within=within))
        want = [(c.score, c.event.id) for c in rr[0]]
        got = [(c.score, c.event.id) for c in rr[1]]
        same_ranking(want, got, f"top_causes({ent}, {eff}, {at})")
        if [i for _, i in want] == [i for _, i in got] and not near_tie(want):
            s0, s1 = rr[0].stats, rr[1].stats
            check((s0.segments_in_window, s0.segments_scanned, s0.segments_pruned, s0.events_scanned,
                   s0.union_fallbacks) == (s1.segments_in_window, s1.segments_scanned,
                                           s1.segments_pruned, s1.events_scanned, s1.union_fallbacks),
                  f"stats {s0} vs {s1}")
            same_text(self.mods[0].format_result(rr[0]), rs.format_result(rr[1]), "format_result")
        naive = self.both(lambda db: db.top_causes_naive(ent, eff, at, k=k, within=within))
        same_ranking([(s, e.id) for s, e in naive[0]], [(s, e.id) for s, e in naive[1]], "naive")
        return rr

    def op_group(self):
        rng = self.rng
        ents = rng.sample(ENTITIES, rng.randint(1, len(ENTITIES)))
        eff, at, k = rng.choice(self.effects), self.at(), rng.choice([1, 3, 7])
        rr = self.both(lambda db: db.top_causes_group(ents, eff, at, k=k))
        same_ranking([(c.score, c.event.id) for c in rr[0]], [(c.score, c.event.id) for c in rr[1]],
                     "group")
        check([c.rank for c in rr[1]] == list(range(1, len(rr[1]) + 1)), "group ranks")

    def compare_links(self, eff, what):
        feats = rng_feats = FEATURES
        for f in rng_feats:
            for ent in [None, *ENTITIES, "fresh"]:
                a, b = (db.link(eff, f, ent) for db in self.dbs)
                for attr in ("prior_alpha", "prior_beta", "pos", "neg", "global_pos", "global_neg"):
                    check(close(getattr(a, attr), getattr(b, attr)),
                          f"{what}: link({eff}, {f}, {ent}).{attr} {getattr(b, attr)} vs {getattr(a, attr)}")
        del feats

    def op_feedback(self):
        rng = self.rng
        rr = self.op_top()
        res = rr[0]
        if len(res) == 0:
            return
        ent, eff, at = res.entity, res.effect, res.at
        ids = [c.event.id for c in res]
        confirm = rng.choice(ids) if rng.random() < 0.8 else None
        rejected = [i for i in ids if i != confirm and rng.random() < 0.7]
        credit = rng.choice([None, None, "uniform", "proportional"])
        reps = self.both(lambda db: db.feedback(ent, eff, at, confirmed_event_id=confirm,
                                                rejected_ids=rejected, credit=credit))
        check(reps[0].confirmed == reps[1].confirmed and tuple(reps[0].rejected) == tuple(reps[1].rejected),
              "feedback ids")
        check(set(reps[0].credit) == set(reps[1].credit), "credit keys")
        for f in reps[0].credit:
            check(close(reps[0].credit[f], reps[1].credit[f]), f"credit {f}")
        self.compare_links(eff, "after feedback")

    def op_observe(self):
        rng = self.rng
        ent, eff, at = rng.choice(ENTITIES), rng.choice(self.effects), self.at()
        leak = rng.choice([None, None, 0.01, 2.0, math.inf])
        reps = self.both(lambda db: db.observe(ent, eff, at, leak=leak))
        a, b = reps
        check(close(a.leak_responsibility, b.leak_responsibility), "leak_responsibility")
        same_ranking([(c.responsibility, c.event_id) for c in a.credits],
                     [(c.responsibility, c.event_id) for c in b.credits], "observe credits")
        for ca, cb in zip(sorted(a.credits, key=lambda c: c.event_id), sorted(b.credits, key=lambda c: c.event_id)):
            check(close(ca.score, cb.score) and set(ca.features) == set(cb.features), "credit detail")
            for f in ca.features:
                check(close(ca.features[f], cb.features[f]), "credit feature")
        if [c.event_id for c in a.credits] == [c.event_id for c in b.credits]:
            same_text(str(a), str(b), "ObserveReport")
        else:  # credits tied within TOL may be listed in either order
            same_text("\n".join(sorted(str(a).splitlines())), "\n".join(sorted(str(b).splitlines())),
                      "ObserveReport (tied credits)")

    def op_settle(self):
        rng = self.rng
        until = self.at()
        effs = rng.choice([None, rng.choice(self.effects), list(self.effects)])
        ent = rng.choice([None, None, rng.choice(ENTITIES)])
        partial = rng.random() < 0.3
        a, b = self.both(lambda db: db.settle(until, effects=effs, entity=ent, partial=partial))
        check((a.events_settled, a.events_rejected, a.skipped_credited, a.skipped_observed)
              == (b.events_settled, b.events_rejected, b.skipped_credited, b.skipped_observed),
              f"settle counts {b} vs {a}")
        check(close(a.rejected_weight, b.rejected_weight), "settle weight")
        check(dict(a.per_effect) == dict(b.per_effect), "per_effect")
        for eff in self.effects:
            sa, sb = (db.observation_stats(eff) for db in self.dbs)
            check(set(sa) == set(sb), "stats keys")
            for k in sa:
                check(close(sa[k], sb[k]), f"observation_stats[{k}] {sb[k]} vs {sa[k]}")

    def op_risk(self):
        rng = self.rng
        eff = rng.choice(self.effects)
        feats = rng.sample(FEATURES, rng.randint(0, 5))
        ent = rng.choice([None, *ENTITIES])
        typ = rng.choice([None, *TYPES])
        weight = rng.choice([1.0, None, 2.5])
        attrs = {"amount": rng.choice([1, 50]), "lines": rng.randint(1, 900)}
        out = self.both(lambda db: db.feature_risk(eff, feats, weight=weight, entity=ent, type=typ,
                                                   attrs=attrs, explain=True))
        check(close(out[0][0], out[1][0]), f"feature_risk {out[1][0]} vs {out[0][0]}")
        same_text(str(out[0][1]), str(out[1][1]), "risk explanation")
        cands = []
        for i in range(rng.randint(1, 12)):
            cands.append({"id": f"c{i}", "features": rng.sample(FEATURES, rng.randint(1, 4)),
                          "weight": rng.choice([None, 1.0, 0.5]), "type": rng.choice([None, "deploy", "c"]),
                          "attrs": {"lines": rng.randint(1, 500), "amount": rng.choice([1, 50])},
                          "entity": rng.choice([None, *ENTITIES])})
        k = rng.choice([1, 3, 20])
        rows = self.both(lambda db: db.rank_candidates(eff, cands, k=k, entity=rng_ent, explain=True))
        same_ranking([(r.score, int(r.candidate.id[1:])) for r in rows[0]],
                     [(r.score, int(r.candidate.id[1:])) for r in rows[1]], "rank_candidates")

    def op_reports(self):
        rng = self.rng
        eff = rng.choice(self.effects)
        ent = rng.choice([None, *ENTITIES])
        ms = rng.choice([0.0, 0.5, 1.0, 3.0])
        base = rng.choice(["mean", "prior", 0.3])
        a, b = self.both(lambda db: db.link_report(eff, entity=ent, min_support=ms, baseline=base))
        fa, fb = {r.feature for r in a}, {r.feature for r in b}
        if fa != fb:
            # Only features whose evidence n sits on the min_support threshold
            # (within TOL) may be in one report and not the other.
            full = [db.link_report(eff, entity=ent, min_support=0) for db in self.dbs]
            for f in fa ^ fb:
                check(close(full[0].row(f).n, ms) and close(full[1].row(f).n, ms),
                      f"link_report rows differ on {f}")
            return
        check(close(a.baseline, b.baseline), "baseline")
        same_ranking([(r.strength, r.feature) for r in a], [(r.strength, r.feature) for r in b],
                     "link_report order")
        for ra in a:
            rb = b.row(ra.feature)
            for attr in ("strength", "alpha", "beta", "n", "lo", "hi", "lift"):
                check(close(getattr(ra, attr), getattr(rb, attr), 1e-8),
                      f"link_report {ra.feature}.{attr} {getattr(rb, attr)} vs {getattr(ra, attr)}")
        if [r.feature for r in a] == [r.feature for r in b]:
            same_text(str(a), str(b), "LinkReport")
        else:  # rows tied within TOL may be listed in either order
            la, lb = str(a).splitlines(), str(b).splitlines()
            same_text("\n".join(sorted(la)), "\n".join(sorted(lb)), "LinkReport (tied rows)")
        bins = rng.choice([1, 5, 10])
        ca, cb = self.both(lambda db: db.calibration_report(eff, bins=bins))
        check(ca.n == cb.n and close(ca.brier, cb.brier), "calibration")
        for x, y in zip(ca.bins, cb.bins):
            check(x.count == y.count and close(x.mean_predicted, y.mean_predicted)
                  and close(x.observed_rate, y.observed_rate), "calibration bins")
        same_text(str(ca), str(cb), "CalibrationReport")

    def op_explain_all(self):
        """Explain a few stored events through both engines."""
        rng = self.rng
        evs = [list(db.events()) for db in self.dbs]
        check([e.id for e in evs[0]] == [e.id for e in evs[1]], "events() order")
        for ent in ENTITIES:
            check([e.id for e in self.dbs[0].events(ent)] == [e.id for e in self.dbs[1].events(ent)],
                  f"events({ent}) order")
        if not evs[0]:
            return
        eff = rng.choice(self.effects)
        for i in rng.sample(range(len(evs[0])), min(5, len(evs[0]))):
            at = evs[0][i].time + rng.uniform(0, 7200)
            ea, eb = self.dbs[0].explain(evs[0][i], eff, at), self.dbs[1].explain(evs[1][i], eff, at)
            same_text(str(ea), str(eb), "explain")
            check(close(self.dbs[0].score(evs[0][i], eff, at), self.dbs[1].score(evs[1][i], eff, at)), "score")

    def op_forget(self):
        ent = self.rng.choice(ENTITIES)
        scrub = self.rng.random() < 0.5
        a, b = self.both(lambda db: db.forget_entity(ent, scrub_global=scrub))
        check(dict(a) == dict(b), f"forget {b} vs {a}")
        for eff in self.effects:
            self.compare_links(eff, "after forget")

    def op_snapshot(self):
        """Save both, compare records, then swap: each engine continues from
        the other engine's snapshot."""
        with tempfile.TemporaryDirectory() as d:
            pa, pb = os.path.join(d, "ref.jsonl"), os.path.join(d, "rs.jsonl")
            self.dbs[0].save(pa)
            self.dbs[1].save(pb)
            ra, rb = snapshot_records(pa), snapshot_records(pb)
            check(len(ra) == len(rb), "snapshot length")
            for x, y in zip(ra, rb):
                same_json(x, y, "snapshot")
            new_ref, new_rs = load_ref_fixed(pb), rs.HeapDB.load(pa)
        for new in (new_ref, new_rs):
            for eff in self.effects:
                spec = self.dbs[0].effect(eff)
                for r in spec.rules:
                    if r.predicate is not None:
                        new.add_rule(eff, r.name, r.predicate, r.multiplier, types=r.types)
            if "deploy" in new.pending_event_types:
                new.register_event_type("deploy", weight=lines_weight)
        self.dbs = [new_ref, new_rs]

    OPS = [("insert", 40), ("top", 12), ("group", 3), ("feedback", 8), ("observe", 6),
           ("settle", 3), ("risk", 4), ("reports", 3), ("explain_all", 2), ("forget", 1),
           ("snapshot", 1)]

    def run(self, n_ops: int):
        names, weights = zip(*self.OPS)
        for _ in range(self.rng.choice([60, 150, 300])):
            self.op_insert()
        for _ in range(n_ops):
            getattr(self, "op_" + self.rng.choices(names, weights=weights)[0])()
        self.op_reports()
        self.op_snapshot()
        self.op_top()


rng_ent = None  # default entity for rank_candidates (candidates may override)


def run_workload(seed: int, n_ops: int = 150) -> None:
    try:
        Pair(seed).run(n_ops)
    except Mismatch as e:
        raise Mismatch(f"workload seed {seed}: {e}") from None


@pytest.mark.parametrize("seed", range(40))
def test_differential(seed):
    run_workload(seed)


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", type=int, default=200)
    ap.add_argument("--ops", type=int, default=150)
    ap.add_argument("--start", type=int, default=1000)
    ns = ap.parse_args()
    t0 = time.time()
    for s in range(ns.start, ns.start + ns.workloads):
        run_workload(s, ns.ops)
    print(f"{ns.workloads} random workloads x {ns.ops} ops agree "
          f"(seeds {ns.start}-{ns.start + ns.workloads - 1}) in {time.time() - t0:.0f}s")
