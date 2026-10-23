"""Bug-inducing-commit ranking: run every method, write results.json.

Usage::

    python -m evals.bic.run                       # both datasets (resumes from the cache)
    python -m evals.bic.run --datasets d4j        # Defects4J only
    python -m evals.bic.run --fresh               # ignore cached per-method results

Per-method results are cached under ``<cache>/runs/`` so an interrupted run
resumes where it stopped. ``python -m evals.bic.report`` renders RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import baselines as B
from . import d4j, dpdk, hdb, signals, sources
from .core import (Case, follow_the_leader, prequential, rank_of, rr, standard_bisection,
                   stream_order, weighted_bisection)

HERE = Path(__file__).parent
METHOD_VERSION = 4
DEV_FRACTION = 0.25        # first quarter of each stream = development slice


# -- running one method with caching -----------------------------------------------------------

class Runner:
    def __init__(self, tag: str, cases: list[Case], fresh: bool, bisect: bool):
        self.tag = tag
        self.stream = stream_order(cases)
        self.dir = sources.cache_dir() / "runs" / f"v{METHOD_VERSION}" / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fresh = fresh
        self.bisect = bisect
        self.methods: dict[str, dict] = {}

    def _path(self, key: str) -> Path:
        return self.dir / (key.replace("/", "_").replace(":", "_").replace(",", "_").replace("=", "-") + ".json")

    def cached(self, key: str):
        p = self._path(key)
        if not self.fresh and p.exists():
            return json.loads(p.read_text())
        return None

    def save(self, key: str, rec: dict) -> dict:
        self._path(key).write_text(json.dumps(rec))
        return rec

    def run(self, key: str, make, *, learn: bool = True, extra=None) -> dict:
        rec = self.cached(key)
        if rec is not None:
            return rec
        t0 = time.time()
        steps: list[int | None] = []

        def hook(case: Case, scores: list[float]):
            if not self.bisect:
                return
            v = case.view
            lab = [c.pos for c in v.cands if c.id in case.labels]
            if not lab:
                steps.append(None)
                return
            order = sorted(range(len(v.cands)), key=lambda i: (-scores[i], v.cands[i].pos))
            w = [0.0] * len(v.cands)
            for r, i in enumerate(order, 1):
                w[v.cands[i].pos] = 1.0 / r
            steps.append(weighted_bisection(w, min(lab)))

        m = make()
        ranks = prequential(self.stream, m, learn=learn, on_rank=hook)
        rec = {"ranks": ranks, "seconds": round(time.time() - t0, 1)}
        if self.bisect:
            rec["bisect"] = steps
        if extra is not None:
            rec.update(extra(m))
        print(f"  [{self.tag}] {key}: MRR {np.mean([rr(r) for r in ranks]):.3f} "
              f"({rec['seconds']}s)", flush=True)
        return self.save(key, rec)

    def add(self, name: str, rec: dict, description: str, group: str, inputs: str, **kw) -> None:
        self.methods[name] = {"description": description, "group": group, "inputs": inputs,
                              "ranks": rec["ranks"], **{k: v for k, v in rec.items() if k != "ranks"}, **kw}

    def ftl(self, name: str, recs: dict[str, dict], description: str, group: str, inputs: str) -> None:
        per = {k: [rr(x) for x in r["ranks"]] for k, r in recs.items()}
        _, chosen = follow_the_leader(per)
        ranks = [recs[c]["ranks"][i] for i, c in enumerate(chosen)]
        extra = {"chosen": chosen, "grid": list(recs)}
        if self.bisect:
            extra["bisect"] = [recs[c]["bisect"][i] for i, c in enumerate(chosen)]
        self.methods[name] = {"description": description, "group": group, "inputs": inputs,
                              "ranks": ranks, **extra}


# -- method suites -------------------------------------------------------------------------------

def heapdb_suite(R: Runner, inputs: str, ablations: bool = True) -> None:
    def links(m):
        rows = sorted(m.links(min_support=20.0), key=lambda r: -r[1])
        return {"links_top": rows[:15], "links_bottom": rows[-8:]}

    R.add("heapdb", R.run("heapdb-nolearn", lambda: hdb.HeapDBRanker(
        hdb.HDBConfig(kernel="exp1y", dose=True, learn=False)), learn=False),
        "HeapDB, no feedback: exp. lag kernel (half-life 1y) x dose (diff size) x prior links (all equal)",
        "heapdb", inputs)
    recs = {}
    for cfg in hdb.GRID:
        recs[cfg.name] = R.run(f"heapdb-{cfg.name}", lambda cfg=cfg: hdb.HeapDBRanker(cfg),
                               extra=links if cfg == hdb.DEFAULT else None)
    d = recs[hdb.DEFAULT.name]
    R.add("heapdb+learned", d, "HeapDB with prequential feedback (confirm the BIC, reject the "
          f"other candidates); default config {hdb.DEFAULT.name}", "heapdb", inputs,
          **{k: d[k] for k in ("links_top", "links_bottom") if k in d})
    R.ftl("heapdb+learned+tuned", recs, "as heapdb+learned; kernel, dose and prior picked per bug by "
          "follow-the-leader over the fixed grid (best MRR on the bugs already seen)", "heapdb", inputs)
    if ablations:
        def only(prefixes):
            return lambda c: tuple(x for x in c.cat if x.startswith(prefixes))
        R.add("abl:signal-buckets-only", R.run("heapdb-abl-signals", lambda: hdb.HeapDBRanker(
            hdb.DEFAULT, features=only(("b:",)))),
            "heapdb+learned with only the bucketed signals (no author/dir/keyword/component features)",
            "ablation", inputs)
        R.add("abl:identity-only", R.run("heapdb-abl-identity", lambda: hdb.HeapDBRanker(
            hdb.DEFAULT, features=only(("author:", "dir:", "kw:", "comp:")))),
            "heapdb+learned with only author/dir/keyword/component features (no signal buckets)",
            "ablation", inputs)


def ltr_suite(R: Runner, inputs: str, retrain_every: int, neg_per_bug, lm_every: int) -> None:
    recs = {}
    for C in (1.0, 0.1):
        recs[f"C={C:g}"] = R.run(f"logreg-C{C:g}", lambda C=C: B.LTR("logreg", "logreg", C=C,
                                 retrain_every=retrain_every, neg_per_bug=neg_per_bug))
    R.add("logreg", recs["C=1"], f"logistic regression (L2, C=1) over every numeric signal + one-hot "
          f"features, retrained on all bugs seen so far every {retrain_every} bug(s)", "learned", inputs)
    R.ftl("logreg+tuned", recs, "logistic regression, C in {1, 0.1} picked by follow-the-leader",
          "learned", inputs)
    R.add("lambdamart", R.run("lambdamart", lambda: B.LTR("lambdamart", "lambdamart",
                                                          retrain_every=lm_every, neg_per_bug=neg_per_bug)),
          f"LightGBM LambdaMART (300 trees, 15 leaves) on the same features, retrained every "
          f"{lm_every} bugs", "learned", inputs)


def linear_suite(R: Runner, sig: list[tuple[str, float]], inputs: str) -> None:
    rec = R.cached("linear-grid")
    if rec is None:
        t0 = time.time()
        g = B.LinearGrid(sig)
        M = g.rr_matrix(R.stream)
        names = g.names()
        rec = R.save("linear-grid", {"names": names, "rr": M.tolist(), "seconds": round(time.time() - t0, 1)})
    per = {n: [row[j] for row in rec["rr"]] for j, n in enumerate(rec["names"])}
    chosen_rr, chosen = follow_the_leader(per)
    ranks = [None if x == 0 else int(round(1 / x)) for x in chosen_rr]
    R.methods["linear+tuned"] = {
        "description": "weighted sum of within-bug z-scored signals (" + ", ".join(n for n, _ in sig) +
                       "), weights in {0, .5, 1} per signal picked by follow-the-leader",
        "group": "learned", "inputs": inputs, "ranks": ranks, "chosen": chosen,
        "grid_size": len(rec["names"])}


def transfer(R: Runner, make, key: str) -> list[int | None]:
    """Leave one project out: learn on every bug of the other projects, rank
    the held-out project's bugs with no feedback from that project."""
    rec = R.cached(key)
    if rec is not None:
        return rec["ranks"]
    t0 = time.time()
    by_id = {}
    projects = sorted({c.view.project for c in R.stream})
    for p in projects:
        m = make()
        m.reset()
        for c in R.stream:
            if c.view.project != p:
                m.learn(c.view, c.labels)
        if hasattr(m, "_fit"):
            m._fit()
        for c in R.stream:
            if c.view.project == p:
                by_id[c.id] = rank_of(c.view, m.rank(c.view), c.labels)
    ranks = [by_id[c.id] for c in R.stream]
    R.save(key, {"ranks": ranks, "seconds": round(time.time() - t0, 1)})
    print(f"  [{R.tag}] {key}: MRR {np.mean([rr(r) for r in ranks]):.3f}", flush=True)
    return ranks


# -- tracks ----------------------------------------------------------------------------------------

R_INPUTS = "report text, failing tests (names, exceptions, stack traces), commit time/order, message, diff, files, size, author"
F_INPUTS = R_INPUTS + " + FL: Ochiai coverage scores, method history depth (FONTE's inputs); candidates = FONTE's C_BIC"


def d4j_track(data: dict, track: str, fresh: bool) -> dict:
    cases = signals.d4j_cases(data, track=track)
    tag = f"d4j-{track}"
    R = Runner(tag, cases, fresh, bisect=True)
    meta_of = {c.id: c.meta for c in cases}
    inputs = R_INPUTS if track == "report" else F_INPUTS
    print(f"[{tag}] {len(cases)} bugs, {sum(len(c.view.cands) for c in cases)} candidates", flush=True)

    def fixed(name, fn, desc, inp, group="baseline"):
        R.add(name, R.run(name, lambda: B.Fixed(name, fn), learn=False), desc, group, inp)

    fixed("recency", B.recency, "most recent commit first", "commit order")
    fixed("file-overlap", lambda v: B.col(v, "trace_src") + B.col(v, "trace_test") + B.col(v, "report_files"),
          "number of touched files named in the stack trace or the report (ties: newest first)",
          "stack trace, report, files")
    fixed("bm25-diff", lambda v: B.col(v, "bm25_diff"), "BM25(report + failure text, commit diff + paths)",
          "report, failure text, diff")
    fixed("bm25-msg", lambda v: B.col(v, "bm25_msg"), "BM25(report + failure text, commit message)",
          "report, failure text, message")
    if track == "fl":
        fixed("fonte", lambda v: B.col(v, "fonte_log"), "FONTE (re-implemented): Ochiai rank votes, "
              "depth decay lambda=0.1, style changes removed (no report text)", "FL scores, method history",
              group="published")
        fixed("fl-max-aggregation", lambda v: B.col(v, "flmax"), "max Ochiai score of the lines a commit "
              "evolved (FONTE's max-aggregation baseline)", "FL scores, method history", group="published")
    fixed("bug2commit (released)", B.published(meta_of, "b2c"), "Bug2Commit's released per-bug ranking "
          "(Murali et al. 2021, as run by the FONTE authors), re-scored on these candidates",
          "report, failing-test symptoms, commit message/files (unsupervised)", group="published")
    fixed("fbl-bert (released)", B.published(meta_of, "fbl"), "FBL-BERT's released per-bug ranking "
          "(Ciborowska & Damevski 2022), re-scored on these candidates",
          "report, commit diffs; model fine-tuned on external JDT data", group="published")

    lin = [("pos_log", -1.0), ("bm25_diff", 1.0), ("bm25_msg", 1.0), ("trace_src", 1.0),
           ("trace_meth", 1.0), ("report_files", 1.0), ("lines_log", 1.0)]
    if track == "fl":
        lin.append(("fonte_log", 1.0))
    linear_suite(R, lin, inputs)
    ltr_suite(R, inputs, retrain_every=1, neg_per_bug=None, lm_every=5)
    heapdb_suite(R, inputs, ablations=(track == "report"))
    if track == "fl":
        # Apples-to-apples with FONTE: HeapDB restricted to FONTE's own inputs
        # (FL vote rank, max Ochiai, method-history depth), learning from feedback.
        R.add("abl:fl-inputs-only", R.run("heapdb-abl-flonly", lambda: hdb.HeapDBRanker(
            hdb.DEFAULT, features=lambda c: tuple(x for x in c.cat
                                                  if x.startswith(("b:fonte=", "b:flmax=", "b:depth="))))),
            "heapdb+learned (default config) with only FONTE's inputs: FL vote rank, max Ochiai score and "
            "method-history depth buckets (no report, no text, no time)", "ablation",
            "FL scores, method history (FONTE's inputs) + feedback on earlier bugs")

    # leave-one-project-out transfer
    R.methods["heapdb+transfer"] = {
        "description": "heapdb+learned (default config) trained only on the other projects' bugs "
                       "(leave one project out), no feedback from the held-out project",
        "group": "transfer", "inputs": inputs,
        "ranks": transfer(R, lambda: hdb.HeapDBRanker(hdb.DEFAULT), "transfer-heapdb")}
    R.methods["logreg+transfer"] = {
        "description": "logistic regression (C=1) trained only on the other projects' bugs",
        "group": "transfer", "inputs": inputs,
        "ranks": transfer(R, lambda: B.LTR("lr-t", "logreg", C=1.0, retrain_every=10**9), "transfer-logreg")}

    # reproduction checks against the released aggregate rankings (max tie rule)
    repro = {}
    if track == "report":
        pub = "Bug2Commit.csv"
        rows = []
        for c in R.stream:
            p = c.meta["published"].get(pub)
            if p is None:
                continue
            sc = B.published(meta_of, "b2c")(c.view)
            rows.append((c.id, rank_of(c.view, sc, c.labels, tie="max"), p))
        repro["bug2commit_vs_released_C"] = {"n": len(rows), "agree": sum(a == b for _, a, b in rows),
                                             "disagree": [r for r in rows if r[1] != r[2]][:20]}
    else:
        pub = "git_line_Ochiai_voting_C_BIC/max-0-0.1.csv"
        rows = []
        for c in R.stream:
            p = c.meta["published"].get(pub)
            if p is None:
                continue
            sc = list(B.col(c.view, "fonte_log"))
            rows.append((c.id, rank_of(c.view, sc, c.labels, tie="max"), p))
        repro["fonte_vs_released_icse23"] = {"n": len(rows), "agree": sum(a == b for _, a, b in rows),
                                             "disagree": [r for r in rows if r[1] != r[2]][:20]}
        repro["fonte_released_icse23_ranks"] = {c.id: c.meta["published"].get(pub) for c in R.stream}
    repro["released_ranks"] = {name: [c.meta["published"].get(name) for c in R.stream]
                               for name in sorted({k for c in R.stream for k in c.meta["published"]})}
    std = []
    for c in R.stream:
        lab = [x.pos for x in c.view.cands if x.id in c.labels]
        std.append(standard_bisection(len(c.view.cands), min(lab)) if lab else None)
    stats = {
        "candidates_java_median": float(np.median([c.meta["n_java"] for c in R.stream])),
        "commits_log_median": float(np.median([c.meta["n_log"] for c in R.stream])),
        "reports_mentioning_bic_id": sum(
            1 for b in data["bugs"] if b["bic"] and any(
                x in (b["br_short"] + b["br_long"]) for x in (b["bic7"], b["bic"][0][:10]))),
        "bugs_total": len(data["bugs"]),
        "bugs_bic_outside_track": sum(1 for c in R.stream if not c.in_scope),
        "bugs_without_candidates": len(data["bugs"]) - len(R.stream),
    }
    return finish(R, cases, extra={"repro": repro, "bisect_standard": std, "stats": stats})


def dpdk_track(data: dict, fresh: bool) -> dict:
    cases = signals.dpdk_cases(data)
    R = Runner("dpdk", cases, fresh, bisect=False)
    inputs = ("fix message (trailers and hex ids removed), fix author, files the fix touches; commit "
              "time/order, message, files, size, author, subsystem prefix")
    print(f"[dpdk] {len(cases)} bugs, {sum(len(c.view.cands) for c in cases)} candidates", flush=True)

    def fixed(name, fn, desc, inp):
        R.add(name, R.run(name, lambda: B.Fixed(name, fn), learn=False), desc, "baseline", inp)

    fixed("recency", B.recency, "most recent commit first (within the file scope)", "commit order")
    fixed("file-overlap", lambda v: B.col(v, "jaccard"), "Jaccard overlap of touched files with the fix "
          "(ties: newest first)", "files")
    fixed("bm25-msg", lambda v: B.col(v, "bm25_msg"), "BM25(fix message, commit message + paths)",
          "messages, paths")
    fixed("same-author", lambda v: B.col(v, "same_author"), "commits by the fix's author first, then newest",
          "authors")
    lin = [("pos_log", -1.0), ("bm25_msg", 1.0), ("jaccard", 1.0), ("shared_dirs", 1.0),
           ("same_author", 1.0), ("lines_log", 1.0), ("cand_is_fix", 1.0)]
    linear_suite(R, lin, inputs)
    ltr_suite(R, inputs, retrain_every=500, neg_per_bug=30, lm_every=1000)
    heapdb_suite(R, inputs, ablations=True)
    return finish(R, cases)


def finish(R: Runner, cases: list[Case], extra: dict | None = None) -> dict:
    n = len(R.stream)
    out = {
        "bugs": {
            "ids": [c.id for c in R.stream],
            "projects": [c.view.project for c in R.stream],
            "times": [c.time for c in R.stream],
            "n_cands": [len(c.view.cands) for c in R.stream],
            "in_scope": [c.in_scope for c in R.stream],
            "provenance": [c.meta.get("provenance") for c in R.stream],
            "lag_days": [min(((c.time - x.time) / 86400.0 for x in c.view.cands if x.id in c.labels),
                             default=None) for c in R.stream],
            "dev_n": int(round(DEV_FRACTION * n)),
        },
        "methods": R.methods,
    }
    if extra:
        out.update(extra)
    return out


def env() -> dict:
    import heapdb
    import lightgbm
    import sklearn
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                             text=True).stdout.strip()
    except OSError:
        sha = ""
    return {"python": platform.python_version(), "heapdb": heapdb.__version__,
            "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__, "cpu": cpu, "cpus": os.cpu_count(),
            "platform": platform.platform(), "git": sha}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="d4j,dpdk")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--fresh", action="store_true", help="ignore cached per-method results")
    ap.add_argument("--allow-mismatch", action="store_true", help="continue if an input sha256 differs")
    a = ap.parse_args(argv)
    t0 = time.time()
    res = {"meta": {"env": env(), "method_version": METHOD_VERSION, "dev_fraction": DEV_FRACTION,
                    "hdb_grid": [asdict(c) | {"name": c.name} for c in hdb.GRID],
                    "hdb_default": hdb.DEFAULT.name,
                    "command": "python -m evals.bic.run " + " ".join(argv if argv is not None else sys.argv[1:])},
           "datasets": {}}
    if os.path.exists(a.out):
        try:
            old = json.loads(Path(a.out).read_text())
            res["datasets"] = old.get("datasets", {})
        except (OSError, ValueError):
            pass
    for ds in a.datasets.split(","):
        if ds == "d4j":
            data = d4j.load(allow_mismatch=a.allow_mismatch)
            meta = dict(data["meta"])
            tracks = {t: d4j_track(data, t, a.fresh) for t in ("report", "fl")}
            res["datasets"]["d4j"] = {"meta": meta, "tracks": tracks}
            del data
        elif ds == "dpdk":
            data = dpdk.load(allow_mismatch=a.allow_mismatch)
            meta = dict(data["meta"])
            res["datasets"]["dpdk"] = {"meta": meta, "tracks": {"fixes": dpdk_track(data, a.fresh)}}
            del data
        else:
            raise SystemExit(f"unknown dataset {ds}")
        res["meta"]["runtime_seconds"] = round(time.time() - t0, 1)
        Path(a.out).write_text(json.dumps(res, separators=(",", ":"), default=str))
        print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
