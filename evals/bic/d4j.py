"""Defects4J BIC dataset from the FONTE artifact (206 bugs), built at run time.

Source: https://github.com/coinse/fonte, branch ``extension``, pinned to
``sources.FONTE_COMMIT`` (MIT licence). Ground truth:
``data/Defects4J/BIC_dataset/combined.csv`` (67 bugs from Wen et al. 2019's
InduceBenchmark + manually curated labels by the FONTE authors).

Per bug we read:

* ``baseline/<bug>/commits/*.json``: every commit before the buggy revision
  that modified at least one ``.java`` file (message, unix time, diff).
  These are the candidates of the *report* track; they are exactly the
  commits Bug2Commit and FBL-BERT ranked.
* ``baseline/<bug>/br_short.txt``, ``br_long.txt``: the bug report.
* ``core/<bug>/failing_tests``: failing test ids, exceptions, stack traces.
* ``core/<bug>/coverage.pkl``, ``git/commits.pkl``, ``git/validation.csv``:
  coverage matrix, per-method commit history (``git log -L``) and the
  style-change check. They give the *FL* track (FONTE's inputs).
* ``baseline/<bug>/ranking_Bug2Commit.csv`` and the FBL-BERT ranking: the
  released outputs of those systems, re-scored here on the same candidates.

Authors come from the upstream GitHub repositories (treeless clones),
matched by commit id, or for Lang/Math (whose Defects4J ids differ from
upstream) by a unique commit timestamp.

The pickles are loaded with an allowlisting unpickler (only the pandas/numpy
reconstructors that the files use), never with ``pickle.load``.
"""

from __future__ import annotations

import bisect
import csv
import gzip
import io
import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import sources
from .text import parse_diff, parse_failures, token_counts

BUILD_VERSION = 3
FONTE_LAMBDA = 0.1   # FONTE's default decay (paper: lambda = 0.1)

#: sha256 of the input-file manifest (every FONTE file read, see Manifest).
EXPECTED_MANIFEST = "3da93f4680d78a6065f4062c7512623566cba4140d569a7ae465da93e5a0fe08"
#: sha256 of the derived author table ("project\tsha\tauthor" lines).
EXPECTED_AUTHORS = "98015d6dfa598950ea1bd2d8e94fe05454365e8edeacf8556ea26568e0a221a2"
ROOT = "data/Defects4J"


# -- safe unpickling ------------------------------------------------------------------

_ALLOWED = {
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.internals.managers", "BlockManager"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.core.arrays.sparse.array", "SparseArray"),
    ("pandas.core.arrays.sparse.dtype", "SparseDtype"),
    ("pandas._libs.sparse", "IntIndex"),
    ("pandas._libs.sparse", "BlockIndex"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("builtins", "slice"),
}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # pandas moved some classes between versions; follow pandas' own
            # rename table, for allowlisted names only.
            from pandas.compat import pickle_compat
            new = getattr(pickle_compat, "_class_locations_map", {}).get((module, name))
            if new is None and name == "SparseDtype":
                new = ("pandas.core.dtypes.dtypes", "SparseDtype")
            if new is None:
                raise
            return super().find_class(*new)


def safe_load(data: bytes):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SafeUnpickler(io.BytesIO(data)).load()


# -- FONTE's FL signals (re-implementation of experiment_utils.vote_for_commits) ------

def fl_signals(cov, commit_df, validation_csv: str, lam: float = FONTE_LAMBDA) -> dict:
    """Per short commit id: FONTE vote (Ochiai, tau=max, alpha=0, decay lam,
    style changes excluded with depth adjustment), max Ochiai score of the
    lines it evolved (max aggregation), number of such lines, minimum depth.
    Also returns C_susp / C_BIC / style sets as in FONTE's notebook."""
    import numpy as np

    # style-change commits: every validated file of the commit is "U"
    per_commit: dict[str, list[bool]] = defaultdict(list)
    for row in csv.reader(io.StringIO(validation_csv)):
        if len(row) >= 3:
            per_commit[row[0][:7]].append(row[2] == "U")
    style = {c for c, v in per_commit.items() if all(v)}

    result = cov["result"].to_numpy(dtype=bool)          # True = passing
    X = cov.drop(columns=["result"])
    cols = list(X.columns)
    try:
        M = X.sparse.to_coo().tocsc()
        ef = np.asarray(M[~result].sum(axis=0)).ravel()
        ep = np.asarray(M[result].sum(axis=0)).ravel()
    except AttributeError:
        A = X.to_numpy(dtype=bool)
        ef = A[~result].sum(axis=0)
        ep = A[result].sum(axis=0)
    nf = int((~result).sum())
    keep = ef > 0                                          # covered by a failing test
    ef, ep = ef[keep].astype(float), ep[keep].astype(float)
    elems = [cols[i] for i in np.flatnonzero(keep)]
    score = ef / np.sqrt(nf * (ef + ep))                   # Ochiai: ef / sqrt((ef+nf')(ef+ep)), ef+nf' = #failing
    # max rank of -score (ties -> worst rank), vote = 1 / max_rank (alpha = 0)
    srt = np.sort(-score)
    max_rank = np.searchsorted(srt, -score, side="right")
    vote_line = 1.0 / max_rank

    by_class: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for (cf, _m, _s, line), v, s in zip(elems, vote_line, score):
        by_class[cf].append((int(line), float(v), float(s)))
    for cf in by_class:
        by_class[cf].sort()

    rows = []
    for r in commit_df.itertuples(index=False):
        rows.append(dict(cf=r.class_file, mn=r.method_name, ms=r.method_signature,
                         b=int(r.begin_line), e=int(r.end_line), c=str(r.commit_hash)[:7],
                         d=int(r.depth)))
    # FONTE's depth adjustment: an excluded (style) commit no longer counts
    # as a later change of that method.
    for r in rows:
        r["nd"] = None if r["c"] in style else r["d"]
    ident = defaultdict(list)
    for r in rows:
        ident[(r["cf"], r["mn"] + r["ms"], r["b"], r["e"])].append(r)
    for group in ident.values():
        for x in group:
            if x["c"] in style:
                for y in group:
                    if y["d"] > x["d"] and y["nd"] is not None:
                        y["nd"] -= 1
    vote: dict[str, float] = defaultdict(float)
    flmax: dict[str, float] = defaultdict(float)
    nlines: dict[str, set] = defaultdict(set)
    dmin: dict[str, int] = {}
    seen: set[str] = set()
    for r in rows:
        lines = by_class.get(r["cf"], [])
        lo = bisect.bisect_left(lines, (r["b"], -1.0, -1.0))
        hi = bisect.bisect_right(lines, (r["e"], 2.0, 2.0))
        if hi <= lo:
            continue
        seen.add(r["c"])                 # FONTE's vote_df index (C_susp)
        if r["nd"] is None:
            continue
        decay = (1.0 - lam) ** r["nd"]
        for line, v, s in lines[lo:hi]:
            vote[r["c"]] += v * decay
            flmax[r["c"]] = max(flmax[r["c"]], s)
            nlines[r["c"]].add((r["cf"], line))
        dmin[r["c"]] = min(dmin.get(r["c"], 10**9), r["nd"])
    return {
        "c_susp": sorted(seen),
        "style": sorted(style & seen),
        "vote": {c: vote.get(c, 0.0) for c in seen},
        "flmax": {c: flmax.get(c, 0.0) for c in seen},
        "nlines": {c: len(nlines.get(c, ())) for c in seen},
        "dmin": {c: dmin.get(c, -1) for c in seen},
    }


# -- authors ---------------------------------------------------------------------------

def author_maps(project_shas: dict[str, dict[str, int]]) -> tuple[dict[str, str], dict]:
    """sha -> author name for every candidate commit, plus match statistics."""
    out: dict[str, str] = {}
    stats = {}
    for proj, shas in sorted(project_shas.items()):
        repo = sources.upstream_repo(proj)
        pinned = sources.UPSTREAM[proj][1]
        log = sources._git("-C", str(repo), "rev-list", "--no-commit-header",
                           "--format=%H%x1f%ct%x1f%an", pinned).decode(
            "utf-8", errors="replace").splitlines()
        by_sha, by_t = {}, defaultdict(list)
        for line in log:
            h, ct, an = line.split("\x1f", 2)
            by_sha[h] = an
            by_t[int(ct)].append(an)
        direct = by_time = 0
        for sha, ts in shas.items():
            if sha in by_sha:
                out[sha] = by_sha[sha]
                direct += 1
            elif len(by_t.get(ts, [])) == 1:
                out[sha] = by_t[ts][0]
                by_time += 1
        stats[proj] = {"commits": len(shas), "by_id": direct, "by_timestamp": by_time,
                       "unknown": len(shas) - direct - by_time}
    return out, stats


# -- build -------------------------------------------------------------------------------

def _csv_rows(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def build(out: Path, verbose: bool = True) -> dict:
    t0 = time.time()
    repo = sources.fonte_repo()
    commit = sources.FONTE_COMMIT
    man = sources.Manifest()
    tree = dict(sources.ls_tree(repo, commit, ROOT))

    def read(paths: list[str]) -> dict[str, bytes]:
        blobs = sources.read_blobs(repo, [tree[p] for p in paths])
        return {p: man.add(p, blobs[tree[p]]) for p in paths}

    gt_path = f"{ROOT}/BIC_dataset/combined.csv"
    gt = {f"{r['pid']}-{r['vid']}b": r for r in _csv_rows(read([gt_path])[gt_path])}
    pub_tree = dict(sources.ls_tree(repo, commit, "experiment_results/ranking"))
    published = {}
    for name in ("git_line_Ochiai_voting_C_BIC/max-0-0.1.csv", "Bug2Commit.csv", "FBL-BERT.csv",
                 "Bug2Commit_C_BIC.csv", "FBL-BERT_C_BIC.csv"):
        p = f"experiment_results/ranking/{name}"
        if p in pub_tree:
            data = man.add(p, sources.read_blobs(repo, [pub_tree[p]])[pub_tree[p]])
            published[name] = {f"{r['pid']}-{r['vid']}b": int(float(r["rank"])) for r in _csv_rows(data)}

    bugs = sorted({p.split("/")[3] for p in tree if p.startswith(f"{ROOT}/core/")})
    assert set(bugs) == set(gt), "core dirs and ground truth disagree"
    commit_files: dict[str, list[str]] = defaultdict(list)
    for p in tree:
        parts = p.split("/")
        if len(parts) == 6 and parts[2] == "baseline" and parts[4] == "commits":
            commit_files[parts[3]].append(p)

    # unique commit blobs (the same commit appears under many bugs)
    oid_of = {p: tree[p] for b in bugs for p in commit_files[b]}
    blobs = sources.read_blobs(repo, list(oid_of.values()))
    oid_sha = {}
    for p, oid in oid_of.items():
        man.add(p, blobs[oid])
    vocab: dict[str, int] = {}

    def enc(cnt: Counter) -> list[list[int]]:
        return [[vocab.setdefault(t, len(vocab)), n] for t, n in sorted(cnt.items())]

    commits: dict[str, dict] = {}
    for oid, raw in blobs.items():
        d = json.loads(raw)
        sha = d["sha"]
        oid_sha[oid] = sha
        if sha in commits:
            continue
        parsed = parse_diff(d["commit"])
        paths_text = " ".join(f for f, _, _ in parsed["files"])
        commits[sha] = {
            "t": int(d["timestamp"]),
            "m": d["log"].strip(),
            "f": parsed["files"],
            "tm": enc(token_counts(d["log"])),
            "td": enc(token_counts(d["commit"] + "\n" + paths_text)),
        }
    if verbose:
        print(f"[d4j] {len(commits)} unique commits read in {time.time() - t0:.0f}s", flush=True)

    out_bugs = []
    project_shas: dict[str, dict[str, int]] = defaultdict(dict)
    for k, bug in enumerate(bugs):
        base = f"{ROOT}/baseline/{bug}"
        core = f"{ROOT}/core/{bug}"
        texts = read([f"{core}/commits.log", f"{core}/failing_tests", f"{base}/br_short.txt",
                      f"{base}/br_long.txt", f"{base}/ranking_Bug2Commit.csv",
                      f"{core}/git/validation.csv"])
        fbl_p = next(p for p in tree if p.startswith(f"{base}/ranking_INDEX_FBLBERT"))
        fbl_raw = read([fbl_p])[fbl_p].decode()
        log = texts[f"{core}/commits.log"].decode().split()
        cand_shas = {oid_sha[oid_of[p]] for p in commit_files[bug]}
        cands = [s for s in log if s in cand_shas]
        assert len(cands) == len(cand_shas), f"{bug}: candidate not in commits.log"
        pid = bug.split("-")[0]
        for s in cands:
            project_shas[pid][s] = commits[s]["t"]
        short = defaultdict(list)
        for s in cands:
            short[s[:7]].append(s)
        bic7 = gt[bug]["commit"][:7]
        bic_full = [s for s in cands if s.startswith(bic7)]
        # released system outputs, keyed by full sha
        b2c = {}
        for row in csv.reader(io.StringIO(texts[f"{base}/ranking_Bug2Commit.csv"].decode())):
            for s in short.get(row[0][:7], []):
                b2c[s] = float(row[3])
        fbl = {}
        for row in csv.reader(io.StringIO(fbl_raw), delimiter="\t"):
            for s in short.get(row[2][:7], []):
                fbl[s] = float(row[5])
        # FL track
        pk = read([f"{core}/coverage.pkl", f"{core}/git/commits.pkl"])
        cov = safe_load(pk[f"{core}/coverage.pkl"])
        cdf = safe_load(pk[f"{core}/git/commits.pkl"])
        fl = fl_signals(cov, cdf, texts[f"{core}/git/validation.csv"].decode())
        del cov, cdf, pk
        unknown_short = [c for c in fl["c_susp"] if c not in short]
        out_bugs.append({
            "id": bug, "project": pid, "vid": gt[bug]["vid"], "bic7": bic7, "bic": bic_full,
            "provenance": gt[bug]["provenance"], "cands": cands, "n_log": len(log),
            "br_short": texts[f"{base}/br_short.txt"].decode("utf-8", errors="replace"),
            "br_long": texts[f"{base}/br_long.txt"].decode("utf-8", errors="replace"),
            "failing": parse_failures(texts[f"{core}/failing_tests"].decode("utf-8", errors="replace")),
            "b2c": b2c, "fbl": fbl,
            "fl": {
                "c_susp": [s for c in fl["c_susp"] for s in short.get(c, [])],
                "style": [s for c in fl["style"] for s in short.get(c, [])],
                "unknown_short": unknown_short,
                **{key: {s: fl[key][c] for c in fl["c_susp"] for s in short.get(c, [])}
                   for key in ("vote", "flmax", "nlines", "dmin")},
            },
            "published": {name: ranks.get(bug) for name, ranks in published.items()},
        })
        if verbose and (k + 1) % 20 == 0:
            print(f"[d4j] {k + 1}/{len(bugs)} bugs ({time.time() - t0:.0f}s)", flush=True)

    authors, astats = author_maps(project_shas)
    for sha, c in commits.items():
        c["a"] = authors.get(sha, "?")
    author_lines = sorted(f"{p}\t{s}\t{authors.get(s, '?')}" for p, d in project_shas.items() for s in d)
    data = {
        "meta": {
            "builder": BUILD_VERSION, "fonte_url": sources.FONTE_URL, "fonte_commit": commit,
            "manifest_sha256": man.digest, "manifest_files": len(man.files),
            "authors_sha256": sources.sha256_text(author_lines), "author_match": astats,
            "upstream": {p: {"repo": s, "commit": c} for p, (s, c) in sources.UPSTREAM.items()},
            "fonte_lambda": FONTE_LAMBDA, "build_seconds": round(time.time() - t0, 1),
        },
        "vocab": [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])],
        "commits": commits,
        "bugs": out_bugs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    if verbose:
        print(f"[d4j] built {out} in {time.time() - t0:.0f}s; manifest {man.digest}", flush=True)
    return data


def load(allow_mismatch: bool = False, rebuild: bool = False) -> dict:
    path = sources.cache_dir() / f"d4j-v{BUILD_VERSION}.json.gz"
    if rebuild or not path.exists():
        data = build(path)
    else:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    m = data["meta"]
    sources.check_pin("FONTE input manifest", m["manifest_sha256"], EXPECTED_MANIFEST, allow_mismatch)
    sources.check_pin("Defects4J author table", m["authors_sha256"], EXPECTED_AUTHORS, allow_mismatch)
    return data
