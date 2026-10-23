"""One signal table per (bug, candidate), shared by every method.

Each candidate gets numeric signals (``Cand.num``) and binary features
(``Cand.cat``). The binary features include fixed-edge buckets of the
numeric signals (``b:<signal>=<bucket>``) plus categorical facts (author,
directory, message keywords, subsystem). The bucket edges are fixed here,
in advance, and are the same for every method: learned baselines get
``num`` and ``cat``; HeapDB gets ``cat`` (its events are sets of features)
and, for the dose, the diff size. No method sees test results unless it is
in the ``fl`` track, where every method sees the same FL signals.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter

import numpy as np
import scipy.sparse as sp

from .core import Cand, Case, CaseView
from .text import class_to_path_suffix, failure_query_text, tokens

KEYWORDS = ("fix", "bug", "refactor", "javadoc", "doc", "test", "format", "checkstyle", "style",
            "typo", "whitespace", "import", "clean", "remove", "add", "new", "merge", "revert",
            "deprecat", "perf", "optimi", "rename", "move", "generic", "warning", "simplif",
            "support", "implement", "improve", "update")

POS_EDGES = ((1, "1"), (3, "2-3"), (10, "4-10"), (30, "11-30"), (100, "31-100"), (300, "101-300"))
LAG_EDGES = ((1, "<1d"), (7, "<7d"), (30, "<30d"), (90, "<90d"), (365, "<1y"), (1095, "<3y"))
RANK_EDGES = ((1, "1"), (2, "2"), (3, "3"), (5, "4-5"), (10, "6-10"), (20, "11-20"), (50, "21-50"),
              (100, "51-100"))
LINES_EDGES = ((5, "<=5"), (20, "<=20"), (100, "<=100"), (500, "<=500"))
FILES_EDGES = ((1, "1"), (3, "2-3"), (10, "4-10"))
COUNT_EDGES = ((0, "0"), (1, "1"), (2, "2"))
JACC_EDGES = ((0.0, "0"), (0.1, "<0.1"), (0.3, "<0.3"), (0.6, "<0.6"))
FL_EDGES = ((0.0, "0"), (0.2, "<0.2"), (0.5, "<0.5"), (0.8, "<0.8"))
DEPTH_EDGES = ((0, "0"), (1, "1"), (4, "2-4"))


def bucket(x: float, edges, last: str | None = None, strict: bool = False) -> str:
    """Label of the first edge with ``x <= edge`` (``x < edge`` if
    ``strict``), else ``last`` (default: ``>`` + the last edge's label, e.g.
    ``>101-300`` = beyond the 101-300 bucket, ``><=500`` = more than 500
    lines). The labels are only names; changing them would reorder the
    one-hot columns of the learned baselines, so they are kept as run."""
    for e, lab in edges:
        if (x < e) if strict else (x <= e):
            return lab
    return last if last is not None else f">{edges[-1][1]}"


_INTERN: dict[str, str] = {}


def _i(s: str) -> str:
    return _INTERN.setdefault(s, sys.intern(s))


# -- BM25 ---------------------------------------------------------------------------

class BM25Index:
    """Term-count matrix (docs x vocab); BM25 is computed per bug over that
    bug's candidates only (IDF and average length from the candidate set)."""

    def __init__(self, docs: list[list[list[int]]], n_vocab: int, k1: float = 1.2, b: float = 0.75):
        rows, cols, vals = [], [], []
        for i, d in enumerate(docs):
            for t, n in d:
                rows.append(i)
                cols.append(t)
                vals.append(n)
        self.X = sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                               shape=(len(docs), n_vocab))
        self.dl = np.asarray(self.X.sum(axis=1)).ravel()
        self.k1, self.b = k1, b

    def scores(self, rows: list[int], query: Counter) -> np.ndarray:
        if not rows or not query:
            return np.zeros(len(rows))
        q = [t for t in query if 0 <= t < self.X.shape[1]]
        if not q:
            return np.zeros(len(rows))
        sub = self.X[rows]
        n = len(rows)
        dl = self.dl[rows]
        avg = max(dl.mean(), 1e-9)
        sq = sub[:, q].tocsc()
        df = np.diff(sq.indptr)
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        qtf = np.asarray([query[t] for t in q], dtype=np.float64)
        sq = sq.tocoo()
        tf = sq.data.astype(np.float64)
        denom = tf + self.k1 * (1 - self.b + self.b * dl[sq.row] / avg)
        contrib = idf[sq.col] * qtf[sq.col] * tf * (self.k1 + 1) / denom
        out = np.zeros(n)
        np.add.at(out, sq.row, contrib)
        return out


def _ranks_desc(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """1-based rank by x descending, ties by recency position."""
    o = np.lexsort((pos, -x))
    r = np.empty(len(x), dtype=np.int64)
    r[o] = np.arange(1, len(x) + 1)
    return r


def _common_cats(proj: str, author: str, files: list, msg: str, ) -> list[str]:
    cats = [f"author:{proj}/{author}"]
    by_dir: Counter = Counter()
    for p, a, d in files:
        by_dir[p.rsplit("/", 1)[0] if "/" in p else "."] += a + d + 1
    for dname, _ in by_dir.most_common(3):
        cats.append(f"dir:{proj}/{dname}")
    lm = msg.lower()
    cats += [f"kw:{k}" for k in KEYWORDS if k in lm]
    return cats


def _cases_from(rows_per_case, names) -> list[Case]:
    return rows_per_case


# -- Defects4J -------------------------------------------------------------------------

D4J_NUM = ("pos_log", "lag_log", "bm25_msg", "bm25_diff", "bm25_msg_rank_log",
           "bm25_diff_rank_log", "trace_src", "trace_test", "trace_meth", "report_files",
           "lines_log", "files_n", "msg_len_log", "lines")
D4J_FL_NUM = ("fonte_log", "fonte_rank_log", "flmax", "fl_lines_log", "depth")


def d4j_cases(data: dict, track: str = "report") -> list[Case]:
    """``track="report"``: candidates = every java-touching commit before the
    buggy revision. ``track="fl"``: candidates = FONTE's C_BIC (commits that
    changed code covered by a failing test, minus style-only changes), and
    every method also gets FONTE's FL signals."""
    assert track in ("report", "fl")
    commits = data["commits"]
    vocab = {t: i for i, t in enumerate(data["vocab"])}
    shas = list(commits)
    row_of = {s: i for i, s in enumerate(shas)}
    idx_msg = BM25Index([commits[s]["tm"] for s in shas], len(vocab))
    idx_diff = BM25Index([commits[s]["td"] for s in shas], len(vocab))
    diff_tokens = {}
    names = D4J_NUM + (D4J_FL_NUM if track == "fl" else ())
    cases = []
    for bug in data["bugs"]:
        proj = bug["project"]
        cands = bug["cands"]
        if track == "fl":
            keep = set(bug["fl"]["c_susp"]) - set(bug["fl"]["style"])
            cands = [s for s in cands if s in keep]
        if not cands:
            continue
        t_fail = max(commits[s]["t"] for s in bug["cands"]) + 1
        fail = bug["failing"]
        report = bug["br_short"] + "\n" + bug["br_long"]
        q = Counter(vocab[t] for t in tokens(report + "\n" + failure_query_text(fail)) if t in vocab)
        rows = [row_of[s] for s in cands]
        pos = np.arange(len(cands))
        s_msg = idx_msg.scores(rows, q)
        s_diff = idx_diff.scores(rows, q)
        r_msg = _ranks_desc(s_msg, pos)
        r_diff = _ranks_desc(s_diff, pos)
        trace_suffix = {class_to_path_suffix(c) for c, _ in fail["frames"]}
        trace_meth = {m.lower() for _, m in fail["frames"] if not m.startswith("<")}
        trace_meth_ids = {vocab[m] for m in trace_meth if m in vocab}
        report_words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", report))
        if track == "fl":
            fl = bug["fl"]
            vote = np.asarray([fl["vote"].get(s, 0.0) for s in cands])
            r_fonte = _ranks_desc(vote, pos)
        X = np.zeros((len(cands), len(names)), dtype=np.float64)
        cat_rows = []
        for i, s in enumerate(cands):
            c = commits[s]
            files = c["f"]
            lines = sum(a + d for _, a, d in files)
            lag = (t_fail - c["t"]) / 86400.0
            tsrc = sum(1 for p, _, _ in files if any(p.endswith(x) for x in trace_suffix)
                       and "/test/" not in p and not p.endswith("Test.java"))
            ttest = sum(1 for p, _, _ in files if any(p.endswith(x) for x in trace_suffix)) - tsrc
            if s not in diff_tokens:
                diff_tokens[s] = {t for t, _ in c["td"]}
            tmeth = len(trace_meth_ids & diff_tokens[s])
            rfiles = sum(1 for p, _, _ in files if p.rsplit("/", 1)[-1][:-5] in report_words)
            mlen = sum(n for _, n in c["tm"])
            vals = [math.log1p(i), math.log1p(lag), s_msg[i], s_diff[i], math.log(r_msg[i]),
                    math.log(r_diff[i]), tsrc, ttest, tmeth, rfiles, math.log1p(lines), len(files),
                    math.log1p(mlen), lines]
            cats = _common_cats(proj, c.get("a", "?"), files, c["m"])
            cats += [
                f"b:pos={bucket(i + 1, POS_EDGES)}", f"b:lag={bucket(lag, LAG_EDGES, '>=3y', strict=True)}",
                f"b:bm25_msg={bucket(int(r_msg[i]), RANK_EDGES, '>100') if s_msg[i] > 0 else 'none'}",
                f"b:bm25_diff={bucket(int(r_diff[i]), RANK_EDGES, '>100') if s_diff[i] > 0 else 'none'}",
                f"b:trace_src={bucket(tsrc, COUNT_EDGES, '3+')}",
                f"b:trace_meth={bucket(tmeth, COUNT_EDGES, '3+')}",
                f"b:report_files={bucket(rfiles, COUNT_EDGES, '3+')}",
                f"b:lines={bucket(lines, LINES_EDGES)}", f"b:files={bucket(len(files), FILES_EDGES)}",
            ]
            if track == "fl":
                v = fl["vote"].get(s, 0.0)
                d = fl["dmin"].get(s, -1)
                vals += [math.log1p(1000 * v), math.log(r_fonte[i]), fl["flmax"].get(s, 0.0),
                         math.log1p(fl["nlines"].get(s, 0)), d]
                cats += [f"b:fonte={bucket(int(r_fonte[i]), RANK_EDGES, '>100')}",
                         f"b:flmax={bucket(fl['flmax'].get(s, 0.0), FL_EDGES, '>=0.8', strict=True)}"
                         if fl["flmax"].get(s, 0.0) > 0 else "b:flmax=0",
                         f"b:depth={bucket(d, DEPTH_EDGES, '5+')}"]
            X[i] = vals
            cat_rows.append(tuple(_i(x) for x in cats))
        cand_objs = tuple(Cand(id=s, time=float(commits[s]["t"]), pos=i, num=X[i], cat=cat_rows[i])
                          for i, s in enumerate(cands))
        view = CaseView(id=bug["id"], project=proj, time=float(t_fail), cands=cand_objs,
                        num_names=names)
        meta = {"X": X, "provenance": bug["provenance"], "b2c": bug["b2c"], "fbl": bug["fbl"],
                "published": bug["published"], "n_log": bug["n_log"], "n_java": len(bug["cands"])}
        cases.append(Case(view=view, labels=frozenset(bug["bic"]), meta=meta))
    return cases


# -- DPDK -------------------------------------------------------------------------------

DPDK_NUM = ("pos_log", "lag_log", "bm25_msg", "bm25_msg_rank_log", "shared_files", "jaccard",
            "shared_dirs", "lines_log", "files_n", "same_author", "cand_is_fix", "msg_len_log",
            "lines")


def dpdk_cases(data: dict) -> list[Case]:
    commits = data["commits"]
    vocab = {t: i for i, t in enumerate(data["vocab"])}
    nv = len(vocab)
    # candidate document = message (trailers removed) + file path tokens
    path_tok = {}
    docs = []
    for c in commits:
        cnt = Counter({t: n for t, n in c["tm"]})
        for p, _, _ in c["f"]:
            if p not in path_tok:
                path_tok[p] = [vocab.setdefault(t, len(vocab)) for t in tokens(p.replace("/", " "))]
            cnt.update(path_tok[p])
        docs.append(sorted(cnt.items()))
    idx = BM25Index([[list(x) for x in d] for d in docs], len(vocab))
    del docs
    names = DPDK_NUM
    cases = []
    for bug in data["bugs"]:
        fx = commits[bug["fix"]]
        cands = bug["cands"]
        if not cands:
            continue
        t_fail = float(fx["t"])
        q = Counter({t: n for t, n in fx["tm"]})
        pos = np.arange(len(cands))
        s_msg = idx.scores(cands, q)
        r_msg = _ranks_desc(s_msg, pos)
        ffiles = {p for p, _, _ in fx["f"]}
        fdirs = {p.rsplit("/", 1)[0] for p in ffiles}
        X = np.zeros((len(cands), len(names)), dtype=np.float64)
        cat_rows = []
        for i, j in enumerate(cands):
            c = commits[j]
            files = c["f"]
            cf = {p for p, _, _ in files}
            shared = len(cf & ffiles)
            jac = shared / max(1, len(cf | ffiles))
            sdirs = len({p.rsplit("/", 1)[0] for p in cf} & fdirs)
            lines = sum(a + d for _, a, d in files)
            lag = (t_fail - c["t"]) / 86400.0
            same = 1.0 if c["a"] == fx["a"] else 0.0
            isfix = 1.0 if c["fix"] else 0.0
            mlen = sum(n for _, n in c["tm"])
            X[i] = [math.log1p(i), math.log1p(lag), s_msg[i], math.log(r_msg[i]), shared, jac, sdirs,
                    math.log1p(lines), len(files), same, isfix, math.log1p(mlen), lines]
            cats = _common_cats("dpdk", c["a"], files, c["subj"])
            cats.append(f"comp:{c['comp']}")
            cats += [
                f"b:pos={bucket(i + 1, POS_EDGES)}", f"b:lag={bucket(lag, LAG_EDGES, '>=3y', strict=True)}",
                f"b:bm25_msg={bucket(int(r_msg[i]), RANK_EDGES, '>100') if s_msg[i] > 0 else 'none'}",
                f"b:shared_files={bucket(shared, COUNT_EDGES, '3+')}",
                f"b:jaccard={bucket(jac, JACC_EDGES, '>=0.6', strict=True) if jac > 0 else '0'}",
                f"b:shared_dirs={bucket(sdirs, COUNT_EDGES, '3+')}",
                f"b:lines={bucket(lines, LINES_EDGES)}", f"b:files={bucket(len(files), FILES_EDGES)}",
            ]
            if same:
                cats.append("b:same_author=1")
            if isfix:
                cats.append("b:cand_is_fix=1")
            X[i] = X[i]
            cat_rows.append(tuple(_i(x) for x in cats))
        cand_objs = tuple(Cand(id=commits[j]["sha"], time=float(commits[j]["t"]), pos=i, num=X[i],
                               cat=cat_rows[i]) for i, j in enumerate(cands))
        view = CaseView(id=bug["id"], project="dpdk", time=t_fail, cands=cand_objs, num_names=names)
        labels = frozenset(commits[j]["sha"] for j in bug["bics"])
        cases.append(Case(view=view, labels=labels, meta={"X": X}))
    return cases
