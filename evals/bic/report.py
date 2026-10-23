"""Render evals/bic/results.json into evals/bic/RESULTS.md.

Every number in RESULTS.md is computed here from the per-bug ranks in
results.json; the prose that states a win, tie or loss is chosen by the
paired bootstrap CI, not written by hand. Paired comparisons: two-sided
paired bootstrap over bugs on the reciprocal rank (10 000 resamples, seed 7).

Usage::

    python -m evals.bic.report [results.json] [-o RESULTS.md]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from .core import KS, paired_bootstrap, rr, summarize

HERE = Path(__file__).parent

#: Numbers as printed in the papers (not recomputed here), for reference.
PUBLISHED = [
    # (row, MRR, Acc@1, Acc@5, setting)
    ("FONTE with SBFL", 0.481, 0.320, 0.689),
    ("Bug2Commit (report + symptom)", 0.330, 0.170, 0.573),
    ("Max aggregation of SBFL scores", 0.288, 0.121, 0.485),
    ("FBL-BERT", 0.246, 0.141, 0.383),
]
PUBLISHED_SRC = ("Table VII of the FONTE extension (An et al., *Identifying Bug Inducing Commits by "
                 "Combining Fault Localisation and Code Change Histories*, arXiv:2502.12922): 206 "
                 "Defects4J bugs, candidates = C_BIC, max tie-breaking")

TRACK_TITLES = {
    ("d4j", "report"): "Defects4J, report track (206 bugs; candidates = every java-touching commit before the buggy revision)",
    ("d4j", "fl"): "Defects4J, FL track (candidates = FONTE's C_BIC; every method also gets FONTE's FL signals)",
    ("dpdk", "fixes"): "DPDK `Fixes:` trailers (fix -> BIC; candidates = earlier commits sharing a file with the fix, newest 300)",
}
HEADLINE_HDB = "heapdb+learned+tuned"
LOGISTIC = "heapdb+logistic+tuned"
LOGISTIC_DEFAULT = "heapdb+logistic"
EQUAL_GROUPS = ("baseline", "learned")
WORD = {"win": "ahead", "tie": "statistically tied", "loss": "behind"}


def rrs(m: dict) -> list[float]:
    return [rr(x) for x in m["ranks"]]


def fmt(x: float, d: int = 3) -> str:
    return "-" if x != x else f"{x:.{d}f}"


def strongest(methods: dict) -> str:
    """Equal-input baseline with the best MRR over the whole stream."""
    cands = [n for n, m in methods.items() if m["group"] in EQUAL_GROUPS]
    return max(cands, key=lambda n: statistics.fmean(rrs(methods[n])))


def verdict(lo: float, hi: float) -> str:
    return "win" if lo > 0 else ("loss" if hi < 0 else "tie")


def cmp_row(a: dict, b: dict, sl=slice(None)) -> tuple[float, float, float]:
    return paired_bootstrap(rrs(a)[sl], rrs(b)[sl])


def table(methods: dict, names: list[str], ref: str | None, sl=slice(None)) -> list[str]:
    L = ["| method | group | Acc@1 | Acc@3 | Acc@5 | Acc@10 | MRR | mean rank | " +
         (f"ΔMRR vs `{ref}` [95% CI] |" if ref else "|"),
         "|---|---|---|---|---|---|---|---|" + ("---|" if ref else "")]
    for n in names:
        m = methods[n]
        s = summarize(m["ranks"][sl])
        row = (f"| `{n}` | {m['group']} | " + " | ".join(fmt(s[f'Acc@{k}']) for k in KS) +
               f" | {fmt(s['MRR'])} | {fmt(s['mean_rank'], 1)} |")
        if ref:
            if n == ref:
                row += " (reference) |"
            else:
                d, lo, hi = cmp_row(m, methods[ref], sl)
                row += f" {d:+.3f} [{lo:+.3f}, {hi:+.3f}] |"
        L.append(row)
    return L


def ordered(methods: dict) -> list[str]:
    order = {"baseline": 0, "published": 1, "learned": 2, "heapdb": 3, "transfer": 4, "ablation": 5}
    return sorted(methods, key=lambda n: (order.get(methods[n]["group"], 9), list(methods).index(n)))


def blocks(n: int, k: int) -> list[tuple[int, int]]:
    edges = [round(i * n / k) for i in range(k + 1)]
    return [(edges[i], edges[i + 1]) for i in range(k)]


# -- sections ---------------------------------------------------------------------------------

def headline(ds: str, tr: str, T: dict) -> list[str]:
    M = T["methods"]
    ref = strongest(M)
    s = summarize(M[HEADLINE_HDB]["ranks"])
    names = [n for n in (HEADLINE_HDB, "heapdb+learned", LOGISTIC, LOGISTIC_DEFAULT, "heapdb", ref, "bm25-diff",
                         "bm25-msg") if n in M]
    names = list(dict.fromkeys(names))
    pub = [n for n in M if M[n]["group"] == "published"]
    L = [f"**{TRACK_TITLES[(ds, tr)]}.** {s['n']} bugs; strongest equal-input baseline: `{ref}`. "
         "Rows in the *published* group are reference rows: other systems run on their own inputs "
         "(listed below the table), re-scored on the same candidates.\n"]
    L += table(M, names + pub, ref)
    if pub:
        L.append("")
        for n in pub:
            L.append(f"- `{n}` inputs: {M[n]['inputs']}.")
    L.append("")
    return L


def _verdicts(M: dict) -> dict[str, list[tuple[str, float, float, float]]]:
    out = {"win": [], "tie": [], "loss": []}
    for nm in ordered(M):
        if nm == HEADLINE_HDB or M[nm]["group"] in ("ablation", "transfer", "heapdb"):
            continue
        d, lo, hi = cmp_row(M[HEADLINE_HDB], M[nm])
        out[verdict(lo, hi)].append((nm, d, lo, hi))
    return out


def logistic_bullets(M: dict, n: int) -> list[str]:
    """The logistic combine mode against noisy-OR HeapDB and the strongest baseline."""
    ref = strongest(M)
    ls = summarize(M[LOGISTIC]["ranks"])
    d1, lo1, hi1 = cmp_row(M[LOGISTIC], M[HEADLINE_HDB])
    d2, lo2, hi2 = cmp_row(M[LOGISTIC], M[ref])
    v1, v2 = verdict(lo1, hi1), verdict(lo2, hi2)
    vs_nor = {"win": "**beats** noisy-OR HeapDB", "tie": "is statistically tied with noisy-OR HeapDB",
              "loss": "**trails** noisy-OR HeapDB"}[v1]
    vs_ref = {"win": f"**beats** `{ref}`", "tie": f"is statistically tied with `{ref}`",
              "loss": f"**trails** `{ref}`"}[v2]
    L = [f"- **Logistic combine mode** (`{LOGISTIC}`, not the default): MRR {ls['MRR']:.3f}, Acc@1 "
         f"{ls['Acc@1']:.3f}, Acc@5 {ls['Acc@5']:.3f}. It {vs_nor} (`{HEADLINE_HDB}`: ΔMRR {d1:+.3f} "
         f"[{lo1:+.3f}, {hi1:+.3f}]) and {vs_ref}, the strongest equal-input baseline (ΔMRR {d2:+.3f} "
         f"[{lo2:+.3f}, {hi2:+.3f}])."]
    k = max(1, round(0.1 * n))
    ref0 = max((nm for nm in M if M[nm]["group"] in EQUAL_GROUPS), key=lambda nm: statistics.fmean(rrs(M[nm])[:k]))
    d, lo, hi = paired_bootstrap(rrs(M[LOGISTIC_DEFAULT])[:k], rrs(M[ref0])[:k])
    d0, lo0, hi0 = paired_bootstrap(rrs(M[LOGISTIC_DEFAULT])[:k], rrs(M["heapdb+learned"])[:k])
    L.append(f"- **Logistic cold start** (first {k} bugs): `{LOGISTIC_DEFAULT}` is {WORD[verdict(lo, hi)]} of "
             f"`{ref0}` (ΔMRR {d:+.3f} [{lo:+.3f}, {hi:+.3f}]) and {WORD[verdict(lo0, hi0)]} of noisy-OR "
             f"`heapdb+learned` (ΔMRR {d0:+.3f} [{lo0:+.3f}, {hi0:+.3f}]).")
    L.append("")
    return L


def logistic_table(res: dict) -> list[str]:
    """One row per track: logistic vs noisy-OR HeapDB vs the strongest equal-input baseline."""
    L = ["### Logistic vs noisy-OR HeapDB vs the strongest equal-input baseline\n",
         "MRR over the whole stream; ΔMRR paired bootstrap, 95% CI. Tuned = follow-the-leader over each "
         "fixed grid; default = the first grid entry, no tuning.\n",
         "| track | noisy-OR tuned | logistic tuned | Δ logistic − noisy-OR | strongest baseline | its MRR | "
         "Δ logistic − baseline | noisy-OR default | logistic default | Δ (defaults) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, D in res["datasets"].items():
        for tr, T in D["tracks"].items():
            M = T["methods"]
            if LOGISTIC not in M:
                continue
            ref = strongest(M)
            mrr = lambda nm: statistics.fmean(rrs(M[nm]))  # noqa: E731
            c = lambda a, b: "{:+.3f} [{:+.3f}, {:+.3f}]".format(*cmp_row(M[a], M[b]))  # noqa: E731
            L.append(f"| {ds}/{tr} | {mrr(HEADLINE_HDB):.3f} | {mrr(LOGISTIC):.3f} | {c(LOGISTIC, HEADLINE_HDB)} | "
                     f"`{ref}` | {mrr(ref):.3f} | {c(LOGISTIC, ref)} | {mrr('heapdb+learned'):.3f} | "
                     f"{mrr(LOGISTIC_DEFAULT):.3f} | {c(LOGISTIC_DEFAULT, 'heapdb+learned')} |")
    L.append("")
    return L


def design_table(res: dict) -> list[str]:
    """Formulations tried on the development slices (pure-Python references)."""
    tracks = [(ds, tr, T) for ds, D in res["datasets"].items() for tr, T in D["tracks"].items() if "design" in T]
    if not tracks:
        return []
    rows = list(tracks[0][2]["design"]["mrr"])
    ref = rows[0]
    L = ["## Combine formulations tried (development slices only)\n",
         "How the logistic mode's formulation was chosen. Each row is a pure-Python reference "
         "implementation (`evals/bic/combine_dev.py`) run with the same protocol (prequential, flat "
         "kernel, no dose) on the **development slice only** (first 25% of each stream), so the choice "
         "did not look at the rest of the streams. (a) = naive-Bayes log-odds sum of the Beta links, "
         "(b) = learned intercept, (c) = online logistic regression (FTRL-Proximal, L2 towards the seeds). "
         "MRR; in brackets the difference to the first row.\n",
         "| formulation | " + " | ".join(f"{ds}/{tr} (bugs 1-{T['design']['dev_n']})" for ds, tr, T in tracks) + " |",
         "|---|" + "---|" * len(tracks)]
    for r in rows:
        cells = []
        for _, _, T in tracks:
            x = T["design"]["mrr"][r]
            cells.append(f"{x:.3f}" if r == ref else f"{x:.3f} ({x - T['design']['mrr'][ref]:+.3f})")
        L.append(f"| {r} | " + " | ".join(cells) + " |")
    L.append("")
    nb = "(a)+(b) naive Bayes, learned base, count credit"
    lr = "(c) online LR, per event, lr 0.1 (= engine default)"
    if nb in rows and lr in rows:
        nb_worse = [f"{ds}/{tr}" for ds, tr, T in tracks if T["design"]["mrr"][nb] < T["design"]["mrr"][ref]]
        lr_worse = [f"{ds}/{tr}" for ds, tr, T in tracks if T["design"]["mrr"][lr] < T["design"]["mrr"][ref]]
        L.append("Reading: multiplying the Beta links' odds (naive Bayes, the simplest option) is "
                 + (f"below noisy-OR on {', '.join(nb_worse)}" if nb_worse else "at or above noisy-OR on every track")
                 + "; naive Bayes counts correlated evidence (several buckets of related signals) once per "
                 "feature. The online logistic regression learns discriminative weights instead and is "
                 + (f"below noisy-OR on {', '.join(lr_worse)}" if lr_worse else "at or above noisy-OR on every track")
                 + " on these slices, so the engine implements (c). Count credit beats the noisy-OR EM credit for "
                 "the naive-Bayes variants, and an intercept seeded at a tiny base rate slows the online learner "
                 "(its negative examples then have almost no gradient), which is why the logistic runs use "
                 "HeapDB's default prior Beta(1, 4) to seed it.\n")
    return L


def summary(res: dict) -> list[str]:
    """Strengths-first summary, every claim chosen by a paired bootstrap CI."""
    L = ["## Summary\n"]
    for ds, D in res["datasets"].items():
        for tr, T in D["tracks"].items():
            M = T["methods"]
            if HEADLINE_HDB not in M:
                continue
            v = _verdicts(M)
            n = len(T["bugs"]["ids"])
            fmt_l = lambda xs: ", ".join(f"`{nm}` ({d:+.3f} [{lo:+.3f}, {hi:+.3f}])" for nm, d, lo, hi in xs)  # noqa: E731
            hs = summarize(M[HEADLINE_HDB]["ranks"])
            L.append(f"**{ds}/{tr}** ({n} bugs). `{HEADLINE_HDB}` reaches MRR {hs['MRR']:.3f}, "
                     f"Acc@1 {hs['Acc@1']:.3f}, Acc@5 {hs['Acc@5']:.3f}.\n")
            if v["win"]:
                L.append(f"- **Wins** (ΔMRR CI above 0) against {fmt_l(v['win'])}.")
            nl = statistics.fmean(rrs(M["heapdb"]))
            hl = statistics.fmean(rrs(M["heapdb+learned"]))
            d, lo, hi = cmp_row(M["heapdb+learned"], M["heapdb"])
            L.append(f"- **Feedback is where the lift comes from**: confirmed/rejected causes take the default "
                     f"configuration from MRR {nl:.3f} (no feedback) to {hl:.3f}, ΔMRR {d:+.3f} [{lo:+.3f}, {hi:+.3f}].")
            k = max(1, round(0.1 * n))
            # strongest equal-input baseline *on this slice*, not overall
            ref0 = max((nm for nm in M if M[nm]["group"] in EQUAL_GROUPS),
                       key=lambda nm: statistics.fmean(rrs(M[nm])[:k]))
            word = {"win": "ahead", "tie": "statistically tied", "loss": "behind"}
            d, lo, hi = paired_bootstrap(rrs(M["heapdb+learned"])[:k], rrs(M[ref0])[:k])
            cold = (f"- **Cold start** (first {k} bugs, 10% of the stream), against the best equal-input "
                    f"baseline on that slice (`{ref0}`): `heapdb+learned` is {word[verdict(lo, hi)]}, "
                    f"ΔMRR {d:+.3f} [{lo:+.3f}, {hi:+.3f}].")
            same = None
            if "abl:fl-inputs-only" in M and "fonte" in M:
                d, lo, hi = cmp_row(M["abl:fl-inputs-only"], M["fonte"])
                same = (f"- **Same inputs as FONTE** (FL scores and method history only, plus feedback on "
                        f"earlier bugs): `abl:fl-inputs-only` MRR {statistics.fmean(rrs(M['abl:fl-inputs-only'])):.3f} "
                        f"vs `fonte` {statistics.fmean(rrs(M['fonte'])):.3f}; HeapDB is {word[verdict(lo, hi)]}, "
                        f"ΔMRR {d:+.3f} [{lo:+.3f}, {hi:+.3f}].")
                if verdict(lo, hi) == "win":
                    L.append(same)
                    same = None
            if v["tie"]:
                L.append(f"- **Ties** (CI spans 0) with {fmt_l(v['tie'])}.")
            if v["loss"]:
                L.append(f"- **Trails** (CI below 0) {fmt_l(v['loss'])}.")
            if same:
                L.append(same)
            L.append(cold)
            L.append("")
            if LOGISTIC in M:
                L += logistic_bullets(M, n)
            ref = strongest(M)
            d, lo, hi = cmp_row(M[HEADLINE_HDB], M[ref])
            share = hs["MRR"] / max(1e-9, statistics.fmean(rrs(M[ref])))
            pub = [nm for nm in M if M[nm]["group"] == "published"]
            beaten = [nm for nm, *_ in v["win"] if nm in pub]
            parts = [f"Against the strongest equal-input baseline, `{ref}`, HeapDB "
                     f"{ {'win': 'is ahead', 'tie': 'is statistically tied', 'loss': 'is behind'}[verdict(lo, hi)] } "
                     f"(ΔMRR {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; {share:.0%} of its MRR)."]
            if pub:
                parts.append(f"It beats {len(beaten)} of {len(pub)} published reference rows on this track "
                             "(those systems take no feedback from these labels; HeapDB and the learned "
                             "baselines do).")
            L.append("*Takeaway:* " + " ".join(parts) + "\n")
    return L


def track_section(ds: str, tr: str, T: dict, meta: dict) -> list[str]:
    M = T["methods"]
    B = T["bugs"]
    ref = strongest(M)
    n = len(B["ids"])
    dev = B["dev_n"]
    L = [f"## {TRACK_TITLES[(ds, tr)]}\n"]
    outside = sum(1 for x in B["in_scope"] if not x)
    L.append(f"{n} bugs, {sum(B['n_cands'])} candidates (median {statistics.median(B['n_cands']):.0f} per bug, "
             f"max {max(B['n_cands'])}). True BIC outside the candidate set: {outside} bugs "
             f"({outside / n:.1%}); they count as misses (RR 0) for every method.\n")
    L.append("All methods, whole stream. ΔMRR is paired against the strongest equal-input baseline "
             f"(`{ref}`). Ties in score are broken newest-first for every method.\n")
    L += table(M, ordered(M), ref)
    L.append("")
    # dev / post-dev
    L.append("### Development slice vs the rest\n")
    L.append(f"The feature buckets and the HeapDB grid were designed on the first {dev} bugs of this "
             f"stream (the development slice, {dev / n:.0%}) and then frozen. That is a statement about "
             "the process, which these outputs cannot prove; what they can show is whether HeapDB drops "
             "more than the baselines from the first slice to the rest.\n")
    L.append(f"| method | MRR bugs 1-{dev} | MRR bugs {dev + 1}-{n} | ΔMRR vs `{ref}` on bugs {dev + 1}-{n} [95% CI] |")
    L.append("|---|---|---|---|")
    seen = set()
    for nm in [x for x in (HEADLINE_HDB, "heapdb+learned", LOGISTIC, LOGISTIC_DEFAULT, "logreg", "lambdamart",
                            "linear+tuned", ref) if x in M]:
        if nm in seen:
            continue
        seen.add(nm)
        a = rrs(M[nm])
        cell = "(reference)" if nm == ref else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*paired_bootstrap(a[dev:], rrs(M[ref])[dev:]))
        L.append(f"| `{nm}` | {statistics.fmean(a[:dev]):.3f} | {statistics.fmean(a[dev:]):.3f} | {cell} |")
    L.append("")
    # learning curve
    k = 5 if n < 1000 else 10
    bl = blocks(n, k)
    curve = [x for x in ("heapdb", "heapdb+learned", HEADLINE_HDB, LOGISTIC_DEFAULT, LOGISTIC, "logreg",
                         "lambdamart", ref) if x in M]
    curve = list(dict.fromkeys(curve))
    L.append("### Learning curve\n")
    L.append("MRR per consecutive block of the chronological stream; *bugs seen* = labelled bugs "
             "available to the learners before the block starts. `heapdb` is HeapDB without feedback; "
             "`heapdb+logistic*` use the logistic combine mode.\n")
    L.append("| block | bugs seen | median candidates | BIC outside | " + " | ".join(f"`{c}`" for c in curve) + " |")
    L.append("|---|---|---|---|" + "---|" * len(curve))
    for a, b in bl:
        L.append(f"| {a + 1}-{b} | {a} | {statistics.median(B['n_cands'][a:b]):.0f} | "
                 f"{sum(1 for x in B['in_scope'][a:b] if not x) / (b - a):.0%} | " +
                 " | ".join(f"{statistics.fmean(rrs(M[c])[a:b]):.3f}" for c in curve) + " |")
    L.append("")
    L.append("Later blocks can be harder for every method (more candidates, more BICs outside the "
             "candidate set), so compare columns within a row, not rows within a column.\n")
    # per project
    if ds == "d4j":
        projs = sorted(set(B["projects"]))
        cols = [x for x in (HEADLINE_HDB, LOGISTIC, ref, "bm25-diff", "fonte", "bug2commit (released)") if x in M]
        cols = list(dict.fromkeys(cols))
        L.append("### Per project (MRR)\n")
        L.append("| project | bugs | " + " | ".join(f"`{c}`" for c in cols) + " |")
        L.append("|---|---|" + "---|" * len(cols))
        for p in projs:
            idx = [i for i, x in enumerate(B["projects"]) if x == p]
            L.append(f"| {p} | {len(idx)} | " + " | ".join(
                f"{statistics.fmean(rrs(M[c])[i] for i in idx):.3f}" for c in cols) + " |")
        L.append("")
        prov = Counter(B["provenance"])
        L.append("By label provenance (MRR): " + "; ".join(
            f"{p} (n={c}): " + ", ".join(
                f"`{x}` {statistics.fmean(rrs(M[x])[i] for i, q in enumerate(B['provenance']) if q == p):.3f}"
                for x in (HEADLINE_HDB, LOGISTIC, ref) if x in M) for p, c in sorted(prov.items())) + ".\n")
    # transfer
    if "heapdb+transfer" in M:
        L.append("### Cross-project transfer (leave one project out)\n")
        L.append("Each project's bugs ranked by a model trained only on the other projects' bugs (all of "
                 "them, regardless of time); no feedback from the held-out project.\n")
        L += table(M, [x for x in ("heapdb+transfer", "logreg+transfer", "heapdb+learned", "logreg") if x in M],
                   "heapdb+learned")
        L.append("")
    # bisection
    if "bisect_standard" in T:
        std = T["bisect_standard"]
        L.append("### Bisection cost\n")
        L.append("Test runs a bisection over this bug's candidates needs to isolate the BIC (FONTE's "
                 "procedure, over the candidates ordered newest-first). *Standard* = plain `git bisect`. "
                 "*Weighted* = FONTE's score-weighted bisection, with each method's ranking as weights "
                 "(1/rank). Bisection assumes the failure can be reproduced at every candidate; only "
                 "in-scope bugs.\n")
        L.append("| ranking used for the weights | mean test runs | vs standard |")
        L.append("|---|---|---|")
        ok = [i for i, x in enumerate(std) if x is not None]
        ms = statistics.fmean(std[i] for i in ok)
        L.append(f"| standard bisection (no ranking) | {ms:.2f} | - |")
        for nm in ordered(M):
            st = M[nm].get("bisect")
            if not st or M[nm]["group"] == "ablation":
                continue
            v = statistics.fmean(st[i] for i in ok if st[i] is not None)
            L.append(f"| `{nm}` | {v:.2f} | {v - ms:+.2f} |")
        L.append("")
    # tuning choices
    tuned = {nm: M[nm] for nm in M if "chosen" in M[nm]}
    if tuned:
        L.append("### Follow-the-leader choices\n")
        L.append("Tuned methods use, for each bug, the grid entry with the best MRR on the bugs already "
                 "seen. Share of bugs ranked with each choice (top 3):\n")
        for nm, m in tuned.items():
            c = Counter(m["chosen"]).most_common(3)
            L.append(f"- `{nm}` (grid of {m.get('grid_size', len(m.get('grid', [])))}): " +
                     "; ".join(f"{k} ({v / n:.0%})" for k, v in c))
        L.append("")
    # links
    hl = M.get("heapdb+learned", {})
    if hl.get("links_top"):
        L.append("### What HeapDB learned\n")
        L.append("Global links after the whole stream (`heapdb+learned`), features with at least 20 "
                 "observations. Strength = posterior mean P(candidate is the BIC | feature), the "
                 "noisy-OR link.\n")
        L.append("| strongest | strength | n | | weakest | strength | n |")
        L.append("|---|---|---|---|---|---|---|")
        top, bot = hl["links_top"][:10], hl["links_bottom"][-8:]
        def cell(r):
            return "| | " if r is None else f"| `{r[0]}` | {r[1]:.2e} | {r[2]:.0f} "
        for i in range(max(len(top), len(bot))):
            a = top[i] if i < len(top) else None
            b = bot[i] if i < len(bot) else None
            L.append(cell(a) + "| " + cell(b) + "|")
        L.append("")
    lg = M.get(LOGISTIC_DEFAULT, {})
    if lg.get("links_top"):
        b = lg.get("base_log_odds")
        L.append(f"Logistic weights after the whole stream (`{LOGISTIC_DEFAULT}`), features with at least 20 "
                 "observations: each feature multiplies the odds that a candidate is the BIC by "
                 "exp(weight)" + (f"; intercept (base log-odds) {b:+.2f}" if b is not None else "") + ".\n")
        L.append("| largest | weight | x odds | n | | smallest | weight | x odds | n |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        top, bot = lg["links_top"][:10], lg["links_bottom"][-8:]
        def wcell(r):
            return "| | | " if r is None else f"| `{r[0]}` | {r[1]:+.2f} | {math.exp(r[1]):.3g} | {r[2]:.0f} "
        for i in range(max(len(top), len(bot))):
            a = top[i] if i < len(top) else None
            c = bot[i] if i < len(bot) else None
            L.append(wcell(a) + "| " + wcell(c) + "|")
        L.append("")
    return L


def d4j_extras(D: dict) -> list[str]:
    L = []
    rep = D["tracks"]["report"]
    fl = D["tracks"]["fl"]
    B = rep["bugs"]
    lags = [x for x in B["lag_days"] if x is not None]
    L.append("## Defects4J: windows, reproduction and published numbers\n")
    L.append("**Window.** The report track ranks the whole java-touching history before the buggy "
             "revision, so no BIC is outside it. The lag from BIC to failure (newest candidate) is long; "
             "a time window would lose this share of BICs:\n")
    L.append("| window | BICs inside | outside |")
    L.append("|---|---|---|")
    for w in (7, 30, 90, 365, 730, 1825):
        inside = sum(1 for x in lags if x <= w)
        L.append(f"| {w} days | {inside} ({inside / len(lags):.0%}) | {len(lags) - inside} |")
    L.append(f"\nMedian lag {statistics.median(lags):.0f} days. FL track: FONTE's C_BIC misses the BIC for "
             f"{sum(1 for x in fl['bugs']['in_scope'] if not x)} of {len(fl['bugs']['in_scope'])} bugs "
             f"({fl.get('stats', {}).get('bugs_without_candidates', 0)} bugs have an empty C_BIC and are dropped from that track).\n")
    st = rep.get("stats", {})
    if st:
        L.append(f"Bug reports that contain the BIC's commit id: {st['reports_mentioning_bic_id']} of "
                 f"{st['bugs_total']}. Median candidates per bug: {st['candidates_java_median']:.0f} "
                 f"java-touching commits of {st['commits_log_median']:.0f} in the history.\n")
    # reproduction
    r = rep["repro"].get("bug2commit_vs_released_C")
    f = fl["repro"].get("fonte_vs_released_icse23")
    L.append("**Reproduction checks** (max tie rule, as in the papers):\n")
    if r:
        L.append(f"- Bug2Commit's released per-bug ranking, re-scored here, gives the same BIC rank as the "
                 f"released aggregate (`Bug2Commit.csv`, full history) for {r['agree']} of {r['n']} bugs.")
    if f:
        L.append(f"- Our FONTE re-implementation (git history, Ochiai, tau=max, alpha=0, lambda=0.1, style "
                 f"changes removed) gives the same BIC rank as FONTE's released ICSE'23 ranking "
                 f"(`max-0-0.1.csv`) for {f['agree']} of {f['n']} bugs.")
    L.append("")
    # released ranks -> metrics
    rel = fl["repro"].get("released_ranks", {})
    ids_fl = fl["bugs"]["ids"]
    rows = []
    for name, label in (("git_line_Ochiai_voting_C_BIC/max-0-0.1.csv", "FONTE (released ranks)"),
                        ("Bug2Commit_C_BIC.csv", "Bug2Commit (released ranks, C_BIC)"),
                        ("FBL-BERT_C_BIC.csv", "FBL-BERT (released ranks, C_BIC)")):
        ranks = [x for x in rel.get(name, []) if x is not None]
        if ranks:
            s = summarize(ranks)
            rows.append(f"| {label} | {s['n']} | {fmt(s['Acc@1'])} | {fmt(s['Acc@5'])} | {fmt(s['MRR'])} |")
    fmax = None
    if "fonte" in fl["methods"]:
        from .core import summarize as _s
        fm = fl["methods"]["fonte"]
        fmax = _s(fm["ranks"])
    L.append("**Published numbers.** " + PUBLISHED_SRC + ". These are the papers' own numbers, as "
             "printed; the inputs differ from the equal-input comparison above (FONTE uses coverage and FL "
             "scores and no report; Bug2Commit uses the report and failing-test symptoms; FBL-BERT uses "
             "the report and a model trained on external data).\n")
    L.append("| as published | MRR | Acc@1 | Acc@5 |")
    L.append("|---|---|---|---|")
    for name, m, a1, a5 in PUBLISHED:
        L.append(f"| {name} | {m:.3f} | {a1:.3f} | {a5:.3f} |")
    L.append("")
    if rows:
        L.append("Released per-bug rankings of the ICSE'23 version (130 of these bugs), metrics recomputed "
                 "here with the max tie rule:\n")
        L.append("| released ranking | bugs | Acc@1 | Acc@5 | MRR |")
        L.append("|---|---|---|---|---|")
        L += rows
        L.append("")
    return L


def analysis(res: dict) -> list[str]:
    L = ["## Analysis\n", "### Where HeapDB wins, ties and loses\n",
         f"`{HEADLINE_HDB}` against every other method of the same track (paired bootstrap over bugs, "
         "ΔMRR, 95% CI; win = CI above 0, loss = CI below 0):\n"]
    L.append("| track | HeapDB | wins against | ties with | loses to |")
    L.append("|---|---|---|---|---|")
    for ds, D in res["datasets"].items():
        for tr, T in D["tracks"].items():
            M = T["methods"]
            for head in (HEADLINE_HDB, LOGISTIC):
                if head not in M:
                    continue
                w, t, l_ = [], [], []
                for nm in ordered(M):
                    if nm == head or M[nm]["group"] in ("ablation",):
                        continue
                    d, lo, hi = cmp_row(M[head], M[nm])
                    {"win": w, "tie": t, "loss": l_}[verdict(lo, hi)].append(f"`{nm}` ({d:+.3f})")
                L.append(f"| {ds}/{tr} | `{head}` | {', '.join(w) or '-'} | {', '.join(t) or '-'} | "
                         f"{', '.join(l_) or '-'} |")
    L.append("")
    L.append("### Where HeapDB ties or loses (where it does not help)\n")
    for ds, D in res["datasets"].items():
        for tr, T in D["tracks"].items():
            M = T["methods"]
            if HEADLINE_HDB not in M:
                continue
            ref = strongest(M)
            d, lo, hi = cmp_row(M[HEADLINE_HDB], M[ref])
            v = verdict(lo, hi)
            word = {"win": "beats", "tie": "ties with", "loss": "trails"}[v]
            L.append(f"- **{ds}/{tr}: against the strongest equal-input baseline.** `{HEADLINE_HDB}` {word} "
                     f"`{ref}`: ΔMRR {d:+.3f} [{lo:+.3f}, {hi:+.3f}].")
            nl = statistics.fmean(rrs(M["heapdb"]))
            hl = statistics.fmean(rrs(M["heapdb+learned"]))
            L.append(f"  Without feedback HeapDB has nothing query-specific to go on (all links equal the "
                     f"prior), so `heapdb` (lag kernel x dose only) scores MRR {nl:.3f}; feedback lifts the "
                     f"default config to {hl:.3f}. Every gain comes from the learned links.")
            n = len(T["bugs"]["ids"])
            k = max(1, round(0.1 * n))
            first = statistics.fmean(rrs(M["heapdb+learned"])[:k])
            bfirst = statistics.fmean(rrs(M[ref])[:k])
            L.append(f"  Cold start: on the first {k} bugs `heapdb+learned` scores MRR {first:.3f} vs "
                     f"{bfirst:.3f} for `{ref}`.")
            if "abl:identity-only" in M:
                a = statistics.fmean(rrs(M["abl:identity-only"]))
                b = statistics.fmean(rrs(M["abl:signal-buckets-only"]))
                L.append(f"  Ablation: with only author/dir/keyword features (no signal buckets) MRR is "
                         f"{a:.3f}; with only the signal buckets {b:.3f} (full default: {hl:.3f}).")
    L.append("")
    return L


STATIC = """### Why noisy-OR is at a disadvantage here

This was our reading of the first results; the logistic combine mode (above)
is the test. HeapDB's default link model is a noisy-OR over *causes*: each feature of an event is an
independent way of producing the effect, and the score adds up (in
probability) the strength of each feature. On this task most features are
not causes but *evidence* about one cause: a high BM25 rank, a file named in
the stack trace and a short lag all point at the same commit. Evidence
combines multiplicatively (as in logistic regression, which works in log
odds), while noisy-OR combines it roughly additively, so a candidate with
one very strong signal and a candidate with several weak ones end up close.
Bucketing continuous signals into binary features also discards the order
inside a bucket, which a model on the raw score keeps. The ablations point the
same way: the signal buckets carry almost all of HeapDB's accuracy, and the
identity features (author, directory, keyword) add little on their own.
The development-slice study shows the multiplication alone is not the whole
story: naive Bayes (odds of the existing Beta links multiplied together)
double-counts correlated buckets, and it is the discriminatively learned
weights of the online logistic regression that make the logistic mode work.

### Threats to validity

- **Labels.** Defects4J BICs come from Wen et al. (2019, test-based) and from
  manual curation by the FONTE authors (who restricted themselves to bugs with
  few candidate commits, an easier subset; see the provenance split). DPDK
  labels are developer-written `Fixes:` trailers; a trailer can name the wrong
  commit or only one of several.
- **Failure time.** Defects4J has no report timestamp in the artifact; the
  effect time is the newest candidate commit (the buggy revision or just
  before it). Lags are therefore lower bounds.
- **Leakage checks.** Methods rank before the label is revealed (tests assert
  that changing later labels never changes earlier ranks). Bug reports are
  the original issue texts; a handful mention a revision or commit (counted
  above). DPDK queries drop every trailer and hex id; a fix message can still
  quote the BIC's subject in prose (counted in the setup).
- **Equal inputs.** HeapDB and the learned baselines are built from the same
  signal table. They differ in learning schedule: HeapDB updates after every
  bug, logistic regression retrains after every bug on Defects4J but every
  500 bugs on DPDK, LambdaMART every 5 / 1000 bugs, and the DPDK learners
  keep 30 random negatives per bug (compute). This favours HeapDB slightly on
  DPDK.
- **Design on the data.** Bucket edges, the prior grid and the feature set
  were designed on the first quarter of each stream (development slice). That
  is a process claim; the split tables report both parts so a larger drop for
  HeapDB than for the baselines would be visible. The `abl:fl-inputs-only`
  row was added after the first full run, because the task called for an
  apples-to-apples row with FONTE's inputs; it uses the default
  configuration, with no tuning.
- **Learning curves drift.** The DPDK stream gets harder over time for every
  method; per-block MRR is comparable across methods, not across blocks.
- **Published systems.** Bug2Commit and FBL-BERT are their released per-bug
  outputs, re-scored on our candidate sets (the reproduction check confirms
  the re-scoring), not re-runs. FONTE is our re-implementation of its
  ICSE'23 configuration, checked against its released ranks.
- **Authors.** FONTE's commit files have no author; authors come from the
  upstream repositories (by commit id, or by unique timestamp for Lang and
  Math, whose Defects4J ids differ). Unmatched commits get author `?`.
"""


def render(res: dict) -> str:
    meta = res["meta"]
    L = ["# Bug-inducing commit ranking: results\n",
         "Generated by `python -m evals.bic.report` from `results.json` (produced by "
         "`python -m evals.bic.run`). Do not edit by hand.\n",
         "<!-- summary:start -->",
         "**Task:** given a failure (Defects4J: bug report + failing tests) or a fix (DPDK), rank the "
         "earlier commits by how likely each one introduced the bug. Bugs are processed in "
         "chronological order; each bug is ranked before its label is revealed (prequential), and "
         "learned methods only ever train on earlier bugs. Acc@k = share of bugs whose BIC is in the "
         "top k; MRR = mean reciprocal rank (0 when the BIC is outside the candidates); mean rank over "
         "bugs whose BIC is a candidate. ΔMRR: paired bootstrap over bugs, 95% CI.\n"]
    L += summary(res)
    L.append("## Headline tables\n")
    L += logistic_table(res)
    for ds, D in res["datasets"].items():
        for tr, T in D["tracks"].items():
            L += headline(ds, tr, T)
    L.append("<!-- summary:end -->\n")
    L.append("## Methods\n")
    seen = {}
    for D in res["datasets"].values():
        for T in D["tracks"].values():
            for nm, m in T["methods"].items():
                seen.setdefault(nm, m)
    for nm, m in seen.items():
        L.append(f"- `{nm}` ({m['group']}): {m['description']}. Inputs: {m['inputs']}.")
    L.append("")
    for ds, D in res["datasets"].items():
        for tr, T in D["tracks"].items():
            L += track_section(ds, tr, T, D["meta"])
        if ds == "d4j":
            L += d4j_extras(D)
        if ds == "dpdk":
            st = D["meta"]["stats"]
            L.append("## DPDK: scope and label statistics\n")
            L.append(f"{st['fix_commits']} commits carry a `Fixes:` trailer; {st['bugs']} fixes have at least one "
                     f"resolvable, earlier BIC ({st.get('unresolved_refs', 0)} unresolvable ids, "
                     f"{st.get('refs_not_earlier', 0)} pointing at a later commit). The BIC shares no file with "
                     f"the fix for {st['bic_no_shared_file']} fixes ({st['bic_no_shared_file'] / st['bugs']:.1%}) "
                     f"and is older than the newest {D['meta']['max_scope']} candidates for "
                     f"{st['bic_beyond_cap']} ({st['bic_beyond_cap'] / st['bugs']:.1%}); both count as misses. "
                     f"After removing trailers and hex ids, {st['query_contains_bic_subject']} fix messages "
                     "still quote the BIC's subject line.\n")
    L += analysis(res)
    L += design_table(res)
    L.append(STATIC)
    L.append("## Setup\n")
    e = meta["env"]
    L.append(f"- Environment: Python {e['python']}, heapdb {e['heapdb']}, numpy {e['numpy']}, "
             f"scikit-learn {e['scikit-learn']}, lightgbm {e['lightgbm']}; {e['cpu']} ({e['cpus']} CPUs); "
             f"{e['platform']}. Git: `{e['git']}`. Runtime of the last `run` invocation: "
             f"{meta.get('runtime_seconds', 0):.0f} s (cached per-method results are reused).")
    L.append(f"- HeapDB grid (follow-the-leader): {', '.join(g['name'] for g in meta['hdb_grid'])}; "
             f"default `{meta['hdb_default']}`.")
    if meta.get("hdb_logistic_grid"):
        L.append(f"- HeapDB logistic grid (follow-the-leader): "
                 f"{', '.join(g['name'] for g in meta['hdb_logistic_grid'])}; default "
                 f"`{meta['hdb_logistic_default']}` (prior Beta(1, 4) seeds the intercept).")
    for ds, D in res["datasets"].items():
        m = D["meta"]
        if ds == "d4j":
            L.append(f"- Defects4J: FONTE artifact {m['fonte_url']} @ `{m['fonte_commit']}` (MIT); input "
                     f"manifest sha256 `{m['manifest_sha256']}` over {m['manifest_files']} files; author "
                     f"table sha256 `{m['authors_sha256']}`.")
            unk = sum(v["unknown"] for v in m["author_match"].values())
            tot = sum(v["commits"] for v in m["author_match"].values())
            bt = sum(v["by_timestamp"] for v in m["author_match"].values())
            L.append(f"- Authors: {tot - unk} of {tot} candidate commits matched ({bt} of them by timestamp); "
                     f"{unk} unknown.")
        else:
            L.append(f"- DPDK: {m['dpdk_url']} @ `{m['dpdk_commit']}`; `git log --numstat` sha256 "
                     f"`{m['log_sha256']}`; {m['commits']} commits.")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=str(HERE / "results.json"))
    ap.add_argument("-o", "--out", default=str(HERE / "RESULTS.md"))
    a = ap.parse_args(argv)
    res = json.loads(Path(a.results).read_text())
    Path(a.out).write_text(render(res))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
