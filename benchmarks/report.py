"""Render a results JSON into markdown.

    python -m benchmarks.report benchmarks/results/<file>.json [-o benchmarks/RESULTS.md]
        [--compare benchmarks/results/<older-run>.json]

``--compare`` (repeatable) adds a "Python engine vs Rust engine" section
that puts the ``heapdb`` numbers of an older run (the pure-Python engine)
next to this run's, one subsection per file.

Every number in the output is copied from the JSON (only formatted). The
hand-written "Reading these results" text lives in
``benchmarks/reading.md`` and is included verbatim, so a re-run regenerates
the tables without touching the interpretation (re-check it against the
new tables).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from .workload import size_label

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL_ORDER = ("heapdb", "bruteforce-rust", "bruteforce", "sqlite-mem", "sqlite-disk", "postgres")
KERNEL_DESC = {
    "lognormal_3h": "log-normal median 3h, sigma 0.6 (support 33h; default)",
    "exp_10m": "exponential, half-life 10 min (support 100 min)",
    "exp_6h": "exponential, half-life 6h (support 60h)",
    "lognormal_narrow": "log-normal median 3h, sigma 0.3 (support 9.96h)",
    "lognormal_wide": "log-normal median 3h, sigma 1.2 (support 365h = 15.2 days)",
    "window_24h": "flat window 0-24h",
}


def ms(x: float | None) -> str:
    if x is None:
        return "-"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def num(x: float | None, nd: int = 0) -> str:
    return "-" if x is None else f"{x:,.{nd}f}"


def mib(b: int | None) -> str:
    return "-" if b is None else f"{b / 2**20:,.0f}"


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (r.get("n_events", 0),
                                       IMPL_ORDER.index(r["impl"]) if r["impl"] in IMPL_ORDER
                                       else 99))


def _run(r: dict[str, Any], kernel: str) -> dict[str, Any] | None:
    for x in r.get("runs", []):
        if x["kernel"] == kernel:
            return x
    return None


def _status(r: dict[str, Any]) -> str | None:
    if r.get("skipped"):
        return f"not measured: {r['skipped']}".replace("not measured: not measured: ",
                                                       "not measured: ")
    if r.get("error"):
        return f"ERROR: {r['error']}"
    return None


def render_compare(new: dict[str, Any], old: dict[str, Any], old_rel: str, heading: str) -> str:
    """Python engine (older run) vs Rust engine (this run), heapdb only."""
    L: list[str] = []
    oe, ne = old["env"], new["env"]
    L.append(f"### {heading}\n")
    L.append(f"`heapdb` rows of the pure-Python engine (from "
             f"[`{os.path.basename(old_rel)}`](results/{os.path.basename(old_rel)}), git "
             f"`{(oe.get('git_sha') or '')[:7]}`, {oe['date_utc']}, host `{oe['host']}`, CPU "
             f"{oe['cpu_model']}) next to this run's Rust engine (git "
             f"`{(ne.get('git_sha') or '')[:7]}`, {ne['date_utc']}, host `{ne['host']}`, CPU "
             f"{ne['cpu_model']}). Same workload generator, seed, queries and kernel; the "
             "Python engine's `bruteforce` used Python scoring, this run's uses Rust scoring. "
             "Speed-up = Python / Rust."
             + (" **The two runs used different CPUs** (same VM type and name, different "
                "host hardware), so the speed-ups include a hardware difference."
                if oe.get("cpu_model") != ne.get("cpu_model") else "") + "\n")
    k = new["config"]["default_kernel"]

    def get(res, impl, n):
        return next((r for r in res["latency"] if r["impl"] == impl and r["n_events"] == n
                     and not _status(r)), None)

    rows = []
    sizes = sorted({r["n_events"] for r in new["latency"]} | {r["n_events"] for r in old["latency"]})
    for n in sizes:
        o, w = get(old, "heapdb", n), get(new, "heapdb", n)
        ox, wx = (_run(o, k) if o else None), (_run(w, k) if w else None)

        def lat(x, key):
            return x["latency"][key] if x and "latency" in x else None

        def sp(a, b):
            return "-" if a is None or b is None or b == 0 else f"{a / b:.1f}x"

        def bpe(r):
            if not r or r.get("rss_delta_bytes") is None:
                return None
            return r["rss_delta_bytes"] / r["n_events"]

        status_o = "not measured" if o is None else None
        status_w = "not measured" if w is None else None
        rows.append([
            size_label(n),
            status_o or ms(lat(ox, "p50_ms")), status_w or ms(lat(wx, "p50_ms")),
            sp(lat(ox, "p50_ms"), lat(wx, "p50_ms")),
            status_o or ms(lat(ox, "p99_ms")), status_w or ms(lat(wx, "p99_ms")),
            status_o or num(o["ingest_events_per_s"]) if o else "not measured",
            num(w["ingest_events_per_s"]) if w else "not measured",
            sp(w["ingest_events_per_s"] if w else None, o["ingest_events_per_s"] if o else None),
            num(bpe(o)) if o else "not measured", num(bpe(w)) if w else "not measured",
        ])
    L.append(table(["N", "Python p50 ms", "Rust p50 ms", "p50 speed-up", "Python p99 ms",
                    "Rust p99 ms", "Python ingest ev/s", "Rust ingest ev/s", "ingest speed-up",
                    "Python RSS B/event", "Rust RSS B/event"], rows))
    L.append("")
    # Sensitivity kernels at the sensitivity size, heapdb p50.
    osens, nsens = old.get("sensitivity") or [], new.get("sensitivity") or []

    def spick(sens, impl, bucket, links):
        return next((r for r in sens if r["impl"] == impl and r["bucket_seconds"] == bucket
                     and r["links"] == links and not _status(r)), None)
    if osens and nsens and osens[0]["n_events"] == nsens[0]["n_events"]:
        b15 = new["config"]["default_bucket"]
        o, w = spick(osens, "heapdb", b15, "skewed"), spick(nsens, "heapdb", b15, "skewed")
        if o and w:
            L.append(f"heapdb p50 by kernel at {size_label(nsens[0]['n_events'])} events "
                     "(15 min buckets, skewed links), ms:\n")
            rows = []
            for x in w.get("runs", []):
                kn = x["kernel"]
                ox = _run(o, kn)
                if not ox or "latency" not in ox or "latency" not in x:
                    continue
                rows.append([f"`{kn}`", ms(ox["latency"]["p50_ms"]), ms(x["latency"]["p50_ms"]),
                             f"{ox['latency']['p50_ms'] / x['latency']['p50_ms']:.1f}x",
                             num(ox.get("events_scored_mean"), 1), num(x.get("events_scored_mean"), 1)])
            L.append(table(["kernel", "Python p50", "Rust p50", "speed-up",
                            "Python events scored", "Rust events scored"], rows))
            L.append("")
    oq, nq = old.get("quality"), new.get("quality")
    if oq and nq and oq.get("seeds") == nq.get("seeds"):
        diffs = []
        for m, per in nq["by_period"].items():
            for p, vals in per.items():
                for k2 in ("hit1", "hit3", "mrr"):
                    try:
                        a, b = oq["by_period"][m][p][k2]["mean"], vals[k2]["mean"]
                    except KeyError:
                        continue
                    if abs(a - b) > 1e-12:
                        diffs.append(f"`{m}` {p} {k2} {a:.3f} -> {b:.3f}")
        if not diffs:
            L.append("Ranking quality (same simulation, same seeds): identical to the Python "
                     "engine's numbers in every cell.\n")
        else:
            L.append("Ranking quality (same simulation, same seeds) differs from the Python "
                     f"engine's numbers in {len(diffs)} cell(s): " + "; ".join(diffs) + ".\n")
    return "\n".join(L)


def render(res: dict[str, Any], json_rel: str,
           compare: list[tuple[dict[str, Any], str]] | None = None) -> str:
    env, cfg = res["env"], res["config"]
    L: list[str] = []
    L.append("# HeapDB benchmark results\n")
    L.append(f"Generated by `python -m benchmarks.report {json_rel}` from the raw results in "
             f"[`{os.path.basename(json_rel)}`](results/{os.path.basename(json_rel)}). "
             "Do not edit the tables by hand: rerun the commands below. Every number here is "
             "measured; nothing is extrapolated. Runs that were not measured say so.\n")

    # -- environment -------------------------------------------------------
    L.append("## Environment\n")
    ram = env.get("ram_bytes")
    rows = [
        ["Date (UTC)", env["date_utc"]],
        ["Host", env["host"]],
        ["CPU", f"{env['cpu_model']} ({env['cpu_count_logical']} logical CPUs)"],
        ["RAM", "-" if ram is None else f"{ram / 2**30:.1f} GiB"],
        ["OS", env["os"]],
        ["Python", f"{env['python']} ({env['python_implementation']})"],
        ["heapdb", env.get("heapdb") or "-"],
        ["Rust", env.get("rustc") or "-"],
        ["SQLite", env["sqlite_version"]],
        ["PostgreSQL", res["postgres"]["description"] if res.get("postgres") else "-"],
        ["psutil / psycopg", f"{env.get('psutil') or '-'} / {env.get('psycopg') or '-'}"],
        ["git", f"`{env['git_sha']}`" + (" (tracked files modified)" if env.get("git_dirty")
                                         else "")],
        ["Total runtime", f"{res['total_seconds']:.0f} s"],
    ]
    L.append(table(["", ""], rows, "ll"))
    L.append("\nThe recorded git sha is the checkout the run started from. The benchmark "
             "code itself was untracked at that point if it was added in the same commit as "
             "the results; the `heapdb` library code is that of the recorded sha.\n")
    L.append("### Exact commands\n")
    L.append("```bash\npip install . -r benchmarks/requirements.txt   # builds the Rust engine (maturin)\n"
             f"{env['command']}\n"
             f"python -m benchmarks.report {json_rel} -o benchmarks/RESULTS.md"
             + "".join(f" \\\n    --compare {rel}" for _, rel in (compare or []))
             + "\n```\n")
    for old, rel in compare or []:
        L.append(f"`{os.path.basename(rel)}` (Python engine) was produced by: "
                 f"`{old['env']['command']}`\n")

    # -- setup -------------------------------------------------------------
    wd = cfg["workload_defaults"]
    L.append("## Setup\n")
    L.append(
        f"* **Workload** (`benchmarks/workload.py`, seed {cfg['seed']}): N events over "
        f"{cfg['entities']} entities, uniformly over {cfg['days']:g} days, so events per "
        f"entity-day grow with N. Each event has 1-{wd['max_features']} features from a "
        f"{wd['vocab']}-token vocabulary with Zipf(s={wd['zipf_s']}) frequencies; dose = "
        "`log1p(lines)/log1p(100)` with `lines ~ 1 + LogNormal(ln 80, 1)` (as in "
        "`examples/code_changes.py`). Link strengths are static Beta priors: "
        f"{wd['strong_fraction']:.0%} of features strong (U(0.3, 0.9)), the rest weak "
        "(U(0.005, 0.08)). The *flat* profile sets every link to 0.1 and every dose to 1.\n"
        f"* **Query**: top-{cfg['k']} causes of one effect for one entity at a random time in "
        f"the second half of the span. {cfg['warmup']} warm-up + {cfg['queries']} timed "
        "queries per run, same query list for every implementation. Latency is wall time "
        "around one query call in the benchmark process (includes the Python driver for "
        "SQLite/PostgreSQL; PostgreSQL over a Unix socket with a prepared statement). "
        "Percentiles are nearest-rank.\n"
        f"* **Verification before timing**: every run first checks {cfg['verify']} queries "
        "per kernel against an independent reference (heapdb's `Kernel.value` and "
        "`LinkTable.noisy_or` over every event in the window, full sort). Scores must match "
        f"within {cfg['tolerance']:g} (relative) and ids must match except for permutations "
        "within tied scores. A mismatch aborts that run and is reported as an error.\n"
        "* **Implementations** (`benchmarks/impls.py`):\n"
        "  * `heapdb`: `HeapDB.top_causes` (segment bounds, best-first scan, early stop), "
        "called from Python; since v0.5 the engine is Rust (PyO3 bindings), so latency "
        "includes the Python call and building the result objects.\n"
        "  * `bruteforce-rust`: `HeapDB.top_causes_naive`: the same Rust engine and scoring "
        "code scores every event in the segments overlapping the window, full sort. No "
        "bounds and no pruning: only the search strategy differs from `heapdb`.\n"
        "  * `bruteforce`: Python `Event` objects in per-entity time arrays (bisect), the "
        "`HeapDB.score` function on every event in the kernel window from a Python loop, "
        "then `heapq.nlargest`. Since v0.5 each `score` call goes into the Rust engine, so "
        "this measures a Python loop over a Rust function.\n"
        "  * `sqlite-mem` / `sqlite-disk`: stdlib `sqlite3`, `events(id, entity, time, "
        "weight)` + `event_features(event_id, feature)` + temp table `links(feature, "
        "ln(1-s))`, index on `events(entity, time)`. **All scoring runs in SQL**: window "
        "filter, kernel (exp/ln/power math), noisy-OR as `1 - exp(sum(ln(1-s)))` with "
        "GROUP BY, ORDER BY, LIMIT. No Python UDFs. `sqlite-disk` is a WAL-mode file "
        "with a 2 GB page cache, so warm queries mostly hit the cache: the difference "
        "to `sqlite-mem` is mainly ingest.\n"
        "  * `postgres`: the same SQL on PostgreSQL via psycopg 3, links passed as a temp "
        "table, ingest with COPY, indexes built after loading.\n"
        "* **Ingest** = inserting all events into the implementation's own structures. For "
        "heapdb that includes maintaining segment summaries and inverted indexes (its "
        "\"index build\" is part of ingest, so the index column is 0). For SQL it is the "
        "bulk insert in one transaction; index build is separate.\n"
        "* **Memory** = RSS delta (psutil) of a fresh worker process across the build, "
        "excluding the generated workload columns. For SQLite in-memory this is the "
        "database pages; for PostgreSQL the client process only, so the server size is "
        "reported as relation size instead.\n")

    # -- latency -----------------------------------------------------------
    lat = _ordered(res["latency"])
    sizes = sorted({r["n_events"] for r in lat})
    L.append("## Query latency by size (default kernel, bucket 15 min)\n")
    L.append(f"Kernel: {KERNEL_DESC.get(cfg['default_kernel'], cfg['default_kernel'])}. "
             "Latency in milliseconds.\n")
    rows = []
    for r in lat:
        st = _status(r)
        x = _run(r, cfg["default_kernel"]) if not st else None
        if x and "error" in x:
            st = f"ERROR: {x['error']}"
        if st:
            rows.append([size_label(r["n_events"]), f"`{r['impl']}`", st, "", "", "", ""])
            continue
        lt = x["latency"]
        rows.append([size_label(r["n_events"]), f"`{r['impl']}`", ms(lt["p50_ms"]),
                     ms(lt["p95_ms"]), ms(lt["p99_ms"]), ms(lt["mean_ms"]),
                     f"{x['verified_queries']} ({x['verified_nonempty']} non-empty)"])
    L.append(table(["N", "impl", "p50", "p95", "p99", "mean", "verified queries"], rows,
                   "llrrrrr"))
    L.append("")

    # relative
    L.append("### heapdb p50 relative to the others\n")
    L.append("`x` = other p50 / heapdb p50. Above 1: heapdb is faster; below 1: heapdb is "
             "slower.\n")
    rows = []
    for n in sizes:
        by = {r["impl"]: _run(r, cfg["default_kernel"]) for r in lat
              if r["n_events"] == n and not _status(r)}
        h = by.get("heapdb")
        if not h or "latency" not in h:
            continue
        row = [size_label(n), ms(h["latency"]["p50_ms"])]
        for other in IMPL_ORDER[1:]:
            o = by.get(other)
            row.append(f"{o['latency']['p50_ms'] / h['latency']['p50_ms']:.2f}x"
                       if o and "latency" in o else "not measured")
        rows.append(row)
    L.append(table(["N", "heapdb p50 (ms)", *[f"vs `{o}`" for o in IMPL_ORDER[1:]]], rows))
    L.append("")

    # pruning
    L.append("## Pruning (heapdb) and work per query\n")
    L.append("Means over the timed queries. *Events in window* is what brute force scores "
             "(also the candidate set the SQL implementations filter to); it is recorded by "
             "the `bruteforce` implementation, so it is `-` where that did not run.\n")
    rows = []
    for n in sizes:
        h = next((r for r in lat if r["n_events"] == n and r["impl"] == "heapdb"
                  and not _status(r)), None)
        b = next((r for r in lat if r["n_events"] == n and r["impl"] == "bruteforce"
                  and not _status(r)), None)
        hx = _run(h, cfg["default_kernel"]) if h else None
        bx = _run(b, cfg["default_kernel"]) if b else None
        if not hx or "latency" not in hx:
            continue
        rows.append([size_label(n), num(bx.get("events_in_window_mean") if bx else None, 1),
                     num(hx["events_scored_mean"], 1), num(hx["events_scored_p99"]),
                     num(hx["segments_in_window_mean"], 1), num(hx["segments_pruned_mean"], 1),
                     pct(hx["segments_pruned_fraction"])])
    L.append(table(["N", "events in window", "heapdb events scored", "scored p99",
                    "segments in window", "segments pruned", "pruned %"], rows))
    L.append("")

    # ingest / index / memory
    L.append("## Ingest, index build and memory\n")
    rows = []
    for r in lat:
        st = _status(r)
        if st:
            rows.append([size_label(r["n_events"]), f"`{r['impl']}`", st, "", "", "", ""])
            continue
        info = r.get("info", {})
        size_note = "-"
        if "db_file_bytes" in info:
            size_note = f"file {mib(info['db_file_bytes'])} MiB"
        elif "relation_bytes" in info:
            size_note = f"relations {mib(info['relation_bytes'])} MiB"
        elif "segments" in info:
            size_note = f"{info['segments']:,} segments"
        rss = r.get("rss_delta_bytes")
        per = "-" if rss is None else f"{rss / r['n_events']:,.0f}"
        rows.append([size_label(r["n_events"]), f"`{r['impl']}`",
                     num(r["ingest_events_per_s"]), f"{r['ingest_seconds']:.2f}",
                     f"{r['index_seconds']:.2f}", f"{mib(rss)} ({per} B/event)", size_note])
    L.append(table(["N", "impl", "ingest events/s", "ingest s", "index build s",
                    "RSS delta MiB", "storage"], rows, "llrrrrl"))
    L.append("\nIndex build: heapdb, none separate (summaries are maintained on insert); "
             "bruteforce, per-entity time arrays; SQLite, `CREATE INDEX` on (entity, time) + "
             "`ANALYZE` (primary keys are built during ingest); PostgreSQL, primary keys + "
             "(entity, time) index + `ANALYZE` after COPY. PostgreSQL's RSS delta is the "
             "client only.\n")
    skipped = cfg.get("skipped_sizes") or []
    if skipped:
        L.append(f"Sizes not run (above `--max-size {size_label(max(cfg['sizes']))}`): "
                 + ", ".join(size_label(n) for n in skipped) + ".\n")

    # -- sensitivity -------------------------------------------------------
    sens = res.get("sensitivity") or []
    if sens:
        sn = sens[0]["n_events"]
        L.append(f"## Sensitivity at {size_label(sn)} events\n")
        L.append("### Kernels (bucket 15 min, skewed links)\n")
        rows = []
        kernels = []
        for r in sens:
            for x in r.get("runs", []):
                if x["kernel"] not in kernels:
                    kernels.append(x["kernel"])

        def pick(impl, bucket, links):
            return next((r for r in sens if r["impl"] == impl and r["bucket_seconds"] == bucket
                         and r["links"] == links), None)
        b15 = cfg["default_bucket"]
        def p50(x):
            return ms(x["latency"]["p50_ms"]) if x and "latency" in x else "-"

        for kn in kernels:
            h, br, b, s = (pick(i, b15, "skewed") for i in
                           ("heapdb", "bruteforce-rust", "bruteforce", "sqlite-mem"))
            hx, brx, bx, sx = (_run(r, kn) if r and not _status(r) else None
                               for r in (h, br, b, s))
            if not hx or "latency" not in hx:
                continue
            ratio = (brx["latency"]["p50_ms"] / hx["latency"]["p50_ms"]) if brx and "latency" in brx else None
            rows.append([f"`{kn}`", num(bx.get("events_in_window_mean") if bx else None, 1),
                         num(hx["events_scored_mean"], 1), pct(hx["segments_pruned_fraction"]),
                         ms(hx["latency"]["p50_ms"]), ms(hx["latency"]["p99_ms"]),
                         p50(brx), p50(bx), p50(sx),
                         "-" if ratio is None else f"{ratio:.2f}x"])
        L.append(table(["kernel", "events in window", "heapdb scored", "pruned %",
                        "heapdb p50", "heapdb p99", "bruteforce-rust p50", "bruteforce p50",
                        "sqlite-mem p50", "bruteforce-rust / heapdb"], rows))
        L.append("\nKernels: " + "; ".join(f"`{k}` = {KERNEL_DESC[k]}" for k in kernels
                                           if k in KERNEL_DESC) + ". Latency in ms.\n")

        L.append("### Bucket size (heapdb only, skewed links)\n")
        rows = []
        buckets = sorted({r["bucket_seconds"] for r in sens if r["impl"] == "heapdb"
                          and r["links"] == "skewed"})
        for kn in kernels:
            row = [f"`{kn}`"]
            for bsec in buckets:
                h = pick("heapdb", bsec, "skewed")
                hx = _run(h, kn) if h and not _status(h) else None
                row.append("-" if not hx or "latency" not in hx else
                           f"{ms(hx['latency']['p50_ms'])} ms / {num(hx['events_scored_mean'], 1)} "
                           f"/ {pct(hx['segments_pruned_fraction'])}")
            rows.append(row)
        L.append("Cells: p50 latency / mean events scored / segments pruned.\n")
        L.append(table(["kernel", *[f"{b / 60:g} min buckets" for b in buckets]], rows))
        rows = []
        for bsec in buckets:
            h = pick("heapdb", bsec, "skewed")
            if h and not _status(h):
                rows.append([f"{bsec / 60:g} min", num(h["ingest_events_per_s"]),
                             f"{mib(h.get('rss_delta_bytes'))}",
                             num(h.get("info", {}).get("segments"))])
        L.append("")
        L.append(table(["bucket", "ingest events/s", "RSS delta MiB", "segments"], rows))
        L.append("")

        L.append("### Adversarial: flat links (every link 0.1, every dose 1)\n")
        L.append("With identical links and doses, segment bounds differ only through the "
                 "kernel and through the number of features per event (noisy-OR of k links "
                 "of 0.1 is 1 - 0.9^k). With the flat window kernel the kernel factor is the "
                 "same for every segment, so only the feature count separates segments: "
                 "pruning is much weaker and heapdb pays for bounding and sorting segments "
                 "on top of scoring.\n")
        rows = []
        for kn in ("lognormal_3h", "window_24h"):
            h, br, b, s = (pick(i, b15, "flat") for i in
                           ("heapdb", "bruteforce-rust", "bruteforce", "sqlite-mem"))
            hx, brx, bx, sx = (_run(r, kn) if r and not _status(r) else None
                               for r in (h, br, b, s))
            if not hx or "latency" not in hx:
                continue
            ratio = (brx["latency"]["p50_ms"] / hx["latency"]["p50_ms"]) if brx and "latency" in brx else None
            rows.append([f"`{kn}`", num(bx.get("events_in_window_mean") if bx else None, 1),
                         num(hx["events_scored_mean"], 1), pct(hx["segments_pruned_fraction"]),
                         ms(hx["latency"]["p50_ms"]), p50(brx), p50(bx), p50(sx),
                         "-" if ratio is None else f"{ratio:.2f}x"])
        L.append(table(["kernel", "events in window", "heapdb scored", "pruned %",
                        "heapdb p50", "bruteforce-rust p50", "bruteforce p50", "sqlite-mem p50",
                        "bruteforce-rust / heapdb"], rows))
        L.append("")
        errs = [r for r in sens if _status(r)]
        for r in errs:
            L.append(f"* `{r['impl']}` bucket {r['bucket_seconds']:g} links {r['links']}: "
                     f"{_status(r)}")
        if errs:
            L.append("")

    # -- quality -----------------------------------------------------------
    q = res.get("quality")
    if q:
        L.append("## Ranking quality against ground truth\n")
        L.append(f"Simulation of `examples/code_changes.py` (unchanged), seeds "
                 f"{q['seeds'][0]}-{q['seeds'][1]} ({q['n_seeds']} seeds, "
                 f"{q['attributions_per_seed_mean']:.1f} attributable effects per seed on "
                 "average). Rank of the true culprit PR among all candidates; mean ± sample "
                 "std over seeds of the per-seed value. Methods (`benchmarks/quality.py`): "
                 "**recency** = most recent eligible deploy in the kernel window first; "
                 "**kernel_only** = kernel × dose × rules, all links equal; **priors_only** = "
                 "HeapDB scoring with the example's prior and no feedback; "
                 "**heapdb_proportional** / **heapdb_uniform** = HeapDB with revert "
                 "feedback (culprit confirmed, other top-3 rejected) and EM vs uniform "
                 "credit.\n")
        periods = list(next(iter(q["by_period"].values())).keys())
        for metric, label in (("hit1", "hit@1"), ("hit3", "hit@3"), ("mrr", "MRR")):
            rows = []
            for m, per in q["by_period"].items():
                rows.append([f"`{m}`"] + [f"{per[p][metric]['mean']:.3f} ± "
                                          f"{per[p][metric]['std']:.3f}" for p in periods])
            L.append(f"**{label}**\n")
            L.append(table(["method", *periods], rows))
            L.append("")
        w = q["seed_wins_second_half"]
        L.append(f"Per seed, days 3.5-7, hit@1: heapdb_proportional > recency on "
                 f"{w['heapdb_beats_recency_hit1']} seeds, recency > heapdb_proportional on "
                 f"{w['recency_beats_heapdb_hit1']}, tied on {w['ties']}.\n")
        L.append("**By effect (whole week), hit@1**\n")
        effs = list(next(iter(q["by_effect"].values())).keys())
        rows = [[f"`{m}`"] + [f"{per[e]['hit1']['mean']:.3f} ± {per[e]['hit1']['std']:.3f}"
                              for e in effs] for m, per in q["by_effect"].items()]
        L.append(table(["method", *effs], rows))
        L.append("")

    if compare:
        L.append("## Python engine vs Rust engine\n")
        same = [c for c in compare if c[0]["env"].get("cpu_model") == env.get("cpu_model")]
        if same and len(same) < len(compare):
            L.append("The same-CPU comparison is the fair one; the other is the original "
                     "Python-engine run, kept for reference.\n")
        for old, rel in sorted(compare, key=lambda c: c[0]["env"].get("cpu_model")
                               != env.get("cpu_model")):
            same_cpu = old["env"].get("cpu_model") == env.get("cpu_model")
            L.append(render_compare(res, old, rel,
                                    "Same CPU (Python engine re-run)" if same_cpu
                                    else "Original Python-engine run (different CPU)"))

    # -- reading -------------------------------------------------------------
    reading = os.path.join(HERE, "reading.md")
    if os.path.exists(reading):
        with open(reading, encoding="utf-8") as fh:
            L.append(fh.read().rstrip() + "\n")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m benchmarks.report")
    ap.add_argument("results")
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "RESULTS.md"))
    ap.add_argument("--compare", action="append", default=[],
                    help="older results JSON (Python engine) for a side-by-side section; "
                         "repeatable")
    args = ap.parse_args(argv)
    with open(args.results, encoding="utf-8") as fh:
        res = json.load(fh)
    root = os.path.dirname(HERE)
    rel = os.path.relpath(os.path.abspath(args.results), root)
    compare = []
    for path in args.compare:
        with open(path, encoding="utf-8") as fh:
            compare.append((json.load(fh), os.path.relpath(os.path.abspath(path), root)))
    md = render(res, rel, compare)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
