"""Benchmark runner.

    python -m benchmarks.run --sizes 10k,100k,1m --out benchmarks/results/<date>-<host>.json
    python -m benchmarks.run --quick                     # CI smoke run (10K, few queries)
    python -m benchmarks.run --max-size 10m ...          # also try 10M (opt-in)

Each (size, implementation[, bucket, link profile]) measurement runs in a
fresh subprocess (:mod:`benchmarks.worker`), so RSS deltas are per
implementation. Every measurement first verifies its top-k on a sample of
queries against an independent reference and aborts that measurement on any
mismatch (the error is recorded in the JSON, not hidden).

Sections:

* ``latency``     -- default kernel (log-normal, median 3h, sigma 0.6, the
                     ``incident`` kernel of examples/code_changes.py), bucket
                     15 min, per size and implementation: p50/p95/p99,
                     ingest, index build, RSS delta, pruning.
* ``sensitivity`` -- at ``--sens-size``: heapdb, bruteforce and sqlite-mem
                     across kernels; heapdb across bucket sizes; and a "flat"
                     link profile (all links equal, all doses 1) where bounds
                     cannot separate segments.
* ``quality``     -- hit@1/hit@3/MRR over seeds (see benchmarks/quality.py).

PostgreSQL: ``--pg-dsn``/``HEAPDB_BENCH_PG_DSN`` if given, else (``--pg
auto``) a throwaway local cluster if server binaries are found, else it is
recorded as not measured with the reason.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from .pg import check_dsn, postgres_server
from .quality import run_quality
from .workload import (DEFAULT_BUCKET, DEFAULT_KERNEL, KERNEL_NAMES, M, SIZES, WorkloadSpec,
                       parse_size, size_label)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATENCY_IMPLS = ("heapdb", "bruteforce-rust", "bruteforce", "sqlite-mem", "sqlite-disk", "postgres")
SENS_IMPLS = ("heapdb", "bruteforce-rust", "bruteforce", "sqlite-mem")
BUCKETS = (5 * M, 15 * M, 60 * M)
TOL = 1e-9


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_bytes() -> int | None:
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def available_bytes() -> int | None:
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _heapdb_version() -> str | None:
    try:
        import heapdb
        engine = "rust" if hasattr(heapdb, "_core") else "python"
        return f"{heapdb.__version__} ({engine} engine)"
    except Exception:  # pragma: no cover
        return None


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def environment(argv: list[str]) -> dict[str, Any]:
    import sqlite3
    try:
        import psutil
        psutil_v = psutil.__version__
    except ImportError:
        psutil_v = None
    try:
        import psycopg
        psycopg_v = psycopg.__version__
    except Exception:
        psycopg_v = None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "date_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "python": sys.version.split()[0],
        "heapdb": _heapdb_version(),
        "rustc": _cmd(["rustc", "--version"]),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "sqlite_version": sqlite3.sqlite_version,
        "psutil": psutil_v,
        "psycopg": psycopg_v,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "command": "python -m benchmarks.run " + " ".join(shlex.quote(a) for a in argv),
        "ci": bool(os.environ.get("CI")),
    }


# ---------------------------------------------------------------------------
# Subprocess workers
# ---------------------------------------------------------------------------


def run_worker(cfg: dict[str, Any], timeout: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="heapdb-bench-") as d:
        cin, cout = os.path.join(d, "cfg.json"), os.path.join(d, "out.json")
        with open(cin, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, "-m", "benchmarks.worker", cin, cout],
                               cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"impl": cfg["impl"], "n_events": cfg["workload"]["n_events"],
                    "error": f"timed out after {timeout:.0f}s"}
        wall = time.perf_counter() - t0
        if not os.path.exists(cout):
            return {"impl": cfg["impl"], "n_events": cfg["workload"]["n_events"],
                    "error": f"worker exited with {p.returncode} "
                             f"(killed? out of memory?): {p.stderr.strip()[-500:]}"}
        with open(cout, encoding="utf-8") as fh:
            res = json.load(fh)
        res["wall_seconds"] = wall
        return res


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def describe(res: dict[str, Any]) -> str:
    if res.get("skipped"):
        return f"SKIPPED: {res['skipped']}"
    if res.get("error"):
        return f"ERROR: {res['error']}"
    parts = [f"ingest {res['ingest_events_per_s']:,.0f} ev/s"]
    for r in res.get("runs", []):
        if "latency" in r:
            parts.append(f"{r['kernel']} p50 {r['latency']['p50_ms']:.3f}ms")
    return ", ".join(parts) + f" [{res.get('wall_seconds', 0):.0f}s]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m benchmarks.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default=None, help="comma list, e.g. 10k,100k,1m (default)")
    ap.add_argument("--max-size", default="1m",
                    help="largest size allowed; 10m is opt-in (--max-size 10m adds it)")
    ap.add_argument("--impls", default=",".join(LATENCY_IMPLS))
    ap.add_argument("--quick", action="store_true",
                    help="CI mode: 10K only, fewer queries, 5 quality seeds")
    ap.add_argument("--queries", type=int, default=None, help="timed queries per run")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--verify", type=int, default=None, help="verified queries per run")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--entities", type=int, default=100)
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sens-size", default=None, help="size for the sensitivity section")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--quality-seeds", type=int, default=None)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--pg", choices=("auto", "off"), default="auto")
    ap.add_argument("--pg-dsn", default=None)
    ap.add_argument("--timeout", type=float, default=3600.0, help="per-worker timeout (s)")
    ap.add_argument("--mem-headroom", type=float, default=0.7,
                    help="skip a run whose estimated RSS exceeds this fraction of free RAM")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    quick = args.quick
    sizes_s = args.sizes or ("10k" if quick else "10k,100k,1m")
    max_n = parse_size(args.max_size)
    sizes = [parse_size(s) for s in sizes_s.split(",")]
    if max_n >= SIZES["10m"] and SIZES["10m"] not in sizes:
        sizes.append(SIZES["10m"])
    skipped_sizes = [n for n in sizes if n > max_n]
    sizes = sorted(n for n in sizes if n <= max_n)
    n_q = args.queries or (100 if quick else 500)
    n_w = args.warmup if args.warmup is not None else (10 if quick else 50)
    n_v = args.verify or (20 if quick else 50)
    sens_n = parse_size(args.sens_size or ("10k" if quick else "1m"))
    q_seeds = args.quality_seeds or (5 if quick else 20)
    impls = [s.strip() for s in args.impls.split(",") if s.strip()]

    env = environment(argv)
    out_path = args.out or os.path.join(
        ROOT, "benchmarks", "results",
        f"{env['date_utc'][:10]}-{env['host']}{'-quick' if quick else ''}.json")
    t_start = time.perf_counter()
    log(f"HeapDB benchmarks -> {out_path}\n{env['cpu_model']}, {env['cpu_count_logical']} cpus, "
        f"python {env['python']}, git {env['git_sha']}")

    results: dict[str, Any] = {
        "schema": 1, "env": env,
        "config": {"sizes": sizes, "skipped_sizes": skipped_sizes, "impls": impls,
                   "queries": n_q, "warmup": n_w, "verify": n_v, "k": args.k,
                   "entities": args.entities, "days": args.days, "seed": args.seed,
                   "default_kernel": DEFAULT_KERNEL, "default_bucket": DEFAULT_BUCKET,
                   "tolerance": TOL, "quick": quick,
                   "sens_size": None if args.no_sensitivity else sens_n,
                   "quality_seeds": None if args.no_quality else q_seeds,
                   "workload_defaults": WorkloadSpec(n_events=0).__dict__},
        "latency": [], "sensitivity": [], "quality": None, "postgres": None,
    }

    def cfg_for(n: int, impl: str, kernels, bucket=DEFAULT_BUCKET, links="skewed",
                dsn=None) -> dict[str, Any]:
        return {"impl": impl, "kernels": list(kernels), "bucket": bucket, "k": args.k,
                "n_queries": n_q, "n_warmup": n_w, "n_verify": n_v, "tol": TOL, "dsn": dsn,
                "workload": {"n_events": n, "n_entities": args.entities, "days": args.days,
                             "seed": args.seed, "links": links}}

    # Bytes per event observed so far, per impl (for the memory pre-check).
    per_event: dict[str, float] = {}

    def run_one(n: int, impl: str, cfg: dict[str, Any], section: str) -> dict[str, Any]:
        est = per_event.get(impl)
        avail = available_bytes()
        if est is not None and avail is not None and est * n > args.mem_headroom * avail:
            res = {"impl": impl, "n_events": n, "skipped":
                   f"estimated RSS {est * n / 2**30:.1f} GiB (from {est:.0f} B/event measured at "
                   f"a smaller size) exceeds {args.mem_headroom:.0%} of available "
                   f"{avail / 2**30:.1f} GiB"}
        else:
            log(f"[{time.perf_counter() - t_start:7.1f}s] {section} {size_label(n)} {impl} "
                f"bucket={cfg['bucket']:g} links={cfg['workload']['links']} "
                f"kernels={','.join(cfg['kernels'])}")
            res = run_worker(cfg, args.timeout)
            if not res.get("error") and res.get("rss_delta_bytes") is not None:
                per_event[impl] = (res["rss_delta_bytes"] + (res["rss_workload_bytes"] or 0)) / n
        res.setdefault("n_events", n)
        res["bucket_seconds"] = cfg["bucket"]
        res["links"] = cfg["workload"]["links"]
        log(f"    {describe(res)}")
        return res

    with _maybe_pg(args, impls) as (dsn, pg_desc):
        results["postgres"] = {"measured": dsn is not None, "description": pg_desc}
        if dsn is None and "postgres" in impls:
            log(f"PostgreSQL not measured: {pg_desc}")
        for n in sizes:
            for impl in impls:
                if impl == "postgres" and dsn is None:
                    results["latency"].append({"impl": impl, "n_events": n,
                                               "skipped": f"not measured: {pg_desc}"})
                    continue
                results["latency"].append(run_one(n, impl, cfg_for(n, impl, [DEFAULT_KERNEL],
                                                                   dsn=dsn), "latency"))
        if not args.no_sensitivity:
            for impl in SENS_IMPLS:
                results["sensitivity"].append(run_one(
                    sens_n, impl, cfg_for(sens_n, impl, KERNEL_NAMES), "sensitivity"))
            for b in BUCKETS:
                if b == DEFAULT_BUCKET:
                    continue  # already measured above with all kernels
                results["sensitivity"].append(run_one(
                    sens_n, "heapdb", cfg_for(sens_n, "heapdb", KERNEL_NAMES, bucket=b),
                    "sensitivity"))
            for impl in SENS_IMPLS:
                results["sensitivity"].append(run_one(
                    sens_n, impl, cfg_for(sens_n, impl, ("lognormal_3h", "window_24h"),
                                          links="flat"), "sensitivity"))

    if not args.no_quality:
        log(f"[{time.perf_counter() - t_start:7.1f}s] quality, {q_seeds} seeds")
        t0 = time.perf_counter()
        results["quality"] = run_quality(q_seeds)
        results["quality"]["seconds"] = time.perf_counter() - t0

    results["total_seconds"] = time.perf_counter() - t_start
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, sort_keys=False)
        fh.write("\n")
    errors = [r for r in results["latency"] + results["sensitivity"] if r.get("error")]
    log(f"done in {results['total_seconds']:.0f}s -> {out_path}"
        + (f" ({len(errors)} runs with errors)" if errors else ""))
    return 1 if errors else 0


class _maybe_pg:
    """Context manager: (dsn, description) or (None, reason)."""

    def __init__(self, args, impls) -> None:
        self.args, self.impls, self.cm = args, impls, None

    def __enter__(self):
        if "postgres" not in self.impls:
            return None, "not requested"
        if self.args.pg == "off" and not self.args.pg_dsn:
            return None, "disabled (--pg off)"
        self.cm = postgres_server(dsn=self.args.pg_dsn, local=True)
        dsn, desc = self.cm.__enter__()
        if dsn is not None:
            err = check_dsn(dsn)
            if err:
                return None, err
            import psycopg
            with psycopg.connect(dsn) as con:
                desc = f"{con.execute('SELECT version()').fetchone()[0]}; {desc}"
        return dsn, desc

    def __exit__(self, *exc):
        if self.cm is not None:
            return self.cm.__exit__(*exc)
        return False


if __name__ == "__main__":
    sys.exit(main())
