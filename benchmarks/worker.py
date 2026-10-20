"""One (size, implementation) latency/ingest/memory measurement.

Run in a fresh subprocess by :mod:`benchmarks.run` so that RSS deltas are
not polluted by earlier runs::

    python -m benchmarks.worker <config.json> <out.json>

Steps: generate the workload -> build (timed ingest + index) -> RSS delta ->
then for each kernel: **verify** top-k on a sample of queries against an
independent reference (:class:`benchmarks.workload.Reference`; abort on any
mismatch, before timing) -> warm-up -> timed queries.
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
import time
from typing import Any

from .impls import compare, make_impl
from .workload import Reference, WorkloadSpec, generate, make_kernel, queries


def rss() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation): an observed value."""
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(-(-p * len(s) // 100)) - 1))
    return s[i]


def summarize_ms(lat: list[float]) -> dict[str, float]:
    ms = [x * 1000.0 for x in lat]
    return {"p50_ms": percentile(ms, 50), "p95_ms": percentile(ms, 95),
            "p99_ms": percentile(ms, 99), "mean_ms": statistics.fmean(ms),
            "max_ms": max(ms)}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    spec = WorkloadSpec(**cfg["workload"])
    kernel_names: list[str] = cfg["kernels"]
    k = cfg["k"]
    out: dict[str, Any] = {"impl": cfg["impl"], "n_events": spec.n_events,
                           "bucket_seconds": cfg.get("bucket"), "links": spec.links, "k": k}
    gc.collect()
    r0 = rss()
    t0 = time.perf_counter()
    wl = generate(spec)
    out["generate_seconds"] = time.perf_counter() - t0
    out["n_feature_rows"] = sum(len(f) for f in wl.features)
    gc.collect()
    r1 = rss()
    opts: dict[str, Any] = {}
    if cfg.get("bucket"):
        opts["bucket"] = cfg["bucket"]
    if cfg.get("dsn"):
        opts["dsn"] = cfg["dsn"]
    impl = make_impl(cfg["impl"], wl, make_kernel(kernel_names[0]), **opts)
    impl.build()
    gc.collect()
    r2 = rss()
    out["ingest_seconds"] = impl.ingest_seconds
    out["ingest_events_per_s"] = spec.n_events / impl.ingest_seconds
    out["index_seconds"] = impl.index_seconds
    out["index_note"] = impl.index_note
    out["rss_workload_bytes"] = None if r0 is None else r1 - r0
    out["rss_delta_bytes"] = None if r1 is None else r2 - r1
    out["rss_after_build_bytes"] = r2
    out["info"] = impl.info()

    reference = Reference(wl)
    out["runs"] = []
    try:
        for kn in kernel_names:
            kernel = make_kernel(kn)
            impl.set_kernel(kn, kernel)
            out["runs"].append(_measure(cfg, wl, impl, reference, kn, kernel, k))
            if "error" in out["runs"][-1]:
                out["error"] = f"{kn}: {out['runs'][-1]['error']}"
                break
    finally:
        impl.close()
    return out


def _measure(cfg, wl, impl, reference, kn, kernel, k) -> dict[str, Any]:
    spec = wl.spec
    res: dict[str, Any] = {"kernel": kn}
    # -- verification (before any timing) ----------------------------------
    vq = queries(wl, cfg["n_verify"], seed=spec.seed + 101)
    mism = []
    nonempty = 0
    for i, (e, a) in enumerate(vq):
        ref = reference.topk(kernel, e, a, k)
        got = impl.query(e, a, k)
        nonempty += bool(ref)
        d = compare(ref, got, cfg["tol"])
        if d:
            mism.append(f"query {i} (entity {e}, at {a!r}): {d}")
    res["verified_queries"] = len(vq)
    res["verified_nonempty"] = nonempty
    if mism:
        res["error"] = f"{len(mism)} of {len(vq)} verification queries disagree with the reference"
        res["verify_mismatches"] = mism[:5]
        return res

    # -- timing ------------------------------------------------------------
    qs = queries(wl, cfg["n_warmup"] + cfg["n_queries"], seed=spec.seed + 202)
    for e, a in qs[:cfg["n_warmup"]]:
        impl.query(e, a, k)
    lat: list[float] = []
    scored: list[int] = []
    in_win_seg: list[int] = []
    pruned: list[int] = []
    window_events: list[int] = []
    pc = time.perf_counter
    for e, a in qs[cfg["n_warmup"]:]:
        t = pc()
        impl.query(e, a, k)
        lat.append(pc() - t)
        st = getattr(impl, "last_stats", None)
        if st is not None:
            scored.append(st.events_scanned)
            in_win_seg.append(st.segments_in_window)
            pruned.append(st.segments_pruned)
        if getattr(impl, "last_window", None) is not None:
            window_events.append(impl.last_window)
    res["n_queries"] = len(lat)
    res["n_warmup"] = cfg["n_warmup"]
    res["latency"] = summarize_ms(lat)
    if scored:
        res["events_scored_mean"] = statistics.fmean(scored)
        res["events_scored_p50"] = percentile(scored, 50)
        res["events_scored_p99"] = percentile(scored, 99)
        res["segments_in_window_mean"] = statistics.fmean(in_win_seg)
        res["segments_pruned_mean"] = statistics.fmean(pruned)
        tot = sum(in_win_seg)
        res["segments_pruned_fraction"] = (sum(pruned) / tot) if tot else None
    if window_events:
        res["events_in_window_mean"] = statistics.fmean(window_events)
    return res


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        cfg = json.load(fh)
    try:
        res = run(cfg)
    except Exception as e:  # report, don't hide
        import traceback
        res = {"impl": cfg.get("impl"), "error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()[-2000:]}
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        json.dump(res, fh)


if __name__ == "__main__":
    main()
