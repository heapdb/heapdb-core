## Reading these results

This section is hand-written (`benchmarks/reading.md`) and refers to the
tables above. If you rerun the benchmarks, re-check it against the new tables.

**Where heapdb wins**

* **Against the pure-Python engine (v0.4).** Measured on the same CPU, the
  Rust engine's p50 is 8.5x lower at 10K, 14.3x at 100K and 19.4x at 1M,
  and 13x-22x lower across the six kernels at 1M. It ingests 4.3x-8.3x
  faster and uses 298 B/event of RSS at 1M instead of 1,502. It scores
  exactly the same number of events per query as the Python engine (same
  segments, same bounds, same pruning), so the gain is the cost per unit of
  work, not a different algorithm. That is what made 10M measurable on this
  7.8 GiB machine (1,701 MiB RSS); the Python engine was not measured at 10M.
* **Against SQLite and PostgreSQL, at every size.** heapdb's p50 is 2.3x
  lower than SQLite in-memory at 10K, 7.1x at 100K, 16x at 1M and 36x at
  10M; against PostgreSQL 11x, 20x, 38x and 124x. The SQL implementations
  have to score every event in the window (459 at 1M), so their latency
  grows with the window, while heapdb scores 19.2 events at 1M and 149.6 at
  10M. Part of the gap at small sizes is per-query overhead (statement
  execution, and for PostgreSQL a Unix-socket round trip), not scoring.
* **Against brute force with the same Rust scoring, at large N.**
  `bruteforce-rust` uses the same engine and scoring code and only skips the
  segment bounds. heapdb's p50 is 1.10x lower at 1M and 1.54x lower at 10M,
  and 1.19x-1.32x lower for four of the six kernels at 1M. Pruning skips
  96.6% (1M) and 96.8% (10M) of the segments in the window.
* **Ranking quality** is unchanged from the Python engine (identical numbers
  in every cell). On the code-changes simulation, learned links with
  feedback rank the true culprit first more often than recency (hit@1 over
  the whole week 0.562 vs 0.367; per seed, better in 17 of 20 second halves).

**Where heapdb loses**

* **Small N, against brute force.** At 10K the window holds 4.6 events and
  only 13% of segments can be pruned; heapdb's p50 is 0.011 ms against
  0.005 ms for both brute-force implementations (0.43x vs `bruteforce-rust`,
  0.49x vs `bruteforce`). At 100K it is still slower than `bruteforce-rust`
  (0.026 vs 0.021 ms, 0.81x). Bounding and ordering segments costs more than
  scoring a few dozen events directly. At these sizes all in-process
  implementations are within a few microseconds, which is the same order as
  the Python call and result-object overhead included in every number.
* **The pruning gain over Rust brute force is modest.** Most of heapdb's
  speed-up over the old engine and over SQL is the Rust scoring loop, not
  pruning: with the same scoring code, brute force is only 1.10x slower at
  1M and 1.54x slower at 10M. A user who only needs top-k over a short
  window could get most of the latency by scanning.
* **Short or uninformative kernels.** With `exp_10m` (23.3 events in the
  window) heapdb is slower than `bruteforce-rust` (0.015 vs 0.011 ms,
  0.72x), and with `lognormal_narrow` roughly tied (0.94x). With flat links
  and the flat 24h window, only feature counts separate segments, 70.6% are
  pruned, heapdb scores 117.9 events and ties brute force (0.073 vs 0.072 ms,
  0.98x).
* **Memory against SQLite.** heapdb (and `bruteforce-rust`, which uses the
  same store) needs 298 B/event at 1M and 178 B/event at 10M; SQLite
  in-memory needs 139 and 140. PostgreSQL's client RSS is not comparable
  (its data is in the server: 3,462 MiB of relations at 10M).
* **Ingest against the Rust brute-force baseline** is 7-21% lower at every
  size, although both build exactly the same store through the same insert
  path; this is run-to-run noise of separate worker processes, not a real
  difference, and shows how noisy the ingest column is. Ingest is
  1.7x-2.1x higher than SQLite and PostgreSQL bulk loads at 1M and 10M.
* **Bucket size is a trade-off, not a free knob.** 5 min buckets prune more
  (10.0 events scored at 1M) but create 592,037 segments (421 MiB) and are
  slower for every kernel; 60 min buckets score the most events but had the
  lowest p50 for five of the six kernels in this run (`exp_10m` was best at
  15 min).
* **Ranking-quality gains from feedback are small.** Most of the
  improvement over recency comes from the kernel and dose (`kernel_only`)
  and from noisy-OR over feature sets (`priors_only`), not from learning.
  Adding feedback moves whole-week hit@1 from 0.536 (priors only) to 0.562,
  well inside one standard deviation across seeds (± 0.076). Recency beat
  HeapDB with feedback in 2 of 20 seeds (second half, hit@1).

**Caveats**

* **Run-to-run noise.** The same configuration (1M, `lognormal_3h`, 15 min
  buckets, skewed links) was measured twice in this run, in separate
  processes: heapdb p50 0.107 ms in the latency section and 0.100 ms in the
  sensitivity section; `bruteforce-rust` 0.119 ms in both. Differences of
  10% or less between implementations should be treated as ties.
* **10M needed a larger memory headroom.** The run used
  `--mem-headroom 0.95` so that heapdb and `bruteforce-rust` (estimated from
  their 1M bytes/event) were allowed at 10M. The Python-scoring `bruteforce`
  (1,138 B/event of Python objects) was still skipped at 10M, so the 10M
  "events in window" column is empty.
* **Two Python-engine baselines.** The original Python-engine run
  (`2026-10-20-vm.json`) reports a different CPU (2.10 GHz) from this run
  (2.80 GHz), although it is the same VM. The Python engine was therefore
  re-run on this CPU (`2026-10-20-vm-python-rerun.json`, from the last
  pure-Python commit, without quality and PostgreSQL). On the faster CPU the
  Python engine was not faster (p50 2.08 vs 1.98 ms at 1M, 0.092 vs 0.074 ms
  at 10K), so the CPU label does not explain the speed-up; the numbers quoted
  above use the same-CPU re-run.
* **`bruteforce` changed meaning in v0.5.** It is still a Python loop, but
  each `score` call now goes into the Rust engine, so it is not comparable
  to the `bruteforce` rows of the Python-engine run.
* **Single machine, small VM.** 2 logical CPUs, 7.8 GiB RAM, shared
  virtualised hardware. Everything is single-threaded. Absolute numbers will
  differ elsewhere; the relative picture may too.
* **Synthetic data.** Uniform event times, Zipf features, independent link
  strengths, a fixed seed. Real logs are bursty (deploy trains, meal times),
  which changes events per segment and pruning. Pruning helps most when a
  few strong events dominate; that is built into the "skewed" profile.
* **Rust vs C, both called from Python.** heapdb and `bruteforce-rust` run
  their scoring in Rust; SQLite and PostgreSQL in C; all are driven from the
  same Python benchmark process, so every latency includes a Python call.
  heapdb vs `bruteforce-rust` isolates the search strategy; heapdb vs SQL
  compares pruning plus a specialised in-memory layout against a general
  B-tree query plan.
* **In-memory vs on-disk.** heapdb is in-memory only (snapshots are
  explicit). `sqlite-disk` and PostgreSQL write to disk but are queried warm
  (page cache / shared buffers), so these are not cold-cache numbers.
* **Static links during latency runs.** No feedback is applied while timing,
  so heapdb's link cache is warm. Feedback clears it, which would add cost.
* **Quality uses the example's own simulator**, whose lag distributions
  match the kernels HeapDB is configured with. That favours kernel-based
  methods over recency more than mismatched real-world lags would.
