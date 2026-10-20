## Reading these results

This section is hand-written (`benchmarks/reading.md`) and refers to the
tables above. If you rerun the benchmarks, re-check it against the new tables.

**Where heapdb wins**

* **Work per query.** Pruning does what it claims. At 1M events with the
  default kernel, brute force scores the 459.0 events in the window on average;
  heapdb scores 19.2 and skips 96.6% of the segments in the window. With the
  wide log-normal kernel (5,055.2 events in window), heapdb scores 19.2. The
  count of events scored stays roughly flat as N grows, while the window
  grows linearly.
* **PostgreSQL**, as set up here (one query per effect over a Unix socket,
  from Python), is slower than every in-process implementation at every
  size, and heapdb's p50 is lower than PostgreSQL's at 10K, 100K and 1M.
  Most of that gap is likely per-query client/server and executor overhead,
  not scoring, so the comparison mainly says that a network round trip costs
  more than a small in-process query.
* **Ranking quality.** On the code-changes simulation, learned links with
  feedback rank the true culprit first more often than recency (hit@1 over
  the whole week 0.562 vs 0.367; per seed, better in 17 of 20 second halves).

**Where heapdb loses**

* **Latency against brute force and SQLite, at every size measured.** Pruning
  saves scoring work, but in this pure-Python implementation it does not
  save wall time: bounding each segment costs Python work too (summaries,
  noisy-OR per distinct feature set, a sort). At 10K the window holds 4.6
  events, so there is nothing to prune and heapdb is about 3x slower than
  brute force (p50 0.074 vs 0.022 ms). At 1M it is still slower than brute
  force and SQLite in the latency table (p50 1.98 ms vs 1.75 and 1.55 ms).
* **Run-to-run noise is about as large as the gap.** The same configuration
  (1M, `lognormal_3h`, 15 min buckets, skewed links) was measured twice in
  this run, in separate processes: in the latency section heapdb 1.98 /
  bruteforce 1.75 ms (heapdb slower), in the sensitivity section heapdb 1.87
  / bruteforce 2.30 ms (heapdb faster). Treat heapdb vs brute force at 1M as
  roughly a tie on this machine. SQLite in-memory is the fastest or tied for
  fastest at 1M in both sections.
* **Flat window kernel with uninformative links.** With every link equal and
  a 24h flat window, 70.6% of segments are pruned, heapdb scores 117.9 events,
  and its p50 is 2.03 ms against 1.33 ms for brute force and 0.977 ms for
  SQLite (brute force / heapdb = 0.65x). With the short exponential kernel
  (`exp_10m`, 23.3 events in window), heapdb is also slower (0.69x).
* **Ingest and memory.** heapdb ingests 39,343 events/s at 1M, against
  242,207 for SQLite in-memory (bulk `executemany`) and 135,974 for the
  brute-force Event lists. It uses 1,501 B/event of RSS, against 652 for
  brute force (the same Event objects without segments) and 131 for SQLite.
  That is why 10M was **not measured** for heapdb or brute force here:
  extrapolating the measured bytes/event gives 15.8 GiB and 7.9 GiB, above
  the free RAM of this 7.8 GiB machine. SQLite and PostgreSQL were measured at
  10M.
* **Bucket size is a trade-off, not a free knob.** Smaller buckets prune more
  (5 min: 10.0 events scored at 1M) but create more segments to bound, which
  costs time and memory (592,037 segments, 1,779 MiB). In this run, 60 min
  buckets had the lowest p50 for `lognormal_3h`, `exp_6h` and
  `lognormal_wide` even though they score the most events; 15 min was best
  for `exp_10m` and `lognormal_narrow`, and 5 min for `window_24h`.
* **Ranking-quality gains from feedback are small.** Most of the
  improvement over recency comes from the kernel and dose (`kernel_only`) and
  from noisy-OR over feature sets (`priors_only`), not from learning. Adding
  feedback moves whole-week hit@1 from 0.536 (priors only) to 0.562, which
  is well inside one standard deviation across seeds (± 0.076). Proportional
  credit beats uniform credit by a similarly small margin (0.562 vs 0.540).
  Recency beat HeapDB with feedback in 2 of 20 seeds (second half, hit@1).
  The simulation is one week (~42 attributable effects per seed), so there is
  little feedback to learn from.

**Caveats**

* **Single machine, small VM.** 2 logical CPUs, 7.8 GiB RAM, shared
  virtualised hardware. Everything is single-threaded. Absolute numbers will
  differ elsewhere; the relative picture may too.
* **Synthetic data.** Uniform event times, Zipf features, independent link
  strengths, a fixed seed. Real logs are bursty (deploy trains, meal times),
  which changes events per segment and pruning. Pruning helps most when a
  few strong events dominate; that is built into the "skewed" profile.
* **Pure Python vs C.** heapdb and brute force are interpreted Python;
  SQLite and PostgreSQL run their scoring in C. Brute force uses exactly the
  same scoring function as heapdb (`HeapDB.score`), so heapdb vs brute force
  isolates the search strategy. heapdb vs SQL mostly compares Python against
  C, not segment pruning against a B-tree.
* **In-memory vs on-disk.** heapdb is in-memory only. `sqlite-disk` and
  PostgreSQL write to disk but are queried warm (page cache / shared
  buffers), so these are not cold-cache numbers. PostgreSQL's server memory is
  not in the RSS column (relation sizes are shown instead).
* **Static links during latency runs.** No feedback is applied while timing,
  so heapdb's link cache is warm. Feedback clears it, which would add cost.
* **Quality uses the example's own simulator**, whose lag distributions match
  the kernels HeapDB is configured with. That favours kernel-based methods
  over recency more than mismatched real-world lags would.
