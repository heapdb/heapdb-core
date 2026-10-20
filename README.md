# HeapDB

**Pop the most likely cause, right now.**

HeapDB is a database whose primary operation is not `SELECT` but *"given what
just happened, what most likely caused it?"*. You append events to a log; when
an effect shows up you ask for the top-k candidate causes. HeapDB ranks them
and explains each score.

```
TOP 3 CAUSES OF bloating FOR me AT now WITHIN 12h EXPLAIN
```

```
  1. #23 meal [coffee,dairy] w=0.8       score=0.5859
      lag 1.50h -> kernel 0.845 x weight 0.8 x link 0.867 [noisy-OR of dairy=0.67, coffee=0.60]
      warning: 'coffee', 'dairy' always co-occur (6 events): feedback cannot separate their effects yet
  2. #22 meal [beans,onion,rice] w=1.5   score=0.3495
  ...
```

> **Status: reference implementation / placeholder, not production.**
> This repository is a small, readable, pure-Python implementation of the
> HeapDB idea. It's meant for reading, experimenting and discussion. It is
> single-process and in-memory (with JSONL snapshots), and it has not been
> hardened, benchmarked or tuned. APIs will change.

## The idea: "pop the top"

Plenty of questions have the same shape:

| Effect                     | Candidate causes                        |
|----------------------------|-----------------------------------------|
| I feel bloated             | meals in the last ~12h                  |
| p99 latency spiked         | deploys, config changes, flag flips     |
| a user churned             | support tickets, outages, price changes |
| a sensor alarmed           | maintenance actions, weather events     |

They all come down to ranking recent events by:

1. **Timing.** Does the lag between event and effect fit how this effect
   usually lags its causes? This is a *lag kernel* per effect type.
2. **Magnitude.** How big was the event (dose, blast radius)? This is the
   event *weight*.
3. **Plausibility.** Do this event's features tend to produce this effect?
   This is a *learned link strength* per (feature, effect).

```
score(event | effect) = kernel(dt) × weight × noisy_OR_f( strength(f, effect) )
```

You almost never want *all* the scores. You want the best few, now, with
reasons, and you want the database to learn when you tell it which one was
right. HeapDB is built around exactly that query, and that is where the name
comes from: conceptually it is a priority queue of suspects that you pop from.

## Quickstart

Requires Python 3.10+. No runtime dependencies.

```bash
pip install -e ".[test]"     # or just put this directory on PYTHONPATH
pytest
python examples/bloating.py
python examples/incident_rca.py
python -m heapdb --demo      # REPL
```

```python
from heapdb import HeapDB, LogNormalKernel

H = 3600
db = HeapDB(bucket_seconds=900)
db.register_effect(
    "bloating",
    kernel=LogNormalKernel(median=2 * H, sigma=0.6),   # peaks ~1.4h, long tail
    priors={"beans": (3, 1), "dairy": (2, 1)},          # Beta(alpha, beta)
)

db.insert(entity="me", time=8 * H,  type="meal", features={"oats", "dairy"})
lunch = db.insert(entity="me", time=13 * H, type="meal", features={"beans", "onion"}, weight=1.5)

res = db.top_causes(entity="me", effect="bloating", at=15 * H, k=3, explain=True)
for c in res:
    print(c.rank, c.event.short(), round(c.score, 3))
    print("   ", c.explanation)
print(res.stats)            # segments scanned / pruned

# Tell HeapDB what really happened. Confirmed features get alpha += 1,
# rejected features get beta += 1.
db.feedback("me", "bloating", at_time=15 * H, confirmed_event_id=lunch.id, rejected_ids=[])

db.save("me.jsonl")
db2 = HeapDB.load("me.jsonl")
```

### Query language

```
TOP <k> CAUSES OF <effect> FOR <entity> [AT <time>] [WITHIN <duration>] [EXPLAIN]
FEEDBACK <effect> FOR <entity> AT <time> CONFIRMED <id> [REJECTED <id>,<id>...]
INSERT <type> FOR <entity> [AT <time>] [FEATURES <f>,<f>...] [WEIGHT <w>]
```

`<time>` can be seconds, `now`, `now-2h` or ISO-8601. `<duration>` can be
`90`, `30m`, `12h` or `2d`. The REPL (`python -m heapdb`) also has meta
commands: `.effect`, `.prior`, `.events`, `.save`, `.load`, `.help`.

## Architecture

```
heapdb/
  events.py         Event record (entity, time, type, features, weight, id)
  kernels.py        Lag kernels: ExponentialDecay, WindowKernel, LogNormalKernel, GammaKernel
  links.py          Beta(alpha, beta) link strengths + noisy-OR aggregation
  segments.py       Per-entity, time-bucketed segments with summaries + inverted index
  store.py          HeapDB: insert, top-k query with pruning, explain, feedback, snapshots
  forward_decay.py  ForwardDecayHeap: time-invariant ordering for exponential decay
  query.py          Query-language parser + executor
  cli.py            `python -m heapdb` REPL
```

### Storage: segmented append-only log

Each entity's events go into fixed-width **time buckets** (`bucket_seconds`).
A `Segment` stores its events plus cheap summaries that only ever grow:

- actual `min_time` / `max_time` of its events,
- `max_weight`,
- the union of features (with counts) and the set of event types,
- an inverted index `feature -> [event ids]`.

### Query: threshold-style top-k with segment pruning

For `top_causes(entity, effect, at, k)`:

1. **Window.** Only segments overlapping `[at - kernel.support, at - kernel.min_lag]`
   can matter. Everything older is never touched.
2. **Bound.** For each segment compute an upper bound on the score of *any*
   of its events, using only the summaries:

   ```
   bound = kernel.max_over(lag range of segment)      # exact for unimodal kernels
         × segment.max_weight
         × noisy_OR(strength(f) for f in segment's feature union)
   ```

   Every factor bounds the matching factor of every event in the segment.
   Noisy-OR is monotone in its feature set, so the union bounds every subset.
3. **Scan best-first and stop early.** Visit segments in descending bound
   order and push scored events into a bounded min-heap of size *k*. Once the
   heap is full and its minimum is `>=` the next segment's bound, nothing left
   can enter the top-k, so we stop. The rest are counted as pruned in
   `QueryResult.stats`.

The answer is **exact**: `tests/test_topk.py` checks it against brute force on
random data across all kernels, several seeds, `WITHIN` caps, type filters and
feedback. In `examples/incident_rca.py`, a week of noisy changes (219 events,
194 segments) is answered by scoring 5 events.

### Learning: Beta links and feedback

Each `(feature, effect)` link is a `Beta(alpha, beta)` belief whose mean
`alpha / (alpha + beta)` is the link strength. Priors encode domain knowledge;
features without one use the effect's `default_prior` (default `Beta(1, 4)`).
`feedback()` is plain Bayesian counting. A deliberately simple **confounding
check** warns in explanations when features of a candidate have *always*
co-occurred and feedback has touched them, because counting cannot tell them
apart. ("Was it the coffee or the milk?")

### ForwardDecayHeap: ordering that doesn't change with time

With an exponential kernel, `score(now) = w · exp(-λ(now - t))`. In log space:

```
log score(now) = [ log w + λ t ]  -  λ now
                   └─ key ──────┘    └─ same for every item ─┘
```

The key is fixed at insert, and the time-dependent term shifts every item
equally, so **a heap ordered by key is correctly ordered at every instant.**
No re-keying is needed as time passes. `ForwardDecayHeap` implements this with
`heapq` (`push`, `pop(now)`, `peek_top(now)`, `top_k`) and supports O(1)
deletion via tombstones that are cleaned up lazily. The RCA example uses it as
a live "top suspect" ticker and checks it against `top_causes` at several
times.

## Roadmap

Ideas, not promises:

- [ ] Benchmarks and guidance for choosing `bucket_seconds`; multi-resolution
      (hierarchical) segment summaries for long lookbacks
- [ ] Per-feature max-weight summaries for tighter bounds
- [ ] Learning kernel parameters (lag distributions) from feedback, not just links
- [ ] Better confounding handling: partial co-occurrence, lift, suggested
      "elimination experiments"
- [ ] Negative / protective links, and effect-specific cause type hierarchies
- [ ] Incremental "standing queries" built on forward decay for exponential kernels
- [ ] Durable storage: write-ahead log, compaction, segment files
- [ ] A Rust/Go core with this implementation kept as the executable spec

## Contributing

Contributions, issues and design discussion are very welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) to get started. Good first issues include
new kernels, benchmarks and query-language features.

## License

Apache License 2.0. See [LICENSE](LICENSE).
