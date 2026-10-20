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
  1. #23 meal@484200 [coffee,dairy] w=0.8         score=0.4528
      lag 1.50h -> kernel 0.845 x weight 0.8 x link 0.670 [noisy-OR of coffee+dairy=0.67 (joint link 0.83, a=5.80, b=1.20) inseparable]
      warning: 'coffee', 'dairy' always co-occur (6 events): feedback cannot separate their effects yet; reported as joint cause coffee+dairy
  2. #22 meal@478800 [beans,onion,rice] w=1.5     score=0.3495
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
| an incident / broken main  | merged PRs and deploys (AI-authored or not) |

They all come down to ranking recent events by:

1. **Timing.** Does the lag between event and effect fit how this effect
   usually lags its causes? This is a *lag kernel* per effect type.
2. **Dose.** How much of it was there (portion size, deploy size, amount)?
   This is the event *weight*: exposure magnitude, default 1.0.
3. **Plausibility.** Do this event's features tend to produce this effect?
   This is a *learned link strength* per (feature, effect), shared across
   entities through a global layer.
4. **Rules.** Optional soft domain rules ("small transactions are rarely
   fraud": ×0.3).

```
score(event | effect) = kernel(dt) × weight × noisy_OR_f( strength(f, effect) ) × Π rules
```

You almost never want *all* the scores. You want the best few, now, with
reasons, and you want the database to learn when you tell it which one was
right. HeapDB is built around exactly that query, and that is where the name
comes from: conceptually it is a priority queue of suspects that you pop from.

## Use case: AI-heavy code changes

When a large share of merged code is written by agents, more changes land
per day, and they get reviewed faster. HeapDB covers both halves of keeping
that safe:

1. **Attribution** (after the fact). An error spike, a failing main check or
   an incident fires: `top_causes(service, "incident", at, k=3, explain=True)`
   ranks the recently merged PRs by timing x dose (`log1p(lines_changed)`) x
   learned links of their features (`migration`, `lockfile`, `review:fast`,
   `path:payments/retry`, ...) x rules (`docs_only` x0). A revert is
   feedback: the culprit is confirmed and the cleared suspects are rejected.
2. **Priority** (before the fact). The same learned links score PRs that
   **have not merged yet**: `feature_risk(effect, features, ...)` and
   `rank_candidates(effect, open_prs, k)` give a pre-merge review queue,
   i.e. which open PRs deserve a careful human review. There is no kernel
   term, because nothing has happened yet.

```python
from heapdb import Candidate
queue = db.rank_candidates("incident", [
    Candidate("PR-1201", {"path:payments/retry", "lockfile", "dep_bump", "ai_authored"},
              type="deploy", attrs={"lines_changed": 60}, entity="payments"),
    Candidate("PR-1203", {"docs_only"}, type="deploy", attrs={"lines_changed": 40},
              entity="search"),          # dropped: the docs_only rule scores it 0
], k=10, explain=True)
for row in queue:
    print(row.rank, row.candidate.id, round(row.score, 3), row.explanation)
```

`examples/code_changes.py` simulates a week of this across three services,
with a hidden ground truth: dangerous feature combinations, plus an
`ai_authored` feature that only correlates with risky traits. It prints
attribution accuracy (first vs second half of the week), learned links,
proportional vs uniform credit for the confounder, calibration, and the
pre-merge queue next to the hidden true risk.

## Quickstart

Requires Python 3.10+. No runtime dependencies.

```bash
pip install -e ".[test]"     # or just put this directory on PYTHONPATH
pytest
python examples/bloating.py
python examples/incident_rca.py
python examples/code_changes.py
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
lunch = db.insert(entity="me", time=13 * H, type="meal", features={"beans", "onion"},
                  weight=1.5)                            # weight = dose (portion)

res = db.top_causes(entity="me", effect="bloating", at=15 * H, k=3, explain=True)
for c in res:
    print(c.rank, c.event.short(), round(c.score, 3))
    print("   ", c.explanation)
print(res.stats)            # segments scanned / pruned

# Tell HeapDB what really happened. Confirmed features share the credit in
# proportion to their strength (EM for noisy-OR); rejected ones get beta += 1.
rep = db.feedback("me", "bloating", at_time=15 * H, confirmed_event_id=lunch.id)
print(rep.credit)           # {'beans': 0.9375, 'onion': 0.25}

print(db.calibration_report("bloating"))

# Pre-event risk of something that has not happened yet (no timing term).
print(db.feature_risk("bloating", {"beans", "rice"}, weight=1.0, entity="me"))

db.save("me.jsonl")
db2 = HeapDB.load("me.jsonl")
```

### Query language

```
TOP <k> CAUSES OF <effect> FOR <entity>[,<entity>...] [AT <time>] [WITHIN <duration>] [EXPLAIN]
FEEDBACK <effect> FOR <entity> AT <time> CONFIRMED <id> [REJECTED <id>,<id>...]
INSERT <type> FOR <entity> [AT <time>] [FEATURES <f>,<f>...] [WEIGHT <dose>] [ATTRS k=v,...]
CALIBRATION OF <effect> [BINS <n>]
RISK OF <effect> FOR <entity> FEATURES <f>,<f>... [WEIGHT <w>] [TYPE <t>] [ATTRS k=v,...] [EXPLAIN]
```

`FOR a,b,c` with several entities runs a group query. `<time>` can be
seconds, `now`, `now-2h` or ISO-8601. `<duration>` can be `90`, `30m`,
`12h` or `2d`. `RISK OF` scores a candidate that has not happened yet (see
`feature_risk` below). Without `WEIGHT`, the dose comes from `ATTRS` via the
type's weight function, else 1.0. The REPL (`python -m heapdb`) also has meta commands:
`.effect`, `.prior`, `.rules`, `.forget`, `.events`, `.save`, `.load` and
`.help`. Rules and weight functions are Python callables, so you register
them through the API, not the query language.

## Architecture

```
heapdb/
  events.py         Event record (entity, time, type, features, weight, id)
  kernels.py        Lag kernels: ExponentialDecay, WindowKernel, LogNormalKernel, GammaKernel
  links.py          Hierarchical Beta links (prior / global / entity), EM credit, joint links
  segments.py       Per-entity, time-bucketed segments with summaries + inverted index
  store.py          HeapDB: insert, top-k + group queries with pruning, rules, explain,
                    feature_risk / rank_candidates, feedback, calibration,
                    forget_entity, snapshots
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
- the distinct `(event type, feature set)` keys with the max weight per key
  (up to `max_sets_per_segment`, default 64),
- an inverted index `feature -> [event ids]`.

None of the summaries contain link strengths, so they never go stale as
links are learned.

### Query: threshold-style top-k with segment pruning

For `top_causes(entity, effect, at, k)`:

1. **Window.** Only segments overlapping `[at - kernel.support, at - kernel.min_lag]`
   can matter. Everything older is never touched.
2. **Bound.** For each segment, compute an upper bound on the score of
   *any* of its events from the summaries and the *current* links:

   ```
   bound = kernel.max_over(lag range of segment)          # exact for unimodal kernels
         × max over distinct (type, feature set) keys of the segment:
               max_weight(key) × noisy_OR(strengths of key's features) × rule_bound(type)
   ```

   Each factor bounds the matching factor of every event with that key, so
   the max over keys bounds every event in the segment. `rule_bound(type)` is
   the product of the type-level rule multipliers. Predicate rules are opaque,
   so the bound counts them as 1. If a segment has more than
   `max_sets_per_segment` distinct keys, it falls back to the v0.1 union
   bound, `max_weight × noisy_OR(union of features) × max rule_bound`, which
   is looser but still valid because noisy-OR is monotone in its feature set.

   *Why not the max single link?* The noisy-OR of a set is **at least** its
   largest member's strength, so max-link would *under*-estimate and prune
   real answers. The per-set bound is tight because it never pairs a heavy
   weight with a strong link that belong to different events.
3. **Scan best-first and stop early.** Visit segments in descending bound
   order and push scored events into a bounded min-heap of size *k*. Once the
   heap is full and its minimum is `>=` the next segment's bound, nothing left
   can enter the top-k, so we stop. The rest are counted as pruned in
   `QueryResult.stats`.

The answer is **exact**. `tests/test_topk.py` checks it against brute force
on random data across all kernels, 12 seeds, `WITHIN` caps, type filters,
soft and hard rules, the union fallback, hierarchical links and interleaved
feedback. It also checks bound ≥ score directly for every event. In
`examples/incident_rca.py`, a week of noisy changes (234 events, 203
segments) is answered by scoring 5 events.

On a sparse-strong dataset (each 10-minute segment has ten heavy-dose weak
events and one light-dose strong one; `test_set_bound_prunes_more_than_union_bound`),
top-3 over 151 segments in window gives:

| bound                  | segments pruned | events scored |
|------------------------|-----------------|---------------|
| v0.1 union bound       | 15 / 151 (10%)  | 1512          |
| v0.2 per-set bound     | 149 / 151 (99%) | 24            |

### Cross-entity group queries

`db.top_causes_group(["checkout", "payments", "search"], "latency_spike", at, k)`
runs the per-entity top-k and k-way merges them with `heapq.merge`. Each
cause carries its `entity`. The result is exact with respect to brute force
over the union of the entities' events. A group query takes exactly one
effect: mixing effects raises, because their scores are on different scales.
**Scores are comparable across entities only to the extent that they are
calibrated.** Entities with more feedback have sharper links, and doses may
be on different scales, so treat cross-entity ranking as a triage aid until
`calibration_report` says otherwise.

### Learning: hierarchical Beta links and proportional credit

Each `(feature, effect)` link is a `Beta(alpha, beta)` belief whose mean
`alpha / (alpha + beta)` is the link strength. Counts are floats.

**Three layers.** An entity's belief combines:

```
alpha = prior_alpha + c·G_pos + own_pos        c = min(1, global_pseudocount / (G_pos + G_neg))
beta  = prior_beta  + c·G_neg + own_neg
```

- the **registered prior**: domain knowledge, or the effect's `default_prior`
  (default `Beta(1, 4)`);
- the **global layer** `G`: evidence pooled over all *other* entities
  (leave-one-out, so an entity's own evidence is never counted twice),
  shrunk to at most `global_pseudocount` (default 5) pseudo-observations;
- the entity's **own** evidence.

A brand-new entity gets sensible answers from population data, and its own
evidence dominates after a handful of observations. Every feedback update
goes to both the entity and the global layer. Use
`register_effect(..., hierarchical=False)` to stop entities borrowing for
that effect, or set `global_pseudocount` to change the cap.

**Proportional credit (EM for noisy-OR).** When an event with features `F` is
confirmed, each feature gets the share of the effect it is responsible for:

```
r_f = s_f / (1 − Π_{g∈F} (1 − s_g))    (clamped to [0, 1])
alpha_f += r_f        beta_f += 1 − r_f
```

A feature that already explains the effect takes most of the credit. A
weak rider that merely co-occurs takes little, instead of being inflated
alongside it (as with v0.1's `alpha += 1` for every feature). **Rejections**
still add `beta_f += 1` to every feature: under noisy-OR, the event failing
to cause the effect means every feature independently failed.
`credit="uniform"` (per effect, or per `feedback()` call) restores the v0.1
update. `feedback()` returns the credit given to each feature.

**Joint causes.** Co-occurrence is still tracked: features whose entity-wide
posting lists are identical have *always* co-occurred, so feedback cannot
separate them ("was it the coffee or the milk?"). When such a group has
received feedback, the explanation presents it as **one joint cause**
(`coffee+dairy`) with an `inseparable` flag and a warning. For every
inseparable set that actually appeared on a confirmed or rejected event,
HeapDB keeps a joint Beta link, keyed by `frozenset` and created lazily. On
confirmation the set gets credit `noisyOR(set) / noisyOR(event)`. On
rejection it gets `beta += 1`. Joint links are used only for reporting:
scoring stays per-feature noisy-OR. After the features have been seen apart,
the joint link is still shown, marked as now separable.

### Weight semantics: dose / exposure

`weight` is the **magnitude of exposure**: portion size, deploy size, amount
moved. It defaults to 1.0 and must be finite and > 0. **Propensity** ("deploys
are riskier than flag flips") does not belong in the weight. It belongs in
links (learned) or rules (declared). Weights can be derived from event
attributes:

```python
db.register_event_type("meal",   weight=lambda a: a.get("portion", 1.0))
db.register_event_type("deploy", weight=lambda a: math.log1p(a["lines_changed"]))
db.insert("svc", t, "deploy", {"orm_upgrade"}, attrs={"lines_changed": 420})
```

An explicit `weight=` always wins over the type's weight function.

### Plausibility rules

```python
db.add_rule("fraud", "small transaction",
            predicate=lambda e: e.attrs.get("amount", 0) < 10, multiplier=0.3)
db.add_rule("latency_spike", "flags are gated", predicate=None, multiplier=0.5,
            types={"flag"})                      # type-level rule
```

Matching events get `score *= multiplier`. The multiplier must be in
`[0, 1]`: 0 is a hard block, and values above 1 are rejected so that segment
bounds stay valid. Every applied rule appears in the explanation
(`x rules small transaction x0.3`). `cause_types` remains the type-level
hard filter. As a temporal sanity check, every kernel returns 0 for negative
lag, and `score()` also returns 0 for any event after the effect.

### Pre-event risk: `feature_risk` and `rank_candidates`

```python
db.feature_risk(effect, features, weight=1.0, entity=None, type=None, attrs=None,
                explain=False)                   # -> score, or (score, Explanation)
db.rank_candidates(effect, candidates, k=10, entity=None, explain=False)
                                                 # -> [RankedCandidate], best first
```

```
risk = weight × noisy_OR_f( strength_entity(f, effect) ) × Π rules
```

This is `score` without the kernel. For any stored event `e`,
`score(e, effect, at) == kernel(at − e.time) × feature_risk(effect, e.features,
e.weight, e.entity, e.type, e.attrs)`; a property test checks this across all
kernels. The details:

- **Links.** Links are entity-aware and use the same hierarchical layers as
  queries. `entity=None` uses the global (population) view.
- **Rules.** Rules are evaluated against a transient, unstored `Event`
  (`id=0`, `time=nan`) built from `type`/`features`/`attrs`. Predicates that
  look at `time` will not match.
- **Type.** If `type` is given, `cause_types` applies (a disallowed type
  scores 0 and the explanation says why), and type-scoped rules can match.
  `weight=None` derives the dose from `attrs` via the type's weight function.
- **Explanation.** `explain=True` returns the same `Explanation` structure
  with `prospective=True` and `kernel_value=1`: per-feature links, applied
  rules, and joint causes / inseparable warnings from the entity's history.
- **Ranking.** `rank_candidates` takes `Candidate` objects or dicts (`id`,
  `features`, `weight`, `type`, `attrs`, `entity`). A candidate's own
  `entity` overrides the argument, so one call can rank PRs across services,
  with the same calibration caveat as group queries. It keeps a bounded
  min-heap of size *k* in one pass, drops zero scores, keeps the earlier
  candidate on ties, and builds explanations only for the returned rows.

### Calibration tracking

Each `feedback()` call records, for every judged event, the score it had
**before** the update, together with the outcome (confirmed or rejected).
`db.calibration_report(effect, bins=10)` (or `CALIBRATION OF effect`)
returns, per equal-width score bin, the count, the mean predicted score and
the observed confirm rate, plus the overall Brier score. Scores are clipped
to [0, 1] for this, since a heavy dose can push a score above 1. There is no
automatic recalibration yet (see Roadmap).

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

### Persistence

`db.save(path)` writes a JSONL snapshot (format version 2). It contains the
meta line, effects (kernel, priors, global layer, per-entity float counts,
joint links, credit mode, rule metadata), events (with `attrs`) and
calibration records. `HeapDB.load` reads both v1 and v2 snapshots.

**Python callables are not serialised.** Type-level rules (`predicate=None`)
are restored automatically. Predicate rules and event-type weight functions
keep only their names: after loading, they are listed in
`db.effect(e).pending_rules` and `db.pending_event_types`, and you must
re-register them with `add_rule` and `register_event_type` under the same
names. Stored weights are kept as they were, so scores of existing events
do not change.

## Privacy

`db.forget_entity(entity)` removes the entity's events, segments,
entity-level and joint links, and calibration records. By default its past
contributions to the **global** link layer stay, because they are aggregated
counts. With `scrub_global=True` they are subtracted as well. That
subtraction is exact because the entity layer is a ledger of exactly what the
entity added to the global layer. With proportional credit, *other*
entities' updates may still have been shaped slightly by strengths they
borrowed from the forgotten entity. The guarantee is that the forgotten
entity's own counts are removed.

Design direction (**not implemented** beyond `forget_entity`):

- **Per-entity encryption keys and crypto-shredding.** Encrypt each entity's
  events and links under its own key. Forgetting then means destroying the
  key, which also covers backups and snapshots.
- **Audit logs.** An append-only record of who queried, gave feedback on or
  forgot which entity, and when.
- **On-device by default for health data.** Personal logs (meals, symptoms)
  stay on the user's device. Only opt-in, aggregated link updates leave it.
- **Differential privacy for shared global stats.** Add calibrated noise to
  the global-layer updates contributed by each entity (and bound each
  entity's contribution), so the population prior cannot be used to infer
  any one entity's history.

## Changelog: v0.3

- **Pre-event risk.** `feature_risk(effect, features, weight=1.0, entity=None,
  type=None, attrs=None, explain=False)` scores a candidate that is not (yet)
  an event: dose × noisy-OR of the current entity-aware links × rules, with
  no kernel term. This is the "priority" half of attribution, e.g. ranking
  open PRs before merge. `rank_candidates(effect, candidates, k, entity=None,
  explain=False)` returns the heap-based top-k as `RankedCandidate`s. The new
  types are `Candidate`, `RankedCandidate` and `RiskResult`.
- **Explanation.** `Explanation` gained `prospective` (True for risk
  explanations, which print `no lag (pre-event)` instead of the lag/kernel
  part).
- **Query language.** `RISK OF <effect> FOR <entity> FEATURES a,b [WEIGHT w]
  [TYPE t] [ATTRS k=v,...] [EXPLAIN]`, also listed in the REPL `.help`.
- **New example.** `examples/code_changes.py` covers AI-heavy code changes:
  a seeded week of ~150 PRs on three services, three effects with
  different kernels, revert feedback, hit@1/hit@3 learning curves,
  proportional vs uniform credit on an `ai_authored` confounder, calibration,
  a group query and a pre-merge review queue.
- **Tests.** Monotonicity in link strength and dose, rules reducing risk,
  equality with `score / kernel` for inserted events (property test across
  kernels), top-k equal to a full sort, and query-language parsing and
  execution.

## Changelog: v0.2

- **Proportional credit.** Confirmations use the noisy-OR EM update
  (`alpha_f += r_f`, `beta_f += 1 − r_f`); rejections still give
  `beta_f += 1`. Counts are floats. `credit="uniform"` restores v0.1.
  `feedback()` returns a `FeedbackReport` with each feature's credit and
  validates all ids before changing anything.
- **Joint causes.** Always-co-occurring features with feedback are reported
  as one joint cause (`beans+onion`) with a lazily created joint Beta link
  (bounded to sets that appeared on feedback events) and an `inseparable`
  flag. Scoring is unchanged.
- **Hierarchical priors.** Registered prior + leave-one-out global evidence
  capped at `global_pseudocount` (default 5) + the entity's own counts. This
  can be switched off per effect (`hierarchical=False`). New:
  `db.link(effect, feature, entity=None)`.
- **Tighter, still exact bounds.** Per-segment distinct `(type, feature set)`
  keys with max weight. The bound is the max over keys, with a union
  fallback above `max_sets_per_segment` (64). On the sparse-strong benchmark
  it prunes 99% of segments versus 10% for the union bound.
- **Weight = dose.** Weight must be > 0. Events have optional `attrs`, and
  `register_event_type(type, weight=fn)` derives weight from them.
- **Rules.** `add_rule(effect, name, predicate, multiplier ∈ [0, 1], types=None)`.
  Rules are applied in score, listed in explanations, and type-level rules
  tighten bounds. Tests check that kernels return 0 for negative lag.
- **Group queries.** `top_causes_group(entities, effect, at, k)` uses a k-way
  merge, works on a single effect only, and puts the entity on every cause.
  The query language gains `TOP k CAUSES OF e FOR a,b,c`.
- **Calibration.** Per-effect (score, outcome) records,
  `calibration_report(effect, bins)` with the Brier score, and
  `CALIBRATION OF e [BINS n]`.
- **Privacy.** `forget_entity(entity, scrub_global=False)`, plus the REPL
  commands `.forget` and `.rules`.
- **Snapshots v2.** These round-trip all of the new state. Callables are
  re-registered by name after load.
- **Breaking changes.** `weight=0` is now rejected. `LinkTable.get()` returns
  an immutable view, so use `db.link(...)`. `BetaCounts` gained
  `global_pos`/`global_neg`. `Explanation` gained `rules`, `joint_causes` and
  `entity`.

## Roadmap

Ideas, not promises:

- [ ] Benchmarks and guidance for choosing `bucket_seconds`; multi-resolution
      (hierarchical) segment summaries for long lookbacks
- [x] ~~Per-feature max-weight summaries for tighter bounds~~: per-set bounds in v0.2
- [ ] Automatic recalibration (Platt / isotonic) from the calibration records,
      so scores can be read as probabilities and compared across entities
- [ ] Privacy: per-entity keys with crypto-shredding, audit log, DP for the
      global layer (see Privacy)
- [ ] Learning kernel parameters (lag distributions) from feedback, not just links
- [ ] Better confounding handling: partial co-occurrence, lift, suggested
      "elimination experiments" (v0.2 has exact co-occurrence + joint links)
- [ ] Negative / protective links, and effect-specific cause type hierarchies
- [ ] Interaction features learned automatically (today a harmful combination such
      as `migration+review:fast` is learned only through its members unless you
      add a cross feature yourself)
- [ ] Incremental "standing queries" built on forward decay for exponential kernels
- [ ] Durable storage: write-ahead log, compaction, segment files
- [ ] A Rust/Go core with this implementation kept as the executable spec

## Contributing

Contributions, issues and design discussion are very welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) to get started. Good first issues include
new kernels, benchmarks and query-language features.

## License

Apache License 2.0. See [LICENSE](LICENSE).
