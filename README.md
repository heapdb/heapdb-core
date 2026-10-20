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

> **Status: early, not production.** Since v0.5 the engine is written in
> Rust (`crates/heapdb`) and Python uses it through PyO3 bindings with the
> same `import heapdb` API as before. It is single-process and in-memory
> (with JSONL snapshots), and it has not been hardened. APIs may change.
> Benchmarks: see below.

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

### Install (Python)

```bash
pip install heapdb
```

Pre-built abi3 wheels on [PyPI](https://pypi.org/project/heapdb/) cover Linux
(x86_64, aarch64), macOS (Intel, Apple silicon) and Windows (x64) for CPython
>= 3.10, so no Rust toolchain is needed. The package has no runtime
dependencies.

To install the latest `main` from source instead
(`pip install git+https://github.com/heapdb/heapdb-core`), you need a **Rust
toolchain** (stable, >= 1.82; install with [rustup](https://rustup.rs)); the
build backend, [maturin](https://www.maturin.rs), is fetched by `pip`.

Releases are published from tags (`vX.Y.Z`) by CI with PyPI trusted
publishing; the tag must match the version in `pyproject.toml`.

### TypeScript client (preview)

`npm install @heapdb/client` connects to a hosted or self-hosted HeapDB server over
the Data API (`connect("heapdb://<key>@<host>/<db>")`). See
[clients/typescript](clients/typescript).

### Develop

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest
maturin develop --release    # build the extension into the venv
pytest                       # Python test suite
cargo test --workspace       # Rust unit, property and fixture tests
python examples/bloating.py
python examples/incident_rca.py
python examples/code_changes.py
python examples/product_analytics.py
python -m heapdb --demo      # REPL (also: `heapdb` script)
cargo run -p heapdb-cli -- --demo          # the same REPL as a native binary
cargo run -p heapdb-cli -- queries.hql     # run a query file
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

### Using the Rust crate directly

The engine is a plain Rust library (`crates/heapdb`) with no Python
dependency:

```rust
use std::sync::Arc;

use heapdb::{EffectOptions, EventRef, HeapDB, Kernel, NewEvent, TopK};

fn main() -> heapdb::Result<()> {
    let h = 3600.0;
    let mut db = HeapDB::new(900.0, 64)?; // bucket_seconds, max_sets_per_segment
    db.register_effect(
        "bloating",
        EffectOptions::new(Kernel::lognormal(2.0 * h, 0.6)?)
            .prior("beans", 3.0, 1.0)
            .prior("dairy", 2.0, 1.0),
    )?;
    // Soft rule: small portions are rarely the cause (predicate = any closure).
    let small = |e: &EventRef<'_>| e.weight < 0.5;
    db.add_rule("bloating", "small portion", Some(Arc::new(small)), 0.4, None)?;

    db.insert(NewEvent::new("me", 8.0 * h, "meal").features(["oats", "dairy"]))?;
    let lunch = db.insert(NewEvent::new("me", 13.0 * h, "meal").features(["beans", "onion"]).weight(1.5))?;

    let res = db.top_causes("me", "bloating", 15.0 * h, TopK::new(3).explain(true))?;
    println!("{res}"); // same text as format_result() in Python

    let rep = db.feedback("me", "bloating", 15.0 * h, Some(lunch.id), &[], None)?;
    println!("{:?}", rep.credit); // [("beans", 0.9375), ("onion", 0.25)]
    db.save("me.jsonl")?;
    Ok(())
}
```

The query language is available too (`heapdb::query::execute_str(&mut db,
"TOP 3 CAUSES OF bloating FOR me AT now EXPLAIN", now)`), as is the REPL
logic (`heapdb::repl::handle_line`).

### Query language

```
TOP <k> CAUSES OF <effect> FOR <entity>[,<entity>...] [AT <time>] [WITHIN <duration>] [EXPLAIN]
FEEDBACK <effect> FOR <entity> AT <time> CONFIRMED <id> [REJECTED <id>,<id>...]
INSERT <type> FOR <entity> [AT <time>] [FEATURES <f>,<f>...] [WEIGHT <dose>] [ATTRS k=v,...]
CALIBRATION OF <effect> [BINS <n>]
RISK OF <effect> FOR <entity> FEATURES <f>,<f>... [WEIGHT <w>] [TYPE <t>] [ATTRS k=v,...] [EXPLAIN]
OBSERVE <effect> FOR <entity> [AT <time>] [LEAK <x>]
SETTLE [<effect>] UNTIL <time> [FOR <entity>]
LINKS OF <effect> [FOR <entity>] [MIN <n>]
```

`FOR a,b,c` with several entities runs a group query. `<time>` can be
seconds, `now`, `now-2h` or ISO-8601. `<duration>` can be `90`, `30m`,
`12h` or `2d`. `RISK OF` scores a candidate that has not happened yet (see
`feature_risk` below). `OBSERVE`, `SETTLE` and `LINKS OF` are the
observational-learning statements (see below). Without `WEIGHT`, the dose comes from `ATTRS` via the
type's weight function, else 1.0. The REPL (`python -m heapdb`) also has meta commands:
`.effect`, `.prior`, `.rules`, `.forget`, `.events`, `.save`, `.load` and
`.help`. Rules and weight functions are callables (Python callables, or Rust
closures / trait objects), so you register them through the API, not the
query language.

## Architecture

A Cargo workspace (the engine and the CLI are pure Rust) plus a maturin
project for the Python package:

```
Cargo.toml, pyproject.toml   workspace; maturin build config for the `heapdb` Python package
crates/heapdb/               the engine (Rust library, no Python dependency)
  src/event.rs               Event, attrs (JSON values), EventRef (what rule predicates see)
  src/kernels.rs             lag kernels: exponential, window, log-normal, gamma (+ mass_fraction)
  src/links.rs               hierarchical Beta links (prior / global / entity), EM credit,
                             joint links, noisy-OR
  src/segments.rs            per-entity, time-bucketed segments + summaries, posting lists
  src/rules.rs               rules (type-level, or Predicate trait objects), WeightFn
  src/store.rs               HeapDB: insert, pruned top-k + group (k-way merge) queries,
                             explain, feature_risk / rank_candidates, feedback,
                             calibration, forget_entity
  src/observe.rs             observe (soft EM), settle (negative evidence, cursors), link_report
  src/stats.rs               betainc (Lentz), beta_ppf / beta_interval, spearman, lgamma
  src/snapshot.rs            JSONL snapshots (reads v1, v2, v3; writes v3)
  src/forward_decay.rs       ForwardDecayHeap<T>
  src/query.rs, src/repl.rs  query language (parser + executor), REPL commands
  src/pyfmt.rs               Python-compatible number formatting for the text output
  tests/                     property tests (proptest), behaviour tests, snapshot fixtures
crates/heapdb-cli/           `heapdb` binary: REPL and query-file runner
bindings/python/
  src/                       PyO3 extension `heapdb._core` (abi3, CPython >= 3.10)
  python/heapdb/             thin package: re-exports (heapdb.store, .links, .query, ...),
                             `_core.pyi` type stubs, `py.typed`, CLI terminal I/O
tests/, examples/, benchmarks/   Python test suite, examples and benchmarks (use the package)
```

Python objects such as `db.effect(name)`, `.links` and `db.entity_log(e)`
are live views into the Rust database. Rule predicates and weight functions
registered from Python are Python callables that the engine calls back
(exceptions propagate unchanged).

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

The answer is **exact**. Noisy-OR multiplies the `(1 - s)` factors in a
canonical order, so events whose features have equal strengths get
bit-identical scores and ties are always broken by event id (the later event
wins). `crates/heapdb/tests/topk_props.rs` (proptest) and `tests/test_topk.py` check it against brute force
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
runs the per-entity top-k and k-way merges them (a heap over the list heads). Each
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

### Observational learning: `observe`, `settle`, `link_report` (v0.4)

Feedback assumes someone says "this event caused that effect". In product
analytics, monitoring and many other settings nobody does: you only see that
a user rage-clicked, converted or churned. v0.4 learns from those bare
occurrences.

**`db.observe(entity, effect, at, leak=None)`**: the effect happened, cause
unknown. This is one soft-EM step. Every candidate `e` in the kernel window is
scored exactly as in `top_causes` (kernel × dose × noisy-OR links × rules),
and gets a responsibility

```
r_e = score_e / (leak + Σ_e' score_e')
```

`leak` is the effect's background rate, i.e. the weight of "it happened for
no event in the window". Set it per effect with
`register_effect(..., leak=0.05)` (the default) or per call. Each feature
of `e` then gets the proportional credit of a confirmation, scaled by
`r_e`: `alpha_f += r_e·share_f`, `beta_f += r_e·(1 − share_f)` with
`share_f = s_f / noisyOR(F)`. Updates go to both the entity and the global
layer. The returned `ObserveReport` lists every event's score,
responsibility and per-feature credit, and the share taken by the leak.
With a huge leak nothing is learned. With a tiny leak, every occurrence is
fully explained by whatever happened in the window.

**`db.settle(until, effects=None, entity=None, partial=False)`**: absence
of the effect is evidence too. Take an event whose kernel support window
`[t + min_lag, t + support]` has fully elapsed by `until`, where the effect
was never observed for its entity in that window. Each of its features gets
`beta_f += w_neg × rules(e)` (`register_effect(..., w_neg=1.0)`). An event
that received soft credit from `observe` is never rejected for the same
effect. A per-(entity, effect) cursor makes `settle` incremental (O(new
events)) and idempotent, and events inserted behind the cursor are queued for
the next call. With `partial=True`, events whose window is still open get
negative evidence scaled by the fraction of kernel mass already elapsed
(topped up later). If the effect does turn up in the window, a later
`observe` undoes it. For one-shot effects (churn, first conversion),
`register_effect(..., repeatable=False)` stops `settle` from rejecting an
entity's events that come after its first occurrence.

**`db.link_report(effect, entity=None, min_support=1, level=0.9,
baseline="mean")`** returns a `LinkReport` with features sorted by posterior
mean. Each row has `alpha`, `beta`, the effective evidence `n` (excluding the
prior), an equal-tailed 90% credible interval and
`lift = strength / baseline`. The baseline is the mean over reported
features, the default prior mean (`"prior"`) or a number. The interval uses
the regularized incomplete beta function (`heapdb.stats.betainc`, Lentz
continued fraction, with a port of CPython's Lanczos `lgamma`) and bisection
for quantiles (`beta_ppf`, `beta_interval`).
`db.observation_stats(effect)` reports running counters, including the
mean leak responsibility.

**Read it as association, not causation.** Links learned this way are
associations *under the model*. Here is what the seeded simulation in
`tests/test_v04_observe.py` shows. Several features follow a noisy-OR
generative model with known per-feature probabilities, over 3,000 users.
`link_report` recovers the ranking of the true effects (Spearman ≈ 0.96),
and every real cause beats every null feature. A **confounder** `z` has no
effect of its own but rides along with the strongest cause 80% of the time.
It is *partly* separated: it stays below the strongest cause and below a
moderate true cause, because proportional credit hands most of a shared
event's credit to the stronger feature. It is *not fully* separated: it ranks
confidently above the true nulls (lift ≈ 1.2). With `credit="uniform"` it
climbs above the moderate cause. The `examples/product_analytics.py` script
shows the same effect in product terms: power users both export more and
convert more, so `event:used_export` shows up as a conversion association
although it does nothing. Selection effects, novelty effects and survivor
bias are not modelled either. Use the report to choose what to test, for
example with an A/B experiment, and not as proof.

### ForwardDecayHeap: ordering that doesn't change with time

With an exponential kernel, `score(now) = w · exp(-λ(now - t))`. In log space:

```
log score(now) = [ log w + λ t ]  -  λ now
                   └─ key ──────┘    └─ same for every item ─┘
```

The key is fixed at insert, and the time-dependent term shifts every item
equally, so **a heap ordered by key is correctly ordered at every instant.**
No re-keying is needed as time passes. `ForwardDecayHeap` implements this with
a binary heap (`push`, `pop(now)`, `peek_top(now)`, `top_k`) and supports O(1)
deletion via tombstones that are cleaned up lazily. The RCA example uses it as
a live "top suspect" ticker and checks it against `top_causes` at several
times.

### Persistence

`db.save(path)` writes a JSONL snapshot (format version 3). It contains the
meta line, effects (kernel, priors, global layer, per-entity float counts,
joint links, credit mode, rule metadata, and since v0.4 `leak`, `w_neg`,
`repeatable` and the observational state: occurrences, credited and rejected
events, settle cursors), events (with `attrs`) and calibration records.
`HeapDB.load` reads v1, v2 and v3 snapshots, including files written by the
pre-0.5 Python engine (checked against fixtures written by the original
v0.1, v0.2 and v0.4 code). The v0.5 engine writes v3 as well.

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

## Changelog: v0.5

- **Engine rewritten in Rust; API unchanged.** The engine is now the
  `heapdb` Rust crate (`crates/heapdb`). Python keeps `import heapdb` with
  the same classes, methods, signatures, defaults and `str()` output, via
  PyO3 bindings (`bindings/python`, abi3 wheels for CPython >= 3.10) built
  with maturin. The existing Python test suite passes unchanged, and all
  examples print the same output. Before the pure-Python engine was removed,
  2,040 random differential workloads compared the two engines (rankings,
  scores to 1e-9, learned links, observe/settle, risk, link reports,
  calibration, snapshots) with no disagreement.
- **Native CLI.** `crates/heapdb-cli` builds a `heapdb` binary: the REPL and
  a runner for query files (`heapdb queries.hql`). The Python `heapdb`
  script uses the same Rust REPL logic and also runs query files.
- **Snapshots.** Snapshots written by the Python engine (v1, v2, v3) load
  unchanged; snapshots are still written as v3. Fix: loading no longer
  re-queues already settled events into the `settle` backlog (the Python
  engine duplicated the saved `late` lists on every load).
- **Deterministic ties.** Noisy-OR is computed in a canonical order, so
  mathematically tied events tie exactly and are ordered by event id. In
  Python the order of `frozenset` iteration could break such ties by a
  last-bit rounding difference.
- **Packaging.** `pyproject.toml` uses maturin (Rust toolchain needed to
  build from source). `.pyi` stubs and `py.typed` ship with the package.
  CI runs `cargo fmt --check`, `clippy -D warnings`, `cargo test` and the
  pytest suite on Linux/macOS/Windows, and builds wheels on tags.
- **Benchmarks.** These are re-run with the Rust engine (now up to 10M
  events), and a new `bruteforce-rust` baseline (same Rust scoring, no
  pruning) is added. On the same CPU, query p50 is 8.5x-19.4x lower than the
  pure-Python engine; see "Python engine vs Rust engine" in
  `benchmarks/RESULTS.md` and the Benchmarks section below.
- **Not ported.** There are no user-defined Python `Kernel` subclasses
  (kernels are the four built-in kinds). Python objects in `attrs` must be
  JSON values (tuples become lists).

## Changelog: v0.4

- **Observational learning.** `observe(entity, effect, at, leak=None)` does
  soft-EM credit for an effect occurrence with no labelled cause:
  `r_e = score_e / (leak + Σ scores)`, then proportional credit scaled by
  `r_e` on the entity and global layers. It returns an `ObserveReport`
  (`EventCredit` per candidate). `register_effect` gained `leak=0.05`,
  `w_neg=1.0` and `repeatable=True`.
- **Negative evidence.** `settle(until, effects=None, entity=None,
  partial=False)` adds `beta_f += w_neg × rules(e)` for events whose window
  passed without the effect. It is incremental via per-(entity, effect)
  cursors, idempotent, and handles late events. Credited events are never
  rejected. `partial=True` scales by elapsed kernel mass
  (`Kernel.mass_fraction`), and a later `observe` undoes it. It returns a
  `SettleReport`. `observation_stats(effect)` gives the running counters.
- **Link reports.** `link_report(effect, entity=None, min_support=1,
  level=0.9, baseline="mean")` returns a `LinkReport` of `LinkRow`s
  (mean, alpha, beta, n, credible interval, lift). These are associations,
  not causal effects.
- **Stats.** New `heapdb.stats`: `betainc` (regularized incomplete beta),
  `beta_ppf`, `beta_interval`, `spearman`. Stdlib only.
- **Query language.** `OBSERVE <effect> FOR <entity> [AT <t>] [LEAK <x>]`,
  `SETTLE [<effect>] UNTIL <t> [FOR <entity>]`,
  `LINKS OF <effect> [FOR <entity>] [MIN <n>]`, also in the REPL `.help`.
- **Snapshots v3.** These persist `leak`, `w_neg`, `repeatable` and the
  observational state. `forget_entity` also drops an entity's observational
  state.
- **New example.** `examples/product_analytics.py` covers synthetic users
  with flag variants, pages, onboarding and exceptions. Its effects are
  `rage_click` (minutes), `converted` (days) and `churned` (weeks). It prints
  a link report per effect against the hidden truth, a confounder caveat
  (`event:used_export`) and "why did this user churn?" EXPLAIN output.
- **Tests.** `betainc` against binomial identities and closed forms, soft
  credit sums, leak → ∞ learning nothing, settle idempotence, incrementality
  and late events, partial settle with undo, snapshot round-trip, query
  parsing, and a seeded noisy-OR simulation with a confounder.

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

## Benchmarks

[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) has the full, reproducible
results: query latency (p50/p95/p99) from 10K to 10M events for HeapDB,
brute force with the same Rust scoring code (`bruteforce-rust`), a Python
brute-force loop, SQLite and PostgreSQL (all checked to return identical
top-k before timing), pruning, ingest, memory, sensitivity to kernels and
bucket sizes, ranking quality against a known ground truth, and the v0.4
pure-Python engine against the v0.5 Rust engine. They were measured on one
2-vCPU VM with synthetic data, so read the "Reading these results" section
there before quoting any number. Rerun with
`pip install . -r benchmarks/requirements.txt && python -m benchmarks.run`.

A few lines copied verbatim from those tables. Latency in ms (p50, p95, p99,
mean), default kernel:

```
| 10K | `heapdb` | 0.011 | 0.021 | 0.049 | 0.012 | 50 (50 non-empty) |
| 10K | `bruteforce-rust` | 0.005 | 0.008 | 0.012 | 0.005 | 50 (50 non-empty) |
| 10K | `sqlite-mem` | 0.025 | 0.043 | 0.064 | 0.026 | 50 (50 non-empty) |
| 1M | `heapdb` | 0.107 | 0.167 | 0.200 | 0.120 | 50 (50 non-empty) |
| 1M | `bruteforce-rust` | 0.119 | 0.157 | 0.173 | 0.125 | 50 (50 non-empty) |
| 1M | `sqlite-mem` | 1.75 | 2.71 | 2.92 | 1.85 | 50 (50 non-empty) |
| 1M | `postgres` | 4.04 | 4.70 | 6.20 | 4.11 | 50 (50 non-empty) |
| 10M | `heapdb` | 0.514 | 0.602 | 0.634 | 0.517 | 50 (50 non-empty) |
| 10M | `bruteforce-rust` | 0.789 | 1.15 | 1.21 | 0.830 | 50 (50 non-empty) |
| 10M | `sqlite-mem` | 18.5 | 29.0 | 35.1 | 20.2 | 50 (50 non-empty) |
| 10M | `postgres` | 63.5 | 129 | 174 | 72.0 | 50 (50 non-empty) |
```

heapdb p50 relative to the others (other p50 / heapdb p50; below 1 means
heapdb is slower; columns: N, heapdb p50, vs `bruteforce-rust`, vs
`bruteforce`, vs `sqlite-mem`, vs `sqlite-disk`, vs `postgres`):

```
| 10K | 0.011 | 0.43x | 0.49x | 2.28x | 2.51x | 11.00x |
| 100K | 0.026 | 0.81x | 1.85x | 7.09x | 7.05x | 20.33x |
| 1M | 0.107 | 1.10x | 4.12x | 16.30x | 16.71x | 37.58x |
| 10M | 0.514 | 1.54x | not measured | 35.93x | 35.89x | 123.58x |
```

Python engine (v0.4) vs Rust engine (v0.5), same CPU (columns: N, Python
p50 ms, Rust p50 ms, p50 speed-up, Python p99 ms, Rust p99 ms, Python ingest
ev/s, Rust ingest ev/s, ingest speed-up, Python RSS B/event, Rust RSS
B/event):

```
| 10K | 0.092 | 0.011 | 8.5x | 0.294 | 0.049 | 75,809 | 328,630 | 4.3x | 2,415 | 808 |
| 100K | 0.377 | 0.026 | 14.3x | 0.607 | 0.056 | 41,414 | 273,492 | 6.6x | 2,228 | 607 |
| 1M | 2.08 | 0.107 | 19.4x | 3.75 | 0.200 | 42,969 | 357,197 | 8.3x | 1,502 | 298 |
| 10M | not measured | 0.514 | - | not measured | 0.634 | not measured | 293,950 | - | not measured | 178 |
```

Pruning (events in window, heapdb events scored, scored p99, segments in
window, segments pruned, pruned %):

```
| 1M | 459.0 | 19.2 | 30 | 129.2 | 124.7 | 96.6% |
| 10M | - | 149.6 | 250 | 133.3 | 129.0 | 96.8% |
```

Where pruning does not pay at 1M: a short kernel, and a flat 24h window with
all links equal (columns: kernel, events in window, heapdb scored, pruned %,
heapdb p50, [heapdb p99,] bruteforce-rust p50, bruteforce p50, sqlite-mem
p50, bruteforce-rust / heapdb):

```
| `exp_10m` | 23.3 | 9.4 | 63.8% | 0.015 | 0.040 | 0.011 | 0.031 | 0.089 | 0.72x |
| `window_24h` | 333.5 | 117.9 | 70.6% | 0.073 | 0.072 | 0.397 | 1.14 | 0.98x |
```

Memory (ingest events/s, ingest s, index build s, RSS delta, storage):

```
| 1M | `heapdb` | 357,197 | 2.80 | 0.00 | 284 (298 B/event) | 279,021 segments |
| 1M | `sqlite-mem` | 185,164 | 5.40 | 1.39 | 132 (139 B/event) | - |
| 10M | `heapdb` | 293,950 | 34.02 | 0.00 | 1,701 (178 B/event) | 288,000 segments |
| 10M | `sqlite-mem` | 173,499 | 57.64 | 20.55 | 1,331 (140 B/event) | - |
```

Ranking quality, hit@1 of the true culprit (days 0-3.5, days 3.5-7, whole
week; mean ± std over 20 seeds; identical for both engines):

```
| `recency` | 0.335 ± 0.097 | 0.411 ± 0.117 | 0.367 ± 0.070 |
| `heapdb_proportional` | 0.536 ± 0.098 | 0.600 ± 0.111 | 0.562 ± 0.076 |
```

In short: the Rust engine is 8-19x faster per query than the pure-Python
engine, ingests 4-8x faster and uses about a fifth of the memory, with
identical results. It is 16x faster than SQLite and 38x faster than
PostgreSQL at 1M (36x and 124x at 10M). Against brute force over the same
Rust store, pruning only pays off at larger N (1.10x at 1M, 1.54x at 10M);
at 10K and 100K, and with very short or uninformative kernels, heapdb is
slower or tied. SQLite still uses less memory per event.

## Roadmap

Ideas, not promises:

- [x] ~~Benchmarks~~: `benchmarks/` (see Benchmarks above)
- [ ] Guidance for choosing `bucket_seconds`; multi-resolution
      (hierarchical) segment summaries for long lookbacks
- [x] ~~Per-feature max-weight summaries for tighter bounds~~: per-set bounds in v0.2
- [ ] Automatic recalibration (Platt / isotonic) from the calibration records,
      so scores can be read as probabilities and compared across entities
- [ ] Privacy: per-entity keys with crypto-shredding, audit log, DP for the
      global layer (see Privacy)
- [ ] Learning kernel parameters (lag distributions) from feedback, not just links
- [ ] Better confounding handling: partial co-occurrence, suggested
      "elimination experiments" (v0.2 has exact co-occurrence + joint links;
      v0.4 adds lift and credible intervals in `link_report`)
- [ ] Observational learning: per-entity propensity adjustment / matched
      comparisons to reduce confounding, learned leak rates, multi-step EM
      (v0.4 does a single online soft-EM step per occurrence)
- [ ] Negative / protective links, and effect-specific cause type hierarchies
- [ ] Interaction features learned automatically (today a harmful combination such
      as `migration+review:fast` is learned only through its members unless you
      add a cross feature yourself)
- [ ] Incremental "standing queries" built on forward decay for exponential kernels
- [ ] Durable storage: write-ahead log, compaction, segment files
- [x] ~~A Rust core~~: v0.5 (the Python engine served as the differential oracle during the port)

## Contributing

Contributions, issues and design discussion are very welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) to get started. Good first issues include
new kernels, benchmarks and query-language features.

## License

Apache License 2.0. See [LICENSE](LICENSE).
