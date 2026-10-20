# Examples

All scripts are self-contained and only need the `heapdb` package (the Rust
engine's Python bindings). From a checkout, build it once with
`pip install .` (or `maturin develop --release` in a virtualenv), then:

```bash
python examples/bloating.py
python examples/incident_rca.py
python examples/code_changes.py
python examples/product_analytics.py
```

## `bloating.py`: personal health, learning from feedback

One person, a daily menu, bloating at 16:00. It shows:

- **Weight = dose.** The `meal` event type derives its weight from
  `attrs["portion"]` via `register_event_type`.
- **Proportional credit.** Each confirmation prints how much credit each
  feature of the latte got (`r_f = s_f / noisyOR`). At the end, the same
  history is replayed with `credit="uniform"` (v0.1 behaviour) for comparison.
  Uniform credit pushes `coffee` up as hard as `dairy` (0.55 vs 0.30).
- **Joint causes.** While `coffee` and `dairy` have only ever arrived
  together, the explanation reports them as one cause `coffee+dairy` with a
  joint link and an `inseparable` flag. Once a dairy-free coffee snack has
  been seen (and rejected), they are separable and per-feature strengths
  take over.
- **Calibration.** `calibration_report("bloating")` shows the mean predicted
  score against the observed confirm rate per bin, plus the Brier score.

## `incident_rca.py`: root-cause analysis across services

A week of changes to `checkout`, then a p99 latency spike. It shows:

- **Pruning.** 234 events, answered by scoring 5. The result is checked
  against brute force.
- **`cause_types` versus rules.** `cause_types` hard-filters `cron`. Soft
  rules multiply a score by a factor in [0, 1]: dry-run config changes get
  x0.1 (a predicate rule on `attrs`) and flag flips get x0.5 (a type-level
  rule, which also tightens pruning bounds). Applied rules are listed in the
  explanation.
- **Dose from attributes.** Deploy weight is
  `log1p(lines_changed) / log1p(100)`.
- **Group query.** `top_causes_group(["checkout", "payments", "search"], ...)`
  ranks changes across services. Scores are comparable across entities only
  once they are calibrated.
- **Hierarchical priors.** Postmortem feedback on `checkout` also updates the
  global layer, so a brand-new service inherits the lesson.
- **ForwardDecayHeap.** A live "hottest change" ticker that agrees with
  `top_causes` at every instant without re-scoring.

## `code_changes.py`: AI-heavy code changes, from attribution to pre-merge risk

A seeded week of ~150 merged PRs across `checkout`, `payments` and `search`,
with features like `path:payments/retry`, `migration`, `lockfile`,
`dep_bump`, `config`, `flag_flip`, `size:large`, `review:fast`,
`review:none`, `ai_authored`, `tests:none` and `docs_only`. Hidden ground
truth makes some combinations dangerous (migration + fast review, lockfile
changes in payments, config without tests). `ai_authored` has no causal
effect but correlates with large, fast-reviewed PRs. Runs in well under a
second. It shows:

- **Three effects, three kernels.** `failing_main_check` (exponential,
  10 min half-life), `error_spike` (log-normal, median 30 min) and
  `incident` (log-normal, median 3 h). `cause_types=["deploy"]` excludes
  on-call handoffs, and a `docs_only` rule (x0) is a hard block. Services are
  separate entities, so a deploy to `search` is never blamed for a
  `payments` incident.
- **Reverts as feedback.** The true culprit is confirmed and the other top-3
  suspects are rejected. hit@1 / hit@3 are printed per effect for the first
  and second half of the week, plus a no-feedback baseline and a 20-seed
  average. One week is small, so expect a modest gain (about +0.07 hit@1 on
  average), not a dramatic one.
- **Proportional vs uniform credit.** The same week replayed with
  `credit="uniform"`: the confounder `ai_authored` ends up with an incident
  link about 3x higher than under proportional credit.
- **Calibration** for `incident` and `error_spike`, and a cross-service
  **group query**.
- **Pre-merge review queue.** `rank_candidates("incident", open_prs, k=10)`
  ranks 10 open PRs by `feature_risk` (dose x noisy-OR of the current links
  x rules, with no timing term), with explanations, next to the hidden true
  risk. The same query is also run as `RISK OF incident FOR payments FEATURES ...`.
  Links are per-feature, so interactions are learned only through their
  members. The script points out where that misleads the queue.

## `product_analytics.py`: observational learning, no feedback at all

1,200 synthetic users over about two months. Each user sits in a feature-flag
variant (`flag:new-editor=test|control`), visits pages (some
`/settings/billing` loads are `perf:slow`), completes or skips onboarding,
uses the editor, invites teammates, exports, and sometimes hits an
exception. Hidden noisy-OR ground truth produces `rage_click` (exponential
kernel, 2 min half-life), `converted` (log-normal, median 1 day) and
`churned` (log-normal, median 14 days). Nobody labels causes: occurrences go
to `observe()` and the rest is handled by `settle()`. Runs in about 1 s. It
shows:

- **`link_report` per effect** with 90% credible intervals and lift, next to
  the hidden truth. Every true cause ranks above every null feature.
- **The confounder caveat.** Power users export a lot and convert more for
  unrelated reasons, so `event:used_export` appears as a conversion
  association although it has no effect.
- **One-shot effects.** `repeatable=False` for `converted` and `churned`.
- **"Why did this user churn?"** `top_causes(..., explain=True)` for one
  churned user.
