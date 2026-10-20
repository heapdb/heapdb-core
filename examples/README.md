# Examples

Both scripts are self-contained (stdlib only) and put the repository root on
`sys.path`, so you can run them from a fresh checkout:

```bash
python examples/bloating.py
python examples/incident_rca.py
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
