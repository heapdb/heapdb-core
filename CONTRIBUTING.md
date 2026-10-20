# Contributing to HeapDB

Thanks for your interest! HeapDB is an early reference implementation, so
ideas, critiques and experiments are just as welcome as code.

## Ground rules

- **Readable over clever.** This code exists to be read. Prefer clear names,
  short functions and a comment explaining *why* over micro-optimisations.
- **Stdlib only** for the `heapdb` package. `pytest` is the only dev dependency.
- **Python 3.10+.**
- **Exactness first.** Any change to the query engine must keep
  `tests/test_topk.py` green: pruned top-k must equal brute force, and
  segment bounds must stay *true* upper bounds (rule multipliers are capped
  at 1 for this reason).

## Getting started

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
pytest
python examples/bloating.py
python examples/incident_rca.py
```

## Good first contributions

- New lag kernels (e.g. Weibull, mixtures). Implement `_raw`, `mode`,
  `support`, `to_dict`; register in `KERNEL_TYPES`; add the kernel to the
  parametrised tests in `tests/test_kernels.py` and `tests/test_topk.py`.
- Better confounding detection (partial co-occurrence, lift).
- Recalibration from `calibration_report` data (Platt / isotonic).
- Benchmarks: segment width vs. pruning rate on realistic workloads.
- Query-language features (`WHERE type = ...`, `RANK BY ...`).

## Pull requests

1. Open an issue first for anything larger than a small fix, so we can agree on
   the approach.
2. Add or update tests. Property-style tests with fixed random seeds are
   preferred for anything algorithmic.
3. Keep the README's architecture section accurate if you change behaviour.

By contributing you agree that your contributions are licensed under the
Apache License 2.0 (see `LICENSE`).
