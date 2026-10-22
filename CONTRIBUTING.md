# Contributing to HeapDB

Thanks for your interest! HeapDB is early-stage, so ideas, critiques and
experiments are just as welcome as code.

## Layout

- `crates/heapdb/`: the engine (Rust library, no Python dependencies).
- `crates/heapdb-cli/`: the native `heapdb` REPL / query-file runner.
- `bindings/python/`: PyO3 bindings (`heapdb._core`) and the thin `heapdb`
  Python package (re-exports, type stubs, CLI I/O).
- `tests/`: the Python test suite (runs against the built extension).
- `examples/`, `benchmarks/`: Python scripts using the `heapdb` package.

## Ground rules

- **Engine logic lives in Rust.** The Python package stays a thin binding;
  new behaviour goes into `crates/heapdb` first, then gets exposed in
  `bindings/python` with the same names and semantics (and a stub in
  `_core.pyi`).
- **Readable over clever.** Clear names, short functions, comments that
  explain *why*. No `unsafe` (the crates use `#![forbid(unsafe_code)]`).
- **Dependencies:** the engine uses a few small crates (`thiserror`, `serde`,
  `serde_json`, `indexmap`, `rustc-hash`, `chrono`); please discuss new ones
  first. The Python package has no runtime dependencies. Benchmark code may
  use `benchmarks/requirements.txt`.
- **Toolchain:** Rust >= 1.82 (stable), Python >= 3.10, maturin >= 1.9.4.
- **Exactness first.** Any change to the query engine must keep the pruned
  top-k equal to brute force (`crates/heapdb/tests/topk_props.rs`,
  `tests/test_topk.py`), and segment bounds must stay *true* upper bounds
  (rule multipliers are capped at 1 for this reason).
- **Output stability.** `str()` of results and the REPL output are part of
  the API; they follow Python's float formatting (`heapdb::pyfmt`).
- **Snapshots.** Old snapshot formats must keep loading
  (`crates/heapdb/tests/fixtures`, written by the original v0.1/v0.2/v0.4
  Python engine).

## Getting started

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest -r benchmarks/requirements.txt
maturin develop --release          # build the extension into the venv
pytest                             # Python suite
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace             # Rust unit, property and fixture tests
python examples/code_changes.py
cargo run -p heapdb-cli -- --demo  # native REPL
```

## Good first contributions

- New lag kernels (e.g. Weibull, mixtures): add a `KernelShape` variant in
  `crates/heapdb/src/kernels.rs` (`raw`, `mode`, `support`, `params`,
  snapshot (de)serialisation), a Python class in
  `bindings/python/src/kernels.rs`, and add it to the property tests and
  `tests/test_kernels.py` / `tests/test_topk.py`.
- Better confounding detection (partial co-occurrence, lift).
- Recalibration from `calibration_report` data (Platt / isotonic).
- Benchmarks: more realistic (bursty) workloads in `benchmarks/workload.py`.
  Never edit `benchmarks/RESULTS.md` by hand: rerun `python -m benchmarks.run`
  and `python -m benchmarks.report`.
- Query-language features (`WHERE type = ...`, `RANK BY ...`).

## Pull requests

1. Open an issue first for anything larger than a small fix, so we can agree on
   the approach.
2. Add or update tests. Property-style tests with fixed seeds are preferred
   for anything algorithmic (`proptest` in Rust, seeded `random` in Python).
3. Keep the README's architecture section accurate if you change behaviour.
4. CI runs `cargo fmt --check`, `clippy -D warnings`, `cargo test` and the
   pytest suite on Linux, macOS and Windows.

By contributing you agree that your contributions are licensed under the
Apache License 2.0 (see `LICENSE`).
