"""The benchmark implementations must agree on a tiny dataset.

PostgreSQL runs only if psycopg is installed and a server is reachable
(``HEAPDB_BENCH_PG_DSN``, or local server binaries); otherwise it is skipped.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.impls import compare, make_impl  # noqa: E402
from benchmarks.pg import postgres_server  # noqa: E402
from benchmarks.quality import METHODS, run_quality  # noqa: E402
from benchmarks.workload import (KERNEL_NAMES, Reference, WorkloadSpec,  # noqa: E402
                                 generate, make_kernel, queries)

SPECS = [
    WorkloadSpec(n_events=3000, n_entities=5, days=10, vocab=200, seed=3),
    WorkloadSpec(n_events=2000, n_entities=4, days=5, vocab=100, links="flat", seed=4),
]


def _check(impl_name, wl, kernel_names, **opts):
    impl = make_impl(impl_name, wl, make_kernel(kernel_names[0]), **opts)
    impl.build()
    ref = Reference(wl)
    nonempty = 0
    try:
        for kn in kernel_names:
            kernel = make_kernel(kn)
            impl.set_kernel(kn, kernel)
            for e, a in queries(wl, 40):
                want = ref.topk(kernel, e, a, 5)
                nonempty += bool(want)
                diff = compare(want, impl.query(e, a, 5))
                assert diff is None, f"{impl_name} {kn} entity {e} at {a}: {diff}"
    finally:
        impl.close()
    assert nonempty > 0


@pytest.mark.parametrize("spec", SPECS, ids=["skewed", "flat"])
@pytest.mark.parametrize("impl", ["heapdb", "bruteforce-rust", "bruteforce", "sqlite-mem", "sqlite-disk"])
def test_impls_agree(impl, spec):
    _check(impl, generate(spec), list(KERNEL_NAMES))


def test_heapdb_bucket_sizes_agree():
    wl = generate(SPECS[0])
    for bucket in (300.0, 3600.0):
        _check("heapdb", wl, list(KERNEL_NAMES), bucket=bucket)


def test_postgres_agrees():
    with postgres_server(local=True, port=55433) as (dsn, why):
        if dsn is None:
            pytest.skip(f"PostgreSQL not available: {why}")
        _check("postgres", generate(SPECS[0]), list(KERNEL_NAMES), dsn=dsn)


def test_compare_allows_tie_permutations_only():
    assert compare([(1.0, 2), (1.0, 1), (0.5, 3)], [(1.0, 1), (1.0, 2), (0.5, 3)]) is None
    assert compare([(1.0, 2), (0.9, 1)], [(1.0, 1), (0.9, 2)]) is not None
    assert compare([(1.0, 2)], [(1.0 + 1e-6, 2)]) is not None
    assert compare([(1.0, 2)], []) is not None
    # The last tied group may be cut by k differently.
    assert compare([(1.0, 5), (0.5, 4)], [(1.0, 5), (0.5, 3)]) is None


def test_quality_smoke():
    q = run_quality(n_seeds=2)
    for m in METHODS:
        h = q["by_period"][m]["all"]["hit1"]["mean"]
        assert 0.0 <= h <= 1.0
