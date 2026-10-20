import math
import random

import pytest

from heapdb import ExponentialDecay, GammaKernel, LogNormalKernel, WindowKernel
from heapdb.kernels import kernel_from_dict

KERNELS = [
    ExponentialDecay(half_life=600),
    WindowKernel(min_lag=300, max_lag=3600),
    LogNormalKernel(median=7200, sigma=0.6),
    GammaKernel(shape=3, scale=1800),
    GammaKernel(shape=1, scale=900),  # degenerates to exponential
]


def test_exponential_half_life():
    k = ExponentialDecay(half_life=600)
    assert k.value(0) == pytest.approx(1.0)
    assert k.value(600) == pytest.approx(0.5)
    assert k.value(1200) == pytest.approx(0.25)
    assert k.support == pytest.approx(6000)
    assert k.value(6001) == 0.0


def test_window_kernel():
    k = WindowKernel(min_lag=300, max_lag=3600)
    assert k.value(299) == 0.0
    assert k.value(300) == 1.0
    assert k.value(3600) == 1.0
    assert k.value(3601) == 0.0
    assert k.max_over(0, 299) == 0.0
    assert k.max_over(0, 400) == 1.0


def test_lognormal_peak_and_shape():
    k = LogNormalKernel(median=7200, sigma=0.6)
    mode = 7200 * math.exp(-0.36)
    assert k.mode == pytest.approx(mode)
    assert k.value(mode) == pytest.approx(1.0)
    assert k.value(mode / 2) < 1.0 and k.value(mode * 2) < 1.0
    assert k.value(0) == 0.0
    assert k.value(k.support + 1) == 0.0


def test_gamma_mode():
    k = GammaKernel(shape=3, scale=1800)
    assert k.mode == pytest.approx(3600)
    assert k.value(3600) == pytest.approx(1.0)
    assert k.value(1800) < 1.0 and k.value(7200) < 1.0


@pytest.mark.parametrize("k", KERNELS, ids=repr)
def test_values_in_unit_interval_and_zero_for_future(k):
    assert k.value(-1) == 0.0
    for i in range(500):
        v = k.value(k.support * i / 400)
        assert 0.0 <= v <= 1.0
    assert k.max_value == 1.0


@pytest.mark.parametrize("k", KERNELS, ids=repr)
def test_negative_lag_is_always_zero(k):
    """Temporal sanity: a cause can never follow its effect."""
    for dt in (-1e-9, -1.0, -k.support / 2, -k.support, -1e12):
        assert k.value(dt) == 0.0
    assert k.max_over(-k.support, -1e-9) == 0.0
    assert k.max_over(-k.support, 0.0) == k.value(0.0)


@pytest.mark.parametrize("k", KERNELS, ids=repr)
def test_max_over_is_a_true_upper_bound(k):
    """max_over must bound value() everywhere in the interval (pruning relies on it)."""
    rng = random.Random(0)
    for _ in range(300):
        a = rng.uniform(-0.1, 1.2) * k.support
        b = a + rng.uniform(0, 0.3) * k.support
        bound = k.max_over(a, b)
        for j in range(21):
            dt = a + (b - a) * j / 20
            assert k.value(dt) <= bound + 1e-12


@pytest.mark.parametrize("k", KERNELS, ids=repr)
def test_roundtrip(k):
    k2 = kernel_from_dict(k.to_dict())
    assert type(k2) is type(k)
    for dt in (0, 100, 1000, 5000):
        assert k2.value(dt) == pytest.approx(k.value(dt))


def test_invalid_params():
    with pytest.raises(ValueError):
        ExponentialDecay(0)
    with pytest.raises(ValueError):
        WindowKernel(10, 5)
    with pytest.raises(ValueError):
        GammaKernel(0.5, 10)
    with pytest.raises(ValueError):
        kernel_from_dict({"kind": "nope"})
