"""Small stdlib-only statistics helpers (v0.4).

* :func:`betainc` -- the regularized incomplete beta function ``I_x(a, b)``
  (the Beta CDF), by the Lentz continued fraction (Numerical Recipes
  ``betacf``) with the usual symmetry switch for fast convergence.
* :func:`beta_ppf` -- Beta quantiles by bisection on :func:`betainc`.
* :func:`beta_interval` -- equal-tailed credible interval of a Beta belief.
* :func:`spearman` -- Spearman rank correlation (average ranks for ties),
  used by tests and examples to compare learned and true rankings.
"""

from __future__ import annotations

import math
from typing import Sequence

_EPS = 1e-15
_TINY = 1e-300


def _betacf(a: float, b: float, x: float, max_iter: int = 500) -> float:
    """Continued fraction for I_x(a, b) (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` = CDF of Beta(a, b) at ``x``."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be > 0")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return min(1.0, max(0.0, front * _betacf(a, b, x) / a))
    return min(1.0, max(0.0, 1.0 - front * _betacf(b, a, 1.0 - x) / b))


def beta_ppf(q: float, a: float, b: float, tol: float = 1e-10) -> float:
    """Quantile of Beta(a, b): the ``x`` with ``I_x(a, b) = q`` (bisection)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def beta_interval(a: float, b: float, level: float = 0.9) -> tuple[float, float]:
    """Equal-tailed ``level`` credible interval of Beta(a, b)."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    tail = (1.0 - level) / 2.0
    return beta_ppf(tail, a, b), beta_ppf(1.0 - tail, a, b)


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation of two equal-length sequences."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need two sequences of equal length >= 2")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)
