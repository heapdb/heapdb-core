"""Lag kernels: how plausible is a cause that happened ``dt`` seconds ago?

A kernel maps a non-negative lag ``dt = effect_time - event_time`` to a value
in ``[0, 1]``. Every kernel in this module is *peak-normalised* (its maximum
value is 1.0) so that kernel values are easy to read in explanations, and is
*truncated* at a finite :attr:`Kernel.support`: events older than the support
window get value 0 and are never even looked at by the query engine.

For pruning, the query engine needs an upper bound of the kernel over a whole
range of lags (one storage segment covers a range of event times, hence a
range of lags). That is :meth:`Kernel.max_over`. All kernels here are
unimodal, so the bound is exact: the peak if the mode lies inside the range,
otherwise the larger of the two endpoint values.

Adding a kernel: subclass :class:`Kernel`, implement ``_raw(dt)`` (unscaled,
for ``dt > 0``), ``mode``, ``support`` and ``to_dict``, and register the class
in ``KERNEL_TYPES`` so snapshots can round-trip it.
"""

from __future__ import annotations

import math
from typing import Any


class Kernel:
    """Base class for unimodal, peak-normalised, truncated lag kernels."""

    #: Short name used in snapshots and the query language.
    name: str = "kernel"

    # -- to be provided by subclasses ---------------------------------------

    @property
    def support(self) -> float:
        """Maximum lookback in seconds. ``value(dt) == 0`` for ``dt > support``."""
        raise NotImplementedError

    @property
    def min_lag(self) -> float:
        """Minimum lag in seconds. ``value(dt) == 0`` for ``dt < min_lag``."""
        return 0.0

    @property
    def mode(self) -> float:
        """Lag at which the kernel peaks."""
        raise NotImplementedError

    def _raw(self, dt: float) -> float:
        """Un-normalised shape for ``min_lag <= dt <= support``."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    # -- shared behaviour ---------------------------------------------------

    @property
    def max_value(self) -> float:
        """Peak value. Always 1.0 because kernels are peak-normalised."""
        return 1.0

    def value(self, dt: float) -> float:
        """Kernel value for a cause that happened ``dt`` seconds before the effect.

        Causes in the future (``dt < 0``) and outside ``[min_lag, support]``
        score 0.
        """
        if dt < self.min_lag or dt > self.support or dt < 0:
            return 0.0
        peak = self._raw(self.mode)
        return min(1.0, self._raw(dt) / peak) if peak > 0 else 0.0

    def max_over(self, dt_lo: float, dt_hi: float) -> float:
        """Upper bound (here: exact maximum) of ``value`` over ``[dt_lo, dt_hi]``.

        Used to bound every event inside a segment without touching them.
        """
        lo = max(dt_lo, self.min_lag, 0.0)
        hi = min(dt_hi, self.support)
        if lo > hi:
            return 0.0
        if lo <= self.mode <= hi:
            return self.max_value
        # Unimodal: the maximum over an interval not containing the mode is at
        # the endpoint closest to the mode.
        return max(self.value(lo), self.value(hi))

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items() if k != "kind")
        return f"{type(self).__name__}({args})"


class ExponentialDecay(Kernel):
    """``value(dt) = 0.5 ** (dt / half_life)``: most recent is most plausible.

    The tail is truncated at ``horizon_half_lives`` half-lives (default 10,
    i.e. value < 0.001) so the support window is finite.
    """

    name = "exp"

    def __init__(self, half_life: float, horizon_half_lives: float = 10.0) -> None:
        if half_life <= 0:
            raise ValueError("half_life must be > 0")
        self.half_life = float(half_life)
        self.horizon_half_lives = float(horizon_half_lives)

    @property
    def support(self) -> float:
        return self.half_life * self.horizon_half_lives

    @property
    def mode(self) -> float:
        return 0.0

    def _raw(self, dt: float) -> float:
        return 0.5 ** (dt / self.half_life)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.name, "half_life": self.half_life,
                "horizon_half_lives": self.horizon_half_lives}


class WindowKernel(Kernel):
    """Flat kernel: 1.0 for ``min_lag <= dt <= max_lag``, else 0.

    Useful for "anything in the last N minutes counts equally", e.g. deploys
    in the 30 minutes before an incident.
    """

    name = "window"

    def __init__(self, min_lag: float = 0.0, max_lag: float = 3600.0) -> None:
        if not 0 <= min_lag <= max_lag:
            raise ValueError("need 0 <= min_lag <= max_lag")
        self._min_lag = float(min_lag)
        self.max_lag = float(max_lag)

    @property
    def support(self) -> float:
        return self.max_lag

    @property
    def min_lag(self) -> float:
        return self._min_lag

    @property
    def mode(self) -> float:
        # Any point of the plateau is a mode; pick the start.
        return self._min_lag

    def _raw(self, dt: float) -> float:
        return 1.0

    def max_over(self, dt_lo: float, dt_hi: float) -> float:
        # Plateau: 1 iff the ranges overlap.
        return 1.0 if max(dt_lo, self._min_lag) <= min(dt_hi, self.max_lag) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.name, "min_lag": self._min_lag, "max_lag": self.max_lag}


class LogNormalKernel(Kernel):
    """Log-normal shaped delay: rises, peaks, then a long right tail.

    Parameterised by the ``median`` lag and log-space ``sigma``. This is a
    natural model for biological delays (digestion, drug onset). The shape is
    the log-normal density scaled so its peak (at ``median * exp(-sigma**2)``)
    is 1.0. Truncated at ``median * exp(tail_sigmas * sigma)``.
    """

    name = "lognormal"

    def __init__(self, median: float, sigma: float = 0.5, tail_sigmas: float = 4.0) -> None:
        if median <= 0 or sigma <= 0:
            raise ValueError("median and sigma must be > 0")
        self.median = float(median)
        self.sigma = float(sigma)
        self.tail_sigmas = float(tail_sigmas)
        self._mu = math.log(self.median)

    @property
    def support(self) -> float:
        return self.median * math.exp(self.tail_sigmas * self.sigma)

    @property
    def mode(self) -> float:
        return math.exp(self._mu - self.sigma ** 2)

    def _raw(self, dt: float) -> float:
        if dt <= 0:
            return 0.0
        z = (math.log(dt) - self._mu) / self.sigma
        return math.exp(-0.5 * z * z) / dt

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.name, "median": self.median, "sigma": self.sigma,
                "tail_sigmas": self.tail_sigmas}


class GammaKernel(Kernel):
    """Gamma shaped delay ``dt**(shape-1) * exp(-dt/scale)``, peak-normalised.

    Requires ``shape >= 1`` so the peak is finite (mode = ``(shape-1)*scale``).
    Truncated at ``mean + tail_stds * std``.
    """

    name = "gamma"

    def __init__(self, shape: float, scale: float, tail_stds: float = 6.0) -> None:
        if shape < 1 or scale <= 0:
            raise ValueError("need shape >= 1 and scale > 0")
        self.shape = float(shape)
        self.scale = float(scale)
        self.tail_stds = float(tail_stds)

    @property
    def support(self) -> float:
        mean = self.shape * self.scale
        std = math.sqrt(self.shape) * self.scale
        return mean + self.tail_stds * std

    @property
    def mode(self) -> float:
        return (self.shape - 1.0) * self.scale

    def _raw(self, dt: float) -> float:
        if dt < 0:
            return 0.0
        # Work in log space to avoid overflow for large shapes.
        if dt == 0:
            return 1.0 if self.shape == 1 else 0.0
        return math.exp((self.shape - 1.0) * math.log(dt) - dt / self.scale)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.name, "shape": self.shape, "scale": self.scale,
                "tail_stds": self.tail_stds}


KERNEL_TYPES: dict[str, type[Kernel]] = {
    ExponentialDecay.name: ExponentialDecay,
    WindowKernel.name: WindowKernel,
    LogNormalKernel.name: LogNormalKernel,
    GammaKernel.name: GammaKernel,
}


def kernel_from_dict(d: dict[str, Any]) -> Kernel:
    """Inverse of ``Kernel.to_dict`` (used by snapshots)."""
    d = dict(d)
    kind = d.pop("kind")
    try:
        cls = KERNEL_TYPES[kind]
    except KeyError:
        raise ValueError(f"unknown kernel kind {kind!r}") from None
    return cls(**d)
