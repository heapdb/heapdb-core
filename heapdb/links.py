"""Learned link strengths between features and effect types.

For every ``(effect_type, feature)`` pair HeapDB keeps a Beta(alpha, beta)
belief about "how likely is this feature to produce this effect". The point
estimate used for ranking is the posterior mean ``alpha / (alpha + beta)``.

* Priors encode domain knowledge ("beans are a usual suspect for bloating").
  Features with no explicit prior fall back to the effect's default prior.
* Feedback is plain Bayesian counting: a confirmed cause adds 1 to ``alpha``
  for each of its features, a rejected candidate adds 1 to ``beta``.

An event usually has several features. We combine them with *noisy-OR*:
``P(effect | event) = 1 - prod_f (1 - s_f)`` -- the event causes the effect
unless every one of its features independently fails to. Noisy-OR is monotone
in the feature set, which the query engine exploits: the noisy-OR over the
*union* of features in a segment upper-bounds every event in that segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class BetaCounts:
    """A Beta(alpha, beta) belief split into prior and observed counts."""

    prior_alpha: float
    prior_beta: float
    pos: float = 0.0  # confirmed observations
    neg: float = 0.0  # rejected observations

    @property
    def alpha(self) -> float:
        return self.prior_alpha + self.pos

    @property
    def beta(self) -> float:
        return self.prior_beta + self.neg

    @property
    def strength(self) -> float:
        """Posterior mean of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)


def noisy_or(strengths: Iterable[float]) -> float:
    """``1 - prod(1 - s)``; 0.0 for an empty iterable."""
    miss = 1.0
    for s in strengths:
        miss *= 1.0 - s
    return 1.0 - miss


class LinkTable:
    """Beta link beliefs for one effect type.

    Only pairs that have an explicit prior or have received feedback are
    materialised; everything else reads the default prior on the fly.
    """

    def __init__(
        self,
        default_prior: tuple[float, float] = (1.0, 4.0),
        priors: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        a, b = default_prior
        if a <= 0 or b <= 0:
            raise ValueError("Beta prior parameters must be > 0")
        self.default_prior = (float(a), float(b))
        self._links: dict[str, BetaCounts] = {}
        for feature, (pa, pb) in (priors or {}).items():
            if pa <= 0 or pb <= 0:
                raise ValueError(f"Beta prior for {feature!r} must be > 0")
            self._links[feature] = BetaCounts(float(pa), float(pb))

    # -- reads --------------------------------------------------------------

    def get(self, feature: str) -> BetaCounts:
        """Counts for ``feature`` (a detached default-prior object if unseen)."""
        link = self._links.get(feature)
        if link is None:
            return BetaCounts(*self.default_prior)
        return link

    def strength(self, feature: str) -> float:
        link = self._links.get(feature)
        if link is None:
            a, b = self.default_prior
            return a / (a + b)
        return link.strength

    def noisy_or(self, features: Iterable[str]) -> float:
        """Aggregate strength of a feature set (see module docstring)."""
        return noisy_or(self.strength(f) for f in features)

    def items(self) -> Iterable[tuple[str, BetaCounts]]:
        return self._links.items()

    # -- writes -------------------------------------------------------------

    def _materialise(self, feature: str) -> BetaCounts:
        link = self._links.get(feature)
        if link is None:
            link = BetaCounts(*self.default_prior)
            self._links[feature] = link
        return link

    def set_prior(self, feature: str, alpha: float, beta: float) -> None:
        """Set (or replace) the prior for one feature, keeping observed counts."""
        if alpha <= 0 or beta <= 0:
            raise ValueError("Beta prior parameters must be > 0")
        link = self._materialise(feature)
        link.prior_alpha, link.prior_beta = float(alpha), float(beta)

    def confirm(self, features: Iterable[str], amount: float = 1.0) -> None:
        for f in features:
            self._materialise(f).pos += amount

    def reject(self, features: Iterable[str], amount: float = 1.0) -> None:
        for f in features:
            self._materialise(f).neg += amount

    # -- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "default_prior": list(self.default_prior),
            "links": {
                f: [c.prior_alpha, c.prior_beta, c.pos, c.neg]
                for f, c in sorted(self._links.items())
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LinkTable":
        table = cls(default_prior=tuple(d["default_prior"]))  # type: ignore[arg-type]
        for f, (pa, pb, pos, neg) in d.get("links", {}).items():
            table._links[f] = BetaCounts(pa, pb, pos, neg)
        return table
