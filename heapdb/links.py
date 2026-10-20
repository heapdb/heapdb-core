"""Learned link strengths between features and effect types.

For every ``(effect_type, feature)`` pair HeapDB keeps a Beta(alpha, beta)
belief about "how likely is this feature to produce this effect". The point
estimate used for ranking is the posterior mean ``alpha / (alpha + beta)``.

Three layers (v0.2: hierarchical priors)
---------------------------------------
1. **Registered prior** -- domain knowledge ("beans are a usual suspect for
   bloating"). Features without one use the effect's ``default_prior``.
2. **Global layer** -- evidence pooled over *all* entities. Every feedback
   update goes to both the entity and the global layer.
3. **Entity layer** -- the entity's own evidence.

An entity's effective belief for a feature is::

    alpha = prior_alpha + c * G_pos + own_pos
    beta  = prior_beta  + c * G_neg + own_neg
    c     = min(1, global_pseudocount / (G_pos + G_neg))

where ``G`` is the global evidence *excluding this entity's own
contributions* (leave-one-out, so an entity's evidence is not counted twice).
Global evidence is thus worth at most ``global_pseudocount`` (default 5)
pseudo-observations: a brand-new entity inherits the population tendency,
and its own evidence dominates once it has more than a handful of
observations. With ``hierarchical=False`` the global layer is still tracked
but not borrowed from (each entity learns alone from the registered prior).

Because every feedback update is applied identically to the entity and the
global layer, the entity layer doubles as the ledger of that entity's
contributions to the global layer. That is what makes
``forget_entity(..., scrub_global=True)`` possible.

Credit assignment (v0.2: proportional)
-------------------------------------
An event usually has several features. We combine them with *noisy-OR*:
``P(effect | event) = 1 - prod_f (1 - s_f)`` -- the event causes the effect
unless every one of its features independently fails to.

* **Confirmation** uses the EM update for noisy-OR. Each feature's
  *responsibility* for the confirmed effect is
  ``r_f = s_f / (1 - prod_g (1 - s_g))`` (clamped to [0, 1]); then
  ``alpha_f += r_f`` and ``beta_f += 1 - r_f``. A feature that is already a
  strong explanation takes most of the credit; a weak rider that merely
  co-occurs takes little, instead of being inflated alongside it.
* **Rejection** adds ``beta_f += 1`` to every feature: under noisy-OR, the
  event not causing the effect means every feature independently failed.
* ``credit="uniform"`` restores the v0.1 behaviour (``alpha_f += 1`` for every
  feature of a confirmed event).

Counts are therefore floats.

Joint links
-----------
Features that have *always* co-occurred cannot be told apart by feedback
("was it the coffee or the milk?"). For reporting, HeapDB keeps a joint
Beta link per entity for each such inseparable feature set that has
appeared on a feedback event (keyed by ``frozenset``), created lazily. Joint
links are bounded by the number of distinct sets ever judged and are *not*
used for scoring, which stays per-feature noisy-OR.

Noisy-OR is monotone in the feature set, which the query engine exploits for
its segment bounds (see :mod:`heapdb.store`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

CREDIT_MODES = ("proportional", "uniform")

#: Pseudo-observation weight of a joint link's prior (see LinkTable.joint).
JOINT_PRIOR_STRENGTH = 2.0


@dataclass(frozen=True)
class BetaCounts:
    """A Beta(alpha, beta) belief, split into where the evidence came from.

    ``alpha = prior_alpha + global_pos + pos`` (likewise for beta). For an
    entity-level belief, ``global_pos``/``global_neg`` are the *shrunk*
    global evidence actually borrowed and ``pos``/``neg`` the entity's own
    (float) counts. For the global (population) view, ``global_*`` hold
    the full pooled counts and ``pos``/``neg`` are 0.
    """

    prior_alpha: float
    prior_beta: float
    pos: float = 0.0
    neg: float = 0.0
    global_pos: float = 0.0
    global_neg: float = 0.0

    @property
    def alpha(self) -> float:
        return self.prior_alpha + self.global_pos + self.pos

    @property
    def beta(self) -> float:
        return self.prior_beta + self.global_neg + self.neg

    @property
    def strength(self) -> float:
        """Posterior mean of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def observations(self) -> float:
        """Own + borrowed evidence (excludes the prior)."""
        return self.pos + self.neg + self.global_pos + self.global_neg


def noisy_or(strengths: Iterable[float]) -> float:
    """``1 - prod(1 - s)``; 0.0 for an empty iterable."""
    miss = 1.0
    for s in strengths:
        miss *= 1.0 - s
    return 1.0 - miss


def responsibilities(strengths: Mapping[str, float]) -> dict[str, float]:
    """EM responsibilities of each feature for a confirmed noisy-OR effect.

    ``r_f = s_f / (1 - prod_g (1 - s_g))``, clamped to [0, 1]. If every
    strength is 0 (noisy-OR is 0) credit is split uniformly.
    """
    if not strengths:
        return {}
    p = noisy_or(strengths.values())
    if p <= 0.0:
        share = 1.0 / len(strengths)
        return {f: share for f in strengths}
    return {f: min(1.0, max(0.0, s / p)) for f, s in strengths.items()}


def _fmt(x: float) -> str:
    """Counts are floats now; print integers without noise."""
    return f"{x:g}" if abs(x - round(x)) < 1e-9 else f"{x:.2f}"


class LinkTable:
    """Hierarchical Beta link beliefs for one effect type.

    Args:
        default_prior: Beta prior for features without an explicit prior.
        priors: feature -> (alpha, beta).
        global_pseudocount: Cap on how many pseudo-observations the global
            layer may contribute to an entity's belief.
        hierarchical: If False, entities do not borrow from the global layer.
    """

    def __init__(
        self,
        default_prior: tuple[float, float] = (1.0, 4.0),
        priors: Mapping[str, tuple[float, float]] | None = None,
        global_pseudocount: float = 5.0,
        hierarchical: bool = True,
    ) -> None:
        a, b = default_prior
        if a <= 0 or b <= 0:
            raise ValueError("Beta prior parameters must be > 0")
        if global_pseudocount < 0:
            raise ValueError("global_pseudocount must be >= 0")
        self.default_prior = (float(a), float(b))
        self.global_pseudocount = float(global_pseudocount)
        self.hierarchical = bool(hierarchical)
        self._priors: dict[str, tuple[float, float]] = {}
        self._global: dict[str, list[float]] = {}            # f -> [pos, neg]
        self._entity: dict[str, dict[str, list[float]]] = {}  # entity -> f -> [pos, neg]
        self._joint: dict[str, dict[frozenset[str], list[float]]] = {}
        # (entity or None) -> feature -> strength. Cleared on every write.
        self._cache: dict[str | None, dict[str, float]] = {}
        for feature, (pa, pb) in (priors or {}).items():
            self.set_prior(feature, pa, pb)

    # -- reads --------------------------------------------------------------

    def prior(self, feature: str) -> tuple[float, float]:
        return self._priors.get(feature, self.default_prior)

    def get(self, feature: str, entity: str | None = None) -> BetaCounts:
        """Effective belief for ``feature``: global view, or for ``entity``."""
        pa, pb = self.prior(feature)
        gp, gn = self._global.get(feature, (0.0, 0.0))
        if entity is None:
            return BetaCounts(pa, pb, global_pos=gp, global_neg=gn)
        own = self._entity.get(entity, {}).get(feature)
        op, on = own if own is not None else (0.0, 0.0)
        bp = bn = 0.0
        if self.hierarchical and self.global_pseudocount > 0:
            # Leave-one-out: borrow only *other* entities' evidence.
            bp, bn = max(0.0, gp - op), max(0.0, gn - on)
            total = bp + bn
            if total > self.global_pseudocount:
                c = self.global_pseudocount / total
                bp, bn = bp * c, bn * c
        return BetaCounts(pa, pb, op, on, bp, bn)

    def strength(self, feature: str, entity: str | None = None) -> float:
        cache = self._cache.get(entity)
        if cache is None:
            cache = self._cache[entity] = {}
        s = cache.get(feature)
        if s is None:
            s = cache[feature] = self.get(feature, entity).strength
        return s

    def strengths(self, features: Iterable[str], entity: str | None = None) -> dict[str, float]:
        return {f: self.strength(f, entity) for f in features}

    def noisy_or(self, features: Iterable[str], entity: str | None = None) -> float:
        """Aggregate strength of a feature set (see module docstring)."""
        return noisy_or(self.strength(f, entity) for f in features)

    def items(self) -> Iterable[tuple[str, BetaCounts]]:
        """Global view of every feature with a prior or global evidence."""
        for f in sorted(set(self._priors) | set(self._global)):
            yield f, self.get(f)

    def entities(self) -> list[str]:
        return sorted(self._entity)

    def entity_counts(self, entity: str) -> dict[str, tuple[float, float]]:
        """The entity's own (pos, neg) counts = its contributions to global."""
        return {f: (p, n) for f, (p, n) in self._entity.get(entity, {}).items()}

    # -- joint links ----------------------------------------------------------

    def joint_prior(self, features: Iterable[str]) -> tuple[float, float]:
        """Prior for a joint link: noisy-OR of the members' prior means,
        worth ``JOINT_PRIOR_STRENGTH`` pseudo-observations."""
        p = noisy_or(a / (a + b) for a, b in map(self.prior, features))
        p = min(max(p, 1e-6), 1 - 1e-6)
        return JOINT_PRIOR_STRENGTH * p, JOINT_PRIOR_STRENGTH * (1 - p)

    def joint(self, entity: str, features: Iterable[str]) -> BetaCounts | None:
        """Joint link for an exact feature set, or None if never judged."""
        fs = frozenset(features)
        counts = self._joint.get(entity, {}).get(fs)
        if counts is None:
            return None
        pa, pb = self.joint_prior(fs)
        return BetaCounts(pa, pb, counts[0], counts[1])

    def joint_sets(self, entity: str) -> list[frozenset[str]]:
        return list(self._joint.get(entity, {}))

    def update_joint(self, entity: str, features: Iterable[str], pos: float, neg: float) -> None:
        fs = frozenset(features)
        counts = self._joint.setdefault(entity, {}).setdefault(fs, [0.0, 0.0])
        counts[0] += pos
        counts[1] += neg

    # -- writes -------------------------------------------------------------

    def set_prior(self, feature: str, alpha: float, beta: float) -> None:
        """Set (or replace) the prior for one feature, keeping observed counts."""
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"Beta prior for {feature!r} must be > 0")
        self._priors[feature] = (float(alpha), float(beta))
        self._cache.clear()

    def update(self, feature: str, pos: float, neg: float, entity: str | None = None) -> None:
        """Add evidence to the global layer and (if given) the entity layer."""
        g = self._global.setdefault(feature, [0.0, 0.0])
        g[0] += pos
        g[1] += neg
        if entity is not None:
            e = self._entity.setdefault(entity, {}).setdefault(feature, [0.0, 0.0])
            e[0] += pos
            e[1] += neg
        self._cache.clear()

    def confirm(
        self,
        features: Iterable[str],
        entity: str | None = None,
        credit: str = "proportional",
    ) -> dict[str, float]:
        """Record a confirmed cause. Returns each feature's credit ``r_f``."""
        if credit not in CREDIT_MODES:
            raise ValueError(f"credit must be one of {CREDIT_MODES}, got {credit!r}")
        feats = sorted(set(features))
        if credit == "uniform":
            resp = {f: 1.0 for f in feats}
        else:
            resp = responsibilities(self.strengths(feats, entity))
        for f in feats:
            r = resp[f]
            self.update(f, r, 1.0 - r, entity)
        return resp

    def reject(self, features: Iterable[str], entity: str | None = None) -> None:
        """Record a rejected candidate: every feature independently failed."""
        for f in sorted(set(features)):
            self.update(f, 0.0, 1.0, entity)

    def forget_entity(self, entity: str, scrub_global: bool = False) -> int:
        """Drop an entity's links (and joint links). Returns #features dropped.

        With ``scrub_global`` the entity's contributions are also subtracted
        from the global layer.
        """
        own = self._entity.pop(entity, {})
        self._joint.pop(entity, None)
        if scrub_global:
            for f, (p, n) in own.items():
                g = self._global.get(f)
                if g is None:
                    continue
                g[0] = max(0.0, g[0] - p)
                g[1] = max(0.0, g[1] - n)
                if g[0] < 1e-12 and g[1] < 1e-12:
                    del self._global[f]
        self._cache.clear()
        return len(own)

    # -- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "default_prior": list(self.default_prior),
            "global_pseudocount": self.global_pseudocount,
            "hierarchical": self.hierarchical,
            "priors": {f: list(p) for f, p in sorted(self._priors.items())},
            "global": {f: list(c) for f, c in sorted(self._global.items())},
            "entities": {
                e: {f: list(c) for f, c in sorted(fs.items())}
                for e, fs in sorted(self._entity.items())
            },
            "joint": {
                e: [[sorted(fs), c[0], c[1]] for fs, c in sorted(js.items(), key=lambda kv: sorted(kv[0]))]
                for e, js in sorted(self._joint.items())
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LinkTable":
        table = cls(
            default_prior=tuple(d["default_prior"]),  # type: ignore[arg-type]
            global_pseudocount=d.get("global_pseudocount", 5.0),
            hierarchical=d.get("hierarchical", True),
        )
        if "links" in d:  # v0.1 snapshot: one pooled table of [pa, pb, pos, neg]
            for f, (pa, pb, pos, neg) in d["links"].items():
                table._priors[f] = (float(pa), float(pb))
                if pos or neg:
                    table._global[f] = [float(pos), float(neg)]
            return table
        table._priors = {f: (float(a), float(b)) for f, (a, b) in d.get("priors", {}).items()}
        table._global = {f: [float(p), float(n)] for f, (p, n) in d.get("global", {}).items()}
        table._entity = {
            e: {f: [float(p), float(n)] for f, (p, n) in fs.items()}
            for e, fs in d.get("entities", {}).items()
        }
        table._joint = {
            e: {frozenset(fs): [float(p), float(n)] for fs, p, n in js}
            for e, js in d.get("joint", {}).items()
        }
        return table
