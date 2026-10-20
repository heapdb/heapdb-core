"""Seeded synthetic workload: deploy-like events, effects at random times.

This is a scaled-up version of the *shape* of ``examples/code_changes.py``
(entity = service, event = a merged change / deploy, dose =
``log1p(lines_changed) / log1p(100)``), not its hand-written vocabulary:

* ``n_entities`` entities (default 100), events spread uniformly over
  ``days`` (default 30), so the number of events inside a kernel window
  grows linearly with N (the sizes compare "more history per entity").
* each event has 1..``max_features`` features drawn from a vocabulary of
  ``vocab`` tokens with Zipf(``zipf_s``) frequencies (duplicates collapse);
* doses: ``lines ~ 1 + LogNormal(log 80, 1.0)``;
* link strengths per feature (independent of frequency rank, from a separate
  seeded stream): profile ``skewed`` gives 3% of features a strength in
  U(0.3, 0.9) and the rest U(0.005, 0.08); profile ``flat`` gives every
  feature 0.1 and every event dose 1.0 (an adversarial case for pruning:
  bounds cannot tell segments apart).

Links are static during latency benchmarks: they are registered as Beta
priors with mean = strength and no feedback is applied, so every
implementation scores with exactly the same numbers.

Everything is deterministic given ``seed``. Data is stored column-wise
(feature tuples of vocabulary indices), and each implementation builds its
own representation from it, so memory deltas are not shared.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

from heapdb import ExponentialDecay, HeapDB, Kernel, LogNormalKernel, WindowKernel

M = 60.0
H = 3600.0
D = 86400.0

EFFECT = "incident"
EVENT_TYPE = "deploy"
PRIOR_STRENGTH = 10.0  # pseudo-observations of the registered link priors

SIZES = {"10k": 10_000, "100k": 100_000, "1m": 1_000_000, "10m": 10_000_000}


def parse_size(s: str) -> int:
    s = s.strip().lower()
    if s in SIZES:
        return SIZES[s]
    return int(float(s))


def size_label(n: int) -> str:
    for k, v in SIZES.items():
        if v == n:
            return k.upper()
    return str(n)


# ---------------------------------------------------------------------------
# Kernels used by the benchmarks
# ---------------------------------------------------------------------------

#: Default kernel = the ``incident`` kernel of examples/code_changes.py.
DEFAULT_KERNEL = "lognormal_3h"
DEFAULT_BUCKET = 15 * M


def make_kernel(name: str) -> Kernel:
    return {
        "lognormal_3h": lambda: LogNormalKernel(median=3 * H, sigma=0.6),
        "exp_10m": lambda: ExponentialDecay(half_life=10 * M),
        "exp_6h": lambda: ExponentialDecay(half_life=6 * H),
        "lognormal_narrow": lambda: LogNormalKernel(median=3 * H, sigma=0.3),
        "lognormal_wide": lambda: LogNormalKernel(median=3 * H, sigma=1.2),
        "window_24h": lambda: WindowKernel(min_lag=0.0, max_lag=24 * H),
    }[name]()


KERNEL_NAMES = ("lognormal_3h", "exp_10m", "exp_6h", "lognormal_narrow",
                "lognormal_wide", "window_24h")


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadSpec:
    n_events: int
    n_entities: int = 100
    days: float = 30.0
    vocab: int = 2000
    zipf_s: float = 1.1
    max_features: int = 6
    strong_fraction: float = 0.03
    links: str = "skewed"  # or "flat"
    seed: int = 0


@dataclass
class Workload:
    spec: WorkloadSpec
    vocab: list[str]
    strengths: list[float]          # per vocabulary index
    entities: list[str]
    # Columns, sorted by time; event id = row index + 1.
    entity: list[int] = field(default_factory=list)
    time: list[float] = field(default_factory=list)
    features: list[tuple[int, ...]] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.time)

    @property
    def span(self) -> float:
        return self.spec.days * D

    def priors(self) -> dict[str, tuple[float, float]]:
        return {f: (s * PRIOR_STRENGTH, (1.0 - s) * PRIOR_STRENGTH)
                for f, s in zip(self.vocab, self.strengths)}


def link_strengths(spec: WorkloadSpec) -> list[float]:
    if spec.links == "flat":
        return [0.1] * spec.vocab
    if spec.links != "skewed":
        raise ValueError(f"unknown link profile {spec.links!r}")
    # Separate stream: strengths do not depend on N.
    rng = random.Random(spec.seed * 1_000_003 + 17)
    return [rng.uniform(0.3, 0.9) if rng.random() < spec.strong_fraction
            else rng.uniform(0.005, 0.08) for _ in range(spec.vocab)]


def generate(spec: WorkloadSpec) -> Workload:
    rng = random.Random(spec.seed)
    vocab = [f"feat:{i:04d}" for i in range(spec.vocab)]
    entities = [f"svc-{i:03d}" for i in range(spec.n_entities)]
    cum = list(itertools.accumulate(1.0 / (r + 1) ** spec.zipf_s for r in range(spec.vocab)))
    idx = range(spec.vocab)
    span = spec.days * D
    flat = spec.links == "flat"
    log100 = math.log1p(100)
    mu = math.log(80)
    ent: list[int] = []
    tim: list[float] = []
    fea: list[tuple[int, ...]] = []
    wei: list[float] = []
    for _ in range(spec.n_events):
        ent.append(rng.randrange(spec.n_entities))
        tim.append(rng.uniform(0.0, span))
        nf = rng.randint(1, spec.max_features)
        fea.append(tuple(sorted(set(rng.choices(idx, cum_weights=cum, k=nf)))))
        lines = int(rng.lognormvariate(mu, 1.0)) + 1
        wei.append(1.0 if flat else math.log1p(lines) / log100)
    order = sorted(range(spec.n_events), key=tim.__getitem__)
    wl = Workload(spec, vocab, link_strengths(spec), entities)
    wl.entity = [ent[i] for i in order]
    wl.time = [tim[i] for i in order]
    wl.features = [fea[i] for i in order]
    wl.weight = [wei[i] for i in order]
    return wl


def queries(wl: Workload, n: int, seed: int | None = None) -> list[tuple[int, float]]:
    """Effects at random (entity, time); times in the second half of the
    span so every kernel used here has a full window."""
    rng = random.Random((wl.spec.seed if seed is None else seed) * 7919 + 1)
    return [(rng.randrange(wl.spec.n_entities), rng.uniform(0.5 * wl.span, wl.span))
            for _ in range(n)]


def make_scorer(wl: Workload, kernel: Kernel, bucket: float = DEFAULT_BUCKET,
                effect: str = EFFECT) -> HeapDB:
    """A HeapDB with the benchmark effect registered (no events)."""
    db = HeapDB(bucket_seconds=bucket)
    db.register_effect(effect, kernel, priors=wl.priors())
    return db


class Reference:
    """Independent top-k straight from the columns, used to verify every
    implementation: heapdb's ``Kernel.value`` and ``LinkTable.noisy_or``
    applied to every event of the entity in the window, then a full sort by
    ``(score desc, id desc)``. No segments, no SQL, no heap."""

    def __init__(self, wl: Workload) -> None:
        import bisect
        self._bisect = bisect
        self.wl = wl
        self.rows: list[list[int]] = [[] for _ in wl.entities]
        for r, e in enumerate(wl.entity):
            self.rows[e].append(r)  # wl is sorted by time
        self.times = [[wl.time[r] for r in rows] for rows in self.rows]
        self.links = make_scorer(wl, ExponentialDecay(1.0)).effect(EFFECT).links

    def topk(self, kernel: Kernel, entity: int, at: float, k: int) -> list[tuple[float, int]]:
        wl = self.wl
        t_lo, t_hi = at - kernel.support, at - max(kernel.min_lag, 0.0)
        ts = self.times[entity]
        i = self._bisect.bisect_left(ts, t_lo)
        j = self._bisect.bisect_right(ts, t_hi)
        ent = wl.entities[entity]
        out = []
        for r in self.rows[entity][i:j]:
            dt = at - wl.time[r]
            if dt < 0:
                continue
            s = kernel.value(dt) * wl.weight[r] * self.links.noisy_or(
                [wl.vocab[f] for f in wl.features[r]], ent)
            if s > 0.0:
                out.append((s, r + 1))
        out.sort(key=lambda x: (-x[0], -x[1]))
        return out[:k]
