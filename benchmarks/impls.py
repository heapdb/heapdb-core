"""The same ranking problem, implemented several ways.

Problem: for effect ``incident`` observed for entity ``x`` at time ``T``,
return the top-k events of ``x`` by::

    score = kernel(T - t) * weight * (1 - prod_{f in features} (1 - s_f))

over events with ``T - kernel.support <= t <= T - kernel.min_lag``, dropping
score 0, ordered by ``(score desc, event id desc)``. Link strengths ``s_f``
are static (no feedback during latency runs) and identical across
implementations.

Implementations
---------------
* ``heapdb``        -- ``HeapDB.top_causes``: segment bounds + best-first scan
                       + early stop.
* ``bruteforce``    -- same Event objects and the *same* scoring function
                       (``HeapDB.score``, i.e. heapdb's kernel + noisy-OR
                       link code) applied to every event in the kernel window
                       (found with ``bisect`` on a per-entity time array),
                       then ``heapq.nlargest``. Since v0.5 the scoring
                       function is the Rust engine's, called once per event
                       from this Python loop.
* ``bruteforce-rust`` -- ``HeapDB.top_causes_naive``: inside the Rust
                       engine, score every event of the entity's segments
                       overlapping the window (same scoring code as
                       ``heapdb``), full sort, cut. No bounds, no pruning:
                       only the search strategy differs from ``heapdb``.
* ``sqlite-mem`` / ``sqlite-disk`` -- stdlib ``sqlite3``. Tables
                       ``events(id, entity, time, weight)``,
                       ``event_features(event_id, feature)`` and a temp table
                       ``links(feature, log_miss = ln(1 - s))``; index on
                       ``events(entity, time)``. **Everything runs in SQL**:
                       window filter, the kernel (``exp``/``ln``/``power``
                       math, no Python UDF), noisy-OR as
                       ``1 - exp(sum(log_miss))`` with ``GROUP BY``, ordering
                       and ``LIMIT``. Python only binds parameters and fetches
                       k rows.
* ``postgres``      -- the same SQL (``LEAST`` instead of 2-arg ``min``) on
                       PostgreSQL via psycopg 3; links passed as a temp table.
"""

from __future__ import annotations

import bisect
import heapq
import math
import os
import sqlite3
import tempfile
import time
from typing import Any

from heapdb import ExponentialDecay, Kernel, LogNormalKernel, WindowKernel
from heapdb.events import Event

from .workload import DEFAULT_BUCKET, EFFECT, EVENT_TYPE, Workload, make_scorer

Result = list[tuple[float, int]]  # (score, event id), best first


def window(kernel: Kernel, at: float) -> tuple[float, float]:
    """Same window as ``HeapDB._window`` (no ``within`` cap)."""
    return at - kernel.support, at - max(kernel.min_lag, 0.0)


class Impl:
    name = "impl"
    #: What the "index" phase means for this implementation.
    index_note = ""

    def __init__(self, wl: Workload, kernel: Kernel, **opts: Any) -> None:
        self.wl = wl
        self.kernel = kernel
        self.opts = opts
        self.ingest_seconds = 0.0
        self.index_seconds = 0.0

    def build(self) -> None:
        raise NotImplementedError

    def set_kernel(self, kernel_name: str, kernel: Kernel) -> None:
        """Switch the kernel used by later queries (data is not rebuilt)."""
        self.kernel = kernel

    def query(self, entity: int, at: float, k: int) -> Result:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def info(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# HeapDB
# ---------------------------------------------------------------------------


class HeapDBImpl(Impl):
    name = "heapdb"
    index_note = "segment summaries + inverted index maintained on insert (in ingest)"

    def build(self) -> None:
        wl = self.wl
        self.db = make_scorer(wl, self.kernel, self.opts.get("bucket", DEFAULT_BUCKET))
        self.effect = EFFECT
        ents, vocab = wl.entities, wl.vocab
        insert = self.db.insert
        t0 = time.perf_counter()
        for e, t, fs, w in zip(wl.entity, wl.time, wl.features, wl.weight):
            insert(ents[e], t, EVENT_TYPE, [vocab[i] for i in fs], weight=w)
        self.ingest_seconds = time.perf_counter() - t0
        self.index_seconds = 0.0
        self.last_stats = None

    def query(self, entity: int, at: float, k: int) -> Result:
        res = self.db.top_causes(self.wl.entities[entity], self.effect, at, k=k)
        self.last_stats = res.stats
        return [(c.score, c.event.id) for c in res.causes]

    def set_kernel(self, kernel_name: str, kernel: Kernel) -> None:
        # One effect per kernel, all sharing the same priors and the same log.
        self.kernel = kernel
        self.effect = f"{EFFECT}:{kernel_name}"
        if self.effect not in self.db.effects:
            self.db.register_effect(self.effect, kernel, priors=self.wl.priors())

    def info(self) -> dict[str, Any]:
        return {"segments": sum(self.db.entity_log(e).num_segments for e in self.db.entities),
                "bucket_seconds": self.db.bucket_seconds}


# ---------------------------------------------------------------------------
# Brute force (same scoring code, no pruning)
# ---------------------------------------------------------------------------


class BruteForceImpl(Impl):
    name = "bruteforce"
    index_note = "per-entity sorted time arrays for bisect"

    def build(self) -> None:
        wl = self.wl
        # Same scoring function as HeapDB: a HeapDB used only for .score().
        self.scorer = make_scorer(wl, self.kernel)
        ents, vocab = wl.entities, wl.vocab
        per: list[list[Event]] = [[] for _ in ents]
        t0 = time.perf_counter()
        for i, (e, t, fs, w) in enumerate(zip(wl.entity, wl.time, wl.features, wl.weight)):
            per[e].append(Event(id=i + 1, entity=ents[e], time=t, type=EVENT_TYPE,
                                features=frozenset(vocab[j] for j in fs), weight=w))
        self.ingest_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        for lst in per:
            lst.sort(key=lambda ev: (ev.time, ev.id))  # already sorted: O(n) check
        self.times = [[ev.time for ev in lst] for lst in per]
        self.events = per
        self.index_seconds = time.perf_counter() - t0
        self.last_window = 0

    def set_kernel(self, kernel_name: str, kernel: Kernel) -> None:
        self.kernel = kernel
        self.scorer = make_scorer(self.wl, kernel)

    def query(self, entity: int, at: float, k: int) -> Result:
        t_lo, t_hi = window(self.kernel, at)
        ts = self.times[entity]
        i = bisect.bisect_left(ts, t_lo)
        j = bisect.bisect_right(ts, t_hi)
        score = self.scorer.score
        evs = self.events[entity]
        self.last_window = j - i
        scored = []
        for ev in evs[i:j]:
            s = score(ev, EFFECT, at)
            if s > 0.0:
                scored.append((s, ev.id))
        return heapq.nlargest(k, scored)


class BruteForceRustImpl(HeapDBImpl):
    """Same Rust engine and data as ``heapdb``; queries skip the bounds and
    scan the whole window (``HeapDB.top_causes_naive``)."""

    name = "bruteforce-rust"

    def query(self, entity: int, at: float, k: int) -> Result:
        out = self.db.top_causes_naive(self.wl.entities[entity], self.effect, at, k=k)
        self.last_stats = None
        return [(s, ev.id) for s, ev in out]


# ---------------------------------------------------------------------------
# SQL (SQLite and PostgreSQL)
# ---------------------------------------------------------------------------


AT = object()  # sentinel: bind the effect time here


def kernel_sql(kernel: Kernel, least: str, ph: str) -> tuple[str, list[Any]]:
    """SQL for ``kernel.value(at - c.time)`` inside the window, with binds.

    Mirrors heapdb.kernels: peak-normalised, ``min(1, raw(dt) / raw(mode))``.
    Returns the expression and the bind values in textual order (``AT``
    marks where the effect time goes).
    """
    dt = f"({ph} - c.time)"
    if isinstance(kernel, ExponentialDecay):
        return f"power(0.5, {dt} / {ph})", [AT, kernel.half_life]
    if isinstance(kernel, WindowKernel):
        return "1.0", []
    if isinstance(kernel, LogNormalKernel):
        mu, sg, peak = kernel._mu, kernel.sigma, kernel._raw(kernel.mode)
        z = f"((ln({dt}) - {ph}) / {ph})"
        expr = (f"CASE WHEN {dt} > 0 THEN {least}(1.0, exp(-0.5 * {z} * {z}) / {dt} / {ph}) "
                f"ELSE 0.0 END")
        return expr, [AT, AT, mu, sg, AT, mu, sg, AT, peak]
    raise NotImplementedError(f"no SQL for kernel {type(kernel).__name__}")


class _SQLBase(Impl):
    least = "min"
    placeholder = "?"

    def _prepare_sql(self) -> None:
        ph = self.placeholder
        kexpr, self._kbinds = kernel_sql(self.kernel, self.least, ph)
        self.sql = f"""
WITH cand AS (
    SELECT id, time, weight FROM events
    WHERE entity = {ph} AND time >= {ph} AND time <= {ph}
), lk AS (
    SELECT f.event_id AS id, 1.0 - exp(sum(l.log_miss)) AS link
    FROM cand c
    JOIN event_features f ON f.event_id = c.id
    JOIN links l ON l.feature = f.feature
    GROUP BY f.event_id
), scored AS (
    SELECT c.id, ({kexpr}) * c.weight * lk.link AS score
    FROM cand c JOIN lk ON lk.id = c.id
)
SELECT score, id FROM scored WHERE score > 0
ORDER BY score DESC, id DESC
LIMIT {ph}
"""

    def set_kernel(self, kernel_name: str, kernel: Kernel) -> None:
        self.kernel = kernel
        self._prepare_sql()

    def _params(self, entity: int, at: float, k: int) -> list[Any]:
        t_lo, t_hi = window(self.kernel, at)
        kp = [at if t is AT else t for t in self._kbinds]
        return [entity, t_lo, t_hi, *kp, k]

    def link_rows(self) -> list[tuple[int, float]]:
        """(feature index, ln(1 - s)) using heapdb's own strength numbers."""
        scorer = make_scorer(self.wl, self.kernel)
        links = scorer.effect(EFFECT).links
        return [(i, math.log1p(-links.strength(f))) for i, f in enumerate(self.wl.vocab)]

    def event_rows(self):
        wl = self.wl
        return ((i + 1, e, t, w) for i, (e, t, w) in enumerate(zip(wl.entity, wl.time, wl.weight)))

    def feature_rows(self):
        for i, fs in enumerate(self.wl.features):
            for f in fs:
                yield (i + 1, f)


class SQLiteImpl(_SQLBase):
    name = "sqlite"
    index_note = "CREATE INDEX events(entity, time) + ANALYZE (PKs built during ingest)"

    def build(self) -> None:
        self.on_disk = bool(self.opts.get("on_disk"))
        self.name = "sqlite-disk" if self.on_disk else "sqlite-mem"
        self._tmpdir = None
        if self.on_disk:
            self._tmpdir = tempfile.mkdtemp(prefix="heapdb-bench-sqlite-")
            self.path = os.path.join(self._tmpdir, "bench.db")
        else:
            self.path = ":memory:"
        con = self.con = sqlite3.connect(self.path, isolation_level=None)
        con.execute("PRAGMA cache_size = -2000000")  # up to ~2 GB page cache: warm queries
        con.execute("PRAGMA temp_store = MEMORY")
        if self.on_disk:
            con.execute("PRAGMA journal_mode = WAL")
            con.execute("PRAGMA synchronous = NORMAL")
        con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, entity INTEGER NOT NULL, "
                    "time REAL NOT NULL, weight REAL NOT NULL)")
        con.execute("CREATE TABLE event_features (event_id INTEGER NOT NULL, "
                    "feature INTEGER NOT NULL, PRIMARY KEY (event_id, feature)) WITHOUT ROWID")
        con.execute("CREATE TEMP TABLE links (feature INTEGER PRIMARY KEY, log_miss REAL NOT NULL)")
        t0 = time.perf_counter()
        con.execute("BEGIN")
        con.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", self.event_rows())
        con.executemany("INSERT INTO event_features VALUES (?, ?)", self.feature_rows())
        con.executemany("INSERT INTO links VALUES (?, ?)", self.link_rows())
        con.execute("COMMIT")
        self.ingest_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        con.execute("CREATE INDEX events_entity_time ON events (entity, time)")
        con.execute("ANALYZE")
        self.index_seconds = time.perf_counter() - t0
        self._prepare_sql()

    def query(self, entity: int, at: float, k: int) -> Result:
        return [(s, i) for s, i in self.con.execute(self.sql, self._params(entity, at, k))]

    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sqlite_version": sqlite3.sqlite_version, "path": self.path}
        if self.on_disk:
            self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            out["db_file_bytes"] = os.path.getsize(self.path)
        return out

    def close(self) -> None:
        self.con.close()
        if self._tmpdir:
            import shutil
            shutil.rmtree(self._tmpdir, ignore_errors=True)


class PostgresImpl(_SQLBase):
    name = "postgres"
    least = "LEAST"
    placeholder = "%s"
    index_note = "PKs + index events(entity, time) + ANALYZE, built after COPY"

    def build(self) -> None:
        import psycopg  # noqa: F401  (optional dependency)
        dsn = self.opts["dsn"]
        self.con = psycopg.connect(dsn, autocommit=True)
        cur = self.cur = self.con.cursor()
        cur.execute("DROP TABLE IF EXISTS event_features, events")
        cur.execute("CREATE TABLE events (id bigint NOT NULL, entity integer NOT NULL, "
                    "time double precision NOT NULL, weight double precision NOT NULL)")
        cur.execute("CREATE TABLE event_features (event_id bigint NOT NULL, "
                    "feature integer NOT NULL)")
        cur.execute("CREATE TEMP TABLE links (feature integer PRIMARY KEY, "
                    "log_miss double precision NOT NULL)")
        t0 = time.perf_counter()
        with self.con.transaction():
            self._copy("COPY events (id, entity, time, weight) FROM STDIN",
                       (f"{i}\t{e}\t{t!r}\t{w!r}\n" for i, e, t, w in self.event_rows()))
            self._copy("COPY event_features (event_id, feature) FROM STDIN",
                       (f"{i}\t{f}\n" for i, f in self.feature_rows()))
            self._copy("COPY links (feature, log_miss) FROM STDIN",
                       (f"{f}\t{v!r}\n" for f, v in self.link_rows()))
        self.ingest_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        cur.execute("ALTER TABLE events ADD PRIMARY KEY (id)")
        cur.execute("ALTER TABLE event_features ADD PRIMARY KEY (event_id, feature)")
        cur.execute("CREATE INDEX events_entity_time ON events (entity, time)")
        cur.execute("ANALYZE events")
        cur.execute("ANALYZE event_features")
        cur.execute("ANALYZE links")
        self.index_seconds = time.perf_counter() - t0
        self._prepare_sql()

    def _copy(self, stmt: str, lines) -> None:
        with self.cur.copy(stmt) as cp:
            buf: list[str] = []
            for line in lines:
                buf.append(line)
                if len(buf) >= 50_000:
                    cp.write("".join(buf))
                    buf.clear()
            if buf:
                cp.write("".join(buf))

    def query(self, entity: int, at: float, k: int) -> Result:
        self.cur.execute(self.sql, self._params(entity, at, k), prepare=True)
        return [(float(s), int(i)) for s, i in self.cur.fetchall()]

    def info(self) -> dict[str, Any]:
        self.cur.execute("SELECT version(), current_setting('shared_buffers'), "
                         "current_setting('work_mem'), "
                         "pg_total_relation_size('events') + pg_total_relation_size('event_features')")
        v, sb, wm, size = self.cur.fetchone()
        return {"server_version": v, "shared_buffers": sb, "work_mem": wm,
                "relation_bytes": int(size)}

    def close(self) -> None:
        try:
            self.cur.execute("DROP TABLE IF EXISTS event_features, events")
        finally:
            self.con.close()


IMPLS: dict[str, type[Impl]] = {
    "heapdb": HeapDBImpl,
    "bruteforce": BruteForceImpl,
    "bruteforce-rust": BruteForceRustImpl,
    "sqlite-mem": SQLiteImpl,
    "sqlite-disk": SQLiteImpl,
    "postgres": PostgresImpl,
}


def make_impl(name: str, wl: Workload, kernel: Kernel, **opts: Any) -> Impl:
    if name == "sqlite-disk":
        opts["on_disk"] = True
    impl = IMPLS[name](wl, kernel, **opts)
    impl.name = name
    return impl


# ---------------------------------------------------------------------------
# Agreement check
# ---------------------------------------------------------------------------


def compare(ref: Result, got: Result, tol: float = 1e-9) -> str | None:
    """None if ``got`` equals ``ref``, else a description of the difference.

    Equal means: same length; rank by rank, scores equal within ``tol``
    (relative, floor 1); and the same ids, except that ids may be permuted
    within a group of scores tied within ``tol`` (float rounding in SQL math
    can reorder exact ties). The last tied group may be cut by ``k``, so for
    it only the scores must match.
    """
    if len(ref) != len(got):
        return f"length {len(got)} != {len(ref)}"

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= tol * max(1.0, abs(a))

    for i, ((rs, ri), (gs, gi)) in enumerate(zip(ref, got)):
        if not close(rs, gs):
            return f"rank {i + 1}: score {gs!r} (id {gi}) != {rs!r} (id {ri})"
    i = 0
    while i < len(ref):
        j = i + 1
        while j < len(ref) and close(ref[i][0], ref[j][0]):
            j += 1
        r_ids = sorted(x for _, x in ref[i:j])
        g_ids = sorted(x for _, x in got[i:j])
        if r_ids != g_ids and j < len(ref):
            return f"ranks {i + 1}-{j}: ids {g_ids} != {r_ids}"
        i = j
    return None
