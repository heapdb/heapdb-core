"""A work queue for humans and coding agents on a keyed heap.

Run:  python examples/work_queue.py
      HEAPDB_TEST_POSTGRES_URL=postgres://... python examples/work_queue.py   # also on Postgres

Story: tickets arrive over a simulated week. Every event on a ticket (a new
comment, a customer ping, a failing check) is a ``push`` whose weight says
how much it matters; activity decays with a 3-day half-life, so a ticket
that keeps getting pinged rises and one that went quiet sinks, without ever
re-scoring the queue. ``set_base`` carries slow-moving priority (severity).
Workers ``pop`` the hottest ticket they are able to take:

* an agent that knows TypeScript and React only takes tickets whose
  ``required_skills`` are a subset of its skills and that are not labelled
  ``needs-human``,
* a human takes anything, preferring ``needs-human`` work.

At the end the script checks the queue against a brute-force score
(``base * sum w * 2^-(now - t)/half_life``) and prints who did what. The
same code runs on ``Heaps()`` (memory) and, when the environment variable is
set, on ``Heaps("postgres://...")``; both give the same assignments.
"""

from __future__ import annotations

import math
import os
import random
import time

import heapdb

DAY = 86400.0
HALF_LIFE = 3 * DAY
T0 = 1_760_000_000.0  # a fixed clock: the run is reproducible

AGENT = {"name": "agent-ts", "skills": ["ts", "react"]}
HUMAN = {"name": "dana"}
AGENT_FILTER = {"all": [
    {"attr": "required_skills", "op": "subset_of", "value": AGENT["skills"]},
    {"not": {"attr": "labels", "op": "contains", "value": "needs-human"}},
]}
HUMAN_FIRST = "labels CONTAINS 'needs-human'"  # same filter language, as a condition


def simulate(heaps: heapdb.Heaps, seed: int = 7) -> tuple[list[tuple[str, str, float]], dict]:
    """Run the week; returns (assignments, brute-force bookkeeping)."""
    rng = random.Random(seed)
    heaps.drop_heap("acme/checkout")
    heaps.create_heap("acme/checkout", HALF_LIFE, landmark=T0, meta={"team": "checkout"})
    pushes: dict[str, list[tuple[float, float]]] = {}
    base: dict[str, float] = {}
    assignments = []
    now = T0
    n = 0
    for hour in range(7 * 24):
        now = T0 + hour * 3600
        # New tickets.
        if rng.random() < 0.35:
            n += 1
            tid = f"CHK-{n:03d}"
            skills = rng.sample(["ts", "react", "go", "sql"], rng.randint(0, 2))
            labels = ["needs-human"] if rng.random() < 0.25 else []
            kind = rng.choice(["bug", "bug", "task", "chore"])
            heaps.push("acme/checkout", tid, 1.0, t=now,
                       attrs={"kind": kind, "required_skills": skills, "labels": labels})
            pushes[tid] = [(1.0, now)]
            sev = {"bug": 3.0, "task": 1.5, "chore": 1.0}[kind]
            heaps.set_base("acme/checkout", tid, sev)
            base[tid] = sev
        # Activity on open tickets: pings weigh more than comments.
        for tid in list(pushes):
            if rng.random() < 0.04:
                w = rng.choice([0.5, 0.5, 2.0])
                if heaps.get("acme/checkout", tid, now=now) is not None:
                    heaps.push("acme/checkout", tid, w, t=now)
                    pushes[tid].append((w, now))
        # Workers pick up work twice a day: the agent takes what it can, the
        # human prefers needs-human tickets and otherwise takes the top one.
        if hour % 12 == 11:
            picks = [(AGENT["name"], heaps.pop("acme/checkout", now=now, filter=AGENT_FILTER))]
            human = heaps.pop("acme/checkout", now=now, filter=HUMAN_FIRST) or heaps.pop("acme/checkout", now=now)
            picks.append((HUMAN["name"], human))
            for who, got in picks:
                if got is not None:
                    assignments.append((who, got.id, got.score))
                    pushes.pop(got.id)
    return assignments, {"pushes": pushes, "base": base, "now": now}


def check_against_brute_force(heaps: heapdb.Heaps, book: dict) -> int:
    now = book["now"]
    want = sorted(
        ((book["base"][t] * sum(w * 0.5 ** ((now - s) / HALF_LIFE) for w, s in ps), t)
         for t, ps in book["pushes"].items()),
        key=lambda x: (-x[0], x[1]))
    got = heaps.top_k("acme/checkout", 1000, now=now)
    assert [r.id for r in got] == [t for _, t in want], "queue order differs from brute force"
    for r, (score, _) in zip(got, want):
        assert math.isclose(r.score, score, rel_tol=1e-9), (r, score)
    return len(got)


def run(label: str, heaps: heapdb.Heaps) -> list[tuple[str, str, float]]:
    started = time.perf_counter()
    assignments, book = simulate(heaps)
    open_n = check_against_brute_force(heaps, book)
    took = time.perf_counter() - started
    print(f"== {label}: {len(assignments)} tickets assigned, {open_n} still open "
          f"(queue == brute force), {took:.2f}s")
    by_worker: dict[str, list[str]] = {}
    for who, tid, _ in assignments:
        by_worker.setdefault(who, []).append(tid)
    for who, tids in by_worker.items():
        print(f"   {who:9} {len(tids):3} tickets, first: {', '.join(tids[:6])}")
    print("   next up:")
    for r in heaps.top_k("acme/checkout", 3, now=book["now"]):
        print(f"     {r}")
    return assignments


def main() -> None:
    mem = run("memory", heapdb.Heaps())
    url = os.environ.get("HEAPDB_TEST_POSTGRES_URL")
    if url and heapdb._core.HAS_POSTGRES:
        pg = run("postgres", heapdb.Heaps(url))
        assert [a[:2] for a in pg] == [a[:2] for a in mem], "postgres assigned different tickets"
        print("postgres made the same assignments as memory")
    elif url:
        print("(this heapdb build has no Postgres support; skipped)")


if __name__ == "__main__":
    main()
