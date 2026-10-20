"""``python -m heapdb``: a minimal REPL over the query language.

Statements::

    TOP <k> CAUSES OF <effect> FOR <entity>[,<entity>...] [AT <time>]
        [WITHIN <duration>] [EXPLAIN]          several entities = group query
    FEEDBACK <effect> FOR <entity> AT <time> CONFIRMED <id> [REJECTED <id>,...]
    INSERT <type> FOR <entity> [AT <time>] [FEATURES f1,f2] [WEIGHT <dose>]
        [ATTRS k=v,k=v]
    CALIBRATION OF <effect> [BINS <n>]
    RISK OF <effect> FOR <entity> FEATURES f1,f2 [WEIGHT <dose>] [TYPE <t>]
        [ATTRS k=v,k=v] [EXPLAIN]      pre-event risk of a candidate that has
                                       not happened yet (e.g. an open PR)

``<time>``: seconds, now, now-2h, ISO-8601. ``<duration>``: 90, 30m, 12h, 2d.

Meta commands::

    .effect <name> <kernel> <args...>   register an effect, e.g.
                                        .effect bloating lognormal 2h 0.6
                                        .effect latency exp 30m
                                        .effect incident window 0 1h
    .prior <effect> <feature> <a> <b>   set a Beta prior for a feature
    .rules <effect>                     list plausibility rules (rules are
                                        Python callables: add them via the API)
    .forget <entity> [scrub]            delete an entity (scrub: also its
                                        contributions to global links)
    .events [entity]                    list stored events
    .save <path> / .load <path>         JSONL snapshot
    .help / .quit

Start with ``--load snapshot.jsonl`` to resume a saved database, and
``--demo`` to preload a small bloating example.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, TextIO

from .kernels import ExponentialDecay, GammaKernel, Kernel, LogNormalKernel, WindowKernel
from .query import QueryError, execute, format_result, parse_duration
from .store import CalibrationReport, HeapDB, QueryResult, RiskResult

HELP = __doc__ or ""


def _make_kernel(kind: str, args: list[str]) -> Kernel:
    kind = kind.lower()
    if kind == "exp":
        return ExponentialDecay(half_life=parse_duration(args[0]))
    if kind == "window":
        return WindowKernel(parse_duration(args[0]), parse_duration(args[1]))
    if kind == "lognormal":
        return LogNormalKernel(median=parse_duration(args[0]), sigma=float(args[1]))
    if kind == "gamma":
        return GammaKernel(shape=float(args[0]), scale=parse_duration(args[1]))
    raise QueryError(f"unknown kernel {kind!r} (exp, window, lognormal, gamma)")


def load_demo(db: HeapDB, now: float) -> None:
    """A few meals over the last day, relative to ``now``."""
    db.register_effect("bloating", LogNormalKernel(median=2 * 3600, sigma=0.6),
                       priors={"beans": (3, 1), "dairy": (2, 1)})
    for hours_ago, feats, w in [(9, "oats,dairy", 1.0), (5, "bread,cheese,dairy", 1.0),
                                (2.5, "beans,onion,rice", 1.5), (1, "apple", 0.5)]:
        db.insert("me", now - hours_ago * 3600, "meal", feats, w)


def handle_line(db: HeapDB, line: str, clock: Callable[[], float]) -> tuple[HeapDB, str | None]:
    """Process one REPL line. Returns the (possibly replaced) db and output."""
    line = line.strip()
    if not line or line.startswith("#"):
        return db, None
    if line.startswith("."):
        parts = line.split()
        cmd, args = parts[0].lower(), parts[1:]
        if cmd in (".quit", ".exit"):
            raise EOFError
        if cmd == ".help":
            return db, HELP
        if cmd == ".effect":
            if len(args) < 3:
                raise QueryError("usage: .effect <name> <kernel> <args...>")
            db.register_effect(args[0], _make_kernel(args[1], args[2:]))
            return db, f"ok: effect {args[0]} with {db.effect(args[0]).kernel!r}"
        if cmd == ".prior":
            if len(args) != 4:
                raise QueryError("usage: .prior <effect> <feature> <alpha> <beta>")
            db.effect(args[0]).links.set_prior(args[1], float(args[2]), float(args[3]))
            return db, f"ok: prior {args[1]} ~ Beta({args[2]}, {args[3]})"
        if cmd == ".rules":
            if len(args) != 1:
                raise QueryError("usage: .rules <effect>")
            spec = db.effect(args[0])
            lines = [f"{r.name}: x{r.multiplier:g}"
                     + (f" for types {sorted(r.types)}" if r.types is not None else "")
                     + ("" if r.predicate else " (type-level)")
                     for r in spec.rules]
            lines += [f"{n}: (pending; re-register after load)" for n in spec.pending_rules]
            return db, "\n".join(lines) or "(no rules)"
        if cmd == ".forget":
            if len(args) not in (1, 2) or (len(args) == 2 and args[1].lower() != "scrub"):
                raise QueryError("usage: .forget <entity> [scrub]")
            out = db.forget_entity(args[0], scrub_global=len(args) == 2)
            return db, "ok: forgot " + ", ".join(f"{v} {k}" for k, v in out.items())
        if cmd == ".events":
            evs = sorted(db.events(args[0] if args else None), key=lambda e: e.time)
            return db, "\n".join(f"{e.entity}: {e.short()}" for e in evs) or "(empty)"
        if cmd == ".save":
            db.save(args[0])
            return db, f"ok: saved {len(db)} events to {args[0]}"
        if cmd == ".load":
            db = HeapDB.load(args[0])
            return db, f"ok: loaded {db!r}"
        raise QueryError(f"unknown command {cmd!r}; try .help")
    out = execute(db, line, clock=clock)
    if isinstance(out, (QueryResult, CalibrationReport, RiskResult)):
        return db, format_result(out)
    return db, out


def repl(db: HeapDB, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout,
         clock: Callable[[], float] = time.time) -> None:
    interactive = stdin.isatty()
    if interactive:
        print("HeapDB reference REPL. Type .help for commands, .quit to exit.", file=stdout)
    while True:
        if interactive:
            stdout.write("heapdb> ")
            stdout.flush()
        line = stdin.readline()
        if not line:
            break
        try:
            db, out = handle_line(db, line, clock)
        except EOFError:
            break
        except (QueryError, KeyError, ValueError, IndexError, OSError) as e:
            out = f"error: {e}"
        if out:
            print(out, file=stdout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m heapdb", description="HeapDB REPL")
    ap.add_argument("--load", help="JSONL snapshot to load at start")
    ap.add_argument("--demo", action="store_true", help="preload a small bloating demo")
    ap.add_argument("--bucket", type=float, default=900.0, help="bucket width in seconds")
    ns = ap.parse_args(argv)
    db = HeapDB.load(ns.load) if ns.load else HeapDB(bucket_seconds=ns.bucket)
    if ns.demo:
        load_demo(db, time.time())
    repl(db)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
