"""``python -m heapdb`` / ``heapdb``: a minimal REPL over the query language.

Line handling (statements and ``.meta`` commands) is implemented in the Rust
engine (``heapdb::repl``, shared with the native ``heapdb`` binary); this
module only does terminal I/O. See ``HELP`` for the commands.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, TextIO

from ._core import HELP, HeapDB, QueryError, handle_line, load_demo

__all__ = ["HELP", "handle_line", "load_demo", "main", "repl"]

_ERRORS = (QueryError, KeyError, ValueError, IndexError, OSError)


def repl(db: HeapDB, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout,
         clock: Callable[[], float] = time.time) -> None:
    interactive = stdin.isatty()
    if interactive:
        print("HeapDB REPL. Type .help for commands, .quit to exit.", file=stdout)
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
        except _ERRORS as e:
            out = f"error: {e}"
        if out:
            print(out, file=stdout)


def run_files(db: HeapDB, paths: list[str], stdout: TextIO = sys.stdout,
              clock: Callable[[], float] = time.time) -> int:
    """Execute query files line by line; returns the number of failed lines."""
    failed = 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    db, out = handle_line(db, line, clock)
                except EOFError:
                    return failed
                except _ERRORS as e:
                    failed += 1
                    out = f"error: {e}"
                if out:
                    print(out, file=stdout)
    return failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="heapdb", description="HeapDB REPL")
    ap.add_argument("--load", help="JSONL snapshot to load at start")
    ap.add_argument("--demo", action="store_true", help="preload a small bloating demo")
    ap.add_argument("--bucket", type=float, default=900.0, help="bucket width in seconds")
    ap.add_argument("files", nargs="*", help="query files to execute (then exit)")
    ns = ap.parse_args(argv)
    db = HeapDB.load(ns.load) if ns.load else HeapDB(bucket_seconds=ns.bucket)
    if ns.demo:
        load_demo(db, time.time())
    if ns.files:
        return 1 if run_files(db, ns.files) else 0
    repl(db)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
