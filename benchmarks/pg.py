"""Find or start a PostgreSQL server for the benchmarks.

Order of preference:

1. ``--pg-dsn`` / ``HEAPDB_BENCH_PG_DSN`` (e.g. the GitHub Actions service
   container): used as is.
2. ``--pg-local``: find ``initdb``/``pg_ctl`` (``PATH`` or
   ``/usr/lib/postgresql/*/bin``), create a throwaway cluster in a temp
   directory, start it on a Unix socket, and stop + delete it afterwards.
   When running as root, the server runs as the ``postgres`` OS user
   (PostgreSQL refuses to run as root), so the directory must be reachable by
   that user; it is created under the system temp dir by default.

If neither works, :func:`postgres_server` yields ``(None, reason)`` and the
benchmark records "not measured" with that reason instead of a number.
"""

from __future__ import annotations

import contextlib
import glob
import os
import shutil
import subprocess
import tempfile
from typing import Iterator

#: Server settings used for the local cluster (recorded in the results).
LOCAL_SETTINGS = {
    "shared_buffers": "1GB",
    "work_mem": "64MB",
    "maintenance_work_mem": "512MB",
    "max_wal_size": "4GB",
}


def _find_bin(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    cands = sorted(glob.glob(f"/usr/lib/postgresql/*/bin/{name}"), reverse=True)
    return cands[0] if cands else None


def _run_as(user: str | None, args: list[str]) -> subprocess.CompletedProcess:
    if user:
        runuser = shutil.which("runuser")
        if runuser:
            args = [runuser, "-u", user, "--", *args]
        else:
            import shlex
            args = ["su", user, "-s", "/bin/sh", "-c", shlex.join(args)]
    return subprocess.run(args, capture_output=True, text=True)


def psycopg_available() -> str | None:
    """None if psycopg imports, else the reason."""
    try:
        import psycopg  # noqa: F401
    except Exception as e:  # pragma: no cover - depends on environment
        return f"psycopg not importable ({type(e).__name__}: {e})"
    return None


@contextlib.contextmanager
def postgres_server(dsn: str | None = None, local: bool = False,
                    port: int = 55432) -> Iterator[tuple[str | None, str]]:
    """Yield ``(dsn, description)``; dsn is None if unavailable (description
    is then the reason)."""
    missing = psycopg_available()
    if missing:
        yield None, missing
        return
    dsn = dsn or os.environ.get("HEAPDB_BENCH_PG_DSN")
    if dsn:
        yield dsn, "external server (dsn given)"
        return
    if not local:
        yield None, "no --pg-dsn / HEAPDB_BENCH_PG_DSN and --pg-local not set"
        return
    initdb, pg_ctl = _find_bin("initdb"), _find_bin("pg_ctl")
    if not (initdb and pg_ctl):
        yield None, "PostgreSQL server binaries (initdb, pg_ctl) not found"
        return
    user = None
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        import pwd
        try:
            pwd.getpwnam("postgres")
        except KeyError:
            yield None, "running as root and no 'postgres' OS user to run the server"
            return
        user = "postgres"
    base = tempfile.mkdtemp(prefix="heapdb-bench-pg-")
    data = os.path.join(base, "data")
    if user:
        shutil.chown(base, user)
        os.chmod(base, 0o755)
    opts = " ".join([f"-p {port}", f"-k {base}", "-c listen_addresses=''"] +
                    [f"-c {k}={v}" for k, v in LOCAL_SETTINGS.items()])
    started = False
    try:
        r = _run_as(user, [initdb, "-D", data, "-A", "trust", "-U", "bench", "--no-sync"])
        if r.returncode != 0:
            yield None, f"initdb failed: {r.stderr.strip()[-300:]}"
            return
        r = _run_as(user, [pg_ctl, "-D", data, "-o", opts, "-l",
                           os.path.join(base, "server.log"), "-w", "start"])
        if r.returncode != 0:
            yield None, f"pg_ctl start failed: {r.stderr.strip()[-300:]}"
            return
        started = True
        yield (f"host={base} port={port} user=bench dbname=postgres",
               f"local throwaway cluster ({initdb.rsplit('/bin/', 1)[0]}), "
               + ", ".join(f"{k}={v}" for k, v in LOCAL_SETTINGS.items()))
    finally:
        if started:
            _run_as(user, [pg_ctl, "-D", data, "-m", "fast", "-w", "stop"])
        shutil.rmtree(base, ignore_errors=True)


def check_dsn(dsn: str) -> str | None:
    """None if we can connect, else the reason."""
    try:
        import psycopg
        with psycopg.connect(dsn, connect_timeout=5) as con:
            con.execute("SELECT 1")
    except Exception as e:
        return f"cannot connect: {type(e).__name__}: {e}"
    return None
