"""heapdb.Heaps: keyed heaps on every backend.

The same seeded operation sequences run against memory (the reference),
native:<dir>, a heapdb-server (``heapdb.connect(...).heaps``, if the
binary is built) and Postgres (if HEAPDB_TEST_POSTGRES_URL is set), and must
give the same results: ids, order, versions; keys and scores within 1e-9
relative. Backends that are not available are skipped with a reason.
"""

import hashlib
import json
import math
import os
import random
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

import pytest

import heapdb
from heapdb import HeapInfo, HeapItem, Heaps, Ranked, heap_filter

T0 = 1_760_000_000.0
LN2 = math.log(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADMIN = "pytest-heaps-admin"
PG_URL = os.environ.get("HEAPDB_TEST_POSTGRES_URL")


# -- unit behaviour (memory) ------------------------------------------------------------


def test_basics_and_reprs():
    h = Heaps()
    assert repr(h) == "Heaps(persist_to='memory')"
    info = h.create_heap("work", half_life=3600, landmark=T0, meta={"team": "pay"})
    assert isinstance(info, HeapInfo)
    assert (info.name, info.half_life, info.landmark, info.size, info.meta) == ("work", 3600.0, T0, 0, {"team": "pay"})
    assert math.isclose(info.lam, LN2 / 3600)
    assert h.create_heap("work", 3600) == info  # idempotent
    with pytest.raises(ValueError, match="already exists"):
        h.create_heap("work", 60)
    it = h.push("work", "CHK-12", 1.5, t=T0, attrs={"kind": "bug", "skills": ["ts"]})
    assert isinstance(it, HeapItem)
    assert (it.version, it.attrs) == (1, {"kind": "bug", "skills": ["ts"]})
    assert repr(it).startswith("HeapItem(id='CHK-12', base=1.0, activity=0.4054651081081644")
    h.push("work", "CHK-13", t=T0 - 3600)
    h.set_base("work", "CHK-13", 4)
    top = h.top_k("work", 5, now=T0 + 3600)
    assert [r.id for r in top] == ["CHK-13", "CHK-12"]  # 4 x 0.25 beats 1.5 x 0.5
    r = top[1]
    assert isinstance(r, Ranked) and r.rank == 2
    assert math.isclose(r.score, 0.75) and math.isclose(r.heat, 0.75)
    assert math.isclose(top[0].score, 4 * 0.25)
    assert repr(r).startswith("Ranked(rank=2, id='CHK-12', score=0.75")
    assert "score=0.75" in str(r)
    assert Ranked.from_dict(r.to_dict()) == r
    assert h.get("work", "nope", now=T0) is None
    with pytest.raises(KeyError, match="unknown heap"):
        h.push("missing", "a")
    with pytest.raises(ValueError, match="weight must be finite and > 0"):
        h.push("work", "a", 0)
    with pytest.raises(ValueError, match="1-200 UTF-8 bytes"):
        h.push("work", "x" * 201)


def test_now_defaults_to_the_clock():
    h = Heaps()
    before = time.time()
    info = h.create_heap("q", 60)
    assert before <= info.landmark <= time.time()
    h.push("q", "a")  # t = now
    (r,) = h.top_k("q")
    assert 0.99 < r.heat <= 1.0 + 1e-9


def test_filters_dicts_and_strings():
    h = Heaps()
    h.create_heap("q", 3600, landmark=0)
    h.push("q", "a", 3, t=0, attrs={"kind": "bug", "skills": ["ts"], "labels": []})
    h.push("q", "b", 2, t=0, attrs={"kind": "task", "labels": ["needs-human"]})
    h.push("q", "c", 1, t=0, attrs={"kind": "epic"})
    f = {"all": [{"attr": "kind", "op": "in", "value": ["bug", "task"]},
                 {"not": {"attr": "labels", "op": "contains", "value": "needs-human"}}]}
    assert [r.id for r in h.top_k("q", 5, now=0, filter=f)] == ["a"]
    s = "kind IN ('bug', 'task') AND NOT labels CONTAINS 'needs-human'"
    assert heap_filter(s) == f
    assert [r.id for r in h.top_k("q", 5, now=0, filter=s)] == ["a"]
    assert [r.id for r in h.top_k("q", 5, now=0, filter="skills SUBSET OF ('ts')")] == ["a", "b", "c"]
    with pytest.raises(ValueError, match=r"filter.all\[0\]: unknown op 'like'"):
        h.top_k("q", filter={"all": [{"attr": "kind", "op": "like", "value": "b"}]})
    with pytest.raises(heapdb.QueryError, match="quote strings"):
        h.top_k("q", filter="kind = bug")
    assert h.pop("q", now=0, filter="kind = 'task'").id == "b"
    assert h.pop("q", now=0, filter="kind = 'task'") is None


def test_execute_statements():
    h = Heaps()
    assert isinstance(h.execute("CREATE HEAP work HALF LIFE 72h LANDMARK 0"), HeapInfo)
    h.execute("""PUSH 'CHK-12' INTO work WEIGHT 1.5 AT 0 ATTRS {"kind": "bug", "required_skills": ["ts"]}""")
    assert h.execute("SET BASE 'CHK-12' IN work TO 4").startswith("ok:")
    got = h.execute("PEEK 5 FROM work AT 100 WHERE kind IN ('bug', 'task') AND required_skills SUBSET OF ('ts', 'react')")
    assert [r.id for r in got] == ["CHK-12"]
    assert [r.id for r in h.execute("POP FROM work WHERE NOT labels CONTAINS 'needs-human'")] == ["CHK-12"]
    assert h.execute("DROP HEAP work") == "ok: dropped heap 'work'"
    with pytest.raises(heapdb.QueryError):
        h.execute("PEEK 0 FROM work")


def test_rebase_keeps_scores():
    h = Heaps()
    h.create_heap("q", 3600, landmark=0)
    rng = random.Random(4)
    for i in range(50):
        h.push("q", f"i{i % 13}", rng.uniform(0.1, 5), t=rng.uniform(0, 86400))
    before = h.top_k("q", 20, now=90000)
    h.rebase("q", now=90000)
    after = h.top_k("q", 20, now=90000)
    assert [r.id for r in before] == [r.id for r in after]
    for a, b in zip(before, after):
        assert math.isclose(a.score, b.score, rel_tol=1e-9) and a.version == b.version


def test_native_persists(tmp_path):
    d = tmp_path / "q"
    with Heaps(f"native:{d}") as h:
        assert h.persist_to == f"native:{d}"
        h.create_heap("q", 3600, landmark=0)
        h.push("q", "a", 2, t=0, attrs={"k": 1})
        h.push("q", "b", 1, t=0)
        h.pop("q", now=10)
    with pytest.raises(ValueError, match="closed"):
        h.top_k("q")
    h2 = Heaps(f"native:{d}")
    assert [(r.id, r.version) for r in h2.top_k("q", 10, now=10)] == [("b", 1)]


def test_persist_to_errors():
    with pytest.raises(ValueError, match="bad persist_to"):
        Heaps("sqlite://x")
    if not heapdb._core.HAS_POSTGRES:
        with pytest.raises(ValueError, match="no Postgres support"):
            Heaps("postgres://localhost/x")


# -- conformance over backends ----------------------------------------------------------

HEAPS = ["work", "acme/checkout", "q é"]
IDS = ["a", "b", "c", "CHK-1", "CHK/2", "é", "it's", "z"]
KINDS = ["bug", "task", "epic"]
SKILLS = ["ts", "react", "go"]


def _attrs(rng):
    a = {}
    if rng.random() < 0.8:
        a["kind"] = rng.choice(KINDS)
    if rng.random() < 0.6:
        a["required_skills"] = rng.sample(SKILLS, rng.randint(0, 2))
    if rng.random() < 0.5:
        a["labels"] = rng.sample(["needs-human", "p1"], rng.randint(0, 2))
    if rng.random() < 0.5:
        a["risk"] = rng.randint(0, 4)
    return a


def _leaf(rng):
    attr = rng.choice(["kind", "required_skills", "labels", "risk", "absent"])
    op = rng.choice(["eq", "ne", "lt", "gte", "in", "not_in", "contains", "contains_all", "subset_of",
                     "exists", "missing"])
    if op in ("exists", "missing"):
        return {"attr": attr, "op": op}
    if op in ("in", "not_in", "contains_all", "subset_of"):
        return {"attr": attr, "op": op, "value": rng.sample(KINDS + SKILLS + [1, 2], rng.randint(0, 3))}
    if op in ("lt", "gte"):
        return {"attr": attr, "op": op, "value": rng.choice([1.5, 3, "bug"])}
    return {"attr": attr, "op": op, "value": rng.choice(KINDS + SKILLS + ["needs-human", 2])}


def _filter(rng, depth=0):
    if depth >= 2 or rng.random() < 0.5:
        return _leaf(rng)
    kind = rng.choice(["all", "any", "not"])
    if kind == "not":
        return {"not": _filter(rng, depth + 1)}
    return {kind: [_filter(rng, depth + 1) for _ in range(rng.randint(0, 3))]}


def _ops(seed, steps=200):
    rng = random.Random(seed)
    ops = [("create", h, 86400.0, T0) for h in HEAPS]
    clock = T0
    for _ in range(steps):
        clock += rng.random() * 3600
        heap, id_ = rng.choice(HEAPS), rng.choice(IDS)
        f = _filter(rng) if rng.random() < 0.6 else None
        x = rng.random()
        if x < 0.40:
            ops.append(("push", heap, id_, rng.uniform(0.05, 10), clock - rng.random() * 86400,
                        _attrs(rng) if rng.random() < 0.5 else None))
        elif x < 0.48:
            ops.append(("base", heap, id_, rng.uniform(0.1, 5)))
        elif x < 0.56:
            ops.append(("attrs", heap, id_, _attrs(rng), rng.random() < 0.5))
        elif x < 0.61:
            ops.append(("remove", heap, id_))
        elif x < 0.66:
            ops.append(("get", heap, id_, clock))
        elif x < 0.84:
            ops.append(("top", heap, rng.randint(0, 5), clock, f))
        elif x < 0.95:
            ops.append(("pop", heap, clock, f))
        elif x < 0.98:
            ops.append(("rebase", heap, clock))
        else:
            ops.append(("push", heap, id_, -1.0, clock, None))  # invalid
    ops.append(("final",))
    return ops


def _norm(x):
    """Comparable form: dicts of results, floats kept for tolerance checks."""
    if isinstance(x, (Ranked, HeapItem, HeapInfo)):
        return x.to_dict()
    if isinstance(x, list):
        return [_norm(i) for i in x]
    return x


def _run(h, op):
    kind = op[0]
    try:
        if kind == "create":
            return h.create_heap(op[1], op[2], landmark=op[3])
        if kind == "push":
            return h.push(op[1], op[2], op[3], t=op[4], attrs=op[5])
        if kind == "base":
            return h.set_base(op[1], op[2], op[3])
        if kind == "attrs":
            return h.set_attrs(op[1], op[2], op[3], merge=op[4])
        if kind == "remove":
            return h.remove(op[1], op[2])
        if kind == "get":
            return h.get(op[1], op[2], now=op[3])
        if kind == "top":
            return h.top_k(op[1], op[2], now=op[3], filter=op[4])
        if kind == "pop":
            return h.pop(op[1], now=op[2], filter=op[3])
        if kind == "rebase":
            return h.rebase(op[1], now=op[2])
        if kind == "final":
            return [h.top_k(name, 1000, now=T0 + 1e6) for name in HEAPS]
    except (KeyError, ValueError) as e:
        return ("error", "unknown_heap" if isinstance(e, KeyError) else "invalid")
    raise AssertionError(op)


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-300)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    return a == b


def _conform(make, seeds=(1, 2, 3, 4, 5)):
    rows = 0
    for seed in seeds:
        ref, other = Heaps(), make()
        try:
            for i, op in enumerate(_ops(seed)):
                a, b = _norm(_run(ref, op)), _norm(_run(other, op))
                assert _same(a, b), f"seed {seed} step {i} {op}:\n  memory: {a}\n  other:  {b}"
                if isinstance(a, list):
                    rows += len(a)
        finally:
            other.close()
    assert rows > 100, "the comparison carried too little data"


def test_native_conforms(tmp_path):
    n = iter(range(100))
    _conform(lambda: Heaps(f"native:{tmp_path / str(next(n))}"))


# -- Postgres ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_url():
    if not PG_URL:
        pytest.skip("set HEAPDB_TEST_POSTGRES_URL (e.g. postgres://postgres:postgres@localhost:5432/postgres) "
                    "to run the Postgres tests")
    if not heapdb._core.HAS_POSTGRES:
        pytest.skip("this heapdb build has no Postgres support (build with --features postgres)")
    # A scratch database per run, created through the extension's own
    # Postgres connection (no Python driver needed).
    name = f"heapdb_py_{os.getpid()}_{secrets.token_hex(4)}"
    def admin(sql):
        heapdb._core._pg_execute(PG_URL, sql)
    admin(f"CREATE DATABASE {name}")
    p = urllib.parse.urlsplit(PG_URL)
    url = urllib.parse.urlunsplit((p.scheme, p.netloc, "/" + name, p.query, ""))
    yield url
    admin(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


def test_postgres_conforms(pg_url):
    def make():
        h = Heaps(pg_url)
        for name in HEAPS:
            h.drop_heap(name)
        return h
    _conform(make)


def test_postgres_persist_to_is_redacted(pg_url):
    p = urllib.parse.urlsplit(pg_url)
    if p.password:
        h = Heaps(pg_url)
        assert f":{p.password}@" not in repr(h) and "***" in h.persist_to
    assert "CREATE SCHEMA IF NOT EXISTS heapdb" in heapdb._core.POSTGRES_SQL


# -- heapdb-server ---------------------------------------------------------------------------


def _server_bin():
    for c in [os.environ.get("HEAPDB_SERVER_BIN")] + [
            os.path.join(ROOT, "target", p, "heapdb-server") for p in ("release", "debug")]:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    exe = _server_bin()
    if exe is None:
        pytest.skip("heapdb-server binary not built: run `cargo build -p heapdb-server` "
                    "(or set HEAPDB_SERVER_BIN) to run the remote heap tests")
    data = tempfile.mkdtemp(prefix="heapdb-heaps-")
    port, iport = _free_port(), _free_port()
    env = dict(os.environ, HEAPDB_ADMIN_TOKEN=ADMIN, HEAPDB_LOG="warn", HEAPDB_FSYNC="0")
    proc = subprocess.Popen([exe, "--data-dir", data, "--listen", f"127.0.0.1:{port}",
                             "--internal-listen", f"127.0.0.1:{iport}"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            urllib.request.urlopen(url + "/readyz", timeout=1).read()
            break
        except OSError:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.fail("heapdb-server did not start")
    yield url
    proc.terminate()
    proc.wait(timeout=30)
    shutil.rmtree(data, ignore_errors=True)


def _remote(url, scopes=("read", "write", "admin")):
    def admin(method, path, body):
        req = urllib.request.Request(url + path, method=method, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {ADMIN}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    db_id = "db_" + "".join(secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(20))
    admin("POST", "/admin/v1/databases", {"database_id": db_id, "org_id": "org_py", "plan": "free"})
    key = "hdb_live_" + secrets.token_hex(16)
    admin("PUT", f"/admin/v1/databases/{db_id}/keys/key_1",
          {"sha256": hashlib.sha256(key.encode()).hexdigest(), "scopes": list(scopes)})
    return heapdb.connect(f"heapdb+http://{key}@{url.split('://')[1]}/{db_id}")


def test_remote_conforms(server):
    _conform(lambda: _remote(server).heaps, seeds=(1, 2, 3))


def test_remote_api(server):
    db = _remote(server)
    h = db.heaps
    assert isinstance(h, heapdb.RemoteHeaps)
    info = h.create_heap("acme/checkout", 3600, landmark=T0, meta={"team": "pay"})
    assert isinstance(info, HeapInfo) and info.meta == {"team": "pay"}
    assert h.heap("nope") is None
    items = h.push_many("acme/checkout", [{"id": "a", "weight": 2, "time": T0, "attrs": {"kind": "bug"}},
                                          {"id": "b/c", "time": T0}])
    assert [i.id for i in items] == ["a", "b/c"]
    with pytest.raises(heapdb.remote.InvalidArgumentError, match=r"items\[1\]"):
        h.push_many("acme/checkout", [{"id": "x"}, {"id": "y", "weight": 0}])
    assert h.heap("acme/checkout").size == 2, "batches are atomic"
    assert h.get("acme/checkout", "b/c", now=T0).id == "b/c"
    assert h.get("acme/checkout", "zzz") is None
    assert [r.id for r in h.top_k("acme/checkout", 5, now=T0, filter="kind = 'bug'")] == ["a"]
    assert [r.id for r in h.execute("PEEK 5 FROM 'acme/checkout' WHERE kind MISSING", now=T0)] == ["b/c"]
    assert h.remove("acme/checkout", "b/c") is True
    assert h.pop("acme/checkout", now=T0).id == "a"
    assert h.pop("acme/checkout", now=T0) is None
    # Scopes: a read-only key can peek but not push.
    ro = _remote(server, scopes=("read",))
    ro_db = heapdb.connect(f"heapdb+http://{ro._key}@{server.split('://')[1]}/{ro.database_id}")
    with pytest.raises(heapdb.remote.PermissionDeniedError):
        ro_db.heaps.create_heap("x", 60)
    assert h.drop_heap("acme/checkout") is True


def test_repl_runs_heap_statements(tmp_path):
    import io

    from heapdb.cli import main, repl
    out = io.StringIO()
    repl(heapdb.HeapDB(), io.StringIO("CREATE HEAP w HALF LIFE 1h LANDMARK 0\nPUSH a INTO w WEIGHT 2 AT 0\n"
                                      "PEEK 3 FROM w AT 3600\nPOP FROM nope\n"), out, heaps=Heaps())
    text = out.getvalue()
    assert "1. 'a'" in text and "score=1 heat=1" in text
    assert "error: \"unknown heap 'nope'; call create_heap first\"" in text
    f = tmp_path / "q.hql"
    f.write_text("CREATE HEAP w HALF LIFE 1h LANDMARK 0\nPUSH a INTO w WEIGHT 2 AT 0\n")
    assert main(["--heaps", f"native:{tmp_path / 'h'}", str(f)]) == 0
    assert [r.id for r in Heaps(f"native:{tmp_path / 'h'}").top_k("w", now=0)] == ["a"]
