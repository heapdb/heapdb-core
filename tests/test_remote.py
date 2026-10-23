"""heapdb.connect(): the remote client against a real single-node heapdb-server.

The server binary is taken from HEAPDB_SERVER_BIN or target/{release,debug}/
heapdb-server (build it with `cargo build -p heapdb-server`); tests that need
it are skipped with a clear message otherwise.
"""

import hashlib
import json
import math
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request

import pytest

import heapdb
from heapdb import Candidate, HeapDB, LogNormalKernel
from heapdb.remote import (AuthenticationError, InvalidArgumentError, NotFoundError, PermissionDeniedError,
                           QueryLanguageError, QuotaExceededError, RateLimitError, RemoteConnectionError,
                           parse_url, weight_attr, where)

H = 3600.0
ADMIN = "pytest-admin-token"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- URL parsing (no server needed) ------------------------------------------------


def test_parse_url_forms():
    assert parse_url("heapdb://hdb_live_abc@api.heapdb.com/db_x") == ("https://api.heapdb.com", "hdb_live_abc", "db_x")
    assert parse_url("heapdb+http://k@127.0.0.1:8080/db_y/") == ("http://127.0.0.1:8080", "k", "db_y")
    for bad in ["https://k@h/db", "heapdb://api.heapdb.com/db", "heapdb://k@h/", "heapdb://k@h/a/b"]:
        with pytest.raises(ValueError):
            parse_url(bad)


def test_connect_env_and_redaction(monkeypatch):
    monkeypatch.setenv("HEAPDB_URL", "heapdb://hdb_live_secret@api.heapdb.com/db_1")
    db = heapdb.connect()
    assert (db.base_url, db.database_id) == ("https://api.heapdb.com", "db_1")
    assert "secret" not in repr(db)
    monkeypatch.setenv("MY_DB", "heapdb+http://k@localhost:1/db_2")
    assert heapdb.connect("MY_DB").database_id == "db_2"
    monkeypatch.delenv("HEAPDB_URL")
    with pytest.raises(ValueError, match="HEAPDB_URL"):
        heapdb.connect()


def test_declarative_only():
    db = heapdb.connect("heapdb+http://k@127.0.0.1:9/db_x", retries=0)
    with pytest.raises(TypeError, match="declarative"):
        db.register_event_type("meal", weight=lambda a: 1.0)
    with pytest.raises(TypeError, match="declarative"):
        db.add_rule("e", "r", lambda e: True, 0.5)
    with pytest.raises(RemoteConnectionError):
        db.info()


# -- server fixture -------------------------------------------------------------------


def _server_bin():
    cands = [os.environ.get("HEAPDB_SERVER_BIN")] + [
        os.path.join(ROOT, "target", p, "heapdb-server") for p in ("release", "debug")]
    for c in cands:
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
                    "(or set HEAPDB_SERVER_BIN) to run the remote-client tests")
    data = tempfile.mkdtemp(prefix="heapdb-server-")
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


def _admin(url, method, path, body=None):
    req = urllib.request.Request(url + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {ADMIN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _new_db(url, limits=None, scopes=("read", "write", "admin"), **kw):
    db_id = "db_" + "".join(secrets.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(20))
    _admin(url, "POST", "/admin/v1/databases", {"database_id": db_id, "org_id": "org_py", "plan": "free",
                                                 "limits": limits or {}})
    key = "hdb_live_" + secrets.token_hex(16)
    _admin(url, "PUT", f"/admin/v1/databases/{db_id}/keys/key_1",
           {"sha256": hashlib.sha256(key.encode()).hexdigest(), "scopes": list(scopes)})
    host = url.split("://")[1]
    return heapdb.connect(f"heapdb+http://{key}@{host}/{db_id}", **kw)


def _schema(db):
    db.register_effect("bloating", LogNormalKernel(median=2 * H, sigma=0.6), priors={"beans": (3, 1), "dairy": (2, 2)},
                       cause_types={"meal", "snack"})
    if isinstance(db, HeapDB):
        db.register_event_type("meal", weight=lambda a: a.get("portion", 1.0))
        db.add_rule("bloating", "tiny", lambda e: e.attrs.get("portion", 1.0) < 0.5, 0.5)
    else:
        db.register_event_type("meal", weight=weight_attr("portion", default=1.0))
        db.add_rule("bloating", "tiny", where("portion", "<", 0.5), 0.5)
    db.add_rule("bloating", "snacks", None, 0.8, types={"snack"})


def _events():
    import random
    rng = random.Random(11)
    feats = ["beans", "onion", "dairy", "rice", "chips", "garlic"]
    out = []
    for i in range(400):
        out.append(dict(entity=f"u{rng.randrange(4)}", time=i * 400.0 + rng.random() * 100,
                        type="snack" if rng.random() < 0.25 else "meal",
                        features=set(rng.sample(feats, rng.randint(1, 3))),
                        attrs={"portion": round(0.2 + rng.random() * 2, 3)}))
    return out


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


# -- server tests --------------------------------------------------------------------------


def test_parity_local_vs_remote(server):
    local, remote = HeapDB(bucket_seconds=900), _new_db(server)
    for db in (local, remote):
        _schema(db)
    evs = _events()
    for e in evs:
        local.insert(e["entity"], e["time"], e["type"], e["features"], attrs=e["attrs"])
    ids = remote.insert_many(evs, batch_size=150)
    assert ids == list(range(1, len(evs) + 1))
    # learning
    for ent, at in (("u1", 30 * H), ("u2", 15 * H)):
        top = local.top_causes(ent, "bloating", at, k=2)
        args = (ent, "bloating", at, top[0].event.id, [top[1].event.id])
        fl, fr = local.feedback(*args), remote.feedback(*args)
        assert fl.confirmed == fr.confirmed and tuple(fl.rejected) == fr.rejected
        assert all(_close(fl.credit[k], fr.credit[k]) for k in fl.credit)
    ol, orr = local.observe("u3", "bloating", 20 * H), remote.observe("u3", "bloating", 20 * H)
    assert [c.event_id for c in ol.credits] == [c.event_id for c in orr.credits]
    assert _close(ol.total_credit, orr.total_credit)
    sl, sr = local.settle(18 * H), remote.settle(18 * H)
    assert (sl.events_settled, sl.events_rejected) == (sr.events_settled, sr.events_rejected)
    # queries
    for ent in ("u0", "u1", "u2", "u3"):
        for at in (5 * H, 22 * H, 40 * H):
            ql = local.top_causes(ent, "bloating", at, k=5, explain=True)
            qr = remote.top_causes(ent, "bloating", at, k=5, explain=True)
            assert len(ql) == len(qr)
            for cl, cr in zip(ql, qr):
                assert (cl.rank, cl.event.id) == (cr.rank, cr.event.id)
                assert _close(cl.score, cr.score)
                assert cl.event.short() == cr.event.short()
                assert cl.event.features == cr.event.features
                assert _close(cl.explanation.kernel_value, cr.explanation.kernel_value)
                assert [r.name for r in cl.explanation.rules] == [r.name for r in cr.explanation.rules]
                assert _close(cl.explanation.score, cr.explanation.score)
    gl = local.top_causes_group(["u0", "u1"], "bloating", 40 * H, k=4)
    gr = remote.top_causes_group(["u0", "u1"], "bloating", 40 * H, k=4)
    assert [c.event.id for c in gl] == [c.event.id for c in gr] and gr.is_group
    assert _close(local.feature_risk("bloating", {"beans", "garlic"}, entity="u1"),
                  remote.feature_risk("bloating", {"beans", "garlic"}, entity="u1"))
    sl, exl = local.feature_risk("bloating", "beans,dairy", weight=2.0, explain=True)
    sr, exr = remote.feature_risk("bloating", "beans,dairy", weight=2.0, explain=True)
    assert _close(sl, sr) and exr.prospective and _close(exl.link_strength, exr.link_strength)
    cands = [Candidate("a", {"beans"}), {"id": "b", "features": ["dairy", "onion"], "weight": 1.5},
             Candidate("c", {"rice"}, type="snack")]
    rl = local.rank_candidates("bloating", cands, k=3, entity="u1")
    rr = remote.rank_candidates("bloating", cands, k=3, entity="u1")
    assert [x.candidate.id for x in rl] == [x.candidate.id for x in rr]
    assert all(_close(a.score, b.score) for a, b in zip(rl, rr))
    ll, lr = local.link_report("bloating", min_support=0), remote.link_report("bloating", min_support=0)
    assert [r.feature for r in ll] == [r.feature for r in lr]
    assert all(_close(a.strength, b.strength) and _close(a.lo, b.lo) for a, b in zip(ll, lr))
    assert lr.row("beans") is not None
    cl, cr = local.calibration_report("bloating", bins=4), remote.calibration_report("bloating", bins=4)
    assert cl.n == cr.n and _close(cl.brier, cr.brier)
    # the query language
    res = remote.execute("TOP 3 CAUSES OF bloating FOR u1 AT 144000 EXPLAIN")
    assert [c.event.id for c in res] == [c.event.id for c in local.top_causes("u1", "bloating", 40 * H, k=3)]
    assert str(res).startswith("Top causes of 'bloating' for u1")
    assert isinstance(remote.execute("LINKS OF bloating"), type(lr))
    msg = remote.execute("INSERT meal FOR u9 AT now-1h FEATURES rice", now=50 * H)
    assert isinstance(msg, str) and "inserted" in msg
    with pytest.raises(QueryLanguageError) as ei:
        remote.execute("SELECT 1")
    assert isinstance(ei.value, heapdb.QueryError)
    # forgetting
    fl, fr = local.forget_entity("u3", scrub_global=True), remote.forget_entity("u3", scrub_global=True)
    assert fl == fr


def test_insert_idempotency_and_info(server):
    db = _new_db(server)
    db.register_effect("e", {"type": "exponential", "params": {"half_life": 60}})
    ev = db.insert("me", 0, "deploy", "orm, lockfile", attrs={"lines": 3}, external_id="x1")
    assert ev.id == 1 and ev.features == {"orm", "lockfile"} and ev.short().startswith("#1 deploy@0")
    again = db.insert("me", 0, "deploy", external_id="x1")
    assert again.id == 1
    n = 2500
    ids = db.insert_many(({"entity": "bulk", "time": float(i), "type": "t", "external_id": f"b{i}"}
                          for i in range(n)))
    assert len(ids) == n and len(set(ids)) == n
    assert db.insert_many([("bulk", 0.0, "t", None, None, None, "b0")]) == [ids[0]]
    info = db.info()
    assert info["database_id"] == db.database_id
    assert info["usage"]["events_stored"] == n + 1
    assert info["limits"]["max_events"] is None
    db.register_event_type("deploy", weight=heapdb.remote.weight_log1p("lines"))
    assert math.isclose(db.insert("me", 1, "deploy", attrs={"lines": 420}).weight, math.log1p(420))


def test_errors(server):
    db = _new_db(server)
    with pytest.raises(NotFoundError) as ei:
        db.top_causes("me", "nope", 0)
    assert ei.value.code == "unknown_effect" and isinstance(ei.value, KeyError) and ei.value.status == 404
    with pytest.raises(InvalidArgumentError):
        db.register_effect("e", {"type": "spline", "params": {}})
    bad = heapdb.connect(f"heapdb+http://hdb_live_wrong@{server.split('://')[1]}/{db.database_id}")
    with pytest.raises(AuthenticationError) as ei:
        bad.info()
    assert ei.value.code == "invalid_api_key"
    ro = _new_db(server, scopes=("read",))
    with pytest.raises(PermissionDeniedError) as ei:
        ro.insert("me", 0, "meal")
    assert ei.value.code == "insufficient_scope"


def test_catalog(server):
    local, db = HeapDB(bucket_seconds=900), _new_db(server)
    # Empty database: empty first pages.
    for page in (db.effects(), db.entities(), db.event_types(), db.occurrences()):
        assert page == [] and page.next_cursor is None and not page.truncated
    assert db.effects().total_effects == 0 and db.event_types().total_events == 0
    for d in (local, db):
        _schema(d)
    assert db.effects() == local.effects == ["bloating"]
    eff = db.effects()[0]
    assert eff.name == "bloating" and eff.kernel["type"] == "lognormal"
    assert eff.kernel["human"]["median_s"] == 2 * H and eff.max_lag > 2 * H
    assert eff.cause_types == ["meal", "snack"] and eff.rules == ["tiny", "snacks"]
    assert eff.priors["top"][0]["feature"] == "beans" and eff.counts["occurrences"] == 0
    assert eff.combine == "noisy_or" and eff.logistic is None
    assert db.entities() == [] and db.event_types()[0].registered and db.occurrences() == []
    # Effects page in registration order; logistic effects report their mode.
    db.register_effect("churn", {"type": "exp", "params": {"half_life": 60}}, combine="logistic", learning_rate=0.5)
    first = db.effects(limit=1)
    assert first == ["bloating"] and first.total_effects == 2 and first.next_cursor
    second = db.effects(limit=1, cursor=first.next_cursor)
    assert second == ["churn"] and second.next_cursor is None
    assert second[0].combine == "logistic" and second[0].logistic["learning_rate"] == 0.5
    with pytest.raises(InvalidArgumentError):
        db.effects(cursor="zz")
    for ent, t in (("acme/web", 0.0), ("acme/api", H), ("checkout", 2 * H), ("acme/api", 3 * H)):
        for d in (local, db):
            d.insert(ent, t, "meal" if t != 2 * H else "snack", {"beans", "onion"} if t else {"dairy"})
    ents = db.entities()
    assert ents == ["acme/api", "checkout", "acme/web"]  # most recent activity first
    assert sorted(ents) == local.entities
    assert ents[0].event_count == 2 and ents[0].last_event_time == 3 * H and ents.next_cursor is None
    page = db.entities("acme/", limit=1)
    assert page == ["acme/api"] and page.next_cursor and page.matched == 2
    assert db.entities("acme/", limit=1, cursor=page.next_cursor) == ["acme/web"]
    assert db.entities("zzz") == [] and db.entities("zzz").matched == 0
    types = db.event_types()
    assert [t.type for t in types] == ["meal", "snack"] and types.total_types == 2 and types.total_events == 4
    meal = types[0]
    assert (meal.type, meal.count, meal.features[0]) == ("meal", 3, "beans")
    assert meal.top_features[0] == ("beans", 2) and meal.weight["kind"] == "attr"
    tp = db.event_types(limit=1)
    assert [t.type for t in tp] == ["meal"] and tp.next_cursor
    assert [t.type for t in db.event_types(limit=1, cursor=tp.next_cursor)] == ["snack"]
    db.observe("acme/api", "bloating", 3.5 * H)
    db.feedback("checkout", "bloating", 2.5 * H, confirmed_event_id=3)
    occ = db.occurrences("bloating")
    assert [(o.entity, o.at, o.sources) for o in occ] == [("acme/api", 3.5 * H, ("observe",)),
                                                          ("checkout", 2.5 * H, ("feedback",))]
    assert occ.next_cursor is None
    assert occ[1].confirmed_event_id == 3 and db.occurrences(entity="checkout", limit=1)[0].entity == "checkout"
    o1 = db.occurrences("bloating", limit=1)
    assert o1 == occ[:1] and o1.next_cursor
    o2 = db.occurrences("bloating", limit=1, cursor=o1.next_cursor)
    assert o2 == occ[1:] and o2.next_cursor is None
    with pytest.raises(NotFoundError):
        db.occurrences("nope")
    # Same scope as the other reads: read works, write-only and admin-only keys do not.
    ro = _new_db(server, scopes=("read",))
    assert ro.effects() == [] and ro.entities() == []
    for scopes in (("write",), ("admin",)):
        k = _new_db(server, scopes=scopes)
        for call in (k.effects, k.entities, k.event_types, k.occurrences):
            with pytest.raises(PermissionDeniedError) as ei:
                call()
            assert ei.value.code == "insufficient_scope"


def test_rate_limit_is_retried(server):
    db = _new_db(server, limits={"max_requests_per_minute": 60}, retries=3)
    db.register_effect("e", {"type": "exp", "params": {"half_life": 60}})
    t = time.monotonic()
    for _ in range(62):  # the bucket holds 60; refills one per second
        db.calibration_report("e")
    assert time.monotonic() - t >= 0.9, "the client waited for Retry-After"
    strict = _new_db(server, limits={"max_requests_per_minute": 1}, retries=0)
    strict.info()
    with pytest.raises(NotFoundError):
        strict.calibration_report("e")
    with pytest.raises(RateLimitError) as ei:
        strict.calibration_report("e")
    assert ei.value.retry_after and ei.value.retry_after >= 1


def test_monthly_quota_is_not_retried(server):
    db = _new_db(server, limits={"max_requests_per_month": 2}, retries=5)
    db.settle(0)
    db.settle(0)
    t = time.monotonic()
    with pytest.raises(QuotaExceededError) as ei:
        db.settle(0)
    assert time.monotonic() - t < 1.0
    assert ei.value.code == "monthly_quota_exceeded" and ei.value.status == 429
    assert ei.value.retry_after > 0
    assert db.info()["usage"]["requests_this_period"] == 2


def test_openapi_document_validates(server):
    doc = json.loads(urllib.request.urlopen(server + "/v1/openapi.json").read())
    validator = pytest.importorskip("openapi_spec_validator")
    validator.validate(doc)
    assert "/v1/databases/{database_id}/events" in doc["paths"]
