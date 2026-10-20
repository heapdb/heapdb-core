import io

import pytest

from heapdb import CalibrationReport, HeapDB, LogNormalKernel, QueryError, execute, format_result, parse
from heapdb.cli import handle_line, repl
from heapdb.query import (
    CalibrationQuery,
    FeedbackStmt,
    InsertStmt,
    TopCausesQuery,
    parse_duration,
    parse_time,
)

H = 3600.0


def test_parse_top_full():
    q = parse("TOP 3 CAUSES OF bloating FOR me AT now WITHIN 12h EXPLAIN")
    assert isinstance(q, TopCausesQuery)
    assert (q.k, q.effect, q.entity, q.within, q.explain) == (3, "bloating", "me", 43200.0, True)
    assert q.at.resolve(1000.0) == 1000.0


def test_parse_top_minimal_and_case():
    q = parse("top 1 cause of latency for svc-a at 12345.5;")
    assert (q.k, q.effect, q.entity, q.within, q.explain) == (1, "latency", "svc-a", None, False)
    assert q.at.resolve(0) == 12345.5


def test_parse_feedback():
    s = parse("FEEDBACK bloating FOR me AT 7200 CONFIRMED 42 REJECTED 7,#8")
    assert s == FeedbackStmt("bloating", "me", parse_time("7200"), 42, (7, 8))


def test_parse_insert():
    s = parse("INSERT meal FOR me AT now-2h FEATURES beans,onion WEIGHT 1.5")
    assert isinstance(s, InsertStmt)
    assert s.features == {"beans", "onion"} and s.weight == 1.5
    assert s.at.resolve(10 * H) == 8 * H


def test_time_and_duration_forms():
    assert parse_duration("90") == 90
    assert parse_duration("30m") == 1800
    assert parse_duration("1.5d") == 129600
    assert parse_time("2024-05-01T00:00:00+00:00").resolve(0) == 1714521600.0


@pytest.mark.parametrize("bad", [
    "",
    "SELECT * FROM t",
    "TOP x CAUSES OF a FOR b",
    "TOP 0 CAUSES OF a FOR b",
    "TOP 3 THINGS OF a FOR b",
    "TOP 3 CAUSES OF a",
    "TOP 3 CAUSES OF a FOR b WITHIN forever",
    "TOP 3 CAUSES OF a FOR b AT yesterday",
    "TOP 3 CAUSES OF a FOR b EXPLAIN PLEASE",
    "FEEDBACK a FOR b AT now",
    "FEEDBACK a FOR b AT now CONFIRMED abc",
])
def test_parse_errors(bad):
    with pytest.raises(QueryError):
        parse(bad)


def test_execute_roundtrip():
    db = HeapDB()
    db.register_effect("bloating", LogNormalKernel(2 * H, 0.6), priors={"beans": (3, 1)})
    clock = lambda: 10 * H  # noqa: E731
    execute(db, "INSERT meal FOR me AT now-2h FEATURES beans,onion WEIGHT 1.5", clock)
    execute(db, "INSERT meal FOR me AT now-1h FEATURES apple", clock)
    res = execute(db, "TOP 2 CAUSES OF bloating FOR me AT now WITHIN 12h EXPLAIN", clock)
    assert res.top.event.features == {"beans", "onion"}
    assert res.top.explanation is not None
    msg = execute(db, f"FEEDBACK bloating FOR me AT now CONFIRMED {res.top.event.id}", clock)
    assert msg.startswith("ok")
    # Proportional credit: beans (0.75) takes r = 0.75 / (1 - 0.25 * 0.8) of it.
    assert db.link("bloating", "beans", "me").alpha == pytest.approx(3 + 0.75 / 0.8)


def test_cli_session(tmp_path):
    snap = tmp_path / "s.jsonl"
    script = "\n".join([
        ".effect bloating lognormal 2h 0.6",
        ".prior bloating beans 3 1",
        "INSERT meal FOR me AT 0 FEATURES beans,onion WEIGHT 1.5",
        "INSERT meal FOR me AT 3600 FEATURES apple",
        "TOP 2 CAUSES OF bloating FOR me AT 7200 EXPLAIN",
        "FEEDBACK bloating FOR me AT 7200 CONFIRMED 1",
        "NONSENSE",
        f".save {snap}",
        ".quit",
    ])
    out = io.StringIO()
    repl(HeapDB(), stdin=io.StringIO(script), stdout=out, clock=lambda: 0.0)
    text = out.getvalue()
    assert "1. #1 meal@0 [beans,onion]" in text
    assert "ok: confirmed #1" in text
    assert "error:" in text
    assert snap.exists()
    db, msg = handle_line(HeapDB(), f".load {snap}", clock=lambda: 0.0)
    assert len(db) == 2 and msg.startswith("ok")


def test_package_doctest():
    import doctest

    import heapdb

    assert doctest.testmod(heapdb).failed == 0


def test_parse_group_and_calibration():
    q = parse("TOP 5 CAUSES OF latency FOR checkout,payments,search AT 100 EXPLAIN")
    assert q.is_group and q.entities == ("checkout", "payments", "search")
    assert parse("TOP 1 CAUSE OF x FOR a").entities == ("a",)
    c = parse("CALIBRATION OF bloating BINS 5")
    assert isinstance(c, CalibrationQuery) and (c.effect, c.bins) == ("bloating", 5)
    assert parse("calibration of x").bins == 10
    s = parse("INSERT txn FOR acct ATTRS amount=5,currency=EUR,rate=1.5")
    assert dict(s.attrs) == {"amount": 5, "currency": "EUR", "rate": 1.5}
    assert s.weight is None
    for bad in ("CALIBRATION bloating", "CALIBRATION OF x BINS 0", "INSERT t FOR a ATTRS amount",
                "TOP 1 CAUSES OF x FOR ,"):
        with pytest.raises(QueryError):
            parse(bad)


def test_execute_group_calibration_and_attrs():
    db = HeapDB()
    db.register_effect("latency", LogNormalKernel(H, 0.6), priors={"pool": (3, 1)})
    db.register_event_type("deploy", weight=lambda a: a.get("lines", 1) / 100)
    clock = lambda: 10 * H  # noqa: E731
    msg = execute(db, "INSERT deploy FOR svc-a AT now-1h FEATURES pool ATTRS lines=250", clock)
    assert "w=2.5" in msg
    execute(db, "INSERT deploy FOR svc-b AT now-1h FEATURES ui", clock)
    res = execute(db, "TOP 3 CAUSES OF latency FOR svc-a,svc-b AT now EXPLAIN", clock)
    assert [c.entity for c in res] == ["svc-a", "svc-b"]
    text = format_result(res)
    assert "for entities svc-a,svc-b" in text and "1. svc-a: #1" in text
    execute(db, "FEEDBACK latency FOR svc-a AT now CONFIRMED 1", clock)
    rep = execute(db, "CALIBRATION OF latency BINS 4", clock)
    assert isinstance(rep, CalibrationReport) and rep.n == 1
    assert "Brier" in format_result(rep)


def test_cli_rules_and_forget():
    db = HeapDB()
    db.register_effect("fx", LogNormalKernel(H, 0.6))
    db.add_rule("fx", "small", lambda e: True, 0.3)
    db.add_rule("fx", "cron", None, 0.1, types={"cron"})
    _, out = handle_line(db, ".rules fx", clock=lambda: 0.0)
    assert "small: x0.3" in out and "cron: x0.1 for types ['cron'] (type-level)" in out
    db.insert("me", 0, "meal", {"a"})
    _, out = handle_line(db, ".forget me scrub", clock=lambda: 0.0)
    assert out.startswith("ok: forgot 1 events") and len(db) == 0
    _, out = handle_line(db, "CALIBRATION OF fx", clock=lambda: 0.0)
    assert "n=0" in out
    _, out = handle_line(db, ".help", clock=lambda: 0.0)
    assert "CALIBRATION OF" in out and "group query" in out
