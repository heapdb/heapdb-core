"""v0.4: observational learning -- observe(), settle(), link_report() and stats."""

import math
import random

import pytest

from heapdb import (
    ExponentialDecay,
    HeapDB,
    LinkReport,
    LogNormalKernel,
    ObserveReport,
    SettleReport,
    WindowKernel,
    beta_interval,
    beta_ppf,
    betainc,
    execute,
    format_result,
    parse,
    spearman,
)
from heapdb.cli import handle_line
from heapdb.query import LinksQuery, ObserveStmt, QueryError, SettleStmt

H = 3600.0
D = 86400.0


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------


def _binom_tail(n: int, k: int, x: float) -> float:
    """P(Binomial(n, x) >= k): equals I_x(k, n - k + 1) for integer params."""
    return sum(math.comb(n, j) * x ** j * (1 - x) ** (n - j) for j in range(k, n + 1))


@pytest.mark.parametrize("a,b,x", [(2, 3, 0.5), (5, 7, 0.3), (1, 9, 0.05), (20, 30, 0.42),
                                   (3, 1, 0.9), (50, 2, 0.97), (7, 7, 0.5)])
def test_betainc_matches_binomial_identity(a, b, x):
    assert betainc(a, b, x) == pytest.approx(_binom_tail(a + b - 1, a, x), abs=1e-10)


def test_betainc_known_values_and_edges():
    assert betainc(2, 3, 0.5) == pytest.approx(11 / 16)
    assert betainc(1, 1, 0.37) == pytest.approx(0.37)
    assert betainc(2.5, 1, 0.4) == pytest.approx(0.4 ** 2.5)       # I_x(a, 1) = x^a
    assert betainc(1, 3.5, 0.2) == pytest.approx(1 - 0.8 ** 3.5)   # I_x(1, b) = 1-(1-x)^b
    assert betainc(0.5, 0.5, 0.25) == pytest.approx(2 / math.pi * math.asin(0.5))  # arcsine
    assert betainc(3, 4, 0.0) == 0.0 and betainc(3, 4, 1.0) == 1.0
    # symmetry I_x(a, b) = 1 - I_{1-x}(b, a)
    assert betainc(4.2, 9.1, 0.33) == pytest.approx(1 - betainc(9.1, 4.2, 0.67))
    with pytest.raises(ValueError):
        betainc(0, 1, 0.5)


def test_beta_ppf_and_interval():
    assert beta_ppf(0.3, 1, 1) == pytest.approx(0.3, abs=1e-8)
    assert beta_ppf(0.5, 7, 7) == pytest.approx(0.5, abs=1e-8)
    assert beta_ppf(0.05, 3, 1) == pytest.approx(0.05 ** (1 / 3), abs=1e-8)
    for a, b, q in [(2, 8, 0.05), (2, 8, 0.95), (40, 160, 0.1), (0.7, 3.3, 0.9)]:
        assert betainc(a, b, beta_ppf(q, a, b)) == pytest.approx(q, abs=1e-8)
    lo, hi = beta_interval(1, 1, 0.9)
    assert (lo, hi) == pytest.approx((0.05, 0.95), abs=1e-8)
    lo, hi = beta_interval(30, 70, 0.9)
    assert lo < 0.3 < hi and hi - lo < 0.2


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2], [5, 5, 9]) == pytest.approx(1.0)


def test_kernel_mass_fraction():
    k = WindowKernel(0, 100)
    assert k.mass_fraction(-1) == 0.0 and k.mass_fraction(100) == 1.0
    assert k.mass_fraction(25) == pytest.approx(0.25, abs=1e-3)
    e = ExponentialDecay(half_life=10)
    assert e.mass_fraction(10) == pytest.approx(0.5 / (1 - 0.5 ** 10), abs=2e-3)
    ln = LogNormalKernel(median=100, sigma=0.5)
    vals = [ln.mass_fraction(t) for t in range(0, int(ln.support) + 10, 7)]
    assert vals == sorted(vals) and vals[-1] == 1.0


# ---------------------------------------------------------------------------
# observe()
# ---------------------------------------------------------------------------


def _db(leak=0.05, credit="proportional", **kw) -> HeapDB:
    db = HeapDB(bucket_seconds=600)
    db.register_effect("rage", ExponentialDecay(half_life=120), leak=leak, credit=credit, **kw)
    return db


def test_observe_soft_credit_sums_correctly():
    db = _db(leak=0.1)
    e1 = db.insert("u", 0, "page", {"flag:x=test", "page:/a"})
    e2 = db.insert("u", 60, "page", {"page:/b"})
    s1, s2 = db.score(e1, "rage", 120), db.score(e2, "rage", 120)
    before = {f: db.link("rage", f, "u") for f in ("flag:x=test", "page:/a", "page:/b")}
    rep = db.observe("u", "rage", 120)
    assert isinstance(rep, ObserveReport)
    total = 0.1 + s1 + s2
    r = {c.event_id: c.responsibility for c in rep.credits}
    assert r[e1.id] == pytest.approx(s1 / total)
    assert r[e2.id] == pytest.approx(s2 / total)
    assert rep.leak_responsibility == pytest.approx(0.1 / total)
    assert rep.total_credit + rep.leak_responsibility == pytest.approx(1.0)
    # Each event's features: alpha + beta grows by exactly r_e; alphas sum to
    # r_e * sum(share_f) with share = proportional credit.
    for ev in (e1, e2):
        for f in ev.features:
            a = db.link("rage", f, "u")
            b0 = before[f]
            assert (a.alpha + a.beta) - (b0.alpha + b0.beta) == pytest.approx(r[ev.id])
    c1 = next(c for c in rep.credits if c.event_id == e1.id)
    assert set(c1.features) == e1.features
    # Two equal features of e1 -> equal share s/(1-(1-s)^2) each.
    s = 0.2
    share = s / (1 - (1 - s) ** 2)
    assert c1.features["flag:x=test"] == pytest.approx(r[e1.id] * share)
    # Global layer got the same update.
    g = db.link("rage", "page:/b")
    assert g.global_pos == pytest.approx(r[e2.id])


def test_observe_uniform_credit_mode():
    db = _db(credit="uniform")
    e = db.insert("u", 0, "page", {"a", "b"})
    rep = db.observe("u", "rage", 10)
    c = rep.credits[0]
    assert c.features == {"a": pytest.approx(c.responsibility), "b": pytest.approx(c.responsibility)}
    assert db.link("rage", "a", "u").neg == pytest.approx(0.0)
    assert e.id in db.effect("rage").obs.credited


def test_observe_respects_rules_and_window():
    db = _db()
    blocked = db.insert("u", 0, "rage_click", {"event:rage_click"})
    ok = db.insert("u", 10, "page", {"page:/a"})
    old = db.insert("u", -10 * 3600, "page", {"page:/old"})  # outside the window
    db.add_rule("rage", "effect-not-cause", None, 0.0, types={"rage_click"})
    rep = db.observe("u", "rage", 30)
    assert [c.event_id for c in rep.credits] == [ok.id]
    assert db.link("rage", "event:rage_click", "u").observations == 0
    assert db.link("rage", "page:/old", "u").observations == 0
    assert old.id not in db.effect("rage").obs.credited and blocked.id not in db.effect("rage").obs.credited


def test_observe_with_no_candidates_is_all_leak():
    db = _db()
    rep = db.observe("nobody", "rage", 100)
    assert rep.credits == () and rep.leak_responsibility == 1.0
    assert db.observation_stats("rage")["mean_leak_responsibility"] == 1.0


@pytest.mark.parametrize("leak", [1e12, math.inf])
def test_huge_leak_means_no_learning(leak):
    db = _db()
    db.insert("u", 0, "page", {"a", "b"})
    before = db.link("rage", "a", "u")
    rep = db.observe("u", "rage", 10, leak=leak)
    after = db.link("rage", "a", "u")
    assert rep.leak_responsibility == pytest.approx(1.0)
    assert after.strength == pytest.approx(before.strength, abs=1e-9)
    assert after.alpha - before.alpha == pytest.approx(0.0, abs=1e-9)


def test_leak_validation():
    with pytest.raises(ValueError):
        _db(leak=0)
    with pytest.raises(ValueError):
        _db(w_neg=-1)
    db = _db()
    with pytest.raises(ValueError):
        db.observe("u", "rage", 1, leak=-0.5)


def test_smaller_leak_gives_more_credit():
    out = []
    for leak in (0.01, 0.5, 5.0):
        db = _db(leak=leak)
        db.insert("u", 0, "page", {"a"})
        out.append(db.observe("u", "rage", 10).total_credit)
    assert out[0] > out[1] > out[2] > 0


# ---------------------------------------------------------------------------
# settle()
# ---------------------------------------------------------------------------


def test_settle_rejects_after_window_and_skips_credited():
    db = _db(w_neg=2.0)
    support = db.effect("rage").kernel.support  # 1200 s
    e1 = db.insert("u", 0, "page", {"a"})
    e2 = db.insert("u", 100, "page", {"b"})
    db.observe("u", "rage", 110)          # credits e1 and e2
    e3 = db.insert("u", 5000, "page", {"c"})
    e4 = db.insert("u", 5100, "page", {"d"})
    rep = db.settle(5000 + support - 1)   # e3's window is not over yet
    assert isinstance(rep, SettleReport)
    assert rep.events_rejected == 0 and rep.skipped_credited == 2
    assert db.link("rage", "c", "u").neg == 0
    rep = db.settle(5000 + support)
    assert rep.events_rejected == 1
    assert db.link("rage", "c", "u").neg == pytest.approx(2.0)
    assert db.link("rage", "a", "u").neg < 1.0  # only soft-credit beta, no rejection
    assert e1.id in db.effect("rage").obs.credited and e2.id in db.effect("rage").obs.credited
    assert db.link("rage", "d", "u").neg == 0
    rep = db.settle(10_000)
    assert rep.events_rejected == 1 and db.link("rage", "d", "u").neg == pytest.approx(2.0)
    assert e4.id in db.effect("rage").obs.rejected


def test_settle_is_idempotent_and_incremental():
    db = _db()
    rng = random.Random(1)
    for i in range(200):
        db.insert(f"u{i % 5}", rng.uniform(0, 20_000), "page", {f"p{rng.randint(0, 9)}"})
    first = db.settle(30_000)
    assert first.events_rejected == 200
    snapshot = {f: db.link("rage", f).global_neg for f in (f"p{i}" for i in range(10))}
    assert sum(snapshot.values()) == pytest.approx(200.0)
    again = db.settle(30_000)
    assert again.events_settled == 0 and again.events_rejected == 0
    assert {f: db.link("rage", f).global_neg for f in snapshot} == snapshot
    # Only new events are examined afterwards (O(new events)).
    for i in range(7):
        db.insert("u0", 40_000 + i, "page", {"new"})
    rep = db.settle(60_000)
    assert rep.events_settled == 7 and rep.events_rejected == 7
    assert db.link("rage", "new").global_neg == pytest.approx(7.0)


def test_settle_handles_late_events_behind_the_cursor():
    db = _db()
    db.insert("u", 0, "page", {"a"})
    db.settle(10_000)
    late = db.insert("u", 50, "page", {"late"})    # arrives after settling
    rep = db.settle(10_000)
    assert rep.events_rejected == 1 and late.id in db.effect("rage").obs.rejected
    assert db.settle(10_000).events_rejected == 0


def test_settle_skips_events_whose_window_saw_the_effect_late():
    # An event that got *no* credit (score 0 due to rule) but whose window
    # contained an occurrence is not rejected either.
    db = _db()
    ev = db.insert("u", 0, "page", {"a"})
    db.observe("u", "rage", 30)
    assert ev.id in db.effect("rage").obs.credited
    rep = db.settle(100_000, effects="rage", entity="u")
    assert rep.events_rejected == 0 and rep.skipped_credited == 1


def test_settle_partial_and_undo_by_observe():
    db = _db()
    k = db.effect("rage").kernel
    ev = db.insert("u", 0, "page", {"a"})
    rep = db.settle(120, partial=True)  # one half-life elapsed
    frac = k.mass_fraction(120)
    assert 0.4 < frac < 0.6
    assert db.link("rage", "a", "u").neg == pytest.approx(frac, abs=1e-9)
    assert rep.events_rejected == 0  # not counted as fully settled
    db.settle(240, partial=True)       # top up, not double-count
    assert db.link("rage", "a", "u").neg == pytest.approx(k.mass_fraction(240), abs=1e-9)
    db.observe("u", "rage", 300)       # effect seen inside the window after all
    b = db.link("rage", "a", "u")
    r = db.effect("rage").obs.credited[ev.id]
    assert b.neg == pytest.approx(r * (1 - 1.0), abs=1e-9)  # single feature: share 1
    assert db.observation_stats("rage")["rejections_undone"] == 1
    assert db.settle(10_000).events_rejected == 0


def test_settle_one_shot_effect_skips_events_after_first_occurrence():
    db = HeapDB(bucket_seconds=600)
    db.register_effect("churn", ExponentialDecay(half_life=60), repeatable=False)
    db.register_effect("again", ExponentialDecay(half_life=60))
    db.insert("u", 0, "page", {"before"})
    db.observe("u", "churn", 5000)
    db.observe("u", "again", 5000)
    db.insert("u", 6000, "page", {"after"})
    rep = db.settle(100_000)
    assert db.link("churn", "before").global_neg == pytest.approx(1.0)
    assert db.link("churn", "after").global_neg == 0.0
    assert db.link("again", "after").global_neg == pytest.approx(1.0)
    assert db.observation_stats("churn")["skipped_terminal"] == 1
    assert rep.skipped_observed == 1


def test_settle_respects_cause_types_and_blocking_rules():
    db = HeapDB(bucket_seconds=600)
    db.register_effect("rage", ExponentialDecay(half_life=60), cause_types={"page"})
    db.add_rule("rage", "no-leave", None, 0.0, types={"leave"})
    db.insert("u", 0, "page", {"a"})
    db.insert("u", 0, "click", {"b"})
    rep = db.settle(10_000)
    assert rep.events_rejected == 1
    assert db.link("rage", "b").global_neg == 0


def test_settle_scales_by_rule_multiplier():
    db = _db()
    db.add_rule("rage", "pageleave-low", None, 0.25, types={"leave"})
    db.insert("u", 0, "leave", {"x"})
    db.settle(10_000)
    assert db.link("rage", "x").global_neg == pytest.approx(0.25)


def test_observe_settle_snapshot_round_trip(tmp_path):
    db = _db(leak=0.2, w_neg=0.5)
    db.register_effect("churn", ExponentialDecay(half_life=60), repeatable=False)
    db.insert("u", 0, "page", {"a"})
    db.insert("u", 3000, "page", {"b"})
    db.observe("u", "rage", 20)
    db.settle(2000)
    p = tmp_path / "s.jsonl"
    db.save(p)
    db2 = HeapDB.load(p)
    spec = db2.effect("rage")
    assert spec.leak == 0.2 and spec.w_neg == 0.5 and spec.repeatable
    assert db2.effect("churn").repeatable is False
    assert spec.obs.credited == db.effect("rage").obs.credited
    assert spec.obs.cursors == db.effect("rage").obs.cursors
    assert db2.settle(2000).events_rejected == 0
    assert db2.settle(10_000, effects="rage").events_rejected == 1
    assert db2.link("rage", "a", "u").strength == pytest.approx(db.link("rage", "a", "u").strength)


def test_forget_entity_drops_observation_state():
    db = _db()
    db.insert("u", 0, "page", {"a"})
    db.observe("u", "rage", 10)
    db.settle(10_000)
    db.forget_entity("u", scrub_global=True)
    obs = db.effect("rage").obs
    assert "u" not in obs.occurrences and "u" not in obs.cursors and not obs.credited
    assert db.link("rage", "a").observations == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# link_report()
# ---------------------------------------------------------------------------


def test_link_report_rows_interval_and_lift():
    db = _db()
    db.effect("rage").links.update("hot", 30, 70)
    db.effect("rage").links.update("cold", 2, 98)
    db.effect("rage").links.update("rare", 1, 0)
    rep = db.link_report("rage", min_support=5)
    assert isinstance(rep, LinkReport)
    assert [r.feature for r in rep] == ["hot", "cold"]
    hot = rep.row("hot")
    assert hot.alpha == 31 and hot.beta == 74 and hot.n == 100
    assert hot.lo < hot.strength < hot.hi
    assert betainc(31, 74, hot.lo) == pytest.approx(0.05, abs=1e-6)
    assert betainc(31, 74, hot.hi) == pytest.approx(0.95, abs=1e-6)
    assert rep.baseline == pytest.approx((hot.strength + rep.row("cold").strength) / 2)
    assert hot.lift == pytest.approx(hot.strength / rep.baseline)
    prior = db.link_report("rage", baseline="prior", min_support=0)
    assert prior.baseline == pytest.approx(0.2) and prior.row("rare") is not None
    assert "not causal" in str(rep)


def test_link_report_entity_view():
    db = _db()
    db.insert("u", 0, "page", {"a"})
    db.observe("u", "rage", 5)
    rep = db.link_report("rage", entity="u", min_support=0.1)
    assert rep.entity == "u" and rep.row("a") is not None


# ---------------------------------------------------------------------------
# query language
# ---------------------------------------------------------------------------


def test_query_language_observe_settle_links():
    assert parse("OBSERVE rage FOR u AT 120") == ObserveStmt("rage", "u", parse("OBSERVE rage FOR u AT 120").at)
    st = parse("observe rage for u at now-5m leak 0.5")
    assert isinstance(st, ObserveStmt) and st.leak == 0.5 and st.at.now_offset == 300
    s = parse("SETTLE UNTIL 900")
    assert isinstance(s, SettleStmt) and s.effect is None and s.until.absolute == 900
    s = parse("SETTLE rage UNTIL now FOR u")
    assert s.effect == "rage" and s.entity == "u"
    q = parse("LINKS OF rage FOR u MIN 3")
    assert q == LinksQuery("rage", "u", 3.0)
    assert parse("LINKS OF rage") == LinksQuery("rage", None, 1.0)
    for bad in ("OBSERVE rage", "OBSERVE rage FOR u LEAK 0", "SETTLE rage",
                "LINKS rage", "LINKS OF rage MIN x", "SETTLE UNTIL 5 BOGUS"):
        with pytest.raises(QueryError):
            parse(bad)

    db = _db()
    db.insert("u", 0, "page", {"a"})
    db.insert("u", 0, "page", {"b"})
    out = execute(db, "OBSERVE rage FOR u AT 30", clock=lambda: 0)
    assert isinstance(out, ObserveReport) and len(out.credits) == 2
    assert "leak" in format_result(out)
    out = execute(db, "SETTLE UNTIL 100000", clock=lambda: 0)
    assert isinstance(out, SettleReport)
    out = execute(db, "LINKS OF rage MIN 0.1", clock=lambda: 0)
    assert isinstance(out, LinkReport) and "90% CI" in format_result(out)


def test_repl_help_and_statements():
    db = _db()
    _, help_text = handle_line(db, ".help", clock=lambda: 0)
    for kw in ("OBSERVE", "SETTLE", "LINKS OF"):
        assert kw in help_text
    db.insert("u", 0, "page", {"a"})
    _, out = handle_line(db, "OBSERVE rage FOR u AT 10", clock=lambda: 0)
    assert "Observed 'rage'" in out
    _, out = handle_line(db, "LINKS OF rage", clock=lambda: 0)
    assert "a" in out


# ---------------------------------------------------------------------------
# synthetic simulation: noisy-OR generative model
# ---------------------------------------------------------------------------

TRUE = {"a": 0.30, "b": 0.15, "c": 0.08, "d": 0.03, "e": 0.0, "f": 0.0, "g": 0.0}


def _simulate(seed: int, n_users: int = 3000, credit: str = "proportional") -> HeapDB:
    """Users do 2-6 actions over 30 days; each action has each of a..g with
    p=0.3. Under noisy-OR every feature f independently triggers the effect
    with probability TRUE[f], after a log-normal lag (median 1 day). Plus 5%
    of users get a background occurrence.

    Feature ``z`` is a CONFOUNDER: it rides along with ``a`` 80% of the time
    (10% otherwise) and has no effect of its own.
    """
    rng = random.Random(seed)
    db = HeapDB(bucket_seconds=6 * H)
    db.register_effect("y", LogNormalKernel(median=D, sigma=0.5), leak=0.05, credit=credit)
    occ = []
    for u in range(n_users):
        ent = f"u{u}"
        for t in sorted(rng.uniform(0, 30 * D) for _ in range(rng.randint(2, 6))):
            feats = {f for f in TRUE if rng.random() < 0.3}
            if ("a" in feats and rng.random() < 0.8) or ("a" not in feats and rng.random() < 0.1):
                feats.add("z")
            if not feats:
                feats = {rng.choice("defg")}
            db.insert(ent, t, "act", feats)
            for f in feats:
                if rng.random() < TRUE.get(f, 0.0):
                    occ.append((t + math.exp(rng.gauss(math.log(D), 0.5)), ent))
        if rng.random() < 0.05:
            occ.append((rng.uniform(0, 32 * D), ent))
    for t, ent in sorted(occ):
        db.observe(ent, "y", t)
    db.settle(40 * D)
    return db


@pytest.mark.parametrize("seed", [7, 11])
def test_simulation_recovers_ranking_of_true_effects(seed):
    db = _simulate(seed)
    rep = db.link_report("y", min_support=5)
    feats = list(TRUE)
    rho = spearman([TRUE[f] for f in feats], [rep.row(f).strength for f in feats])
    assert rho >= 0.8
    # The strongest true cause is on top; every real cause beats every null.
    assert rep.rows[0].feature == "a"
    null_max = max(rep.row(f).strength for f in "efg")
    assert min(rep.row(f).strength for f in "abc") > null_max
    # Credible intervals of the top cause and the nulls do not overlap.
    assert rep.row("a").lo > max(rep.row(f).hi for f in "efg")
    st = db.observation_stats("y")
    assert 0.0 < st["mean_leak_responsibility"] < 0.5


def test_simulation_confounder_is_only_partly_separated():
    """Honest check of what observational credit does with a confounder.

    ``z`` has NO effect but co-occurs with ``a`` (the strongest cause).
    Soft/proportional credit gives ``z`` *less* than ``a`` (``a`` also fires
    without ``z``, and within an event ``a``'s higher strength takes more of
    the share) but ``z`` still ranks clearly above the true nulls -- about as
    high as a genuine moderate cause. Observational learning reduces, but
    does not remove, confounding; an experiment is needed to tell ``z``
    from a real cause.
    """
    db = _simulate(7)
    rep = db.link_report("y", min_support=5)
    z, a = rep.row("z"), rep.row("a")
    null_max = max(rep.row(f).strength for f in "efg")
    assert z.strength < a.strength              # partly separated
    assert z.strength > null_max                # ... but not fully
    assert z.lo > max(rep.row(f).hi for f in "efg")  # and confidently so
    assert z.lift > 1.0

    # Proportional credit keeps z below the moderate true cause b; v0.1-style
    # uniform credit separates less: z climbs above b and closer to a.
    assert z.strength < rep.row("b").strength
    uni = _simulate(7, credit="uniform").link_report("y", min_support=5)
    assert uni.row("z").strength > uni.row("b").strength
    assert uni.row("z").strength / uni.row("a").strength > z.strength / a.strength
