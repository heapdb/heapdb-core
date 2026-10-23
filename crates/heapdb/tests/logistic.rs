//! The `logistic` combine mode: link math (including degenerate cases),
//! learning directions, explanations, persistence, and property tests
//! (bounded, monotone, single-feature order, exact pruned top-k).

use std::path::PathBuf;

use heapdb::{sigmoid, Combine, Credit, EffectOptions, HeapDB, Kernel, NewEvent, TopK};
use proptest::prelude::*;

const H: f64 = 3600.0;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

fn logistic_db(opts: EffectOptions) -> HeapDB {
    let mut db = HeapDB::new(900.0, 64).unwrap();
    db.register_effect("fx", opts.combine(Combine::Logistic)).unwrap();
    db
}

fn window() -> Kernel {
    Kernel::window(0.0, 10.0 * H).unwrap()
}

fn link_of(db: &HeapDB, entity: &str, feats: &[&str]) -> f64 {
    let ev = heapdb::Event {
        id: 0,
        entity: entity.into(),
        time: 0.0,
        event_type: "t".into(),
        features: {
            let mut v: Vec<String> = feats.iter().map(|s| s.to_string()).collect();
            v.sort();
            v.dedup();
            v
        },
        weight: 1.0,
        attrs: Default::default(),
    };
    db.explain(&ev, "fx", H).unwrap().link_strength
}

// -- the math -------------------------------------------------------------------

#[test]
fn no_features_link_is_the_base_rate() {
    let db = logistic_db(EffectOptions::new(window()));
    // Default prior Beta(1, 4): base odds 1:4, link 0.2.
    assert!(close(link_of(&db, "me", &[]), 0.2));
    let spec = db.effect("fx").unwrap();
    assert_eq!(spec.combine(), Combine::Logistic);
    assert!(close(spec.links().base_log_odds().unwrap(), (0.25f64).ln()));
}

#[test]
fn features_at_the_prior_leave_the_link_at_the_base_rate() {
    let db = logistic_db(EffectOptions::new(window()));
    for fs in [&["a"][..], &["a", "b"], &["a", "b", "c", "d", "e"]] {
        assert!(close(link_of(&db, "me", fs), 0.2), "{fs:?}");
    }
    // ... unlike noisy-OR, where every extra feature adds probability.
    let mut nor = HeapDB::new(900.0, 64).unwrap();
    nor.register_effect("fx", EffectOptions::new(window())).unwrap();
    assert!(link_of(&nor, "me", &["a", "b"]) > link_of(&nor, "me", &["a"]));
}

#[test]
fn registered_priors_seed_odds_multipliers() {
    // Beta(3, 1): mean 0.75 = odds 3; against the base odds 1/4 that is a
    // multiplier of 12. One such feature: link = its prior mean.
    let db = logistic_db(
        EffectOptions::new(window())
            .prior("x", 3.0, 1.0)
            .prior("y", 3.0, 1.0),
    );
    assert!(close(db.effect("fx").unwrap().links().weight_seed(None), 0.0));
    assert!(close(link_of(&db, "me", &["x"]), 0.75));
    // Two: base odds 1/4 x 12 x 12 = 36 -> 36/37.
    assert!(close(link_of(&db, "me", &["x", "y"]), 36.0 / 37.0));
    // A feature at the default prior multiplies by 1.
    assert!(close(link_of(&db, "me", &["x", "z"]), 0.75));
}

#[test]
fn extreme_priors_stay_finite_and_bounded() {
    for (a, b) in [
        (1e-12, 1e12),
        (1e12, 1e-12),
        (1e-300, 1.0),
        (1.0, 1e-300),
        (1e-3, 10.0),
    ] {
        let db = logistic_db(
            EffectOptions::new(window())
                .default_prior(a, b)
                .prior("hi", 1e6, 1e-6)
                .prior("lo", 1e-6, 1e6),
        );
        for fs in [&[][..], &["hi"], &["lo"], &["hi", "lo"], &["hi", "lo", "x"]] {
            let l = link_of(&db, "me", fs);
            assert!(l.is_finite() && (0.0..=1.0).contains(&l), "{a} {b} {fs:?} -> {l}");
        }
    }
    // Moderate log-odds stay strictly inside (0, 1).
    let db = logistic_db(
        EffectOptions::new(window())
            .prior("hi", 1e6, 1.0)
            .prior("lo", 1.0, 1e6),
    );
    for fs in [&["hi"][..], &["lo"], &["hi", "lo"]] {
        let l = link_of(&db, "me", fs);
        assert!(l > 0.0 && l < 1.0, "{fs:?} -> {l}");
    }
}

#[test]
fn bad_params_and_modes_are_rejected() {
    let mut db = HeapDB::new(900.0, 64).unwrap();
    for o in [
        EffectOptions::new(window())
            .combine(Combine::Logistic)
            .learning_rate(0.0),
        EffectOptions::new(window())
            .combine(Combine::Logistic)
            .learning_rate(f64::NAN),
        EffectOptions::new(window()).combine(Combine::Logistic).l2(-1.0),
    ] {
        assert!(db.register_effect("fx", o).is_err());
    }
    assert!(Combine::parse("noisy-or").is_err());
    assert_eq!(Combine::parse("logistic").unwrap().as_str(), "logistic");
    assert_eq!(Combine::parse("noisy_or").unwrap(), Combine::NoisyOr);
}

#[test]
fn credit_defaults_follow_the_combine_mode() {
    let mut db = HeapDB::new(900.0, 64).unwrap();
    db.register_effect("a", EffectOptions::new(window())).unwrap();
    db.register_effect("b", EffectOptions::new(window()).combine(Combine::Logistic))
        .unwrap();
    db.register_effect(
        "c",
        EffectOptions::new(window())
            .combine(Combine::Logistic)
            .credit(Credit::Proportional),
    )
    .unwrap();
    assert_eq!(db.effect("a").unwrap().credit(), Credit::Proportional);
    assert_eq!(db.effect("b").unwrap().credit(), Credit::Uniform);
    assert_eq!(db.effect("c").unwrap().credit(), Credit::Proportional);
    assert_eq!(db.effect("a").unwrap().combine(), Combine::NoisyOr);
    assert!(db.effect("a").unwrap().logistic_params().is_none());
    let p = db.effect("b").unwrap().logistic_params().unwrap();
    assert_eq!((p.learning_rate, p.l2), (0.1, 1.0));
}

// -- learning -----------------------------------------------------------------------

fn bloat() -> (HeapDB, u64, u64) {
    let mut db = logistic_db(EffectOptions::new(Kernel::lognormal(2.0 * H, 0.6).unwrap()));
    let a = db
        .insert(NewEvent::new("me", 0.0, "meal").features(["beans", "onion"]))
        .unwrap()
        .id;
    let b = db
        .insert(NewEvent::new("me", 600.0, "meal").features(["rice", "onion"]))
        .unwrap()
        .id;
    (db, a, b)
}

#[test]
fn feedback_moves_weights_the_right_way() {
    let (mut db, a, b) = bloat();
    let w = |db: &HeapDB, f: &str| {
        db.effect("fx")
            .unwrap()
            .links()
            .weight(db.vocab().features.get(f))
            .unwrap()
    };
    let before = db.score(&db.get(a).unwrap(), "fx", 2.0 * H).unwrap();
    let rep = db.feedback("me", "fx", 2.0 * H, Some(a), &[b], None).unwrap();
    // Uniform (count) credit on the Beta ledger under logistic.
    assert!(rep.credit.iter().all(|(_, r)| *r == 1.0), "{:?}", rep.credit);
    assert!(w(&db, "beans") > 0.0, "confirmed-only feature goes up");
    assert!(w(&db, "rice") < 0.0, "rejected-only feature goes down");
    let (wb, wo, wr) = (w(&db, "beans"), w(&db, "onion"), w(&db, "rice"));
    assert!(wb > wo && wo > wr, "{wb} {wo} {wr}");
    let after = db.score(&db.get(a).unwrap(), "fx", 2.0 * H).unwrap();
    assert!(after > before);
    // The confirmed event's link is now above the rejected one's (the lag
    // kernel still weighs in the score; one step at lr 0.1 is small).
    assert!(link_of(&db, "me", &["beans", "onion"]) > link_of(&db, "me", &["rice", "onion"]));
}

#[test]
fn observe_and_settle_train_the_weights() {
    let (mut db, a, _) = bloat();
    let links = |db: &HeapDB| db.effect("fx").unwrap().links().clone();
    let id = |db: &HeapDB, f: &str| db.vocab().features.get(f);
    db.observe("me", "fx", 2.0 * H, None).unwrap();
    let l = links(&db);
    for f in ["beans", "onion", "rice"] {
        assert!(l.weight(id(&db, f)).unwrap() > 0.0, "{f} credited by observe");
    }
    // A meal whose window passes without the effect: negative example.
    db.insert(NewEvent::new("you", 0.0, "meal").features(["kale"]))
        .unwrap();
    db.settle(30.0 * H, None, Some("you"), false).unwrap();
    assert!(links(&db).weight(id(&db, "kale")).unwrap() < 0.0);
    // Partial settle then observe: the observe takes the negative back.
    let mut db2 = logistic_db(EffectOptions::new(Kernel::window(0.0, 4.0 * H).unwrap()));
    db2.insert(NewEvent::new("x", 0.0, "meal").features(["k"]))
        .unwrap();
    db2.settle(2.0 * H, None, None, true).unwrap();
    let neg = links(&db2).weight(id(&db2, "k")).unwrap();
    assert!(neg < 0.0);
    db2.observe("x", "fx", 3.0 * H, None).unwrap();
    assert!(links(&db2).weight(id(&db2, "k")).unwrap() > neg);
    let _ = a;
}

#[test]
fn scrub_global_removes_a_lone_entitys_contributions() {
    let (mut db, a, b) = bloat();
    db.feedback("me", "fx", 2.0 * H, Some(a), &[b], None).unwrap();
    let l = db.effect("fx").unwrap().links();
    assert!(l.logistic().unwrap().num_features() > 0);
    let rep = db.forget_entity("me", true);
    assert!(rep.events == 2);
    let l = db.effect("fx").unwrap().links();
    for f in ["beans", "onion", "rice"] {
        assert!(l.weight(db.vocab().features.get(f)).unwrap().abs() < 1e-12, "{f}");
    }
    assert!(close(l.base_log_odds().unwrap(), (0.25f64).ln()));
}

// -- explanations ---------------------------------------------------------------------

#[test]
fn explanation_is_base_odds_times_multipliers() {
    let (mut db, a, b) = bloat();
    db.feedback("me", "fx", 2.0 * H, Some(a), &[b], None).unwrap();
    let ex = db.explain(&db.get(a).unwrap(), "fx", 2.0 * H).unwrap();
    assert_eq!(ex.combine(), "logistic");
    let b0 = ex.base_log_odds.unwrap();
    let z: f64 = b0 + ex.features.iter().map(|c| c.log_odds.unwrap()).sum::<f64>();
    assert!(close(ex.log_odds().unwrap(), z));
    assert!(close(ex.link_strength, sigmoid(z)));
    let mut odds = ex.base_odds().unwrap();
    for c in &ex.features {
        odds *= c.odds_multiplier().unwrap();
    }
    assert!((odds - ex.odds().unwrap()).abs() <= 1e-12 * odds);
    assert!(close(
        ex.score(),
        db.score(&db.get(a).unwrap(), "fx", 2.0 * H).unwrap()
    ));
    // Strongest multiplier first.
    let ws: Vec<f64> = ex.features.iter().map(|c| c.log_odds.unwrap()).collect();
    assert!(ws.windows(2).all(|w| w[0] >= w[1]), "{ws:?}");
    let text = ex.to_string();
    assert!(
        text.contains("[logistic: base odds") && text.contains("-> link"),
        "{text}"
    );
    // Prospective risk explains the same way.
    let c = heapdb::Candidate::new(["beans", "onion"]);
    let (s, rex) = db.feature_risk("fx", &c, true).unwrap();
    let rex = rex.unwrap();
    assert!(close(s, rex.link_strength) && rex.base_log_odds.is_some());
}

#[test]
fn noisy_or_explanations_have_no_logistic_fields() {
    let mut db = HeapDB::new(900.0, 64).unwrap();
    db.register_effect("fx", EffectOptions::new(window()).prior("x", 3.0, 1.0))
        .unwrap();
    let ev = db
        .insert(NewEvent::new("me", 0.0, "t").features(["x", "y"]))
        .unwrap();
    let ex = db.explain(&ev, "fx", H).unwrap();
    assert_eq!(ex.combine(), "noisy_or");
    assert!(ex.base_log_odds.is_none() && ex.log_odds().is_none());
    assert!(ex.features.iter().all(|c| c.log_odds.is_none()));
    assert!(ex.to_string().contains("[noisy-OR of"));
    let rep = db.link_report("fx", &Default::default()).unwrap();
    assert!(rep.rows.iter().all(|r| r.log_odds.is_none()));
}

// -- persistence ---------------------------------------------------------------------

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn snapshot_round_trips_the_mode_and_the_weights() {
    let (mut db, a, b) = bloat();
    db.register_effect("plain", EffectOptions::new(window())).unwrap();
    db.feedback("me", "fx", 2.0 * H, Some(a), &[b], None).unwrap();
    db.observe("me", "fx", 3.0 * H, None).unwrap();
    let text = db.snapshot_string();
    assert!(text.contains("\"combine\":\"logistic\""));
    let db2 = HeapDB::load_from(text.as_bytes(), "mem").unwrap();
    assert_eq!(db2.snapshot_string(), text, "writing is a fixed point");
    assert_eq!(db2.effect("fx").unwrap().combine(), Combine::Logistic);
    assert_eq!(db2.effect("plain").unwrap().combine(), Combine::NoisyOr);
    assert_eq!(
        db2.effect("fx").unwrap().logistic_params(),
        db.effect("fx").unwrap().logistic_params()
    );
    let q = |d: &HeapDB| {
        d.top_causes("me", "fx", 2.0 * H, TopK::new(5).explain(true))
            .unwrap()
            .causes
    };
    assert_eq!(q(&db), q(&db2));
    // Learning continues identically after a reload (the ledger came along).
    let mut db3 = HeapDB::load_from(text.as_bytes(), "mem").unwrap();
    db.feedback("me", "fx", 4.0 * H, Some(b), &[a], None).unwrap();
    db3.feedback("me", "fx", 4.0 * H, Some(b), &[a], None).unwrap();
    assert_eq!(db.snapshot_string(), db3.snapshot_string());
}

#[test]
fn noisy_or_snapshots_have_no_combine_key_and_old_ones_load_as_noisy_or() {
    let mut db = HeapDB::new(900.0, 64).unwrap();
    db.register_effect("fx", EffectOptions::new(window())).unwrap();
    assert!(!db.snapshot_string().contains("combine"));
    for v in 1..=3 {
        let db = HeapDB::load(fixture(&format!("snapshot_v{v}.jsonl"))).unwrap();
        for e in db.effects() {
            assert_eq!(db.effect(&e).unwrap().combine(), Combine::NoisyOr);
        }
    }
}

// -- properties ---------------------------------------------------------------------

const FEATS: [&str; 8] = ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"];

fn prior_strategy() -> impl Strategy<Value = (f64, f64)> {
    prop_oneof![
        (0.05..20.0f64, 0.05..20.0f64),
        (1e-4..1e-2f64, 1.0..50.0f64),
        (1.0..50.0f64, 1e-4..1e-2f64),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 200, .. ProptestConfig::default() })]

    /// The link is `sigmoid(log-odds)`: always in [0, 1] and finite, and
    /// strictly inside (0, 1) unless the log-odds are so large that the
    /// sigmoid saturates in f64 (|z| > ~36.7).
    #[test]
    fn link_is_bounded(default in prior_strategy(),
                       priors in proptest::collection::vec((0..8usize, prior_strategy()), 0..8),
                       fs in proptest::collection::btree_set(0..8usize, 0..8)) {
        let mut o = EffectOptions::new(window()).default_prior(default.0, default.1);
        for (f, (a, b)) in &priors { o = o.prior(FEATS[*f], *a, *b); }
        let db = logistic_db(o);
        let names: Vec<&str> = fs.iter().map(|&f| FEATS[f]).collect();
        let l = link_of(&db, "me", &names);
        prop_assert!(l.is_finite() && (0.0..=1.0).contains(&l), "{}", l);
        let z = db.effect("fx").unwrap().links().log_odds_opt(
            &names.iter().map(|n| db.vocab().features.get(n)).collect::<Vec<_>>());
        if z.abs() < 30.0 {
            prop_assert!(l > 0.0 && l < 1.0, "z {} -> {}", z, l);
        }
    }

    /// Raising one feature's prior mean never lowers the link of an event
    /// that has it, whatever the other features are.
    #[test]
    fn link_is_monotone_in_feature_strength(p1 in 0.001..0.999f64, p2 in 0.001..0.999f64, k in 0.5..20.0f64,
                                            others in proptest::collection::vec((1..8usize, prior_strategy()), 0..5),
                                            fs in proptest::collection::btree_set(1..8usize, 0..6)) {
        let (lo, hi) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
        let make = |p: f64| {
            let mut o = EffectOptions::new(window()).prior("f0", k * p, k * (1.0 - p));
            for (f, (a, b)) in &others { o = o.prior(FEATS[*f], *a, *b); }
            logistic_db(o)
        };
        let mut names: Vec<&str> = fs.iter().map(|&f| FEATS[f]).collect();
        names.push("f0");
        prop_assert!(link_of(&make(lo), "me", &names) <= link_of(&make(hi), "me", &names) * (1.0 + 1e-12));
    }

    /// Before any learning, single-feature events rank exactly as under
    /// noisy-OR (the weight seed of a feature is its prior's log-odds
    /// against the default prior, a monotone map of the prior mean). After
    /// learning the orders can differ: the weights are fitted
    /// discriminatively, not counted.
    #[test]
    fn single_feature_order_matches_noisy_or_before_learning(
        priors in proptest::collection::vec(prior_strategy(), 8)) {
        let mut o = EffectOptions::new(window());
        for (f, (a, b)) in priors.iter().enumerate() { o = o.prior(FEATS[f], *a, *b); }
        let lg = logistic_db(o.clone());
        let mut nor = HeapDB::new(900.0, 64).unwrap();
        nor.register_effect("fx", o).unwrap();
        for i in 0..8 {
            for j in 0..8 {
                let (a, b) = (link_of(&lg, "me", &[FEATS[i]]), link_of(&lg, "me", &[FEATS[j]]));
                let (c, d) = (link_of(&nor, "me", &[FEATS[i]]), link_of(&nor, "me", &[FEATS[j]]));
                if (c - d).abs() > 1e-9 * c.max(d) {
                    prop_assert_eq!(a < b, c < d, "{} vs {}", FEATS[i], FEATS[j]);
                }
            }
        }
    }

    /// The pruned top-k equals brute force under logistic, before and after
    /// feedback / observe / settle, and segment bounds (set and union) are
    /// upper bounds of every score.
    #[test]
    fn pruned_topk_is_exact_under_logistic(
        events in proptest::collection::vec((0..2usize, 0.0..(2.0 * 86400.0), proptest::collection::btree_set(0..8usize, 0..4), 0.05..4.0f64), 1..150),
        max_sets in prop_oneof![Just(64usize), Just(2), Just(0)],
        lr in prop_oneof![Just(0.1), Just(0.5), Just(2.0)],
        steps in proptest::collection::vec((0..2usize, 0.0..(2.0 * 86400.0), 0..3u8), 1..15)) {
        let mut db = HeapDB::new(1800.0, max_sets).unwrap();
        db.register_effect("fx", EffectOptions::new(Kernel::lognormal(2.0 * H, 0.8).unwrap())
            .combine(Combine::Logistic).learning_rate(lr).prior("f0", 3.0, 1.0)).unwrap();
        for (e, t, fs, w) in &events {
            db.insert(NewEvent::new(["a", "b"][*e], *t, "t").features(fs.iter().map(|&f| FEATS[f])).weight(*w)).unwrap();
        }
        let check = |db: &HeapDB, at: f64| -> Result<(), TestCaseError> {
            for ent in ["a", "b"] {
                let got = db.top_causes(ent, "fx", at, TopK::new(4)).unwrap();
                let want = db.top_causes_naive(ent, "fx", at, 4, None).unwrap();
                let g: Vec<(u64, f64)> = got.causes.iter().map(|c| (c.event.id, c.score)).collect();
                let w: Vec<(u64, f64)> = want.iter().map(|(s, e)| (e.id, *s)).collect();
                prop_assert_eq!(g, w);
                let (t_lo, t_hi) = db.window("fx", at, None).unwrap();
                for b in db.segment_buckets(ent, Some((t_lo, t_hi))) {
                    let info = db.segment_info(ent, b).unwrap();
                    for use_sets in [true, false] {
                        let (bound, _) = db.segment_bound("fx", ent, b, ent, at, t_lo, t_hi, use_sets).unwrap();
                        for &id in &info.event_ids {
                            let ev = db.get(id).unwrap();
                            if t_lo <= ev.time && ev.time <= t_hi {
                                let sc = db.score(&ev, "fx", at).unwrap();
                                prop_assert!(sc <= bound * (1.0 + 1e-9), "score {} > bound {}", sc, bound);
                            }
                        }
                    }
                }
            }
            Ok(())
        };
        for (e, at, op) in steps {
            let ent = ["a", "b"][e];
            match op {
                0 => {
                    let r = db.top_causes(ent, "fx", at, TopK::new(3)).unwrap();
                    if r.causes.len() >= 2 {
                        db.feedback(ent, "fx", at, Some(r.causes[1].event.id), &[r.causes[0].event.id], None).unwrap();
                    }
                }
                1 => { db.observe(ent, "fx", at, None).unwrap(); }
                _ => { db.settle(at, None, None, at as u64 % 2 == 0).unwrap(); }
            }
            check(&db, at)?;
        }
    }
}
