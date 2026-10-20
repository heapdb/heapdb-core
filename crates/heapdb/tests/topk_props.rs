//! Property tests: the pruned top-k equals brute force, segment bounds are
//! true upper bounds, group queries equal brute force over the union, and
//! all of this survives learning (feedback, observe, settle).

use std::sync::Arc;

use heapdb::{Credit, EffectOptions, EventRef, HeapDB, Kernel, NewEvent, TopK};
use proptest::prelude::*;

const FEATURES: [&str; 12] = [
    "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11",
];
const TYPES: [&str; 3] = ["a", "b", "c"];
const ENTITIES: [&str; 3] = ["e1", "e2", "e3"];

#[derive(Debug, Clone)]
struct Ev {
    entity: usize,
    time: f64,
    ty: usize,
    feats: Vec<usize>,
    weight: f64,
    amount: i64,
}

#[derive(Debug, Clone)]
struct Setup {
    kernel: u8,
    bucket: f64,
    max_sets: usize,
    cause_types: bool,
    hierarchical: bool,
    priors: Vec<(usize, f64, f64)>,
    pred_rule: bool,
    type_rule: bool,
    block_rule: bool,
    events: Vec<Ev>,
}

fn kernel(k: u8) -> Kernel {
    match k % 5 {
        0 => Kernel::exponential(3600.0).unwrap(),
        1 => Kernel::window(600.0, 4.0 * 3600.0).unwrap(),
        2 => Kernel::lognormal(7200.0, 0.7).unwrap(),
        3 => Kernel::gamma(2.5, 2400.0).unwrap(),
        _ => Kernel::gamma(1.0, 900.0).unwrap(),
    }
}

fn ev_strategy() -> impl Strategy<Value = Ev> {
    (
        0..2usize,
        0.0..(2.0 * 86400.0),
        0..3usize,
        proptest::collection::btree_set(0..12usize, 0..4),
        prop_oneof![Just(0.05), Just(0.3), Just(1.0), Just(2.0), 0.01..8.0f64],
        prop_oneof![Just(1i64), Just(5), Just(50), Just(500)],
    )
        .prop_map(|(entity, time, ty, feats, weight, amount)| Ev {
            entity,
            time,
            ty,
            feats: feats.into_iter().collect(),
            weight,
            amount,
        })
}

fn setup_strategy() -> impl Strategy<Value = Setup> {
    (
        0..5u8,
        prop_oneof![Just(300.0), Just(600.0), Just(3600.0)],
        prop_oneof![Just(64usize), Just(2), Just(0)],
        any::<bool>(),
        any::<bool>(),
        proptest::collection::vec((0..12usize, 0.2..5.0f64, 0.2..5.0f64), 0..6),
        (any::<bool>(), any::<bool>(), any::<bool>()),
        proptest::collection::vec(ev_strategy(), 1..250),
    )
        .prop_map(
            |(kernel, bucket, max_sets, cause_types, hierarchical, priors, (p, t, b), events)| Setup {
                kernel,
                bucket,
                max_sets,
                cause_types,
                hierarchical,
                priors,
                pred_rule: p,
                type_rule: t,
                block_rule: b,
                events,
            },
        )
}

fn build(s: &Setup) -> HeapDB {
    let mut db = HeapDB::new(s.bucket, s.max_sets).unwrap();
    let mut opts = EffectOptions::new(kernel(s.kernel)).hierarchical(s.hierarchical);
    for &(f, a, b) in &s.priors {
        opts = opts.prior(FEATURES[f], a, b);
    }
    if s.cause_types {
        opts = opts.cause_types(["a", "b"]);
    }
    db.register_effect("fx", opts).unwrap();
    if s.pred_rule {
        let small = |e: &EventRef<'_>| e.attrs.get("amount").and_then(|v| v.as_i64()).unwrap_or(100) < 10;
        db.add_rule("fx", "small", Some(Arc::new(small)), 0.3, None)
            .unwrap();
    }
    if s.type_rule {
        db.add_rule("fx", "type-c", None, 0.5, Some(vec!["c".into()]))
            .unwrap();
    }
    if s.block_rule {
        let f0 = |e: &EventRef<'_>| e.features.contains("f0");
        db.add_rule("fx", "block-f0", Some(Arc::new(f0)), 0.0, None)
            .unwrap();
    }
    for e in &s.events {
        let mut attrs = heapdb::Attrs::new();
        attrs.insert("amount".into(), e.amount.into());
        db.insert(
            NewEvent::new(ENTITIES[e.entity], e.time, TYPES[e.ty])
                .features(e.feats.iter().map(|&f| FEATURES[f]))
                .weight(e.weight)
                .attrs(attrs),
        )
        .unwrap();
    }
    db
}

fn assert_exact(
    db: &HeapDB,
    entity: &str,
    at: f64,
    k: usize,
    within: Option<f64>,
) -> Result<(), TestCaseError> {
    let got = db
        .top_causes(entity, "fx", at, TopK::new(k).within(within))
        .unwrap();
    let want = db.top_causes_naive(entity, "fx", at, k, within).unwrap();
    let g: Vec<(u64, f64)> = got.causes.iter().map(|c| (c.event.id, c.score)).collect();
    let w: Vec<(u64, f64)> = want.iter().map(|(s, e)| (e.id, *s)).collect();
    prop_assert_eq!(g, w);
    let st = got.stats;
    prop_assert_eq!(st.segments_scanned + st.segments_pruned, st.segments_in_window);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 160, .. ProptestConfig::default() })]

    #[test]
    fn pruned_topk_equals_brute_force(
        s in setup_strategy(),
        queries in proptest::collection::vec((0..3usize, 0.0..(2.2 * 86400.0), 1..12usize,
                                              prop_oneof![Just(None), Just(Some(3600.0)), Just(Some(7200.0))]), 1..12),
    ) {
        let db = build(&s);
        for (e, at, k, within) in queries {
            assert_exact(&db, ENTITIES[e], at, k, within)?;
        }
    }

    #[test]
    fn segment_bounds_are_upper_bounds(s in setup_strategy(), ats in proptest::collection::vec(0.0..(2.2 * 86400.0), 1..6)) {
        let db = build(&s);
        for at in ats {
            let (t_lo, t_hi) = db.window("fx", at, None).unwrap();
            for ent in &ENTITIES[..2] {
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
        }
    }

    #[test]
    fn group_equals_brute_force_over_union(s in setup_strategy(), at in 0.0..(2.2 * 86400.0), k in 1..8usize) {
        let db = build(&s);
        let got = db.top_causes_group(&ENTITIES, "fx", at, TopK::new(k)).unwrap();
        let mut all: Vec<(f64, u64)> = ENTITIES
            .iter()
            .flat_map(|e| db.top_causes_naive(e, "fx", at, usize::MAX, None).unwrap())
            .map(|(s, e)| (s, e.id))
            .collect();
        all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(b.1.cmp(&a.1)));
        all.truncate(k);
        let g: Vec<(f64, u64)> = got.causes.iter().map(|c| (c.score, c.event.id)).collect();
        prop_assert_eq!(g, all);
        let ranks: Vec<usize> = got.causes.iter().map(|c| c.rank).collect();
        prop_assert_eq!(ranks, (1..=got.causes.len()).collect::<Vec<_>>());
    }

    #[test]
    fn still_exact_after_learning(s in setup_strategy(), steps in proptest::collection::vec((0..2usize, 0.0..(2.0 * 86400.0), 0..3u8), 1..25), uniform in any::<bool>()) {
        let mut db = build(&s);
        if uniform {
            db.effect_mut("fx").unwrap().set_credit(Credit::Uniform);
        }
        for (e, at, op) in steps {
            let ent = ENTITIES[e];
            match op {
                0 => {
                    let r = db.top_causes(ent, "fx", at, TopK::new(3)).unwrap();
                    if r.causes.len() >= 2 {
                        // Confirm a weaker candidate, reject the top: maximal churn.
                        db.feedback(ent, "fx", at, Some(r.causes[1].event.id), &[r.causes[0].event.id], None).unwrap();
                    }
                }
                1 => { db.observe(ent, "fx", at, None).unwrap(); }
                _ => { db.settle(at, None, None, at as u64 % 2 == 0).unwrap(); }
            }
            for x in ENTITIES {
                assert_exact(&db, x, at, 3, None)?;
            }
        }
    }

    #[test]
    fn snapshot_round_trip_preserves_answers(s in setup_strategy(), at in 0.0..(2.2 * 86400.0)) {
        let mut db = build(&s);
        if let Some(c) = db.top_causes("e1", "fx", at, TopK::new(2)).unwrap().causes.first() {
            let id = c.event.id;
            db.feedback("e1", "fx", at, Some(id), &[], None).unwrap();
        }
        db.observe("e2", "fx", at, None).unwrap();
        db.settle(at, None, None, false).unwrap();
        let text = db.snapshot_string();
        let db2 = HeapDB::load_from(text.as_bytes(), "mem").unwrap();
        // Predicate rules turn into pending_rules on the first load; from then
        // on writing is a fixed point.
        let text2 = db2.snapshot_string();
        if !(s.pred_rule || s.block_rule) {
            prop_assert_eq!(&text2, &text);
        }
        let db3 = HeapDB::load_from(text2.as_bytes(), "mem").unwrap();
        prop_assert_eq!(db3.snapshot_string(), text2);
        for e in ENTITIES {
            let a = db.top_causes(e, "fx", at, TopK::new(5)).unwrap();
            let b = db2.top_causes(e, "fx", at, TopK::new(5)).unwrap();
            // Predicate rules are not persisted: compare only without them.
            if !(s.pred_rule || s.block_rule) {
                prop_assert_eq!(a.causes, b.causes);
            }
        }
    }
}
