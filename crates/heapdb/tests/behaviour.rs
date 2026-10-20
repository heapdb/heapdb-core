//! Behaviour of the public Rust API (ports of the key checks of the Python
//! test suite).

use std::sync::Arc;

use heapdb::observe::{Baseline, LinkReportOptions};
use heapdb::query::{execute_str, parse, Output, Statement};
use heapdb::repl::{handle_line, Outcome};
use heapdb::{
    Attrs, Candidate, Credit, EffectOptions, Error, EventRef, ForwardDecayHeap, HeapDB, Kernel, NewEvent,
    TopK,
};

const H: f64 = 3600.0;
const DAY: f64 = 24.0 * H;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn bloating_db(credit: Credit) -> HeapDB {
    let mut db = HeapDB::new(900.0, 64).unwrap();
    db.register_effect(
        "bloating",
        EffectOptions::new(Kernel::lognormal(2.0 * H, 0.8).unwrap())
            .prior("beans", 4.0, 1.0)
            .cause_types(["meal"])
            .credit(credit),
    )
    .unwrap();
    db
}

fn meal(db: &mut HeapDB, t: f64, feats: &[&str]) -> u64 {
    db.insert(NewEvent::new("me", t, "meal").features(feats.iter().copied()))
        .unwrap()
        .id
}

#[test]
fn feedback_updates_beta_counts_and_reranks() {
    let mut db = bloating_db(Credit::Proportional);
    let a = meal(&mut db, 0.0, &["beans", "onion"]);
    let b = meal(&mut db, H, &["dairy"]);
    db.feedback("me", "bloating", 3.0 * H, Some(b), &[a], None)
        .unwrap();
    let l = |f: &str| db.link("bloating", f, None).unwrap();
    assert_eq!((l("dairy").alpha(), l("dairy").beta()), (2.0, 4.0));
    assert_eq!((l("beans").alpha(), l("beans").beta()), (4.0, 2.0));
    assert_eq!((l("onion").alpha(), l("onion").beta()), (1.0, 5.0));

    let mut db = bloating_db(Credit::Proportional);
    let mut tops = Vec::new();
    for day in 0..6 {
        let t0 = day as f64 * DAY;
        let beans = meal(&mut db, t0, &["beans"]);
        let dairy = meal(&mut db, t0 + 0.5 * H, &["dairy"]);
        let at = t0 + 2.5 * H;
        tops.push(
            db.top_causes("me", "bloating", at, TopK::new(2)).unwrap().causes[0]
                .event
                .features
                .clone(),
        );
        db.feedback("me", "bloating", at, Some(dairy), &[beans], None)
            .unwrap();
    }
    assert_eq!(tops[0], ["beans"]);
    assert_eq!(tops[5], ["dairy"]);
}

#[test]
fn proportional_credit_and_validation() {
    let mut db = bloating_db(Credit::Proportional);
    let ev = meal(&mut db, 0.0, &["beans", "onion"]);
    let rep = db
        .feedback("me", "bloating", 2.0 * H, Some(ev), &[], None)
        .unwrap();
    let p = 1.0 - 0.2 * 0.8;
    assert_eq!(rep.credit[0].0, "beans");
    assert!(close(rep.credit[0].1, 0.8 / p) && close(rep.credit[1].1, 0.2 / p));
    let rep = db
        .feedback("me", "bloating", 2.0 * H, Some(ev), &[], Some(Credit::Uniform))
        .unwrap();
    assert_eq!(
        rep.credit,
        vec![("beans".to_string(), 1.0), ("onion".to_string(), 1.0)]
    );
    // Validation is atomic.
    let before = db.link("bloating", "beans", Some("me")).unwrap();
    assert!(matches!(
        db.feedback("me", "bloating", 2.0 * H, Some(ev), &[999], None),
        Err(Error::UnknownEvent(999))
    ));
    assert!(db
        .feedback("me", "bloating", 2.0 * H, Some(ev), &[ev], None)
        .is_err());
    assert!(db.feedback("me", "bloating", -1.0, Some(ev), &[], None).is_err());
    assert!(db
        .feedback("you", "bloating", 2.0 * H, Some(ev), &[], None)
        .is_err());
    assert!(matches!(
        db.feedback("me", "nope", 1.0, Some(ev), &[], None),
        Err(Error::UnknownEffect(_))
    ));
    assert_eq!(db.link("bloating", "beans", Some("me")).unwrap(), before);
    assert_eq!(
        Credit::parse("bogus").unwrap_err().to_string(),
        "credit must be one of ('proportional', 'uniform'), got 'bogus'"
    );
}

#[test]
fn joint_causes_and_warnings() {
    let mut db = bloating_db(Credit::Proportional);
    let mut latte = 0;
    for d in 0..3 {
        latte = meal(&mut db, d as f64 * DAY, &["coffee", "dairy"]);
        db.feedback("me", "bloating", d as f64 * DAY + 2.0 * H, Some(latte), &[], None)
            .unwrap();
    }
    let ev = db.get(latte).unwrap();
    let ex = db.explain(&ev, "bloating", 2.0 * DAY + 2.0 * H).unwrap();
    assert!(ex.inseparable());
    assert_eq!(ex.joint_causes.len(), 1);
    assert_eq!(ex.joint_causes[0].name(), "coffee+dairy");
    assert!(close(ex.joint_causes[0].joint.unwrap().pos, 3.0));
    let text = ex.to_string();
    assert!(
        text.contains("coffee+dairy=") && text.contains("joint link"),
        "{text}"
    );
    assert!(ex.warnings[0].contains("'coffee', 'dairy' always co-occur (3 events)"));
    meal(&mut db, 3.0 * DAY, &["coffee"]);
    let ex = db.explain(&ev, "bloating", 2.0 * DAY + 2.0 * H).unwrap();
    assert!(!ex.inseparable() && ex.warnings.is_empty());
    assert!(ex.to_string().contains("now separable"));
}

#[test]
fn hierarchical_links_and_forget() {
    let mut db = HeapDB::default();
    db.register_effect(
        "bloating",
        EffectOptions::new(Kernel::lognormal(2.0 * H, 0.8).unwrap()).cause_types(["meal"]),
    )
    .unwrap();
    for u in 0..10 {
        for d in 0..4 {
            let ent = format!("u{u}");
            let id = db
                .insert(NewEvent::new(ent.as_str(), d as f64 * DAY, "meal").features(["dairy"]))
                .unwrap()
                .id;
            db.feedback(&ent, "bloating", d as f64 * DAY + 2.0 * H, Some(id), &[], None)
                .unwrap();
        }
    }
    let new = db.link("bloating", "dairy", Some("newbie")).unwrap();
    assert!(close(new.global_pos + new.global_neg, 5.0) && close(new.strength(), 0.6));
    assert!(close(
        db.link("bloating", "dairy", None).unwrap().strength(),
        41.0 / 45.0
    ));
    let own = db.link("bloating", "dairy", Some("u0")).unwrap();
    let before = db.link("bloating", "dairy", None).unwrap();
    let r = db.forget_entity("u0", true);
    assert_eq!((r.events, r.links, r.calibration_records), (4, 1, 4));
    let after = db.link("bloating", "dairy", None).unwrap();
    assert!(close(after.global_pos, before.global_pos - own.pos));
    assert!(db.entity_log("u0").is_none());
}

#[test]
fn rules_weights_and_risk() {
    let mut db = HeapDB::new(600.0, 64).unwrap();
    db.register_event_type(
        "deploy",
        Some(Arc::new(|a: &Attrs| {
            (a["lines_changed"].as_f64().unwrap()).ln_1p()
        })),
    );
    db.register_effect(
        "incident",
        EffectOptions::new(Kernel::lognormal(3.0 * H, 0.7).unwrap())
            .prior("migration", 3.0, 2.0)
            .prior("docs_only", 1.0, 20.0)
            .cause_types(["deploy"]),
    )
    .unwrap();
    let fast = |e: &EventRef<'_>| {
        e.attrs
            .get("review_minutes")
            .and_then(|v| v.as_f64())
            .unwrap_or(99.0)
            < 5.0
    };
    db.add_rule("incident", "fast review", Some(Arc::new(fast)), 0.5, None)
        .unwrap();
    db.add_rule(
        "incident",
        "deploys canaried",
        None,
        0.8,
        Some(vec!["deploy".into()]),
    )
    .unwrap();
    assert!(db.add_rule("incident", "bad", None, 1.5, None).is_err());
    let mut attrs = Attrs::new();
    attrs.insert("review_minutes".into(), 2.into());
    let c = Candidate {
        features: vec!["migration".into()],
        weight: Some(1.0),
        event_type: Some("deploy".into()),
        attrs,
        entity: None,
    };
    let (s, ex) = db.feature_risk("incident", &c, true).unwrap();
    assert!(close(s, 0.6 * 0.5 * 0.8));
    let ex = ex.unwrap();
    assert!(ex.prospective && close(ex.score(), s));
    assert!(ex.to_string().starts_with(
        "no lag (pre-event) -> weight 1 x link 0.600 x rules fast review x0.5, deploys canaried x0.8"
    ));
    let cron = Candidate {
        event_type: Some("cron".into()),
        ..Candidate::new(["migration"])
    };
    let (s, ex) = db.feature_risk("incident", &cron, true).unwrap();
    assert_eq!(s, 0.0);
    assert_eq!(ex.unwrap().rules[0].name, "type 'cron' not in cause_types");
    // Weight derived from attrs.
    let mut a = Attrs::new();
    a.insert("lines_changed".into(), 99.into());
    let d = db
        .insert(
            NewEvent::new("svc", 0.0, "deploy")
                .features(["migration"])
                .attrs(a),
        )
        .unwrap();
    assert!(close(d.weight, 99f64.ln_1p()));
    let mut zero = Attrs::new();
    zero.insert("lines_changed".into(), 0.into());
    assert!(db
        .insert(NewEvent::new("svc", 0.0, "deploy").attrs(zero))
        .is_err());
    // rank_candidates: ties keep the earlier candidate; zero scores dropped.
    let cands = vec![
        Candidate::new(["config"]),
        Candidate::new(["config"]),
        Candidate {
            event_type: Some("cron".into()),
            ..Candidate::new(["migration"])
        },
    ];
    let rows = db
        .rank_candidates("incident", &cands, 5, Some("svc"), false)
        .unwrap();
    assert_eq!(rows.iter().map(|r| r.index).collect::<Vec<_>>(), vec![0, 1]);
}

#[test]
fn observe_settle_and_link_report() {
    let mut db = HeapDB::new(600.0, 64).unwrap();
    db.register_effect(
        "rage",
        EffectOptions::new(Kernel::exponential(120.0).unwrap())
            .leak(0.1)
            .w_neg(2.0),
    )
    .unwrap();
    let e1 = db
        .insert(NewEvent::new("u", 0.0, "page").features(["flag:x=test", "page:/a"]))
        .unwrap();
    let e2 = db
        .insert(NewEvent::new("u", 60.0, "page").features(["page:/b"]))
        .unwrap();
    let (s1, s2) = (
        db.score(&e1, "rage", 120.0).unwrap(),
        db.score(&e2, "rage", 120.0).unwrap(),
    );
    let rep = db.observe("u", "rage", 120.0, None).unwrap();
    let total = 0.1 + s1 + s2;
    assert!(close(rep.leak_responsibility, 0.1 / total));
    assert!(close(rep.total_credit() + rep.leak_responsibility, 1.0));
    assert!(rep
        .to_string()
        .starts_with("Observed 'rage' for u at t=120: 2 candidate(s), leak 0.1 took"));
    let e3 = db
        .insert(NewEvent::new("u", 5000.0, "page").features(["c"]))
        .unwrap();
    let st = db.settle(5000.0 + 1200.0, None, None, false).unwrap();
    assert_eq!((st.events_rejected, st.skipped_credited), (1, 2));
    assert!(close(db.link("rage", "c", Some("u")).unwrap().neg, 2.0));
    assert_eq!(
        db.settle(5000.0 + 1200.0, None, None, false)
            .unwrap()
            .events_settled,
        0
    );
    assert!(db
        .effect("rage")
        .unwrap()
        .observations()
        .rejected
        .contains_key(&e3.id));
    let report = db
        .link_report(
            "rage",
            &LinkReportOptions {
                min_support: 0.1,
                baseline: Baseline::Prior,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(close(report.baseline, 0.2));
    let c = report.row("c").unwrap();
    assert!(c.lo < c.strength && c.strength < c.hi);
    assert!(close(heapdb::betainc(c.alpha, c.beta, c.lo).unwrap(), 0.05));
    assert!(report.to_string().contains("not causal effects"));
    assert!(db.observe("u", "rage", 1.0, Some(-1.0)).is_err());
    let stats = db.observation_stats("rage").unwrap();
    assert_eq!(stats["occurrences"], 1.0);
}

#[test]
fn calibration_report() {
    let mut db = HeapDB::default();
    db.register_effect(
        "fx",
        EffectOptions::new(Kernel::window(0.0, H).unwrap())
            .prior("good", 9.0, 1.0)
            .prior("bad", 1.0, 9.0),
    )
    .unwrap();
    for i in 0..10 {
        let t = i as f64 * 10.0 * H;
        let good = db
            .insert(NewEvent::new("e", t, "x").features(["good"]))
            .unwrap()
            .id;
        let bad = db
            .insert(NewEvent::new("e", t, "x").features(["bad"]))
            .unwrap()
            .id;
        if i < 9 {
            db.feedback("e", "fx", t + 60.0, Some(good), &[bad], None)
                .unwrap();
        } else {
            db.feedback("e", "fx", t + 60.0, None, &[good, bad], None)
                .unwrap();
        }
    }
    let rep = db.calibration_report("fx", 10).unwrap();
    assert_eq!(rep.n, 20);
    assert!(rep.brier.unwrap() > 0.0 && rep.brier.unwrap() < 0.1);
    assert!(close(db.calibration_records("fx").unwrap()[0].score, 0.9));
    assert!(rep.to_string().contains("Brier="));
    assert!(db.calibration_report("fx", 0).is_err());
}

#[test]
fn query_language_and_repl() {
    let mut db = HeapDB::default();
    db.register_effect(
        "bloating",
        EffectOptions::new(Kernel::lognormal(2.0 * H, 0.6).unwrap()).prior("beans", 3.0, 1.0),
    )
    .unwrap();
    let now = 10.0 * H;
    execute_str(
        &mut db,
        "INSERT meal FOR me AT now-2h FEATURES beans,onion WEIGHT 1.5",
        now,
    )
    .unwrap();
    execute_str(&mut db, "INSERT meal FOR me AT now-1h FEATURES apple", now).unwrap();
    let Output::Query(r) = execute_str(
        &mut db,
        "TOP 2 CAUSES OF bloating FOR me AT now WITHIN 12h EXPLAIN",
        now,
    )
    .unwrap() else {
        panic!()
    };
    assert_eq!(r.causes[0].event.features, ["beans", "onion"]);
    let msg = execute_str(&mut db, "FEEDBACK bloating FOR me AT now CONFIRMED 1", now).unwrap();
    assert_eq!(msg.to_string(), "ok: confirmed #1");
    assert!(close(
        db.link("bloating", "beans", Some("me")).unwrap().alpha(),
        3.0 + 0.75 / 0.8
    ));
    assert!(matches!(
        parse("RISK OF x FOR y FEATURES a").unwrap(),
        Statement::Risk(_)
    ));
    assert!(matches!(parse("TOP 0 CAUSES OF a FOR b"), Err(Error::Query(_))));

    let mut db = HeapDB::default();
    let mut clock = || Ok(0.0);
    for line in [
        ".effect bloating lognormal 2h 0.6",
        ".prior bloating beans 3 1",
        "INSERT meal FOR me AT 0 FEATURES beans,onion WEIGHT 1.5",
        "TOP 2 CAUSES OF bloating FOR me AT 7200 EXPLAIN",
    ] {
        handle_line(&mut db, line, &mut clock).unwrap();
    }
    let Outcome::Output(Some(out)) = handle_line(&mut db, ".rules bloating", &mut clock).unwrap() else {
        panic!()
    };
    assert_eq!(out, "(no rules)");
    assert!(handle_line(&mut db, "NONSENSE", &mut clock).is_err());
    assert!(matches!(
        handle_line(&mut db, ".quit", &mut clock).unwrap(),
        Outcome::Quit
    ));
    let Outcome::Output(Some(help)) = handle_line(&mut db, ".help", &mut clock).unwrap() else {
        panic!()
    };
    assert!(help.contains("CALIBRATION OF") && help.contains("group query"));
}

#[test]
fn forward_decay_matches_top_causes() {
    let mut db = HeapDB::new(600.0, 64).unwrap();
    db.register_effect("x", EffectOptions::new(Kernel::exponential(1200.0).unwrap()))
        .unwrap();
    let mut h = ForwardDecayHeap::from_half_life(1200.0).unwrap();
    for (i, (t, w)) in [(0.0, 3.0), (600.0, 1.0), (1500.0, 0.5), (3000.0, 0.2)]
        .iter()
        .enumerate()
    {
        let ev = db
            .insert(
                NewEvent::new("s", *t, "change")
                    .features([format!("f{i}")])
                    .weight(*w),
            )
            .unwrap();
        let ex = db.explain(&ev, "x", ev.time).unwrap();
        h.push(
            ev.id,
            ex.weight * ex.link_strength * ex.rule_multiplier(),
            ev.time,
        )
        .unwrap();
    }
    for now in [3000.0, 4000.0, 9000.0] {
        let top = *h.peek_top(now).unwrap().item;
        let r = db.most_likely_cause("s", "x", now).unwrap().unwrap();
        assert_eq!(top, r.event.id);
    }
}
