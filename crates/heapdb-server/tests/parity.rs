//! Engine parity: the same operations over HTTP and on the in-process
//! engine give identical rankings and scores (to 1e-9).

mod common;

use std::sync::Arc;

use common::*;
use heapdb::{Candidate, EffectOptions, HeapDB, Kernel, LinkReportOptions, NewEvent, TopK};
use heapdb_server::server::local_backends;
use serde_json::{json, Map, Value};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f64 {
        (self.next() % 1_000_000) as f64 / 1_000_000.0
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_matches_in_process_engine() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = test_config(dir.path());
    let n = TestNode::start(cfg.clone(), local_backends(&cfg).unwrap());
    let db = create_db(&n.url).await;

    // In-process engine with the same schema.
    let mut local = HeapDB::new(900.0, 64).unwrap();
    local
        .register_effect(
            "bloating",
            EffectOptions::new(Kernel::lognormal(2.0 * H, 0.6).unwrap())
                .prior("beans", 3.0, 1.0)
                .prior("dairy", 2.0, 2.0)
                .cause_types(["meal", "snack"]),
        )
        .unwrap();
    local.register_event_type(
        "meal",
        Some(Arc::new(|a: &Map<String, Value>| {
            a.get("portion").and_then(Value::as_f64).unwrap_or(1.0)
        })),
    );
    local
        .add_rule(
            "bloating",
            "tiny",
            Some(Arc::new(|e: &heapdb::EventRef<'_>| {
                e.attrs
                    .get("portion")
                    .and_then(Value::as_f64)
                    .is_some_and(|p| p < 0.5)
            })),
            0.5,
            None,
        )
        .unwrap();
    local
        .add_rule("bloating", "snacks", None, 0.8, Some(vec!["snack".into()]))
        .unwrap();
    demo_schema(&db).await;

    let feats = ["beans", "onion", "dairy", "rice", "chips", "garlic", "wheat"];
    let mut rng = Lcg(7);
    let mut batch = Vec::new();
    for i in 0..600 {
        let ent = format!("u{}", rng.next() % 5);
        let t = i as f64 * 300.0 + rng.unit() * 200.0;
        let ty = if rng.next().is_multiple_of(4) {
            "snack"
        } else {
            "meal"
        };
        let mut fs: Vec<String> = (0..1 + rng.next() % 3)
            .map(|_| feats[(rng.next() % 7) as usize].to_string())
            .collect();
        fs.sort();
        fs.dedup();
        let portion = 0.2 + rng.unit() * 2.0;
        let mut attrs = Map::new();
        attrs.insert("portion".into(), json!(portion));
        local
            .insert(
                NewEvent::new(ent.clone(), t, ty)
                    .features(fs.clone())
                    .attrs(attrs.clone()),
            )
            .unwrap();
        batch.push(json!({"entity": ent, "time": t, "type": ty, "features": fs, "attrs": attrs}));
        if batch.len() == 200 {
            db.post("/events", json!({"events": batch})).await.ok();
            batch = Vec::new();
        }
    }
    // Learning on both sides.
    for (ent, at) in [("u1", 40.0 * H), ("u2", 20.0 * H)] {
        let r = local.top_causes(ent, "bloating", at, TopK::new(3)).unwrap();
        let (conf, rej) = (r.causes[0].event.id, r.causes[1].event.id);
        local
            .feedback(ent, "bloating", at, Some(conf), &[rej], None)
            .unwrap();
        db.post(
            "/feedback",
            json!({"entity": ent, "effect": "bloating", "at": at, "confirmed_event_id": conf, "rejected_ids": [rej]}),
        )
        .await
        .ok();
    }
    local.observe("u3", "bloating", 30.0 * H, None).unwrap();
    db.post(
        "/observe",
        json!({"entity": "u3", "effect": "bloating", "at": 30.0 * H}),
    )
    .await
    .ok();
    local.settle(25.0 * H, None, None, false).unwrap();
    db.post("/settle", json!({"until": 25.0 * H})).await.ok();

    let mut compared = 0;
    for ent in ["u0", "u1", "u2", "u3", "u4"] {
        for at in [5.0 * H, 17.5 * H, 33.0 * H, 49.0 * H] {
            let want = local
                .top_causes(ent, "bloating", at, TopK::new(5).explain(true))
                .unwrap();
            let got = db
                .post(
                    "/query/top-causes",
                    json!({"entity": ent, "effect": "bloating", "at": at, "k": 5, "explain": true}),
                )
                .await
                .ok();
            let gc = got["causes"].as_array().unwrap();
            assert_eq!(gc.len(), want.causes.len());
            for (g, w) in gc.iter().zip(&want.causes) {
                assert_eq!(g["event"]["id"].as_u64().unwrap(), w.event.id, "{ent} at {at}");
                assert!(close(g["score"].as_f64().unwrap(), w.score), "{ent} at {at}");
                let ex = w.explanation.as_ref().unwrap();
                assert!(close(
                    g["explanation"]["link_strength"].as_f64().unwrap(),
                    ex.link_strength
                ));
                assert_eq!(
                    g["explanation"]["rules"].as_array().unwrap().len(),
                    ex.rules.len()
                );
                compared += 1;
            }
        }
    }
    assert!(compared > 30, "{compared}");
    // Group query.
    let want = local
        .top_causes_group(&["u0", "u1", "u2"], "bloating", 33.0 * H, TopK::new(6))
        .unwrap();
    let got = db
        .post(
            "/query/top-causes",
            json!({"entities": ["u0", "u1", "u2"], "effect": "bloating", "at": 33.0 * H, "k": 6}),
        )
        .await
        .ok();
    for (g, w) in got["causes"].as_array().unwrap().iter().zip(&want.causes) {
        assert_eq!(g["event"]["id"].as_u64().unwrap(), w.event.id);
        assert!(close(g["score"].as_f64().unwrap(), w.score));
    }
    // Risk and candidates.
    let c = Candidate {
        features: vec!["beans".into(), "garlic".into()],
        weight: Some(1.3),
        event_type: None,
        attrs: Map::new(),
        entity: Some("u1".into()),
    };
    let (want, _) = local.feature_risk("bloating", &c, false).unwrap();
    let got = db
        .post(
            "/query/risk",
            json!({"effect": "bloating", "features": ["beans", "garlic"], "weight": 1.3, "entity": "u1"}),
        )
        .await
        .ok();
    assert!(close(got["score"].as_f64().unwrap(), want));
    // Links report.
    let want = local
        .link_report(
            "bloating",
            &LinkReportOptions {
                min_support: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
    let got = db.get("/links?effect=bloating&min_support=0").await.ok();
    let rows = got["rows"].as_array().unwrap();
    assert_eq!(rows.len(), want.rows.len());
    for (g, w) in rows.iter().zip(&want.rows) {
        assert_eq!(g["feature"].as_str().unwrap(), w.feature);
        assert!(close(g["strength"].as_f64().unwrap(), w.strength));
    }
    // ... and parity survives a restart (snapshot + WAL replay).
    n.shutdown();
    let n = TestNode::start(cfg.clone(), local_backends(&cfg).unwrap());
    let db = db.on(&n.url);
    let want = local
        .top_causes("u1", "bloating", 49.0 * H, TopK::new(5))
        .unwrap();
    let got = db
        .post(
            "/query/top-causes",
            json!({"entity": "u1", "effect": "bloating", "at": 49.0 * H, "k": 5}),
        )
        .await
        .ok();
    for (g, w) in got["causes"].as_array().unwrap().iter().zip(&want.causes) {
        assert_eq!(g["event"]["id"].as_u64().unwrap(), w.event.id);
        assert!(close(g["score"].as_f64().unwrap(), w.score));
    }
    n.shutdown();
}
