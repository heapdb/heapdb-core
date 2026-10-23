//! Catalog endpoints: GET /effects, /entities, /event-types, /occurrences.

mod common;

use common::*;
use heapdb_server::server::local_backends;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};

fn node() -> (TestNode, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let cfg = test_config(dir.path());
    let b = local_backends(&cfg).unwrap();
    (TestNode::start(cfg, b), dir)
}

fn names(v: &Value, list: &str, key: &str) -> Vec<String> {
    v[list]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e[key].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catalog_endpoints() {
    let (n, _d) = node();
    let db = create_db(&n.url).await;

    // Empty database: empty lists, no next page, never errors.
    let e = db.get("/effects").await.ok();
    assert_eq!(e, json!({"effects": [], "next_cursor": null, "total_effects": 0}));
    let e = db.get("/entities").await.ok();
    assert_eq!(
        e,
        json!({"entities": [], "next_cursor": null, "matched": 0, "scanned": 0,
               "total_entities": 0, "truncated": false})
    );
    let e = db.get("/event-types").await.ok();
    assert_eq!(
        e,
        json!({"event_types": [], "next_cursor": null, "total_types": 0,
               "events_scanned": 0, "total_events": 0, "truncated": false})
    );
    let e = db.get("/occurrences").await.ok();
    assert_eq!(
        e,
        json!({"occurrences": [], "next_cursor": null, "truncated": false})
    );

    demo_schema(&db).await;
    db.post(
        "/effects",
        json!({"name": "incident", "kernel": {"type": "exponential", "params": {"half_life": 86400}},
               "combine": "logistic", "learning_rate": 0.5}),
    )
    .await
    .ok();
    let mut events = Vec::new();
    for (i, ent) in ["acme/web", "acme/api", "checkout", "me"].iter().enumerate() {
        for j in 0..=i {
            events.push(json!({
                "entity": ent, "time": (i as f64) * H + (j as f64) * 60.0,
                "type": if j % 2 == 0 { "meal" } else { "snack" },
                "features": ["beans", if j == 0 { "onion" } else { "chips" }],
            }));
        }
    }
    db.post("/events", json!({"events": events})).await.ok();

    // Effects: kernel in request and human form, priors, counts.
    let eff = db.get("/effects").await.ok();
    assert_eq!(names(&eff, "effects", "name"), ["bloating", "incident"]);
    assert_eq!(
        (eff["next_cursor"].clone(), eff["total_effects"].clone()),
        (Value::Null, json!(2))
    );
    let b = &eff["effects"][0];
    assert_eq!(
        (b["combine"].clone(), b["logistic"].clone()),
        (json!("noisy_or"), Value::Null)
    );
    let inc = &eff["effects"][1];
    assert_eq!(inc["combine"], "logistic");
    assert_eq!(inc["credit"], "uniform");
    assert_eq!(inc["logistic"]["learning_rate"], json!(0.5));
    assert_eq!(inc["logistic"]["l2"], json!(1.0));
    assert!(inc["logistic"]["base_log_odds"].is_number());
    // Effect pages.
    let p1 = db.get("/effects?limit=1").await.ok();
    assert_eq!(names(&p1, "effects", "name"), ["bloating"]);
    let cur = p1["next_cursor"].as_str().unwrap().to_string();
    let p2 = db.get(&format!("/effects?limit=1&cursor={cur}")).await.ok();
    assert_eq!(names(&p2, "effects", "name"), ["incident"]);
    assert_eq!(p2["next_cursor"], Value::Null);
    for bad in [
        "/effects?limit=0",
        "/effects?cursor=zz",
        "/effects?cursor=6e6f7065",
    ] {
        assert_eq!(db.get(bad).await.code(), "bad_request", "{bad}");
    }
    assert_eq!(b["kernel"]["type"], "lognormal");
    assert_eq!(b["kernel"]["params"]["median"], json!(2.0 * H));
    assert_eq!(b["kernel"]["human"]["median_s"], json!(2.0 * H));
    assert!(b["kernel"]["text"]
        .as_str()
        .unwrap()
        .starts_with("typical lag 2h"));
    assert_eq!(b["cause_types"], json!(["meal", "snack"]));
    assert_eq!(b["credit"], "proportional");
    assert_eq!(b["priors"]["count"], 2);
    assert_eq!(b["priors"]["top"][0]["feature"], "beans");
    assert_eq!(b["priors"]["top"][0]["strength"], json!(0.75));
    assert_eq!(b["rules"], json!(["tiny", "snacks"]));
    assert_eq!(b["counts"]["occurrences"], 0);
    assert_eq!(
        eff["effects"][1]["kernel"]["human"]["half_life_s"],
        json!(86400.0)
    );
    assert_eq!(eff["effects"][1]["cause_types"], Value::Null);

    // Entities: most recent first, prefix, pagination, limits.
    let all = db.get("/entities").await.ok();
    assert_eq!(
        names(&all, "entities", "name"),
        ["me", "checkout", "acme/api", "acme/web"]
    );
    assert_eq!(all["entities"][0]["event_count"], 4);
    assert_eq!(all["entities"][0]["last_event_time"], json!(3.0 * H + 180.0));
    assert_eq!(all["next_cursor"], Value::Null);
    assert_eq!(all["total_entities"], 4);
    let p1 = db.get("/entities?prefix=acme%2F&limit=1").await.ok();
    assert_eq!(names(&p1, "entities", "name"), ["acme/api"]);
    assert_eq!(p1["matched"], 2);
    let cur = p1["next_cursor"].as_str().unwrap().to_string();
    let p2 = db
        .get(&format!("/entities?prefix=acme/&limit=1&cursor={cur}"))
        .await
        .ok();
    assert_eq!(names(&p2, "entities", "name"), ["acme/web"]);
    assert_eq!(p2["next_cursor"], Value::Null);
    let none = db.get("/entities?prefix=zzz").await.ok();
    assert_eq!(
        (none["entities"].clone(), none["matched"].clone()),
        (json!([]), json!(0))
    );
    for bad in [
        "/entities?limit=0",
        "/entities?limit=abc",
        "/entities?cursor=nope",
    ] {
        assert_eq!(db.get(bad).await.code(), "bad_request", "{bad}");
    }
    assert_eq!(
        db.get("/entities?limit=100000").await.ok()["entities"]
            .as_array()
            .unwrap()
            .len(),
        4,
        "an oversized limit is capped, not rejected"
    );

    // Event types: counts and top features (registered types included).
    let et = db.get("/event-types?features=1").await.ok();
    assert_eq!(names(&et, "event_types", "type"), ["meal", "snack"]);
    assert_eq!(
        (et["next_cursor"].clone(), et["total_types"].clone()),
        (Value::Null, json!(2))
    );
    let p1 = db.get("/event-types?limit=1").await.ok();
    assert_eq!(names(&p1, "event_types", "type"), ["meal"]);
    let cur = p1["next_cursor"].as_str().unwrap().to_string();
    let p2 = db.get(&format!("/event-types?limit=1&cursor={cur}")).await.ok();
    assert_eq!(names(&p2, "event_types", "type"), ["snack"]);
    assert_eq!(p2["next_cursor"], Value::Null);
    let past = db.get("/event-types?cursor=ff").await.ok();
    assert_eq!(
        (past["event_types"].clone(), past["next_cursor"].clone()),
        (json!([]), Value::Null)
    );
    for bad in [
        "/event-types?features=0",
        "/event-types?limit=x",
        "/event-types?cursor=-1",
    ] {
        assert_eq!(db.get(bad).await.code(), "bad_request", "{bad}");
    }
    let meal = &et["event_types"][0];
    assert_eq!(meal["count"], 6);
    assert_eq!(meal["registered"], true);
    assert_eq!(meal["weight"]["kind"], "attr");
    assert_eq!(meal["top_features"], json!([{"feature": "beans", "count": 6}]));
    assert_eq!(et["event_types"][1]["registered"], false);
    assert_eq!(
        (
            et["events_scanned"].clone(),
            et["total_events"].clone(),
            et["truncated"].clone()
        ),
        (json!(10), json!(10), json!(false))
    );

    // Occurrences: observe and feedback, newest first, filters, limit.
    for (ent, at) in [("me", 3.5 * H), ("checkout", 2.5 * H), ("me", 4.0 * H)] {
        db.post("/observe", json!({"entity": ent, "effect": "bloating", "at": at}))
            .await
            .ok();
    }
    db.post(
        "/feedback",
        json!({"entity": "me", "effect": "bloating", "at": 4.0 * H, "confirmed_event_id": 7}),
    )
    .await
    .ok();
    let occ = db.get("/occurrences?effect=bloating").await.ok();
    let rows = occ["occurrences"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["entity"], "me");
    assert_eq!(rows[0]["at"], json!(4.0 * H));
    assert_eq!(rows[0]["sources"], json!(["observe", "feedback"]));
    assert_eq!(rows[0]["confirmed_event_id"], 7);
    assert_eq!(rows[2]["entity"], "checkout");
    assert_eq!(occ["next_cursor"], Value::Null);
    // Occurrence pages: one at a time, same rows in the same order.
    let mut paged = Vec::new();
    let mut url = "/occurrences?effect=bloating&limit=1".to_string();
    loop {
        let p = db.get(&url).await.ok();
        let rows = p["occurrences"].as_array().unwrap();
        assert_eq!(rows.len(), 1);
        paged.push(rows[0].clone());
        match p["next_cursor"].as_str() {
            Some(c) => url = format!("/occurrences?effect=bloating&limit=1&cursor={c}"),
            None => break,
        }
    }
    assert_eq!(&paged, rows);
    for bad in ["/occurrences?limit=0", "/occurrences?cursor=abc"] {
        assert_eq!(db.get(bad).await.code(), "bad_request", "{bad}");
    }
    let one = db.get("/occurrences?entity=checkout&limit=5").await.ok();
    assert_eq!(one["occurrences"].as_array().unwrap().len(), 1);
    assert_eq!(
        db.get("/occurrences?limit=1").await.ok()["occurrences"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let e = db.get("/occurrences?effect=nope").await;
    assert_eq!((e.status, e.code()), (StatusCode::NOT_FOUND, "unknown_effect"));
    let counts = &db.get("/effects").await.ok()["effects"][0]["counts"];
    assert_eq!(counts["occurrences"], 3);
    assert_eq!(counts["occurrence_entities"], 2);
    assert_eq!(counts["feedback"], 1);
    assert_eq!(counts["feedback_confirmed"], 1);

    // Wrong methods.
    assert_eq!(db.delete("/effects").await.code(), "method_not_allowed");
    assert_eq!(
        db.post("/occurrences", json!({})).await.code(),
        "method_not_allowed"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catalog_needs_read_scope_and_is_counted() {
    let (n, _d) = node();
    let db = create_db(&n.url).await;
    demo_schema(&db).await;
    let paths = ["/effects", "/entities", "/event-types", "/occurrences"];
    // The same scope rules as the existing read endpoints, e.g. GET /calibration.
    let reference = "/calibration?effect=bloating";

    let wo = db.with_key(&add_key(&n.url, &db.id, "key_wo", &["write"]).await);
    let ad = db.with_key(&add_key(&n.url, &db.id, "key_ad", &["admin"]).await);
    for p in paths.iter().copied().chain([reference]) {
        for k in [&wo, &ad] {
            let r = k.get(p).await;
            assert_eq!(
                (r.status, r.code()),
                (StatusCode::FORBIDDEN, "insufficient_scope"),
                "{p}"
            );
        }
    }
    let ro = db.with_key(&add_key(&n.url, &db.id, "key_ro", &["read"]).await);
    // Without a key, or with a wrong one: unauthenticated, like every Data API call.
    for p in paths {
        let r = db.with_key("hdb_nope").get(p).await;
        assert_eq!(r.status, StatusCode::UNAUTHORIZED, "{p}");
    }
    let before = ro.get("").await.ok()["usage"]["requests_this_period"]
        .as_u64()
        .unwrap();
    for p in paths {
        ro.get(p).await.ok();
    }
    // The schema change endpoints keep their admin scope.
    let r = ro
        .post(
            "/effects",
            json!({"name": "x", "kernel": {"type": "exp", "params": {"half_life": 1}}}),
        )
        .await;
    assert_eq!(r.code(), "insufficient_scope");
    // Counted like every Data API request (usage is flushed asynchronously).
    let mut after = 0;
    for _ in 0..50 {
        after = ro.get("").await.ok()["usage"]["requests_this_period"]
            .as_u64()
            .unwrap();
        if after >= before + 4 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert_eq!(after, before + 4);
}

fn keys(v: &Value) -> Vec<String> {
    let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
    k.sort();
    k
}

/// Every field a catalog response (and each of its rows) carries is in the
/// OpenAPI document, and every documented field is present.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catalog_matches_openapi() {
    let (n, _d) = node();
    let db = create_db(&n.url).await;
    demo_schema(&db).await;
    db.post(
        "/effects",
        json!({"name": "churn", "kernel": {"type": "exponential", "params": {"half_life": 60}}, "combine": "logistic"}),
    )
    .await
    .ok();
    db.post(
        "/events",
        json!({"events": [{"entity": "me", "time": 0.0, "type": "meal", "features": ["beans"]}]}),
    )
    .await
    .ok();
    db.post("/observe", json!({"entity": "me", "effect": "bloating", "at": H}))
        .await
        .ok();
    let doc = send(
        &http(),
        Method::GET,
        &format!("{}/v1/openapi.json", n.url),
        None,
        None,
    )
    .await
    .ok();
    let schema = |name: &str| doc["components"]["schemas"][name].clone();
    let props = |s: &Value| {
        let mut k: Vec<String> = s["properties"].as_object().unwrap().keys().cloned().collect();
        k.sort();
        k
    };
    for (path, list, row) in [
        ("/effects", "effects", "EffectInfo"),
        ("/entities", "entities", "EntityInfo"),
        ("/event-types", "event_types", "EventTypeInfo"),
        ("/occurrences", "occurrences", "OccurrenceInfo"),
    ] {
        let op = &doc["paths"][format!("/v1/databases/{{database_id}}{path}")]["get"];
        let res = &op["responses"]["200"]["content"]["application/json"]["schema"];
        let got = db.get(&format!("{path}?limit=1")).await.ok();
        assert_eq!(keys(&got), props(res), "{path}");
        let mut req: Vec<String> = res["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        req.sort();
        assert_eq!(req, props(res), "{path}: every field is always present");
        let rows = got[list].as_array().unwrap();
        assert!(!rows.is_empty(), "{path}");
        let s = schema(row);
        for r in rows {
            assert_eq!(keys(r), props(&s), "{path} row");
        }
        let mut req: Vec<String> = s["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        req.sort();
        assert_eq!(req, props(&s), "{row}: every field is always present");
        let params: Vec<&str> = op["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["name"].as_str().unwrap())
            .collect();
        assert!(
            params.contains(&"limit") && params.contains(&"cursor"),
            "{path}: {params:?}"
        );
    }
    // POST /effects and POST /event-types are still documented next to the GETs.
    for p in ["/effects", "/event-types"] {
        assert!(doc["paths"][format!("/v1/databases/{{database_id}}{p}")]["post"].is_object());
    }
}
