//! Keyed heaps over HTTP: every route, scopes, validation, batch atomicity,
//! percent-encoded names, quotas, /hql heap statements, and recovery after
//! a restart (WAL replay and snapshots), on in-memory storage.

mod common;

use std::time::Duration;

use common::*;
use heapdb_server::storage::Backends;
use reqwest::StatusCode;
use serde_json::{json, Value};

const T0: f64 = 1_760_000_000.0;

fn node_on(b: &Backends, tweak: impl FnOnce(&mut heapdb_server::config::Config)) -> TestNode {
    let mut cfg = test_config(&std::env::temp_dir());
    tweak(&mut cfg);
    TestNode::start(cfg, b.clone())
}

fn ids(v: &Value) -> Vec<String> {
    v["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn heap_routes() {
    let b = Backends::memory();
    let n = node_on(&b, |_| {});
    let db = create_db(&n.url).await;
    let h = "/heaps/acme%2Fcheckout"; // name "acme/checkout"

    // Create (idempotent; conflicting spec -> 409).
    let info = db
        .put(
            h,
            json!({"half_life": 3600, "landmark": T0, "meta": {"team": "pay"}}),
        )
        .await
        .ok();
    assert_eq!(
        info,
        json!({"name": "acme/checkout", "half_life": 3600.0, "landmark": T0, "meta": {"team": "pay"}, "size": 0})
    );
    assert_eq!(db.put(h, json!({"half_life": 3600})).await.ok(), info);
    let r = db.put(h, json!({"half_life": 60})).await;
    assert_eq!((r.status, r.code()), (StatusCode::CONFLICT, "heap_conflict"));
    let r = db.put(h, json!({"half_life": -1})).await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::BAD_REQUEST, "invalid_argument")
    );
    let r = db.put(h, json!({"half_life": 1, "bogus": 1})).await;
    assert_eq!(r.code(), "bad_request", "unknown fields are rejected");
    let r = db
        .put(&format!("/heaps/{}", "x".repeat(201)), json!({"half_life": 1}))
        .await;
    assert_eq!(r.code(), "invalid_argument");

    // Push (batched, atomic, defaults weight 1 / time now).
    let r = db
        .post(
            &format!("{h}/push"),
            json!({"items": [
                {"id": "CHK-12", "weight": 1.5, "time": T0, "attrs": {"kind": "bug", "required_skills": ["ts"]}},
                {"id": "CHK/13", "weight": 1, "time": T0, "attrs": {"kind": "task", "labels": ["needs-human"]}},
                {"id": "CHK-12", "weight": 0.5, "time": T0}
            ]}),
        )
        .await
        .ok();
    assert_eq!(r["items"][2]["version"], json!(2));
    assert_eq!(
        r["items"][2]["attrs"]["kind"],
        json!("bug"),
        "attrs kept when omitted"
    );
    assert!((r["items"][2]["key"].as_f64().unwrap() - 2f64.ln()).abs() < 1e-12);
    let bad = db
        .post(
            &format!("{h}/push"),
            json!({"items": [{"id": "new", "weight": 1, "time": T0}, {"id": "bad", "weight": 0}]}),
        )
        .await;
    assert_eq!(bad.code(), "invalid_argument");
    assert!(bad.body["error"]["message"]
        .as_str()
        .unwrap()
        .starts_with("items[1]: weight"));
    assert_eq!(
        db.get(h).await.ok()["size"],
        json!(2),
        "a failing batch stores nothing"
    );
    let r = db.post(&format!("{h}/push"), json!({"items": []})).await;
    assert_eq!(r.code(), "bad_request");
    let big: Vec<Value> = (0..1001).map(|i| json!({"id": format!("i{i}")})).collect();
    let r = db.post(&format!("{h}/push"), json!({"items": big})).await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::PAYLOAD_TOO_LARGE, "batch_too_large")
    );
    let r = db.post("/heaps/nope/push", json!({"items": [{"id": "a"}]})).await;
    assert_eq!((r.status, r.code()), (StatusCode::NOT_FOUND, "unknown_heap"));

    // Base and attrs.
    let r = db
        .post(
            &format!("{h}/base"),
            json!({"items": [{"id": "CHK-12", "base": 4}, {"id": "idle", "base": 9}]}),
        )
        .await
        .ok();
    assert_eq!(
        r["items"][1]["key"],
        Value::Null,
        "set_base creates the item unranked"
    );
    let r = db
        .post(
            &format!("{h}/attrs"),
            json!({"items": [{"id": "CHK/13", "attrs": {"risk": 2}, "merge": true}]}),
        )
        .await
        .ok();
    assert_eq!(
        r["items"][0]["attrs"],
        json!({"kind": "task", "labels": ["needs-human"], "risk": 2})
    );

    // Top with filters; unranked items are not listed.
    let top = db
        .post(&format!("{h}/top"), json!({"k": 10, "now": T0 + 3600.0}))
        .await
        .ok();
    assert_eq!(ids(&top), ["CHK-12", "CHK/13"]);
    let r0 = &top["items"][0];
    assert_eq!((r0["rank"].clone(), r0["base"].clone()), (json!(1), json!(4.0)));
    assert!(
        (r0["score"].as_f64().unwrap() - 4.0).abs() < 1e-9,
        "base 4 x heat 1 after one half-life: {r0}"
    );
    let f = json!({"all": [
        {"attr": "kind", "op": "in", "value": ["bug", "task"]},
        {"not": {"attr": "labels", "op": "contains", "value": "needs-human"}}
    ]});
    let top = db
        .post(&format!("{h}/top"), json!({"k": 5, "now": T0, "filter": f}))
        .await
        .ok();
    assert_eq!(ids(&top), ["CHK-12"]);
    let r = db
        .post(
            &format!("{h}/top"),
            json!({"filter": {"attr": "kind", "op": "like", "value": "b"}}),
        )
        .await;
    assert_eq!(r.code(), "invalid_argument");
    assert!(r.body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("unknown op 'like'"));

    // Get / remove items by percent-encoded id.
    let g = db.get(&format!("{h}/items/CHK%2F13?now={}", T0)).await.ok();
    assert_eq!((g["id"].clone(), g["rank"].clone()), (json!("CHK/13"), json!(0)));
    assert_eq!(db.get(&format!("{h}/items/missing")).await.code(), "unknown_item");
    assert_eq!(
        db.delete(&format!("{h}/items/CHK%2F13")).await.ok(),
        json!({"removed": true})
    );
    assert_eq!(
        db.delete(&format!("{h}/items/CHK%2F13")).await.ok(),
        json!({"removed": false})
    );

    // Pop (filter, empty body allowed) and rebase.
    let p = db
        .post(
            &format!("{h}/pop"),
            json!({"now": T0, "filter": {"attr": "kind", "op": "eq", "value": "task"}}),
        )
        .await
        .ok();
    assert_eq!(p, json!({"item": null}));
    let p = db.post(&format!("{h}/pop"), json!({"now": T0})).await.ok();
    assert_eq!(p["item"]["id"], json!("CHK-12"));
    db.post(&format!("{h}/push"), json!({"items": [{"id": "z", "time": T0}]}))
        .await
        .ok();
    let before = db
        .post(&format!("{h}/top"), json!({"now": T0 + 100.0}))
        .await
        .ok();
    let info = db
        .post(&format!("{h}/rebase"), json!({"now": T0 + 50.0}))
        .await
        .ok();
    assert_eq!(info["landmark"], json!(T0 + 50.0));
    let after = db
        .post(&format!("{h}/top"), json!({"now": T0 + 100.0}))
        .await
        .ok();
    let s = |v: &Value| v["items"][0]["score"].as_f64().unwrap();
    assert!((s(&before) - s(&after)).abs() < 1e-12);

    // List, drop.
    let l = db.get("/heaps").await.ok();
    assert_eq!(l["heaps"][0]["name"], json!("acme/checkout"));
    assert_eq!(db.delete(h).await.ok(), json!({"dropped": true}));
    assert_eq!(db.delete(h).await.ok(), json!({"dropped": false}));
    assert_eq!(db.get(h).await.code(), "unknown_heap");
    let r = db.post("/heaps/x/frobnicate", json!({})).await;
    assert_eq!(r.code(), "not_found");

    // OpenAPI lists the routes.
    let doc = send(
        &http(),
        reqwest::Method::GET,
        &format!("{}/v1/openapi.json", n.url),
        None,
        None,
    )
    .await
    .ok();
    for p in [
        "/heaps",
        "/heaps/{heap}",
        "/heaps/{heap}/push",
        "/heaps/{heap}/items/{item}",
        "/heaps/{heap}/pop",
    ] {
        assert!(
            doc["paths"][format!("/v1/databases/{{database_id}}{p}")].is_object(),
            "{p}"
        );
    }
    n.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn heap_scopes_hql_and_quotas() {
    let b = Backends::memory();
    let n = node_on(&b, |_| {});
    let db = create_db(&n.url).await;
    db.put("/heaps/w", json!({"half_life": 3600, "landmark": T0}))
        .await
        .ok();
    let reader = db.with_key(&add_key(&n.url, &db.id, "key_r", &["read"]).await);
    let writer = db.with_key(&add_key(&n.url, &db.id, "key_w", &["write"]).await);
    let forbidden = |r: Resp| {
        assert_eq!(
            (r.status, r.code()),
            (StatusCode::FORBIDDEN, "insufficient_scope"),
            "{}",
            r.body
        )
    };
    // Admin routes.
    forbidden(writer.put("/heaps/x", json!({"half_life": 1})).await);
    forbidden(writer.delete("/heaps/w").await);
    forbidden(writer.post("/heaps/w/rebase", json!({})).await);
    // Write routes.
    forbidden(
        reader
            .post("/heaps/w/push", json!({"items": [{"id": "a"}]}))
            .await,
    );
    forbidden(reader.post("/heaps/w/pop", json!({})).await);
    forbidden(reader.delete("/heaps/w/items/a").await);
    writer
        .post(
            "/heaps/w/push",
            json!({"items": [{"id": "a", "time": T0, "attrs": {"kind": "bug"}}]}),
        )
        .await
        .ok();
    // Read routes.
    reader.get("/heaps").await.ok();
    reader.post("/heaps/w/top", json!({"now": T0})).await.ok();
    reader.get("/heaps/w/items/a").await.ok();

    // /hql: PEEK read, PUSH/POP write, CREATE/DROP/REBASE admin.
    let hql = |q: &str| json!({"query": q, "now": T0 + 60.0});
    let r = reader
        .post("/hql", hql("PEEK 3 FROM w WHERE kind IN ('bug')"))
        .await
        .ok();
    assert_eq!(r["kind"], json!("heap_items"));
    assert_eq!(r["result"]["items"][0]["id"], json!("a"));
    assert!(r["text"].as_str().unwrap().starts_with("Top of heap 'w'"));
    forbidden(reader.post("/hql", hql("PUSH b INTO w WEIGHT 2")).await);
    forbidden(writer.post("/hql", hql("CREATE HEAP v HALF LIFE 1h")).await);
    forbidden(writer.post("/hql", hql("REBASE w")).await);
    let r = writer
        .post("/hql", hql(r#"PUSH b INTO w WEIGHT 2 ATTRS {"kind": "task"}"#))
        .await
        .ok();
    assert_eq!(r["kind"], json!("message"));
    let r = writer
        .post("/hql", hql("POP FROM w WHERE NOT kind = 'bug'"))
        .await
        .ok();
    assert_eq!(r["result"]["items"][0]["id"], json!("b"));
    let r = db.post("/hql", hql("CREATE HEAP v HALF LIFE 2h")).await.ok();
    assert_eq!(
        (r["kind"].clone(), r["result"]["half_life"].clone()),
        (json!("heap"), json!(7200.0))
    );
    assert_eq!(
        db.post("/hql", hql("SHOW HEAPS")).await.ok()["result"]["heaps"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    let r = db.post("/hql", hql("PEEK 3 FROM w WHERE kind LIKE 'x'")).await;
    assert_eq!(r.code(), "query_error");
    let r = db.post("/hql", hql("POP FROM nope")).await;
    assert_eq!(r.code(), "unknown_heap");

    // Quotas count heap requests like any other.
    let q = create_db_with(
        &n.url,
        json!({"max_requests_per_minute": 3}),
        &["read", "write", "admin"],
    )
    .await;
    q.put("/heaps/w", json!({"half_life": 60})).await.ok();
    q.post("/heaps/w/push", json!({"items": [{"id": "a"}]}))
        .await
        .ok();
    q.post("/heaps/w/top", json!({})).await.ok();
    let r = q.post("/heaps/w/top", json!({})).await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::TOO_MANY_REQUESTS, "rate_limited")
    );
    let s = create_db_with(
        &n.url,
        json!({"max_storage_bytes": 1}),
        &["read", "write", "admin"],
    )
    .await;
    s.put("/heaps/w", json!({"half_life": 60})).await; // the first write may fit
    let r = s.post("/heaps/w/push", json!({"items": [{"id": "a"}]})).await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::PAYLOAD_TOO_LARGE, "storage_quota_exceeded")
    );
    let info = q.get("").await.ok();
    assert!(info["usage"]["requests_this_period"].as_u64().unwrap() >= 3);
    n.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn heaps_survive_restarts() {
    let b = Backends::memory();
    // Snapshots after every 4 records, so both paths are exercised: the
    // first restart recovers from a snapshot + WAL tail.
    let snap = |c: &mut heapdb_server::config::Config| {
        c.snapshot_records = 4;
        fast_leases(c);
    };
    let n = node_on(&b, snap);
    let db = create_db(&n.url).await;
    db.put("/heaps/q%20one", json!({"half_life": 3600, "landmark": T0}))
        .await
        .ok();
    for i in 0..10 {
        db.post(
            "/heaps/q%20one/push",
            json!({"items": [{"id": format!("i{i}"), "weight": 1.0 + i as f64, "time": T0 + i as f64 * 60.0, "attrs": {"lane": i % 2}}]}),
        )
        .await
        .ok();
    }
    db.post("/heaps/q%20one/pop", json!({"now": T0 + 900.0}))
        .await
        .ok();
    db.post("/hql", json!({"query": "SET BASE i0 IN 'q one' TO 7", "now": T0}))
        .await
        .ok();
    let q = json!({"k": 20, "now": T0 + 1000.0, "filter": {"attr": "lane", "op": "eq", "value": 0}});
    let before = db.post("/heaps/q%20one/top", q.clone()).await.ok();
    assert_eq!(before["items"].as_array().unwrap().len(), 5);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let snaps = b.objects.list(&format!("db/{}/snap/", db.id)).await.unwrap();
    assert!(!snaps.is_empty(), "a snapshot was written");
    // Crash: no final snapshot, recovery = last snapshot + WAL replay.
    n.kill();
    let n = node_on(&b, snap);
    let db = db.on(&n.url);
    let after = db.post_retry("/heaps/q%20one/top", q.clone()).await.ok();
    assert_eq!(after, before, "WAL replay after a crash");
    // Graceful restart: recovery from the final snapshot.
    db.post(
        "/heaps/q%20one/push",
        json!({"items": [{"id": "late", "time": T0, "attrs": {"lane": 0}}]}),
    )
    .await
    .ok();
    let before = db.post("/heaps/q%20one/top", q.clone()).await.ok();
    n.shutdown();
    let n = node_on(&b, snap);
    let db = db.on(&n.url);
    let after = db.post_retry("/heaps/q%20one/top", q).await.ok();
    assert_eq!(after, before, "snapshot after a graceful restart");
    let snaps = b.objects.list(&format!("db/{}/snap/", db.id)).await.unwrap();
    let newest = b.objects.get(snaps.last().unwrap()).await.unwrap().unwrap();
    assert!(newest.starts_with(br#"{"kind":"heapdb_server","format":2"#));
    n.shutdown();
}
