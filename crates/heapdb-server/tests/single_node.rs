//! Single-node Data API and Admin API flows over HTTP.

mod common;

use std::time::Duration;

use chrono::{TimeZone, Utc};
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

fn ids(v: &Value) -> Vec<u64> {
    v["causes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["event"]["id"].as_u64().unwrap())
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_api_flow() {
    let (n, _d) = node();
    let db = create_db(&n.url).await;
    demo_schema(&db).await;

    // Batched insert with an idempotency key; weights derived from attrs.
    let batch = json!({"events": [
        {"entity": "me", "time": 0.0, "type": "meal", "features": ["beans", "onion"], "attrs": {"portion": 1.5}, "external_id": "m1"},
        {"entity": "me", "time": H, "type": "meal", "features": ["dairy"], "attrs": {"portion": 0.25}, "external_id": "m2"},
        {"entity": "me", "time": 1.5 * H, "type": "snack", "features": ["chips"]},
        {"entity": "you", "time": 0.5 * H, "type": "meal", "features": ["beans"]},
        {"entity": "a/b c", "time": 0.0, "type": "meal", "features": ["x"]}
    ]});
    let r = db.post("/events", batch.clone()).await.ok();
    assert_eq!(r["ids"], json!([1, 2, 3, 4, 5]));
    assert!(r.get("events").is_none());
    // A retried request with external ids does not insert duplicates.
    let again = db
        .post(
            "/events",
            json!({"events": [batch["events"][0].clone(), batch["events"][1].clone()]}),
        )
        .await
        .ok();
    assert_eq!(again["ids"], json!([1, 2]));
    let r = db
        .post(
            "/events",
            json!({"events": [{"entity": "me", "time": 1.8 * H, "type": "meal", "features": ["beans"]}], "return_events": true}),
        )
        .await
        .ok();
    assert_eq!(r["events"][0]["weight"], json!(1.0));
    assert_eq!(r["events"][0]["type"], json!("meal"));

    // Top causes, explained.
    let q = db
        .post(
            "/query/top-causes",
            json!({"entity": "me", "effect": "bloating", "at": 3.0 * H, "k": 10, "explain": true}),
        )
        .await
        .ok();
    let c = &q["causes"];
    assert_eq!(c.as_array().unwrap().len(), 4);
    assert_eq!(c[0]["rank"], json!(1));
    let first = c
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["event"]["id"] == json!(1))
        .unwrap();
    assert_eq!(first["event"]["features"], json!(["beans", "onion"]));
    assert_eq!(first["event"]["weight"], json!(1.5));
    let ex = &c[0]["explanation"];
    for k in [
        "dt",
        "kernel_value",
        "weight",
        "link_strength",
        "features",
        "rules",
        "joint_causes",
        "warnings",
        "score",
    ] {
        assert!(ex.get(k).is_some(), "explanation lacks {k}: {ex}");
    }
    assert!((ex["score"].as_f64().unwrap() - c[0]["score"].as_f64().unwrap()).abs() < 1e-12);
    for k in [
        "segments_in_window",
        "segments_scanned",
        "segments_pruned",
        "events_scanned",
    ] {
        assert!(q["stats"][k].is_u64());
    }
    // The tiny-portion rule shows up in the dairy meal's explanation.
    let dairy = c
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["event"]["id"] == json!(2))
        .unwrap();
    assert_eq!(dairy["explanation"]["rules"][0]["name"], json!("tiny"));

    // Group query.
    let g = db
        .post(
            "/query/top-causes",
            json!({"entities": ["me", "you"], "effect": "bloating", "at": 3.0 * H, "k": 5}),
        )
        .await
        .ok();
    assert_eq!(g["entities"], json!(["me", "you"]));
    assert!(ids(&g).contains(&4));
    assert_eq!(
        db.post("/query/top-causes", json!({"effect": "bloating", "at": 1.0}))
            .await
            .code(),
        "bad_request"
    );
    let e = db
        .post(
            "/query/top-causes",
            json!({"entity": "me", "effect": "nope", "at": 1.0}),
        )
        .await;
    assert_eq!((e.status, e.code()), (StatusCode::NOT_FOUND, "unknown_effect"));

    // Risk: single and candidates.
    let r = db
        .post(
            "/query/risk",
            json!({"effect": "bloating", "features": ["beans"], "entity": "me", "explain": true}),
        )
        .await
        .ok();
    assert!(r["score"].as_f64().unwrap() > 0.0);
    assert_eq!(r["explanation"]["prospective"], json!(true));
    let r = db
        .post(
            "/query/risk",
            json!({"effect": "bloating", "k": 2, "entity": "me", "candidates": [
                {"id": "lunch-a", "features": ["dairy"]},
                {"id": "lunch-b", "features": ["beans", "onion"], "weight": 2.0},
                {"id": 7, "features": ["rice"]}
            ]}),
        )
        .await
        .ok();
    let cs = r["candidates"].as_array().unwrap();
    assert_eq!(cs.len(), 2);
    assert_eq!(cs[0]["candidate"]["id"], json!("lunch-b"));
    assert_eq!(cs[0]["rank"], json!(1));

    // Learning.
    let f = db
        .post(
            "/feedback",
            json!({"entity": "me", "effect": "bloating", "at": 3.0 * H, "confirmed_event_id": 1, "rejected_ids": [2]}),
        )
        .await
        .ok();
    assert_eq!(f["confirmed"], json!(1));
    assert_eq!(f["rejected"], json!([2]));
    assert!(f["credit"]["beans"].as_f64().unwrap() > 0.0);
    let o = db
        .post(
            "/observe",
            json!({"entity": "you", "effect": "bloating", "at": 2.5 * H}),
        )
        .await
        .ok();
    assert_eq!(o["credits"][0]["event_id"], json!(4));
    let s = db.post("/settle", json!({"until": 100.0 * H})).await.ok();
    for k in [
        "until",
        "events_settled",
        "events_rejected",
        "rejected_weight",
        "skipped_credited",
        "skipped_observed",
        "per_effect",
    ] {
        assert!(s.get(k).is_some(), "settle lacks {k}: {s}");
    }
    assert!(s.get("settled").is_none());

    // Reports.
    let l = db.get("/links?effect=bloating&min_support=0").await.ok();
    assert!(l["rows"]
        .as_array()
        .unwrap()
        .iter()
        .any(|r| r["feature"] == json!("beans")));
    let l = db
        .get("/links?effect=bloating&entity=me&features=beans,dairy&min_support=0")
        .await
        .ok();
    assert_eq!(l["entity"], json!("me"));
    let cal = db.get("/calibration?effect=bloating&bins=5").await.ok();
    assert_eq!(cal["n"], json!(2));
    assert_eq!(cal["bins"].as_array().unwrap().len(), 5);
    assert_eq!(db.get("/links").await.code(), "bad_request");

    // Query language.
    let h = db
        .post(
            "/hql",
            json!({"query": "TOP 2 CAUSES OF bloating FOR me AT 10800 EXPLAIN"}),
        )
        .await
        .ok();
    assert_eq!(h["kind"], json!("top_causes"));
    assert!(h["text"]
        .as_str()
        .unwrap()
        .starts_with("Top causes of 'bloating' for me"));
    let h = db
        .post(
            "/hql",
            json!({"query": "INSERT meal FOR me AT now-1h FEATURES rice", "now": 20.0 * H}),
        )
        .await
        .ok();
    assert_eq!(h["kind"], json!("message"));
    assert!(h["result"].as_str().unwrap().contains("#7 meal@68400"));
    assert_eq!(
        db.post("/hql", json!({"query": "SELECT 1"})).await.code(),
        "query_error"
    );

    // Forget a percent-encoded entity.
    let fg = db.delete("/entities/a%2Fb%20c?scrub_global=true").await.ok();
    assert_eq!(fg["counts"]["events"], json!(1));

    // Info.
    let info = db.get("").await.ok();
    assert_eq!(info["database_id"], json!(db.id));
    assert_eq!(info["usage"]["events_stored"], json!(6));
    assert!(info["limits"]["max_events"].is_null());
    let info2 = db.get("/").await.ok();
    assert_eq!(info2["database_id"], json!(db.id));

    // Declarative only: code is rejected.
    let bad = db
        .post(
            "/event-types",
            json!({"type": "x", "weight": {"kind": "python", "code": "lambda a: 1"}}),
        )
        .await;
    assert_eq!(bad.status, StatusCode::BAD_REQUEST);
    let bad = db
        .post(
            "/rules",
            json!({"effect": "bloating", "name": "r", "multiplier": 2.0}),
        )
        .await;
    assert_eq!(bad.code(), "invalid_argument");
    n.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_scopes_suspend_revoke() {
    let (n, _d) = node();
    let db = create_db(&n.url).await;
    demo_schema(&db).await;
    let c = http();

    let r = send(&c, Method::GET, &db.url(""), None, None).await;
    assert_eq!((r.status, r.code()), (StatusCode::UNAUTHORIZED, "unauthorized"));
    let r = db.with_key("hdb_live_nope").get("").await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::UNAUTHORIZED, "invalid_api_key")
    );
    let other = create_db(&n.url).await;
    let r = db.with_key(&other.key).get("").await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::FORBIDDEN, "key_not_for_database")
    );

    let ro = db.with_key(&add_key(&n.url, &db.id, "key_ro", &["read"]).await);
    ro.get("").await.ok();
    ro.post(
        "/query/top-causes",
        json!({"entity": "me", "effect": "bloating", "at": 1.0}),
    )
    .await
    .ok();
    let r = ro
        .post(
            "/events",
            json!({"events": [{"entity": "me", "time": 0, "type": "meal"}]}),
        )
        .await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::FORBIDDEN, "insufficient_scope")
    );
    assert_eq!(
        ro.post("/hql", json!({"query": "INSERT meal FOR me AT 0"}))
            .await
            .code(),
        "insufficient_scope"
    );
    ro.post("/hql", json!({"query": "TOP 1 CAUSES OF bloating FOR me AT 0"}))
        .await
        .ok();

    let rw = db.with_key(&add_key(&n.url, &db.id, "key_rw", &["read", "write"]).await);
    rw.post(
        "/events",
        json!({"events": [{"entity": "me", "time": 0, "type": "meal"}]}),
    )
    .await
    .ok();
    for (path, body) in [
        (
            "/effects",
            json!({"name": "x", "kernel": {"type": "exp", "params": {"half_life": 60}}}),
        ),
        ("/event-types", json!({"type": "deploy"})),
        (
            "/rules",
            json!({"effect": "bloating", "name": "r", "multiplier": 0.5}),
        ),
    ] {
        assert_eq!(rw.post(path, body).await.code(), "insufficient_scope", "{path}");
    }
    assert_eq!(rw.delete("/entities/me").await.code(), "insufficient_scope");

    // Expired key.
    let k = heapdb_server::ids::new_api_key();
    admin(
        &c,
        Method::PUT,
        &format!("{}/admin/v1/databases/{}/keys/key_old", n.url, db.id),
        Some(json!({"sha256": heapdb_server::ids::sha256_hex(k.as_bytes()), "scopes": ["read"], "expires_at": 1.0})),
    )
    .await
    .ok();
    assert_eq!(db.with_key(&k).get("").await.code(), "key_expired");

    // Revocation is immediate (cache invalidated).
    rw.get("").await.ok();
    let r = admin(
        &c,
        Method::DELETE,
        &format!("{}/admin/v1/databases/{}/keys/key_rw", n.url, db.id),
        None,
    )
    .await;
    assert_eq!(r.ok()["revoked"], json!(true));
    assert_eq!(rw.get("").await.code(), "invalid_api_key");

    // Suspension.
    let url = format!("{}/admin/v1/databases/{}", n.url, db.id);
    admin(&c, Method::PATCH, &url, Some(json!({"suspended": true})))
        .await
        .ok();
    let r = db.get("").await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::FORBIDDEN, "database_suspended")
    );
    admin(&c, Method::PATCH, &url, Some(json!({"suspended": false})))
        .await
        .ok();
    db.get("").await.ok();

    // Admin auth.
    let r = send(
        &c,
        Method::GET,
        &format!("{}/admin/v1/health", n.url),
        Some("wrong"),
        None,
    )
    .await;
    assert_eq!((r.status, r.code()), (StatusCode::UNAUTHORIZED, "unauthorized"));
    let r = send(&c, Method::GET, &format!("{}/admin/v1/health", n.url), None, None).await;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED);
    admin(&c, Method::GET, &format!("{}/admin/v1/health", n.url), None)
        .await
        .ok();
    n.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_disabled_without_token() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(dir.path());
    cfg.admin_token = None;
    let b = local_backends(&cfg).unwrap();
    let n = TestNode::start(cfg, b);
    let r = admin(&http(), Method::GET, &format!("{}/admin/v1/health", n.url), None).await;
    assert_eq!((r.status, r.code()), (StatusCode::FORBIDDEN, "admin_disabled"));
    n.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn storage_event_and_rate_quotas() {
    let (n, _d) = node();
    let db = create_db_with(&n.url, json!({"max_events": 3}), &["read", "write", "admin"]).await;
    demo_schema(&db).await;
    let ev = |t: f64| json!({"entity": "me", "time": t, "type": "meal", "features": ["beans"]});
    db.post("/events", json!({"events": [ev(0.0), ev(1.0)]}))
        .await
        .ok();
    let r = db.post("/events", json!({"events": [ev(2.0), ev(3.0)]})).await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::PAYLOAD_TOO_LARGE, "events_quota_exceeded")
    );
    db.post("/events", json!({"events": [ev(2.0)]})).await.ok();
    let big: Vec<Value> = (0..1001).map(|i| ev(i as f64)).collect();
    let r = db.post("/events", json!({"events": big})).await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::PAYLOAD_TOO_LARGE, "batch_too_large")
    );

    let db2 = create_db_with(
        &n.url,
        json!({"max_storage_bytes": 1}),
        &["read", "write", "admin"],
    )
    .await;
    db2.post(
        "/effects",
        json!({"name": "e", "kernel": {"type": "exp", "params": {"half_life": 60}}}),
    )
    .await
    .ok();
    let r = db2.post("/events", json!({"events": [ev(0.0)]})).await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::PAYLOAD_TOO_LARGE, "storage_quota_exceeded")
    );

    let db3 = create_db_with(&n.url, json!({"max_requests_per_minute": 3}), &["read"]).await;
    for _ in 0..3 {
        db3.get("/calibration?effect=x").await; // 404 unknown_effect, but admitted
    }
    let r = db3.get("/calibration?effect=x").await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::TOO_MANY_REQUESTS, "rate_limited")
    );
    let ra = r.retry_after().expect("Retry-After");
    assert!((1..=20).contains(&ra), "{ra}");
    // Info is not rate limited.
    db3.get("").await.ok();
    n.shutdown();
}

fn month_before(ts: i64) -> i64 {
    Utc.timestamp_opt(ts, 0)
        .unwrap()
        .checked_sub_months(chrono::Months::new(1))
        .unwrap()
        .timestamp()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn monthly_quota_rollover_and_durability() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = test_config(dir.path());
    let n = TestNode::start(cfg.clone(), local_backends(&cfg).unwrap());
    let db = create_db_with(
        &n.url,
        json!({"max_requests_per_month": 3}),
        &["read", "write", "admin"],
    )
    .await;
    let c = http();
    for _ in 0..3 {
        db.post("/settle", json!({"until": 0})).await.ok();
    }
    let r = db.post("/settle", json!({"until": 0})).await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::TOO_MANY_REQUESTS, "monthly_quota_exceeded")
    );
    let now = Utc::now().timestamp();
    let (_, end) = heapdb_server::quota::billing_period(None, now);
    let ra = r.retry_after().unwrap() as i64;
    assert!((ra - (end - now)).abs() <= 2, "Retry-After {ra} vs {}", end - now);
    // Info keeps working and shows the count.
    let info = db.get("").await.ok();
    assert_eq!(info["usage"]["requests_this_period"], json!(3));
    assert_eq!(info["limits"]["max_requests_per_month"], json!(3));
    tokio::time::sleep(Duration::from_millis(300)).await;
    let u = admin(
        &c,
        Method::GET,
        &format!("{}/admin/v1/usage?since=0", n.url),
        None,
    )
    .await
    .ok();
    let row = u["databases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["database_id"] == json!(db.id))
        .unwrap()
        .clone();
    assert_eq!(row["requests_this_period"], json!(3));
    assert_eq!(row["requests"], json!(3));

    // Durable: a restarted node still refuses.
    n.shutdown();
    let n = TestNode::start(cfg.clone(), local_backends(&cfg).unwrap());
    let db = db.on(&n.url);
    assert_eq!(
        db.post("/settle", json!({"until": 0})).await.code(),
        "monthly_quota_exceeded"
    );

    // Roll the period over: anchor the billing month so the current period
    // ends in ~3 s (a new anchor starts a new period), exhaust it, cross
    // the boundary.
    let end_soon = Utc::now().timestamp() + 3;
    admin(
        &c,
        Method::PATCH,
        &format!("{}/admin/v1/databases/{}", n.url, db.id),
        Some(json!({"limits": {"billing_period_start": month_before(end_soon)}})),
    )
    .await
    .ok();
    for _ in 0..3 {
        db.post("/settle", json!({"until": 0})).await.ok();
    }
    let r = db.post("/settle", json!({"until": 0})).await;
    assert_eq!(r.code(), "monthly_quota_exceeded", "{}", r.body);
    assert!(r.retry_after().unwrap() <= 4);
    let wait = (end_soon * 1000 + 200) - Utc::now().timestamp_millis();
    tokio::time::sleep(Duration::from_millis(wait.max(0) as u64)).await;
    db.post("/settle", json!({"until": 0})).await.ok();
    let info = db.get("").await.ok();
    assert_eq!(info["usage"]["requests_this_period"], json!(1));
    assert_eq!(info["usage"]["period_start"], json!(end_soon));
    // Unlimited (Pro sends null).
    admin(
        &c,
        Method::PATCH,
        &format!("{}/admin/v1/databases/{}", n.url, db.id),
        Some(json!({"limits": {"max_requests_per_month": null}})),
    )
    .await
    .ok();
    for _ in 0..5 {
        db.post("/settle", json!({"until": 0})).await.ok();
    }
    n.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn admin_api() {
    let (n, _d) = node();
    let c = http();
    let base = format!("{}/admin/v1/databases", n.url);
    let id = heapdb_server::ids::new_database_id();
    let body = json!({"database_id": id, "org_id": "org_a", "plan": "free", "limits": {"max_events": 10, "max_requests_per_month": 100000}});
    let r = admin(&c, Method::POST, &base, Some(body.clone())).await;
    assert_eq!(r.status, StatusCode::CREATED);
    let r = admin(&c, Method::POST, &base, Some(body)).await;
    assert_eq!(r.status, StatusCode::OK, "idempotent create");
    let r = admin(
        &c,
        Method::POST,
        &base,
        Some(json!({"database_id": id, "org_id": "org_b", "plan": "free"})),
    )
    .await;
    assert_eq!((r.status, r.code()), (StatusCode::CONFLICT, "database_exists"));
    let r = admin(
        &c,
        Method::POST,
        &base,
        Some(json!({"database_id": "db_bad", "org_id": "org_a"})),
    )
    .await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    let r = admin(
        &c,
        Method::PATCH,
        &format!("{base}/{id}"),
        Some(json!({"plan": "pro", "limits": {"max_events": null, "max_storage_bytes": 5000}})),
    )
    .await
    .ok();
    assert_eq!(r["plan"], json!("pro"));
    assert_eq!(
        r["limits"],
        json!({"max_storage_bytes": 5000, "max_requests_per_month": 100000})
    );
    let key = add_key(&n.url, &id, "key_1", &["read", "write", "admin"]).await;
    let db = Db {
        c: c.clone(),
        base: n.url.clone(),
        id: id.clone(),
        key,
    };
    db.post(
        "/effects",
        json!({"name": "e", "kernel": {"type": "exp", "params": {"half_life": 60}}}),
    )
    .await
    .ok();
    db.post(
        "/events",
        json!({"events": [{"entity": "a", "time": 0, "type": "t"}]}),
    )
    .await
    .ok();
    let h = admin(&c, Method::GET, &format!("{}/admin/v1/health", n.url), None)
        .await
        .ok();
    assert_eq!(h["status"], json!("ok"));
    assert_eq!(h["cluster"], json!("off"));
    tokio::time::sleep(Duration::from_millis(300)).await;
    let u = admin(
        &c,
        Method::GET,
        &format!("{}/admin/v1/usage?since=0", n.url),
        None,
    )
    .await
    .ok();
    let row = u["databases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["database_id"] == json!(id))
        .unwrap()
        .clone();
    assert_eq!(row["events_written"], json!(1));
    assert_eq!(row["events_stored"], json!(1));
    assert!(row["storage_bytes"].as_u64().unwrap() > 0);
    let future = Utc::now().timestamp() + 7200;
    let u = admin(
        &c,
        Method::GET,
        &format!("{}/admin/v1/usage?since={future}", n.url),
        None,
    )
    .await
    .ok();
    let row = u["databases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["database_id"] == json!(id))
        .unwrap()
        .clone();
    assert_eq!(
        row["requests"],
        json!(0),
        "hourly buckets before `since` are excluded"
    );
    // Delete removes everything.
    let r = admin(&c, Method::DELETE, &format!("{base}/{id}"), None)
        .await
        .ok();
    assert_eq!(r["deleted"], json!(true));
    assert_eq!(db.get("").await.code(), "invalid_api_key");
    let r = admin(&c, Method::DELETE, &format!("{base}/{id}"), None)
        .await
        .ok();
    assert_eq!(r["deleted"], json!(false));
    assert!(n.manager.get_open(&id).is_none());
    let keys = n
        .manager
        .ctx
        .backends
        .objects
        .list(&format!("db/{id}/"))
        .await
        .unwrap();
    assert!(keys.is_empty(), "{keys:?}");
    n.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_conventions() {
    let (n, _d) = node();
    let db = create_db(&n.url).await;
    let c = http();
    let doc = send(&c, Method::GET, &format!("{}/v1/openapi.json", n.url), None, None)
        .await
        .ok();
    assert_eq!(doc["openapi"], json!("3.1.0"));
    assert!(doc["paths"]["/v1/databases/{database_id}/query/top-causes"]["post"].is_object());
    // x-request-id is echoed or generated.
    let r = c
        .get(db.url(""))
        .bearer_auth(&db.key)
        .header("x-request-id", "abc-123")
        .send()
        .await
        .unwrap();
    assert_eq!(r.headers()["x-request-id"], "abc-123");
    let r = c.get(format!("{}/healthz", n.url)).send().await.unwrap();
    assert!(r.headers()["x-request-id"].to_str().unwrap().starts_with("req_"));
    // CORS preflight.
    let r = c
        .request(Method::OPTIONS, db.url("/events"))
        .header("origin", "https://app.example")
        .header("access-control-request-method", "POST")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::NO_CONTENT);
    assert_eq!(r.headers()["access-control-allow-origin"], "*");
    assert!(r.headers()["access-control-allow-headers"]
        .to_str()
        .unwrap()
        .contains("Authorization"));
    let r = db.get("").await;
    assert_eq!(
        r.headers["access-control-expose-headers"],
        "x-request-id, retry-after"
    );
    // JSON errors everywhere.
    let r = send(&c, Method::GET, &format!("{}/nope", n.url), None, None).await;
    assert_eq!((r.status, r.code()), (StatusCode::NOT_FOUND, "not_found"));
    let r = db.get("/events").await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::METHOD_NOT_ALLOWED, "method_not_allowed")
    );
    let r = send(
        &c,
        Method::GET,
        &format!("{}/v1/databases/db_nope", n.url),
        Some(&db.key),
        None,
    )
    .await;
    assert_eq!(r.code(), "database_not_found");
    let huge = "x".repeat(9 << 20);
    // Its own client: the server answers 413 without reading the whole body and closes the
    // connection, so a pooled client could reuse that dead connection for the next request
    // (intermittent "BrokenPipe" on /readyz in CI).
    let r = http()
        .post(db.url("/events"))
        .bearer_auth(&db.key)
        .body(huge)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let v: Value = r.json().await.unwrap();
    assert_eq!(v["error"]["code"], json!("payload_too_large"));
    // Health.
    send(&c, Method::GET, &format!("{}/readyz", n.url), None, None)
        .await
        .ok();
    let m = c
        .get(format!("{}/metrics", n.internal))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("heapdb_http_responses_total"), "{m}");
    n.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn graceful_restart_and_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = test_config(dir.path());
    cfg.snapshot_records = 5;
    let n = TestNode::start(cfg.clone(), local_backends(&cfg).unwrap());
    let db = create_db(&n.url).await;
    demo_schema(&db).await;
    for i in 0..12 {
        db.post(
            "/events",
            json!({"events": [{"entity": "me", "time": i as f64 * 600.0, "type": "meal", "features": [format!("f{i}"), "beans"], "attrs": {"portion": 0.3 + i as f64 / 10.0}}]}),
        )
        .await
        .ok();
    }
    let q = json!({"entity": "me", "effect": "bloating", "at": 3.0 * H, "k": 5, "explain": true});
    let before = db.post("/query/top-causes", q.clone()).await.ok();
    tokio::time::sleep(Duration::from_millis(1200)).await;
    let objs = n
        .manager
        .ctx
        .backends
        .objects
        .list(&format!("db/{}/snap/", db.id))
        .await
        .unwrap();
    assert!(!objs.is_empty(), "a snapshot was written after 5 records");
    n.shutdown();
    let n = TestNode::start(cfg.clone(), local_backends(&cfg).unwrap());
    let db = db.on(&n.url);
    let after = db.post("/query/top-causes", q).await.ok();
    assert_eq!(
        before["causes"], after["causes"],
        "rules and weight functions survive restarts"
    );
    let wal = n
        .manager
        .ctx
        .backends
        .objects
        .list(&format!("db/{}/wal/", db.id))
        .await
        .unwrap();
    assert!(wal.len() <= 2, "covered WAL objects were deleted: {wal:?}");
    n.shutdown();
}

/// `combine: "logistic"` over the Data API: registered from JSON, explained
/// as base odds x multipliers, reported with weights, and recovered
/// bit-identically after a restart (WAL replay).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn logistic_effect_over_http_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = test_config(dir.path());
    let n = TestNode::start(cfg.clone(), local_backends(&cfg).unwrap());
    let db = create_db(&n.url).await;
    let bad = db
        .post(
            "/effects",
            json!({"name": "x", "kernel": {"type": "window", "params": {"min_lag": 0, "max_lag": 7200}}, "combine": "nope"}),
        )
        .await;
    assert_eq!(bad.code(), "invalid_argument", "{}", bad.body);
    let bad = db
        .post(
            "/effects",
            json!({"name": "x", "kernel": {"type": "window", "params": {"min_lag": 0, "max_lag": 7200}}, "combine": "logistic", "learning_rate": 0}),
        )
        .await;
    assert_eq!(bad.code(), "invalid_argument", "{}", bad.body);
    db.post_retry(
        "/effects",
        json!({
            "name": "incident",
            "kernel": {"type": "window", "params": {"min_lag": 0, "max_lag": 6.0 * H}},
            "priors": {"migration": [3, 1]},
            "combine": "logistic", "learning_rate": 0.5, "l2": 1.0
        }),
    )
    .await
    .ok();
    db.post(
        "/events",
        json!({"events": [
            {"entity": "svc", "time": 0.0, "type": "deploy", "features": ["migration", "team:a"]},
            {"entity": "svc", "time": 600.0, "type": "deploy", "features": ["config", "team:a"]},
            {"entity": "svc", "time": 1200.0, "type": "deploy", "features": ["config", "team:b"]}
        ]}),
    )
    .await
    .ok();
    let fb = db
        .post(
            "/feedback",
            json!({"entity": "svc", "effect": "incident", "at": H, "confirmed_event_id": 1, "rejected_ids": [2, 3]}),
        )
        .await
        .ok();
    // Uniform (count) credit is the logistic default.
    assert_eq!(fb["credit"]["migration"], json!(1.0));
    let q = json!({"entity": "svc", "effect": "incident", "at": H, "k": 3, "explain": true});
    let r = db.post("/query/top-causes", q.clone()).await.ok();
    let ex = &r["causes"][0]["explanation"];
    assert_eq!(ex["combine"], json!("logistic"));
    for k in ["base_log_odds", "base_odds", "log_odds", "odds"] {
        assert!(ex[k].is_number(), "{k} missing: {ex}");
    }
    let z = ex["base_log_odds"].as_f64().unwrap()
        + ex["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["log_odds"].as_f64().unwrap())
            .sum::<f64>();
    assert!((z - ex["log_odds"].as_f64().unwrap()).abs() < 1e-12);
    let link = ex["link_strength"].as_f64().unwrap();
    assert!((link - 1.0 / (1.0 + (-z).exp())).abs() < 1e-12);
    for c in ex["features"].as_array().unwrap() {
        let w = c["log_odds"].as_f64().unwrap();
        assert!((c["odds_multiplier"].as_f64().unwrap() - w.exp()).abs() < 1e-12 * w.exp().max(1.0));
    }
    assert!(ex["text"].as_str().unwrap().contains("logistic: base odds"));
    assert_eq!(r["causes"][0]["event"]["id"], json!(1));
    let l = db.get("/links?effect=incident&min_support=0").await.ok();
    assert!(l["rows"]
        .as_array()
        .unwrap()
        .iter()
        .all(|x| x["log_odds"].is_number()));

    n.shutdown();
    let n = TestNode::start(cfg.clone(), local_backends(&cfg).unwrap());
    let db = db.on(&n.url);
    let r2 = db.post("/query/top-causes", q).await.ok();
    assert_eq!(r2["causes"], r["causes"]);
    n.shutdown();
}
