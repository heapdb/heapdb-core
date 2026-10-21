//! Clustered mode: 2-3 nodes sharing storage.
//!
//! Every scenario runs on the in-memory fakes of S3 and DynamoDB (same
//! conditional semantics, always on). With the `cluster-aws` feature and
//! `HEAPDB_TEST_AWS_ENDPOINT` set (LocalStack, moto, or MinIO + DynamoDB
//! Local behind one endpoint), the same scenarios also run against the
//! emulator through the real AWS SDK backends:
//!
//! ```text
//! moto_server -p 5000 &
//! HEAPDB_TEST_AWS_ENDPOINT=http://127.0.0.1:5000 \
//!   cargo test -p heapdb-server --features cluster-aws --test cluster
//! ```

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::*;
use heapdb_server::config::{ClusterMode, Config};
use heapdb_server::storage::Backends;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};

#[derive(Clone)]
enum Env {
    Fake(Backends),
    #[cfg(feature = "cluster-aws")]
    Aws(Config),
}

impl Env {
    fn fake() -> Env {
        Env::Fake(Backends::memory())
    }

    #[cfg(feature = "cluster-aws")]
    async fn aws() -> Option<Env> {
        let endpoint = std::env::var("HEAPDB_TEST_AWS_ENDPOINT").ok()?;
        // Standard AWS configuration, pointed at the emulator.
        std::env::set_var("AWS_ENDPOINT_URL", &endpoint);
        std::env::set_var("HEAPDB_S3_FORCE_PATH_STYLE", "1");
        for (k, v) in [
            ("AWS_REGION", "us-east-1"),
            ("AWS_ACCESS_KEY_ID", "test"),
            ("AWS_SECRET_ACCESS_KEY", "test"),
        ] {
            if std::env::var(k).is_err() {
                std::env::set_var(k, v);
            }
        }
        let mut c = Config::single_node(std::env::temp_dir());
        c.cluster = ClusterMode::Aws;
        c.s3_bucket = "heapdb-test".into();
        c.s3_prefix = format!("t{}/", heapdb_server::ids::new_node_id());
        c.ddb_leases = "heapdb-test-leases".into();
        c.ddb_registry = "heapdb-test-registry".into();
        c.ddb_usage = "heapdb-test-usage".into();
        heapdb_server::storage::aws::ensure_resources(&c)
            .await
            .expect("emulator reachable");
        Some(Env::Aws(c))
    }

    fn config(&self, dir: &std::path::Path) -> Config {
        let mut c = test_config(dir);
        #[cfg(feature = "cluster-aws")]
        if let Env::Aws(a) = self {
            c.s3_bucket = a.s3_bucket.clone();
            c.s3_prefix = a.s3_prefix.clone();
            c.ddb_leases = a.ddb_leases.clone();
            c.ddb_registry = a.ddb_registry.clone();
            c.ddb_usage = a.ddb_usage.clone();
        }
        c.cluster = ClusterMode::Aws;
        fast_leases(&mut c);
        c
    }

    fn node_with(&self, tweak: impl FnOnce(&mut Config)) -> TestNode {
        let dir = std::env::temp_dir();
        let mut cfg = self.config(&dir);
        tweak(&mut cfg);
        match self {
            Env::Fake(b) => TestNode::start(cfg, b.clone()),
            #[cfg(feature = "cluster-aws")]
            Env::Aws(_) => {
                let c2 = cfg.clone();
                TestNode::start_with(
                    cfg,
                    Box::new(move || {
                        Box::pin(async move { heapdb_server::storage::aws::backends(&c2).await.unwrap() })
                    }),
                )
            }
        }
    }

    fn node(&self) -> TestNode {
        self.node_with(|_| {})
    }

    /// Backends for test-side inspection (built in the calling runtime).
    async fn backends(&self) -> Backends {
        match self {
            Env::Fake(b) => b.clone(),
            #[cfg(feature = "cluster-aws")]
            Env::Aws(c) => heapdb_server::storage::aws::backends(c).await.unwrap(),
        }
    }

    fn prefix(&self) -> String {
        match self {
            Env::Fake(_) => String::new(),
            #[cfg(feature = "cluster-aws")]
            Env::Aws(c) => c.s3_prefix.clone(),
        }
    }
}

fn owners<'a>(nodes: &'a [&'a TestNode], db: &str) -> Vec<&'a TestNode> {
    nodes.iter().copied().filter(|n| n.owns(db)).collect()
}

fn ev(entity: &str, t: f64, xid: &str) -> Value {
    json!({"entity": entity, "time": t, "type": "meal", "features": ["beans", "x"], "external_id": xid})
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

async fn forwarding(env: Env) {
    let (a, b, c) = (env.node(), env.node(), env.node());
    let db = create_db(&a.url).await;
    demo_schema(&db.on(&b.url)).await; // first touch on b: b acquires
    assert!(b.owns(&db.id));
    assert!(!a.owns(&db.id) && !c.owns(&db.id));
    // Writes and reads through every node land on the owner.
    for (i, n) in [&a, &b, &c].iter().enumerate() {
        db.on(&n.url)
            .post(
                "/events",
                json!({"events": [ev("me", i as f64 * 600.0, &format!("x{i}"))]}),
            )
            .await
            .ok();
    }
    for n in [&a, &b, &c] {
        let q = db
            .on(&n.url)
            .post(
                "/query/top-causes",
                json!({"entity": "me", "effect": "bloating", "at": 7200.0, "k": 5}),
            )
            .await
            .ok();
        assert_eq!(q["causes"].as_array().unwrap().len(), 3);
    }
    assert_eq!(owners(&[&a, &b, &c], &db.id).len(), 1);
    let fwd = |n: &TestNode| {
        n.manager
            .ctx
            .metrics
            .forwards
            .load(std::sync::atomic::Ordering::Relaxed)
    };
    assert!(fwd(&a) >= 2 && fwd(&c) >= 2 && fwd(&b) == 0);
    // A forwarded request is never forwarded again.
    let r = send(
        &http(),
        Method::GET,
        &format!("{}/v1/databases/{}", a.internal, db.id),
        Some(&db.key),
        None,
    )
    .await;
    let r2 = http()
        .get(format!("{}/v1/databases/{}", a.internal, db.id))
        .bearer_auth(&db.key)
        .header("x-heapdb-forwarded", "1")
        .send()
        .await
        .unwrap();
    assert!(r.status.is_success());
    assert_eq!(r2.status(), StatusCode::SERVICE_UNAVAILABLE);
    let v: Value = r2.json().await.unwrap();
    assert_eq!(v["error"]["code"], json!("not_owner"));
    // Revocation through any node is effective immediately on the owner.
    let k2 = add_key(&c.url, &db.id, "key_2", &["read"]).await;
    db.with_key(&k2).on(&a.url).get("").await.ok();
    admin(
        &http(),
        Method::DELETE,
        &format!("{}/admin/v1/databases/{}/keys/key_2", c.url, db.id),
        None,
    )
    .await
    .ok();
    assert_eq!(
        db.with_key(&k2).on(&b.url).get("").await.code(),
        "invalid_api_key"
    );
    // Suspension too.
    admin(
        &http(),
        Method::PATCH,
        &format!("{}/admin/v1/databases/{}", c.url, db.id),
        Some(json!({"suspended": true})),
    )
    .await
    .ok();
    assert_eq!(db.on(&b.url).get("").await.code(), "database_suspended");
    for n in [a, b, c] {
        n.shutdown();
    }
}

async fn failover_keeps_acked_writes(env: Env) {
    let (a, b) = (env.node(), env.node());
    let db = create_db(&a.url).await;
    demo_schema(&db.on(&a.url)).await;
    assert!(a.owns(&db.id));
    let acked: Arc<Mutex<Vec<(String, u64)>>> = Arc::default();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut tasks = Vec::new();
    for w in 0..4 {
        // Half the writers go through b (forwarded to a), half to a directly.
        let target = if w % 2 == 0 { b.url.clone() } else { a.url.clone() };
        let db = db.on(&target).clone();
        let fallback = db.on(&b.url);
        let (acked, stop) = (acked.clone(), stop.clone());
        tasks.push(tokio::spawn(async move {
            let mut i = 0;
            while !stop.load(std::sync::atomic::Ordering::SeqCst) {
                let xid = format!("w{w}-{i}");
                let body = json!({"events": [ev(&format!("e{w}"), i as f64 * 60.0, &xid)]});
                let r =
                    db.c.post(db.url("/events"))
                        .bearer_auth(&db.key)
                        .json(&body)
                        .send()
                        .await;
                let r = match r {
                    Ok(r) => r,
                    Err(_) => fallback
                        .c
                        .post(fallback.url("/events"))
                        .bearer_auth(&db.key)
                        .json(&body)
                        .send()
                        .await
                        .unwrap(),
                };
                if r.status().is_success() {
                    let v: Value = r.json().await.unwrap();
                    acked.lock().unwrap().push((xid, v["ids"][0].as_u64().unwrap()));
                } else {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                i += 1;
            }
        }));
    }
    tokio::time::sleep(Duration::from_millis(800)).await;
    let before = acked.lock().unwrap().len();
    a.kill(); // crash the owner mid-traffic
              // Writers keep going through b; b takes over once the lease expires.
    tokio::time::sleep(Duration::from_millis(3500)).await;
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    for t in tasks {
        t.await.unwrap();
    }
    let acked = acked.lock().unwrap().clone();
    assert!(before > 10, "writes before the crash: {before}");
    assert!(acked.len() > before, "writes resumed after failover");
    assert!(b.owns(&db.id));
    let dbb = db.on(&b.url);
    for chunk in acked.chunks(500) {
        let events: Vec<Value> = chunk.iter().map(|(x, _)| ev("replay", 0.0, x)).collect();
        let r = dbb.post_retry("/events", json!({"events": events})).await.ok();
        let got: Vec<u64> = r["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let want: Vec<u64> = chunk.iter().map(|(_, i)| *i).collect();
        assert_eq!(got, want, "an acknowledged write was lost in failover");
    }
    b.shutdown();
}

async fn fencing_rejects_stale_owner(env: Env) {
    // Long leases: the stale owner keeps believing it owns the database.
    let slow = |c: &mut Config| {
        c.lease_ttl = Duration::from_secs(20);
        c.lease_renew = Duration::from_secs(10);
    };
    let a = env.node_with(slow);
    let b = env.node_with(slow);
    let db = create_db(&a.url).await;
    demo_schema(&db.on(&a.url)).await;
    db.on(&a.url)
        .post("/events", json!({"events": [ev("me", 0.0, "a1")]}))
        .await
        .ok();
    let lease = a.manager.get_open(&db.id).unwrap().fence.lease();
    // Simulate a's lease expiring (e.g. a long pause): expire it in storage.
    let backends = env.backends().await;
    backends.leases.release(&lease).await.unwrap();
    // b takes over (epoch + 1) on the next request it receives.
    let r = db
        .on(&b.url)
        .post_retry("/events", json!({"events": [ev("me", 60.0, "b1")]}))
        .await
        .ok();
    let b1 = r["ids"][0].as_u64().unwrap();
    assert!(b.owns(&db.id));
    let now = backends.leases.get(&db.id).await.unwrap().unwrap();
    assert_eq!(now.epoch, lease.epoch + 1);
    assert!(a.owns(&db.id), "a has not noticed yet");
    // The stale owner's late write is not acknowledged ...
    let r = db
        .on(&a.url)
        .post("/events", json!({"events": [ev("me", 120.0, "stale")]}))
        .await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::SERVICE_UNAVAILABLE, "lease_lost"),
        "{}",
        r.body
    );
    assert!(r.retry_after().is_some());
    // ... even though its WAL object reached storage before the re-check.
    let wal = backends
        .objects
        .list(&format!("{}db/{}/wal/", env.prefix(), db.id))
        .await
        .unwrap();
    let stale_epoch = format!("/{:020}-", lease.epoch);
    assert!(wal.iter().any(|k| k.contains(&stale_epoch)), "{wal:?}");
    // b never saw it, and a fresh recovery (node c) ignores it too.
    let q = json!({"entity": "me", "effect": "bloating", "at": 3600.0, "k": 10});
    let on_b = db.on(&b.url).post("/query/top-causes", q.clone()).await.ok();
    assert_eq!(on_b["causes"].as_array().unwrap().len(), 2);
    b.shutdown(); // releases the lease
    let c = env.node();
    let on_c = db.on(&c.url).post_retry("/query/top-causes", q).await.ok();
    assert_eq!(on_c["causes"], on_b["causes"]);
    let r = db
        .on(&c.url)
        .post("/events", json!({"events": [ev("me", 120.0, "stale")]}))
        .await
        .ok();
    assert!(
        r["ids"][0].as_u64().unwrap() > b1,
        "the stale event was never stored"
    );
    assert_eq!(
        db.on(&c.url).get("").await.ok()["usage"]["events_stored"],
        json!(3)
    );
    // The stale node eventually fences itself.
    drop(a);
    c.shutdown();
}

async fn concurrent_lease_race(env: Env) {
    let nodes = [env.node(), env.node(), env.node()];
    let db = create_db(&nodes[0].url).await;
    let mut tasks = Vec::new();
    for i in 0..30 {
        let d = db.on(&nodes[i % 3].url);
        tasks.push(tokio::spawn(async move { d.get_retry("").await.status }));
    }
    for t in tasks {
        assert!(t.await.unwrap().is_success());
    }
    let refs: Vec<&TestNode> = nodes.iter().collect();
    assert_eq!(owners(&refs, &db.id).len(), 1, "exactly one owner");
    let acquired: u64 = nodes
        .iter()
        .map(|n| {
            n.manager
                .ctx
                .metrics
                .leases_acquired
                .load(std::sync::atomic::Ordering::Relaxed)
        })
        .sum();
    assert_eq!(acquired, 1, "one lease acquisition won the race");
    let lease = env.backends().await.leases.get(&db.id).await.unwrap().unwrap();
    assert_eq!(lease.epoch, 1);
    for n in nodes {
        n.shutdown();
    }
}

async fn usage_survives_failover(env: Env) {
    let (a, b) = (env.node(), env.node());
    let db = create_db_with(
        &a.url,
        json!({"max_requests_per_month": 5}),
        &["read", "write", "admin"],
    )
    .await;
    // First touch on a; the next requests go through b and are forwarded,
    // so each is counted once (by the owner).
    db.on(&a.url)
        .post(
            "/effects",
            json!({"name": "e", "kernel": {"type": "exp", "params": {"half_life": 60}}}),
        )
        .await
        .ok();
    for i in 0..4 {
        db.on(&b.url)
            .post(
                "/events",
                json!({"events": [ev("me", i as f64, &format!("u{i}"))]}),
            )
            .await
            .ok();
    }
    assert!(a.owns(&db.id));
    assert_eq!(
        db.on(&b.url).post("/settle", json!({"until": 0})).await.code(),
        "monthly_quota_exceeded"
    );
    tokio::time::sleep(Duration::from_millis(400)).await; // > usage flush interval
    a.kill();
    let r = db.on(&b.url).post_retry("/settle", json!({"until": 0})).await;
    assert_eq!(
        (r.status, r.code()),
        (StatusCode::TOO_MANY_REQUESTS, "monthly_quota_exceeded")
    );
    assert!(b.owns(&db.id));
    let info = db.on(&b.url).get("").await.ok();
    assert_eq!(info["usage"]["requests_this_period"], json!(5));
    assert_eq!(info["usage"]["events_stored"], json!(4));
    let u = admin(
        &http(),
        Method::GET,
        &format!("{}/admin/v1/usage?since=0", b.url),
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
    assert_eq!(row["requests_this_period"], json!(5));
    assert_eq!(row["requests"], json!(5));
    assert_eq!(row["events_written"], json!(4));
    b.shutdown();
}

async fn graceful_handoff(env: Env) {
    let (a, b) = (env.node(), env.node());
    let db = create_db(&a.url).await;
    demo_schema(&db.on(&a.url)).await;
    db.on(&a.url)
        .post("/events", json!({"events": [ev("me", 0.0, "g1")]}))
        .await
        .ok();
    a.shutdown(); // snapshot + lease release: b takes over at once
    let t = std::time::Instant::now();
    let r = db
        .on(&b.url)
        .post("/events", json!({"events": [ev("me", 60.0, "g2")]}))
        .await
        .ok();
    assert!(t.elapsed() < Duration::from_secs(1));
    assert_eq!(r["ids"], json!([2]));
    b.shutdown();
}

macro_rules! scenarios {
    ($($name:ident),* $(,)?) => {
        mod fake {
            use super::*;
            $(
                #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
                async fn $name() {
                    super::$name(Env::fake()).await;
                }
            )*
        }
        #[cfg(feature = "cluster-aws")]
        mod emulator {
            use super::*;
            $(
                #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
                async fn $name() {
                    match Env::aws().await {
                        Some(env) => super::$name(env).await,
                        None => eprintln!("skipped: set HEAPDB_TEST_AWS_ENDPOINT to run against an emulator"),
                    }
                }
            )*
        }
    };
}

scenarios!(
    forwarding,
    failover_keeps_acked_writes,
    fencing_rejects_stale_owner,
    concurrent_lease_race,
    usage_survives_failover,
    graceful_handoff,
);
