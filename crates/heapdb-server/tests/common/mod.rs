//! Test harness: in-process nodes (each on its own runtime thread so that
//! `kill` is a real crash: tasks dropped mid-flight, nothing flushed, no
//! lease released) and a small HTTP client.

#![allow(dead_code)]

use std::path::Path;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use heapdb_server::config::Config;
use heapdb_server::ids::{new_api_key, new_database_id, sha256_hex};
use heapdb_server::manager::Manager;
use heapdb_server::server::Server;
use heapdb_server::storage::Backends;
use reqwest::header::HeaderMap;
use reqwest::StatusCode;
use serde_json::{json, Value};
use tokio::sync::oneshot;

pub const ADMIN: &str = "test-admin-token";

enum Stop {
    Graceful,
    Kill,
}

/// A node running on its own runtime.
pub struct TestNode {
    pub url: String,
    pub internal: String,
    pub manager: Arc<Manager>,
    stop: Option<oneshot::Sender<Stop>>,
    thread: Option<JoinHandle<()>>,
}

/// Config for tests: ephemeral ports, admin token, short timings.
pub fn test_config(dir: &Path) -> Config {
    let mut c = Config::single_node(dir);
    c.listen = "127.0.0.1:0".parse().unwrap();
    c.internal_listen = "127.0.0.1:0".parse().unwrap();
    c.admin_token = Some(ADMIN.into());
    c.usage_flush = Duration::from_millis(100);
    c.fsync = false;
    c
}

/// Cluster-ish timings (leases expire fast so failover tests are quick).
pub fn fast_leases(c: &mut Config) {
    c.lease_ttl = Duration::from_millis(1500);
    c.lease_renew = Duration::from_millis(300);
}

/// Builds a node's backends inside the node's runtime.
pub type BackendsFactory =
    Box<dyn FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = Backends> + Send>> + Send>;

impl TestNode {
    pub fn start(cfg: Config, backends: Backends) -> TestNode {
        Self::start_with(cfg, Box::new(move || Box::pin(async move { backends })))
    }

    pub fn start_with(cfg: Config, factory: BackendsFactory) -> TestNode {
        let (tx, rx) = std::sync::mpsc::channel();
        let (stop_tx, stop_rx) = oneshot::channel::<Stop>();
        let thread = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            let kill = rt.block_on(async move {
                let backends = factory().await;
                let server = Server::start(cfg, backends).await.expect("server starts");
                tx.send((server.public_addr, server.internal_addr, server.manager.clone()))
                    .unwrap();
                match stop_rx.await {
                    Ok(Stop::Graceful) | Err(_) => {
                        server.shutdown().await;
                        false
                    }
                    Ok(Stop::Kill) => {
                        server.abort();
                        true
                    }
                }
            });
            if kill {
                rt.shutdown_background();
            }
        });
        let (p, i, manager) = rx.recv().expect("node started");
        TestNode {
            url: format!("http://{p}"),
            internal: format!("http://{i}"),
            manager,
            stop: Some(stop_tx),
            thread: Some(thread),
        }
    }

    pub fn node_id(&self) -> String {
        self.manager.ctx.node_id.clone()
    }

    /// Databases open on this node.
    pub fn owns(&self, db: &str) -> bool {
        self.manager.get_open(db).is_some_and(|d| d.is_open())
    }

    fn stop(&mut self, how: Stop) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(how);
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }

    /// Graceful shutdown (flush, snapshot, release).
    pub fn shutdown(mut self) {
        self.stop(Stop::Graceful);
    }

    /// Crash: drop every task at once.
    pub fn kill(mut self) {
        self.stop(Stop::Kill);
    }
}

impl Drop for TestNode {
    fn drop(&mut self) {
        self.stop(Stop::Kill);
    }
}

/// Response of a test request.
pub struct Resp {
    pub status: StatusCode,
    pub body: Value,
    pub headers: HeaderMap,
}

impl Resp {
    pub fn code(&self) -> &str {
        self.body["error"]["code"].as_str().unwrap_or("")
    }
    pub fn retry_after(&self) -> Option<u64> {
        self.headers
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
    }
    #[track_caller]
    pub fn ok(self) -> Value {
        assert!(
            self.status.is_success(),
            "expected success, got {} {}",
            self.status,
            self.body
        );
        self.body
    }
}

pub fn http() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap()
}

pub async fn send(
    c: &reqwest::Client,
    method: reqwest::Method,
    url: &str,
    token: Option<&str>,
    body: Option<Value>,
) -> Resp {
    let mut r = c.request(method, url);
    if let Some(t) = token {
        r = r.bearer_auth(t);
    }
    if let Some(b) = body {
        r = r.json(&b);
    }
    let resp = r.send().await.expect("request sent");
    let status = resp.status();
    let headers = resp.headers().clone();
    let text = resp.text().await.unwrap_or_default();
    let body = serde_json::from_str(&text).unwrap_or(Value::String(text));
    Resp {
        status,
        body,
        headers,
    }
}

/// A database client bound to one node URL.
#[derive(Clone)]
pub struct Db {
    pub c: reqwest::Client,
    pub base: String,
    pub id: String,
    pub key: String,
}

impl Db {
    pub fn on(&self, node_url: &str) -> Db {
        Db {
            base: node_url.to_string(),
            ..self.clone()
        }
    }
    pub fn with_key(&self, key: &str) -> Db {
        Db {
            key: key.to_string(),
            ..self.clone()
        }
    }
    pub fn url(&self, path: &str) -> String {
        format!("{}/v1/databases/{}{}", self.base, self.id, path)
    }
    pub async fn post(&self, path: &str, body: Value) -> Resp {
        send(
            &self.c,
            reqwest::Method::POST,
            &self.url(path),
            Some(&self.key),
            Some(body),
        )
        .await
    }
    pub async fn get(&self, path: &str) -> Resp {
        send(
            &self.c,
            reqwest::Method::GET,
            &self.url(path),
            Some(&self.key),
            None,
        )
        .await
    }
    pub async fn put(&self, path: &str, body: Value) -> Resp {
        send(
            &self.c,
            reqwest::Method::PUT,
            &self.url(path),
            Some(&self.key),
            Some(body),
        )
        .await
    }
    pub async fn delete(&self, path: &str) -> Resp {
        send(
            &self.c,
            reqwest::Method::DELETE,
            &self.url(path),
            Some(&self.key),
            None,
        )
        .await
    }
    /// POST, retrying retryable errors (503, rate limits) for up to 20 s.
    pub async fn post_retry(&self, path: &str, body: Value) -> Resp {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            let r = self.post(path, body.clone()).await;
            if r.status == StatusCode::SERVICE_UNAVAILABLE && std::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
            return r;
        }
    }
    pub async fn get_retry(&self, path: &str) -> Resp {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            let r = self.get(path).await;
            if r.status == StatusCode::SERVICE_UNAVAILABLE && std::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
            return r;
        }
    }
}

pub async fn admin(c: &reqwest::Client, method: reqwest::Method, url: &str, body: Option<Value>) -> Resp {
    send(c, method, url, Some(ADMIN), body).await
}

/// Create a database (with `limits`) and a key with `scopes` via the Admin API.
pub async fn create_db_with(node_url: &str, limits: Value, scopes: &[&str]) -> Db {
    let c = http();
    let id = new_database_id();
    admin(
        &c,
        reqwest::Method::POST,
        &format!("{node_url}/admin/v1/databases"),
        Some(json!({"database_id": id, "org_id": "org_test", "plan": "free", "limits": limits})),
    )
    .await
    .ok();
    let key = add_key(node_url, &id, "key_main", scopes).await;
    Db {
        c,
        base: node_url.to_string(),
        id,
        key,
    }
}

pub async fn create_db(node_url: &str) -> Db {
    create_db_with(node_url, json!({}), &["read", "write", "admin"]).await
}

pub async fn add_key(node_url: &str, id: &str, key_id: &str, scopes: &[&str]) -> String {
    let key = new_api_key();
    admin(
        &http(),
        reqwest::Method::PUT,
        &format!("{node_url}/admin/v1/databases/{id}/keys/{key_id}"),
        Some(json!({"sha256": sha256_hex(key.as_bytes()), "scopes": scopes})),
    )
    .await
    .ok();
    key
}

pub const H: f64 = 3600.0;

/// The standard demo schema.
pub async fn demo_schema(db: &Db) {
    db.post_retry(
        "/effects",
        json!({
            "name": "bloating",
            "kernel": {"type": "lognormal", "params": {"median": 2.0 * H, "sigma": 0.6}},
            "priors": {"beans": [3, 1], "dairy": [2, 2]},
            "cause_types": ["meal", "snack"],
        }),
    )
    .await
    .ok();
    db.post_retry(
        "/event-types",
        json!({"type": "meal", "weight": {"kind": "attr", "attr": "portion", "value": 1.0}}),
    )
    .await
    .ok();
    db.post_retry(
        "/rules",
        json!({"effect": "bloating", "name": "tiny", "multiplier": 0.5,
               "predicate": {"attr": "portion", "op": "lt", "value": 0.5}}),
    )
    .await
    .ok();
    db.post_retry(
        "/rules",
        json!({"effect": "bloating", "name": "snacks", "multiplier": 0.8, "types": ["snack"]}),
    )
    .await
    .ok();
}
