//! Crash recovery of the real binary: SIGKILL while writes are in flight,
//! restart on the same data directory, every acknowledged write is there.

mod common;

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::*;
use serde_json::{json, Value};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn spawn(dir: &std::path::Path, port: u16, iport: u16) -> Child {
    Command::new(env!("CARGO_BIN_EXE_heapdb-server"))
        .arg("--data-dir")
        .arg(dir)
        .arg("--listen")
        .arg(format!("127.0.0.1:{port}"))
        .arg("--internal-listen")
        .arg(format!("127.0.0.1:{iport}"))
        .env("HEAPDB_ADMIN_TOKEN", ADMIN)
        .env("HEAPDB_SNAPSHOT_RECORDS", "50")
        .env("HEAPDB_LOG", "warn")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("binary runs")
}

async fn wait_ready(url: &str) {
    let c = http();
    for _ in 0..200 {
        if let Ok(r) = c.get(format!("{url}/readyz")).send().await {
            if r.status().is_success() {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server at {url} did not become ready");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn kill_9_loses_no_acknowledged_write() {
    let dir = tempfile::tempdir().unwrap();
    let (port, iport) = (free_port(), free_port());
    let url = format!("http://127.0.0.1:{port}");
    let mut child = spawn(dir.path(), port, iport);
    wait_ready(&url).await;
    let db = create_db(&url).await;
    demo_schema(&db).await;

    // Writers hammer the server; we record (external_id, id) of every ack.
    let acked: Arc<Mutex<Vec<(String, u64)>>> = Arc::default();
    let mut tasks = Vec::new();
    for w in 0..6 {
        let db = db.clone();
        let acked = acked.clone();
        tasks.push(tokio::spawn(async move {
            for i in 0.. {
                let xid = format!("w{w}-{i}");
                let body = json!({"events": [{"entity": format!("e{w}"), "time": i as f64 * 60.0, "type": "meal",
                    "features": ["beans", format!("f{}", i % 7)], "external_id": xid}]});
                let r = db.c.post(db.url("/events")).bearer_auth(&db.key).json(&body).send().await;
                let Ok(r) = r else { break };
                if !r.status().is_success() {
                    break;
                }
                let Ok(v) = r.json::<Value>().await else { break };
                acked.lock().unwrap().push((xid, v["ids"][0].as_u64().unwrap()));
            }
        }));
    }
    tokio::time::sleep(Duration::from_millis(1500)).await;
    child.kill().unwrap(); // SIGKILL
    child.wait().unwrap();
    for t in tasks {
        let _ = t.await;
    }
    let acked = acked.lock().unwrap().clone();
    assert!(acked.len() > 50, "only {} writes acknowledged", acked.len());

    let mut child = spawn(dir.path(), port, iport);
    wait_ready(&url).await;
    // Replaying every acknowledged insert (same external_id) must return the
    // original id: the event survived and dedupe state was recovered.
    for chunk in acked.chunks(500) {
        let events: Vec<Value> = chunk
            .iter()
            .map(|(x, _)| json!({"entity": "replay", "time": 0, "type": "meal", "external_id": x}))
            .collect();
        let r = db.post("/events", json!({"events": events})).await.ok();
        let got: Vec<u64> = r["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let want: Vec<u64> = chunk.iter().map(|(_, id)| *id).collect();
        assert_eq!(got, want);
    }
    let info = db.get("").await.ok();
    assert!(info["usage"]["events_stored"].as_u64().unwrap() >= acked.len() as u64);
    // Queries work on the recovered state (rules + weight functions restored).
    db.post(
        "/query/top-causes",
        json!({"entity": "e0", "effect": "bloating", "at": 3600.0, "explain": true}),
    )
    .await
    .ok();
    child.kill().unwrap();
    child.wait().unwrap();
}
