//! Snapshots written by the original Python implementations (v0.1 -> format
//! v1, v0.2 -> v2, v0.4 -> v3) load, and answer exactly as the v0.4
//! reference implementation did after loading them.

use std::path::PathBuf;

use heapdb::{HeapDB, NewEvent, TopK};
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

fn check_against_expected(db: &mut HeapDB, ver: u32) {
    let exp: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture(&format!("snapshot_v{ver}.expected.json"))).unwrap(),
    )
    .unwrap();
    let mut n = 0;
    for a in exp["answers"].as_array().unwrap() {
        let (eff, ent, at) = (
            a["effect"].as_str().unwrap(),
            a["entity"].as_str().unwrap(),
            a["at"].as_f64().unwrap(),
        );
        let r = db.top_causes(ent, eff, at, TopK::new(5).explain(true)).unwrap();
        let want: Vec<(u64, f64)> = a["causes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| (c[0].as_u64().unwrap(), c[1].as_f64().unwrap()))
            .collect();
        let got: Vec<(u64, f64)> = r.causes.iter().map(|c| (c.event.id, c.score)).collect();
        assert_eq!(want.len(), got.len(), "v{ver} {eff} {ent} {at}");
        for (w, g) in want.iter().zip(&got) {
            assert_eq!(w.0, g.0, "v{ver} {eff} {ent} {at}");
            assert!(close(w.1, g.1), "v{ver} {eff} {ent} {at}: {} vs {}", g.1, w.1);
        }
        assert_eq!(
            r.to_string(),
            a["text"].as_str().unwrap(),
            "v{ver} text {eff} {ent} {at}"
        );
        n += got.len();
    }
    assert!(n > 100);
    for (k, v) in exp["links"].as_object().unwrap() {
        let parts: Vec<&str> = k.split('|').collect();
        let ent = if parts[2].is_empty() { None } else { Some(parts[2]) };
        let b = db.link(parts[0], parts[1], ent).unwrap();
        assert!(close(b.alpha(), v[0].as_f64().unwrap()), "v{ver} {k} alpha");
        assert!(close(b.beta(), v[1].as_f64().unwrap()), "v{ver} {k} beta");
    }
    let meta = &exp["meta"];
    let pending: Vec<String> = db.pending_event_types().iter().cloned().collect();
    let want_pending: Vec<String> = meta["pending_event_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(pending, want_pending);
    for (eff, rules) in meta["pending_rules"].as_object().unwrap() {
        let want: Vec<&str> = rules
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(db.effect(eff).unwrap().pending_rules(), want.as_slice());
    }
    let ev = db.insert(NewEvent::new("x", 0.0, "t")).unwrap();
    assert_eq!(ev.id, meta["next_id"].as_u64().unwrap());
}

#[test]
fn loads_v1_v2_v3_snapshots_like_the_reference() {
    for ver in 1..=3 {
        let mut db = HeapDB::load(fixture(&format!("snapshot_v{ver}.jsonl"))).unwrap();
        check_against_expected(&mut db, ver);
    }
}

#[test]
fn rewritten_snapshots_answer_identically() {
    for ver in 1..=3 {
        let db = HeapDB::load(fixture(&format!("snapshot_v{ver}.jsonl"))).unwrap();
        let text = db.snapshot_string();
        assert!(text.starts_with("{\"kind\":\"meta\",\"version\":3"));
        let mut db2 = HeapDB::load_from(text.as_bytes(), "memory").unwrap();
        check_against_expected(&mut db2, ver);
        // Writing again is a fixed point.
        let db3 = HeapDB::load_from(text.as_bytes(), "memory").unwrap();
        assert_eq!(db3.snapshot_string(), text);
    }
}

#[test]
fn bad_snapshots_are_rejected() {
    assert!(HeapDB::load_from(&b""[..], "x").is_err());
    assert!(HeapDB::load_from(&b"{\"kind\":\"event\"}\n"[..], "x").is_err());
    assert!(HeapDB::load_from(
        &b"{\"kind\":\"meta\",\"bucket_seconds\":900,\"next_id\":1}\n{\"kind\":\"zzz\"}\n"[..],
        "x"
    )
    .is_err());
    assert!(HeapDB::load_from(&b"not json\n"[..], "x").is_err());
}

#[test]
fn infinite_leak_round_trips_as_bare_token() {
    use heapdb::{EffectOptions, Kernel};
    let mut db = HeapDB::default();
    db.register_effect(
        "x",
        EffectOptions::new(Kernel::exponential(60.0).unwrap()).leak(f64::INFINITY),
    )
    .unwrap();
    let s = db.snapshot_string();
    assert!(s.contains("\"leak\":Infinity"), "{s}");
    let db2 = HeapDB::load_from(s.as_bytes(), "m").unwrap();
    assert_eq!(db2.effect("x").unwrap().leak(), f64::INFINITY);
}
