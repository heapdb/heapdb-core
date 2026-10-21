//! Property tests for keyed heaps (fixed seeds):
//!
//! * ordering by key equals ordering by score at many times,
//! * push accumulation equals brute force `sum w * exp(-lambda (now - t))`,
//! * `rebase` preserves current scores and the order,
//! * filters match a straightforward reference evaluator,
//! * `NativeHeaps` replays to the identical state after random op
//!   sequences and simulated crashes (torn last line).

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use heapdb::event::Attrs;
use heapdb::heaps::{lambda, Filter, HeapBackend, HeapSpec, LeafOp, MemoryHeaps, NativeHeaps};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};
use serde_json::{json, Value};

fn cfg(cases: u32, seed: u64) -> Config {
    Config {
        cases,
        rng_seed: RngSeed::Fixed(seed),
        failure_persistence: None,
        ..Config::default()
    }
}

const HL: f64 = 3600.0;
const IDS: [&str; 8] = ["a", "b", "c", "d", "e", "f", "g", "h"];

fn close(a: f64, b: f64, rel: f64) -> bool {
    a == b || (a - b).abs() <= rel * a.abs().max(b.abs())
}

/// (item index, weight, time) pushes.
fn pushes() -> impl Strategy<Value = Vec<(usize, f64, f64)>> {
    proptest::collection::vec(
        (
            0..IDS.len(),
            prop_oneof![Just(1.0), 0.01..50.0f64],
            0.0..(5.0 * 86400.0),
        ),
        1..60,
    )
}

fn heap_with(ps: &[(usize, f64, f64)], bases: &[(usize, f64)]) -> MemoryHeaps {
    let mut m = MemoryHeaps::new();
    m.create_heap(&HeapSpec::new("q", HL).landmark(86400.0), 0.0)
        .unwrap();
    for &(i, w, t) in ps {
        m.push("q", IDS[i], w, t, None).unwrap();
    }
    for &(i, b) in bases {
        m.set_base("q", IDS[i], b).unwrap();
    }
    m
}

/// Brute force: base * sum w exp(-lambda (now - t)), per pushed item.
fn brute(ps: &[(usize, f64, f64)], bases: &[(usize, f64)], now: f64) -> BTreeMap<&'static str, (f64, f64)> {
    let lam = lambda(HL);
    let mut heat: BTreeMap<&str, f64> = BTreeMap::new();
    for &(i, w, t) in ps {
        *heat.entry(IDS[i]).or_default() += w * (-lam * (now - t)).exp();
    }
    let mut base: BTreeMap<&str, f64> = BTreeMap::new();
    for &(i, b) in bases {
        base.insert(IDS[i], b);
    }
    heat.into_iter()
        .map(|(id, h)| (id, (base.get(id).copied().unwrap_or(1.0) * h, h)))
        .collect()
}

proptest! {
    #![proptest_config(cfg(200, 0x6865_6170_7331))]

    #[test]
    fn key_order_is_score_order_at_every_time(
        ps in pushes(),
        bases in proptest::collection::vec((0..IDS.len(), 0.05..20.0f64), 0..6),
        nows in proptest::collection::vec(0.0..(10.0 * 86400.0), 1..8),
    ) {
        let m = heap_with(&ps, &bases);
        for now in nows {
            let got = m.top_k("q", IDS.len(), now, None).unwrap();
            // Scores are non-increasing down the list (up to rounding) ...
            for w in got.windows(2) {
                prop_assert!(w[0].score >= w[1].score * (1.0 - 1e-12), "{} < {}", w[0].score, w[1].score);
            }
            // ... and the order equals sorting the brute-force scores
            // wherever they differ by more than rounding.
            let want = brute(&ps, &bases, now);
            prop_assert_eq!(got.len(), want.len());
            for w in got.windows(2) {
                let (a, b) = (want[w[0].id.as_str()].0, want[w[1].id.as_str()].0);
                prop_assert!(a >= b * (1.0 - 1e-9), "{} ({}) ranked above {} ({})", w[0].id, a, w[1].id, b);
            }
        }
    }

    #[test]
    fn push_accumulation_equals_brute_force(
        ps in pushes(),
        bases in proptest::collection::vec((0..IDS.len(), 0.05..20.0f64), 0..6),
        now in (4.0 * 86400.0)..(6.0 * 86400.0),
    ) {
        let m = heap_with(&ps, &bases);
        for (id, (score, heat)) in brute(&ps, &bases, now) {
            let r = m.get("q", id, now).unwrap().unwrap();
            prop_assert!(close(r.heat, heat, 1e-9), "{id}: heat {} vs {}", r.heat, heat);
            prop_assert!(close(r.score, score, 1e-9), "{id}: score {} vs {}", r.score, score);
        }
    }

    #[test]
    fn rebase_preserves_scores_and_order(
        ps in pushes(),
        bases in proptest::collection::vec((0..IDS.len(), 0.05..20.0f64), 0..6),
        at in 0.0..(10.0 * 86400.0),
        later in 0.0..(2.0 * 86400.0),
        new_hl in prop_oneof![Just(None), (60.0..86400.0f64).prop_map(Some)],
    ) {
        let mut m = heap_with(&ps, &bases);
        let before = m.top_k("q", IDS.len(), at, None).unwrap();
        let info = m.rebase("q", at, new_hl).unwrap();
        prop_assert_eq!(info.landmark, at);
        let after = m.top_k("q", IDS.len(), at, None).unwrap();
        let ids = |v: &[heapdb::Ranked]| v.iter().map(|r| r.id.clone()).collect::<Vec<_>>();
        prop_assert_eq!(ids(&before), ids(&after));
        for (b, a) in before.iter().zip(&after) {
            prop_assert!(close(a.score, b.score, 1e-9), "{} vs {}", a.score, b.score);
            prop_assert!(close(a.heat, b.heat, 1e-9));
            prop_assert_eq!(a.version, b.version);
        }
        // Without a new half-life the order stays right later on, too.
        if new_hl.is_none() {
            let later_want = heap_with(&ps, &bases).top_k("q", IDS.len(), at + later, None).unwrap();
            let later_got = m.top_k("q", IDS.len(), at + later, None).unwrap();
            for (w, g) in later_want.iter().zip(&later_got) {
                prop_assert!(close(w.score, g.score, 1e-9));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Filters vs a reference evaluator
// ---------------------------------------------------------------------------

fn scalar() -> impl Strategy<Value = Value> {
    prop_oneof![
        (0..4i64).prop_map(Value::from),
        Just(json!(1.0)),
        Just(json!(2.5)),
        prop_oneof![Just("a"), Just("b"), Just("ab"), Just("")].prop_map(Value::from),
        any::<bool>().prop_map(Value::from),
    ]
}

fn attr_value() -> impl Strategy<Value = Value> {
    prop_oneof![
        3 => scalar(),
        2 => proptest::collection::vec(scalar(), 0..4).prop_map(Value::from),
        1 => Just(Value::Null),
    ]
}

fn attrs() -> impl Strategy<Value = Attrs> {
    proptest::collection::btree_map(prop_oneof![Just("x"), Just("y"), Just("z")], attr_value(), 0..3)
        .prop_map(|m| m.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn leaf() -> impl Strategy<Value = Value> {
    let attr = prop_oneof![Just("x"), Just("y"), Just("z"), Just("w")];
    let ops = [
        "eq",
        "ne",
        "lt",
        "lte",
        "gt",
        "gte",
        "in",
        "not_in",
        "contains",
        "contains_all",
        "subset_of",
        "exists",
        "missing",
    ];
    (
        attr,
        0..ops.len(),
        scalar(),
        proptest::collection::vec(scalar(), 0..4),
    )
        .prop_map(move |(a, o, s, list)| {
            let op = ops[o];
            match op {
                "exists" | "missing" => json!({"attr": a, "op": op}),
                "in" | "not_in" | "contains_all" | "subset_of" => json!({"attr": a, "op": op, "value": list}),
                "lt" | "lte" | "gt" | "gte" if s.is_boolean() => json!({"attr": a, "op": op, "value": 1}),
                _ => json!({"attr": a, "op": op, "value": s}),
            }
        })
}

fn filter() -> impl Strategy<Value = Value> {
    leaf().prop_recursive(3, 24, 4, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..4).prop_map(|v| json!({"all": v})),
            proptest::collection::vec(inner.clone(), 0..4).prop_map(|v| json!({"any": v})),
            inner.prop_map(|v| json!({"not": v})),
        ]
    })
}

/// The reference: the table in docs/heaps.md, written as plainly as possible.
fn reference(f: &Value, a: &Attrs) -> bool {
    if let Some(cs) = f.get("all") {
        return cs.as_array().unwrap().iter().all(|c| reference(c, a));
    }
    if let Some(cs) = f.get("any") {
        return cs.as_array().unwrap().iter().any(|c| reference(c, a));
    }
    if let Some(c) = f.get("not") {
        return !reference(c, a);
    }
    fn eq(x: &Value, y: &Value) -> bool {
        match (x, y) {
            (Value::Number(p), Value::Number(q)) => p.as_f64() == q.as_f64(),
            (Value::Array(p), Value::Array(q)) => {
                p.len() == q.len() && p.iter().zip(q).all(|(u, v)| eq(u, v))
            }
            _ => x == y,
        }
    }
    fn member(list: &[Value], x: &Value) -> bool {
        list.iter().any(|y| eq(x, y))
    }
    let lhs = a.get(f["attr"].as_str().unwrap()).filter(|v| !v.is_null());
    let op = f["op"].as_str().unwrap();
    let v = f.get("value").cloned().unwrap_or(Value::Null);
    let empty = vec![];
    let list = v.as_array().unwrap_or(&empty);
    let Some(x) = lhs else {
        return matches!(op, "ne" | "not_in" | "missing" | "subset_of");
    };
    let cmp = || match (x, &v) {
        (Value::Number(p), Value::Number(q)) => p.as_f64().unwrap().partial_cmp(&q.as_f64().unwrap()),
        (Value::String(p), Value::String(q)) => Some(p.cmp(q)),
        _ => None,
    };
    use std::cmp::Ordering::*;
    match op {
        "eq" => eq(x, &v),
        "ne" => !eq(x, &v),
        "lt" => cmp() == Some(Less),
        "lte" => matches!(cmp(), Some(Less | Equal)),
        "gt" => cmp() == Some(Greater),
        "gte" => matches!(cmp(), Some(Greater | Equal)),
        "in" => member(list, x),
        "not_in" => !member(list, x),
        "contains" => match x {
            Value::Array(xs) => member(xs, &v),
            Value::String(s) => v.as_str().is_some_and(|t| s.contains(t)),
            _ => false,
        },
        "contains_all" => x.as_array().is_some_and(|xs| list.iter().all(|y| member(xs, y))),
        "subset_of" => x.as_array().is_some_and(|xs| xs.iter().all(|y| member(list, y))),
        "exists" => true,
        "missing" => false,
        _ => unreachable!(),
    }
}

proptest! {
    #![proptest_config(cfg(400, 0x6669_6c74))]

    #[test]
    fn filters_match_the_reference(f in filter(), items in proptest::collection::vec(attrs(), 1..12)) {
        let parsed = Filter::from_json(&f).unwrap();
        prop_assert_eq!(Filter::from_json(&parsed.to_json()).unwrap(), parsed.clone());
        for a in &items {
            prop_assert_eq!(parsed.matches(a), reference(&f, a), "filter {} attrs {:?}", f, a);
        }
        // And top_k with the filter = brute-force filtering of the full order.
        let mut m = MemoryHeaps::new();
        m.create_heap(&HeapSpec::new("q", HL).landmark(0.0), 0.0).unwrap();
        for (i, a) in items.iter().enumerate() {
            m.push("q", &format!("i{i:02}"), 1.0 + (i % 3) as f64, i as f64 * 10.0, Some(a)).unwrap();
        }
        let all = m.top_k("q", 100, 500.0, None).unwrap();
        let want: Vec<String> = all.iter().filter(|r| reference(&f, &r.attrs)).take(3).map(|r| r.id.clone()).collect();
        let got: Vec<String> = m.top_k("q", 3, 500.0, Some(&parsed)).unwrap().into_iter().map(|r| r.id).collect();
        prop_assert_eq!(got, want);
    }
}

#[test]
fn leaf_op_names_round_trip() {
    for op in ["eq", "subset_of", "contains_all", "missing"] {
        assert_eq!(LeafOp::parse(op).unwrap().as_str(), op);
    }
}

// ---------------------------------------------------------------------------
// NativeHeaps replay
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Op {
    Push(usize, f64, f64, Option<Attrs>),
    Base(usize, f64),
    Attrs(usize, Attrs, bool),
    Remove(usize),
    Pop(f64, Option<Value>),
    Rebase(f64, Option<f64>),
    Drop,
    Bad,
    Crash,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (0..IDS.len(), 0.1..10.0f64, 0.0..86400.0f64, proptest::option::of(attrs())).prop_map(|(i, w, t, a)| Op::Push(i, w, t, a)),
        2 => (0..IDS.len(), 0.1..10.0f64).prop_map(|(i, b)| Op::Base(i, b)),
        2 => (0..IDS.len(), attrs(), any::<bool>()).prop_map(|(i, a, m)| Op::Attrs(i, a, m)),
        1 => (0..IDS.len()).prop_map(Op::Remove),
        2 => (0.0..86400.0f64, proptest::option::of(leaf())).prop_map(|(n, f)| Op::Pop(n, f)),
        1 => (0.0..86400.0f64, proptest::option::of(600.0..7200.0f64)).prop_map(|(n, h)| Op::Rebase(n, h)),
        1 => Just(Op::Drop),
        1 => Just(Op::Bad),
        2 => Just(Op::Crash),
    ]
}

fn scratch() -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let d = std::env::temp_dir().join(format!(
        "heapdb-props-{}-{}",
        std::process::id(),
        N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ));
    let _ = fs::remove_dir_all(&d);
    d
}

fn run(h: &mut dyn HeapBackend, op: &Op) -> String {
    let r = match op {
        Op::Push(i, w, t, a) => h.push("q", IDS[*i], *w, *t, a.as_ref()).map(|x| format!("{x:?}")),
        Op::Base(i, b) => h.set_base("q", IDS[*i], *b).map(|x| format!("{x:?}")),
        Op::Attrs(i, a, m) => h.set_attrs("q", IDS[*i], a, *m).map(|x| format!("{x:?}")),
        Op::Remove(i) => h.remove("q", IDS[*i]).map(|x| format!("{x:?}")),
        Op::Pop(now, f) => {
            let f = f.as_ref().map(|v| Filter::from_json(v).unwrap());
            h.pop("q", *now, f.as_ref()).map(|x| format!("{x:?}"))
        }
        Op::Rebase(now, hl) => h.rebase("q", *now, *hl).map(|x| format!("{x:?}")),
        Op::Drop => h
            .drop_heap("q")
            .and_then(|_| h.create_heap(&HeapSpec::new("q", HL).landmark(0.0), 0.0))
            .map(|x| format!("{x:?}")),
        Op::Bad => h.push("q", "a", -1.0, 0.0, None).map(|x| format!("{x:?}")),
        Op::Crash => Ok(String::new()),
    };
    r.unwrap_or_else(|e| format!("error: {e}"))
}

proptest! {
    #![proptest_config(cfg(60, 0x6e61_7469_7665))]

    #[test]
    fn native_replays_to_identical_state(ops in proptest::collection::vec(op(), 1..80), every in 1..20u64, torn in any::<bool>()) {
        let dir = scratch();
        let mut mem = MemoryHeaps::new();
        let mut nat = NativeHeaps::open(&dir).unwrap().with_checkpoint_every(every);
        mem.create_heap(&HeapSpec::new("q", HL).landmark(0.0), 0.0).unwrap();
        nat.create_heap(&HeapSpec::new("q", HL).landmark(0.0), 0.0).unwrap();
        for op in &ops {
            if let Op::Crash = op {
                // Simulated crash: drop without any shutdown, maybe leave a
                // torn (unacknowledged) line, reopen.
                drop(nat);
                if torn {
                    let mut f = fs::OpenOptions::new().append(true).create(true).open(dir.join("ops.jsonl")).unwrap();
                    f.write_all(br#"{"seq":1000000,"op":"push","args":{"heap":"q","it"#).unwrap();
                }
                nat = NativeHeaps::open(&dir).unwrap().with_checkpoint_every(every);
                prop_assert_eq!(nat.state().to_records(), mem.to_records());
                continue;
            }
            let a = run(&mut mem, op);
            let b = run(&mut nat, op);
            prop_assert_eq!(a, b);
        }
        drop(nat);
        let reopened = NativeHeaps::open(&dir).unwrap();
        prop_assert_eq!(reopened.state().to_records(), mem.to_records());
        fs::remove_dir_all(&dir).unwrap();
    }
}
