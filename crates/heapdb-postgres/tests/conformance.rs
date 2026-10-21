//! Conformance: the same seeded random operation sequences against
//! `MemoryHeaps`, `NativeHeaps` and `PostgresHeaps`, with identical results
//! (ids, order, versions, booleans, errors by kind; keys and scores within
//! 1e-9 relative).
//!
//! Postgres runs when `HEAPDB_TEST_POSTGRES_URL` is set (a server URL; each
//! run creates and drops a scratch database). Otherwise the Postgres parts
//! print a visible skip and the suite still compares memory and native.
//!
//! ```text
//! HEAPDB_TEST_POSTGRES_URL=postgres://postgres:postgres@localhost:5432/postgres \
//!   cargo test -p heapdb-postgres
//! ```

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Barrier, Mutex};

use heapdb::event::Attrs;
use heapdb::heaps::{HeapBackend, HeapSpec, MemoryHeaps, NativeHeaps};
use heapdb::{Error, Filter, HeapInfo, HeapItem, Ranked};
use heapdb_postgres::PostgresHeaps;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Scratch databases
// ---------------------------------------------------------------------------

/// A scratch database, dropped at the end.
struct Scratch {
    admin_url: String,
    name: String,
    url: String,
}

fn with_db(url: &str, db: &str) -> String {
    // Replace the path (database) of a postgres:// URL.
    let (scheme, rest) = url.split_once("://").expect("a postgres:// URL");
    let (authority, query) = match rest.split_once('/') {
        Some((a, tail)) => (a, tail.split_once('?').map(|(_, q)| q)),
        None => (rest, None),
    };
    match query {
        Some(q) => format!("{scheme}://{authority}/{db}?{q}"),
        None => format!("{scheme}://{authority}/{db}"),
    }
}

impl Scratch {
    fn new(tag: &str) -> Option<Scratch> {
        let admin_url = std::env::var("HEAPDB_TEST_POSTGRES_URL").ok()?;
        let name = format!(
            "heapdb_test_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        );
        let mut c = heapdb_postgres::connect(&admin_url).expect("HEAPDB_TEST_POSTGRES_URL reachable");
        c.batch_execute(&format!("CREATE DATABASE {name}")).unwrap();
        let url = with_db(&admin_url, &name);
        Some(Scratch { admin_url, name, url })
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Ok(mut c) = heapdb_postgres::connect(&self.admin_url) {
            let _ = c.batch_execute(&format!("DROP DATABASE IF EXISTS {} WITH (FORCE)", self.name));
        }
    }
}

fn skip(what: &str) {
    eprintln!("SKIPPED {what}: set HEAPDB_TEST_POSTGRES_URL (e.g. postgres://postgres:postgres@localhost:5432/postgres) to run it");
}

fn tmpdir(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let d = std::env::temp_dir().join(format!(
        "heapdb-conf-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    ));
    let _ = std::fs::remove_dir_all(&d);
    d
}

// ---------------------------------------------------------------------------
// A small seeded generator (xorshift*) for the operation sequences
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
    fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }
}

const HEAPS: [&str; 3] = ["work", "acme/checkout", "q é"];
const IDS: [&str; 10] = ["a", "b", "c", "CHK-1", "CHK-2", "x/y", "é", "it's", "z", "Z"];
const KINDS: [&str; 4] = ["bug", "task", "epic", "chore"];
const SKILLS: [&str; 4] = ["ts", "react", "go", "sql"];
const T0: f64 = 1_760_000_000.0;

fn scalar(r: &mut Rng) -> Value {
    match r.below(6) {
        0 => json!(r.below(4)),
        1 => json!(r.below(4) as f64 + 0.5),
        2 => json!(r.pick(&KINDS)),
        3 => json!(r.chance(0.5)),
        4 => json!(1.0),
        _ => json!(r.pick(&["", "b", "bu", "Bug"])),
    }
}

fn list(r: &mut Rng, pool: &[&str]) -> Value {
    let n = r.below(4);
    Value::Array((0..n).map(|_| json!(r.pick(pool))).collect())
}

fn attrs(r: &mut Rng) -> Attrs {
    let mut a = Attrs::new();
    if r.chance(0.8) {
        a.insert("kind".into(), json!(r.pick(&KINDS)));
    }
    if r.chance(0.6) {
        a.insert("required_skills".into(), list(r, &SKILLS));
    }
    if r.chance(0.5) {
        a.insert("labels".into(), list(r, &["needs-human", "p1", "flaky"]));
    }
    if r.chance(0.5) {
        a.insert("risk".into(), json!(r.below(5)));
    }
    if r.chance(0.3) {
        a.insert("blocked".into(), json!(r.chance(0.5)));
    }
    if r.chance(0.2) {
        a.insert("note".into(), Value::Null);
    }
    if r.chance(0.2) {
        a.insert("nested".into(), json!({"k": [1, {"x": r.below(2)}]}));
    }
    a
}

fn leaf(r: &mut Rng) -> Value {
    let attr = *r.pick(&[
        "kind",
        "required_skills",
        "labels",
        "risk",
        "blocked",
        "note",
        "nested",
        "absent",
    ]);
    let op = *r.pick(&[
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
    ]);
    match op {
        "exists" | "missing" => json!({"attr": attr, "op": op}),
        "in" | "not_in" | "contains_all" | "subset_of" => {
            let pool: Vec<Value> = (0..r.below(4))
                .map(|_| match r.below(3) {
                    0 => json!(r.pick(&SKILLS)),
                    1 => json!(r.pick(&KINDS)),
                    _ => scalar(r),
                })
                .collect();
            json!({"attr": attr, "op": op, "value": pool})
        }
        "lt" | "lte" | "gt" | "gte" => {
            let v = if r.chance(0.5) {
                json!(r.below(5) as f64 - 0.5)
            } else {
                json!(r.pick(&KINDS))
            };
            json!({"attr": attr, "op": op, "value": v})
        }
        "contains" => {
            let v = match r.below(3) {
                0 => json!(r.pick(&SKILLS)),
                1 => json!(r.pick(&["needs-human", "p1"])),
                _ => scalar(r),
            };
            json!({"attr": attr, "op": op, "value": v})
        }
        _ => {
            let v = if attr == "nested" {
                json!({"k": [1.0, {"x": 0}]})
            } else {
                scalar(r)
            };
            json!({"attr": attr, "op": op, "value": v})
        }
    }
}

fn filter(r: &mut Rng, depth: usize) -> Value {
    if depth >= 2 || r.chance(0.5) {
        return leaf(r);
    }
    match r.below(3) {
        0 => {
            Value::Array((0..r.below(4)).map(|_| filter(r, depth + 1)).collect()).pipe(|v| json!({"all": v}))
        }
        1 => {
            Value::Array((0..r.below(4)).map(|_| filter(r, depth + 1)).collect()).pipe(|v| json!({"any": v}))
        }
        _ => json!({"not": filter(r, depth + 1)}),
    }
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl Pipe for Value {}

// ---------------------------------------------------------------------------
// Operations and comparable outcomes
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Op {
    Create(String, f64, Option<f64>, Option<Attrs>),
    Drop(String),
    Info(String),
    List,
    Push(String, String, f64, f64, Option<Attrs>),
    Base(String, String, f64),
    Attrs(String, String, Attrs, bool),
    Remove(String, String),
    Get(String, String, f64),
    Top(String, usize, f64, Option<Value>),
    Pop(String, f64, Option<Value>),
    Rebase(String, f64, Option<f64>),
}

fn gen_op(r: &mut Rng, clock: &mut f64) -> Op {
    *clock += r.unit() * 3600.0;
    let heap = r.pick(&HEAPS).to_string();
    let id = r.pick(&IDS).to_string();
    let now = *clock;
    let f = |r: &mut Rng| if r.chance(0.6) { Some(filter(r, 0)) } else { None };
    match r.below(100) {
        0..=3 => Op::Create(
            heap,
            *r.pick(&[3600.0, 86400.0, 259200.0]),
            if r.chance(0.5) { Some(T0) } else { None },
            if r.chance(0.3) { Some(attrs(r)) } else { None },
        ),
        4 => Op::Drop(heap),
        5..=6 => Op::Info(heap),
        7 => Op::List,
        8..=42 => {
            let w = if r.chance(0.2) {
                1.0
            } else {
                0.01 + r.unit() * 20.0
            };
            let t = now - r.unit() * 3.0 * 86400.0;
            let a = if r.chance(0.5) { Some(attrs(r)) } else { None };
            // Occasionally invalid arguments.
            let w = if r.chance(0.02) { -1.0 } else { w };
            Op::Push(heap, id, w, t, a)
        }
        43..=50 => Op::Base(
            heap,
            id,
            if r.chance(0.03) {
                0.0
            } else {
                0.1 + r.unit() * 10.0
            },
        ),
        51..=58 => Op::Attrs(heap, id, attrs(r), r.chance(0.5)),
        59..=63 => Op::Remove(heap, id),
        64..=68 => Op::Get(heap, id, now),
        69..=85 => Op::Top(heap, r.below(6), now, f(r)),
        86..=95 => Op::Pop(heap, now, f(r)),
        _ => Op::Rebase(
            heap,
            now,
            if r.chance(0.3) {
                Some(*r.pick(&[3600.0, 7200.0]))
            } else {
                None
            },
        ),
    }
}

/// An outcome, compared with float tolerance.
#[derive(Debug)]
enum Out {
    Info(Option<HeapInfo>),
    Infos(Vec<HeapInfo>),
    Item(HeapItem),
    Bool(bool),
    Ranked(Vec<Ranked>),
    Err(&'static str),
}

fn kind(e: &Error) -> &'static str {
    match e {
        Error::UnknownHeap(_) => "unknown_heap",
        Error::HeapConflict(_) => "conflict",
        Error::Invalid(_) => "invalid",
        other => panic!("unexpected backend error: {other}"),
    }
}

fn run(b: &mut dyn HeapBackend, op: &Op) -> Out {
    let res: heapdb::Result<Out> = (|| {
        Ok(match op {
            Op::Create(n, hl, l, m) => {
                let spec = HeapSpec {
                    name: n.clone(),
                    half_life: *hl,
                    landmark: *l,
                    meta: m.clone(),
                };
                Out::Info(Some(b.create_heap(&spec, T0)?))
            }
            Op::Drop(n) => Out::Bool(b.drop_heap(n)?),
            Op::Info(n) => Out::Info(b.heap(n)?),
            Op::List => Out::Infos(b.heaps()?),
            Op::Push(h, id, w, t, a) => Out::Item(b.push(h, id, *w, *t, a.as_ref())?),
            Op::Base(h, id, x) => Out::Item(b.set_base(h, id, *x)?),
            Op::Attrs(h, id, a, m) => Out::Item(b.set_attrs(h, id, a, *m)?),
            Op::Remove(h, id) => Out::Bool(b.remove(h, id)?),
            Op::Get(h, id, now) => Out::Ranked(b.get(h, id, *now)?.into_iter().collect()),
            Op::Top(h, k, now, f) => {
                let f = f.as_ref().map(|v| Filter::from_json(v).unwrap());
                Out::Ranked(b.top_k(h, *k, *now, f.as_ref())?)
            }
            Op::Pop(h, now, f) => {
                let f = f.as_ref().map(|v| Filter::from_json(v).unwrap());
                Out::Ranked(b.pop(h, *now, f.as_ref())?.into_iter().collect())
            }
            Op::Rebase(h, now, hl) => Out::Info(Some(b.rebase(h, *now, *hl)?)),
        })
    })();
    res.unwrap_or_else(|e| Out::Err(kind(&e)))
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs()) || (a.abs() < 1e-300 && b.abs() < 1e-300)
}

fn close_opt(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => close(x, y),
        (None, None) => true,
        _ => false,
    }
}

fn same_info(a: &HeapInfo, b: &HeapInfo) -> bool {
    a.name == b.name
        && a.half_life == b.half_life
        && close(a.landmark, b.landmark)
        && a.meta == b.meta
        && a.size == b.size
}

fn same_item(a: &HeapItem, b: &HeapItem) -> bool {
    a.id == b.id
        && a.base == b.base
        && close_opt(a.activity, b.activity)
        && close_opt(a.key, b.key)
        && json_same(&a.attrs, &b.attrs)
        && a.version == b.version
}

/// Attrs equal as JSON values (jsonb normalises `1.0` vs `1` etc.).
fn json_same(a: &Attrs, b: &Attrs) -> bool {
    fn eq(x: &Value, y: &Value) -> bool {
        match (x, y) {
            (Value::Number(p), Value::Number(q)) => p.as_f64() == q.as_f64(),
            (Value::Array(p), Value::Array(q)) => {
                p.len() == q.len() && p.iter().zip(q).all(|(u, v)| eq(u, v))
            }
            (Value::Object(p), Value::Object(q)) => {
                p.len() == q.len() && p.iter().all(|(k, v)| q.get(k).is_some_and(|w| eq(v, w)))
            }
            _ => x == y,
        }
    }
    eq(&Value::Object(a.clone()), &Value::Object(b.clone()))
}

fn same_ranked(a: &Ranked, b: &Ranked) -> bool {
    a.rank == b.rank
        && a.id == b.id
        && close_opt(a.key, b.key)
        && close(a.score, b.score)
        && close(a.heat, b.heat)
        && a.base == b.base
        && json_same(&a.attrs, &b.attrs)
        && a.version == b.version
}

fn same(a: &Out, b: &Out) -> bool {
    match (a, b) {
        (Out::Info(x), Out::Info(y)) => match (x, y) {
            (Some(p), Some(q)) => same_info(p, q),
            (None, None) => true,
            _ => false,
        },
        (Out::Infos(x), Out::Infos(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_info(p, q)),
        (Out::Item(x), Out::Item(y)) => same_item(x, y),
        (Out::Bool(x), Out::Bool(y)) => x == y,
        (Out::Ranked(x), Out::Ranked(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_ranked(p, q))
        }
        (Out::Err(x), Out::Err(y)) => x == y,
        _ => false,
    }
}

/// Run `steps` seeded ops on every backend; assert identical outcomes.
fn conform(seed: u64, steps: usize, backends: &mut [(&str, &mut dyn HeapBackend)]) {
    let mut r = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut clock = T0;
    // Always start with the heaps existing (most ops then do something).
    let mut ops: Vec<Op> = HEAPS
        .iter()
        .map(|h| Op::Create(h.to_string(), 86400.0, Some(T0), None))
        .collect();
    ops.extend((0..steps).map(|_| gen_op(&mut r, &mut clock)));
    // Guard against a vacuous comparison: count outcomes that carry data.
    let (mut ranked_rows, mut errors, mut filtered_hits) = (0usize, 0usize, 0usize);
    for (i, op) in ops.iter().enumerate() {
        let outs: Vec<Out> = backends.iter_mut().map(|(_, b)| run(*b, op)).collect();
        match &outs[0] {
            Out::Ranked(v) => {
                ranked_rows += v.len();
                if matches!(op, Op::Top(_, _, _, Some(_)) | Op::Pop(_, _, Some(_))) {
                    filtered_hits += v.len();
                }
            }
            Out::Err(_) => errors += 1,
            _ => {}
        }
        for (j, o) in outs.iter().enumerate().skip(1) {
            assert!(
                same(&outs[0], o),
                "seed {seed} step {i}: {op:?}\n  {}: {:?}\n  {}: {:?}",
                backends[0].0,
                outs[0],
                backends[j].0,
                o
            );
        }
    }
    assert!(
        ranked_rows > 50 && filtered_hits > 5 && errors > 0 && errors < steps / 2,
        "seed {seed}: too few meaningful outcomes (rows {ranked_rows}, filtered {filtered_hits}, errors {errors})"
    );
    // Final state: every heap, every item, in order.
    for h in HEAPS {
        let outs: Vec<Out> = backends
            .iter_mut()
            .map(|(_, b)| run(*b, &Op::Top(h.to_string(), 1000, clock, None)))
            .collect();
        for o in &outs[1..] {
            assert!(same(&outs[0], o), "seed {seed}: final state of {h} differs");
        }
    }
}

const SEEDS: [u64; 12] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233];
const STEPS: usize = 300;

#[test]
fn memory_and_native_conform() {
    for seed in SEEDS {
        let dir = tmpdir("mn");
        let mut mem = MemoryHeaps::new();
        let mut nat = NativeHeaps::open(&dir).unwrap().with_checkpoint_every(37);
        conform(seed, STEPS, &mut [("memory", &mut mem), ("native", &mut nat)]);
        // Reopened from disk it is still the same.
        drop(nat);
        let mut nat = NativeHeaps::open(&dir).unwrap();
        for h in HEAPS {
            let a = mem.top_k(h, 1000, T0 + 1e5, None).ok();
            let b = nat.top_k(h, 1000, T0 + 1e5, None).ok();
            assert_eq!(a, b);
        }
        let _ = nat.drop_heap("work");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn postgres_conforms() {
    let Some(db) = Scratch::new("conf") else {
        return skip("postgres_conforms");
    };
    let mut pg = heapdb_postgres::open(&db.url).unwrap();
    for seed in SEEDS {
        for h in HEAPS {
            pg.drop_heap(h).unwrap();
        }
        let dir = tmpdir("pg");
        let mut mem = MemoryHeaps::new();
        let mut nat = NativeHeaps::open(&dir).unwrap().with_checkpoint_every(50);
        conform(
            seed,
            STEPS,
            &mut [("memory", &mut mem), ("native", &mut nat), ("postgres", &mut pg)],
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn postgres_edge_cases() {
    let Some(db) = Scratch::new("edge") else {
        return skip("postgres_edge_cases");
    };
    let mut pg = heapdb_postgres::open(&db.url).unwrap();
    // Installing again is a no-op upgrade.
    heapdb_postgres::install(pg.client_mut()).unwrap();
    assert_eq!(
        heapdb_postgres::installed_version(pg.client_mut())
            .unwrap()
            .as_deref(),
        Some(heapdb_postgres::SQL_VERSION)
    );
    let mut mem = MemoryHeaps::new();
    let both: &mut [(&str, &mut dyn HeapBackend)] = &mut [("memory", &mut mem), ("postgres", &mut pg)];
    let ops = vec![
        Op::Create("e".into(), 1.0, Some(0.0), None),
        // Ties by id (equal keys), unranked items excluded.
        Op::Push("e".into(), "b".into(), 1.0, 0.0, None),
        Op::Push("e".into(), "a".into(), 1.0, 0.0, None),
        Op::Base("e".into(), "u".into(), 5.0),
        Op::Top("e".into(), 10, 0.0, None),
        // logaddexp far apart (difference > 40) and close together.
        Op::Push("e".into(), "a".into(), 1.0, -100.0, None),
        Op::Push("e".into(), "a".into(), 1.0, 30.0, None),
        Op::Push("e".into(), "a".into(), 1e-300, 0.0, None),
        // Scores underflow to 0 and overflow to Infinity without errors.
        Op::Top("e".into(), 10, 1e6, None),
        Op::Push("e".into(), "far".into(), 1.0, 5000.0, None),
        Op::Get("e".into(), "far".into(), 0.0),
        Op::Get("e".into(), "u".into(), 0.0),
        Op::Get("e".into(), "nope".into(), 0.0),
        // Overflowing key: rejected by both.
        Op::Push("e".into(), "huge".into(), 1.0, 1e300, None),
        Op::Base("e".into(), "a".into(), 1e300),
        Op::Top("e".into(), 10, 10.0, None),
        Op::Rebase("e".into(), 100.0, Some(2.0)),
        Op::Top("e".into(), 10, 110.0, None),
        Op::Create("e".into(), 2.0, None, None),
        Op::Create("e".into(), 3.0, None, None),
        Op::Get("missing-heap".into(), "a".into(), 0.0),
        Op::Remove("missing-heap".into(), "a".into()),
        Op::Drop("missing-heap".into()),
        Op::Push("e".into(), String::new(), 1.0, 0.0, None),
        Op::Push("e".into(), "x".repeat(201), 1.0, 0.0, None),
        Op::Push("e".into(), "é".repeat(100), 1.0, 0.0, None),
        Op::Push("e".into(), "n".into(), f64::NAN, 0.0, None),
        Op::Push("e".into(), "n".into(), 1.0, f64::INFINITY, None),
        Op::Top("e".into(), 0, 0.0, None),
    ];
    for (i, op) in ops.iter().enumerate() {
        let outs: Vec<Out> = both.iter_mut().map(|(_, b)| run(*b, op)).collect();
        assert!(
            same(&outs[0], &outs[1]),
            "step {i}: {op:?}\n  memory: {:?}\n  postgres: {:?}",
            outs[0],
            outs[1]
        );
    }
}

#[test]
fn concurrent_pops_never_share_an_item() {
    let Some(db) = Scratch::new("pop") else {
        return skip("concurrent_pops_never_share_an_item");
    };
    let mut pg = heapdb_postgres::open(&db.url).unwrap();
    pg.create_heap(&HeapSpec::new("q", 3600.0).landmark(T0), T0)
        .unwrap();
    const N: usize = 800;
    for i in 0..N {
        let mut a = Attrs::new();
        a.insert("lane".into(), json!(i % 2));
        pg.push(
            "q",
            &format!("item-{i:04}"),
            1.0 + (i % 17) as f64,
            T0 + (i % 50) as f64,
            Some(&a),
        )
        .unwrap();
    }
    let got: Arc<Mutex<Vec<String>>> = Arc::default();
    let barrier = Arc::new(Barrier::new(16));
    let threads: Vec<_> = (0..16)
        .map(|t| {
            let (url, got, barrier) = (db.url.clone(), got.clone(), barrier.clone());
            std::thread::spawn(move || {
                let mut h = PostgresHeaps::connect(&url).unwrap();
                // Half the poppers filter, so filtered and unfiltered pops race too.
                let f = (t % 2 == 0)
                    .then(|| Filter::from_json(&json!({"attr": "lane", "op": "eq", "value": 0})).unwrap());
                barrier.wait();
                let mut mine = Vec::new();
                loop {
                    let r = h.pop("q", T0 + 100.0, f.as_ref()).unwrap();
                    match r {
                        Some(r) => mine.push(r.id),
                        None if f.is_some() => break,
                        None => {
                            // Items may still be locked by other poppers: check emptiness.
                            if h.heap("q").unwrap().unwrap().size == 0 {
                                break;
                            }
                        }
                    }
                }
                got.lock().unwrap().extend(mine);
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    let got = got.lock().unwrap();
    let unique: HashSet<&String> = got.iter().collect();
    assert_eq!(unique.len(), got.len(), "an item was popped twice");
    assert_eq!(got.len(), N, "every item was popped exactly once");
}

#[test]
fn row_level_security_applies_through_the_functions() {
    let Some(db) = Scratch::new("rls") else {
        return skip("row_level_security_applies_through_the_functions");
    };
    let mut owner = heapdb_postgres::open(&db.url).unwrap();
    owner
        .create_heap(&HeapSpec::new("q", 3600.0).landmark(T0), T0)
        .unwrap();
    for (id, tenant) in [("a1", "acme"), ("a2", "acme"), ("g1", "globex")] {
        let mut a = Attrs::new();
        a.insert("tenant".into(), json!(tenant));
        owner.push("q", id, 1.0, T0, Some(&a)).unwrap();
    }
    let role = format!("heapdb_rls_{}", std::process::id());
    owner
        .client_mut()
        .batch_execute(&format!(
            "DROP ROLE IF EXISTS {role};
             CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS;
             GRANT USAGE ON SCHEMA heapdb TO {role};
             GRANT SELECT, UPDATE ON heapdb.heaps TO {role};
             GRANT SELECT, INSERT, UPDATE, DELETE ON heapdb.items TO {role};
             ALTER TABLE heapdb.items ENABLE ROW LEVEL SECURITY;
             CREATE POLICY tenant_isolation ON heapdb.items
                 USING (attrs ->> 'tenant' = current_setting('app.tenant', true))
                 WITH CHECK (attrs ->> 'tenant' = current_setting('app.tenant', true));"
        ))
        .unwrap();
    let mut app = PostgresHeaps::connect(&db.url).unwrap();
    app.client_mut()
        .batch_execute(&format!("SET ROLE {role}; SET app.tenant = 'acme'"))
        .unwrap();
    let ids = |v: Vec<Ranked>| v.into_iter().map(|r| r.id).collect::<Vec<_>>();
    assert_eq!(
        ids(app.top_k("q", 10, T0, None).unwrap()),
        ["a1", "a2"],
        "only acme rows are visible"
    );
    assert!(app.get("q", "g1", T0).unwrap().is_none());
    // pop never reaches another tenant's item ...
    assert_eq!(app.pop("q", T0, None).unwrap().unwrap().id, "a1");
    assert_eq!(app.pop("q", T0, None).unwrap().unwrap().id, "a2");
    assert!(app.pop("q", T0, None).unwrap().is_none());
    // ... remove cannot delete it, and writes are checked by the policy.
    assert!(!app.remove("q", "g1").unwrap());
    let mut other = Attrs::new();
    other.insert("tenant".into(), json!("globex"));
    let e = app.push("q", "sneaky", 1.0, T0, Some(&other)).unwrap_err();
    assert!(
        matches!(e, Error::Backend(ref m) if m.contains("row-level security")),
        "{e}"
    );
    // The owner (not subject to the policy) still sees the other tenant.
    assert_eq!(ids(owner.top_k("q", 10, T0, None).unwrap()), ["g1"]);
    drop(app);
    owner
        .client_mut()
        .batch_execute(&format!(
            "DROP POLICY tenant_isolation ON heapdb.items; ALTER TABLE heapdb.items DISABLE ROW LEVEL SECURITY;
             REVOKE ALL ON ALL TABLES IN SCHEMA heapdb FROM {role}; REVOKE ALL ON SCHEMA heapdb FROM {role};
             DROP ROLE {role};"
        ))
        .unwrap();
}

#[test]
fn sql_functions_are_security_invoker() {
    let Some(db) = Scratch::new("secdef") else {
        return skip("sql_functions_are_security_invoker");
    };
    let mut pg = heapdb_postgres::open(&db.url).unwrap();
    let n: i64 = pg
        .client_mut()
        .query_one(
            "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
             WHERE n.nspname = 'heapdb' AND p.prosecdef",
            &[],
        )
        .unwrap()
        .get(0);
    assert_eq!(n, 0, "no SECURITY DEFINER functions");
}
