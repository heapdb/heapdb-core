//! Keyed heaps: named, durable priority queues of keyed items ordered by
//! forward decay (see `docs/heaps.md`).
//!
//! A heap has a half-life (`lambda = ln 2 / half_life`) and a landmark time
//! `L`. Each item, identified by a string id, carries
//!
//! * `base`: a slow-moving priority weight (> 0, default 1),
//! * `activity = ln sum_i w_i * exp(lambda * (t_i - L))` over its pushes
//!   (`None` until the first push),
//! * `key = ln(base) + activity` (`None` while `activity` is `None`),
//! * `attrs`: a JSON object that [`Filter`]s are evaluated against,
//! * `version`: incremented on every change.
//!
//! The score at `now` is `exp(key - lambda * (now - L)) = base * heat(now)`
//! with `heat(now) = sum_i w_i * exp(-lambda * (now - t_i))`. The second
//! term is the same for every item, so ordering by key (descending, ties by
//! id ascending) is the ordering by score at every instant.
//!
//! Where the state lives is pluggable ([`HeapBackend`], chosen with a
//! [`PersistTo`]): [`MemoryHeaps`] (process memory), [`NativeHeaps`]
//! (append-only JSONL op log + checkpoints in a directory), and, in the
//! separate `heapdb-postgres` crate, tables and SQL functions in Postgres.
//! heapdb-server keeps heaps in each database's WAL and snapshots. All
//! backends compute keys with the same formulas ([`logaddexp`],
//! [`decay_exp`]), so they agree to the last bit or close to it.

mod filter;
pub mod hql;
mod memory;
mod native;
mod ops;

use std::f64::consts::LN_2;
use std::fmt;
use std::path::PathBuf;

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::event::Attrs;
use crate::pyfmt::{fmt_g6, format_time, repr_f64, repr_str};

pub use filter::{Filter, Leaf, LeafOp, MAX_FILTER_DEPTH, MAX_FILTER_NODES};
pub use memory::{AttrsArgs, BaseArgs, MemoryHeaps, PushArgs};
pub use native::{NativeHeaps, DEFAULT_CHECKPOINT_EVERY};
pub use ops::HeapOp;

/// Maximum length of heap names and item ids, in UTF-8 bytes.
pub const MAX_NAME_BYTES: usize = 200;

/// `ln(f64::MAX)`: `exp` of anything larger overflows to infinity.
pub const MAX_EXP_ARG: f64 = 709.782712893384;

/// Below this, `exp` is (treated as) 0. Postgres raises an underflow error
/// for `exp(x)` with `x < -745.13...`; both engines return 0 here instead.
pub const MIN_EXP_ARG: f64 = -745.0;

/// Above this difference, `ln(1 + exp(-d))` rounds to exactly 0 in `f64`
/// (`exp(-36.7) < 2^-53`), so [`logaddexp`] skips the `exp`. This changes
/// no result; it keeps Postgres' `exp` away from its underflow error.
pub const LOGADDEXP_CUTOFF: f64 = 40.0;

// ---------------------------------------------------------------------------
// Numerics (mirrored exactly by sql/postgres/heapdb.sql)
// ---------------------------------------------------------------------------

/// Decay rate of a half-life: `ln 2 / half_life`.
pub fn lambda(half_life: f64) -> f64 {
    LN_2 / half_life
}

/// `ln(exp(a) + exp(b))` without overflow:
/// `max + ln(1 + exp(-|a - b|))`.
///
/// `ln(1 + x)` rather than `ln_1p(x)` because Postgres has no `log1p`; the
/// difference is below one ulp of the result.
pub fn logaddexp(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    let d = hi - lo;
    if d > LOGADDEXP_CUTOFF {
        hi
    } else {
        hi + (1.0 + (-d).exp()).ln()
    }
}

/// `exp(x)`, with `x < MIN_EXP_ARG` giving 0 and `x > MAX_EXP_ARG` giving
/// infinity (Postgres would raise an error in both cases).
pub fn decay_exp(x: f64) -> f64 {
    if x < MIN_EXP_ARG {
        0.0
    } else if x > MAX_EXP_ARG {
        f64::INFINITY
    } else {
        x.exp()
    }
}

/// The activity contributed by one push: `ln(weight) + lambda * (t - L)`.
pub fn push_activity(weight: f64, t: f64, lam: f64, landmark: f64) -> f64 {
    weight.ln() + lam * (t - landmark)
}

/// `key = ln(base) + activity` (`-0.0` is normalised to `0.0`, so key
/// order is the same as in Postgres, where the two compare equal).
pub fn item_key(base: f64, activity: Option<f64>) -> Option<f64> {
    activity.map(|a| base.ln() + a + 0.0)
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Heap names and item ids: 1-200 UTF-8 bytes, no NUL character (Postgres
/// text cannot hold one).
pub fn check_name(what: &str, s: &str) -> Result<()> {
    if s.is_empty() || s.len() > MAX_NAME_BYTES {
        return Err(Error::invalid(format!(
            "{what} must be 1-{MAX_NAME_BYTES} UTF-8 bytes, got {} bytes",
            s.len()
        )));
    }
    if s.contains('\0') {
        return Err(Error::invalid(format!("{what} must not contain NUL characters")));
    }
    Ok(())
}

/// A finite number `> 0` (weights, bases, half-lives).
pub fn check_positive(what: &str, x: f64) -> Result<()> {
    if !(x.is_finite() && x > 0.0) {
        return Err(Error::invalid(format!(
            "{what} must be finite and > 0, got {}",
            repr_f64(x)
        )));
    }
    Ok(())
}

/// A finite number (times).
pub fn check_finite(what: &str, x: f64) -> Result<()> {
    if !x.is_finite() {
        return Err(Error::invalid(format!(
            "{what} must be finite, got {}",
            repr_f64(x)
        )));
    }
    Ok(())
}

/// Attributes (and filter values) must not contain NUL characters in keys
/// or strings: Postgres `jsonb` rejects `\u0000`.
pub fn check_json(what: &str, v: &Value) -> Result<()> {
    fn walk(v: &Value) -> bool {
        match v {
            Value::String(s) => !s.contains('\0'),
            Value::Array(a) => a.iter().all(walk),
            Value::Object(m) => m.iter().all(|(k, v)| !k.contains('\0') && walk(v)),
            _ => true,
        }
    }
    if walk(v) {
        Ok(())
    } else {
        Err(Error::invalid(format!("{what} must not contain NUL characters")))
    }
}

/// [`check_json`] for an attribute object.
pub fn check_attrs(attrs: &Attrs) -> Result<()> {
    if attrs
        .iter()
        .all(|(k, v)| !k.contains('\0') && check_json("attrs", v).is_ok())
    {
        Ok(())
    } else {
        Err(Error::invalid("attrs must not contain NUL characters"))
    }
}

/// An activity or shifted value that left the `f64` range.
pub(crate) fn too_far(what: &str) -> Error {
    Error::invalid(format!(
        "{what} is too far from the heap's landmark (the key is not finite); rebase the heap"
    ))
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// What `create_heap` asks for.
///
/// `landmark` and `meta` are optional: when the heap already exists, an
/// omitted value matches whatever it has (so a repeated
/// `create_heap(name, half_life)` is idempotent); for a new heap the
/// landmark defaults to `now` and `meta` to `{}`.
#[derive(Clone, Debug, PartialEq)]
pub struct HeapSpec {
    /// Heap name (1-200 UTF-8 bytes).
    pub name: String,
    /// Half-life in seconds (finite, > 0).
    pub half_life: f64,
    /// Landmark time `L` (Unix seconds).
    pub landmark: Option<f64>,
    /// Free-form metadata (JSON object).
    pub meta: Option<Attrs>,
}

impl HeapSpec {
    /// A spec with the default landmark and metadata.
    pub fn new(name: impl Into<String>, half_life: f64) -> Self {
        HeapSpec {
            name: name.into(),
            half_life,
            landmark: None,
            meta: None,
        }
    }

    /// Set the landmark.
    pub fn landmark(mut self, landmark: f64) -> Self {
        self.landmark = Some(landmark);
        self
    }

    /// Set the metadata.
    pub fn meta(mut self, meta: Attrs) -> Self {
        self.meta = Some(meta);
        self
    }

    /// Check the fields.
    pub fn validate(&self) -> Result<()> {
        check_name("heap name", &self.name)?;
        check_positive("half_life", self.half_life)?;
        if let Some(l) = self.landmark {
            check_finite("landmark", l)?;
        }
        if let Some(m) = &self.meta {
            check_attrs(m).map_err(|_| Error::invalid("meta must not contain NUL characters"))?;
        }
        Ok(())
    }

    /// `Err(HeapConflict)` unless this spec matches an existing heap.
    pub fn check_matches(&self, have: &HeapInfo) -> Result<()> {
        let mut diffs = Vec::new();
        if self.half_life != have.half_life {
            diffs.push(format!(
                "half_life {} (requested {})",
                repr_f64(have.half_life),
                repr_f64(self.half_life)
            ));
        }
        if let Some(l) = self.landmark.filter(|l| *l != have.landmark) {
            diffs.push(format!(
                "landmark {} (requested {})",
                repr_f64(have.landmark),
                repr_f64(l)
            ));
        }
        if self.meta.as_ref().is_some_and(|m| *m != have.meta) {
            diffs.push("different meta".to_string());
        }
        if diffs.is_empty() {
            Ok(())
        } else {
            Err(Error::HeapConflict(format!(
                "heap {} already exists with {}",
                repr_str(&self.name),
                diffs.join(", ")
            )))
        }
    }
}

/// A heap's settings and size.
#[derive(Clone, Debug, PartialEq)]
pub struct HeapInfo {
    /// Name.
    pub name: String,
    /// Half-life in seconds.
    pub half_life: f64,
    /// Landmark time `L`.
    pub landmark: f64,
    /// Metadata.
    pub meta: Attrs,
    /// Number of items (ranked or not).
    pub size: u64,
}

impl HeapInfo {
    /// Decay rate `ln 2 / half_life`.
    pub fn lambda(&self) -> f64 {
        lambda(self.half_life)
    }

    /// `{name, half_life, landmark, meta, size}`.
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("name".into(), self.name.clone().into());
        m.insert("half_life".into(), self.half_life.into());
        m.insert("landmark".into(), self.landmark.into());
        m.insert("meta".into(), Value::Object(self.meta.clone()));
        m.insert("size".into(), self.size.into());
        Value::Object(m)
    }
}

impl fmt::Display for HeapInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "heap {}: half_life={} landmark={} items={}",
            repr_str(&self.name),
            fmt_g6(self.half_life),
            format_time(self.landmark),
            self.size
        )
    }
}

/// An item as stored (returned by `push`, `set_base`, `set_attrs`).
#[derive(Clone, Debug, PartialEq)]
pub struct HeapItem {
    /// Item id.
    pub id: String,
    /// Priority weight (> 0).
    pub base: f64,
    /// Log-sum of decayed pushes relative to the landmark (`None`: never pushed).
    pub activity: Option<f64>,
    /// `ln(base) + activity` (`None`: not ranked).
    pub key: Option<f64>,
    /// Attributes.
    pub attrs: Attrs,
    /// Incremented on every change.
    pub version: u64,
}

impl HeapItem {
    /// `{id, base, activity, key, attrs, version}`.
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("id".into(), self.id.clone().into());
        m.insert("base".into(), self.base.into());
        m.insert("activity".into(), self.activity.into());
        m.insert("key".into(), self.key.into());
        m.insert("attrs".into(), Value::Object(self.attrs.clone()));
        m.insert("version".into(), self.version.into());
        Value::Object(m)
    }
}

impl fmt::Display for HeapItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} base={} key={} v{}",
            repr_str(&self.id),
            fmt_g6(self.base),
            self.key.map_or("None".to_string(), repr_f64),
            self.version
        )?;
        write_attrs(f, &self.attrs)
    }
}

/// An item with its score and heat at `now` (returned by `top_k`, `pop`, `get`).
#[derive(Clone, Debug, PartialEq)]
pub struct Ranked {
    /// 1-based position in a `top_k` / `pop` result; 0 from `get`.
    pub rank: usize,
    /// Item id.
    pub id: String,
    /// `ln(base) + activity` (`None` only from `get` of an unranked item).
    pub key: Option<f64>,
    /// `exp(key - lambda * (now - L)) = base * heat` (0 when unranked).
    pub score: f64,
    /// `sum_i w_i * exp(-lambda * (now - t_i))` (0 when unranked).
    pub heat: f64,
    /// Priority weight.
    pub base: f64,
    /// Attributes.
    pub attrs: Attrs,
    /// Version.
    pub version: u64,
}

impl Ranked {
    /// Score and heat of `item` at `now` in a heap with decay rate `lam` and
    /// landmark `landmark`.
    pub fn of(rank: usize, item: &HeapItem, lam: f64, landmark: f64, now: f64) -> Ranked {
        let shift = lam * (now - landmark);
        Ranked {
            rank,
            id: item.id.clone(),
            key: item.key,
            score: item.key.map_or(0.0, |k| decay_exp(k - shift)),
            heat: item.activity.map_or(0.0, |a| decay_exp(a - shift)),
            base: item.base,
            attrs: item.attrs.clone(),
            version: item.version,
        }
    }

    /// `{rank, id, key, score, heat, base, attrs, version}`. A non-finite
    /// score (an item pushed far in the future) is the string `"Infinity"`.
    pub fn to_json(&self) -> Value {
        let num = |x: f64| -> Value {
            if x.is_finite() {
                Value::from(x)
            } else {
                Value::from("Infinity")
            }
        };
        let mut m = Map::new();
        m.insert("rank".into(), self.rank.into());
        m.insert("id".into(), self.id.clone().into());
        m.insert("key".into(), self.key.into());
        m.insert("score".into(), num(self.score));
        m.insert("heat".into(), num(self.heat));
        m.insert("base".into(), self.base.into());
        m.insert("attrs".into(), Value::Object(self.attrs.clone()));
        m.insert("version".into(), self.version.into());
        Value::Object(m)
    }
}

impl fmt::Display for Ranked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<28} score={} heat={} base={} v{}",
            repr_str(&self.id),
            fmt_g6(self.score),
            fmt_g6(self.heat),
            fmt_g6(self.base),
            self.version
        )?;
        write_attrs(f, &self.attrs)
    }
}

fn write_attrs(f: &mut fmt::Formatter<'_>, attrs: &Attrs) -> fmt::Result {
    if attrs.is_empty() {
        return Ok(());
    }
    let s = serde_json::to_string(attrs).map_err(|_| fmt::Error)?;
    write!(f, " {s}")
}

/// The result of `PEEK` / `POP` in the query language.
#[derive(Clone, Debug, PartialEq)]
pub struct HeapTop {
    /// Heap name.
    pub heap: String,
    /// The time scores are computed at.
    pub now: f64,
    /// True for `POP` (at most one item, now removed).
    pub popped: bool,
    /// Ranked items, best first.
    pub items: Vec<Ranked>,
}

impl fmt::Display for HeapTop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let what = if self.popped { "Popped from" } else { "Top of" };
        write!(
            f,
            "{what} heap {} at t={}:",
            repr_str(&self.heap),
            format_time(self.now)
        )?;
        if self.items.is_empty() {
            f.write_str(if self.popped {
                "\n  (nothing to pop)"
            } else {
                "\n  (no ranked items)"
            })?;
        }
        for r in &self.items {
            write!(f, "\n  {}. {r}", r.rank)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// JSON parsing of results (clients of heapdb-server)
// ---------------------------------------------------------------------------

fn field<'a>(m: &'a Map<String, Value>, k: &str) -> Result<&'a Value> {
    m.get(k)
        .ok_or_else(|| Error::invalid(format!("missing field {}", repr_str(k))))
}

fn as_num(v: &Value, k: &str) -> Result<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) if s == "Infinity" => Some(f64::INFINITY),
        Value::String(s) if s == "-Infinity" => Some(f64::NEG_INFINITY),
        Value::String(s) if s == "NaN" => Some(f64::NAN),
        _ => None,
    }
    .ok_or_else(|| Error::invalid(format!("field {} must be a number", repr_str(k))))
}

fn num_field(m: &Map<String, Value>, k: &str) -> Result<f64> {
    as_num(field(m, k)?, k)
}

fn opt_num_field(m: &Map<String, Value>, k: &str) -> Result<Option<f64>> {
    match m.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => as_num(v, k).map(Some),
    }
}

fn str_field(m: &Map<String, Value>, k: &str) -> Result<String> {
    field(m, k)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Error::invalid(format!("field {} must be a string", repr_str(k))))
}

fn u64_field(m: &Map<String, Value>, k: &str) -> Result<u64> {
    field(m, k)?
        .as_u64()
        .ok_or_else(|| Error::invalid(format!("field {} must be a non-negative integer", repr_str(k))))
}

fn obj_field(m: &Map<String, Value>, k: &str) -> Result<Attrs> {
    match m.get(k) {
        None | Some(Value::Null) => Ok(Attrs::new()),
        Some(Value::Object(o)) => Ok(o.clone()),
        Some(_) => Err(Error::invalid(format!("field {} must be an object", repr_str(k)))),
    }
}

impl HeapInfo {
    /// Inverse of [`HeapInfo::to_json`].
    pub fn from_json(m: &Map<String, Value>) -> Result<HeapInfo> {
        Ok(HeapInfo {
            name: str_field(m, "name")?,
            half_life: num_field(m, "half_life")?,
            landmark: num_field(m, "landmark")?,
            meta: obj_field(m, "meta")?,
            size: u64_field(m, "size")?,
        })
    }
}

impl HeapItem {
    /// Inverse of [`HeapItem::to_json`].
    pub fn from_json(m: &Map<String, Value>) -> Result<HeapItem> {
        Ok(HeapItem {
            id: str_field(m, "id")?,
            base: num_field(m, "base")?,
            activity: opt_num_field(m, "activity")?,
            key: opt_num_field(m, "key")?,
            attrs: obj_field(m, "attrs")?,
            version: u64_field(m, "version")?,
        })
    }
}

impl Ranked {
    /// Inverse of [`Ranked::to_json`].
    pub fn from_json(m: &Map<String, Value>) -> Result<Ranked> {
        Ok(Ranked {
            rank: u64_field(m, "rank")? as usize,
            id: str_field(m, "id")?,
            key: opt_num_field(m, "key")?,
            score: num_field(m, "score")?,
            heat: num_field(m, "heat")?,
            base: num_field(m, "base")?,
            attrs: obj_field(m, "attrs")?,
            version: u64_field(m, "version")?,
        })
    }
}

// ---------------------------------------------------------------------------
// The backend trait
// ---------------------------------------------------------------------------

/// Where heaps live. Every backend implements the same semantics and passes
/// the same conformance suite (`crates/heapdb-postgres/tests/conformance.rs`).
///
/// Errors: an unknown heap is [`Error::UnknownHeap`] (except `heap` and
/// `drop_heap`, which report absence); bad arguments are
/// [`Error::Invalid`]; storage failures are [`Error::Backend`] or
/// [`Error::Io`].
pub trait HeapBackend {
    /// Create a heap, or check that the existing one matches `spec`
    /// (`Error::HeapConflict` otherwise). A new heap's landmark defaults to `now`.
    fn create_heap(&mut self, spec: &HeapSpec, now: f64) -> Result<HeapInfo>;
    /// Delete a heap and its items; false if it did not exist.
    fn drop_heap(&mut self, name: &str) -> Result<bool>;
    /// One heap.
    fn heap(&self, name: &str) -> Result<Option<HeapInfo>>;
    /// All heaps, by name.
    fn heaps(&self) -> Result<Vec<HeapInfo>>;
    /// `activity <- logaddexp(activity, ln weight + lambda (t - L))`; creates
    /// the item (base 1) if missing; `attrs` replaces the attributes when given.
    fn push(&mut self, heap: &str, id: &str, weight: f64, t: f64, attrs: Option<&Attrs>) -> Result<HeapItem>;
    /// Set the base; creates the item (unranked) if missing.
    fn set_base(&mut self, heap: &str, id: &str, base: f64) -> Result<HeapItem>;
    /// Replace (or shallow-merge) the attributes; creates the item
    /// (unranked, base 1) if missing.
    fn set_attrs(&mut self, heap: &str, id: &str, attrs: &Attrs, merge: bool) -> Result<HeapItem>;
    /// Delete an item; false if it did not exist.
    fn remove(&mut self, heap: &str, id: &str) -> Result<bool>;
    /// One item with its score and heat at `now` (rank 0).
    fn get(&self, heap: &str, id: &str, now: f64) -> Result<Option<Ranked>>;
    /// The `k` best ranked items matching `filter`, highest key first (ties: id ascending).
    fn top_k(&self, heap: &str, k: usize, now: f64, filter: Option<&Filter>) -> Result<Vec<Ranked>>;
    /// `top_k(.., 1, ..)` and `remove`, atomically.
    fn pop(&mut self, heap: &str, now: f64, filter: Option<&Filter>) -> Result<Option<Ranked>>;
    /// Move the landmark to `now` (optionally changing the half-life),
    /// keeping every item's heat at `now`: `activity <- activity - lambda_old (now - L_old)`.
    /// Item versions do not change.
    fn rebase(&mut self, heap: &str, now: f64, half_life: Option<f64>) -> Result<HeapInfo>;
}

// ---------------------------------------------------------------------------
// persist_to
// ---------------------------------------------------------------------------

/// Where heaps live: `memory`, `native:<dir>` or `postgres://...`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PersistTo {
    /// [`MemoryHeaps`].
    Memory,
    /// [`NativeHeaps`] in this directory.
    Native(PathBuf),
    /// `PostgresHeaps` (crate `heapdb-postgres`) at this connection URL.
    Postgres(String),
}

impl PersistTo {
    /// Parse `memory` (or an empty string), `native:<dir>`, or a
    /// `postgres://` / `postgresql://` URL.
    pub fn parse(s: &str) -> Result<PersistTo> {
        let t = s.trim();
        if t.is_empty() || t.eq_ignore_ascii_case("memory") {
            return Ok(PersistTo::Memory);
        }
        if let Some(dir) = t.strip_prefix("native:") {
            if dir.is_empty() {
                return Err(Error::invalid("persist_to native:<dir> needs a directory"));
            }
            return Ok(PersistTo::Native(PathBuf::from(dir)));
        }
        let lower = t.to_ascii_lowercase();
        if lower.starts_with("postgres://") || lower.starts_with("postgresql://") {
            return Ok(PersistTo::Postgres(t.to_string()));
        }
        Err(Error::invalid(format!(
            "bad persist_to {} (expected \"memory\", \"native:<dir>\" or \"postgres://...\")",
            repr_str(s)
        )))
    }

    /// Open the backend. `Postgres` is implemented in the `heapdb-postgres`
    /// crate (`heapdb_postgres::open`), so this returns an error for it.
    pub fn open(&self) -> Result<Box<dyn HeapBackend + Send>> {
        match self {
            PersistTo::Memory => Ok(Box::new(MemoryHeaps::new())),
            PersistTo::Native(dir) => Ok(Box::new(NativeHeaps::open(dir)?)),
            PersistTo::Postgres(_) => Err(Error::invalid(
                "persist_to postgres://... is provided by the heapdb-postgres crate \
                 (heapdb_postgres::open); the heapdb engine has only memory and native:<dir>",
            )),
        }
    }
}

impl std::str::FromStr for PersistTo {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        PersistTo::parse(s)
    }
}

impl fmt::Display for PersistTo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PersistTo::Memory => f.write_str("memory"),
            PersistTo::Native(d) => write!(f, "native:{}", d.display()),
            // Never print credentials.
            PersistTo::Postgres(url) => f.write_str(&redact_url(url)),
        }
    }
}

/// `postgres://user:secret@host/db` -> `postgres://user:***@host/db`.
pub fn redact_url(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_string();
    };
    let (auth_end, _) = rest
        .char_indices()
        .find(|&(_, c)| c == '/' || c == '?')
        .unwrap_or((rest.len(), ' '));
    let (authority, tail) = rest.split_at(auth_end);
    match authority.rsplit_once('@') {
        Some((userinfo, host)) => {
            let user = userinfo.split_once(':').map_or(userinfo, |(u, _)| u);
            format!("{scheme}://{user}:***@{host}{tail}")
        }
        None => url.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numerics() {
        assert_eq!(logaddexp(0.0, 0.0), LN_2);
        assert_eq!(logaddexp(1000.0, 0.0), 1000.0);
        assert_eq!(logaddexp(-5.0, 3.0), logaddexp(3.0, -5.0));
        assert!((logaddexp(2.0, 1.0) - (2f64.exp() + 1f64.exp()).ln()).abs() < 1e-15);
        assert_eq!(decay_exp(-800.0), 0.0);
        assert_eq!(decay_exp(800.0), f64::INFINITY);
        assert_eq!(decay_exp(0.0), 1.0);
        assert_eq!(MAX_EXP_ARG, f64::MAX.ln());
        // The cutoff changes nothing: ln(1 + exp(-d)) is already 0 there.
        assert_eq!((1.0 + (-LOGADDEXP_CUTOFF).exp()).ln(), 0.0);
    }

    #[test]
    fn persist_to() {
        assert_eq!(PersistTo::parse("").unwrap(), PersistTo::Memory);
        assert_eq!(PersistTo::parse("Memory").unwrap(), PersistTo::Memory);
        assert_eq!(
            PersistTo::parse("native:/tmp/q").unwrap(),
            PersistTo::Native("/tmp/q".into())
        );
        let pg = PersistTo::parse("postgresql://u:pw@h:5432/db?sslmode=require").unwrap();
        assert_eq!(pg.to_string(), "postgresql://u:***@h:5432/db?sslmode=require");
        for bad in ["native:", "sqlite://x", "s3://b"] {
            assert!(matches!(PersistTo::parse(bad), Err(Error::Invalid(_))), "{bad}");
        }
        let e = pg.open().err().unwrap().to_string();
        assert!(e.contains("heapdb-postgres"), "{e}");
        assert_eq!(redact_url("postgres://h/db"), "postgres://h/db");
        assert_eq!(redact_url("postgres://u@h/db"), "postgres://u:***@h/db");
    }

    #[test]
    fn validation() {
        assert!(check_name("item id", "").is_err());
        assert!(check_name("item id", &"x".repeat(201)).is_err());
        assert!(check_name("item id", &"é".repeat(100)).is_ok());
        assert!(check_name("item id", "a\0b").is_err());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(check_positive("weight", bad).is_err());
        }
        let mut a = Attrs::new();
        a.insert("k".into(), serde_json::json!(["ok", {"x": "b\u{0}"}]));
        assert!(check_attrs(&a).is_err());
    }

    #[test]
    fn json_round_trips() {
        let mut m = MemoryHeaps::new();
        let info = m
            .create_heap(&HeapSpec::new("q", 60.0).landmark(0.0), 0.0)
            .unwrap();
        let it = m.push("q", "a", 1.0, 1e6, None).unwrap();
        let r = m.get("q", "a", 0.0).unwrap().unwrap();
        assert_eq!(r.score, f64::INFINITY);
        let back = |v: Value| v.as_object().unwrap().clone();
        assert_eq!(HeapInfo::from_json(&back(info.to_json())).unwrap(), info);
        assert_eq!(HeapItem::from_json(&back(it.to_json())).unwrap(), it);
        assert_eq!(Ranked::from_json(&back(r.to_json())).unwrap(), r);
        assert!(Ranked::from_json(&Map::new()).is_err());
    }
}
