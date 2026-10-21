//! A tiny query language for HeapDB.
//!
//! Grammar (keywords are case-insensitive; `[...]` is optional):
//!
//! ```text
//! TOP <k> CAUSES OF <effect> FOR <entity>[,<entity>...] [AT <time>]
//!     [WITHIN <duration>] [EXPLAIN]
//! CALIBRATION OF <effect> [BINS <n>]
//! RISK OF <effect> FOR <entity> FEATURES <f>[,<f>...] [WEIGHT <w>] [TYPE <t>]
//!     [ATTRS k=v[,k=v...]] [EXPLAIN]
//! FEEDBACK <effect> FOR <entity> AT <time> CONFIRMED <event_id>
//!          [REJECTED <id>[,<id>...]]
//! INSERT <type> FOR <entity> [AT <time>] [FEATURES <f>[,<f>...]] [WEIGHT <w>]
//!     [ATTRS k=v[,k=v...]]
//! OBSERVE <effect> FOR <entity> [AT <time>] [LEAK <x>]
//! SETTLE [<effect>] UNTIL <time> [FOR <entity>]
//! LINKS OF <effect> [FOR <entity>] [MIN <n>]
//! ```
//!
//! Heap statements (`CREATE HEAP`, `PUSH`, `PEEK`, `POP`, ...) are parsed by
//! [`crate::heaps::hql`] and run against a [`HeapBackend`] with
//! [`execute_with`].
//!
//! `<time>` is a number of seconds, `now`, `now-<duration>` or an ISO-8601
//! datetime; `<duration>` is a number with an optional unit `s`, `m`, `h`,
//! `d`. Statements are tokenised like Python's `shlex.split` (quotes work).

use std::collections::BTreeSet;
use std::fmt;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::event::Attrs;
use crate::heaps::hql::{self, HeapOutput, HeapStmt};
use crate::heaps::HeapBackend;
use crate::observe::LinkReportOptions;
use crate::pyfmt::repr_str;
use crate::results::{CalibrationReport, LinkReport, ObserveReport, QueryResult, RiskResult, SettleReport};
use crate::store::{Candidate, HeapDB, NewEvent, TopK};

/// Either an absolute time or `now - offset`, resolved at execution.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct TimeExpr {
    /// Absolute time (seconds), if given.
    pub absolute: Option<f64>,
    /// Offset before `now` (seconds) when `absolute` is `None`.
    pub now_offset: f64,
}

impl TimeExpr {
    /// `now`.
    pub const NOW: TimeExpr = TimeExpr {
        absolute: None,
        now_offset: 0.0,
    };

    /// Resolve against the clock.
    pub fn resolve(&self, now: f64) -> f64 {
        self.absolute.unwrap_or(now - self.now_offset)
    }
}

/// An attribute value parsed from `ATTRS k=v`: int, else float, else string.
pub type AttrPairs = Vec<(String, Value)>;

/// `TOP k CAUSES OF ...`.
#[derive(Clone, Debug, PartialEq)]
pub struct TopCausesQuery {
    /// k.
    pub k: usize,
    /// Effect.
    pub effect: String,
    /// Entity (comma-joined for groups).
    pub entity: String,
    /// Effect time.
    pub at: TimeExpr,
    /// Lookback cap.
    pub within: Option<f64>,
    /// Attach explanations.
    pub explain: bool,
    /// All entities named in `FOR` (more than one: group query).
    pub entities: Vec<String>,
}

impl TopCausesQuery {
    /// More than one entity.
    pub fn is_group(&self) -> bool {
        self.entities.len() > 1
    }
}

/// `CALIBRATION OF ...`.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationQuery {
    /// Effect.
    pub effect: String,
    /// Number of bins.
    pub bins: usize,
}

/// `RISK OF ...`.
#[derive(Clone, Debug, PartialEq)]
pub struct RiskQuery {
    /// Effect.
    pub effect: String,
    /// Entity.
    pub entity: String,
    /// Features.
    pub features: BTreeSet<String>,
    /// Dose (`None`: derived from attrs, else 1).
    pub weight: Option<f64>,
    /// Event type.
    pub event_type: Option<String>,
    /// Attributes.
    pub attrs: AttrPairs,
    /// Attach an explanation.
    pub explain: bool,
}

/// `FEEDBACK ...`.
#[derive(Clone, Debug, PartialEq)]
pub struct FeedbackStmt {
    /// Effect.
    pub effect: String,
    /// Entity.
    pub entity: String,
    /// Effect time.
    pub at: TimeExpr,
    /// Confirmed event id.
    pub confirmed: u64,
    /// Rejected event ids.
    pub rejected: Vec<u64>,
}

/// `INSERT ...`.
#[derive(Clone, Debug, PartialEq)]
pub struct InsertStmt {
    /// Event type.
    pub event_type: String,
    /// Entity.
    pub entity: String,
    /// Time.
    pub at: TimeExpr,
    /// Features.
    pub features: BTreeSet<String>,
    /// Dose.
    pub weight: Option<f64>,
    /// Attributes.
    pub attrs: AttrPairs,
}

/// `OBSERVE ...`.
#[derive(Clone, Debug, PartialEq)]
pub struct ObserveStmt {
    /// Effect.
    pub effect: String,
    /// Entity.
    pub entity: String,
    /// Occurrence time.
    pub at: TimeExpr,
    /// Leak override.
    pub leak: Option<f64>,
}

/// `SETTLE ...`.
#[derive(Clone, Debug, PartialEq)]
pub struct SettleStmt {
    /// Horizon.
    pub until: TimeExpr,
    /// One effect (None: all).
    pub effect: Option<String>,
    /// One entity (None: all).
    pub entity: Option<String>,
}

/// `LINKS OF ...`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinksQuery {
    /// Effect.
    pub effect: String,
    /// Entity view.
    pub entity: Option<String>,
    /// Minimum evidence.
    pub min_support: f64,
}

/// A parsed statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Statement {
    /// `TOP`
    Top(TopCausesQuery),
    /// `FEEDBACK`
    Feedback(FeedbackStmt),
    /// `INSERT`
    Insert(InsertStmt),
    /// `CALIBRATION`
    Calibration(CalibrationQuery),
    /// `RISK`
    Risk(RiskQuery),
    /// `OBSERVE`
    Observe(ObserveStmt),
    /// `SETTLE`
    Settle(SettleStmt),
    /// `LINKS`
    Links(LinksQuery),
    /// A heap statement (`CREATE HEAP`, `PUSH`, `PEEK`, `POP`, ...).
    Heap(HeapStmt),
}

/// The result of executing a statement.
#[derive(Clone, Debug)]
pub enum Output {
    /// `TOP`
    Query(QueryResult),
    /// `CALIBRATION`
    Calibration(CalibrationReport),
    /// `RISK`
    Risk(RiskResult),
    /// `LINKS`
    Links(LinkReport),
    /// `OBSERVE`
    Observe(ObserveReport),
    /// `SETTLE`
    Settle(SettleReport),
    /// `FEEDBACK` / `INSERT`: a message.
    Message(String),
    /// A heap statement.
    Heap(HeapOutput),
}

impl fmt::Display for Output {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Output::Query(r) => r.fmt(f),
            Output::Calibration(r) => r.fmt(f),
            Output::Risk(r) => r.fmt(f),
            Output::Links(r) => r.fmt(f),
            Output::Observe(r) => r.fmt(f),
            Output::Settle(r) => r.fmt(f),
            Output::Message(m) => f.write_str(m),
            Output::Heap(h) => h.fmt(f),
        }
    }
}

// ---------------------------------------------------------------------------
// Lexical helpers (Python-compatible number parsing)
// ---------------------------------------------------------------------------

fn strip_underscores(s: &str) -> Option<String> {
    // Python allows single underscores between digits only.
    if !s.contains('_') {
        return Some(s.to_string());
    }
    let b = s.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if c == b'_' {
            let ok = i > 0 && i + 1 < b.len() && b[i - 1].is_ascii_digit() && b[i + 1].is_ascii_digit();
            if !ok {
                return None;
            }
        }
    }
    Some(s.replace('_', ""))
}

/// Python `int(s)` for base-10 strings.
pub fn py_int(s: &str) -> Option<i64> {
    let t = s.trim();
    let t = strip_underscores(t)?;
    let digits = t.strip_prefix(['+', '-']).unwrap_or(&t);
    if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    t.parse().ok()
}

/// Python `float(s)`.
pub fn py_float(s: &str) -> Option<f64> {
    let t = s.trim();
    let (sign, body) = match t.as_bytes().first() {
        Some(b'-') => (-1.0, &t[1..]),
        Some(b'+') => (1.0, &t[1..]),
        _ => (1.0, t),
    };
    let low = body.to_ascii_lowercase();
    if low == "inf" || low == "infinity" {
        return Some(sign * f64::INFINITY);
    }
    if low == "nan" {
        return Some(f64::NAN);
    }
    let body = strip_underscores(body)?;
    let ok = !body.is_empty()
        && body
            .bytes()
            .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
        && body.bytes().any(|c| c.is_ascii_digit());
    if !ok {
        return None;
    }
    body.parse::<f64>().ok().map(|x| sign * x)
}

/// `"12h"` -> 43200. A bare number means seconds.
pub fn parse_duration(text: &str) -> Result<f64> {
    let t = text.trim();
    let bad = || {
        Error::query(format!(
            "bad duration {} (expected e.g. 90, 30m, 12h, 2d)",
            repr_str(text)
        ))
    };
    let (num, unit) = match t.chars().last() {
        Some(c) if matches!(c.to_ascii_lowercase(), 's' | 'm' | 'h' | 'd') => {
            (&t[..t.len() - 1], c.to_ascii_lowercase())
        }
        _ => (t, 's'),
    };
    // ^\d+(\.\d+)?$
    let mut parts = num.splitn(2, '.');
    let int = parts.next().unwrap_or("");
    let frac = parts.next();
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    if !digits(int) || frac.is_some_and(|f| !digits(f)) {
        return Err(bad());
    }
    let v: f64 = num.parse().map_err(|_| bad())?;
    let mult = match unit {
        's' => 1.0,
        'm' => 60.0,
        'h' => 3600.0,
        _ => 86400.0,
    };
    Ok(v * mult)
}

/// Parse a `<time>`: seconds, `now`, `now-<duration>` or ISO-8601.
pub fn parse_time(text: &str) -> Result<TimeExpr> {
    let t = text.trim();
    let low = t.to_lowercase();
    if low == "now" {
        return Ok(TimeExpr::NOW);
    }
    if low.starts_with("now-") {
        return Ok(TimeExpr {
            absolute: None,
            now_offset: parse_duration(&t[4..])?,
        });
    }
    if let Some(x) = py_float(t) {
        return Ok(TimeExpr {
            absolute: Some(x),
            now_offset: 0.0,
        });
    }
    if let Some(x) = iso_timestamp(t) {
        return Ok(TimeExpr {
            absolute: Some(x),
            now_offset: 0.0,
        });
    }
    Err(Error::query(format!(
        "bad time {} (expected seconds, now, now-2h or ISO-8601)",
        repr_str(text)
    )))
}

/// `datetime.fromisoformat(s).timestamp()`: naive times are local time.
pub fn iso_timestamp(s: &str) -> Option<f64> {
    use chrono::{FixedOffset, Local, NaiveDate, NaiveDateTime, NaiveTime, TimeZone};
    let s = s.trim();
    if !s.is_ascii() || s.len() < 8 {
        return None;
    }
    let (date, rest) = if s.len() >= 10 && s.as_bytes()[4] == b'-' {
        (NaiveDate::parse_from_str(&s[..10], "%Y-%m-%d").ok()?, &s[10..])
    } else {
        (NaiveDate::parse_from_str(&s[..8], "%Y%m%d").ok()?, &s[8..])
    };
    let (time, tz) = if rest.is_empty() {
        (NaiveTime::from_hms_opt(0, 0, 0)?, "")
    } else {
        let rest = &rest[1..]; // any single separator character
        let tz_at = rest.find(['Z', 'z', '+', '-']).unwrap_or(rest.len());
        let (tpart, tz) = rest.split_at(tz_at);
        (parse_iso_time(tpart)?, tz)
    };
    let ndt = NaiveDateTime::new(date, time);
    let ts = if tz.is_empty() {
        Local.from_local_datetime(&ndt).earliest()?.timestamp_micros()
    } else {
        let off = if tz.eq_ignore_ascii_case("z") {
            0
        } else {
            let sign = if tz.starts_with('-') { -1 } else { 1 };
            let d: String = tz[1..].chars().filter(|c| c.is_ascii_digit()).collect();
            let (h, m, sec) = match d.len() {
                2 => (d[..2].parse::<i32>().ok()?, 0, 0),
                4 => (d[..2].parse().ok()?, d[2..4].parse().ok()?, 0),
                6 => (d[..2].parse().ok()?, d[2..4].parse().ok()?, d[4..6].parse().ok()?),
                _ => return None,
            };
            sign * (h * 3600 + m * 60 + sec)
        };
        FixedOffset::east_opt(off)?
            .from_local_datetime(&ndt)
            .single()?
            .timestamp_micros()
    };
    Some(ts as f64 / 1e6)
}

fn parse_iso_time(t: &str) -> Option<chrono::NaiveTime> {
    let (main, frac) = match t.find(['.', ',']) {
        Some(i) => (&t[..i], Some(&t[i + 1..])),
        None => (t, None),
    };
    let d: String = main.chars().filter(|c| *c != ':').collect();
    if !d.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (h, m, s) = match d.len() {
        2 => (d[..2].parse().ok()?, 0, 0),
        4 => (d[..2].parse().ok()?, d[2..4].parse().ok()?, 0),
        6 => (d[..2].parse().ok()?, d[2..4].parse().ok()?, d[4..6].parse().ok()?),
        _ => return None,
    };
    let micros = match frac {
        None => 0,
        Some(f) if !f.is_empty() && f.bytes().all(|c| c.is_ascii_digit()) => {
            let f6: String = f.chars().chain(std::iter::repeat('0')).take(6).collect();
            f6.parse().ok()?
        }
        _ => return None,
    };
    chrono::NaiveTime::from_hms_micro_opt(h, m, s, micros)
}

/// `shlex.split` (POSIX mode, no comments).
pub fn shlex_split(s: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_word = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            ' ' | '\t' | '\r' | '\n' => {
                if in_word {
                    out.push(std::mem::take(&mut cur));
                    in_word = false;
                }
            }
            '\\' => {
                in_word = true;
                match chars.next() {
                    Some(n) => cur.push(n),
                    None => return Err(Error::query("No escaped character")),
                }
            }
            '\'' => {
                in_word = true;
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(n) => cur.push(n),
                        None => return Err(Error::query("No closing quotation")),
                    }
                }
            }
            '"' => {
                in_word = true;
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(n @ ('"' | '\\')) => cur.push(n),
                            Some(n) => {
                                cur.push('\\');
                                cur.push(n);
                            }
                            None => return Err(Error::query("No escaped character")),
                        },
                        Some(n) => cur.push(n),
                        None => return Err(Error::query("No closing quotation")),
                    }
                }
            }
            c => {
                in_word = true;
                cur.push(c);
            }
        }
    }
    if in_word {
        out.push(cur);
    }
    Ok(out)
}

struct Tokens {
    toks: Vec<String>,
    i: usize,
}

impl Tokens {
    fn new(text: &str) -> Result<Self> {
        let t = text.trim().trim_end_matches(';');
        Ok(Tokens {
            toks: shlex_split(t)?,
            i: 0,
        })
    }
    fn peek(&self) -> Option<&str> {
        self.toks.get(self.i).map(String::as_str)
    }
    fn peek_kw(&self, kws: &[&str]) -> bool {
        self.peek()
            .is_some_and(|p| kws.contains(&p.to_uppercase().as_str()))
    }
    fn next(&mut self, what: &str) -> Result<String> {
        match self.toks.get(self.i) {
            None => Err(Error::query(format!("unexpected end of query, expected {what}"))),
            Some(t) => {
                self.i += 1;
                Ok(t.clone())
            }
        }
    }
    fn expect(&mut self, kw: &str) -> Result<()> {
        let tok = self.next(kw)?;
        if tok.to_uppercase() != kw {
            return Err(Error::query(format!("expected {kw}, got {}", repr_str(&tok))));
        }
        Ok(())
    }
    fn done(&self) -> bool {
        self.i >= self.toks.len()
    }
}

fn int_tok(tok: &str, what: &str) -> Result<i64> {
    py_int(tok.trim_start_matches('#'))
        .ok_or_else(|| Error::query(format!("{what} must be an integer, got {}", repr_str(tok))))
}

fn id_tok(tok: &str) -> Result<u64> {
    let v = int_tok(tok, "event id")?;
    u64::try_from(v).map_err(|_| Error::query(format!("event id must be >= 0, got {}", repr_str(tok))))
}

fn id_list(tok: &str) -> Result<Vec<u64>> {
    tok.split(',').filter(|x| !x.is_empty()).map(id_tok).collect()
}

fn features_tok(tok: &str) -> BTreeSet<String> {
    tok.split(',')
        .filter(|f| !f.is_empty())
        .map(str::to_string)
        .collect()
}

/// `ATTRS` value: int, else float, else string.
pub fn attr_value(v: &str) -> Value {
    if let Some(i) = py_int(v) {
        return Value::from(i);
    }
    if let Some(x) = py_float(v) {
        return crate::snapshot::num(x);
    }
    Value::from(v)
}

fn parse_attrs(tok: &str) -> Result<AttrPairs> {
    let mut out = Vec::new();
    for part in tok.split(',') {
        if part.is_empty() {
            continue;
        }
        match part.split_once('=') {
            Some((k, v)) if !k.is_empty() => out.push((k.to_string(), attr_value(v))),
            _ => {
                return Err(Error::query(format!(
                    "bad attribute {} (expected key=value)",
                    repr_str(part)
                )))
            }
        }
    }
    Ok(out)
}

fn weight_tok(tok: &str) -> Result<f64> {
    py_float(tok).ok_or_else(|| Error::query("WEIGHT must be a number"))
}

fn number_tok(tok: &str, what: &str) -> Result<f64> {
    py_float(tok).ok_or_else(|| Error::query(format!("{what} must be a number, got {}", repr_str(tok))))
}

/// Parse one statement.
pub fn parse(text: &str) -> Result<Statement> {
    if hql::is_heap_statement(text) {
        return Ok(Statement::Heap(hql::parse(text)?));
    }
    let mut ts = Tokens::new(text)?;
    let head = ts.next("a statement")?.to_uppercase();
    let stmt = match head.as_str() {
        "TOP" => Statement::Top(parse_top(&mut ts)?),
        "FEEDBACK" => Statement::Feedback(parse_feedback(&mut ts)?),
        "INSERT" => Statement::Insert(parse_insert(&mut ts)?),
        "CALIBRATION" => Statement::Calibration(parse_calibration(&mut ts)?),
        "RISK" => Statement::Risk(parse_risk(&mut ts)?),
        "OBSERVE" => Statement::Observe(parse_observe(&mut ts)?),
        "SETTLE" => Statement::Settle(parse_settle(&mut ts)?),
        "LINKS" => Statement::Links(parse_links(&mut ts)?),
        _ => {
            return Err(Error::query(format!(
                "unknown statement {} (expected TOP, RISK, FEEDBACK, INSERT, CALIBRATION, OBSERVE, SETTLE, LINKS, or a heap statement: CREATE HEAP, PUSH, SET BASE, SET ATTRS, PEEK, POP, REMOVE, REBASE, DROP HEAP, SHOW HEAPS)",
                repr_str(&head)
            )))
        }
    };
    if !ts.done() {
        return Err(Error::query(format!(
            "unexpected trailing input: {}",
            repr_str(&ts.toks[ts.i..].join(" "))
        )));
    }
    Ok(stmt)
}

fn parse_top(ts: &mut Tokens) -> Result<TopCausesQuery> {
    let k = int_tok(&ts.next("k")?, "k")?;
    if k < 1 {
        return Err(Error::query("k must be >= 1"));
    }
    if !ts.peek_kw(&["CAUSE", "CAUSES"]) {
        let got = ts.peek().map_or("None".to_string(), repr_str);
        return Err(Error::query(format!("expected CAUSES, got {got}")));
    }
    ts.next("CAUSES")?;
    ts.expect("OF")?;
    let effect = ts.next("effect name")?;
    ts.expect("FOR")?;
    let entity_tok = ts.next("entity")?;
    let mut entities: Vec<String> = Vec::new();
    for e in entity_tok.split(',').filter(|e| !e.is_empty()) {
        if !entities.iter().any(|x| x == e) {
            entities.push(e.to_string());
        }
    }
    if entities.is_empty() {
        return Err(Error::query(format!("bad entity list {}", repr_str(&entity_tok))));
    }
    let (mut at, mut within, mut explain) = (TimeExpr::NOW, None, false);
    while !ts.done() {
        let kw = ts.next("clause")?.to_uppercase();
        match kw.as_str() {
            "AT" => at = parse_time(&ts.next("time")?)?,
            "WITHIN" => within = Some(parse_duration(&ts.next("duration")?)?),
            "EXPLAIN" => explain = true,
            _ => {
                return Err(Error::query(format!(
                    "unexpected {} (expected AT, WITHIN or EXPLAIN)",
                    repr_str(&kw)
                )))
            }
        }
    }
    Ok(TopCausesQuery {
        k: k as usize,
        effect,
        entity: entities.join(","),
        at,
        within,
        explain,
        entities,
    })
}

fn parse_calibration(ts: &mut Tokens) -> Result<CalibrationQuery> {
    ts.expect("OF")?;
    let effect = ts.next("effect name")?;
    let mut bins = 10;
    if ts.peek_kw(&["BINS"]) {
        ts.next("BINS")?;
        let b = int_tok(&ts.next("bins")?, "bins")?;
        if b < 1 {
            return Err(Error::query("BINS must be >= 1"));
        }
        bins = b as usize;
    }
    Ok(CalibrationQuery { effect, bins })
}

fn parse_risk(ts: &mut Tokens) -> Result<RiskQuery> {
    ts.expect("OF")?;
    let effect = ts.next("effect name")?;
    ts.expect("FOR")?;
    let entity = ts.next("entity")?;
    let mut q = RiskQuery {
        effect,
        entity,
        features: BTreeSet::new(),
        weight: None,
        event_type: None,
        attrs: Vec::new(),
        explain: false,
    };
    while !ts.done() {
        let kw = ts.next("clause")?.to_uppercase();
        match kw.as_str() {
            "FEATURES" => q.features = features_tok(&ts.next("features")?),
            "WEIGHT" => q.weight = Some(weight_tok(&ts.next("weight")?)?),
            "TYPE" => q.event_type = Some(ts.next("event type")?),
            "ATTRS" => q.attrs = parse_attrs(&ts.next("attributes")?)?,
            "EXPLAIN" => q.explain = true,
            _ => {
                return Err(Error::query(format!(
                    "unexpected {} (expected FEATURES, WEIGHT, TYPE, ATTRS or EXPLAIN)",
                    repr_str(&kw)
                )))
            }
        }
    }
    if q.features.is_empty() {
        return Err(Error::query("RISK needs FEATURES <f>[,<f>...]"));
    }
    Ok(q)
}

fn parse_feedback(ts: &mut Tokens) -> Result<FeedbackStmt> {
    let effect = ts.next("effect name")?;
    ts.expect("FOR")?;
    let entity = ts.next("entity")?;
    ts.expect("AT")?;
    let at = parse_time(&ts.next("time")?)?;
    ts.expect("CONFIRMED")?;
    let confirmed = id_tok(&ts.next("event id")?)?;
    let mut rejected = Vec::new();
    if ts.peek_kw(&["REJECTED"]) {
        ts.next("REJECTED")?;
        rejected = id_list(&ts.next("event ids")?)?;
    }
    Ok(FeedbackStmt {
        effect,
        entity,
        at,
        confirmed,
        rejected,
    })
}

fn parse_insert(ts: &mut Tokens) -> Result<InsertStmt> {
    let event_type = ts.next("event type")?;
    ts.expect("FOR")?;
    let entity = ts.next("entity")?;
    let mut s = InsertStmt {
        event_type,
        entity,
        at: TimeExpr::NOW,
        features: BTreeSet::new(),
        weight: None,
        attrs: Vec::new(),
    };
    while !ts.done() {
        let kw = ts.next("clause")?.to_uppercase();
        match kw.as_str() {
            "AT" => s.at = parse_time(&ts.next("time")?)?,
            "FEATURES" => s.features = features_tok(&ts.next("features")?),
            "WEIGHT" => s.weight = Some(weight_tok(&ts.next("weight")?)?),
            "ATTRS" => s.attrs = parse_attrs(&ts.next("attributes")?)?,
            _ => {
                return Err(Error::query(format!(
                    "unexpected {} (expected AT, FEATURES, WEIGHT or ATTRS)",
                    repr_str(&kw)
                )))
            }
        }
    }
    Ok(s)
}

fn parse_observe(ts: &mut Tokens) -> Result<ObserveStmt> {
    let effect = ts.next("effect name")?;
    ts.expect("FOR")?;
    let entity = ts.next("entity")?;
    let (mut at, mut leak) = (TimeExpr::NOW, None);
    while !ts.done() {
        let kw = ts.next("clause")?.to_uppercase();
        match kw.as_str() {
            "AT" => at = parse_time(&ts.next("time")?)?,
            "LEAK" => {
                let l = number_tok(&ts.next("leak")?, "LEAK")?;
                if l.is_nan() || l <= 0.0 {
                    return Err(Error::query("LEAK must be > 0"));
                }
                leak = Some(l);
            }
            _ => {
                return Err(Error::query(format!(
                    "unexpected {} (expected AT or LEAK)",
                    repr_str(&kw)
                )))
            }
        }
    }
    Ok(ObserveStmt {
        effect,
        entity,
        at,
        leak,
    })
}

fn parse_settle(ts: &mut Tokens) -> Result<SettleStmt> {
    let mut effect = None;
    if !ts.peek_kw(&["UNTIL"]) {
        effect = Some(ts.next("effect name or UNTIL")?);
    }
    ts.expect("UNTIL")?;
    let until = parse_time(&ts.next("time")?)?;
    let mut entity = None;
    if ts.peek_kw(&["FOR"]) {
        ts.next("FOR")?;
        entity = Some(ts.next("entity")?);
    }
    Ok(SettleStmt {
        until,
        effect,
        entity,
    })
}

fn parse_links(ts: &mut Tokens) -> Result<LinksQuery> {
    ts.expect("OF")?;
    let effect = ts.next("effect name")?;
    let (mut entity, mut min_support) = (None, 1.0);
    while !ts.done() {
        let kw = ts.next("clause")?.to_uppercase();
        match kw.as_str() {
            "FOR" => entity = Some(ts.next("entity")?),
            "MIN" => {
                min_support = number_tok(&ts.next("min support")?, "MIN")?;
                if min_support < 0.0 {
                    return Err(Error::query("MIN must be >= 0"));
                }
            }
            _ => {
                return Err(Error::query(format!(
                    "unexpected {} (expected FOR or MIN)",
                    repr_str(&kw)
                )))
            }
        }
    }
    Ok(LinksQuery {
        effect,
        entity,
        min_support,
    })
}

fn attrs_of(pairs: &AttrPairs) -> Attrs {
    let mut m = Attrs::new();
    for (k, v) in pairs {
        m.insert(k.clone(), v.clone());
    }
    m
}

/// Run a statement. `now` resolves `now` / `now-<d>` times. Heap
/// statements need a heap backend: use [`execute_with`].
pub fn execute(db: &mut HeapDB, stmt: &Statement, now: f64) -> Result<Output> {
    execute_inner(db, None, stmt, now)
}

/// [`execute`], running heap statements against `heaps`.
pub fn execute_with(
    db: &mut HeapDB,
    heaps: &mut dyn HeapBackend,
    stmt: &Statement,
    now: f64,
) -> Result<Output> {
    execute_inner(db, Some(heaps), stmt, now)
}

fn execute_inner(
    db: &mut HeapDB,
    heaps: Option<&mut dyn HeapBackend>,
    stmt: &Statement,
    now: f64,
) -> Result<Output> {
    Ok(match stmt {
        Statement::Heap(h) => match heaps {
            Some(backend) => Output::Heap(hql::execute(backend, h, now)?),
            None => {
                return Err(Error::query(
                    "heap statements need a heap backend (heapdb::query::execute_with, or heapdb.Heaps in Python)",
                ))
            }
        },
        Statement::Top(q) => {
            let opts = TopK {
                k: q.k,
                explain: q.explain,
                within: q.within,
                use_sets: true,
            };
            let at = q.at.resolve(now);
            if q.is_group() {
                let ents: Vec<&str> = q.entities.iter().map(String::as_str).collect();
                Output::Query(db.top_causes_group(&ents, &q.effect, at, opts)?)
            } else {
                Output::Query(db.top_causes(&q.entity, &q.effect, at, opts)?)
            }
        }
        Statement::Calibration(q) => Output::Calibration(db.calibration_report(&q.effect, q.bins)?),
        Statement::Risk(q) => {
            let c = Candidate {
                features: q.features.iter().cloned().collect(),
                weight: q.weight,
                event_type: q.event_type.clone(),
                attrs: attrs_of(&q.attrs),
                entity: Some(q.entity.clone()),
            };
            let (score, explanation) = db.feature_risk(&q.effect, &c, q.explain)?;
            Output::Risk(RiskResult {
                effect: q.effect.clone(),
                entity: Some(q.entity.clone()),
                features: q.features.iter().cloned().collect(),
                event_type: q.event_type.clone(),
                score,
                explanation,
            })
        }
        Statement::Feedback(s) => {
            db.feedback(
                &s.entity,
                &s.effect,
                s.at.resolve(now),
                Some(s.confirmed),
                &s.rejected,
                None,
            )?;
            let mut msg = format!("ok: confirmed #{}", s.confirmed);
            if !s.rejected.is_empty() {
                let r: Vec<String> = s.rejected.iter().map(|i| format!("#{i}")).collect();
                msg.push_str(&format!(", rejected {}", r.join(", ")));
            }
            Output::Message(msg)
        }
        Statement::Observe(s) => {
            Output::Observe(db.observe(&s.entity, &s.effect, s.at.resolve(now), s.leak)?)
        }
        Statement::Settle(s) => {
            let effs: Option<Vec<&str>> = s.effect.as_deref().map(|e| vec![e]);
            Output::Settle(db.settle(s.until.resolve(now), effs.as_deref(), s.entity.as_deref(), false)?)
        }
        Statement::Links(q) => Output::Links(db.link_report(
            &q.effect,
            &LinkReportOptions {
                entity: q.entity.clone(),
                min_support: q.min_support,
                ..Default::default()
            },
        )?),
        Statement::Insert(s) => {
            let ev = db.insert(NewEvent {
                entity: s.entity.clone(),
                time: s.at.resolve(now),
                event_type: s.event_type.clone(),
                features: s.features.iter().cloned().collect(),
                weight: s.weight,
                attrs: attrs_of(&s.attrs),
            })?;
            Output::Message(format!("ok: inserted {}", ev.short()))
        }
    })
}

/// Parse and run a statement.
pub fn execute_str(db: &mut HeapDB, text: &str, now: f64) -> Result<Output> {
    execute(db, &parse(text)?, now)
}

/// Parse and run a statement, with heap statements going to `heaps`.
pub fn execute_str_with(
    db: &mut HeapDB,
    heaps: &mut dyn HeapBackend,
    text: &str,
    now: f64,
) -> Result<Output> {
    execute_with(db, heaps, &parse(text)?, now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_forms() {
        let Statement::Top(q) = parse("TOP 3 CAUSES OF bloating FOR me AT now WITHIN 12h EXPLAIN").unwrap()
        else {
            panic!()
        };
        assert_eq!((q.k, q.within, q.explain), (3, Some(43200.0), true));
        let Statement::Top(q) = parse("top 1 cause of latency for svc-a at 12345.5;").unwrap() else {
            panic!()
        };
        assert_eq!(q.at.resolve(0.0), 12345.5);
        let Statement::Feedback(f) =
            parse("FEEDBACK bloating FOR me AT 7200 CONFIRMED 42 REJECTED 7,#8").unwrap()
        else {
            panic!()
        };
        assert_eq!((f.confirmed, f.rejected), (42, vec![7, 8]));
        assert_eq!(parse_duration("1.5d").unwrap(), 129600.0);
        assert_eq!(
            parse_time("2024-05-01T00:00:00+00:00").unwrap().resolve(0.0),
            1714521600.0
        );
        for bad in [
            "",
            "SELECT * FROM t",
            "TOP x CAUSES OF a FOR b",
            "TOP 0 CAUSES OF a FOR b",
            "TOP 3 THINGS OF a FOR b",
            "TOP 3 CAUSES OF a",
            "TOP 3 CAUSES OF a FOR b WITHIN forever",
            "TOP 3 CAUSES OF a FOR b AT yesterday",
            "TOP 3 CAUSES OF a FOR b EXPLAIN PLEASE",
            "FEEDBACK a FOR b AT now",
            "FEEDBACK a FOR b AT now CONFIRMED abc",
            "TOP 1 CAUSES OF x FOR ,",
            "INSERT t FOR a ATTRS amount",
        ] {
            assert!(matches!(parse(bad), Err(Error::Query(_))), "{bad}");
        }
    }

    #[test]
    fn shlex_quotes() {
        assert_eq!(
            shlex_split("a 'b c' \"d\\\"e\" f\\ g").unwrap(),
            vec!["a", "b c", "d\"e", "f g"]
        );
        assert!(shlex_split("'oops").is_err());
    }

    #[test]
    fn numbers() {
        assert_eq!(py_int(" +1_000 "), Some(1000));
        assert_eq!(py_int("1.5"), None);
        assert_eq!(py_float("1e3"), Some(1000.0));
        assert_eq!(py_float("-inf"), Some(f64::NEG_INFINITY));
        assert_eq!(py_float("abc"), None);
        assert_eq!(attr_value("EUR"), Value::from("EUR"));
        assert_eq!(attr_value("5"), Value::from(5));
    }
}
