//! Heap statements of the query language.
//!
//! ```text
//! CREATE HEAP <heap> HALF LIFE <duration> [LANDMARK <time>] [META {json}]
//! DROP HEAP <heap>
//! SHOW HEAPS
//! PUSH <id> INTO <heap> WEIGHT <w> [AT <time>] [ATTRS {json}]
//! SET BASE <id> IN <heap> TO <base>
//! SET ATTRS <id> IN <heap> TO {json} [MERGE]
//! PEEK <k> FROM <heap> [AT <time>] [WHERE <condition>]
//! POP FROM <heap> [AT <time>] [WHERE <condition>]
//! REMOVE <id> FROM <heap>
//! REBASE <heap> [AT <time>] [HALF LIFE <duration>]
//! ```
//!
//! Names and ids are bare words (`work`, `CHK-12`) or single-quoted
//! strings (`'acme/checkout'`, `''` escapes a quote). Conditions:
//!
//! ```text
//! cond  := and (OR and)*          and := unary (AND unary)*
//! unary := NOT unary | ( cond ) | pred
//! pred  := attr (= | != | <> | < | <= | > | >=) value
//!        | attr [NOT] IN (values)       | attr CONTAINS value
//!        | attr CONTAINS ALL (values)   | attr SUBSET OF (values)
//!        | attr EXISTS                  | attr MISSING
//! value := 'string' | number | TRUE | FALSE
//! ```
//!
//! They compile to the JSON [`Filter`]s of `docs/heaps.md`.

use std::fmt;

use serde_json::Value;

use super::filter::{Filter, LeafOp};
use super::{HeapBackend, HeapInfo, HeapSpec, HeapTop};
use crate::error::{Error, Result};
use crate::event::Attrs;
use crate::pyfmt::{repr_f64, repr_str};
use crate::query::{parse_duration, parse_time, py_float, py_int, TimeExpr};

/// The keywords that start a heap statement.
pub const HEAP_KEYWORDS: [&str; 9] = [
    "CREATE", "DROP", "SHOW", "PUSH", "SET", "PEEK", "POP", "REMOVE", "REBASE",
];

/// Which API scope a statement needs on heapdb-server.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// `PEEK`, `SHOW HEAPS`.
    Read,
    /// `PUSH`, `SET`, `POP`, `REMOVE`.
    Write,
    /// `CREATE`, `DROP`, `REBASE`.
    Admin,
}

/// A parsed heap statement.
#[derive(Clone, Debug, PartialEq)]
pub enum HeapStmt {
    /// `CREATE HEAP`
    Create {
        /// Heap.
        heap: String,
        /// Half-life (seconds).
        half_life: f64,
        /// Landmark (default: now).
        landmark: Option<TimeExpr>,
        /// Metadata.
        meta: Option<Attrs>,
    },
    /// `DROP HEAP`
    Drop {
        /// Heap.
        heap: String,
    },
    /// `SHOW HEAPS`
    Show,
    /// `PUSH`
    Push {
        /// Heap.
        heap: String,
        /// Item id.
        id: String,
        /// Weight.
        weight: f64,
        /// Time (default now).
        at: TimeExpr,
        /// Replacement attributes.
        attrs: Option<Attrs>,
    },
    /// `SET BASE`
    SetBase {
        /// Heap.
        heap: String,
        /// Item id.
        id: String,
        /// Base.
        base: f64,
    },
    /// `SET ATTRS`
    SetAttrs {
        /// Heap.
        heap: String,
        /// Item id.
        id: String,
        /// Attributes.
        attrs: Attrs,
        /// Shallow merge instead of replace.
        merge: bool,
    },
    /// `PEEK`
    Peek {
        /// Heap.
        heap: String,
        /// k.
        k: usize,
        /// Clock (default now).
        at: TimeExpr,
        /// Condition.
        filter: Option<Filter>,
    },
    /// `POP`
    Pop {
        /// Heap.
        heap: String,
        /// Clock (default now).
        at: TimeExpr,
        /// Condition.
        filter: Option<Filter>,
    },
    /// `REMOVE`
    Remove {
        /// Heap.
        heap: String,
        /// Item id.
        id: String,
    },
    /// `REBASE`
    Rebase {
        /// Heap.
        heap: String,
        /// New landmark (default now).
        at: TimeExpr,
        /// New half-life.
        half_life: Option<f64>,
    },
}

impl HeapStmt {
    /// The scope the statement needs.
    pub fn access(&self) -> Access {
        match self {
            HeapStmt::Peek { .. } | HeapStmt::Show => Access::Read,
            HeapStmt::Push { .. }
            | HeapStmt::SetBase { .. }
            | HeapStmt::SetAttrs { .. }
            | HeapStmt::Pop { .. }
            | HeapStmt::Remove { .. } => Access::Write,
            HeapStmt::Create { .. } | HeapStmt::Drop { .. } | HeapStmt::Rebase { .. } => Access::Admin,
        }
    }

    /// True unless the statement only reads.
    pub fn is_mutating(&self) -> bool {
        self.access() != Access::Read
    }
}

/// The result of a heap statement.
#[derive(Clone, Debug, PartialEq)]
pub enum HeapOutput {
    /// `PEEK` / `POP`.
    Top(HeapTop),
    /// `CREATE HEAP` / `REBASE`.
    Info(HeapInfo),
    /// `SHOW HEAPS`.
    List(Vec<HeapInfo>),
    /// Everything else.
    Message(String),
}

impl fmt::Display for HeapOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeapOutput::Top(t) => t.fmt(f),
            HeapOutput::Info(i) => write!(f, "ok: {i}"),
            HeapOutput::List(v) if v.is_empty() => f.write_str("(no heaps)"),
            HeapOutput::List(v) => {
                let lines: Vec<String> = v.iter().map(ToString::to_string).collect();
                f.write_str(&lines.join("\n"))
            }
            HeapOutput::Message(m) => f.write_str(m),
        }
    }
}

/// Run a heap statement against a backend. `now` resolves `now` times.
pub fn execute(h: &mut dyn HeapBackend, stmt: &HeapStmt, now: f64) -> Result<HeapOutput> {
    Ok(match stmt {
        HeapStmt::Create {
            heap,
            half_life,
            landmark,
            meta,
        } => {
            let spec = HeapSpec {
                name: heap.clone(),
                half_life: *half_life,
                landmark: landmark.map(|l| l.resolve(now)),
                meta: meta.clone(),
            };
            HeapOutput::Info(h.create_heap(&spec, now)?)
        }
        HeapStmt::Drop { heap } => HeapOutput::Message(if h.drop_heap(heap)? {
            format!("ok: dropped heap {}", repr_str(heap))
        } else {
            format!("ok: no heap {}", repr_str(heap))
        }),
        HeapStmt::Show => HeapOutput::List(h.heaps()?),
        HeapStmt::Push {
            heap,
            id,
            weight,
            at,
            attrs,
        } => {
            let it = h.push(heap, id, *weight, at.resolve(now), attrs.as_ref())?;
            HeapOutput::Message(format!("ok: pushed {it}"))
        }
        HeapStmt::SetBase { heap, id, base } => {
            let it = h.set_base(heap, id, *base)?;
            HeapOutput::Message(format!("ok: {it}"))
        }
        HeapStmt::SetAttrs {
            heap,
            id,
            attrs,
            merge,
        } => {
            let it = h.set_attrs(heap, id, attrs, *merge)?;
            HeapOutput::Message(format!("ok: {it}"))
        }
        HeapStmt::Peek { heap, k, at, filter } => {
            let t = at.resolve(now);
            HeapOutput::Top(HeapTop {
                heap: heap.clone(),
                now: t,
                popped: false,
                items: h.top_k(heap, *k, t, filter.as_ref())?,
            })
        }
        HeapStmt::Pop { heap, at, filter } => {
            let t = at.resolve(now);
            HeapOutput::Top(HeapTop {
                heap: heap.clone(),
                now: t,
                popped: true,
                items: h.pop(heap, t, filter.as_ref())?.into_iter().collect(),
            })
        }
        HeapStmt::Remove { heap, id } => HeapOutput::Message(if h.remove(heap, id)? {
            format!("ok: removed {} from {}", repr_str(id), repr_str(heap))
        } else {
            format!("ok: {} was not in {}", repr_str(id), repr_str(heap))
        }),
        HeapStmt::Rebase { heap, at, half_life } => {
            HeapOutput::Info(h.rebase(heap, at.resolve(now), *half_life)?)
        }
    })
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    /// A bare word (keyword, name, number, `now-2h`).
    Word(String),
    /// `'...'`.
    Str(String),
    /// `{...}` JSON object text.
    Json(String),
    /// `( ) , = != <> < <= > >=`.
    Punct(&'static str),
}

impl Tok {
    fn describe(&self) -> String {
        match self {
            Tok::Word(w) => repr_str(w),
            Tok::Str(s) => format!("string {}", repr_str(s)),
            Tok::Json(_) => "a JSON object".into(),
            Tok::Punct(p) => repr_str(p),
        }
    }
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '.' | ':' | '/' | '@' | '+' | '#')
}

fn lex(text: &str) -> Result<Vec<Tok>> {
    let mut out = Vec::new();
    let cs: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() || (c == ';' && cs[i + 1..].iter().all(|c| c.is_whitespace() || *c == ';')) {
            i += 1;
            continue;
        }
        match c {
            '\'' => {
                let mut s = String::new();
                i += 1;
                loop {
                    match cs.get(i) {
                        None => return Err(Error::query("unterminated string (missing closing ')")),
                        Some('\'') if cs.get(i + 1) == Some(&'\'') => {
                            s.push('\'');
                            i += 2;
                        }
                        Some('\'') => {
                            i += 1;
                            break;
                        }
                        Some(&ch) => {
                            s.push(ch);
                            i += 1;
                        }
                    }
                }
                out.push(Tok::Str(s));
            }
            '{' => {
                // Capture a balanced JSON object, skipping braces in strings.
                let start = i;
                let (mut depth, mut in_str, mut esc) = (0usize, false, false);
                while i < cs.len() {
                    let ch = cs[i];
                    i += 1;
                    if in_str {
                        match ch {
                            _ if esc => esc = false,
                            '\\' => esc = true,
                            '"' => in_str = false,
                            _ => {}
                        }
                        continue;
                    }
                    match ch {
                        '"' => in_str = true,
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if depth != 0 {
                    return Err(Error::query("unterminated JSON object (missing closing })"));
                }
                out.push(Tok::Json(cs[start..i].iter().collect()));
            }
            '(' | ')' | ',' | '=' => {
                out.push(Tok::Punct(match c {
                    '(' => "(",
                    ')' => ")",
                    ',' => ",",
                    _ => "=",
                }));
                i += 1;
            }
            '!' | '<' | '>' => {
                let next = cs.get(i + 1).copied();
                let (p, n) = match (c, next) {
                    ('!', Some('=')) => ("!=", 2),
                    ('<', Some('>')) => ("!=", 2),
                    ('<', Some('=')) => ("<=", 2),
                    ('>', Some('=')) => (">=", 2),
                    ('<', _) => ("<", 1),
                    ('>', _) => (">", 1),
                    _ => return Err(Error::query("unexpected '!' (did you mean !=?)")),
                };
                out.push(Tok::Punct(p));
                i += n;
            }
            c if is_word_char(c) => {
                let start = i;
                while i < cs.len() && is_word_char(cs[i]) {
                    i += 1;
                }
                out.push(Tok::Word(cs[start..i].iter().collect()));
            }
            c => {
                return Err(Error::query(format!(
                    "unexpected character {}",
                    repr_str(&c.to_string())
                )))
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

struct P {
    toks: Vec<Tok>,
    i: usize,
}

impl P {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i)
    }

    fn peek_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Word(w)) if w.eq_ignore_ascii_case(kw))
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        if self.peek_kw(kw) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn next(&mut self, what: &str) -> Result<Tok> {
        let t = self
            .toks
            .get(self.i)
            .cloned()
            .ok_or_else(|| Error::query(format!("unexpected end of query, expected {what}")))?;
        self.i += 1;
        Ok(t)
    }

    fn expect_kw(&mut self, kw: &str) -> Result<()> {
        match self.next(kw)? {
            Tok::Word(w) if w.eq_ignore_ascii_case(kw) => Ok(()),
            t => Err(Error::query(format!("expected {kw}, got {}", t.describe()))),
        }
    }

    fn expect_punct(&mut self, p: &str) -> Result<()> {
        match self.next(p)? {
            Tok::Punct(q) if q == p => Ok(()),
            t => Err(Error::query(format!(
                "expected {}, got {}",
                repr_str(p),
                t.describe()
            ))),
        }
    }

    /// A heap name or item id: a bare word or a quoted string.
    fn name(&mut self, what: &str) -> Result<String> {
        match self.next(what)? {
            Tok::Word(w) | Tok::Str(w) => Ok(w),
            t => Err(Error::query(format!("expected {what}, got {}", t.describe()))),
        }
    }

    fn word(&mut self, what: &str) -> Result<String> {
        match self.next(what)? {
            Tok::Word(w) | Tok::Str(w) => Ok(w),
            t => Err(Error::query(format!("expected {what}, got {}", t.describe()))),
        }
    }

    fn number(&mut self, what: &str) -> Result<f64> {
        let w = self.word(what)?;
        py_float(&w).ok_or_else(|| Error::query(format!("{what} must be a number, got {}", repr_str(&w))))
    }

    fn json(&mut self, what: &str) -> Result<Attrs> {
        match self.next(what)? {
            Tok::Json(s) => match serde_json::from_str::<Value>(&s) {
                Ok(Value::Object(m)) => Ok(m),
                Ok(_) => Err(Error::query(format!("{what} must be a JSON object"))),
                Err(e) => Err(Error::query(format!("{what} is not valid JSON: {e}"))),
            },
            t => Err(Error::query(format!(
                "expected {what} as {{json}}, got {}",
                t.describe()
            ))),
        }
    }

    fn time(&mut self) -> Result<TimeExpr> {
        parse_time(&self.word("time")?)
    }

    fn duration(&mut self) -> Result<f64> {
        parse_duration(&self.word("duration")?)
    }

    fn done(&self) -> bool {
        self.i >= self.toks.len()
    }

    fn unexpected(&self, expected: &str) -> Error {
        match self.peek() {
            Some(t) => Error::query(format!("unexpected {} (expected {expected})", t.describe())),
            None => Error::query(format!("unexpected end of query, expected {expected}")),
        }
    }
}

/// True if `text` starts with a heap statement keyword.
pub fn is_heap_statement(text: &str) -> bool {
    let first: String = text
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphabetic())
        .collect();
    HEAP_KEYWORDS.iter().any(|k| k.eq_ignore_ascii_case(&first))
}

/// Parse one heap statement.
pub fn parse(text: &str) -> Result<HeapStmt> {
    let mut p = P {
        toks: lex(text)?,
        i: 0,
    };
    let head = p.word("a statement")?.to_uppercase();
    let stmt = match head.as_str() {
        "CREATE" => {
            p.expect_kw("HEAP")?;
            let heap = p.name("heap name")?;
            let (mut half_life, mut landmark, mut meta) = (None, None, None);
            while !p.done() {
                if p.eat_kw("HALF") {
                    p.expect_kw("LIFE")?;
                    half_life = Some(p.duration()?);
                } else if p.eat_kw("LANDMARK") {
                    landmark = Some(p.time()?);
                } else if p.eat_kw("META") {
                    meta = Some(p.json("META")?);
                } else {
                    return Err(p.unexpected("HALF LIFE, LANDMARK or META"));
                }
            }
            let half_life =
                half_life.ok_or_else(|| Error::query("CREATE HEAP needs HALF LIFE <duration>"))?;
            HeapStmt::Create {
                heap,
                half_life,
                landmark,
                meta,
            }
        }
        "DROP" => {
            p.expect_kw("HEAP")?;
            HeapStmt::Drop {
                heap: p.name("heap name")?,
            }
        }
        "SHOW" => {
            p.expect_kw("HEAPS")?;
            HeapStmt::Show
        }
        "PUSH" => {
            let id = p.name("item id")?;
            p.expect_kw("INTO")?;
            let heap = p.name("heap name")?;
            let (mut weight, mut at, mut attrs) = (None, TimeExpr::NOW, None);
            while !p.done() {
                if p.eat_kw("WEIGHT") {
                    weight = Some(p.number("WEIGHT")?);
                } else if p.eat_kw("AT") {
                    at = p.time()?;
                } else if p.eat_kw("ATTRS") {
                    attrs = Some(p.json("ATTRS")?);
                } else {
                    return Err(p.unexpected("WEIGHT, AT or ATTRS"));
                }
            }
            HeapStmt::Push {
                heap,
                id,
                weight: weight.ok_or_else(|| Error::query("PUSH needs WEIGHT <w>"))?,
                at,
                attrs,
            }
        }
        "SET" => {
            let what = p.word("BASE or ATTRS")?.to_uppercase();
            let id = p.name("item id")?;
            p.expect_kw("IN")?;
            let heap = p.name("heap name")?;
            p.expect_kw("TO")?;
            match what.as_str() {
                "BASE" => HeapStmt::SetBase {
                    heap,
                    id,
                    base: p.number("base")?,
                },
                "ATTRS" => {
                    let attrs = p.json("ATTRS")?;
                    let merge = p.eat_kw("MERGE");
                    HeapStmt::SetAttrs {
                        heap,
                        id,
                        attrs,
                        merge,
                    }
                }
                _ => {
                    return Err(Error::query(format!(
                        "expected SET BASE or SET ATTRS, got SET {what}"
                    )))
                }
            }
        }
        "PEEK" => {
            let kt = p.word("k")?;
            let k = py_int(&kt)
                .filter(|k| *k >= 1)
                .ok_or_else(|| Error::query(format!("k must be an integer >= 1, got {}", repr_str(&kt))))?;
            p.expect_kw("FROM")?;
            let heap = p.name("heap name")?;
            let (at, filter) = at_where(&mut p)?;
            HeapStmt::Peek {
                heap,
                k: k as usize,
                at,
                filter,
            }
        }
        "POP" => {
            p.expect_kw("FROM")?;
            let heap = p.name("heap name")?;
            let (at, filter) = at_where(&mut p)?;
            HeapStmt::Pop { heap, at, filter }
        }
        "REMOVE" => {
            let id = p.name("item id")?;
            p.expect_kw("FROM")?;
            HeapStmt::Remove {
                heap: p.name("heap name")?,
                id,
            }
        }
        "REBASE" => {
            let heap = p.name("heap name")?;
            let (mut at, mut half_life) = (TimeExpr::NOW, None);
            while !p.done() {
                if p.eat_kw("AT") {
                    at = p.time()?;
                } else if p.eat_kw("HALF") {
                    p.expect_kw("LIFE")?;
                    half_life = Some(p.duration()?);
                } else {
                    return Err(p.unexpected("AT or HALF LIFE"));
                }
            }
            HeapStmt::Rebase { heap, at, half_life }
        }
        _ => {
            return Err(Error::query(format!(
                "unknown heap statement {}",
                repr_str(&head)
            )))
        }
    };
    if !p.done() {
        return Err(p.unexpected("end of statement"));
    }
    Ok(stmt)
}

fn at_where(p: &mut P) -> Result<(TimeExpr, Option<Filter>)> {
    let (mut at, mut filter) = (TimeExpr::NOW, None);
    while !p.done() {
        if p.eat_kw("AT") {
            at = p.time()?;
        } else if p.eat_kw("WHERE") {
            filter = Some(parse_or(p)?);
            // WHERE must come last.
            if !p.done() {
                return Err(p.unexpected("AND, OR or end of statement"));
            }
        } else {
            return Err(p.unexpected("AT or WHERE"));
        }
    }
    Ok((at, filter))
}

/// Parse the text of a `WHERE` condition on its own.
pub fn parse_condition(text: &str) -> Result<Filter> {
    let mut p = P {
        toks: lex(text)?,
        i: 0,
    };
    let f = parse_or(&mut p)?;
    if !p.done() {
        return Err(p.unexpected("AND, OR or end of condition"));
    }
    Ok(f)
}

fn parse_or(p: &mut P) -> Result<Filter> {
    let mut parts = vec![parse_and(p)?];
    while p.eat_kw("OR") {
        parts.push(parse_and(p)?);
    }
    Ok(if parts.len() == 1 {
        parts.remove(0)
    } else {
        Filter::Any(parts)
    })
}

fn parse_and(p: &mut P) -> Result<Filter> {
    let mut parts = vec![parse_unary(p)?];
    while p.eat_kw("AND") {
        parts.push(parse_unary(p)?);
    }
    Ok(if parts.len() == 1 {
        parts.remove(0)
    } else {
        Filter::All(parts)
    })
}

fn parse_unary(p: &mut P) -> Result<Filter> {
    if p.eat_kw("NOT") {
        return Ok(Filter::Not(Box::new(parse_unary(p)?)));
    }
    if p.peek() == Some(&Tok::Punct("(")) {
        p.i += 1;
        let f = parse_or(p)?;
        p.expect_punct(")")?;
        return Ok(f);
    }
    parse_pred(p)
}

fn value(p: &mut P) -> Result<Value> {
    match p.next("a value")? {
        Tok::Str(s) => Ok(Value::from(s)),
        Tok::Word(w) if w.eq_ignore_ascii_case("true") => Ok(Value::Bool(true)),
        Tok::Word(w) if w.eq_ignore_ascii_case("false") => Ok(Value::Bool(false)),
        Tok::Word(w) => {
            if let Some(i) = py_int(&w) {
                Ok(Value::from(i))
            } else if let Some(x) = py_float(&w).filter(|x| x.is_finite()) {
                Ok(Value::from(x))
            } else {
                Err(Error::query(format!(
                    "expected a value ('string', number, TRUE or FALSE), got {} (quote strings: '{w}')",
                    repr_str(&w)
                )))
            }
        }
        t => Err(Error::query(format!("expected a value, got {}", t.describe()))),
    }
}

fn value_list(p: &mut P) -> Result<Value> {
    p.expect_punct("(")?;
    let mut out = Vec::new();
    if p.peek() == Some(&Tok::Punct(")")) {
        p.i += 1;
        return Ok(Value::Array(out));
    }
    loop {
        out.push(value(p)?);
        match p.next(", or )")? {
            Tok::Punct(",") => continue,
            Tok::Punct(")") => break,
            t => return Err(Error::query(format!("expected , or ), got {}", t.describe()))),
        }
    }
    Ok(Value::Array(out))
}

fn parse_pred(p: &mut P) -> Result<Filter> {
    let attr = match p.next("an attribute")? {
        Tok::Word(w) | Tok::Str(w) => w,
        t => {
            return Err(Error::query(format!(
                "expected an attribute name, got {}",
                t.describe()
            )))
        }
    };
    let leaf = |op, v| Ok(Filter::leaf(attr.clone(), op, v));
    match p.next("an operator")? {
        Tok::Punct(op) => {
            let op = match op {
                "=" => LeafOp::Eq,
                "!=" => LeafOp::Ne,
                "<" => LeafOp::Lt,
                "<=" => LeafOp::Lte,
                ">" => LeafOp::Gt,
                ">=" => LeafOp::Gte,
                other => return Err(Error::query(format!("unexpected {} after {}", repr_str(other), repr_str(&attr)))),
            };
            let v = value(p)?;
            if matches!(op, LeafOp::Lt | LeafOp::Lte | LeafOp::Gt | LeafOp::Gte) && v.is_boolean() {
                return Err(Error::query("<, <=, >, >= compare numbers or strings"));
            }
            leaf(op, Some(v))
        }
        Tok::Word(w) => match w.to_uppercase().as_str() {
            "IN" => leaf(LeafOp::In, Some(value_list(p)?)),
            "NOT" => {
                p.expect_kw("IN")?;
                leaf(LeafOp::NotIn, Some(value_list(p)?))
            }
            "CONTAINS" => {
                if p.eat_kw("ALL") {
                    leaf(LeafOp::ContainsAll, Some(value_list(p)?))
                } else {
                    leaf(LeafOp::Contains, Some(value(p)?))
                }
            }
            "SUBSET" => {
                p.expect_kw("OF")?;
                leaf(LeafOp::SubsetOf, Some(value_list(p)?))
            }
            "EXISTS" => leaf(LeafOp::Exists, None),
            "MISSING" => leaf(LeafOp::Missing, None),
            _ => Err(Error::query(format!(
                "unknown operator {} after {} (expected =, !=, <, <=, >, >=, IN, NOT IN, CONTAINS, CONTAINS ALL, SUBSET OF, EXISTS or MISSING)",
                repr_str(&w),
                repr_str(&attr)
            ))),
        },
        t => Err(Error::query(format!("expected an operator after {}, got {}", repr_str(&attr), t.describe()))),
    }
}

impl fmt::Display for HeapStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Debug-ish but compact; used in reprs only.
        match self {
            HeapStmt::Create { heap, half_life, .. } => {
                write!(
                    f,
                    "CREATE HEAP {} HALF LIFE {}",
                    repr_str(heap),
                    repr_f64(*half_life)
                )
            }
            other => write!(f, "{other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heaps::MemoryHeaps;
    use serde_json::json;

    #[test]
    fn doc_statements() {
        let mut m = MemoryHeaps::new();
        let now = 1_760_000_000.0;
        let mut run = |q: &str| execute(&mut m, &parse(q).unwrap(), now).unwrap().to_string();
        assert!(run("CREATE HEAP work HALF LIFE 72h;").starts_with("ok: heap 'work': half_life=259200"));
        assert!(run(
            r#"PUSH 'CHK-12' INTO work WEIGHT 1.5 ATTRS {"kind": "bug", "required_skills": ["ts"]};"#
        )
        .starts_with("ok: pushed 'CHK-12'"));
        run(r#"PUSH CHK-13 INTO work WEIGHT 1 AT now-1h ATTRS {"kind": "task", "labels": ["needs-human"]}"#);
        assert!(run("SET BASE 'CHK-12' IN work TO 4;").contains("base=4"));
        let peek = run(
            "PEEK 5 FROM work WHERE kind IN ('bug', 'task') AND required_skills SUBSET OF ('ts', 'react');",
        );
        assert!(
            peek.starts_with("Top of heap 'work' at t=2025-10-09T08:53:20Z:\n  1. 'CHK-12'"),
            "{peek}"
        );
        assert!(
            peek.contains("2. 'CHK-13'"),
            "missing attribute counts as [] for SUBSET OF: {peek}"
        );
        let pop = run("POP FROM work WHERE NOT labels CONTAINS 'needs-human';");
        assert!(
            pop.starts_with("Popped from heap 'work'") && pop.contains("'CHK-12'"),
            "{pop}"
        );
        assert!(run("POP FROM work WHERE kind = 'bug'").contains("(nothing to pop)"));
        assert_eq!(
            run("REMOVE 'CHK-13' FROM work;"),
            "ok: removed 'CHK-13' from 'work'"
        );
        assert!(run("REBASE work;").starts_with("ok: heap 'work'"));
        assert!(run("SHOW HEAPS").starts_with("heap 'work'"));
        assert_eq!(run("DROP HEAP work"), "ok: dropped heap 'work'");
    }

    #[test]
    fn conditions() {
        let c = |s: &str| parse_condition(s).unwrap().to_json();
        assert_eq!(
            c("a = 1 OR b != 'x' AND NOT (c >= 2.5 OR d EXISTS)"),
            json!({"any": [
                {"attr": "a", "op": "eq", "value": 1},
                {"all": [
                    {"attr": "b", "op": "ne", "value": "x"},
                    {"not": {"any": [{"attr": "c", "op": "gte", "value": 2.5}, {"attr": "d", "op": "exists"}]}}
                ]}
            ]})
        );
        assert_eq!(
            c("tags CONTAINS ALL ('a', 'b') AND x NOT IN () AND y MISSING AND z <> true AND w<=3"),
            json!({"all": [
                {"attr": "tags", "op": "contains_all", "value": ["a", "b"]},
                {"attr": "x", "op": "not_in", "value": []},
                {"attr": "y", "op": "missing"},
                {"attr": "z", "op": "ne", "value": true},
                {"attr": "w", "op": "lte", "value": 3}
            ]})
        );
        assert_eq!(
            c("name = 'it''s'"),
            json!({"attr": "name", "op": "eq", "value": "it's"})
        );
        for (bad, msg) in [
            ("kind = bug", "quote strings"),
            ("kind LIKE 'x'", "unknown operator 'LIKE'"),
            ("(a = 1", "expected )"),
            ("a IN 'x'", "expected '('"),
            ("a = 1 b = 2", "unexpected 'b'"),
            ("a > true", "compare numbers or strings"),
            ("a = 'x", "unterminated string"),
        ] {
            let e = parse_condition(bad).unwrap_err().to_string();
            assert!(e.contains(msg), "{bad}: {e}");
        }
    }

    #[test]
    fn parse_errors() {
        for (bad, msg) in [
            ("CREATE HEAP w", "HALF LIFE"),
            ("CREATE HEAP w HALF LIFE forever", "bad duration"),
            ("PUSH a INTO w", "needs WEIGHT"),
            ("PUSH a INTO w WEIGHT x", "WEIGHT must be a number"),
            ("PUSH a INTO w WEIGHT 1 ATTRS [1]", "unexpected character"),
            ("PUSH a INTO w WEIGHT 1 ATTRS {\"a\": }", "not valid JSON"),
            ("PEEK 0 FROM w", "k must be an integer >= 1"),
            ("PEEK 3 FROM w WHERE a = 1 AT now", "unexpected 'AT'"),
            ("SET COLOR a IN w TO 1", "SET BASE or SET ATTRS"),
            ("REBASE w SOON", "AT or HALF LIFE"),
            ("DROP TABLE w", "expected HEAP"),
        ] {
            let e = parse(bad).unwrap_err();
            assert!(matches!(e, Error::Query(_)), "{bad}");
            assert!(e.to_string().contains(msg), "{bad}: {e}");
        }
        let HeapStmt::SetAttrs { merge, attrs, .. } =
            parse(r#"set attrs 'x y' in 'acme/q' to {"a": "}"} merge"#).unwrap()
        else {
            panic!()
        };
        assert!(merge);
        assert_eq!(attrs["a"], json!("}"));
        assert!(is_heap_statement("  peek 3 from w") && !is_heap_statement("TOP 3 CAUSES OF x FOR y"));
        assert_eq!(parse("POP FROM w").unwrap().access(), Access::Write);
        assert_eq!(parse("PEEK 1 FROM w").unwrap().access(), Access::Read);
        assert_eq!(parse("REBASE w HALF LIFE 1d").unwrap().access(), Access::Admin);
    }
}
