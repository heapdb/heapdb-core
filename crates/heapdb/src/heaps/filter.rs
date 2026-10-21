//! Filters on item attributes: a JSON tree of `{"all": [...]}`,
//! `{"any": [...]}`, `{"not": node}` and leaves `{"attr", "op", "value"}`.
//!
//! Leaf semantics (`docs/heaps.md`). A null attribute counts as missing.
//!
//! | op | true when |
//! | --- | --- |
//! | `eq`, `ne` | JSON equality (numbers compared numerically, at any depth) |
//! | `lt`, `lte`, `gt`, `gte` | both numbers, or both strings (byte-wise); mixed types never match |
//! | `in`, `not_in` | `value` (an array) contains / does not contain the attribute |
//! | `contains` | array attribute contains `value`, or string attribute contains the substring `value` |
//! | `contains_all` | array attribute contains every element of the array `value` |
//! | `subset_of` | every element of the array attribute is in the array `value` (missing counts as `[]`) |
//! | `exists`, `missing` | attribute present and non-null / absent or null |
//!
//! A missing attribute matches only `ne`, `not_in`, `missing` and
//! `subset_of`. A present attribute of the wrong shape (a scalar for
//! `contains_all` / `subset_of`) does not match. `{"all": []}` is true,
//! `{"any": []}` is false.
//!
//! `sql/postgres/heapdb.sql` (`heapdb.matches`) implements the same rules.

use std::cmp::Ordering;
use std::fmt;

use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::event::Attrs;
use crate::pyfmt::repr_str;

/// Deepest allowed nesting of `all` / `any` / `not`.
pub const MAX_FILTER_DEPTH: usize = 32;
/// Most nodes allowed in one filter.
pub const MAX_FILTER_NODES: usize = 1000;

/// A leaf operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafOp {
    /// `eq`
    Eq,
    /// `ne`
    Ne,
    /// `lt`
    Lt,
    /// `lte`
    Lte,
    /// `gt`
    Gt,
    /// `gte`
    Gte,
    /// `in`
    In,
    /// `not_in`
    NotIn,
    /// `contains`
    Contains,
    /// `contains_all`
    ContainsAll,
    /// `subset_of`
    SubsetOf,
    /// `exists`
    Exists,
    /// `missing`
    Missing,
}

const OPS: [(&str, LeafOp); 13] = [
    ("eq", LeafOp::Eq),
    ("ne", LeafOp::Ne),
    ("lt", LeafOp::Lt),
    ("lte", LeafOp::Lte),
    ("gt", LeafOp::Gt),
    ("gte", LeafOp::Gte),
    ("in", LeafOp::In),
    ("not_in", LeafOp::NotIn),
    ("contains", LeafOp::Contains),
    ("contains_all", LeafOp::ContainsAll),
    ("subset_of", LeafOp::SubsetOf),
    ("exists", LeafOp::Exists),
    ("missing", LeafOp::Missing),
];

impl LeafOp {
    /// Wire name.
    pub fn as_str(self) -> &'static str {
        OPS.iter()
            .find(|(_, o)| *o == self)
            .map(|(n, _)| *n)
            .expect("every op is listed")
    }

    /// Parse a wire name.
    pub fn parse(s: &str) -> Option<LeafOp> {
        OPS.iter().find(|(n, _)| *n == s).map(|(_, o)| *o)
    }

    /// The value this op needs.
    fn value_kind(self) -> ValueKind {
        match self {
            LeafOp::Exists | LeafOp::Missing => ValueKind::None,
            LeafOp::In | LeafOp::NotIn | LeafOp::ContainsAll | LeafOp::SubsetOf => ValueKind::Array,
            LeafOp::Lt | LeafOp::Lte | LeafOp::Gt | LeafOp::Gte => ValueKind::NumberOrString,
            LeafOp::Eq | LeafOp::Ne | LeafOp::Contains => ValueKind::NonNull,
        }
    }
}

enum ValueKind {
    None,
    Array,
    NumberOrString,
    NonNull,
}

/// `{"attr", "op", "value"}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Leaf {
    /// Top-level attribute name.
    pub attr: String,
    /// Operator.
    pub op: LeafOp,
    /// Right-hand side (`None` for `exists` / `missing`).
    pub value: Option<Value>,
}

/// A filter tree.
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    /// Every child matches (true when empty).
    All(Vec<Filter>),
    /// Some child matches (false when empty).
    Any(Vec<Filter>),
    /// The child does not match.
    Not(Box<Filter>),
    /// A comparison on one attribute.
    Leaf(Leaf),
}

/// JSON equality with numbers compared as doubles, at any depth.
pub(crate) fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
        }
        _ => a == b,
    }
}

fn json_cmp(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64()?.partial_cmp(&y.as_f64()?),
        // Byte-wise, i.e. by code point (Postgres: COLLATE "C").
        (Value::String(x), Value::String(y)) => Some(x.as_bytes().cmp(y.as_bytes())),
        _ => None,
    }
}

fn has(list: &[Value], x: &Value) -> bool {
    list.iter().any(|y| json_eq(x, y))
}

impl Leaf {
    /// Evaluate on attributes.
    pub fn matches(&self, attrs: &Attrs) -> bool {
        let lhs = attrs.get(&self.attr).filter(|v| !v.is_null());
        let rhs = self.value.as_ref().unwrap_or(&Value::Null);
        let rhs_list = rhs.as_array().map(Vec::as_slice).unwrap_or(&[]);
        let Some(lhs) = lhs else {
            return matches!(
                self.op,
                LeafOp::Ne | LeafOp::NotIn | LeafOp::Missing | LeafOp::SubsetOf
            );
        };
        use Ordering::*;
        match self.op {
            LeafOp::Exists => true,
            LeafOp::Missing => false,
            LeafOp::Eq => json_eq(lhs, rhs),
            LeafOp::Ne => !json_eq(lhs, rhs),
            LeafOp::Lt => json_cmp(lhs, rhs) == Some(Less),
            LeafOp::Lte => matches!(json_cmp(lhs, rhs), Some(Less | Equal)),
            LeafOp::Gt => json_cmp(lhs, rhs) == Some(Greater),
            LeafOp::Gte => matches!(json_cmp(lhs, rhs), Some(Greater | Equal)),
            LeafOp::In => has(rhs_list, lhs),
            LeafOp::NotIn => !has(rhs_list, lhs),
            LeafOp::Contains => match (lhs, rhs) {
                (Value::Array(a), x) => has(a, x),
                (Value::String(s), Value::String(t)) => s.contains(t.as_str()),
                _ => false,
            },
            LeafOp::ContainsAll => match lhs {
                Value::Array(a) => rhs_list.iter().all(|x| has(a, x)),
                _ => false,
            },
            LeafOp::SubsetOf => match lhs {
                Value::Array(a) => a.iter().all(|x| has(rhs_list, x)),
                _ => false,
            },
        }
    }

    fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("attr".into(), self.attr.clone().into());
        m.insert("op".into(), self.op.as_str().into());
        if let Some(v) = &self.value {
            m.insert("value".into(), v.clone());
        }
        Value::Object(m)
    }
}

impl Filter {
    /// A leaf.
    pub fn leaf(attr: impl Into<String>, op: LeafOp, value: Option<Value>) -> Filter {
        Filter::Leaf(Leaf {
            attr: attr.into(),
            op,
            value,
        })
    }

    /// Evaluate on attributes.
    pub fn matches(&self, attrs: &Attrs) -> bool {
        match self {
            Filter::All(cs) => cs.iter().all(|c| c.matches(attrs)),
            Filter::Any(cs) => cs.iter().any(|c| c.matches(attrs)),
            Filter::Not(c) => !c.matches(attrs),
            Filter::Leaf(l) => l.matches(attrs),
        }
    }

    /// Parse and validate a JSON filter. Errors name the offending node by
    /// its path, e.g. `filter.all[1].not: unknown op "like"`.
    pub fn from_json(v: &Value) -> Result<Filter> {
        let mut nodes = 0;
        parse_node(v, "filter", 0, &mut nodes)
    }

    /// Parse a filter from JSON text.
    pub fn parse_str(s: &str) -> Result<Filter> {
        let v: Value =
            serde_json::from_str(s).map_err(|e| Error::invalid(format!("filter is not valid JSON: {e}")))?;
        Self::from_json(&v)
    }

    /// Canonical JSON form (what `heapdb.matches` in Postgres takes).
    pub fn to_json(&self) -> Value {
        let list = |cs: &[Filter]| Value::Array(cs.iter().map(Filter::to_json).collect());
        let one = |k: &str, v: Value| {
            let mut m = Map::new();
            m.insert(k.into(), v);
            Value::Object(m)
        };
        match self {
            Filter::All(cs) => one("all", list(cs)),
            Filter::Any(cs) => one("any", list(cs)),
            Filter::Not(c) => one("not", c.to_json()),
            Filter::Leaf(l) => l.to_json(),
        }
    }
}

fn bad(path: &str, msg: impl fmt::Display) -> Error {
    Error::invalid(format!("{path}: {msg}"))
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

fn parse_node(v: &Value, path: &str, depth: usize, nodes: &mut usize) -> Result<Filter> {
    *nodes += 1;
    if *nodes > MAX_FILTER_NODES {
        return Err(Error::invalid(format!(
            "filter has more than {MAX_FILTER_NODES} nodes"
        )));
    }
    if depth > MAX_FILTER_DEPTH {
        return Err(bad(path, format!("nested deeper than {MAX_FILTER_DEPTH} levels")));
    }
    let Value::Object(m) = v else {
        return Err(bad(
            path,
            format!(
                "expected an object ({{\"all\": [...]}}, {{\"any\": [...]}}, {{\"not\": ...}} or {{\"attr\", \"op\", \"value\"}}), got {}",
                kind_of(v)
            ),
        ));
    };
    for comb in ["all", "any", "not"] {
        if let Some(child) = m.get(comb) {
            if m.len() != 1 {
                return Err(bad(path, format!("{comb:?} must be the only key of its node")));
            }
            if comb == "not" {
                let p = format!("{path}.not");
                return Ok(Filter::Not(Box::new(parse_node(child, &p, depth + 1, nodes)?)));
            }
            let Value::Array(items) = child else {
                return Err(bad(
                    path,
                    format!("{comb:?} takes an array of filters, got {}", kind_of(child)),
                ));
            };
            let cs = items
                .iter()
                .enumerate()
                .map(|(i, c)| parse_node(c, &format!("{path}.{comb}[{i}]"), depth + 1, nodes))
                .collect::<Result<Vec<_>>>()?;
            return Ok(if comb == "all" {
                Filter::All(cs)
            } else {
                Filter::Any(cs)
            });
        }
    }
    parse_leaf(m, path).map(Filter::Leaf)
}

fn parse_leaf(m: &Map<String, Value>, path: &str) -> Result<Leaf> {
    if let Some(k) = m.keys().find(|k| !matches!(k.as_str(), "attr" | "op" | "value")) {
        return Err(bad(
            path,
            format!(
                "unknown key {} (a leaf has \"attr\", \"op\" and \"value\"; combine with \"all\", \"any\", \"not\")",
                repr_str(k)
            ),
        ));
    }
    let attr = match m.get("attr") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(Value::String(_)) => return Err(bad(path, "\"attr\" must not be empty")),
        Some(other) => {
            return Err(bad(
                path,
                format!("\"attr\" must be a string, got {}", kind_of(other)),
            ))
        }
        None => return Err(bad(path, "missing \"attr\"")),
    };
    let op_name = match m.get("op") {
        Some(Value::String(s)) => s.as_str(),
        Some(other) => {
            return Err(bad(
                path,
                format!("\"op\" must be a string, got {}", kind_of(other)),
            ))
        }
        None => return Err(bad(path, "missing \"op\"")),
    };
    let Some(op) = LeafOp::parse(op_name) else {
        let names: Vec<&str> = OPS.iter().map(|(n, _)| *n).collect();
        return Err(bad(
            path,
            format!(
                "unknown op {} (expected one of {})",
                repr_str(op_name),
                names.join(", ")
            ),
        ));
    };
    let value = m.get("value");
    let ctx = |msg: &str| bad(path, format!("op {} {msg}", repr_str(op_name)));
    match (op.value_kind(), value) {
        (ValueKind::None, None) => {}
        (ValueKind::None, Some(_)) => return Err(ctx("takes no \"value\"")),
        (_, None) => return Err(ctx("needs a \"value\"")),
        (ValueKind::Array, Some(Value::Array(_))) => {}
        (ValueKind::Array, Some(v)) => return Err(ctx(&format!("needs an array value, got {}", kind_of(v)))),
        (ValueKind::NumberOrString, Some(Value::Number(_) | Value::String(_))) => {}
        (ValueKind::NumberOrString, Some(v)) => {
            return Err(ctx(&format!("needs a number or a string, got {}", kind_of(v))))
        }
        (ValueKind::NonNull, Some(Value::Null)) => {
            return Err(ctx(
                "needs a non-null value (use \"exists\" / \"missing\" to test for null)",
            ))
        }
        (ValueKind::NonNull, Some(_)) => {}
    }
    if let Some(v) = value {
        super::check_json("filter value", v).map_err(|e| bad(path, e))?;
    }
    Ok(Leaf {
        attr,
        op,
        value: value.cloned(),
    })
}

impl Serialize for Filter {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        self.to_json().serialize(s)
    }
}

impl<'de> Deserialize<'de> for Filter {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        Filter::from_json(&v).map_err(de::Error::custom)
    }
}

impl fmt::Display for Filter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&serde_json::to_string(&self.to_json()).map_err(|_| fmt::Error)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn attrs(v: Value) -> Attrs {
        v.as_object().unwrap().clone()
    }

    fn f(v: Value) -> Filter {
        Filter::from_json(&v).unwrap()
    }

    #[test]
    fn doc_example() {
        let flt = f(json!({"all": [
            {"attr": "kind", "op": "in", "value": ["bug", "task"]},
            {"attr": "required_skills", "op": "subset_of", "value": ["ts", "react"]},
            {"not": {"attr": "labels", "op": "contains", "value": "needs-human"}},
            {"attr": "blocked", "op": "ne", "value": true}
        ]}));
        assert!(flt.matches(&attrs(
            json!({"kind": "bug", "required_skills": ["ts"], "labels": []})
        )));
        assert!(flt.matches(&attrs(json!({"kind": "task"}))));
        assert!(!flt.matches(&attrs(json!({"kind": "bug", "labels": ["needs-human"]}))));
        assert!(!flt.matches(&attrs(json!({"kind": "bug", "required_skills": ["go"]}))));
        assert!(!flt.matches(&attrs(json!({"kind": "bug", "blocked": true}))));
        assert!(!flt.matches(&attrs(json!({"kind": "epic"}))));
        assert_eq!(Filter::from_json(&flt.to_json()).unwrap(), flt);
    }

    #[test]
    fn leaf_semantics() {
        let a = attrs(json!({"n": 1, "s": "abc", "tags": ["x", 2.0], "z": null, "o": {"k": [1]}}));
        let t = |v: Value| f(v).matches(&a);
        assert!(t(json!({"attr": "n", "op": "eq", "value": 1.0})));
        assert!(t(json!({"attr": "o", "op": "eq", "value": {"k": [1.0]}})));
        assert!(t(json!({"attr": "n", "op": "ne", "value": "1"})));
        assert!(t(json!({"attr": "n", "op": "lt", "value": 1.5})));
        assert!(
            !t(json!({"attr": "n", "op": "lt", "value": "2"})),
            "mixed types never match"
        );
        assert!(!t(json!({"attr": "n", "op": "gte", "value": "0"})));
        assert!(t(json!({"attr": "s", "op": "gte", "value": "abc"})));
        assert!(t(json!({"attr": "s", "op": "gt", "value": "ab"})));
        assert!(t(json!({"attr": "s", "op": "contains", "value": "bc"})));
        assert!(t(json!({"attr": "s", "op": "contains", "value": ""})));
        assert!(t(json!({"attr": "tags", "op": "contains", "value": 2})));
        assert!(!t(json!({"attr": "n", "op": "contains", "value": 1})));
        assert!(t(
            json!({"attr": "tags", "op": "contains_all", "value": [2, "x"]})
        ));
        assert!(t(json!({"attr": "tags", "op": "contains_all", "value": []})));
        assert!(!t(json!({"attr": "s", "op": "contains_all", "value": []})));
        assert!(t(
            json!({"attr": "tags", "op": "subset_of", "value": ["x", 2, "y"]})
        ));
        assert!(!t(json!({"attr": "tags", "op": "subset_of", "value": ["x"]})));
        assert!(!t(json!({"attr": "s", "op": "subset_of", "value": ["abc"]})));
        assert!(t(json!({"attr": "n", "op": "in", "value": [0, 1]})));
        assert!(t(json!({"attr": "n", "op": "not_in", "value": [0, 2]})));
        assert!(t(json!({"attr": "s", "op": "exists"})));
        assert!(t(json!({"attr": "z", "op": "missing"})), "null counts as missing");
        // A missing attribute matches only ne, not_in, missing, subset_of.
        for (op, v, want) in [
            ("eq", json!(1), false),
            ("ne", json!(1), true),
            ("lt", json!(1), false),
            ("lte", json!(1), false),
            ("gt", json!(1), false),
            ("gte", json!(1), false),
            ("in", json!([1]), false),
            ("not_in", json!([1]), true),
            ("contains", json!(1), false),
            ("contains_all", json!([]), false),
            ("subset_of", json!([]), true),
            ("exists", Value::Null, false),
            ("missing", Value::Null, true),
        ] {
            let mut leaf = json!({"attr": "nope", "op": op});
            if !v.is_null() {
                leaf["value"] = v;
            }
            assert_eq!(t(leaf.clone()), want, "{leaf}");
            leaf["attr"] = json!("z");
            assert_eq!(t(leaf.clone()), want, "{leaf}");
        }
        assert!(t(json!({"all": []})));
        assert!(!t(json!({"any": []})));
        assert!(t(json!({"not": {"any": []}})));
    }

    #[test]
    fn errors_name_the_node() {
        let err = |v: Value| Filter::from_json(&v).unwrap_err().to_string();
        assert_eq!(
            err(json!({"all": [{"attr": "a", "op": "eq", "value": 1}, {"not": {"attr": "b", "op": "like", "value": "x"}}]})),
            "filter.all[1].not: unknown op 'like' (expected one of eq, ne, lt, lte, gt, gte, in, not_in, contains, contains_all, subset_of, exists, missing)"
        );
        assert!(err(json!({"attr": "a", "op": "in", "value": "x"}))
            .contains("needs an array value, got a string"));
        assert!(err(json!({"attr": "a", "op": "eq"})).contains("needs a \"value\""));
        assert!(err(json!({"attr": "a", "op": "eq", "value": null})).contains("non-null"));
        assert!(err(json!({"attr": "a", "op": "exists", "value": 1})).contains("takes no"));
        assert!(err(json!({"attr": "a", "op": "lt", "value": [1]})).contains("number or a string"));
        assert!(err(json!({"attr": "a", "op": "eq", "value": 1, "x": 2})).contains("unknown key 'x'"));
        assert!(err(json!({"all": [], "any": []})).contains("only key"));
        assert!(err(json!({"all": {}})).contains("array of filters"));
        assert!(err(json!([1])).starts_with("filter: expected an object"));
        assert!(err(json!({"op": "eq", "value": 1})).contains("missing \"attr\""));
        let mut deep = json!({"attr": "a", "op": "exists"});
        for _ in 0..40 {
            deep = json!({"not": deep});
        }
        assert!(err(deep).contains("deeper"));
        assert!(Filter::parse_str("{nope")
            .unwrap_err()
            .to_string()
            .contains("not valid JSON"));
    }
}
