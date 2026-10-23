//! Declarative schema: effects (kernel + priors), event-type weight
//! functions and rule predicates. No code is uploaded: weight functions
//! and predicates are small JSON descriptions evaluated by the server.

use std::any::Any;
use std::sync::Arc;

use heapdb::{Attrs, BoxError, Combine, Credit, EffectOptions, EventRef, Kernel, Predicate, WeightFn};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{ApiError, ApiResult};
use crate::json::{as_num, Num};

// ---------------------------------------------------------------------------
// Effects
// ---------------------------------------------------------------------------

/// `{type, params}` of a lag kernel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSpec {
    /// `exponential` (alias `exp`), `window`, `lognormal`, `gamma`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Kernel parameters, e.g. `{"median": 7200, "sigma": 0.6}`.
    #[serde(default)]
    pub params: Map<String, Value>,
}

impl KernelSpec {
    /// Build the engine kernel.
    pub fn build(&self) -> ApiResult<Kernel> {
        let kind = match self.kind.to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
            "exp" | "exponential" | "exponential_decay" | "exponentialdecay" => "exp",
            "window" | "window_kernel" | "windowkernel" => "window",
            "lognormal" | "log_normal" | "lognormal_kernel" | "lognormalkernel" => "lognormal",
            "gamma" | "gamma_kernel" | "gammakernel" => "gamma",
            other => return Err(ApiError::invalid(format!("unknown kernel type {other:?}"))),
        };
        let mut m = Map::new();
        m.insert("kind".into(), kind.into());
        for (k, v) in &self.params {
            m.insert(k.clone(), v.clone());
        }
        Ok(Kernel::from_json(&m)?)
    }
}

/// Body of `POST /effects` (also the WAL record of `register_effect`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectReq {
    /// Effect name.
    pub name: String,
    /// Lag kernel.
    pub kernel: KernelSpec,
    /// feature -> `[alpha, beta]` prior.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priors: Option<indexmap_like::OrderedPriors>,
    /// Default `[alpha, beta]` prior (default `[1, 4]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_prior: Option<[f64; 2]>,
    /// Only these event types can be causes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cause_types: Option<Vec<String>>,
    /// `proportional` or `uniform` (default: `proportional` for
    /// `noisy_or`, `uniform` for `logistic`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credit: Option<String>,
    /// How features combine into the link: `noisy_or` (default) or
    /// `logistic`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combine: Option<String>,
    /// Logistic learning rate (FTRL alpha, default 0.1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f64>,
    /// Logistic L2 penalty towards the prior seeds (default 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub l2: Option<f64>,
    /// Global-layer pseudocount (default 5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_pseudocount: Option<f64>,
    /// Hierarchical links (default true).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hierarchical: Option<bool>,
    /// Background score of `observe` (default 0.05; may be `"Infinity"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leak: Option<Num>,
    /// Weight of negative evidence from `settle` (default 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub w_neg: Option<f64>,
    /// Effect can recur (default true).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeatable: Option<bool>,
}

/// Priors keep their request order (it fixes the interning order, so
/// results are bit-identical to the in-process engine).
pub mod indexmap_like {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde_json::{Map, Value};

    /// Ordered `feature -> [alpha, beta]`.
    #[derive(Clone, Debug, PartialEq, Default)]
    pub struct OrderedPriors(pub Vec<(String, (f64, f64))>);

    impl Serialize for OrderedPriors {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let m: Map<String, Value> = self
                .0
                .iter()
                .map(|(k, (a, b))| (k.clone(), Value::from(vec![*a, *b])))
                .collect();
            m.serialize(s)
        }
    }

    impl<'de> Deserialize<'de> for OrderedPriors {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let m = Map::<String, Value>::deserialize(d)?;
            let mut out = Vec::with_capacity(m.len());
            for (k, v) in m {
                let pair = match &v {
                    Value::Array(a) if a.len() == 2 => (a[0].as_f64(), a[1].as_f64()),
                    Value::Object(o) => (
                        o.get("alpha").and_then(Value::as_f64),
                        o.get("beta").and_then(Value::as_f64),
                    ),
                    _ => (None, None),
                };
                match pair {
                    (Some(a), Some(b)) => out.push((k, (a, b))),
                    _ => {
                        return Err(serde::de::Error::custom(format!(
                            "prior of {k:?} must be [alpha, beta]"
                        )))
                    }
                }
            }
            Ok(OrderedPriors(out))
        }
    }
}

impl EffectReq {
    /// Engine options.
    pub fn options(&self) -> ApiResult<EffectOptions> {
        let mut o = EffectOptions::new(self.kernel.build()?);
        if let Some(p) = &self.priors {
            o.priors = p.0.clone();
        }
        if let Some([a, b]) = self.default_prior {
            o.default_prior = (a, b);
        }
        if let Some(c) = &self.cause_types {
            o.cause_types = Some(c.clone());
        }
        if let Some(c) = &self.credit {
            o.credit = Some(Credit::parse(c)?);
        }
        if let Some(c) = &self.combine {
            o.combine = Combine::parse(c)?;
        }
        if let Some(lr) = self.learning_rate {
            o.logistic.learning_rate = lr;
        }
        if let Some(l2) = self.l2 {
            o.logistic.l2 = l2;
        }
        if let Some(g) = self.global_pseudocount {
            o.global_pseudocount = g;
        }
        if let Some(h) = self.hierarchical {
            o.hierarchical = h;
        }
        if let Some(l) = self.leak {
            o.leak = l.0;
        }
        if let Some(w) = self.w_neg {
            o.w_neg = w;
        }
        if let Some(r) = self.repeatable {
            o.repeatable = r;
        }
        Ok(o)
    }
}

// ---------------------------------------------------------------------------
// Weight functions
// ---------------------------------------------------------------------------

/// Kind of a declarative weight function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightKind {
    /// `ln(1 + attrs[attr])`.
    Log1p,
    /// `attrs[attr]`.
    Attr,
    /// A constant `value`.
    Const,
}

/// A declarative weight function: `{kind, attr?, value?}`. For `attr` and
/// `log1p`, `value` is the attribute's fallback when it is missing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightSpec {
    /// Kind.
    pub kind: WeightKind,
    /// Attribute name (`attr`, `log1p`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attr: Option<String>,
    /// Constant (`const`) or fallback attribute value (`attr`, `log1p`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
}

impl WeightSpec {
    /// Check the description is complete.
    pub fn validate(&self) -> ApiResult<()> {
        match self.kind {
            WeightKind::Const => match self.value {
                Some(v) if v.is_finite() && v > 0.0 => Ok(()),
                _ => Err(ApiError::invalid("weight kind const needs a finite value > 0")),
            },
            WeightKind::Attr | WeightKind::Log1p => match &self.attr {
                Some(a) if !a.is_empty() => Ok(()),
                _ => Err(ApiError::invalid(format!(
                    "weight kind {} needs an attr",
                    serde_json::to_value(self.kind).unwrap_or_default()
                ))),
            },
        }
    }

    /// Evaluate on an event's attributes.
    pub fn eval(&self, attrs: &Attrs) -> Result<f64, String> {
        let attr_value = |name: &str| -> Result<f64, String> {
            match attrs.get(name) {
                Some(v) if !v.is_null() => {
                    as_num(v).ok_or_else(|| format!("attribute {name:?} must be a number, got {v}"))
                }
                _ => self
                    .value
                    .ok_or_else(|| format!("attribute {name:?} is missing (and no fallback value)")),
            }
        };
        match self.kind {
            WeightKind::Const => self.value.ok_or_else(|| "const weight without value".to_string()),
            WeightKind::Attr => attr_value(self.attr.as_deref().unwrap_or_default()),
            WeightKind::Log1p => attr_value(self.attr.as_deref().unwrap_or_default()).map(f64::ln_1p),
        }
    }

    /// As an engine weight function.
    pub fn weight_fn(&self) -> Arc<dyn WeightFn> {
        Arc::new(DeclWeight(self.clone()))
    }
}

struct DeclWeight(WeightSpec);

impl WeightFn for DeclWeight {
    fn weight(&self, attrs: &Attrs) -> Result<f64, BoxError> {
        self.0.eval(attrs).map_err(BoxError::from)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Predicates and rules
// ---------------------------------------------------------------------------

/// Comparison operator of a predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// `==`
    #[serde(alias = "==")]
    Eq,
    /// `!=`
    #[serde(alias = "!=")]
    Ne,
    /// `<`
    #[serde(alias = "<")]
    Lt,
    /// `<=`
    #[serde(alias = "<=", alias = "lte")]
    Le,
    /// `>`
    #[serde(alias = ">")]
    Gt,
    /// `>=`
    #[serde(alias = ">=", alias = "gte")]
    Ge,
    /// value is a list containing the attribute
    In,
    /// value is a list not containing the attribute
    NotIn,
    /// attribute (string or list) contains value
    Contains,
    /// attribute is present (not null)
    Exists,
    /// attribute is absent or null
    Missing,
}

/// A declarative predicate on event attributes: `{attr, op, value?, default?}`.
/// A missing attribute takes `default` if given; otherwise only `missing`
/// (and `ne`, `not_in`) can match.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredicateSpec {
    /// Attribute name.
    pub attr: String,
    /// Operator.
    pub op: Op,
    /// Right-hand side.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    /// Value used when the attribute is missing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
}

fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        _ => a == b,
    }
}

fn json_cmp(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64()?.partial_cmp(&y.as_f64()?),
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

impl PredicateSpec {
    /// Check the description is complete.
    pub fn validate(&self) -> ApiResult<()> {
        let needs_value = !matches!(self.op, Op::Exists | Op::Missing);
        if needs_value && self.value.is_none() {
            return Err(ApiError::invalid(format!(
                "predicate op {:?} needs a value",
                self.op
            )));
        }
        if matches!(self.op, Op::In | Op::NotIn) && !matches!(self.value, Some(Value::Array(_))) {
            return Err(ApiError::invalid("predicate ops in / not_in need a list value"));
        }
        Ok(())
    }

    /// Evaluate on attributes.
    pub fn eval(&self, attrs: &Attrs) -> bool {
        let present = attrs.get(&self.attr).filter(|v| !v.is_null());
        match self.op {
            Op::Exists => return present.is_some(),
            Op::Missing => return present.is_none(),
            _ => {}
        }
        let rhs = self.value.as_ref().unwrap_or(&Value::Null);
        let Some(lhs) = present.or(self.default.as_ref()) else {
            return matches!(self.op, Op::Ne | Op::NotIn);
        };
        use std::cmp::Ordering::*;
        match self.op {
            Op::Eq => json_eq(lhs, rhs),
            Op::Ne => !json_eq(lhs, rhs),
            Op::Lt => json_cmp(lhs, rhs) == Some(Less),
            Op::Le => matches!(json_cmp(lhs, rhs), Some(Less | Equal)),
            Op::Gt => json_cmp(lhs, rhs) == Some(Greater),
            Op::Ge => matches!(json_cmp(lhs, rhs), Some(Greater | Equal)),
            Op::In => rhs.as_array().is_some_and(|a| a.iter().any(|x| json_eq(lhs, x))),
            Op::NotIn => !rhs.as_array().is_some_and(|a| a.iter().any(|x| json_eq(lhs, x))),
            Op::Contains => match (lhs, rhs) {
                (Value::String(s), Value::String(t)) => s.contains(t.as_str()),
                (Value::Array(a), x) => a.iter().any(|y| json_eq(x, y)),
                _ => false,
            },
            Op::Exists | Op::Missing => unreachable!("handled above"),
        }
    }
}

struct DeclPredicate(PredicateSpec);

impl Predicate for DeclPredicate {
    fn matches(&self, event: &EventRef<'_>) -> Result<bool, BoxError> {
        Ok(self.0.eval(event.attrs))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A rule: body of `POST /rules` (minus `effect`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleSpec {
    /// Rule name (unique per effect).
    pub name: String,
    /// Multiplier in `[0, 1]`.
    pub multiplier: f64,
    /// Restrict to these event types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub types: Option<Vec<String>>,
    /// Attribute predicate (`None`: type-level rule).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<PredicateSpec>,
}

impl RuleSpec {
    /// The engine predicate.
    pub fn predicate_fn(&self) -> Option<Arc<dyn Predicate>> {
        self.predicate
            .as_ref()
            .map(|p| Arc::new(DeclPredicate(p.clone())) as Arc<dyn Predicate>)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn attrs(v: Value) -> Attrs {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn weights() {
        let w: WeightSpec = serde_json::from_value(json!({"kind": "log1p", "attr": "lines"})).unwrap();
        assert!((w.eval(&attrs(json!({"lines": 420}))).unwrap() - 421f64.ln()).abs() < 1e-12);
        assert!(w.eval(&attrs(json!({}))).is_err());
        let w: WeightSpec =
            serde_json::from_value(json!({"kind": "attr", "attr": "portion", "value": 1.0})).unwrap();
        assert_eq!(w.eval(&attrs(json!({"portion": 2}))).unwrap(), 2.0);
        assert_eq!(w.eval(&attrs(json!({}))).unwrap(), 1.0);
        let w: WeightSpec = serde_json::from_value(json!({"kind": "const", "value": 3})).unwrap();
        assert_eq!(w.eval(&attrs(json!({}))).unwrap(), 3.0);
        assert!(serde_json::from_value::<WeightSpec>(json!({"kind": "python", "code": "x"})).is_err());
        let bad: WeightSpec = serde_json::from_value(json!({"kind": "const"})).unwrap();
        assert!(bad.validate().is_err());
    }

    #[test]
    fn predicates() {
        let p: PredicateSpec =
            serde_json::from_value(json!({"attr": "amount", "op": "<", "value": 10, "default": 0})).unwrap();
        assert!(p.eval(&attrs(json!({"amount": 5}))));
        assert!(!p.eval(&attrs(json!({"amount": 50}))));
        assert!(p.eval(&attrs(json!({}))));
        let p: PredicateSpec =
            serde_json::from_value(json!({"attr": "cur", "op": "in", "value": ["EUR", "USD"]})).unwrap();
        assert!(p.eval(&attrs(json!({"cur": "EUR"}))));
        assert!(!p.eval(&attrs(json!({"cur": "GBP"}))));
        assert!(!p.eval(&attrs(json!({}))));
        let p: PredicateSpec = serde_json::from_value(json!({"attr": "x", "op": "exists"})).unwrap();
        assert!(p.eval(&attrs(json!({"x": 0}))) && !p.eval(&attrs(json!({"x": null}))));
        let p: PredicateSpec =
            serde_json::from_value(json!({"attr": "tags", "op": "contains", "value": "a"})).unwrap();
        assert!(p.eval(&attrs(json!({"tags": ["a", "b"]}))));
        assert!(p.eval(&attrs(json!({"tags": "cat"}))));
        let p: PredicateSpec = serde_json::from_value(json!({"attr": "n", "op": "eq", "value": 1})).unwrap();
        assert!(p.eval(&attrs(json!({"n": 1.0}))));
    }

    #[test]
    fn kernels() {
        let k = KernelSpec {
            kind: "LogNormal".into(),
            params: json!({"median": 7200, "sigma": 0.6}).as_object().unwrap().clone(),
        };
        assert_eq!(k.build().unwrap(), Kernel::lognormal(7200.0, 0.6).unwrap());
        let k = KernelSpec {
            kind: "exponential".into(),
            params: json!({"half_life": 60}).as_object().unwrap().clone(),
        };
        assert_eq!(k.build().unwrap(), Kernel::exponential(60.0).unwrap());
        let k = KernelSpec {
            kind: "spline".into(),
            params: Map::new(),
        };
        assert!(k.build().is_err());
    }
}
