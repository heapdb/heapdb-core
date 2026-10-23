//! JSON forms of engine results (stable snake_case field names, documented
//! in `docs/server.md` and `/v1/openapi.json`).
//!
//! Non-finite numbers (a `leak` of infinity, NaN statistics) are encoded as
//! the strings `"Infinity"`, `"-Infinity"` and `"NaN"`, and accepted in
//! that form wherever a request takes a number that may be non-finite.

use heapdb::query::Output;
use heapdb::{
    AppliedRule, BetaCounts, CalibrationReport, Cause, Event, EventCredit, Explanation, FeatureContribution,
    FeedbackReport, ForgetReport, JointCause, LinkReport, ObserveReport, QueryResult, QueryStats,
    RankedCandidate, RiskResult, SettleReport,
};
use serde::de::{self, Deserializer};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{json, Map, Value};

/// A number that may be non-finite on the wire.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Num(pub f64);

/// JSON value of a float (non-finite as strings).
pub fn num(x: f64) -> Value {
    if x.is_finite() {
        Value::from(x)
    } else if x.is_nan() {
        Value::from("NaN")
    } else if x > 0.0 {
        Value::from("Infinity")
    } else {
        Value::from("-Infinity")
    }
}

fn opt_num(x: Option<f64>) -> Value {
    x.map_or(Value::Null, num)
}

/// Parse a number or one of the non-finite strings.
pub fn as_num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => match s.as_str() {
            "NaN" | "nan" => Some(f64::NAN),
            "Infinity" | "inf" | "+Infinity" => Some(f64::INFINITY),
            "-Infinity" | "-inf" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

impl Serialize for Num {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        num(self.0).serialize(s)
    }
}

impl<'de> Deserialize<'de> for Num {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v = Value::deserialize(d)?;
        as_num(&v)
            .map(Num)
            .ok_or_else(|| de::Error::custom("expected a number (or \"Infinity\", \"-Infinity\", \"NaN\")"))
    }
}

/// An event: `{id, entity, time, type, features, weight, attrs}`.
pub fn event(e: &Event) -> Value {
    json!({
        "id": e.id,
        "entity": e.entity,
        "time": num(e.time),
        "type": e.event_type,
        "features": e.features,
        "weight": num(e.weight),
        "attrs": Value::Object(e.attrs.clone()),
    })
}

/// A Beta link belief.
pub fn beta(b: &BetaCounts) -> Value {
    json!({
        "alpha": num(b.alpha()),
        "beta": num(b.beta()),
        "strength": num(b.strength()),
        "prior_alpha": num(b.prior_alpha),
        "prior_beta": num(b.prior_beta),
        "pos": num(b.pos),
        "neg": num(b.neg),
        "global_pos": num(b.global_pos),
        "global_neg": num(b.global_neg),
    })
}

fn contribution(c: &FeatureContribution) -> Value {
    let mut v = json!({"feature": c.feature, "strength": num(c.strength), "alpha": num(c.alpha), "beta": num(c.beta)});
    // Logistic only (noisy-OR explanations keep their exact shape).
    if let (Some(w), Value::Object(m)) = (c.log_odds, &mut v) {
        m.insert("log_odds".into(), num(w));
        m.insert("odds_multiplier".into(), num(w.exp()));
    }
    v
}

fn applied_rule(r: &AppliedRule) -> Value {
    json!({"name": r.name, "multiplier": num(r.multiplier)})
}

fn joint_cause(j: &JointCause) -> Value {
    json!({
        "features": j.features,
        "name": j.name(),
        "strength": num(j.strength),
        "joint": j.joint.as_ref().map_or(Value::Null, beta),
        "inseparable": j.inseparable,
    })
}

/// An explanation (`kernel x weight x link x rules`).
pub fn explanation(e: &Explanation) -> Value {
    let mut v = json!({
        "dt": num(e.dt),
        "kernel_value": num(e.kernel_value),
        "weight": num(e.weight),
        "link_strength": num(e.link_strength),
        "features": e.features.iter().map(contribution).collect::<Vec<_>>(),
        "warnings": e.warnings,
        "rules": e.rules.iter().map(applied_rule).collect::<Vec<_>>(),
        "joint_causes": e.joint_causes.iter().map(joint_cause).collect::<Vec<_>>(),
        "entity": e.entity,
        "prospective": e.prospective,
        "rule_multiplier": num(e.rule_multiplier()),
        "score": num(e.score()),
        "inseparable": e.inseparable(),
        "text": e.to_string(),
    });
    // Logistic only: base odds x multiplier per feature = odds -> link.
    if let (Some(b), Some(z), Value::Object(m)) = (e.base_log_odds, e.log_odds(), &mut v) {
        m.insert("combine".into(), e.combine().into());
        m.insert("base_log_odds".into(), num(b));
        m.insert("base_odds".into(), num(b.exp()));
        m.insert("log_odds".into(), num(z));
        m.insert("odds".into(), num(z.exp()));
    }
    v
}

fn opt_explanation(e: &Option<Explanation>) -> Option<Value> {
    e.as_ref().map(explanation)
}

/// One ranked cause.
pub fn cause(c: &Cause) -> Value {
    let mut m = Map::new();
    m.insert("rank".into(), c.rank.into());
    m.insert("event".into(), event(&c.event));
    m.insert("score".into(), num(c.score));
    if let Some(x) = opt_explanation(&c.explanation) {
        m.insert("explanation".into(), x);
    }
    Value::Object(m)
}

/// Work counters.
pub fn stats(s: &QueryStats) -> Value {
    json!({
        "segments_in_window": s.segments_in_window,
        "segments_scanned": s.segments_scanned,
        "segments_pruned": s.segments_pruned,
        "events_scanned": s.events_scanned,
        "union_fallbacks": s.union_fallbacks,
    })
}

/// A `top_causes` / `top_causes_group` result.
pub fn query_result(r: &QueryResult) -> Value {
    json!({
        "entity": r.entity,
        "entities": r.entities,
        "effect": r.effect,
        "at": num(r.at),
        "causes": r.causes.iter().map(cause).collect::<Vec<_>>(),
        "stats": stats(&r.stats),
        "text": r.to_string(),
    })
}

/// A `feature_risk` result.
pub fn risk_result(r: &RiskResult) -> Value {
    let mut m = Map::new();
    m.insert("effect".into(), r.effect.clone().into());
    m.insert("entity".into(), r.entity.clone().into());
    m.insert("features".into(), json!(r.features));
    m.insert("type".into(), r.event_type.clone().into());
    m.insert("score".into(), num(r.score));
    if let Some(x) = opt_explanation(&r.explanation) {
        m.insert("explanation".into(), x);
    }
    Value::Object(m)
}

/// One `rank_candidates` row; `candidate` echoes the request's candidate.
pub fn ranked_candidate(r: &RankedCandidate, candidate: &Value) -> Value {
    let mut m = Map::new();
    m.insert("rank".into(), r.rank.into());
    m.insert("index".into(), r.index.into());
    m.insert("score".into(), num(r.score));
    m.insert("entity".into(), r.entity.clone().into());
    m.insert("candidate".into(), candidate.clone());
    if let Some(x) = opt_explanation(&r.explanation) {
        m.insert("explanation".into(), x);
    }
    Value::Object(m)
}

fn pairs(v: &[(String, f64)]) -> Value {
    Value::Object(v.iter().map(|(k, x)| (k.clone(), num(*x))).collect())
}

/// A feedback report.
pub fn feedback(r: &FeedbackReport) -> Value {
    json!({"confirmed": r.confirmed, "rejected": r.rejected, "credit": pairs(&r.credit)})
}

fn event_credit(c: &EventCredit) -> Value {
    json!({
        "event_id": c.event_id,
        "score": num(c.score),
        "responsibility": num(c.responsibility),
        "features": pairs(&c.features),
    })
}

/// An observe report.
pub fn observe(r: &ObserveReport) -> Value {
    json!({
        "entity": r.entity,
        "effect": r.effect,
        "at": num(r.at),
        "leak": num(r.leak),
        "leak_responsibility": num(r.leak_responsibility),
        "total_credit": num(r.total_credit()),
        "credits": r.credits.iter().map(event_credit).collect::<Vec<_>>(),
        "text": r.to_string(),
    })
}

/// A settle report (the full report object).
pub fn settle(r: &SettleReport) -> Value {
    json!({
        "until": num(r.until),
        "events_settled": r.events_settled,
        "events_rejected": r.events_rejected,
        "rejected_weight": num(r.rejected_weight),
        "skipped_credited": r.skipped_credited,
        "skipped_observed": r.skipped_observed,
        "per_effect": Value::Object(r.per_effect.iter().map(|(k, n)| (k.clone(), Value::from(*n))).collect()),
        "text": r.to_string(),
    })
}

/// A link report.
pub fn link_report(r: &LinkReport) -> Value {
    json!({
        "effect": r.effect,
        "entity": r.entity,
        "baseline": num(r.baseline),
        "level": num(r.level),
        "rows": r.rows.iter().map(|x| {
            let mut v = json!({
                "feature": x.feature,
                "strength": num(x.strength),
                "alpha": num(x.alpha),
                "beta": num(x.beta),
                "n": num(x.n),
                "lo": num(x.lo),
                "hi": num(x.hi),
                "lift": num(x.lift),
            });
            if let (Some(w), Value::Object(m)) = (x.log_odds, &mut v) {
                m.insert("log_odds".into(), num(w));
            }
            v
        }).collect::<Vec<_>>(),
        "text": r.to_string(),
    })
}

/// A calibration report.
pub fn calibration(r: &CalibrationReport) -> Value {
    json!({
        "effect": r.effect,
        "n": r.n,
        "brier": opt_num(r.brier),
        "bins": r.bins.iter().map(|b| json!({
            "lo": num(b.lo),
            "hi": num(b.hi),
            "count": b.count,
            "mean_predicted": opt_num(b.mean_predicted),
            "observed_rate": opt_num(b.observed_rate),
        })).collect::<Vec<_>>(),
        "text": r.to_string(),
    })
}

/// What `forget_entity` removed.
pub fn forget(r: &ForgetReport) -> Value {
    json!({
        "events": r.events,
        "segments": r.segments,
        "links": r.links,
        "calibration_records": r.calibration_records,
    })
}

/// A query-language result: `{kind, result, text}`.
pub fn output(o: &Output) -> Value {
    let (kind, result) = match o {
        Output::Query(r) => ("top_causes", query_result(r)),
        Output::Calibration(r) => ("calibration", calibration(r)),
        Output::Risk(r) => ("risk", risk_result(r)),
        Output::Links(r) => ("links", link_report(r)),
        Output::Observe(r) => ("observe", observe(r)),
        Output::Settle(r) => ("settle", settle(r)),
        Output::Message(m) => ("message", Value::from(m.clone())),
    };
    json!({"kind": kind, "result": result, "text": o.to_string()})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_round_trip() {
        for x in [f64::INFINITY, f64::NEG_INFINITY, 1.5] {
            let v = serde_json::to_value(Num(x)).unwrap();
            let back: Num = serde_json::from_value(v).unwrap();
            assert_eq!(back.0, x);
        }
        let v = serde_json::to_value(Num(f64::NAN)).unwrap();
        assert_eq!(v, Value::from("NaN"));
        assert!(serde_json::from_value::<Num>(Value::from("x")).is_err());
    }
}
