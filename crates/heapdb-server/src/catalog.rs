//! Catalog reads (`GET /effects`, `/entities`, `/event-types`,
//! `/occurrences`): what a database holds, so people and consoles know
//! what to ask. Every call is bounded (see the `*_LIMIT` constants and the
//! engine's scan caps), pages with an opaque `cursor` / `next_cursor`, and
//! reports `truncated` when a scan stopped early.

use std::collections::HashMap;

use heapdb::catalog::{MAX_SCAN_ENTITIES, MAX_SCAN_EVENTS, MAX_SCAN_RECORDS};
use heapdb::{Kernel, KernelShape};
use serde_json::{json, Map, Value};

use crate::error::{ApiError, ApiResult};
use crate::json::num;
use crate::state::DbState;

/// Default and maximum `limit` of `GET /effects`.
pub const EFFECTS_LIMIT: (usize, usize) = (100, 500);
/// Default and maximum `limit` of `GET /event-types`.
pub const EVENT_TYPES_LIMIT: (usize, usize) = (100, 1000);
/// Default and maximum `limit` of `GET /entities`.
pub const ENTITIES_LIMIT: (usize, usize) = (50, 500);
/// Default and maximum `limit` of `GET /occurrences`.
pub const OCCURRENCES_LIMIT: (usize, usize) = (20, 200);
/// Default and maximum `features` of `GET /event-types`.
pub const TOP_FEATURES: (usize, usize) = (20, 100);
/// Priors listed per effect by `GET /effects`.
pub const TOP_PRIORS: usize = 20;

/// Parse an optional positive integer parameter, capped at `max`.
pub fn limit_param(
    q: &HashMap<String, String>,
    name: &str,
    (default, max): (usize, usize),
) -> ApiResult<usize> {
    match q.get(name).map(String::as_str) {
        None | Some("") => Ok(default),
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(n.min(max)),
            _ => Err(ApiError::bad_request(format!(
                "{name} must be a positive integer (at most {max}), got {v:?}"
            ))),
        },
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn bad_cursor() -> ApiError {
    ApiError::bad_request("cursor is not a cursor returned by this endpoint")
}

fn unhex(c: &str) -> ApiResult<String> {
    if !c.len().is_multiple_of(2) || !c.is_ascii() {
        return Err(bad_cursor());
    }
    let bytes = (0..c.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&c[i..i + 2], 16))
        .collect::<Result<Vec<u8>, _>>()
        .map_err(|_| bad_cursor())?;
    String::from_utf8(bytes).map_err(|_| bad_cursor())
}

/// Opaque cursor for a position `(time, names...)`: the time's bits, then
/// each name hex-encoded, separated by `.`.
pub fn encode_cursor(t: f64, names: &[&str]) -> String {
    let mut c = format!("{:016x}", t.to_bits());
    for n in names {
        c.push('.');
        c.push_str(&hex(n.as_bytes()));
    }
    c
}

/// Inverse of [`encode_cursor`] for `n` names.
pub fn decode_cursor(c: &str, n: usize) -> ApiResult<(f64, Vec<String>)> {
    let mut parts = c.split('.');
    let t = parts.next().filter(|t| t.len() == 16).ok_or_else(bad_cursor)?;
    let t = u64::from_str_radix(t, 16).map_err(|_| bad_cursor())?;
    let names = parts.map(unhex).collect::<ApiResult<Vec<String>>>()?;
    if names.len() != n {
        return Err(bad_cursor());
    }
    Ok((f64::from_bits(t), names))
}

/// Opaque cursor naming an effect (`GET /effects`).
pub fn encode_name_cursor(name: &str) -> String {
    hex(name.as_bytes())
}

/// Opaque cursor for an offset into a computed list (`GET /event-types`).
pub fn encode_offset_cursor(offset: usize) -> String {
    format!("{offset:x}")
}

fn decode_offset_cursor(c: &str) -> ApiResult<usize> {
    if c.is_empty() || c.len() > 16 {
        return Err(bad_cursor());
    }
    usize::from_str_radix(c, 16).map_err(|_| bad_cursor())
}

fn cursor_param(c: Option<&str>) -> Option<&str> {
    c.filter(|c| !c.is_empty())
}

/// `2700` -> `45m`, `9000` -> `2h 30m`, `259200` -> `3d`.
pub fn duration_text(s: f64) -> String {
    if !s.is_finite() {
        return "forever".into();
    }
    let s = s.max(0.0).round() as u64;
    let (d, h, m, sec) = (s / 86400, (s % 86400) / 3600, (s % 3600) / 60, s % 60);
    match (d, h, m) {
        (0, 0, 0) => format!("{sec}s"),
        (0, 0, _) => format!("{m}m"),
        (0, _, 0) => format!("{h}h"),
        (0, _, _) => format!("{h}h {m}m"),
        (_, 0, _) => format!("{d}d"),
        _ => format!("{d}d {h}h"),
    }
}

/// Kernel as `{type, params, human, text}`: `params` as accepted by
/// `POST /effects`, `human` in seconds with `_s` names.
pub fn kernel_json(k: &Kernel) -> Value {
    let params: Map<String, Value> = k.params().into_iter().map(|(n, v)| (n.into(), num(v))).collect();
    let max_lag = k.support();
    let (ty, human, text) = match *k.shape() {
        KernelShape::Exponential { half_life, .. } => (
            "exponential",
            json!({"half_life_s": num(half_life)}),
            format!("halves every {}", duration_text(half_life)),
        ),
        KernelShape::Window { min_lag, max_lag } => (
            "window",
            json!({"min_lag_s": num(min_lag)}),
            format!(
                "flat from {} to {}",
                duration_text(min_lag),
                duration_text(max_lag)
            ),
        ),
        KernelShape::LogNormal { median, sigma, .. } => (
            "lognormal",
            json!({"median_s": num(median), "sigma": num(sigma)}),
            format!("typical lag {}", duration_text(median)),
        ),
        KernelShape::Gamma { shape, scale, .. } => (
            "gamma",
            json!({"mean_s": num(shape * scale), "shape": num(shape)}),
            format!("typical lag {}", duration_text(shape * scale)),
        ),
    };
    let mut human = human.as_object().cloned().unwrap_or_default();
    human.insert("peak_s".into(), num(k.mode()));
    human.insert("max_lag_s".into(), num(max_lag));
    json!({
        "type": ty,
        "params": params,
        "human": human,
        "text": format!("{text}, looks back up to {}", duration_text(max_lag)),
    })
}

impl DbState {
    /// `GET /effects`.
    pub fn catalog_effects(&self, limit: usize, cursor: Option<&str>) -> ApiResult<Value> {
        let after = cursor_param(cursor).map(unhex).transpose()?;
        let page = self
            .db
            .effect_summaries(after.as_deref(), limit, TOP_PRIORS)
            .map_err(|_| bad_cursor())?;
        let effects: Vec<Value> = page
            .effects
            .iter()
            .map(|e| {
                json!({
                    "name": e.name,
                    "kernel": kernel_json(&e.kernel),
                    "credit": e.credit.as_str(),
                    "combine": e.combine.as_str(),
                    "logistic": e.logistic.map(|l| json!({
                        "learning_rate": num(l.learning_rate),
                        "l2": num(l.l2),
                        "base_log_odds": e.base_log_odds.map(num),
                    })),
                    "cause_types": e.cause_types,
                    "priors": {
                        "count": e.num_priors,
                        "default": [num(e.default_prior.0), num(e.default_prior.1)],
                        "top": e.priors.iter().map(|p| json!({
                            "feature": p.feature, "alpha": num(p.alpha), "beta": num(p.beta),
                            "strength": num(p.strength()),
                        })).collect::<Vec<_>>(),
                    },
                    "rules": e.rules,
                    "hierarchical": e.hierarchical,
                    "global_pseudocount": num(e.global_pseudocount),
                    "leak": num(e.leak),
                    "w_neg": num(e.w_neg),
                    "repeatable": e.repeatable,
                    "counts": {
                        "occurrences": e.occurrences,
                        "occurrence_entities": e.occurrence_entities,
                        "feedback": e.feedback_records,
                        "feedback_confirmed": e.feedback_confirmed,
                        "learned_features": e.learned_features,
                    },
                })
            })
            .collect();
        Ok(json!({
            "effects": effects,
            "next_cursor": page.next.as_deref().map(encode_name_cursor),
            "total_effects": page.total,
        }))
    }

    /// `GET /entities`.
    pub fn catalog_entities(
        &self,
        prefix: Option<&str>,
        limit: usize,
        cursor: Option<&str>,
    ) -> ApiResult<Value> {
        let after = cursor_param(cursor).map(|c| decode_cursor(c, 1)).transpose()?;
        let page = self.db.entity_summaries(
            prefix,
            limit,
            after.as_ref().map(|(t, n)| (*t, n[0].as_str())),
            MAX_SCAN_ENTITIES,
        );
        let entities: Vec<Value> = page
            .entities
            .iter()
            .map(|e| {
                json!({
                    "name": e.name, "event_count": e.events,
                    "first_event_time": num(e.first_time), "last_event_time": num(e.last_time),
                })
            })
            .collect();
        Ok(json!({
            "entities": entities,
            "next_cursor": page.next.map(|(t, n)| encode_cursor(t, &[&n])),
            "matched": page.matched,
            "scanned": page.scanned,
            "total_entities": page.total,
            "truncated": page.truncated,
        }))
    }

    /// `GET /event-types`.
    pub fn catalog_event_types(
        &self,
        top_features: usize,
        limit: usize,
        cursor: Option<&str>,
    ) -> ApiResult<Value> {
        let start = cursor_param(cursor)
            .map(decode_offset_cursor)
            .transpose()?
            .unwrap_or(0);
        let c = self.db.event_type_summaries(top_features, MAX_SCAN_EVENTS);
        let total = c.types.len();
        let end = start.saturating_add(limit).min(total);
        let types: Vec<Value> = c
            .types
            .iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .map(|t| {
                json!({
                    "type": t.name,
                    "count": t.events,
                    "last_event_time": t.last_time.map(num),
                    "registered": t.registered,
                    "weight": self.schema.event_types.get(&t.name).cloned().flatten(),
                    "top_features": t.features.iter().map(|(f, n)| json!({"feature": f, "count": n})).collect::<Vec<_>>(),
                })
            })
            .collect();
        Ok(json!({
            "event_types": types,
            "next_cursor": (end < total).then(|| encode_offset_cursor(end)),
            "total_types": total,
            "events_scanned": c.events_scanned,
            "total_events": c.total_events,
            "truncated": c.truncated,
        }))
    }

    /// `GET /occurrences`.
    pub fn catalog_occurrences(
        &self,
        effect: Option<&str>,
        entity: Option<&str>,
        limit: usize,
        cursor: Option<&str>,
    ) -> ApiResult<Value> {
        let after = cursor_param(cursor).map(|c| decode_cursor(c, 2)).transpose()?;
        let after = after.as_ref().map(|(t, n)| (*t, n[0].as_str(), n[1].as_str()));
        let p = self
            .db
            .recent_occurrences(effect, entity, limit, after, MAX_SCAN_RECORDS)?;
        let rows: Vec<Value> = p
            .occurrences
            .iter()
            .map(|o| {
                let mut sources = Vec::new();
                if o.observed {
                    sources.push("observe");
                }
                if o.feedback {
                    sources.push("feedback");
                }
                json!({
                    "effect": o.effect, "entity": o.entity, "at": num(o.at), "sources": sources,
                    "confirmed_event_id": o.confirmed_event_id, "rejected": o.rejected,
                })
            })
            .collect();
        Ok(json!({
            "occurrences": rows,
            "next_cursor": p.next.map(|(t, ent, eff)| encode_cursor(t, &[&ent, &eff])),
            "truncated": p.truncated,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursors_round_trip() {
        for (t, n) in [(1.5e9, "acme/web"), (-3.25, ""), (0.0, "ünï code & co.")] {
            let c = encode_cursor(t, &[n]);
            assert!(c.bytes().all(|b| b.is_ascii_hexdigit() || b == b'.'));
            assert_eq!(decode_cursor(&c, 1).unwrap(), (t, vec![n.to_string()]));
            let c2 = encode_cursor(t, &[n, "effect.x"]);
            assert_eq!(
                decode_cursor(&c2, 2).unwrap(),
                (t, vec![n.to_string(), "effect.x".to_string()])
            );
            assert!(decode_cursor(&c2, 1).is_err());
            assert_eq!(unhex(&encode_name_cursor(n)).unwrap(), n);
        }
        for bad in [
            "",
            "zz",
            "0123456789abcdef",
            "0123456789abcdef0",
            "0123456789abcdef.zz",
            "0123456789abcdef.ff",
            "0123456789abcdef.6",
            "0123456789abcdeg.61",
        ] {
            assert_eq!(decode_cursor(bad, 1).unwrap_err().code, "bad_request", "{bad}");
        }
        assert_eq!(decode_offset_cursor(&encode_offset_cursor(300)).unwrap(), 300);
        for bad in ["", "-1", "x", "11111111111111111"] {
            assert!(decode_offset_cursor(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn durations_and_kernels() {
        assert_eq!(duration_text(45.0), "45s");
        assert_eq!(duration_text(2700.0), "45m");
        assert_eq!(duration_text(9000.0), "2h 30m");
        assert_eq!(duration_text(86400.0 * 3.0), "3d");
        assert_eq!(duration_text(86400.0 + 7200.0), "1d 2h");
        let k = kernel_json(&Kernel::lognormal(2700.0, 0.7).unwrap());
        assert_eq!(k["type"], "lognormal");
        assert_eq!(k["params"]["median"], 2700.0);
        assert_eq!(k["human"]["median_s"], 2700.0);
        assert!(k["text"]
            .as_str()
            .unwrap()
            .starts_with("typical lag 45m, looks back up to"));
        let k = kernel_json(&Kernel::exponential(3600.0).unwrap());
        assert_eq!(k["type"], "exponential");
        assert_eq!(k["human"]["half_life_s"], 3600.0);
        assert_eq!(k["human"]["max_lag_s"], 36000.0);
    }

    #[test]
    fn limits() {
        let q = |v: &str| HashMap::from([("limit".to_string(), v.to_string())]);
        assert_eq!(limit_param(&HashMap::new(), "limit", (50, 500)).unwrap(), 50);
        assert_eq!(limit_param(&q("7"), "limit", (50, 500)).unwrap(), 7);
        assert_eq!(limit_param(&q("100000"), "limit", (50, 500)).unwrap(), 500);
        for bad in ["0", "-1", "x", "1.5"] {
            assert!(limit_param(&q(bad), "limit", (50, 500)).is_err());
        }
    }
}
