//! `GET /v1/openapi.json`: OpenAPI 3.1 description of the Data API.

use serde_json::{json, Map, Value};

fn r(name: &str) -> Value {
    json!({"$ref": format!("#/components/schemas/{name}")})
}

fn num() -> Value {
    json!({"oneOf": [{"type": "number"}, {"type": "string", "enum": ["Infinity", "-Infinity", "NaN"]}]})
}

fn arr(items: Value) -> Value {
    json!({"type": "array", "items": items})
}

fn obj(props: Value, required: &[&str]) -> Value {
    let mut m = Map::new();
    m.insert("type".into(), "object".into());
    m.insert("properties".into(), props);
    if !required.is_empty() {
        m.insert("required".into(), json!(required));
    }
    Value::Object(m)
}

fn nullable(v: Value) -> Value {
    json!({"oneOf": [v, {"type": "null"}]})
}

fn schemas() -> Value {
    let mut s = Map::new();
    let mut put = |k: &str, v: Value| {
        s.insert(k.to_string(), v);
    };
    put(
        "Error",
        obj(
            json!({"error": obj(json!({"code": {"type": "string"}, "message": {"type": "string"}}), &["code", "message"])}),
            &["error"],
        ),
    );
    put("Number", num());
    put(
        "Event",
        obj(
            json!({
                "id": {"type": "integer"}, "entity": {"type": "string"}, "time": {"type": "number"},
                "type": {"type": "string"}, "features": arr(json!({"type": "string"})),
                "weight": {"type": "number"}, "attrs": {"type": "object"}
            }),
            &["id", "entity", "time", "type", "features", "weight", "attrs"],
        ),
    );
    put(
        "EventInput",
        obj(
            json!({
                "entity": {"type": "string"}, "time": {"type": "number"}, "type": {"type": "string"},
                "features": arr(json!({"type": "string"})), "weight": {"type": "number", "exclusiveMinimum": 0},
                "attrs": {"type": "object"}, "external_id": {"type": "string", "description": "Idempotency key: a repeated insert returns the first event's id."}
            }),
            &["entity", "time", "type"],
        ),
    );
    put(
        "BetaCounts",
        obj(
            json!({
                "alpha": {"type": "number"}, "beta": {"type": "number"}, "strength": {"type": "number"},
                "prior_alpha": {"type": "number"}, "prior_beta": {"type": "number"},
                "pos": {"type": "number"}, "neg": {"type": "number"},
                "global_pos": {"type": "number"}, "global_neg": {"type": "number"}
            }),
            &[],
        ),
    );
    put(
        "Explanation",
        obj(
            json!({
                "dt": {"type": "number"}, "kernel_value": {"type": "number"}, "weight": {"type": "number"},
                "link_strength": {"type": "number"},
                "features": arr(obj(json!({"feature": {"type": "string"}, "strength": {"type": "number"}, "alpha": {"type": "number"}, "beta": {"type": "number"}}), &[])),
                "warnings": arr(json!({"type": "string"})),
                "rules": arr(obj(json!({"name": {"type": "string"}, "multiplier": {"type": "number"}}), &[])),
                "joint_causes": arr(obj(json!({
                    "features": arr(json!({"type": "string"})), "name": {"type": "string"},
                    "strength": {"type": "number"}, "joint": nullable(r("BetaCounts")), "inseparable": {"type": "boolean"}
                }), &[])),
                "entity": {"type": "string"}, "prospective": {"type": "boolean"},
                "rule_multiplier": {"type": "number"}, "score": {"type": "number"},
                "inseparable": {"type": "boolean"}, "text": {"type": "string"}
            }),
            &[],
        ),
    );
    put(
        "Cause",
        obj(
            json!({"rank": {"type": "integer"}, "event": r("Event"), "score": {"type": "number"}, "explanation": r("Explanation")}),
            &["rank", "event", "score"],
        ),
    );
    put(
        "QueryStats",
        obj(
            json!({
                "segments_in_window": {"type": "integer"}, "segments_scanned": {"type": "integer"},
                "segments_pruned": {"type": "integer"}, "events_scanned": {"type": "integer"},
                "union_fallbacks": {"type": "integer"}
            }),
            &[],
        ),
    );
    put(
        "QueryResult",
        obj(
            json!({
                "entity": {"type": "string"}, "entities": arr(json!({"type": "string"})),
                "effect": {"type": "string"}, "at": {"type": "number"},
                "causes": arr(r("Cause")), "stats": r("QueryStats"), "text": {"type": "string"}
            }),
            &["causes", "stats"],
        ),
    );
    put(
        "RiskResult",
        obj(
            json!({
                "effect": {"type": "string"}, "entity": nullable(json!({"type": "string"})),
                "features": arr(json!({"type": "string"})), "type": nullable(json!({"type": "string"})),
                "score": {"type": "number"}, "explanation": r("Explanation")
            }),
            &["score"],
        ),
    );
    put(
        "Candidate",
        obj(
            json!({
                "id": {}, "features": arr(json!({"type": "string"})), "weight": nullable(json!({"type": "number"})),
                "type": nullable(json!({"type": "string"})), "attrs": nullable(json!({"type": "object"})),
                "entity": nullable(json!({"type": "string"}))
            }),
            &[],
        ),
    );
    put(
        "RankedCandidates",
        obj(
            json!({"candidates": arr(obj(json!({
                "rank": {"type": "integer"}, "index": {"type": "integer"}, "candidate": r("Candidate"),
                "score": {"type": "number"}, "entity": nullable(json!({"type": "string"})), "explanation": r("Explanation")
            }), &["rank", "candidate", "score"]))}),
            &["candidates"],
        ),
    );
    put(
        "FeedbackReport",
        obj(
            json!({"confirmed": nullable(json!({"type": "integer"})), "rejected": arr(json!({"type": "integer"})), "credit": {"type": "object", "additionalProperties": {"type": "number"}}}),
            &[],
        ),
    );
    put(
        "ObserveReport",
        obj(
            json!({
                "entity": {"type": "string"}, "effect": {"type": "string"}, "at": {"type": "number"},
                "leak": num(), "leak_responsibility": {"type": "number"}, "total_credit": {"type": "number"},
                "credits": arr(obj(json!({
                    "event_id": {"type": "integer"}, "score": {"type": "number"}, "responsibility": {"type": "number"},
                    "features": {"type": "object", "additionalProperties": {"type": "number"}}
                }), &[])),
                "text": {"type": "string"}
            }),
            &[],
        ),
    );
    put(
        "SettleReport",
        obj(
            json!({
                "until": {"type": "number"}, "events_settled": {"type": "integer"}, "events_rejected": {"type": "integer"},
                "rejected_weight": {"type": "number"}, "skipped_credited": {"type": "integer"},
                "skipped_observed": {"type": "integer"}, "per_effect": {"type": "object", "additionalProperties": {"type": "integer"}},
                "text": {"type": "string"}
            }),
            &[],
        ),
    );
    put(
        "LinkReport",
        obj(
            json!({
                "effect": {"type": "string"}, "entity": nullable(json!({"type": "string"})),
                "baseline": {"type": "number"}, "level": {"type": "number"},
                "rows": arr(obj(json!({
                    "feature": {"type": "string"}, "strength": {"type": "number"}, "alpha": {"type": "number"},
                    "beta": {"type": "number"}, "n": {"type": "number"}, "lo": {"type": "number"},
                    "hi": {"type": "number"}, "lift": {"type": "number"}
                }), &[])),
                "text": {"type": "string"}
            }),
            &[],
        ),
    );
    put(
        "CalibrationReport",
        obj(
            json!({
                "effect": {"type": "string"}, "n": {"type": "integer"}, "brier": nullable(json!({"type": "number"})),
                "bins": arr(obj(json!({
                    "lo": {"type": "number"}, "hi": {"type": "number"}, "count": {"type": "integer"},
                    "mean_predicted": nullable(json!({"type": "number"})), "observed_rate": nullable(json!({"type": "number"}))
                }), &[])),
                "text": {"type": "string"}
            }),
            &[],
        ),
    );
    put(
        "KernelSpec",
        obj(
            json!({
                "type": {"type": "string", "enum": ["exponential", "exp", "window", "lognormal", "gamma"]},
                "params": {"type": "object", "description": "half_life, horizon_half_lives | min_lag, max_lag | median, sigma, tail_sigmas | shape, scale, tail_stds", "additionalProperties": {"type": "number"}}
            }),
            &["type"],
        ),
    );
    put(
        "EffectRequest",
        obj(
            json!({
                "name": {"type": "string"}, "kernel": r("KernelSpec"),
                "priors": {"type": "object", "additionalProperties": arr(json!({"type": "number"}))},
                "default_prior": arr(json!({"type": "number"})), "cause_types": arr(json!({"type": "string"})),
                "credit": {"type": "string", "enum": ["proportional", "uniform"]},
                "global_pseudocount": {"type": "number"}, "hierarchical": {"type": "boolean"},
                "leak": num(), "w_neg": {"type": "number"}, "repeatable": {"type": "boolean"}
            }),
            &["name", "kernel"],
        ),
    );
    put(
        "WeightSpec",
        obj(
            json!({
                "kind": {"type": "string", "enum": ["log1p", "attr", "const"]},
                "attr": {"type": "string"},
                "value": {"type": "number", "description": "const: the weight; attr/log1p: value used when the attribute is missing"}
            }),
            &["kind"],
        ),
    );
    put(
        "Predicate",
        obj(
            json!({
                "attr": {"type": "string"},
                "op": {"type": "string", "enum": ["eq", "ne", "lt", "lte", "le", "gt", "gte", "ge", "in", "not_in", "contains", "exists", "missing"]},
                "value": {}, "default": {"description": "value used when the attribute is missing"}
            }),
            &["attr", "op"],
        ),
    );
    put(
        "DatabaseInfo",
        obj(
            json!({
                "database_id": {"type": "string"}, "org_id": {"type": "string"}, "plan": {"type": "string"},
                "suspended": {"type": "boolean"},
                "usage": obj(json!({
                    "requests_this_period": {"type": "integer"}, "period_start": {"type": "integer"},
                    "period_end": {"type": "integer"}, "events_stored": {"type": "integer"}, "storage_bytes": {"type": "integer"}
                }), &[]),
                "limits": obj(json!({
                    "max_events": nullable(json!({"type": "integer"})),
                    "max_requests_per_minute": nullable(json!({"type": "integer"})),
                    "max_storage_bytes": nullable(json!({"type": "integer"})),
                    "max_requests_per_month": nullable(json!({"type": "integer"})),
                    "billing_period_start": {"type": "integer"}
                }), &[])
            }),
            &["database_id", "plan", "usage", "limits"],
        ),
    );
    put("Ok", obj(json!({"ok": {"type": "boolean"}}), &["ok"]));
    Value::Object(s)
}

fn body(schema: Value) -> Value {
    json!({"required": true, "content": {"application/json": {"schema": schema}}})
}

fn resp(schema: Value) -> Value {
    json!({
        "200": {"description": "OK", "content": {"application/json": {"schema": schema}}},
        "default": {"description": "Error", "content": {"application/json": {"schema": r("Error")}}}
    })
}

fn op(summary: &str, scope: &str, req: Option<Value>, res: Value, params: Value) -> Value {
    let mut m = Map::new();
    m.insert("summary".into(), summary.into());
    m.insert("description".into(), format!("Required scope: {scope}.").into());
    if let Some(b) = req {
        m.insert("requestBody".into(), body(b));
    }
    m.insert("parameters".into(), params);
    m.insert("responses".into(), resp(res));
    Value::Object(m)
}

fn qp(name: &str, ty: &str, required: bool) -> Value {
    json!({"name": name, "in": "query", "required": required, "schema": {"type": ty}})
}

/// The OpenAPI document.
pub fn document() -> Value {
    let base = "/v1/databases/{database_id}";
    let none = json!([]);
    let mut paths = Map::new();
    let mut add = |p: &str, method: &str, v: Value| {
        let e = paths
            .entry(format!("{base}{p}"))
            .or_insert_with(|| json!({"parameters": [{"name": "database_id", "in": "path", "required": true, "schema": {"type": "string"}}]}));
        e.as_object_mut().expect("object").insert(method.into(), v);
    };
    add(
        "",
        "get",
        op(
            "Database usage and limits",
            "any",
            None,
            r("DatabaseInfo"),
            none.clone(),
        ),
    );
    add(
        "/effects",
        "post",
        op(
            "register_effect",
            "admin",
            Some(r("EffectRequest")),
            r("Ok"),
            none.clone(),
        ),
    );
    add(
        "/event-types",
        "post",
        op(
            "register_event_type (declarative weight functions only)",
            "admin",
            Some(obj(
                json!({"type": {"type": "string"}, "weight": r("WeightSpec")}),
                &["type"],
            )),
            r("Ok"),
            none.clone(),
        ),
    );
    add(
        "/rules",
        "post",
        op(
            "add_rule (declarative predicates only)",
            "admin",
            Some(obj(
                json!({
                    "effect": {"type": "string"}, "name": {"type": "string"},
                    "multiplier": {"type": "number", "minimum": 0, "maximum": 1},
                    "types": arr(json!({"type": "string"})), "predicate": r("Predicate")
                }),
                &["effect", "name", "multiplier"],
            )),
            r("Ok"),
            none.clone(),
        ),
    );
    add(
        "/events",
        "post",
        op(
            "insert (batched, at most 1000 events)",
            "write",
            Some(obj(
                json!({"events": json!({"type": "array", "maxItems": 1000, "items": r("EventInput")}), "return_events": {"type": "boolean"}}),
                &["events"],
            )),
            obj(
                json!({"ids": arr(json!({"type": "integer"})), "events": arr(r("Event"))}),
                &["ids"],
            ),
            none.clone(),
        ),
    );
    add(
        "/query/top-causes",
        "post",
        op(
            "top_causes / top_causes_group",
            "read",
            Some(obj(
                json!({
                    "entity": {"type": "string"}, "entities": arr(json!({"type": "string"})),
                    "effect": {"type": "string"}, "at": {"type": "number"}, "k": {"type": "integer", "minimum": 1},
                    "within": {"type": "number"}, "explain": {"type": "boolean"}
                }),
                &["effect", "at"],
            )),
            r("QueryResult"),
            none.clone(),
        ),
    );
    add(
        "/query/risk",
        "post",
        op(
            "feature_risk (features) or rank_candidates (candidates)",
            "read",
            Some(obj(
                json!({
                    "effect": {"type": "string"}, "features": arr(json!({"type": "string"})),
                    "weight": {"type": "number"}, "entity": {"type": "string"}, "type": {"type": "string"},
                    "attrs": {"type": "object"}, "explain": {"type": "boolean"},
                    "candidates": arr(r("Candidate")), "k": {"type": "integer", "minimum": 1}
                }),
                &["effect"],
            )),
            json!({"oneOf": [r("RiskResult"), r("RankedCandidates")]}),
            none.clone(),
        ),
    );
    add(
        "/feedback",
        "post",
        op(
            "feedback",
            "write",
            Some(obj(
                json!({
                    "entity": {"type": "string"}, "effect": {"type": "string"}, "at": {"type": "number"},
                    "confirmed_event_id": {"type": "integer"}, "rejected_ids": arr(json!({"type": "integer"})),
                    "credit": {"type": "string", "enum": ["proportional", "uniform"]}
                }),
                &["entity", "effect", "at"],
            )),
            r("FeedbackReport"),
            none.clone(),
        ),
    );
    add(
        "/observe",
        "post",
        op(
            "observe",
            "write",
            Some(obj(
                json!({"entity": {"type": "string"}, "effect": {"type": "string"}, "at": {"type": "number"}, "leak": num()}),
                &["entity", "effect", "at"],
            )),
            r("ObserveReport"),
            none.clone(),
        ),
    );
    add(
        "/settle",
        "post",
        op(
            "settle",
            "write",
            Some(obj(
                json!({
                    "until": {"type": "number"},
                    "effects": {"oneOf": [{"type": "string"}, arr(json!({"type": "string"}))]},
                    "entity": {"type": "string"}, "partial": {"type": "boolean"}
                }),
                &["until"],
            )),
            r("SettleReport"),
            none.clone(),
        ),
    );
    add(
        "/links",
        "get",
        op(
            "link_report",
            "read",
            None,
            r("LinkReport"),
            json!([
                qp("effect", "string", true),
                qp("entity", "string", false),
                qp("min_support", "number", false),
                qp("level", "number", false),
                qp("baseline", "string", false),
                qp("features", "string", false)
            ]),
        ),
    );
    add(
        "/calibration",
        "get",
        op(
            "calibration_report",
            "read",
            None,
            r("CalibrationReport"),
            json!([qp("effect", "string", true), qp("bins", "integer", false)]),
        ),
    );
    add(
        "/entities/{entity}",
        "delete",
        op(
            "forget_entity",
            "admin",
            None,
            obj(
                json!({"counts": obj(json!({
                    "events": {"type": "integer"}, "segments": {"type": "integer"},
                    "links": {"type": "integer"}, "calibration_records": {"type": "integer"}
                }), &[])}),
                &["counts"],
            ),
            json!([
                {"name": "entity", "in": "path", "required": true, "schema": {"type": "string"}},
                qp("scrub_global", "boolean", false)
            ]),
        ),
    );
    add(
        "/hql",
        "post",
        op(
            "Run one query-language statement",
            "read (TOP, RISK, LINKS, CALIBRATION) or write (INSERT, FEEDBACK, OBSERVE, SETTLE)",
            Some(obj(
                json!({"query": {"type": "string"}, "now": {"type": "number"}}),
                &["query"],
            )),
            obj(
                json!({
                    "kind": {"type": "string", "enum": ["top_causes", "calibration", "risk", "links", "observe", "settle", "message"]},
                    "result": {}, "text": {"type": "string"}
                }),
                &["kind", "result", "text"],
            ),
            none,
        ),
    );
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "HeapDB Data API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Ranked, explainable attribution over an append-only event stream. Errors are {\"error\": {\"code\", \"message\"}}; 429 and 503 carry Retry-After. Non-finite numbers are the strings Infinity, -Infinity and NaN.",
            "license": {"name": "Apache-2.0", "identifier": "Apache-2.0"}
        },
        "servers": [{"url": "https://api.heapdb.com"}],
        "security": [{"apiKey": []}],
        "paths": Value::Object(paths),
        "components": {
            "schemas": schemas(),
            "securitySchemes": {"apiKey": {"type": "http", "scheme": "bearer", "description": "hdb_live_... API key"}}
        }
    })
}
