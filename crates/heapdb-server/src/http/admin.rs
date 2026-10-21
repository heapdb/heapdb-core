//! Admin API (contract §5): `/admin/v1/...`, bearer `HEAPDB_ADMIN_TOKEN`
//! (compared in constant time). Also the internal owner-invalidation hook.

use std::collections::BTreeMap;

use axum::body::Bytes;
use axum::extract::{Path, RawQuery, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::Router;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::{bearer, query_pairs, AppState};
use crate::database::{db_prefix, now_secs};
use crate::error::{ApiError, ApiResult};
use crate::ids::{ct_eq, is_database_id, is_key_id, is_org_id, is_sha256_hex};
use crate::quota::billing_period;
use crate::storage::{DbRecord, KeyRecord, Limits, Scope, GAUGE_BUCKET};

/// Admin routes.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/admin/v1/databases", post(create_db))
        .route(
            "/admin/v1/databases/{id}",
            axum::routing::patch(patch_db).delete(delete_db).get(get_db),
        )
        .route(
            "/admin/v1/databases/{id}/keys/{key_id}",
            put(put_key).delete(delete_key),
        )
        .route("/admin/v1/usage", get(usage))
        .route("/admin/v1/health", get(health))
}

fn check_admin(st: &AppState, headers: &HeaderMap) -> ApiResult<()> {
    let Some(token) = st.mgr.ctx.cfg.admin_token.as_deref().filter(|t| !t.is_empty()) else {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "admin_disabled",
            "the Admin API is disabled (set HEAPDB_ADMIN_TOKEN)",
        ));
    };
    match bearer(headers) {
        Some(b) if ct_eq(b.as_bytes(), token.as_bytes()) => Ok(()),
        _ => Err(ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "invalid admin token",
        )),
    }
}

fn parse<T: for<'de> Deserialize<'de>>(b: &Bytes) -> ApiResult<T> {
    serde_json::from_slice(b).map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))
}

fn record_json(r: &DbRecord) -> Value {
    json!({
        "database_id": r.database_id,
        "org_id": r.org_id,
        "plan": r.plan,
        "limits": r.limits,
        "suspended": r.suspended,
        "created_at": r.created_at,
        "key_ids": r.keys.keys().collect::<Vec<_>>(),
    })
}

fn check_db_id(id: &str) -> ApiResult<()> {
    if is_database_id(id) {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "invalid database id {id:?} (expected db_ + 20 base62 characters)"
        )))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateReq {
    database_id: String,
    org_id: String,
    #[serde(default = "default_plan")]
    plan: String,
    #[serde(default)]
    limits: Limits,
}

fn default_plan() -> String {
    "free".into()
}

async fn create_db(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> ApiResult<Response> {
    check_admin(&st, &headers)?;
    let req: CreateReq = parse(&body)?;
    check_db_id(&req.database_id)?;
    if !is_org_id(&req.org_id) {
        return Err(ApiError::bad_request(format!("invalid org id {:?}", req.org_id)));
    }
    let rec = DbRecord {
        database_id: req.database_id.clone(),
        org_id: req.org_id,
        plan: req.plan,
        limits: req.limits,
        suspended: false,
        created_at: now_secs(),
        keys: BTreeMap::new(),
    };
    let reg = &st.mgr.ctx.backends.registry;
    match reg.create_database(&rec).await? {
        None => {
            st.mgr.invalidate(&rec.database_id, None, false);
            Ok((StatusCode::CREATED, axum::Json(record_json(&rec))).into_response())
        }
        Some(existing) if existing.org_id == rec.org_id => {
            Ok(axum::Json(record_json(&existing)).into_response())
        }
        Some(_) => Err(ApiError::new(
            StatusCode::CONFLICT,
            "database_exists",
            format!("database {} exists in another org", rec.database_id),
        )),
    }
}

async fn get_db(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    check_admin(&st, &headers)?;
    check_db_id(&id)?;
    match st.mgr.ctx.backends.registry.get_database(&id).await? {
        Some(r) => Ok(axum::Json(record_json(&r)).into_response()),
        None => Err(ApiError::database_not_found(&id)),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PatchReq {
    #[serde(default)]
    plan: Option<String>,
    /// Merged into the current limits; `null` values mean unlimited.
    #[serde(default)]
    limits: Option<Map<String, Value>>,
    #[serde(default)]
    suspended: Option<bool>,
}

async fn patch_db(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Bytes,
) -> ApiResult<Response> {
    check_admin(&st, &headers)?;
    check_db_id(&id)?;
    let req: PatchReq = parse(&body)?;
    let reg = &st.mgr.ctx.backends.registry;
    let mut rec = reg
        .get_database(&id)
        .await?
        .ok_or_else(|| ApiError::database_not_found(&id))?;
    if let Some(p) = req.plan {
        rec.plan = p;
    }
    if let Some(patch) = req.limits {
        let mut cur = match serde_json::to_value(&rec.limits) {
            Ok(Value::Object(m)) => m,
            _ => Map::new(),
        };
        for (k, v) in patch {
            if v.is_null() {
                cur.remove(&k);
            } else {
                cur.insert(k, v);
            }
        }
        rec.limits = serde_json::from_value(Value::Object(cur))
            .map_err(|e| ApiError::bad_request(format!("invalid limits: {e}")))?;
    }
    if let Some(s) = req.suspended {
        rec.suspended = s;
    }
    reg.put_database(&rec).await?;
    st.mgr.invalidate(&id, None, false);
    st.mgr.notify_owner(&id, None, false).await;
    Ok(axum::Json(record_json(&rec)).into_response())
}

async fn delete_db(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    check_admin(&st, &headers)?;
    check_db_id(&id)?;
    let b = st.mgr.ctx.backends.clone();
    let Some(rec) = b.registry.get_database(&id).await? else {
        return Ok(axum::Json(json!({"deleted": false})).into_response());
    };
    for sha in rec.keys.values() {
        b.registry.delete_key(sha).await?;
    }
    b.registry.delete_database(&id).await?;
    st.mgr.invalidate(&id, None, true);
    st.mgr.notify_owner(&id, None, true).await;
    if let Some(db) = st.mgr.get_open(&id) {
        st.mgr.close_now(db, false, true).await;
    }
    let prefix = db_prefix(&st.mgr.ctx.cfg.s3_prefix, &id);
    for key in b.objects.list(&prefix).await? {
        b.objects.delete(&key).await?;
    }
    b.usage.delete_gauges(&id).await?;
    tracing::info!(database_id = %id, "database deleted");
    Ok(axum::Json(json!({"deleted": true})).into_response())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyReq {
    sha256: String,
    scopes: Vec<Scope>,
    #[serde(default)]
    expires_at: Option<f64>,
}

async fn put_key(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path((id, key_id)): Path<(String, String)>,
    body: Bytes,
) -> ApiResult<Response> {
    check_admin(&st, &headers)?;
    check_db_id(&id)?;
    if !is_key_id(&key_id) {
        return Err(ApiError::bad_request(format!("invalid key id {key_id:?}")));
    }
    let req: KeyReq = parse(&body)?;
    let sha = req.sha256.to_ascii_lowercase();
    if !is_sha256_hex(&sha) {
        return Err(ApiError::bad_request("sha256 must be 64 hex digits"));
    }
    if req.scopes.is_empty() {
        return Err(ApiError::bad_request("scopes must not be empty"));
    }
    let reg = &st.mgr.ctx.backends.registry;
    let mut rec = reg
        .get_database(&id)
        .await?
        .ok_or_else(|| ApiError::database_not_found(&id))?;
    if let Some(other) = reg.get_key(&sha).await? {
        if other.database_id != id || other.key_id != key_id {
            return Err(ApiError::new(
                StatusCode::CONFLICT,
                "key_exists",
                "a key with this hash is already registered",
            ));
        }
    }
    let old = rec.keys.insert(key_id.clone(), sha.clone());
    if let Some(old) = old.as_ref().filter(|o| **o != sha) {
        reg.delete_key(old).await?;
    }
    let mut scopes = req.scopes;
    scopes.sort();
    scopes.dedup();
    let key = KeyRecord {
        key_id,
        sha256: sha.clone(),
        database_id: id.clone(),
        scopes,
        expires_at: req.expires_at,
    };
    reg.put_key(&key).await?;
    reg.put_database(&rec).await?;
    st.mgr.invalidate(&id, Some(&sha), false);
    if let Some(old) = &old {
        st.mgr.invalidate(&id, Some(old), false);
        st.mgr.notify_owner(&id, Some(old), false).await;
    }
    Ok(axum::Json(serde_json::to_value(&key).unwrap_or_default()).into_response())
}

async fn delete_key(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path((id, key_id)): Path<(String, String)>,
) -> ApiResult<Response> {
    check_admin(&st, &headers)?;
    check_db_id(&id)?;
    let reg = &st.mgr.ctx.backends.registry;
    let Some(mut rec) = reg.get_database(&id).await? else {
        return Err(ApiError::database_not_found(&id));
    };
    let Some(sha) = rec.keys.remove(&key_id) else {
        return Ok(axum::Json(json!({"revoked": false})).into_response());
    };
    reg.delete_key(&sha).await?;
    reg.put_database(&rec).await?;
    st.mgr.invalidate(&id, Some(&sha), false);
    st.mgr.notify_owner(&id, Some(&sha), false).await;
    Ok(axum::Json(json!({"revoked": true})).into_response())
}

async fn usage(State(st): State<AppState>, headers: HeaderMap, RawQuery(q): RawQuery) -> ApiResult<Response> {
    check_admin(&st, &headers)?;
    let q = query_pairs(q.as_deref())?;
    let since: f64 = match q.get("since") {
        Some(s) => s
            .parse()
            .map_err(|_| ApiError::bad_request("since must be Unix seconds"))?,
        None => 0.0,
    };
    // Counts held by databases open here are flushed first.
    for db in st.mgr.open_databases() {
        db.flush_usage().await;
    }
    let b = &st.mgr.ctx.backends;
    let items = b.usage.scan().await?;
    let records: BTreeMap<String, DbRecord> = b
        .registry
        .list_databases()
        .await?
        .into_iter()
        .map(|r| (r.database_id.clone(), r))
        .collect();
    let now = now_secs() as i64;
    let since_hour = (since.max(0.0) as i64).div_euclid(3600) * 3600;
    #[derive(Default)]
    struct Row {
        requests: u64,
        events_written: u64,
        events_stored: u64,
        storage_bytes: u64,
        periods: BTreeMap<i64, u64>,
    }
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    for id in records.keys() {
        rows.entry(id.clone()).or_default();
    }
    for it in items {
        let row = rows.entry(it.database_id.clone()).or_default();
        if it.bucket == GAUGE_BUCKET {
            row.events_stored = it.events_stored;
            row.storage_bytes = it.storage_bytes;
        } else if let Some(h) = it.bucket.strip_prefix("h#").and_then(|h| h.parse::<i64>().ok()) {
            if h >= since_hour {
                row.requests += it.requests;
                row.events_written += it.events_written;
            }
        } else if let Some(p) = it.bucket.strip_prefix("p#").and_then(|p| p.parse::<i64>().ok()) {
            row.periods.insert(p, it.requests);
        }
    }
    let out: Vec<Value> = rows
        .into_iter()
        .map(|(id, r)| {
            let anchor = records.get(&id).and_then(|x| x.limits.billing_period_start);
            let (ps, pe) = billing_period(anchor, now);
            json!({
                "database_id": id,
                "requests": r.requests,
                "events_written": r.events_written,
                "events_stored": r.events_stored,
                "storage_bytes": r.storage_bytes,
                "requests_this_period": r.periods.get(&ps).copied().unwrap_or(0),
                "period_start": ps,
                "period_end": pe,
            })
        })
        .collect();
    Ok(axum::Json(json!({
        "since": since,
        "generated_at": now_secs(),
        "granularity_seconds": 3600,
        "databases": out,
    }))
    .into_response())
}

async fn health(State(st): State<AppState>, headers: HeaderMap) -> ApiResult<Response> {
    check_admin(&st, &headers)?;
    let c = &st.mgr.ctx;
    Ok(axum::Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "node_id": c.node_id,
        "node_addr": c.node_addr,
        "cluster": match c.cfg.cluster { crate::config::ClusterMode::Off => "off", crate::config::ClusterMode::Aws => "aws" },
        "open_databases": st.mgr.open_databases().len(),
    }))
    .into_response())
}

#[derive(Deserialize, Default)]
struct InvalidateReq {
    #[serde(default)]
    key_sha256: Option<String>,
    #[serde(default)]
    drop: bool,
}

/// Internal: another node changed the registry for a database we own.
pub async fn invalidate(
    State(st): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    body: Bytes,
) -> ApiResult<Response> {
    if st
        .mgr
        .ctx
        .cfg
        .admin_token
        .as_deref()
        .is_some_and(|t| !t.is_empty())
    {
        check_admin(&st, &headers)?;
    }
    let req: InvalidateReq = if body.is_empty() {
        InvalidateReq::default()
    } else {
        parse(&body)?
    };
    st.mgr.invalidate(&id, req.key_sha256.as_deref(), req.drop);
    if req.drop {
        if let Some(db) = st.mgr.get_open(&id) {
            st.mgr.close_now(db, false, true).await;
        }
    }
    Ok(axum::Json(json!({"ok": true})).into_response())
}
