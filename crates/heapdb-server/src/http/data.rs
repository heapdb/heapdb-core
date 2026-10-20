//! Data API (contract §4): `/v1/databases/{id}/...`.
//!
//! One handler parses the route, authenticates at the edge, then serves
//! locally or proxies the raw request to the owner node.

use std::sync::atomic::Ordering;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use heapdb::{Baseline, LinkReportOptions};
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::{bearer, percent_decode, query_pairs, AppState};
use crate::database::{now_secs, OpenDb};
use crate::error::{ApiError, ApiResult};
use crate::ids::is_database_id;
use crate::manager::{Need, Route};
use crate::schema::EffectReq;
use crate::state::{
    is_mutating, EventTypeReq, FeedbackReq, HqlReq, InsertReq, ObserveReq, RiskReq, RuleReq, SettleReq,
    TopCausesReq, WalOp, MAX_BATCH,
};
use crate::storage::{DbRecord, Scope};

/// Header marking a request forwarded by another node.
pub const FORWARDED_HEADER: &str = "x-heapdb-forwarded";

/// A Data API route.
#[derive(Clone, Debug, PartialEq)]
pub enum DataOp {
    /// `GET /`
    Info,
    /// `POST /effects`
    Effects,
    /// `POST /event-types`
    EventTypes,
    /// `POST /rules`
    Rules,
    /// `POST /events`
    Events,
    /// `POST /query/top-causes`
    TopCauses,
    /// `POST /query/risk`
    Risk,
    /// `POST /feedback`
    Feedback,
    /// `POST /observe`
    Observe,
    /// `POST /settle`
    Settle,
    /// `GET /links`
    Links,
    /// `GET /calibration`
    Calibration,
    /// `DELETE /entities/{entity}`
    Forget(String),
    /// `POST /hql`
    Hql,
}

fn method_not_allowed(method: &Method, path: &str) -> ApiError {
    ApiError::new(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        format!("{method} is not allowed on {path}"),
    )
}

/// Parse `(method, path after /v1/databases/{id})`.
pub fn parse_route(method: &Method, rest: &str) -> ApiResult<DataOp> {
    let r = rest.trim_end_matches('/');
    let want = |m: Method, op: DataOp| -> ApiResult<DataOp> {
        if *method == m {
            Ok(op)
        } else {
            Err(method_not_allowed(method, rest))
        }
    };
    if let Some(ent) = r.strip_prefix("/entities/") {
        if ent.is_empty() || ent.contains('/') {
            return Err(ApiError::not_found(format!("no route {rest}")));
        }
        return want(Method::DELETE, DataOp::Forget(percent_decode(ent)?));
    }
    match r {
        "" => want(Method::GET, DataOp::Info),
        "/effects" => want(Method::POST, DataOp::Effects),
        "/event-types" => want(Method::POST, DataOp::EventTypes),
        "/rules" => want(Method::POST, DataOp::Rules),
        "/events" => want(Method::POST, DataOp::Events),
        "/query/top-causes" => want(Method::POST, DataOp::TopCauses),
        "/query/risk" => want(Method::POST, DataOp::Risk),
        "/feedback" => want(Method::POST, DataOp::Feedback),
        "/observe" => want(Method::POST, DataOp::Observe),
        "/settle" => want(Method::POST, DataOp::Settle),
        "/links" => want(Method::GET, DataOp::Links),
        "/calibration" => want(Method::GET, DataOp::Calibration),
        "/hql" => want(Method::POST, DataOp::Hql),
        _ => Err(ApiError::not_found(format!("no route {rest}"))),
    }
}

fn body<T: DeserializeOwned>(b: &Bytes) -> ApiResult<T> {
    if b.is_empty() {
        return Err(ApiError::bad_request("request body must be a JSON object"));
    }
    serde_json::from_slice(b).map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))
}

fn need_of(op: &DataOp, b: &Bytes) -> ApiResult<Need> {
    Ok(match op {
        DataOp::Info => Need::Any,
        DataOp::Effects | DataOp::EventTypes | DataOp::Rules | DataOp::Forget(_) => Need::Scope(Scope::Admin),
        DataOp::Events | DataOp::Feedback | DataOp::Observe | DataOp::Settle => Need::Scope(Scope::Write),
        DataOp::TopCauses | DataOp::Risk | DataOp::Links | DataOp::Calibration => Need::Scope(Scope::Read),
        DataOp::Hql => {
            let req: HqlReq = body(b)?;
            let stmt = heapdb::parse(&req.query)?;
            Need::Scope(if is_mutating(&stmt) {
                Scope::Write
            } else {
                Scope::Read
            })
        }
    })
}

fn json_ok(v: Value) -> Response {
    (StatusCode::OK, axum::Json(v)).into_response()
}

/// The Data API handler (`/v1/databases/{id}` and everything below).
pub async fn handle(
    State(st): State<AppState>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    raw: Bytes,
) -> Response {
    match handle_inner(&st, &method, &uri, &headers, raw).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn handle_inner(
    st: &AppState,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    raw: Bytes,
) -> ApiResult<Response> {
    let path = uri.path();
    let after = path
        .strip_prefix("/v1/databases/")
        .ok_or_else(|| ApiError::not_found("no route"))?;
    let (id, rest) = match after.find('/') {
        Some(i) => (&after[..i], &after[i..]),
        None => (after, ""),
    };
    if !is_database_id(id) {
        return Err(ApiError::database_not_found(id));
    }
    let op = parse_route(method, rest)?;
    let need = need_of(&op, &raw)?;
    let (_, rec) = st.mgr.authenticate(id, bearer(headers), need).await?;
    let forwarded = headers.contains_key(FORWARDED_HEADER);
    match st.mgr.route(id, forwarded).await? {
        Route::Forward(addr) => forward(st, &addr, id, method, uri, headers, raw).await,
        Route::Local(db) => serve(&db, &rec, op, uri.query(), raw).await.map(json_ok),
    }
}

async fn serve(db: &OpenDb, rec: &DbRecord, op: DataOp, query: Option<&str>, raw: Bytes) -> ApiResult<Value> {
    if op == DataOp::Info {
        return db.info(rec).await;
    }
    db.admit(rec).await?;
    match op {
        DataOp::Info => unreachable!("handled above"),
        DataOp::Effects => {
            let req: EffectReq = body(&raw)?;
            db.write(WalOp::RegisterEffect(req)).await
        }
        DataOp::EventTypes => {
            let req: EventTypeReq = body(&raw)?;
            db.write(WalOp::RegisterEventType(req)).await
        }
        DataOp::Rules => {
            let req: RuleReq = body(&raw)?;
            db.write(WalOp::AddRule(req)).await
        }
        DataOp::Events => {
            let req: InsertReq = body(&raw)?;
            if req.events.is_empty() {
                return Err(ApiError::bad_request("events must not be empty"));
            }
            if req.events.len() > MAX_BATCH {
                return Err(ApiError::new(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "batch_too_large",
                    format!("at most {MAX_BATCH} events per request, got {}", req.events.len()),
                ));
            }
            db.check_write_quota(rec, req.events.len())?;
            let mut out = db.write(WalOp::Insert(req.events)).await?;
            if !req.return_events {
                if let Some(m) = out.as_object_mut() {
                    m.remove("events");
                }
            }
            Ok(out)
        }
        DataOp::TopCauses => {
            let req: TopCausesReq = body(&raw)?;
            db.read(|s| s.top_causes(&req)).await
        }
        DataOp::Risk => {
            let req: RiskReq = body(&raw)?;
            db.read(|s| s.risk(&req)).await
        }
        DataOp::Feedback => {
            let req: FeedbackReq = body(&raw)?;
            db.check_write_quota(rec, 0)?;
            db.write(WalOp::Feedback(req)).await
        }
        DataOp::Observe => {
            let req: ObserveReq = body(&raw)?;
            db.check_write_quota(rec, 0)?;
            db.write(WalOp::Observe(req)).await
        }
        DataOp::Settle => {
            let req: SettleReq = body(&raw)?;
            db.check_write_quota(rec, 0)?;
            db.write(WalOp::Settle(req)).await
        }
        DataOp::Links => {
            let q = query_pairs(query)?;
            let effect = q
                .get("effect")
                .cloned()
                .ok_or_else(|| ApiError::bad_request("effect is required"))?;
            let mut opts = LinkReportOptions {
                entity: q.get("entity").cloned().filter(|e| !e.is_empty()),
                ..Default::default()
            };
            if let Some(m) = q.get("min_support") {
                opts.min_support = parse_f64("min_support", m)?;
            }
            if let Some(l) = q.get("level") {
                opts.level = parse_f64("level", l)?;
            }
            if let Some(b) = q.get("baseline") {
                opts.baseline = match b.as_str() {
                    "mean" => Baseline::Mean,
                    "prior" => Baseline::Prior,
                    x => Baseline::Value(parse_f64("baseline", x)?),
                };
            }
            if let Some(f) = q.get("features") {
                opts.features = Some(
                    f.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect(),
                );
            }
            db.read(|s| s.links(&effect, &opts)).await
        }
        DataOp::Calibration => {
            let q = query_pairs(query)?;
            let effect = q
                .get("effect")
                .cloned()
                .ok_or_else(|| ApiError::bad_request("effect is required"))?;
            let bins = match q.get("bins") {
                Some(b) => b
                    .parse::<usize>()
                    .map_err(|_| ApiError::bad_request("bins must be a positive integer"))?,
                None => 10,
            };
            db.read(|s| s.calibration(&effect, bins)).await
        }
        DataOp::Forget(entity) => {
            let q = query_pairs(query)?;
            let scrub_global = match q.get("scrub_global").map(String::as_str) {
                None | Some("") | Some("false") | Some("0") => false,
                Some("true") | Some("1") => true,
                Some(x) => {
                    return Err(ApiError::bad_request(format!(
                        "scrub_global must be true or false, got {x:?}"
                    )))
                }
            };
            db.write(WalOp::Forget { entity, scrub_global }).await
        }
        DataOp::Hql => {
            let req: HqlReq = body(&raw)?;
            let now = req.now.unwrap_or_else(now_secs);
            let stmt = heapdb::parse(&req.query)?;
            if is_mutating(&stmt) {
                db.check_write_quota(rec, usize::from(matches!(stmt, heapdb::Statement::Insert(_))))?;
                db.write(WalOp::Hql {
                    query: req.query,
                    now,
                })
                .await
            } else {
                db.read(|s| s.hql_read(&stmt, now)).await
            }
        }
    }
}

fn parse_f64(name: &str, v: &str) -> ApiResult<f64> {
    v.parse::<f64>()
        .map_err(|_| ApiError::bad_request(format!("{name} must be a number, got {v:?}")))
}

async fn forward(
    st: &AppState,
    addr: &str,
    id: &str,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    raw: Bytes,
) -> ApiResult<Response> {
    let pq = uri.path_and_query().map_or(uri.path(), |p| p.as_str());
    let mut req = st
        .mgr
        .http
        .request(method.clone(), format!("http://{addr}{pq}"))
        .header(FORWARDED_HEADER, "1")
        .body(raw);
    for h in [header::AUTHORIZATION, header::CONTENT_TYPE, header::ACCEPT] {
        if let Some(v) = headers.get(&h) {
            req = req.header(h, v);
        }
    }
    if let Some(v) = headers.get("x-request-id") {
        req = req.header("x-request-id", v);
    }
    st.mgr.ctx.metrics.forwards.fetch_add(1, Ordering::Relaxed);
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            st.mgr
                .ctx
                .metrics
                .forward_failures
                .fetch_add(1, Ordering::Relaxed);
            st.mgr.forget_lease(id);
            tracing::warn!(database_id = %id, owner = %addr, error = %e, "forward failed");
            return Err(ApiError::unavailable(
                "owner_unreachable",
                "the node owning this database is unreachable; retry",
            ));
        }
    };
    let status = resp.status();
    let mut out = HeaderMap::new();
    for h in [header::CONTENT_TYPE, header::RETRY_AFTER] {
        if let Some(v) = resp.headers().get(&h) {
            out.insert(h, v.clone());
        }
    }
    let bytes = resp.bytes().await.map_err(|e| {
        st.mgr.forget_lease(id);
        ApiError::unavailable(
            "owner_unreachable",
            format!("reading the owner's response failed: {e}"),
        )
    })?;
    if !out.contains_key(header::CONTENT_TYPE) {
        out.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    }
    Ok((status, out, bytes).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes() {
        assert_eq!(parse_route(&Method::GET, "").unwrap(), DataOp::Info);
        assert_eq!(parse_route(&Method::GET, "/").unwrap(), DataOp::Info);
        assert_eq!(
            parse_route(&Method::POST, "/query/top-causes").unwrap(),
            DataOp::TopCauses
        );
        assert_eq!(
            parse_route(&Method::DELETE, "/entities/a%2Fb%20c").unwrap(),
            DataOp::Forget("a/b c".into())
        );
        assert_eq!(
            parse_route(&Method::GET, "/events").unwrap_err().code,
            "method_not_allowed"
        );
        assert_eq!(parse_route(&Method::GET, "/nope").unwrap_err().code, "not_found");
    }
}
