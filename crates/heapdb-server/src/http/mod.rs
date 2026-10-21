//! HTTP routers and middleware.
//!
//! * Public listener (`HEAPDB_LISTEN`): Data API, Admin API, OpenAPI
//!   document, `/healthz`, `/readyz`.
//! * Internal listener (`HEAPDB_INTERNAL_LISTEN`, node security group
//!   only): forwarded Data API requests, `/metrics`, owner invalidation,
//!   health.
//!
//! Every response carries `x-request-id`; errors are always
//! `{"error": {"code", "message"}}`; `/v1/*` answers CORS preflights.

use std::collections::HashMap;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use serde_json::json;

use crate::error::{ApiError, ApiResult};
use crate::manager::Manager;
use std::sync::Arc;

pub mod admin;
pub mod data;
pub mod openapi;

/// Shared handler state.
#[derive(Clone)]
pub struct AppState {
    /// The node's database manager.
    pub mgr: Arc<Manager>,
}

/// `Authorization: Bearer <token>`.
pub fn bearer(headers: &HeaderMap) -> Option<&str> {
    let v = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, rest) = v.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| rest.trim())
}

/// Percent-decode a path segment or query component.
pub fn percent_decode(s: &str) -> ApiResult<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                let hex = std::str::from_utf8(&b[i + 1..i + 3])
                    .map_err(|_| ApiError::bad_request("bad percent-encoding"))?;
                let v =
                    u8::from_str_radix(hex, 16).map_err(|_| ApiError::bad_request("bad percent-encoding"))?;
                out.push(v);
                i += 3;
            }
            b'%' => return Err(ApiError::bad_request("bad percent-encoding")),
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| ApiError::bad_request("percent-encoding is not UTF-8"))
}

/// Parse a query string (`+` is a space; later keys win).
pub fn query_pairs(q: Option<&str>) -> ApiResult<HashMap<String, String>> {
    let mut m = HashMap::new();
    for part in q.unwrap_or("").split('&').filter(|p| !p.is_empty()) {
        let (k, v) = part.split_once('=').unwrap_or((part, ""));
        m.insert(
            percent_decode(&k.replace('+', " "))?,
            percent_decode(&v.replace('+', " "))?,
        );
    }
    Ok(m)
}

fn code_for_status(s: StatusCode) -> &'static str {
    match s.as_u16() {
        400 => "bad_request",
        401 => "unauthorized",
        403 => "forbidden",
        404 => "not_found",
        405 => "method_not_allowed",
        408 => "timeout",
        409 => "conflict",
        413 => "payload_too_large",
        415 => "unsupported_media_type",
        422 => "bad_request",
        429 => "rate_limited",
        503 => "unavailable",
        _ if s.is_server_error() => "internal",
        _ => "error",
    }
}

fn new_request_id() -> String {
    use rand::Rng;
    let n: u128 = rand::rng().random();
    format!("req_{n:032x}")
}

const CORS_ALLOW_HEADERS: &str = "Authorization, Content-Type, Accept, User-Agent, X-Request-Id";
const CORS_EXPOSE_HEADERS: &str = "x-request-id, retry-after";
const CORS_METHODS: &str = "GET, POST, PUT, PATCH, DELETE, OPTIONS";

async fn common(State(st): State<AppState>, req: Request, next: Next) -> Response {
    let started = Instant::now();
    let rid = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty() && s.len() <= 128 && s.bytes().all(|b| b.is_ascii_graphic()))
        .map(str::to_string)
        .unwrap_or_else(new_request_id);
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let surface = if path.starts_with("/v1/databases/") {
        "data"
    } else if path.starts_with("/admin/") {
        "admin"
    } else if path.starts_with("/internal/") {
        "internal"
    } else {
        "other"
    };
    let is_data = path.starts_with("/v1/");
    let mut resp = if is_data && method == Method::OPTIONS {
        let mut r = StatusCode::NO_CONTENT.into_response();
        let h = r.headers_mut();
        h.insert(
            "access-control-allow-methods",
            HeaderValue::from_static(CORS_METHODS),
        );
        h.insert(
            "access-control-allow-headers",
            HeaderValue::from_static(CORS_ALLOW_HEADERS),
        );
        h.insert("access-control-max-age", HeaderValue::from_static("86400"));
        r
    } else {
        next.run(req).await
    };
    // Errors are always JSON.
    let status = resp.status();
    let is_json = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"));
    if (status.is_client_error() || status.is_server_error()) && !is_json {
        let retry = resp.headers().get(header::RETRY_AFTER).cloned();
        let body = std::mem::replace(resp.body_mut(), Body::empty());
        let text = axum::body::to_bytes(body, 64 * 1024).await.unwrap_or_default();
        let msg = String::from_utf8_lossy(&text).trim().to_string();
        let msg = if msg.is_empty() {
            status.canonical_reason().unwrap_or("error").to_string()
        } else {
            msg
        };
        let mut e = ApiError::new(status, code_for_status(status), msg);
        if let Some(r) = retry.and_then(|v| v.to_str().ok().and_then(|s| s.parse().ok())) {
            e = e.retry_after(r);
        }
        resp = e.into_response();
    }
    let h = resp.headers_mut();
    if let Ok(v) = HeaderValue::from_str(&rid) {
        h.insert("x-request-id", v);
    }
    if is_data {
        h.insert("access-control-allow-origin", HeaderValue::from_static("*"));
        h.insert(
            "access-control-expose-headers",
            HeaderValue::from_static(CORS_EXPOSE_HEADERS),
        );
    }
    let status = resp.status().as_u16();
    st.mgr.ctx.metrics.http(surface, status);
    tracing::info!(
        request_id = %rid,
        method = %method,
        path = %path,
        status,
        latency_ms = started.elapsed().as_secs_f64() * 1000.0,
        "request"
    );
    resp
}

async fn healthz() -> Response {
    axum::Json(json!({"status": "ok", "version": env!("CARGO_PKG_VERSION")})).into_response()
}

async fn readyz(State(st): State<AppState>) -> Response {
    let b = &st.mgr.ctx.backends;
    let (o, l, r) = tokio::join!(b.objects.ping(), b.leases.ping(), b.registry.ping());
    let errs: Vec<String> = [o, l, r]
        .into_iter()
        .filter_map(|x| x.err())
        .map(|e| e.to_string())
        .collect();
    if errs.is_empty() {
        axum::Json(json!({"ready": true})).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({"ready": false, "errors": errs})),
        )
            .into_response()
    }
}

async fn metrics(State(st): State<AppState>) -> Response {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        st.mgr.ctx.metrics.render(),
    )
        .into_response()
}

async fn openapi_json() -> Response {
    axum::Json(openapi::document()).into_response()
}

async fn fallback() -> ApiError {
    ApiError::not_found("no such route")
}

fn data_routes() -> Router<AppState> {
    Router::new()
        .route("/v1/databases/{id}", any(data::handle))
        .route("/v1/databases/{id}/", any(data::handle))
        .route("/v1/databases/{id}/{*rest}", any(data::handle))
}

/// Router of the public listener.
pub fn public_router(st: AppState) -> Router {
    let limit = st.mgr.ctx.cfg.max_body_bytes;
    data_routes()
        .merge(admin::routes())
        .route("/v1/openapi.json", get(openapi_json))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(limit))
        .layer(middleware::from_fn_with_state(st.clone(), common))
        .with_state(st)
}

/// Router of the internal listener.
pub fn internal_router(st: AppState) -> Router {
    let limit = st.mgr.ctx.cfg.max_body_bytes;
    data_routes()
        .route("/internal/v1/databases/{id}/invalidate", post(admin::invalidate))
        .route("/metrics", get(metrics))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(limit))
        .layer(middleware::from_fn_with_state(st.clone(), common))
        .with_state(st)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoding() {
        assert_eq!(percent_decode("a%2Fb%20c").unwrap(), "a/b c");
        assert_eq!(percent_decode("plain").unwrap(), "plain");
        assert!(percent_decode("bad%2").is_err());
        assert!(percent_decode("bad%zz").is_err());
        let q = query_pairs(Some("effect=bloat+ing&entity=a%26b&x")).unwrap();
        assert_eq!(q["effect"], "bloat ing");
        assert_eq!(q["entity"], "a&b");
        assert_eq!(q["x"], "");
    }
}
