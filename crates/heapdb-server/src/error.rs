//! API errors: `{"error": {"code", "message"}}` with an HTTP status and,
//! for 429/503, a `Retry-After` header.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::storage::StoreError;

/// An error returned by the Data or Admin API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    /// HTTP status.
    pub status: StatusCode,
    /// Stable machine-readable code (see `docs/server.md`).
    pub code: &'static str,
    /// Human-readable message.
    pub message: String,
    /// Seconds for the `Retry-After` header.
    pub retry_after: Option<u64>,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}: {}", self.status.as_u16(), self.code, self.message)
    }
}

impl std::error::Error for ApiError {}

impl ApiError {
    /// A new error.
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        ApiError {
            status,
            code,
            message: message.into(),
            retry_after: None,
        }
    }

    /// Attach `Retry-After` (seconds, at least 1).
    pub fn retry_after(mut self, secs: u64) -> Self {
        self.retry_after = Some(secs.max(1));
        self
    }

    /// 400 `bad_request`: malformed JSON or parameters.
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "bad_request", msg)
    }

    /// 400 `invalid_argument`: rejected by the engine (bad value or type).
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_argument", msg)
    }

    /// 404 `not_found`.
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found", msg)
    }

    /// 404 `database_not_found`.
    pub fn database_not_found(id: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "database_not_found",
            format!("database {id} does not exist"),
        )
    }

    /// 500 `internal`.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", msg)
    }

    /// 503 with a retryable code and `Retry-After: 1`.
    pub fn unavailable(code: &'static str, msg: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, code, msg).retry_after(1)
    }

    /// True for errors a client may retry (429 except quota codes, 502-504).
    pub fn is_retryable(&self) -> bool {
        match self.status.as_u16() {
            429 => self.code == "rate_limited",
            502..=504 => true,
            _ => false,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({"error": {"code": self.code, "message": self.message}});
        let mut resp = (self.status, axum::Json(body)).into_response();
        if let Some(s) = self.retry_after {
            if let Ok(v) = HeaderValue::from_str(&s.to_string()) {
                resp.headers_mut().insert(header::RETRY_AFTER, v);
            }
        }
        resp
    }
}

impl From<heapdb::Error> for ApiError {
    fn from(e: heapdb::Error) -> Self {
        use heapdb::Error as E;
        match &e {
            E::UnknownEffect(_) => Self::new(StatusCode::NOT_FOUND, "unknown_effect", e.to_string()),
            E::UnknownEvent(_) => Self::new(StatusCode::NOT_FOUND, "unknown_event", e.to_string()),
            E::UnknownHeap(_) => Self::new(StatusCode::NOT_FOUND, "unknown_heap", e.to_string()),
            E::HeapConflict(_) => Self::new(StatusCode::CONFLICT, "heap_conflict", e.to_string()),
            E::Query(_) => Self::new(StatusCode::BAD_REQUEST, "query_error", e.to_string()),
            E::Invalid(_) | E::Type(_) | E::MissingArgument | E::Callback(_) => Self::invalid(e.to_string()),
            E::Snapshot(_) | E::Io(_) | E::Backend(_) => Self::internal(e.to_string()),
        }
    }
}

impl From<StoreError> for ApiError {
    fn from(e: StoreError) -> Self {
        Self::unavailable("storage_unavailable", e.to_string())
    }
}

/// `Result` with [`ApiError`].
pub type ApiResult<T> = Result<T, ApiError>;

/// Shorthand for building JSON error bodies in tests and docs.
pub fn error_body(code: &str, message: &str) -> serde_json::Value {
    json!({"error": {"code": code, "message": message}})
}
