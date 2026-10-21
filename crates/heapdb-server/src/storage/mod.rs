//! Storage backends behind traits.
//!
//! * [`ObjectStore`]: WAL segments and snapshots (S3, local disk, memory).
//! * [`LeaseStore`]: per-database ownership leases with epoch fencing
//!   (DynamoDB, memory, memory persisted to a file).
//! * [`Registry`]: databases (plan, limits, suspended) and API key hashes.
//! * [`UsageStore`]: durable, additive usage counters.
//!
//! Single-node mode uses [`local::LocalObjectStore`] plus the in-memory
//! stores persisted to JSON files; cluster mode uses `aws` (feature
//! `cluster-aws`); tests use the in-memory stores, which implement the
//! same conditional semantics as DynamoDB and S3.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

#[cfg(feature = "cluster-aws")]
pub mod aws;
pub mod local;
pub mod memory;

/// A storage failure.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum StoreError {
    /// A conditional write lost (key exists, epoch changed, ...).
    #[error("conditional write failed: {0}")]
    Condition(String),
    /// The backend could not be reached or failed.
    #[error("storage unavailable: {0}")]
    Unavailable(String),
    /// Stored data could not be decoded.
    #[error("corrupt stored data: {0}")]
    Corrupt(String),
}

/// `Result` with [`StoreError`].
pub type StoreResult<T> = Result<T, StoreError>;

/// Blob storage with put-if-absent (S3 `If-None-Match: *`).
#[async_trait]
pub trait ObjectStore: Send + Sync {
    /// Write `key` unless it exists (`StoreError::Condition` if it does).
    /// Returns only once the object is durable.
    async fn put_if_absent(&self, key: &str, body: Bytes) -> StoreResult<()>;
    /// Read an object.
    async fn get(&self, key: &str) -> StoreResult<Option<Bytes>>;
    /// Keys under `prefix`, sorted.
    async fn list(&self, prefix: &str) -> StoreResult<Vec<String>>;
    /// Delete (missing keys are fine).
    async fn delete(&self, key: &str) -> StoreResult<()>;
    /// Reachability check for `/readyz`.
    async fn ping(&self) -> StoreResult<()>;
}

/// Ownership lease of one database (DynamoDB table `heapdb-leases`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    /// Database id (partition key).
    pub database_id: String,
    /// Owning node id.
    pub owner_node: String,
    /// Owner's private `ip:port` (internal listener).
    pub owner_addr: String,
    /// Monotonic fencing token.
    pub epoch: u64,
    /// Expiry, Unix milliseconds.
    pub expires_at_ms: u64,
}

impl Lease {
    /// Not expired at `now_ms`.
    pub fn live(&self, now_ms: u64) -> bool {
        self.expires_at_ms > now_ms
    }
}

/// Leases with conditional writes on `(epoch, expires_at)`.
#[async_trait]
pub trait LeaseStore: Send + Sync {
    /// Strongly consistent read.
    async fn get(&self, database_id: &str) -> StoreResult<Option<Lease>>;
    /// Take the lease with `epoch = prev.epoch + 1` (or 1), conditional on
    /// the item still being exactly `prev` (or absent). The caller checks
    /// that `prev` is expired. `StoreError::Condition` if another node won.
    async fn acquire(
        &self,
        database_id: &str,
        node: &str,
        addr: &str,
        prev: Option<&Lease>,
        expires_at_ms: u64,
    ) -> StoreResult<Lease>;
    /// Extend a held lease, conditional on owner and epoch.
    async fn renew(&self, lease: &Lease, expires_at_ms: u64) -> StoreResult<Lease>;
    /// Give a lease up (expire it now), conditional on owner and epoch.
    async fn release(&self, lease: &Lease) -> StoreResult<()>;
    /// Reachability check for `/readyz`.
    async fn ping(&self) -> StoreResult<()>;
}

/// Limits of a database (contract §5).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    /// Maximum stored events (`None`: unlimited).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_events: Option<u64>,
    /// Token bucket size and refill per minute (`None`: unlimited).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_requests_per_minute: Option<u64>,
    /// Maximum storage bytes (`None`: unlimited).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_storage_bytes: Option<u64>,
    /// Data API requests per billing period (`None`: unlimited).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_requests_per_month: Option<u64>,
    /// Start of the billing period anchor, Unix seconds (`None`: first day
    /// of the current UTC month).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub billing_period_start: Option<i64>,
    /// Control-plane only; stored and ignored by the data plane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_databases_per_org: Option<u64>,
}

/// A database in the registry (`pk = db#<id>`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DbRecord {
    /// `db_...`.
    pub database_id: String,
    /// `org_...`.
    pub org_id: String,
    /// Plan name (informational for the data plane).
    pub plan: String,
    /// Limits.
    pub limits: Limits,
    /// Suspended databases answer 403 `database_suspended`.
    pub suspended: bool,
    /// Unix seconds.
    pub created_at: f64,
    /// key_id -> sha256 of the database's keys.
    #[serde(default)]
    pub keys: BTreeMap<String, String>,
}

/// Permission scopes of an API key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Queries and reports.
    Read,
    /// Inserts and learning (feedback, observe, settle).
    Write,
    /// Schema calls: effects, event types, rules, forget_entity.
    Admin,
}

/// An API key (`pk = key#<sha256>`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KeyRecord {
    /// `key_...`.
    pub key_id: String,
    /// Lowercase hex SHA-256 of the key.
    pub sha256: String,
    /// Database the key belongs to.
    pub database_id: String,
    /// Granted scopes.
    pub scopes: Vec<Scope>,
    /// Expiry, Unix seconds.
    #[serde(default)]
    pub expires_at: Option<f64>,
}

/// Database and key registry.
#[async_trait]
pub trait Registry: Send + Sync {
    /// Read a database record.
    async fn get_database(&self, id: &str) -> StoreResult<Option<DbRecord>>;
    /// Create unless present; returns the existing record if there is one.
    async fn create_database(&self, rec: &DbRecord) -> StoreResult<Option<DbRecord>>;
    /// Replace a database record.
    async fn put_database(&self, rec: &DbRecord) -> StoreResult<()>;
    /// Delete a database record (keys are deleted by the caller).
    async fn delete_database(&self, id: &str) -> StoreResult<()>;
    /// All database records.
    async fn list_databases(&self) -> StoreResult<Vec<DbRecord>>;
    /// Look a key up by hash.
    async fn get_key(&self, sha256: &str) -> StoreResult<Option<KeyRecord>>;
    /// Store a key.
    async fn put_key(&self, rec: &KeyRecord) -> StoreResult<()>;
    /// Delete a key by hash.
    async fn delete_key(&self, sha256: &str) -> StoreResult<()>;
    /// Reachability check for `/readyz`.
    async fn ping(&self) -> StoreResult<()>;
}

/// One usage item: hourly bucket `h#<unix hour start>`, billing period
/// `p#<period start>`, or `gauge` (current stored events / bytes).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageItem {
    /// Database id.
    pub database_id: String,
    /// Bucket name.
    pub bucket: String,
    /// Requests counted.
    #[serde(default)]
    pub requests: u64,
    /// Events written.
    #[serde(default)]
    pub events_written: u64,
    /// Gauge: events stored.
    #[serde(default)]
    pub events_stored: u64,
    /// Gauge: storage bytes.
    #[serde(default)]
    pub storage_bytes: u64,
}

/// Durable usage counters (DynamoDB `heapdb-usage`). Counter updates are
/// atomic additions, so late flushes from a fenced node never lose or
/// overwrite counts.
#[async_trait]
pub trait UsageStore: Send + Sync {
    /// `requests += r, events_written += w` on one bucket.
    async fn add(
        &self,
        database_id: &str,
        bucket: &str,
        requests: u64,
        events_written: u64,
    ) -> StoreResult<()>;
    /// Set the `gauge` bucket.
    async fn set_gauges(&self, database_id: &str, events_stored: u64, storage_bytes: u64) -> StoreResult<()>;
    /// Read one bucket.
    async fn get(&self, database_id: &str, bucket: &str) -> StoreResult<Option<UsageItem>>;
    /// Every item (for `GET /admin/v1/usage`).
    async fn scan(&self) -> StoreResult<Vec<UsageItem>>;
    /// Remove a database's gauge (on delete).
    async fn delete_gauges(&self, database_id: &str) -> StoreResult<()>;
}

/// All backends of a node.
#[derive(Clone)]
pub struct Backends {
    /// WAL + snapshots.
    pub objects: Arc<dyn ObjectStore>,
    /// Leases.
    pub leases: Arc<dyn LeaseStore>,
    /// Registry.
    pub registry: Arc<dyn Registry>,
    /// Usage.
    pub usage: Arc<dyn UsageStore>,
}

impl Backends {
    /// Fresh in-memory backends (tests, fake cluster).
    pub fn memory() -> Self {
        Backends {
            objects: Arc::new(memory::MemObjectStore::default()),
            leases: Arc::new(memory::MemLeaseStore::default()),
            registry: Arc::new(memory::MemRegistry::default()),
            usage: Arc::new(memory::MemUsageStore::default()),
        }
    }
}

/// Hourly usage bucket name.
pub fn hour_bucket(unix_secs: i64) -> String {
    format!("h#{}", unix_secs.div_euclid(3600) * 3600)
}

/// Billing period bucket name.
pub fn period_bucket(period_start: i64) -> String {
    format!("p#{period_start}")
}

/// Gauge bucket name.
pub const GAUGE_BUCKET: &str = "gauge";
