//! In-memory backends with the conditional semantics of S3 and DynamoDB.
//!
//! Used by tests (including the in-process cluster tests, where several
//! nodes share one instance) and, persisted to JSON files, as the
//! registry/lease/usage stores of single-node mode.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use super::{
    DbRecord, KeyRecord, Lease, LeaseStore, ObjectStore, Registry, StoreError, StoreResult, UsageItem,
    UsageStore, GAUGE_BUCKET,
};
use crate::storage::local::{load_json, save_json_atomic};

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

/// In-memory object store (a fake S3 bucket).
#[derive(Default)]
pub struct MemObjectStore {
    objects: Mutex<BTreeMap<String, Bytes>>,
    /// Fault injection: fail every put.
    pub fail_puts: AtomicBool,
    /// Number of successful puts.
    pub puts: AtomicU64,
}

impl MemObjectStore {
    /// Keys currently stored (sorted).
    pub fn keys(&self) -> Vec<String> {
        lock(&self.objects).keys().cloned().collect()
    }
}

#[async_trait]
impl ObjectStore for MemObjectStore {
    async fn put_if_absent(&self, key: &str, body: Bytes) -> StoreResult<()> {
        if self.fail_puts.load(Ordering::SeqCst) {
            return Err(StoreError::Unavailable("injected put failure".into()));
        }
        let mut m = lock(&self.objects);
        if m.contains_key(key) {
            return Err(StoreError::Condition(format!("object {key} exists")));
        }
        m.insert(key.to_string(), body);
        self.puts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn get(&self, key: &str) -> StoreResult<Option<Bytes>> {
        Ok(lock(&self.objects).get(key).cloned())
    }

    async fn list(&self, prefix: &str) -> StoreResult<Vec<String>> {
        Ok(lock(&self.objects)
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .map(|(k, _)| k.clone())
            .collect())
    }

    async fn delete(&self, key: &str) -> StoreResult<()> {
        lock(&self.objects).remove(key);
        Ok(())
    }

    async fn ping(&self) -> StoreResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Leases
// ---------------------------------------------------------------------------

/// In-memory lease table (a fake DynamoDB `heapdb-leases`), optionally
/// persisted to a JSON file.
#[derive(Default)]
pub struct MemLeaseStore {
    leases: Mutex<HashMap<String, Lease>>,
    file: Option<PathBuf>,
    /// Fault injection: fail every call.
    pub unavailable: AtomicBool,
}

impl MemLeaseStore {
    /// Persisted to `path`. Single-node mode: leases left by a previous
    /// process are expired at once (keeping their epochs), because there is
    /// no other node that could still hold them.
    pub fn persistent(path: PathBuf) -> std::io::Result<Self> {
        let mut leases: HashMap<String, Lease> = load_json(&path)?.unwrap_or_default();
        for l in leases.values_mut() {
            l.expires_at_ms = 0;
        }
        let s = MemLeaseStore {
            leases: Mutex::new(leases),
            file: Some(path),
            unavailable: AtomicBool::new(false),
        };
        s.save(&lock(&s.leases))?;
        Ok(s)
    }

    fn save(&self, m: &HashMap<String, Lease>) -> std::io::Result<()> {
        match &self.file {
            Some(p) => save_json_atomic(p, m, true),
            None => Ok(()),
        }
    }

    fn check(&self) -> StoreResult<()> {
        if self.unavailable.load(Ordering::SeqCst) {
            Err(StoreError::Unavailable("injected lease store failure".into()))
        } else {
            Ok(())
        }
    }

    /// Overwrite a lease (tests: simulate clock skew / manual expiry).
    pub fn force(&self, lease: Lease) {
        lock(&self.leases).insert(lease.database_id.clone(), lease);
    }
}

fn io_err(e: std::io::Error) -> StoreError {
    StoreError::Unavailable(e.to_string())
}

#[async_trait]
impl LeaseStore for MemLeaseStore {
    async fn get(&self, database_id: &str) -> StoreResult<Option<Lease>> {
        self.check()?;
        Ok(lock(&self.leases).get(database_id).cloned())
    }

    async fn acquire(
        &self,
        database_id: &str,
        node: &str,
        addr: &str,
        prev: Option<&Lease>,
        expires_at_ms: u64,
    ) -> StoreResult<Lease> {
        self.check()?;
        let mut m = lock(&self.leases);
        let cur = m.get(database_id);
        if cur != prev {
            return Err(StoreError::Condition(format!("lease of {database_id} changed")));
        }
        let lease = Lease {
            database_id: database_id.to_string(),
            owner_node: node.to_string(),
            owner_addr: addr.to_string(),
            epoch: prev.map_or(1, |p| p.epoch + 1),
            expires_at_ms,
        };
        m.insert(database_id.to_string(), lease.clone());
        self.save(&m).map_err(io_err)?;
        Ok(lease)
    }

    async fn renew(&self, lease: &Lease, expires_at_ms: u64) -> StoreResult<Lease> {
        self.check()?;
        let mut m = lock(&self.leases);
        match m.get_mut(&lease.database_id) {
            Some(cur) if cur.epoch == lease.epoch && cur.owner_node == lease.owner_node => {
                cur.expires_at_ms = expires_at_ms;
                let out = cur.clone();
                self.save(&m).map_err(io_err)?;
                Ok(out)
            }
            _ => Err(StoreError::Condition(format!(
                "lease of {} lost",
                lease.database_id
            ))),
        }
    }

    async fn release(&self, lease: &Lease) -> StoreResult<()> {
        self.check()?;
        let mut m = lock(&self.leases);
        match m.get_mut(&lease.database_id) {
            Some(cur) if cur.epoch == lease.epoch && cur.owner_node == lease.owner_node => {
                cur.expires_at_ms = 0;
                self.save(&m).map_err(io_err)?;
                Ok(())
            }
            _ => Err(StoreError::Condition(format!(
                "lease of {} lost",
                lease.database_id
            ))),
        }
    }

    async fn ping(&self) -> StoreResult<()> {
        self.check()
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[derive(Default, Serialize, Deserialize)]
struct RegistryData {
    databases: BTreeMap<String, DbRecord>,
    keys: BTreeMap<String, KeyRecord>,
}

/// In-memory registry (a fake DynamoDB `heapdb-registry`), optionally
/// persisted to a JSON file.
#[derive(Default)]
pub struct MemRegistry {
    data: Mutex<RegistryData>,
    file: Option<PathBuf>,
    /// Number of `get_key` calls (tests check caching).
    pub key_lookups: AtomicU64,
}

impl MemRegistry {
    /// Persisted to `path`.
    pub fn persistent(path: PathBuf) -> std::io::Result<Self> {
        let data: RegistryData = load_json(&path)?.unwrap_or_default();
        Ok(MemRegistry {
            data: Mutex::new(data),
            file: Some(path),
            key_lookups: AtomicU64::new(0),
        })
    }

    fn save(&self, d: &RegistryData) -> StoreResult<()> {
        match &self.file {
            Some(p) => save_json_atomic(p, d, true).map_err(io_err),
            None => Ok(()),
        }
    }
}

#[async_trait]
impl Registry for MemRegistry {
    async fn get_database(&self, id: &str) -> StoreResult<Option<DbRecord>> {
        Ok(lock(&self.data).databases.get(id).cloned())
    }

    async fn create_database(&self, rec: &DbRecord) -> StoreResult<Option<DbRecord>> {
        let mut d = lock(&self.data);
        if let Some(existing) = d.databases.get(&rec.database_id) {
            return Ok(Some(existing.clone()));
        }
        d.databases.insert(rec.database_id.clone(), rec.clone());
        self.save(&d)?;
        Ok(None)
    }

    async fn put_database(&self, rec: &DbRecord) -> StoreResult<()> {
        let mut d = lock(&self.data);
        d.databases.insert(rec.database_id.clone(), rec.clone());
        self.save(&d)
    }

    async fn delete_database(&self, id: &str) -> StoreResult<()> {
        let mut d = lock(&self.data);
        d.databases.remove(id);
        self.save(&d)
    }

    async fn list_databases(&self) -> StoreResult<Vec<DbRecord>> {
        Ok(lock(&self.data).databases.values().cloned().collect())
    }

    async fn get_key(&self, sha256: &str) -> StoreResult<Option<KeyRecord>> {
        self.key_lookups.fetch_add(1, Ordering::SeqCst);
        Ok(lock(&self.data).keys.get(sha256).cloned())
    }

    async fn put_key(&self, rec: &KeyRecord) -> StoreResult<()> {
        let mut d = lock(&self.data);
        d.keys.insert(rec.sha256.clone(), rec.clone());
        self.save(&d)
    }

    async fn delete_key(&self, sha256: &str) -> StoreResult<()> {
        let mut d = lock(&self.data);
        d.keys.remove(sha256);
        self.save(&d)
    }

    async fn ping(&self) -> StoreResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// In-memory usage table (a fake DynamoDB `heapdb-usage`), optionally
/// persisted to a JSON file.
#[derive(Default)]
pub struct MemUsageStore {
    items: Mutex<BTreeMap<String, UsageItem>>,
    file: Option<PathBuf>,
}

fn ukey(db: &str, bucket: &str) -> String {
    format!("{db}|{bucket}")
}

impl MemUsageStore {
    /// Persisted to `path`.
    pub fn persistent(path: PathBuf) -> std::io::Result<Self> {
        let items: BTreeMap<String, UsageItem> = load_json(&path)?.unwrap_or_default();
        Ok(MemUsageStore {
            items: Mutex::new(items),
            file: Some(path),
        })
    }

    fn save(&self, m: &BTreeMap<String, UsageItem>) -> StoreResult<()> {
        match &self.file {
            Some(p) => save_json_atomic(p, m, true).map_err(io_err),
            None => Ok(()),
        }
    }
}

#[async_trait]
impl UsageStore for MemUsageStore {
    async fn add(
        &self,
        database_id: &str,
        bucket: &str,
        requests: u64,
        events_written: u64,
    ) -> StoreResult<()> {
        let mut m = lock(&self.items);
        let it = m.entry(ukey(database_id, bucket)).or_insert_with(|| UsageItem {
            database_id: database_id.to_string(),
            bucket: bucket.to_string(),
            ..Default::default()
        });
        it.requests += requests;
        it.events_written += events_written;
        self.save(&m)
    }

    async fn set_gauges(&self, database_id: &str, events_stored: u64, storage_bytes: u64) -> StoreResult<()> {
        let mut m = lock(&self.items);
        let it = m
            .entry(ukey(database_id, GAUGE_BUCKET))
            .or_insert_with(|| UsageItem {
                database_id: database_id.to_string(),
                bucket: GAUGE_BUCKET.to_string(),
                ..Default::default()
            });
        it.events_stored = events_stored;
        it.storage_bytes = storage_bytes;
        self.save(&m)
    }

    async fn get(&self, database_id: &str, bucket: &str) -> StoreResult<Option<UsageItem>> {
        Ok(lock(&self.items).get(&ukey(database_id, bucket)).cloned())
    }

    async fn scan(&self) -> StoreResult<Vec<UsageItem>> {
        Ok(lock(&self.items).values().cloned().collect())
    }

    async fn delete_gauges(&self, database_id: &str) -> StoreResult<()> {
        let mut m = lock(&self.items);
        m.remove(&ukey(database_id, GAUGE_BUCKET));
        self.save(&m)
    }
}
