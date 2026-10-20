//! An open database on its owner node: engine state behind a lock, the
//! group-committing WAL writer, recovery, snapshots, lease fencing, usage
//! counters and request admission (monthly quota, rate limit).
//!
//! Write path: apply the op to the engine under the write lock, assign the
//! next `seq`, hand the record to the committer (still under the lock, so
//! WAL order = apply order), release the lock and wait. The committer
//! batches records arriving within the group-commit window into one object
//! `db/<id>/wal/<epoch>-<last seq>.jsonl`, checks the lease, puts the object
//! (put-if-absent), re-checks the lease with a consistent read and only
//! then acknowledges. Reads wait until everything they observed is durable,
//! so no client ever sees state that a crash could take back.
//!
//! Any failure after an op was applied (storage, lost lease) *poisons* the
//! database: pending and later requests get a retryable 503 and the
//! manager drops it; the next request recovers it from storage under a new
//! epoch.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use crate::config::Config;
use crate::error::{ApiError, ApiResult};
use crate::metrics::Metrics;
use crate::quota::{billing_period, TokenBucket};
use crate::state::{DbState, WalOp, WalRecord};
use crate::storage::{hour_bucket, period_bucket, Backends, DbRecord, Lease, StoreError};

/// Milliseconds since the Unix epoch.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Seconds since the Unix epoch (float).
pub fn now_secs() -> f64 {
    now_ms() as f64 / 1000.0
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

// ---------------------------------------------------------------------------
// Object keys
// ---------------------------------------------------------------------------

/// `<prefix>db/<id>/`.
pub fn db_prefix(prefix: &str, id: &str) -> String {
    format!("{prefix}db/{id}/")
}

/// WAL object key.
pub fn wal_key(prefix: &str, id: &str, epoch: u64, last_seq: u64) -> String {
    format!("{prefix}db/{id}/wal/{epoch:020}-{last_seq:020}.jsonl")
}

/// Snapshot key.
pub fn snap_key(prefix: &str, id: &str, epoch: u64, seq: u64) -> String {
    format!("{prefix}db/{id}/snap/{epoch:020}-{seq:020}.jsonl")
}

/// `(epoch, seq)` of a WAL or snapshot key.
pub fn parse_key(key: &str) -> Option<(u64, u64)> {
    let name = key.rsplit('/').next()?.strip_suffix(".jsonl")?;
    let (e, s) = name.split_once('-')?;
    Some((e.parse().ok()?, s.parse().ok()?))
}

// ---------------------------------------------------------------------------
// Fencing
// ---------------------------------------------------------------------------

/// The lease of an open database plus the checks every ack depends on.
pub struct Fence {
    lease: Mutex<Lease>,
    backends: Backends,
    ttl: Duration,
    margin: Duration,
}

fn lease_lost(msg: impl Into<String>) -> ApiError {
    ApiError::unavailable("lease_lost", msg)
}

impl Fence {
    /// Current lease.
    pub fn lease(&self) -> Lease {
        lock(&self.lease).clone()
    }

    /// Local view: the lease has more than the safety margin left.
    pub fn check_local(&self) -> ApiResult<()> {
        let l = lock(&self.lease);
        if now_ms() + self.margin.as_millis() as u64 >= l.expires_at_ms {
            return Err(lease_lost(format!(
                "lease of {} (epoch {}) is about to expire without renewal",
                l.database_id, l.epoch
            )));
        }
        Ok(())
    }

    /// Strongly consistent read: we still own the lease at this epoch.
    pub async fn check_remote(&self) -> ApiResult<()> {
        let mine = self.lease();
        let cur = self
            .backends
            .leases
            .get(&mine.database_id)
            .await
            .map_err(|e| lease_lost(format!("cannot verify lease: {e}")))?;
        match cur {
            Some(l) if l.epoch == mine.epoch && l.owner_node == mine.owner_node && l.live(now_ms()) => Ok(()),
            _ => Err(lease_lost(format!(
                "lease of {} (epoch {}) is held by another node",
                mine.database_id, mine.epoch
            ))),
        }
    }

    /// Extend the lease.
    pub async fn renew(&self) -> Result<(), StoreError> {
        let cur = self.lease();
        let new = self
            .backends
            .leases
            .renew(&cur, now_ms() + self.ttl.as_millis() as u64)
            .await?;
        *lock(&self.lease) = new;
        Ok(())
    }

    /// Give the lease up.
    pub async fn release(&self) {
        let cur = self.lease();
        if let Err(e) = self.backends.leases.release(&cur).await {
            tracing::debug!(database_id = %cur.database_id, error = %e, "lease release failed");
        }
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

#[derive(Default)]
struct UsageState {
    /// Current billing period `[start, end)`; `None` before the first load.
    period: Option<(i64, i64)>,
    /// Requests in the current period (durable + pending).
    period_count: u64,
    pending_hourly: BTreeMap<String, (u64, u64)>,
    pending_period: BTreeMap<i64, u64>,
    gauges_dirty: bool,
}

// ---------------------------------------------------------------------------
// Open database
// ---------------------------------------------------------------------------

const OPEN: u8 = 0;
const CLOSING: u8 = 1;
const CLOSED: u8 = 2;
const POISONED: u8 = 3;

struct WalItem {
    seq: u64,
    line: Vec<u8>,
    done: oneshot::Sender<ApiResult<()>>,
}

/// Node-wide context shared by open databases.
pub struct NodeCtx {
    /// Configuration.
    pub cfg: Arc<Config>,
    /// Storage.
    pub backends: Backends,
    /// This node's id.
    pub node_id: String,
    /// This node's internal address (advertised in leases).
    pub node_addr: String,
    /// Metrics.
    pub metrics: Arc<Metrics>,
}

/// A database open on this (owner) node.
pub struct OpenDb {
    /// Database id.
    pub id: String,
    /// Owner epoch.
    pub epoch: u64,
    /// Lease + checks.
    pub fence: Arc<Fence>,
    state: RwLock<DbState>,
    wal_tx: mpsc::UnboundedSender<WalItem>,
    durable: watch::Receiver<u64>,
    status: Arc<AtomicU8>,
    close_claimed: std::sync::atomic::AtomicBool,
    last_used_ms: AtomicU64,
    snap_seq: AtomicU64,
    snap_bytes: AtomicU64,
    wal_bytes: AtomicU64,
    last_snapshot: Mutex<Instant>,
    snapshot_lock: tokio::sync::Mutex<()>,
    usage: Mutex<UsageState>,
    limiter: Mutex<Option<TokenBucket>>,
    committer: Mutex<Option<JoinHandle<()>>>,
    ctx: Arc<NodeCtx>,
}

impl Drop for OpenDb {
    fn drop(&mut self) {
        if let Some(h) = lock(&self.committer).take() {
            h.abort();
        }
    }
}

struct Recovered {
    state: DbState,
    snap_seq: u64,
    snap_bytes: u64,
    wal_bytes: u64,
}

fn corrupt(id: &str, msg: impl std::fmt::Display) -> ApiError {
    tracing::error!(database_id = %id, error = %msg, "recovery failed");
    ApiError::internal(format!("recovery of {id} failed: {msg}"))
}

/// Rebuild a database from storage: the latest valid snapshot plus the
/// valid WAL records after it, in `seq` order.
///
/// Fencing rule: epoch `e'` starts at `start(e')`, the seq of its first
/// record (its `epoch_start` marker). A record or snapshot `(e, s)` is
/// valid iff no later epoch `e' > e` has `start(e') <= s`, so writes a
/// fenced owner made after a newer owner took over are ignored.
async fn recover(ctx: &NodeCtx, id: &str) -> ApiResult<Recovered> {
    let prefix = db_prefix(&ctx.cfg.s3_prefix, id);
    let objects = &ctx.backends.objects;
    let mut records: Vec<WalRecord> = Vec::new();
    let mut wal_bytes = 0u64;
    for key in objects.list(&format!("{prefix}wal/")).await? {
        if parse_key(&key).is_none() {
            continue;
        }
        let Some(body) = objects.get(&key).await? else {
            continue;
        };
        wal_bytes += body.len() as u64;
        for line in body.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
            let r: WalRecord =
                serde_json::from_slice(line).map_err(|e| corrupt(id, format!("{key}: {e}")))?;
            records.push(r);
        }
    }
    let mut start: BTreeMap<u64, u64> = BTreeMap::new();
    for r in &records {
        let s = start.entry(r.epoch).or_insert(r.seq);
        *s = (*s).min(r.seq);
    }
    let fenced = |e: u64, s: u64| start.range(e + 1..).any(|(_, &st)| st <= s);
    let mut snaps: Vec<(u64, u64, String)> = objects
        .list(&format!("{prefix}snap/"))
        .await?
        .into_iter()
        .filter_map(|k| parse_key(&k).map(|(e, s)| (e, s, k)))
        .collect();
    snaps.sort();
    let mut state = DbState::new();
    let (mut snap_seq, mut snap_bytes) = (0u64, 0u64);
    for (e, s, key) in snaps.iter().rev() {
        if fenced(*e, *s) {
            tracing::warn!(database_id = %id, key = %key, "ignoring fenced snapshot");
            continue;
        }
        let Some(body) = objects.get(key).await? else {
            continue;
        };
        match DbState::from_snapshot(&body) {
            Ok((st, h)) => {
                state = st;
                snap_seq = h.seq;
                snap_bytes = body.len() as u64;
                break;
            }
            Err(err) => tracing::error!(database_id = %id, key = %key, error = %err, "unreadable snapshot"),
        }
    }
    let mut valid: Vec<WalRecord> = records
        .into_iter()
        .filter(|r| r.seq > snap_seq && !fenced(r.epoch, r.seq))
        .collect();
    valid.sort_by_key(|r| r.seq);
    for (expect, r) in (snap_seq + 1..).zip(valid.iter()) {
        if r.seq != expect {
            return Err(corrupt(
                id,
                format!("WAL gap: expected seq {expect}, found {}", r.seq),
            ));
        }
        state
            .apply(&r.op)
            .map_err(|e| corrupt(id, format!("replaying seq {}: {e}", r.seq)))?;
        state.seq = r.seq;
    }
    Ok(Recovered {
        state,
        snap_seq,
        snap_bytes,
        wal_bytes,
    })
}

async fn committer(
    ctx: Arc<NodeCtx>,
    id: String,
    epoch: u64,
    fence: Arc<Fence>,
    status: Arc<AtomicU8>,
    mut rx: mpsc::UnboundedReceiver<WalItem>,
    durable: watch::Sender<u64>,
) {
    let window = ctx.cfg.group_commit;
    while let Some(first) = rx.recv().await {
        let mut batch = vec![first];
        let deadline = tokio::time::Instant::now() + window;
        while batch.len() < 1000 {
            tokio::select! {
                item = rx.recv() => match item {
                    Some(i) => batch.push(i),
                    None => break,
                },
                _ = tokio::time::sleep_until(deadline) => break,
            }
        }
        if status.load(Ordering::SeqCst) == POISONED {
            for it in batch {
                let _ = it.done.send(Err(ApiError::unavailable(
                    "database_unavailable",
                    "database is being reloaded; retry",
                )));
            }
            continue;
        }
        let last = batch.last().map_or(0, |i| i.seq);
        let mut body = Vec::with_capacity(batch.iter().map(|i| i.line.len() + 1).sum());
        for it in &batch {
            body.extend_from_slice(&it.line);
            body.push(b'\n');
        }
        let key = wal_key(&ctx.cfg.s3_prefix, &id, epoch, last);
        let res: ApiResult<()> = async {
            fence.check_local()?;
            ctx.backends
                .objects
                .put_if_absent(&key, Bytes::from(body))
                .await?;
            fence.check_remote().await
        }
        .await;
        match res {
            Ok(()) => {
                ctx.metrics.wal_commits.fetch_add(1, Ordering::Relaxed);
                ctx.metrics
                    .wal_records
                    .fetch_add(batch.len() as u64, Ordering::Relaxed);
                let _ = durable.send(last);
                for it in batch {
                    let _ = it.done.send(Ok(()));
                }
            }
            Err(e) => {
                ctx.metrics.wal_failures.fetch_add(1, Ordering::Relaxed);
                if e.code == "lease_lost" {
                    ctx.metrics.leases_lost.fetch_add(1, Ordering::Relaxed);
                }
                tracing::warn!(database_id = %id, epoch, error = %e, "WAL commit failed; poisoning database");
                status.store(POISONED, Ordering::SeqCst);
                let e = ApiError::unavailable(
                    if e.code == "lease_lost" {
                        "lease_lost"
                    } else {
                        "storage_unavailable"
                    },
                    format!("write not acknowledged: {}", e.message),
                );
                for it in batch {
                    let _ = it.done.send(Err(e.clone()));
                }
            }
        }
    }
}

impl OpenDb {
    /// Recover `id` from storage under `lease`, start its committer and
    /// write the epoch-start marker.
    pub async fn open(ctx: Arc<NodeCtx>, id: &str, lease: Lease) -> ApiResult<Arc<OpenDb>> {
        let rec = recover(&ctx, id).await?;
        let epoch = lease.epoch;
        let fence = Arc::new(Fence {
            lease: Mutex::new(lease),
            backends: ctx.backends.clone(),
            ttl: ctx.cfg.lease_ttl,
            margin: ctx.cfg.lease_margin(),
        });
        let status = Arc::new(AtomicU8::new(OPEN));
        let (tx, rx) = mpsc::unbounded_channel();
        let (dtx, drx) = watch::channel(rec.state.seq);
        let handle = tokio::spawn(committer(
            ctx.clone(),
            id.to_string(),
            epoch,
            fence.clone(),
            status.clone(),
            rx,
            dtx,
        ));
        let db = Arc::new(OpenDb {
            id: id.to_string(),
            epoch,
            fence,
            state: RwLock::new(rec.state),
            wal_tx: tx,
            durable: drx,
            status,
            close_claimed: std::sync::atomic::AtomicBool::new(false),
            last_used_ms: AtomicU64::new(now_ms()),
            snap_seq: AtomicU64::new(rec.snap_seq),
            snap_bytes: AtomicU64::new(rec.snap_bytes),
            wal_bytes: AtomicU64::new(rec.wal_bytes),
            last_snapshot: Mutex::new(Instant::now()),
            snapshot_lock: tokio::sync::Mutex::new(()),
            usage: Mutex::new(UsageState::default()),
            limiter: Mutex::new(None),
            committer: Mutex::new(Some(handle)),
            ctx: ctx.clone(),
        });
        db.write(WalOp::EpochStart).await?;
        ctx.metrics.opens.fetch_add(1, Ordering::Relaxed);
        tracing::info!(database_id = %id, epoch, seq = db.seq(), "database opened");
        Ok(db)
    }

    /// Last applied seq.
    pub fn seq(&self) -> u64 {
        self.state.read().unwrap_or_else(|p| p.into_inner()).seq
    }

    /// True while serving.
    pub fn is_open(&self) -> bool {
        self.status.load(Ordering::SeqCst) == OPEN
    }

    /// True once a failure means the database must be reloaded.
    pub fn is_poisoned(&self) -> bool {
        self.status.load(Ordering::SeqCst) == POISONED
    }

    /// Mark the database unusable (lost lease, failed write).
    pub fn poison(&self, why: &str) {
        if self.status.swap(POISONED, Ordering::SeqCst) != POISONED {
            tracing::warn!(database_id = %self.id, epoch = self.epoch, reason = why, "database poisoned");
        }
    }

    fn check_status(&self, writing: bool) -> ApiResult<()> {
        match self.status.load(Ordering::SeqCst) {
            OPEN => Ok(()),
            CLOSING if !writing => Ok(()),
            CLOSING | CLOSED => Err(ApiError::unavailable(
                "database_moving",
                "database is closing on this node; retry",
            )),
            _ => Err(ApiError::unavailable(
                "database_unavailable",
                "database is being reloaded; retry",
            )),
        }
    }

    /// Claim the right to close this database (true for the first caller).
    pub fn claim_close(&self) -> bool {
        !self.close_claimed.swap(true, Ordering::SeqCst)
    }

    /// Mark use (idle tracking / LRU).
    pub fn touch(&self) {
        self.last_used_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// Last use, Unix ms.
    pub fn last_used_ms(&self) -> u64 {
        self.last_used_ms.load(Ordering::Relaxed)
    }

    /// Estimated storage bytes (snapshot + WAL since).
    pub fn storage_bytes(&self) -> u64 {
        self.snap_bytes.load(Ordering::Relaxed) + self.wal_bytes.load(Ordering::Relaxed)
    }

    /// Stored events.
    pub fn events_stored(&self) -> u64 {
        self.state.read().unwrap_or_else(|p| p.into_inner()).db.len() as u64
    }

    /// Apply a mutation and wait until it is durable.
    pub async fn write(&self, op: WalOp) -> ApiResult<Value> {
        let events = match &op {
            WalOp::Insert(e) => e.len() as u64,
            _ => 0,
        };
        let (rx, result) = {
            let mut st = self.state.write().unwrap_or_else(|p| p.into_inner());
            self.check_status(true)?;
            let result = match st.apply(&op) {
                Ok(v) => v,
                Err(e) => {
                    if e.status == StatusCode::INTERNAL_SERVER_ERROR {
                        self.poison("apply failed after validation");
                    }
                    return Err(e);
                }
            };
            st.seq += 1;
            let seq = st.seq;
            let rec = WalRecord {
                epoch: self.epoch,
                seq,
                ts: now_secs(),
                op,
            };
            let line = serde_json::to_vec(&rec).map_err(|e| ApiError::internal(e.to_string()))?;
            self.wal_bytes.fetch_add(line.len() as u64 + 1, Ordering::Relaxed);
            let (tx, rx) = oneshot::channel();
            if self.wal_tx.send(WalItem { seq, line, done: tx }).is_err() {
                self.poison("committer stopped");
                return Err(ApiError::unavailable(
                    "database_unavailable",
                    "WAL writer stopped; retry",
                ));
            }
            (rx, result)
        };
        match rx.await {
            Ok(Ok(())) => {
                if events > 0 {
                    let mut u = lock(&self.usage);
                    u.pending_hourly
                        .entry(hour_bucket(now_secs() as i64))
                        .or_default()
                        .1 += events;
                    u.gauges_dirty = true;
                }
                Ok(result)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                self.poison("committer dropped a write");
                Err(ApiError::unavailable(
                    "database_unavailable",
                    "WAL writer stopped; retry",
                ))
            }
        }
    }

    async fn wait_durable(&self, seq: u64) -> ApiResult<()> {
        let mut rx = self.durable.clone();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            if *rx.borrow_and_update() >= seq {
                return Ok(());
            }
            if self.is_poisoned() {
                return Err(ApiError::unavailable(
                    "database_unavailable",
                    "database is being reloaded; retry",
                ));
            }
            match tokio::time::timeout_at(deadline, rx.changed()).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) | Err(_) => {
                    return Err(ApiError::unavailable(
                        "database_unavailable",
                        "WAL did not become durable; retry",
                    ))
                }
            }
        }
    }

    /// Run a read; returns once all state it observed is durable.
    pub async fn read<T>(&self, f: impl FnOnce(&DbState) -> ApiResult<T>) -> ApiResult<T> {
        let (res, seq) = {
            let st = self.state.read().unwrap_or_else(|p| p.into_inner());
            self.check_status(false)?;
            (f(&st), st.seq)
        };
        self.wait_durable(seq).await?;
        res
    }

    // -- admission and usage ----------------------------------------------------

    async fn ensure_period(&self, rec: &DbRecord, now: i64) -> ApiResult<(i64, i64)> {
        let (ps, pe) = billing_period(rec.limits.billing_period_start, now);
        if lock(&self.usage).period.map(|p| p.0) == Some(ps) {
            return Ok((ps, pe));
        }
        let stored = self
            .ctx
            .backends
            .usage
            .get(&self.id, &period_bucket(ps))
            .await?
            .map_or(0, |i| i.requests);
        let mut u = lock(&self.usage);
        if u.period.map(|p| p.0) != Some(ps) {
            let pending = u.pending_period.get(&ps).copied().unwrap_or(0);
            u.period = Some((ps, pe));
            u.period_count = stored + pending;
        }
        Ok((ps, pe))
    }

    /// Admit and count one Data API request: monthly quota, then the
    /// per-minute token bucket.
    pub async fn admit(&self, rec: &DbRecord) -> ApiResult<()> {
        let now = now_secs() as i64;
        let (ps, pe) = self.ensure_period(rec, now).await?;
        let mut u = lock(&self.usage);
        if let Some(max) = rec.limits.max_requests_per_month {
            if u.period_count >= max {
                return Err(ApiError::new(
                    StatusCode::TOO_MANY_REQUESTS,
                    "monthly_quota_exceeded",
                    format!(
                        "monthly request quota of {max} reached; the billing period ends at {pe} (Unix seconds)"
                    ),
                )
                .retry_after((pe - now).max(1) as u64));
            }
        }
        if let Some(rpm) = rec.limits.max_requests_per_minute {
            let mut lim = lock(&self.limiter);
            if lim.as_ref().map(TokenBucket::rpm) != Some(rpm) {
                *lim = Some(TokenBucket::new(rpm));
            }
            if let Err(wait) = lim.as_mut().expect("set above").take(Instant::now()) {
                return Err(ApiError::new(
                    StatusCode::TOO_MANY_REQUESTS,
                    "rate_limited",
                    format!("rate limit of {rpm} requests per minute exceeded"),
                )
                .retry_after(wait));
            }
        }
        u.period_count += 1;
        *u.pending_period.entry(ps).or_default() += 1;
        u.pending_hourly.entry(hour_bucket(now)).or_default().0 += 1;
        Ok(())
    }

    /// Storage / event quotas before a write.
    pub fn check_write_quota(&self, rec: &DbRecord, new_events: usize) -> ApiResult<()> {
        if let Some(max) = rec.limits.max_storage_bytes {
            if self.storage_bytes() >= max {
                return Err(ApiError::new(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "storage_quota_exceeded",
                    format!("storage quota of {max} bytes reached"),
                ));
            }
        }
        if let Some(max) = rec.limits.max_events {
            let have = self.events_stored();
            if have + new_events as u64 > max {
                return Err(ApiError::new(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "events_quota_exceeded",
                    format!("event quota of {max} reached ({have} stored, {new_events} more requested)"),
                ));
            }
        }
        Ok(())
    }

    /// `{database_id, plan, usage, limits}` for `GET /v1/databases/{id}`.
    pub async fn info(&self, rec: &DbRecord) -> ApiResult<Value> {
        let now = now_secs() as i64;
        let (ps, pe) = self.ensure_period(rec, now).await?;
        let count = lock(&self.usage).period_count;
        let l = &rec.limits;
        Ok(json!({
            "database_id": rec.database_id,
            "org_id": rec.org_id,
            "plan": rec.plan,
            "suspended": rec.suspended,
            "usage": {
                "requests_this_period": count,
                "period_start": ps,
                "period_end": pe,
                "events_stored": self.events_stored(),
                "storage_bytes": self.storage_bytes(),
            },
            "limits": {
                "max_events": l.max_events,
                "max_requests_per_minute": l.max_requests_per_minute,
                "max_storage_bytes": l.max_storage_bytes,
                "max_requests_per_month": l.max_requests_per_month,
                "billing_period_start": l.billing_period_start.unwrap_or(ps),
            },
        }))
    }

    /// Write pending usage counters (atomic additions) and gauges.
    pub async fn flush_usage(&self) {
        let (hourly, period, gauges) = {
            let mut u = lock(&self.usage);
            let g = std::mem::take(&mut u.gauges_dirty) || !u.pending_hourly.is_empty();
            (
                std::mem::take(&mut u.pending_hourly),
                std::mem::take(&mut u.pending_period),
                g,
            )
        };
        let store = &self.ctx.backends.usage;
        for (bucket, (r, w)) in hourly {
            if let Err(e) = store.add(&self.id, &bucket, r, w).await {
                tracing::warn!(database_id = %self.id, error = %e, "usage flush failed; will retry");
                let mut u = lock(&self.usage);
                let x = u.pending_hourly.entry(bucket).or_default();
                x.0 += r;
                x.1 += w;
            }
        }
        for (ps, r) in period {
            if let Err(e) = store.add(&self.id, &period_bucket(ps), r, 0).await {
                tracing::warn!(database_id = %self.id, error = %e, "usage flush failed; will retry");
                *lock(&self.usage).pending_period.entry(ps).or_default() += r;
            }
        }
        if gauges {
            let (n, b) = (self.events_stored(), self.storage_bytes());
            if store.set_gauges(&self.id, n, b).await.is_err() {
                lock(&self.usage).gauges_dirty = true;
            }
        }
    }

    // -- snapshots ------------------------------------------------------------------

    /// True if a snapshot is due (record count or interval).
    pub fn snapshot_due(&self, cfg: &Config) -> bool {
        let pending = self.seq().saturating_sub(self.snap_seq.load(Ordering::Relaxed));
        pending >= cfg.snapshot_records
            || (pending > 1 && lock(&self.last_snapshot).elapsed() >= cfg.snapshot_interval)
    }

    /// Write a snapshot of the durable state and delete what it covers.
    /// Returns false if there was nothing new.
    pub async fn snapshot(&self) -> ApiResult<bool> {
        let _g = self.snapshot_lock.lock().await;
        let (bytes, seq) = {
            let st = self.state.read().unwrap_or_else(|p| p.into_inner());
            if st.seq <= self.snap_seq.load(Ordering::Relaxed) {
                return Ok(false);
            }
            (st.snapshot(&self.id, self.epoch, now_secs()), st.seq)
        };
        self.wait_durable(seq).await?;
        let prefix = &self.ctx.cfg.s3_prefix;
        let objects = &self.ctx.backends.objects;
        let len = bytes.len() as u64;
        self.fence.check_local()?;
        match objects
            .put_if_absent(&snap_key(prefix, &self.id, self.epoch, seq), Bytes::from(bytes))
            .await
        {
            Ok(()) | Err(StoreError::Condition(_)) => {}
            Err(e) => return Err(e.into()),
        }
        self.fence.check_remote().await?;
        self.snap_seq.store(seq, Ordering::Relaxed);
        self.snap_bytes.store(len, Ordering::Relaxed);
        // Approximate: WAL appended while the snapshot was written is dropped
        // from the estimate.
        self.wal_bytes.store(0, Ordering::Relaxed);
        *lock(&self.last_snapshot) = Instant::now();
        self.ctx.metrics.snapshots.fetch_add(1, Ordering::Relaxed);
        lock(&self.usage).gauges_dirty = true;
        // Cleanup, still fenced: WAL covered by this snapshot (and anything
        // from older epochs), and snapshots older than the previous one.
        let dbp = db_prefix(prefix, &self.id);
        self.fence.check_local()?;
        for key in objects.list(&format!("{dbp}wal/")).await? {
            if let Some((e, s)) = parse_key(&key) {
                if e < self.epoch || (e == self.epoch && s <= seq) {
                    objects.delete(&key).await?;
                }
            }
        }
        let mut snaps: Vec<(u64, u64, String)> = objects
            .list(&format!("{dbp}snap/"))
            .await?
            .into_iter()
            .filter_map(|k| parse_key(&k).map(|(e, s)| (e, s, k)))
            .filter(|&(e, s, _)| (e, s) <= (self.epoch, seq))
            .collect();
        snaps.sort();
        if snaps.len() > 2 {
            for (_, _, k) in &snaps[..snaps.len() - 2] {
                objects.delete(k).await?;
            }
        }
        tracing::info!(database_id = %self.id, epoch = self.epoch, seq, bytes = len, "snapshot written");
        Ok(true)
    }

    // -- closing ------------------------------------------------------------------------

    /// Stop serving: wait for pending writes, optionally snapshot, flush
    /// usage, release the lease.
    pub async fn close_graceful(&self, snapshot: bool) {
        let seq = {
            let st = self.state.write().unwrap_or_else(|p| p.into_inner());
            let _ = self
                .status
                .compare_exchange(OPEN, CLOSING, Ordering::SeqCst, Ordering::SeqCst);
            st.seq
        };
        let healthy = !self.is_poisoned() && self.wait_durable(seq).await.is_ok();
        if healthy && snapshot {
            if let Err(e) = self.snapshot().await {
                tracing::warn!(database_id = %self.id, error = %e, "final snapshot failed");
            }
        }
        self.flush_usage().await;
        let _ = self
            .status
            .compare_exchange(CLOSING, CLOSED, Ordering::SeqCst, Ordering::SeqCst);
        if let Some(h) = lock(&self.committer).take() {
            h.abort();
        }
        self.fence.release().await;
        self.ctx.metrics.closes.fetch_add(1, Ordering::Relaxed);
        tracing::info!(database_id = %self.id, epoch = self.epoch, seq, "database closed");
    }

    /// Stop serving at once (lease lost or database deleted): no snapshot,
    /// no lease release, pending writes fail with a retryable 503.
    pub async fn close_fenced(&self, release: bool) {
        self.poison("fenced");
        self.flush_usage().await;
        if let Some(h) = lock(&self.committer).take() {
            h.abort();
        }
        if release {
            self.fence.release().await;
        }
        self.ctx.metrics.closes.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys() {
        let k = wal_key("p/", "db_x", 3, 42);
        assert_eq!(k, "p/db/db_x/wal/00000000000000000003-00000000000000000042.jsonl");
        assert_eq!(parse_key(&k), Some((3, 42)));
        assert_eq!(parse_key("p/db/db_x/wal/junk"), None);
        assert!(wal_key("", "d", 1, 9) < wal_key("", "d", 1, 10));
    }
}
