//! Which databases are open here, who owns the others, and the background
//! work (lease renewal, usage flushes, snapshots, idle close, LRU eviction).
//!
//! A request for database `d` on node `n`:
//! 1. `d` open on `n` → serve;
//! 2. a live lease of another node → forward to its `owner_addr`
//!    (forwarded requests are never forwarded again: 503 `not_owner`);
//! 3. otherwise acquire the lease (conditional write, `epoch + 1`),
//!    recover from storage and serve.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use tokio::sync::OnceCell;
use tokio_util::sync::CancellationToken;

use crate::database::{now_ms, now_secs, NodeCtx, OpenDb};
use crate::error::{ApiError, ApiResult};
use crate::ids::sha256_hex;
use crate::storage::{DbRecord, KeyRecord, Lease, Scope, StoreError};

/// Where a request is served.
pub enum Route {
    /// Here.
    Local(Arc<OpenDb>),
    /// On the owner, at this internal address.
    Forward(String),
}

/// Required permission of a Data API route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Need {
    /// Any valid key of the database.
    Any,
    /// A scope.
    Scope(Scope),
}

type Cell = Arc<OnceCell<Arc<OpenDb>>>;

struct Cached<T> {
    at: Instant,
    value: T,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Open databases and caches of one node.
pub struct Manager {
    /// Node context.
    pub ctx: Arc<NodeCtx>,
    open: Mutex<HashMap<String, Cell>>,
    db_cache: Mutex<HashMap<String, Cached<Option<DbRecord>>>>,
    key_cache: Mutex<HashMap<String, Cached<Option<KeyRecord>>>>,
    lease_cache: Mutex<HashMap<String, Cached<Lease>>>,
    /// HTTP client for forwarding and owner notifications.
    pub http: reqwest::Client,
    shutting_down: AtomicBool,
}

const NEGATIVE_TTL: Duration = Duration::from_secs(1);
const LEASE_CACHE_TTL: Duration = Duration::from_millis(500);

impl Manager {
    /// New manager.
    pub fn new(ctx: Arc<NodeCtx>) -> Arc<Self> {
        Arc::new(Manager {
            ctx,
            open: Mutex::new(HashMap::new()),
            db_cache: Mutex::new(HashMap::new()),
            key_cache: Mutex::new(HashMap::new()),
            lease_cache: Mutex::new(HashMap::new()),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .pool_idle_timeout(Duration::from_secs(30))
                .build()
                .expect("HTTP client"),
            shutting_down: AtomicBool::new(false),
        })
    }

    // -- registry caches ----------------------------------------------------------

    /// Database record (cached for at most `auth_cache_ttl`).
    pub async fn db_record(&self, id: &str) -> ApiResult<Option<DbRecord>> {
        let ttl = self.ctx.cfg.auth_cache_ttl;
        if let Some(c) = lock(&self.db_cache).get(id) {
            let t = if c.value.is_some() {
                ttl
            } else {
                ttl.min(NEGATIVE_TTL)
            };
            if c.at.elapsed() < t {
                return Ok(c.value.clone());
            }
        }
        let v = self.ctx.backends.registry.get_database(id).await?;
        lock(&self.db_cache).insert(
            id.to_string(),
            Cached {
                at: Instant::now(),
                value: v.clone(),
            },
        );
        Ok(v)
    }

    async fn key_record(&self, sha: &str) -> ApiResult<Option<KeyRecord>> {
        let ttl = self.ctx.cfg.auth_cache_ttl;
        if let Some(c) = lock(&self.key_cache).get(sha) {
            let t = if c.value.is_some() {
                ttl
            } else {
                ttl.min(NEGATIVE_TTL)
            };
            if c.at.elapsed() < t {
                return Ok(c.value.clone());
            }
        }
        let v = self.ctx.backends.registry.get_key(sha).await?;
        lock(&self.key_cache).insert(
            sha.to_string(),
            Cached {
                at: Instant::now(),
                value: v.clone(),
            },
        );
        Ok(v)
    }

    /// Forget cached registry data of a database (and one key, or all its keys).
    pub fn invalidate(&self, id: &str, key_sha: Option<&str>, all_keys: bool) {
        lock(&self.db_cache).remove(id);
        let mut kc = lock(&self.key_cache);
        if let Some(sha) = key_sha {
            kc.remove(sha);
        }
        if all_keys {
            kc.retain(|_, c| c.value.as_ref().is_none_or(|k| k.database_id != id));
        }
    }

    /// Authenticate a Data API request.
    pub async fn authenticate(
        &self,
        id: &str,
        bearer: Option<&str>,
        need: Need,
    ) -> ApiResult<(KeyRecord, DbRecord)> {
        let Some(token) = bearer.filter(|t| !t.is_empty()) else {
            return Err(ApiError::new(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "missing Authorization: Bearer <api_key>",
            ));
        };
        let sha = sha256_hex(token.as_bytes());
        let Some(key) = self.key_record(&sha).await? else {
            return Err(ApiError::new(
                StatusCode::UNAUTHORIZED,
                "invalid_api_key",
                "invalid API key",
            ));
        };
        if key.database_id != id {
            return Err(ApiError::new(
                StatusCode::FORBIDDEN,
                "key_not_for_database",
                "this API key belongs to another database",
            ));
        }
        if key.expires_at.is_some_and(|t| t <= now_secs()) {
            return Err(ApiError::new(
                StatusCode::UNAUTHORIZED,
                "key_expired",
                "API key expired",
            ));
        }
        let Some(rec) = self.db_record(id).await? else {
            return Err(ApiError::database_not_found(id));
        };
        if let Need::Scope(s) = need {
            if !key.scopes.contains(&s) {
                let name = serde_json::to_value(s).unwrap_or_default();
                return Err(ApiError::new(
                    StatusCode::FORBIDDEN,
                    "insufficient_scope",
                    format!("this API key lacks the {} scope", name.as_str().unwrap_or("?")),
                ));
            }
        }
        if rec.suspended {
            return Err(ApiError::new(
                StatusCode::FORBIDDEN,
                "database_suspended",
                format!("database {id} is suspended"),
            ));
        }
        Ok((key, rec))
    }

    // -- routing --------------------------------------------------------------------------

    fn get_cell(&self, id: &str) -> Option<Cell> {
        lock(&self.open).get(id).cloned()
    }

    /// Open databases (initialised ones).
    pub fn open_databases(&self) -> Vec<Arc<OpenDb>> {
        lock(&self.open)
            .values()
            .filter_map(|c| c.get().cloned())
            .collect()
    }

    fn remove(&self, db: &Arc<OpenDb>) {
        let mut m = lock(&self.open);
        if let Some(c) = m.get(&db.id) {
            if c.get().is_some_and(|x| Arc::ptr_eq(x, db)) {
                m.remove(&db.id);
            }
        }
        self.ctx.metrics.open_databases.store(
            m.values().filter(|c| c.get().is_some()).count() as u64,
            Ordering::Relaxed,
        );
    }

    /// Close a database in the background (fenced or graceful).
    pub fn spawn_close(self: &Arc<Self>, db: Arc<OpenDb>, graceful: bool, release: bool) {
        if !db.claim_close() {
            return;
        }
        let me = self.clone();
        tokio::spawn(async move {
            if graceful {
                db.close_graceful(true).await;
            } else {
                db.close_fenced(release).await;
            }
            me.remove(&db);
        });
    }

    /// Close now and wait.
    pub async fn close_now(self: &Arc<Self>, db: Arc<OpenDb>, graceful: bool, release: bool) {
        if db.claim_close() {
            if graceful {
                db.close_graceful(true).await;
            } else {
                db.close_fenced(release).await;
            }
        }
        self.remove(&db);
    }

    /// Decide where a request for `id` is served.
    pub async fn route(self: &Arc<Self>, id: &str, forwarded: bool) -> ApiResult<Route> {
        if self.shutting_down.load(Ordering::SeqCst) {
            return Err(ApiError::unavailable(
                "shutting_down",
                "node is shutting down; retry",
            ));
        }
        for _ in 0..4 {
            if let Some(cell) = self.get_cell(id) {
                match cell.get() {
                    Some(db) if db.is_open() => {
                        db.touch();
                        return Ok(Route::Local(db.clone()));
                    }
                    Some(db) => {
                        if db.is_poisoned() {
                            self.spawn_close(db.clone(), false, true);
                        }
                        return Err(ApiError::unavailable(
                            "database_moving",
                            "database is being reloaded or moved; retry",
                        ));
                    }
                    None => {
                        // Being opened by another request: wait for it.
                        if let Some(db) = self.open_with(id, None, cell).await? {
                            return Ok(Route::Local(db));
                        }
                        continue;
                    }
                }
            }
            let now = now_ms();
            if let Some(c) = lock(&self.lease_cache).get(id) {
                if c.at.elapsed() < LEASE_CACHE_TTL
                    && c.value.live(now)
                    && c.value.owner_node != self.ctx.node_id
                {
                    if forwarded {
                        return Err(not_owner());
                    }
                    return Ok(Route::Forward(c.value.owner_addr.clone()));
                }
            }
            let lease = self.ctx.backends.leases.get(id).await?;
            match lease {
                Some(l) if l.live(now) && l.owner_node != self.ctx.node_id => {
                    let addr = l.owner_addr.clone();
                    lock(&self.lease_cache).insert(
                        id.to_string(),
                        Cached {
                            at: Instant::now(),
                            value: l,
                        },
                    );
                    if forwarded {
                        return Err(not_owner());
                    }
                    return Ok(Route::Forward(addr));
                }
                prev => {
                    let cell = lock(&self.open)
                        .entry(id.to_string())
                        .or_insert_with(|| Arc::new(OnceCell::new()))
                        .clone();
                    if let Some(db) = self.open_with(id, Some(prev), cell).await? {
                        return Ok(Route::Local(db));
                    }
                }
            }
        }
        Err(ApiError::unavailable(
            "database_moving",
            "database ownership is changing; retry",
        ))
    }

    /// Initialise `cell` (acquire + recover) or wait for whoever does.
    /// `Ok(None)`: lost the lease race; re-route.
    async fn open_with(
        self: &Arc<Self>,
        id: &str,
        prev: Option<Option<Lease>>,
        cell: Cell,
    ) -> ApiResult<Option<Arc<OpenDb>>> {
        let ctx = self.ctx.clone();
        let res = cell
            .get_or_try_init(|| async {
                let prev = match prev {
                    Some(p) => p,
                    None => ctx.backends.leases.get(id).await?,
                };
                if let Some(l) = &prev {
                    if l.live(now_ms()) && l.owner_node != ctx.node_id {
                        return Err(ApiError::unavailable("lease_race", "lease held elsewhere"));
                    }
                }
                let expires = now_ms() + ctx.cfg.lease_ttl.as_millis() as u64;
                let lease = ctx
                    .backends
                    .leases
                    .acquire(id, &ctx.node_id, &ctx.node_addr, prev.as_ref(), expires)
                    .await
                    .map_err(|e| match e {
                        StoreError::Condition(m) => ApiError::unavailable("lease_race", m),
                        other => other.into(),
                    })?;
                ctx.metrics.leases_acquired.fetch_add(1, Ordering::Relaxed);
                tracing::info!(database_id = %id, epoch = lease.epoch, node = %ctx.node_id, "lease acquired");
                match OpenDb::open(ctx.clone(), id, lease.clone()).await {
                    Ok(db) => Ok(db),
                    Err(e) => {
                        let _ = ctx.backends.leases.release(&lease).await;
                        Err(e)
                    }
                }
            })
            .await;
        match res {
            Ok(db) => {
                let n = lock(&self.open).values().filter(|c| c.get().is_some()).count();
                self.ctx.metrics.open_databases.store(n as u64, Ordering::Relaxed);
                Ok(Some(db.clone()))
            }
            Err(e) => {
                let mut m = lock(&self.open);
                if m.get(id)
                    .is_some_and(|c| Arc::ptr_eq(c, &cell) && c.get().is_none())
                {
                    m.remove(id);
                }
                if e.code == "lease_race" {
                    Ok(None)
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Forget a cached lease (after a failed forward).
    pub fn forget_lease(&self, id: &str) {
        lock(&self.lease_cache).remove(id);
    }

    /// The open instance of `id`, if any.
    pub fn get_open(&self, id: &str) -> Option<Arc<OpenDb>> {
        self.get_cell(id).and_then(|c| c.get().cloned())
    }

    /// Tell the owner of `id` (if it is another live node) to drop cached
    /// registry data, or to close the database (`drop_db`).
    pub async fn notify_owner(&self, id: &str, key_sha: Option<&str>, drop_db: bool) {
        let Ok(Some(l)) = self.ctx.backends.leases.get(id).await else {
            return;
        };
        if !l.live(now_ms()) || l.owner_node == self.ctx.node_id {
            return;
        }
        let mut req = self
            .http
            .post(format!(
                "http://{}/internal/v1/databases/{id}/invalidate",
                l.owner_addr
            ))
            .timeout(Duration::from_secs(5))
            .header("content-type", "application/json")
            .body(serde_json::json!({"key_sha256": key_sha, "drop": drop_db}).to_string());
        if let Some(t) = &self.ctx.cfg.admin_token {
            req = req.bearer_auth(t);
        }
        if let Err(e) = req.send().await {
            tracing::warn!(database_id = %id, owner = %l.owner_addr, error = %e, "owner notification failed");
        }
    }

    // -- background work -----------------------------------------------------------------

    /// Renew leases of open databases; fence those that are lost.
    pub async fn renew_leases(self: &Arc<Self>) {
        for db in self.open_databases() {
            if !db.is_open() {
                continue;
            }
            match db.fence.renew().await {
                Ok(()) => {}
                Err(StoreError::Condition(m)) => {
                    tracing::warn!(database_id = %db.id, epoch = db.epoch, reason = %m, "lease lost; fencing");
                    self.ctx.metrics.leases_lost.fetch_add(1, Ordering::Relaxed);
                    self.close_now(db, false, false).await;
                }
                Err(e) => {
                    tracing::warn!(database_id = %db.id, error = %e, "lease renewal failed");
                    if db.fence.check_local().is_err() {
                        self.ctx.metrics.leases_lost.fetch_add(1, Ordering::Relaxed);
                        self.close_now(db, false, false).await;
                    }
                }
            }
        }
    }

    /// One maintenance pass.
    pub async fn maintain(self: &Arc<Self>, flush_usage: bool) {
        let cfg = self.ctx.cfg.clone();
        let mut dbs = self.open_databases();
        for db in &dbs {
            if db.is_poisoned() {
                self.spawn_close(db.clone(), false, true);
                continue;
            }
            if !db.is_open() {
                continue;
            }
            if flush_usage {
                db.flush_usage().await;
            }
            if db.snapshot_due(&cfg) {
                if let Err(e) = db.snapshot().await {
                    tracing::warn!(database_id = %db.id, error = %e, "snapshot failed");
                }
            }
            if now_ms().saturating_sub(db.last_used_ms()) >= cfg.idle_close.as_millis() as u64 {
                tracing::info!(database_id = %db.id, "closing idle database");
                self.spawn_close(db.clone(), true, true);
            }
        }
        // Memory budget (estimate: 3x the serialised size), LRU first.
        if cfg.memory_budget_mb > 0 {
            dbs.retain(|d| d.is_open());
            let est = |d: &Arc<OpenDb>| d.storage_bytes().saturating_mul(3);
            let mut total: u64 = dbs.iter().map(est).sum();
            self.ctx.metrics.open_bytes.store(total, Ordering::Relaxed);
            let budget = cfg.memory_budget_mb << 20;
            if total > budget {
                dbs.sort_by_key(|d| d.last_used_ms());
                for d in dbs {
                    if total <= budget {
                        break;
                    }
                    total = total.saturating_sub(est(&d));
                    tracing::info!(database_id = %d.id, "evicting database (memory budget)");
                    self.ctx.metrics.evictions.fetch_add(1, Ordering::Relaxed);
                    self.spawn_close(d, true, true);
                }
            }
        }
    }

    /// Run renewal and maintenance until `stop` fires.
    pub async fn run_background(self: Arc<Self>, stop: CancellationToken) {
        let cfg = self.ctx.cfg.clone();
        let mut renew = tokio::time::interval(cfg.lease_renew);
        let tick = cfg
            .usage_flush
            .min(Duration::from_millis(500))
            .max(Duration::from_millis(20));
        let mut maint = tokio::time::interval(tick);
        let mut last_flush = Instant::now();
        renew.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        maint.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = stop.cancelled() => break,
                _ = renew.tick() => self.renew_leases().await,
                _ = maint.tick() => {
                    let flush = last_flush.elapsed() >= cfg.usage_flush;
                    if flush {
                        last_flush = Instant::now();
                    }
                    self.maintain(flush).await;
                }
            }
        }
    }

    /// Graceful shutdown: refuse new requests, then flush, snapshot and
    /// release every open database.
    pub async fn shutdown(self: &Arc<Self>) {
        self.shutting_down.store(true, Ordering::SeqCst);
        let dbs = self.open_databases();
        let tasks: Vec<_> = dbs
            .into_iter()
            .map(|db| {
                let me = self.clone();
                tokio::spawn(async move { me.close_now(db, true, true).await })
            })
            .collect();
        for t in tasks {
            let _ = t.await;
        }
    }
}

fn not_owner() -> ApiError {
    ApiError::unavailable("not_owner", "database is owned by another node; retry")
}
