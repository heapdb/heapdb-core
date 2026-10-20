//! Server configuration (contract §6 "Config (env)").
//!
//! Every setting has an environment variable; the binary also accepts the
//! main ones as command-line flags (flags win). Tunables beyond the contract
//! (lease timing, group-commit window, snapshot cadence, ...) exist mainly
//! for tests and are documented in `docs/server.md`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Where registry, leases, usage and data live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterMode {
    /// Single node: everything on local disk (`--data-dir`).
    Off,
    /// Several nodes: S3 (WAL + snapshots) and DynamoDB (leases, registry, usage).
    Aws,
}

impl std::str::FromStr for ClusterMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "" | "single" | "none" => Ok(ClusterMode::Off),
            "aws" => Ok(ClusterMode::Aws),
            other => Err(format!("unknown cluster mode {other:?} (expected off or aws)")),
        }
    }
}

/// Full server configuration.
#[derive(Clone, Debug)]
pub struct Config {
    /// Public listener (Data + Admin API, health).
    pub listen: SocketAddr,
    /// Internal listener (forwarded requests, `/metrics`, invalidation).
    pub internal_listen: SocketAddr,
    /// Cluster mode.
    pub cluster: ClusterMode,
    /// This node's private `ip:port` of the internal listener, advertised in leases.
    pub node_addr: Option<String>,
    /// S3 bucket (cluster mode).
    pub s3_bucket: String,
    /// Key prefix inside the bucket (e.g. `prod/`).
    pub s3_prefix: String,
    /// DynamoDB table of leases.
    pub ddb_leases: String,
    /// DynamoDB table of the database/key registry.
    pub ddb_registry: String,
    /// DynamoDB table of usage counters.
    pub ddb_usage: String,
    /// Bearer token of the Admin API (`None`: Admin API disabled).
    pub admin_token: Option<String>,
    /// Data directory (single node) or local cache directory (cluster).
    pub data_dir: PathBuf,
    /// Memory budget for open databases (LRU eviction above it); 0 = unlimited.
    pub memory_budget_mb: u64,
    /// Lease duration.
    pub lease_ttl: Duration,
    /// Lease renewal interval.
    pub lease_renew: Duration,
    /// Group-commit window of the WAL.
    pub group_commit: Duration,
    /// Snapshot at least this often (if anything changed).
    pub snapshot_interval: Duration,
    /// ... or after this many WAL records.
    pub snapshot_records: u64,
    /// Close databases idle for this long.
    pub idle_close: Duration,
    /// Registry/key cache lifetime (contract: at most 30 s).
    pub auth_cache_ttl: Duration,
    /// How often usage counters are flushed to the usage store.
    pub usage_flush: Duration,
    /// Maximum request body size.
    pub max_body_bytes: usize,
    /// fsync WAL objects and snapshots (single node).
    pub fsync: bool,
}

impl Config {
    /// Defaults for a single node storing everything under `data_dir`.
    pub fn single_node(data_dir: impl Into<PathBuf>) -> Self {
        Config {
            listen: "0.0.0.0:8080".parse().expect("valid"),
            internal_listen: "0.0.0.0:9090".parse().expect("valid"),
            cluster: ClusterMode::Off,
            node_addr: None,
            s3_bucket: String::new(),
            s3_prefix: String::new(),
            ddb_leases: "heapdb-leases".into(),
            ddb_registry: "heapdb-registry".into(),
            ddb_usage: "heapdb-usage".into(),
            admin_token: None,
            data_dir: data_dir.into(),
            memory_budget_mb: 1024,
            lease_ttl: Duration::from_secs(30),
            lease_renew: Duration::from_secs(10),
            group_commit: Duration::from_millis(5),
            snapshot_interval: Duration::from_secs(300),
            snapshot_records: 10_000,
            idle_close: Duration::from_secs(600),
            auth_cache_ttl: Duration::from_secs(30),
            usage_flush: Duration::from_secs(1),
            max_body_bytes: 8 << 20,
            fsync: true,
        }
    }

    /// Apply the tuning environment variables (`HEAPDB_LEASE_TTL_MS`, ...).
    /// Main settings are read by the binary's argument parser.
    pub fn apply_tuning_env(&mut self) -> Result<(), String> {
        fn ms(name: &str) -> Result<Option<Duration>, String> {
            match std::env::var(name) {
                Ok(v) if !v.trim().is_empty() => v
                    .trim()
                    .parse::<u64>()
                    .map(|x| Some(Duration::from_millis(x)))
                    .map_err(|e| format!("{name}: {e}")),
                _ => Ok(None),
            }
        }
        fn int(name: &str) -> Result<Option<u64>, String> {
            match std::env::var(name) {
                Ok(v) if !v.trim().is_empty() => v
                    .trim()
                    .parse::<u64>()
                    .map(Some)
                    .map_err(|e| format!("{name}: {e}")),
                _ => Ok(None),
            }
        }
        if let Some(d) = ms("HEAPDB_LEASE_TTL_MS")? {
            self.lease_ttl = d;
        }
        if let Some(d) = ms("HEAPDB_LEASE_RENEW_MS")? {
            self.lease_renew = d;
        }
        if let Some(d) = ms("HEAPDB_GROUP_COMMIT_MS")? {
            self.group_commit = d.min(Duration::from_millis(10));
        }
        if let Some(d) = ms("HEAPDB_SNAPSHOT_INTERVAL_MS")? {
            self.snapshot_interval = d;
        }
        if let Some(n) = int("HEAPDB_SNAPSHOT_RECORDS")? {
            self.snapshot_records = n.max(1);
        }
        if let Some(d) = ms("HEAPDB_IDLE_CLOSE_MS")? {
            self.idle_close = d;
        }
        if let Some(d) = ms("HEAPDB_AUTH_CACHE_TTL_MS")? {
            self.auth_cache_ttl = d.min(Duration::from_secs(30));
        }
        if let Some(d) = ms("HEAPDB_USAGE_FLUSH_MS")? {
            self.usage_flush = d;
        }
        if let Some(n) = int("HEAPDB_MAX_BODY_BYTES")? {
            self.max_body_bytes = n as usize;
        }
        if let Ok(v) = std::env::var("HEAPDB_FSYNC") {
            self.fsync = !matches!(v.trim(), "0" | "false" | "off");
        }
        Ok(())
    }

    /// Safety margin: a node stops acknowledging writes once its lease has
    /// less than this left without a successful renewal.
    pub fn lease_margin(&self) -> Duration {
        (self.lease_ttl / 6).max(Duration::from_millis(50))
    }
}
