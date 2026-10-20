//! Prometheus text metrics (internal port, `GET /metrics`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Node-wide counters.
#[derive(Default)]
pub struct Metrics {
    /// HTTP responses by (surface, status).
    http: Mutex<BTreeMap<(String, u16), u64>>,
    /// WAL group commits.
    pub wal_commits: AtomicU64,
    /// WAL records committed.
    pub wal_records: AtomicU64,
    /// WAL commit failures.
    pub wal_failures: AtomicU64,
    /// Snapshots written.
    pub snapshots: AtomicU64,
    /// Requests forwarded to another node.
    pub forwards: AtomicU64,
    /// Forwarding failures.
    pub forward_failures: AtomicU64,
    /// Leases acquired.
    pub leases_acquired: AtomicU64,
    /// Leases lost (fenced).
    pub leases_lost: AtomicU64,
    /// Databases opened (recovered).
    pub opens: AtomicU64,
    /// Databases closed.
    pub closes: AtomicU64,
    /// Databases evicted under memory pressure.
    pub evictions: AtomicU64,
    /// Currently open databases (gauge).
    pub open_databases: AtomicU64,
    /// Estimated bytes of open databases (gauge).
    pub open_bytes: AtomicU64,
}

impl Metrics {
    /// Count one HTTP response.
    pub fn http(&self, surface: &str, status: u16) {
        let mut m = self.http.lock().unwrap_or_else(|p| p.into_inner());
        *m.entry((surface.to_string(), status)).or_default() += 1;
    }

    /// Prometheus exposition format.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("# HELP heapdb_http_responses_total HTTP responses by API surface and status.\n");
        out.push_str("# TYPE heapdb_http_responses_total counter\n");
        for ((s, c), n) in self.http.lock().unwrap_or_else(|p| p.into_inner()).iter() {
            out.push_str(&format!(
                "heapdb_http_responses_total{{surface=\"{s}\",status=\"{c}\"}} {n}\n"
            ));
        }
        let counters: [(&str, &str, &AtomicU64); 11] = [
            (
                "heapdb_wal_commits_total",
                "WAL group commits.",
                &self.wal_commits,
            ),
            (
                "heapdb_wal_records_total",
                "WAL records committed.",
                &self.wal_records,
            ),
            (
                "heapdb_wal_failures_total",
                "Failed WAL commits.",
                &self.wal_failures,
            ),
            ("heapdb_snapshots_total", "Snapshots written.", &self.snapshots),
            (
                "heapdb_forwards_total",
                "Requests forwarded to the owner node.",
                &self.forwards,
            ),
            (
                "heapdb_forward_failures_total",
                "Failed forwards.",
                &self.forward_failures,
            ),
            (
                "heapdb_leases_acquired_total",
                "Leases acquired.",
                &self.leases_acquired,
            ),
            (
                "heapdb_leases_lost_total",
                "Leases lost (fenced).",
                &self.leases_lost,
            ),
            (
                "heapdb_database_opens_total",
                "Databases opened (recovered).",
                &self.opens,
            ),
            ("heapdb_database_closes_total", "Databases closed.", &self.closes),
            (
                "heapdb_database_evictions_total",
                "Databases evicted under the memory budget.",
                &self.evictions,
            ),
        ];
        for (name, help, v) in counters {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {}\n",
                v.load(Ordering::Relaxed)
            ));
        }
        let gauges: [(&str, &str, &AtomicU64); 2] = [
            (
                "heapdb_open_databases",
                "Databases open on this node.",
                &self.open_databases,
            ),
            (
                "heapdb_open_database_bytes",
                "Estimated bytes of open databases.",
                &self.open_bytes,
            ),
        ];
        for (name, help, v) in gauges {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {}\n",
                v.load(Ordering::Relaxed)
            ));
        }
        out
    }
}
