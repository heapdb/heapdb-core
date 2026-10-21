//! # heapdb-postgres
//!
//! HeapDB keyed heaps stored in PostgreSQL (`persist_to = "postgres://..."`).
//!
//! The engine is plain SQL: [`SQL`] (`sql/postgres/heapdb.sql`) installs
//! schema `heapdb` with tables `heapdb.heaps` / `heapdb.items` and the
//! functions `heapdb.create_heap`, `push`, `set_base`, `set_attrs`,
//! `remove`, `get`, `top_k`, `pop`, `rebase`, `drop_heap` and
//! `heapdb.matches`. Any language can call them; [`PostgresHeaps`]
//! implements [`HeapBackend`] on top of them with the synchronous
//! `postgres` crate, and the conformance suite (`tests/conformance.rs`)
//! checks it against the in-memory engine.
//!
//! ```no_run
//! use heapdb::heaps::{HeapBackend, HeapSpec};
//!
//! let mut heaps = heapdb_postgres::open("postgres://app@localhost/app")?; // installs the SQL if needed
//! heaps.create_heap(&HeapSpec::new("acme/checkout", 3.0 * 86400.0), 1.76e9)?;
//! heaps.push("acme/checkout", "CHK-12", 1.5, 1.76e9, None)?;
//! let top = heaps.top_k("acme/checkout", 5, 1.76e9 + 3600.0, None)?;
//! # Ok::<(), heapdb::Error>(())
//! ```
//!
//! ## TLS
//!
//! `sslmode` follows libpq: `disable`; `prefer` (the default) and `require`
//! encrypt the connection but, like libpq, do not verify the server
//! certificate; `verify-ca` verifies the chain and `verify-full` also the
//! host name, against the platform's trust store plus `sslrootcert=<file>`
//! when given.
//!
//! TLS is rustls via `tokio-postgres-rustls` (pure Rust, `ring` provider)
//! rather than `postgres-native-tls`: nothing links against OpenSSL, so the
//! abi3 Python wheels (manylinux, macOS, Windows) build and behave the same
//! everywhere, and the dependency set is small (rustls, ring, the platform
//! root store).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::cell::RefCell;

use heapdb::event::Attrs;
use heapdb::heaps::{check_attrs, check_finite, check_name, check_positive, HeapBackend};
use heapdb::{Error, Filter, HeapInfo, HeapItem, HeapSpec, Ranked, Result};
use postgres::{Client, NoTls, Row};
use serde_json::Value;

mod tls;

/// The SQL engine (`sql/postgres/heapdb.sql`). Idempotent: running it
/// again upgrades an installed schema in place.
pub const SQL: &str = include_str!("../../../sql/postgres/heapdb.sql");

/// The version [`SQL`] installs into `heapdb.schema_version`.
pub const SQL_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Install or upgrade the SQL engine on a connection (runs [`SQL`], which
/// has its own transaction; do not call inside one).
pub fn install(conn: &mut Client) -> Result<()> {
    conn.batch_execute(SQL).map_err(pg_err(None))
}

/// The installed version, or `None` if the schema is not installed.
pub fn installed_version(conn: &mut Client) -> Result<Option<String>> {
    let exists: bool = conn
        .query_one("SELECT to_regclass('heapdb.schema_version') IS NOT NULL", &[])
        .map_err(pg_err(None))?
        .get(0);
    if !exists {
        return Ok(None);
    }
    let row = conn
        .query_opt("SELECT version FROM heapdb.schema_version", &[])
        .map_err(pg_err(None))?;
    Ok(row.map(|r| r.get(0)))
}

/// Connect to `url` and install (or upgrade) the SQL engine unless this
/// version is already installed.
pub fn open(url: &str) -> Result<PostgresHeaps> {
    let mut h = PostgresHeaps::connect(url)?;
    if installed_version(h.client_mut())?.as_deref() != Some(SQL_VERSION) {
        install(h.client_mut())?;
    }
    Ok(h)
}

/// Connect to a Postgres URL (`postgres://` or `postgresql://`, or a
/// key=value connection string), with TLS per `sslmode`.
pub fn connect(url: &str) -> Result<Client> {
    let (cfg, mode, root) = tls::parse(url)?;
    let r = match mode {
        tls::Mode::Disable => cfg.connect(NoTls),
        _ => cfg.connect(tls::connector(mode, root.as_deref())?),
    };
    r.map_err(|e| {
        Error::Backend(format!(
            "cannot connect to {}: {e}",
            heapdb::heaps::redact_url(url)
        ))
    })
}

/// Map a Postgres error: SQLSTATE `HD400` -> [`Error::Invalid`], `HD404`
/// -> [`Error::UnknownHeap`], `HD409` -> [`Error::HeapConflict`], anything
/// else -> [`Error::Backend`].
fn pg_err(heap: Option<&str>) -> impl Fn(postgres::Error) -> Error + '_ {
    move |e| {
        if let Some(db) = e.as_db_error() {
            let msg = db.message().to_string();
            return match db.code().code() {
                "HD400" => Error::Invalid(msg),
                "HD404" => Error::UnknownHeap(heap.unwrap_or_default().to_string()),
                "HD409" => Error::HeapConflict(msg),
                _ => Error::Backend(format!("postgres: {msg}")),
            };
        }
        Error::Backend(format!("postgres: {e}"))
    }
}

fn obj(v: Value) -> Attrs {
    match v {
        Value::Object(m) => m,
        _ => Attrs::new(),
    }
}

fn info(r: &Row) -> HeapInfo {
    HeapInfo {
        name: r.get("name"),
        half_life: r.get("half_life"),
        landmark: r.get("landmark"),
        meta: obj(r.get("meta")),
        size: r.get::<_, i64>("size") as u64,
    }
}

fn item(r: &Row) -> HeapItem {
    HeapItem {
        id: r.get("id"),
        base: r.get("base"),
        activity: r.get("activity"),
        key: r.get("key"),
        attrs: obj(r.get("attrs")),
        version: r.get::<_, i64>("version") as u64,
    }
}

fn ranked(r: &Row) -> Ranked {
    Ranked {
        rank: r.get::<_, i64>("rank") as usize,
        id: r.get("id"),
        key: r.get("key"),
        score: r.get("score"),
        heat: r.get("heat"),
        base: r.get("base"),
        attrs: obj(r.get("attrs")),
        version: r.get::<_, i64>("version") as u64,
    }
}

/// Keyed heaps in Postgres (schema `heapdb`), over one connection.
///
/// Each call is one statement (one transaction unless the caller opened
/// one with `client_mut().batch_execute("BEGIN")`). Concurrent poppers on
/// other connections never receive the same item (`FOR UPDATE SKIP LOCKED`).
pub struct PostgresHeaps {
    // `HeapBackend`'s read methods take `&self`; the client needs `&mut`.
    client: RefCell<Client>,
}

impl std::fmt::Debug for PostgresHeaps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PostgresHeaps")
    }
}

impl PostgresHeaps {
    /// Connect without installing anything (the schema must exist).
    pub fn connect(url: &str) -> Result<PostgresHeaps> {
        Ok(Self::from_client(connect(url)?))
    }

    /// Use an existing connection.
    pub fn from_client(client: Client) -> PostgresHeaps {
        PostgresHeaps {
            client: RefCell::new(client),
        }
    }

    /// The connection (for transactions, `SET ROLE`, ...).
    pub fn client_mut(&mut self) -> &mut Client {
        self.client.get_mut()
    }

    /// Give the connection back.
    pub fn into_client(self) -> Client {
        self.client.into_inner()
    }

    fn query(
        &self,
        heap: Option<&str>,
        sql: &str,
        params: &[&(dyn postgres::types::ToSql + Sync)],
    ) -> Result<Vec<Row>> {
        self.client.borrow_mut().query(sql, params).map_err(pg_err(heap))
    }

    fn one_item(
        &self,
        heap: &str,
        sql: &str,
        params: &[&(dyn postgres::types::ToSql + Sync)],
    ) -> Result<HeapItem> {
        let rows = self.query(Some(heap), sql, params)?;
        rows.first()
            .map(item)
            .ok_or_else(|| Error::Backend("postgres: function returned no row".into()))
    }
}

/// Filters are validated in Rust first, for the same error messages as the
/// other backends; the SQL validates again (`heapdb.check_filter`).
fn filter_json(f: Option<&Filter>) -> Option<Value> {
    f.map(Filter::to_json)
}

impl HeapBackend for PostgresHeaps {
    fn create_heap(&mut self, spec: &HeapSpec, now: f64) -> Result<HeapInfo> {
        spec.validate()?;
        let meta = spec.meta.clone().map(Value::Object);
        let rows = self.query(
            Some(&spec.name),
            "SELECT * FROM heapdb.create_heap($1, $2, $3::float8, $4::jsonb, $5)",
            &[&spec.name, &spec.half_life, &spec.landmark, &meta, &now],
        )?;
        rows.first()
            .map(info)
            .ok_or_else(|| Error::Backend("postgres: create_heap returned no row".into()))
    }

    fn drop_heap(&mut self, name: &str) -> Result<bool> {
        let rows = self.query(Some(name), "SELECT heapdb.drop_heap($1)", &[&name])?;
        Ok(rows[0].get(0))
    }

    fn heap(&self, name: &str) -> Result<Option<HeapInfo>> {
        Ok(self
            .query(Some(name), "SELECT * FROM heapdb.heap_info($1)", &[&name])?
            .first()
            .map(info))
    }

    fn heaps(&self) -> Result<Vec<HeapInfo>> {
        Ok(self
            .query(None, "SELECT * FROM heapdb.list_heaps()", &[])?
            .iter()
            .map(info)
            .collect())
    }

    fn push(&mut self, heap: &str, id: &str, weight: f64, t: f64, attrs: Option<&Attrs>) -> Result<HeapItem> {
        check_name("item id", id)?;
        check_positive("weight", weight)?;
        check_finite("time", t)?;
        if let Some(a) = attrs {
            check_attrs(a)?;
        }
        let a = attrs.cloned().map(Value::Object);
        self.one_item(
            heap,
            "SELECT * FROM heapdb.push($1, $2, $3, $4, $5::jsonb)",
            &[&heap, &id, &weight, &t, &a],
        )
    }

    fn set_base(&mut self, heap: &str, id: &str, base: f64) -> Result<HeapItem> {
        check_name("item id", id)?;
        check_positive("base", base)?;
        self.one_item(
            heap,
            "SELECT * FROM heapdb.set_base($1, $2, $3)",
            &[&heap, &id, &base],
        )
    }

    fn set_attrs(&mut self, heap: &str, id: &str, attrs: &Attrs, merge: bool) -> Result<HeapItem> {
        check_name("item id", id)?;
        check_attrs(attrs)?;
        let a = Value::Object(attrs.clone());
        self.one_item(
            heap,
            "SELECT * FROM heapdb.set_attrs($1, $2, $3, $4)",
            &[&heap, &id, &a, &merge],
        )
    }

    fn remove(&mut self, heap: &str, id: &str) -> Result<bool> {
        let rows = self.query(Some(heap), "SELECT heapdb.remove($1, $2)", &[&heap, &id])?;
        Ok(rows[0].get(0))
    }

    fn get(&self, heap: &str, id: &str, now: f64) -> Result<Option<Ranked>> {
        check_finite("now", now)?;
        Ok(self
            .query(
                Some(heap),
                "SELECT * FROM heapdb.get($1, $2, $3)",
                &[&heap, &id, &now],
            )?
            .first()
            .map(ranked))
    }

    fn top_k(&self, heap: &str, k: usize, now: f64, filter: Option<&Filter>) -> Result<Vec<Ranked>> {
        check_finite("now", now)?;
        let k = i64::try_from(k).unwrap_or(i64::MAX);
        let f = filter_json(filter);
        Ok(self
            .query(
                Some(heap),
                "SELECT * FROM heapdb.top_k($1, $2, $3, $4::jsonb)",
                &[&heap, &k, &now, &f],
            )?
            .iter()
            .map(ranked)
            .collect())
    }

    fn pop(&mut self, heap: &str, now: f64, filter: Option<&Filter>) -> Result<Option<Ranked>> {
        check_finite("now", now)?;
        let f = filter_json(filter);
        Ok(self
            .query(
                Some(heap),
                "SELECT * FROM heapdb.pop($1, $2, $3::jsonb)",
                &[&heap, &now, &f],
            )?
            .first()
            .map(ranked))
    }

    fn rebase(&mut self, heap: &str, now: f64, half_life: Option<f64>) -> Result<HeapInfo> {
        check_finite("now", now)?;
        if let Some(h) = half_life {
            check_positive("half_life", h)?;
        }
        let rows = self.query(
            Some(heap),
            "SELECT * FROM heapdb.rebase($1, $2, $3::float8)",
            &[&heap, &now, &half_life],
        )?;
        rows.first()
            .map(info)
            .ok_or_else(|| Error::Backend("postgres: rebase returned no row".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sql_version_matches_the_crate() {
        assert!(
            SQL.contains(&format!("'{SQL_VERSION}'")),
            "sql/postgres/heapdb.sql must install schema_version {SQL_VERSION}"
        );
    }
}
