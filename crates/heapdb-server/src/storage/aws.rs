//! AWS backends (feature `cluster-aws`): S3 for WAL + snapshots, DynamoDB
//! for leases, the registry and usage counters.
//!
//! Tables (all on-demand):
//! * `heapdb-leases`: partition key `database_id` (S); attributes
//!   `owner_node` (S), `owner_addr` (S), `epoch` (N), `expires_at` (N, Unix ms).
//! * `heapdb-registry`: partition key `pk` (S): `db#<id>` and
//!   `key#<sha256>` items, each with a JSON `record` (S).
//! * `heapdb-usage`: partition key `database_id` (S), sort key `bucket` (S);
//!   numeric `requests`, `events_written`, `events_stored`, `storage_bytes`.
//!
//! Endpoints follow the standard AWS configuration (`AWS_REGION`,
//! `AWS_ENDPOINT_URL`, task role credentials). Set
//! `HEAPDB_S3_FORCE_PATH_STYLE=1` for S3-compatible emulators.

use std::collections::HashMap;

use async_trait::async_trait;
use aws_sdk_dynamodb::types::{
    AttributeDefinition, AttributeValue, BillingMode, KeySchemaElement, KeyType, ScalarAttributeType,
};
use aws_sdk_dynamodb::Client as Ddb;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client as S3;
use bytes::Bytes;

use super::{
    Backends, DbRecord, KeyRecord, Lease, LeaseStore, ObjectStore, Registry, StoreError, StoreResult,
    UsageItem, UsageStore, GAUGE_BUCKET,
};
use crate::config::Config;

fn unavailable(e: impl std::fmt::Display) -> StoreError {
    StoreError::Unavailable(e.to_string())
}

fn sdk_err<E, R>(e: aws_sdk_s3::error::SdkError<E, R>) -> StoreError
where
    E: std::error::Error + Send + Sync + 'static,
    R: std::fmt::Debug,
{
    StoreError::Unavailable(format!("{}", aws_sdk_s3::error::DisplayErrorContext(e)))
}

fn ddb_err<E, R>(e: aws_sdk_dynamodb::error::SdkError<E, R>) -> StoreError
where
    E: std::error::Error + aws_sdk_dynamodb::error::ProvideErrorMetadata + Send + Sync + 'static,
    R: std::fmt::Debug,
{
    if let aws_sdk_dynamodb::error::SdkError::ServiceError(s) = &e {
        if s.err().code() == Some("ConditionalCheckFailedException") {
            return StoreError::Condition("conditional check failed".into());
        }
    }
    StoreError::Unavailable(format!("{}", aws_sdk_dynamodb::error::DisplayErrorContext(e)))
}

/// Build all backends from the configuration.
pub async fn backends(cfg: &Config) -> Result<Backends, String> {
    if cfg.s3_bucket.is_empty() {
        return Err("HEAPDB_S3_BUCKET is required in cluster mode".into());
    }
    let (s3, ddb) = clients().await;
    Ok(Backends {
        objects: std::sync::Arc::new(S3Store {
            s3,
            bucket: cfg.s3_bucket.clone(),
        }),
        leases: std::sync::Arc::new(DdbLeases {
            ddb: ddb.clone(),
            table: cfg.ddb_leases.clone(),
        }),
        registry: std::sync::Arc::new(DdbRegistry {
            ddb: ddb.clone(),
            table: cfg.ddb_registry.clone(),
        }),
        usage: std::sync::Arc::new(DdbUsage {
            ddb,
            table: cfg.ddb_usage.clone(),
        }),
    })
}

async fn clients() -> (S3, Ddb) {
    let conf = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let force_path =
        std::env::var("HEAPDB_S3_FORCE_PATH_STYLE").is_ok_and(|v| matches!(v.as_str(), "1" | "true"));
    let s3conf = aws_sdk_s3::config::Builder::from(&conf)
        .force_path_style(force_path)
        .build();
    (S3::from_conf(s3conf), Ddb::new(&conf))
}

/// Create the bucket and tables if missing (development, emulators, tests).
pub async fn ensure_resources(cfg: &Config) -> Result<(), String> {
    let (s3, ddb) = clients().await;
    if s3.head_bucket().bucket(&cfg.s3_bucket).send().await.is_err() {
        s3.create_bucket()
            .bucket(&cfg.s3_bucket)
            .send()
            .await
            .map(|_| ())
            .map_err(|e| format!("create bucket: {}", aws_sdk_s3::error::DisplayErrorContext(e)))
            .or_else(|e| {
                if e.contains("BucketAlready") {
                    Ok(())
                } else {
                    Err(e)
                }
            })?;
    }
    let existing = ddb
        .list_tables()
        .send()
        .await
        .map_err(|e| format!("list tables: {}", aws_sdk_dynamodb::error::DisplayErrorContext(e)))?
        .table_names
        .unwrap_or_default();
    let specs: [(&str, &str, Option<&str>); 3] = [
        (&cfg.ddb_leases, "database_id", None),
        (&cfg.ddb_registry, "pk", None),
        (&cfg.ddb_usage, "database_id", Some("bucket")),
    ];
    for (table, pk, sk) in specs {
        if existing.iter().any(|t| t == table) {
            continue;
        }
        let attr = |n: &str| {
            AttributeDefinition::builder()
                .attribute_name(n)
                .attribute_type(ScalarAttributeType::S)
                .build()
                .expect("valid attribute")
        };
        let key = |n: &str, t: KeyType| {
            KeySchemaElement::builder()
                .attribute_name(n)
                .key_type(t)
                .build()
                .expect("valid key")
        };
        let mut req = ddb
            .create_table()
            .table_name(table)
            .billing_mode(BillingMode::PayPerRequest)
            .attribute_definitions(attr(pk))
            .key_schema(key(pk, KeyType::Hash));
        if let Some(sk) = sk {
            req = req
                .attribute_definitions(attr(sk))
                .key_schema(key(sk, KeyType::Range));
        }
        if let Err(e) = req.send().await {
            let msg = format!("{}", aws_sdk_dynamodb::error::DisplayErrorContext(e));
            if !msg.contains("ResourceInUseException") {
                return Err(format!("create table {table}: {msg}"));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// S3
// ---------------------------------------------------------------------------

/// S3 object store.
pub struct S3Store {
    s3: S3,
    bucket: String,
}

#[async_trait]
impl ObjectStore for S3Store {
    async fn put_if_absent(&self, key: &str, body: Bytes) -> StoreResult<()> {
        let r = self
            .s3
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .if_none_match("*")
            .body(ByteStream::from(body))
            .send()
            .await;
        match r {
            Ok(_) => Ok(()),
            Err(aws_sdk_s3::error::SdkError::ServiceError(s))
                if matches!(s.raw().status().as_u16(), 409 | 412) =>
            {
                Err(StoreError::Condition(format!("object {key} exists")))
            }
            Err(e) => Err(sdk_err(e)),
        }
    }

    async fn get(&self, key: &str) -> StoreResult<Option<Bytes>> {
        match self.s3.get_object().bucket(&self.bucket).key(key).send().await {
            Ok(o) => Ok(Some(o.body.collect().await.map_err(unavailable)?.into_bytes())),
            Err(aws_sdk_s3::error::SdkError::ServiceError(s)) if s.err().is_no_such_key() => Ok(None),
            Err(e) => Err(sdk_err(e)),
        }
    }

    async fn list(&self, prefix: &str) -> StoreResult<Vec<String>> {
        let mut out = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let r = self
                .s3
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(prefix)
                .set_continuation_token(token.take())
                .send()
                .await
                .map_err(sdk_err)?;
            out.extend(r.contents().iter().filter_map(|o| o.key().map(str::to_string)));
            match r.next_continuation_token() {
                Some(t) if r.is_truncated().unwrap_or(false) => token = Some(t.to_string()),
                _ => break,
            }
        }
        out.sort();
        Ok(out)
    }

    async fn delete(&self, key: &str) -> StoreResult<()> {
        self.s3
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(sdk_err)?;
        Ok(())
    }

    async fn ping(&self) -> StoreResult<()> {
        self.s3
            .head_bucket()
            .bucket(&self.bucket)
            .send()
            .await
            .map_err(sdk_err)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DynamoDB helpers
// ---------------------------------------------------------------------------

fn s(v: &str) -> AttributeValue {
    AttributeValue::S(v.to_string())
}

fn n(v: u64) -> AttributeValue {
    AttributeValue::N(v.to_string())
}

fn get_s(item: &HashMap<String, AttributeValue>, k: &str) -> StoreResult<String> {
    item.get(k)
        .and_then(|v| v.as_s().ok())
        .cloned()
        .ok_or_else(|| StoreError::Corrupt(format!("missing string attribute {k}")))
}

fn get_n(item: &HashMap<String, AttributeValue>, k: &str) -> u64 {
    item.get(k)
        .and_then(|v| v.as_n().ok())
        .and_then(|x| x.parse::<u64>().ok())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Leases
// ---------------------------------------------------------------------------

/// DynamoDB lease table.
pub struct DdbLeases {
    ddb: Ddb,
    table: String,
}

fn lease_of(item: &HashMap<String, AttributeValue>) -> StoreResult<Lease> {
    Ok(Lease {
        database_id: get_s(item, "database_id")?,
        owner_node: get_s(item, "owner_node")?,
        owner_addr: get_s(item, "owner_addr")?,
        epoch: get_n(item, "epoch"),
        expires_at_ms: get_n(item, "expires_at"),
    })
}

#[async_trait]
impl LeaseStore for DdbLeases {
    async fn get(&self, database_id: &str) -> StoreResult<Option<Lease>> {
        let r = self
            .ddb
            .get_item()
            .table_name(&self.table)
            .key("database_id", s(database_id))
            .consistent_read(true)
            .send()
            .await
            .map_err(ddb_err)?;
        r.item().map(lease_of).transpose()
    }

    async fn acquire(
        &self,
        database_id: &str,
        node: &str,
        addr: &str,
        prev: Option<&Lease>,
        expires_at_ms: u64,
    ) -> StoreResult<Lease> {
        let epoch = prev.map_or(1, |p| p.epoch + 1);
        let mut req = self
            .ddb
            .put_item()
            .table_name(&self.table)
            .item("database_id", s(database_id))
            .item("owner_node", s(node))
            .item("owner_addr", s(addr))
            .item("epoch", n(epoch))
            .item("expires_at", n(expires_at_ms));
        req = match prev {
            None => req.condition_expression("attribute_not_exists(database_id)"),
            Some(p) => req
                .condition_expression("epoch = :e AND expires_at = :x AND owner_node = :o")
                .expression_attribute_values(":e", n(p.epoch))
                .expression_attribute_values(":x", n(p.expires_at_ms))
                .expression_attribute_values(":o", s(&p.owner_node)),
        };
        req.send().await.map_err(ddb_err)?;
        Ok(Lease {
            database_id: database_id.to_string(),
            owner_node: node.to_string(),
            owner_addr: addr.to_string(),
            epoch,
            expires_at_ms,
        })
    }

    async fn renew(&self, lease: &Lease, expires_at_ms: u64) -> StoreResult<Lease> {
        self.ddb
            .update_item()
            .table_name(&self.table)
            .key("database_id", s(&lease.database_id))
            .update_expression("SET expires_at = :x")
            .condition_expression("epoch = :e AND owner_node = :o")
            .expression_attribute_values(":x", n(expires_at_ms))
            .expression_attribute_values(":e", n(lease.epoch))
            .expression_attribute_values(":o", s(&lease.owner_node))
            .send()
            .await
            .map_err(ddb_err)?;
        Ok(Lease {
            expires_at_ms,
            ..lease.clone()
        })
    }

    async fn release(&self, lease: &Lease) -> StoreResult<()> {
        self.ddb
            .update_item()
            .table_name(&self.table)
            .key("database_id", s(&lease.database_id))
            .update_expression("SET expires_at = :x")
            .condition_expression("epoch = :e AND owner_node = :o")
            .expression_attribute_values(":x", n(0))
            .expression_attribute_values(":e", n(lease.epoch))
            .expression_attribute_values(":o", s(&lease.owner_node))
            .send()
            .await
            .map_err(ddb_err)?;
        Ok(())
    }

    async fn ping(&self) -> StoreResult<()> {
        self.get("__ping__").await.map(|_| ())
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// DynamoDB registry table.
pub struct DdbRegistry {
    ddb: Ddb,
    table: String,
}

impl DdbRegistry {
    async fn get_json<T: serde::de::DeserializeOwned>(&self, pk: &str) -> StoreResult<Option<T>> {
        let r = self
            .ddb
            .get_item()
            .table_name(&self.table)
            .key("pk", s(pk))
            .consistent_read(true)
            .send()
            .await
            .map_err(ddb_err)?;
        match r.item() {
            None => Ok(None),
            Some(it) => serde_json::from_str(&get_s(it, "record")?)
                .map(Some)
                .map_err(|e| StoreError::Corrupt(e.to_string())),
        }
    }

    async fn put_json<T: serde::Serialize>(&self, pk: &str, v: &T, only_new: bool) -> StoreResult<()> {
        let body = serde_json::to_string(v).map_err(|e| StoreError::Corrupt(e.to_string()))?;
        let mut req = self
            .ddb
            .put_item()
            .table_name(&self.table)
            .item("pk", s(pk))
            .item("record", s(&body));
        if only_new {
            req = req.condition_expression("attribute_not_exists(pk)");
        }
        req.send().await.map_err(ddb_err)?;
        Ok(())
    }

    async fn delete_pk(&self, pk: &str) -> StoreResult<()> {
        self.ddb
            .delete_item()
            .table_name(&self.table)
            .key("pk", s(pk))
            .send()
            .await
            .map_err(ddb_err)?;
        Ok(())
    }
}

#[async_trait]
impl Registry for DdbRegistry {
    async fn get_database(&self, id: &str) -> StoreResult<Option<DbRecord>> {
        self.get_json(&format!("db#{id}")).await
    }

    async fn create_database(&self, rec: &DbRecord) -> StoreResult<Option<DbRecord>> {
        match self.put_json(&format!("db#{}", rec.database_id), rec, true).await {
            Ok(()) => Ok(None),
            Err(StoreError::Condition(_)) => self.get_database(&rec.database_id).await,
            Err(e) => Err(e),
        }
    }

    async fn put_database(&self, rec: &DbRecord) -> StoreResult<()> {
        self.put_json(&format!("db#{}", rec.database_id), rec, false)
            .await
    }

    async fn delete_database(&self, id: &str) -> StoreResult<()> {
        self.delete_pk(&format!("db#{id}")).await
    }

    async fn list_databases(&self) -> StoreResult<Vec<DbRecord>> {
        let mut out = Vec::new();
        let mut start: Option<HashMap<String, AttributeValue>> = None;
        loop {
            let r = self
                .ddb
                .scan()
                .table_name(&self.table)
                .filter_expression("begins_with(pk, :p)")
                .expression_attribute_values(":p", s("db#"))
                .set_exclusive_start_key(start.take())
                .send()
                .await
                .map_err(ddb_err)?;
            for it in r.items() {
                let rec: DbRecord = serde_json::from_str(&get_s(it, "record")?)
                    .map_err(|e| StoreError::Corrupt(e.to_string()))?;
                out.push(rec);
            }
            match r.last_evaluated_key() {
                Some(k) if !k.is_empty() => start = Some(k.clone()),
                _ => break,
            }
        }
        Ok(out)
    }

    async fn get_key(&self, sha256: &str) -> StoreResult<Option<KeyRecord>> {
        self.get_json(&format!("key#{sha256}")).await
    }

    async fn put_key(&self, rec: &KeyRecord) -> StoreResult<()> {
        self.put_json(&format!("key#{}", rec.sha256), rec, false).await
    }

    async fn delete_key(&self, sha256: &str) -> StoreResult<()> {
        self.delete_pk(&format!("key#{sha256}")).await
    }

    async fn ping(&self) -> StoreResult<()> {
        self.get_json::<serde_json::Value>("ping#").await.map(|_| ())
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

/// DynamoDB usage table.
pub struct DdbUsage {
    ddb: Ddb,
    table: String,
}

fn usage_of(it: &HashMap<String, AttributeValue>) -> StoreResult<UsageItem> {
    Ok(UsageItem {
        database_id: get_s(it, "database_id")?,
        bucket: get_s(it, "bucket")?,
        requests: get_n(it, "requests"),
        events_written: get_n(it, "events_written"),
        events_stored: get_n(it, "events_stored"),
        storage_bytes: get_n(it, "storage_bytes"),
    })
}

#[async_trait]
impl UsageStore for DdbUsage {
    async fn add(
        &self,
        database_id: &str,
        bucket: &str,
        requests: u64,
        events_written: u64,
    ) -> StoreResult<()> {
        self.ddb
            .update_item()
            .table_name(&self.table)
            .key("database_id", s(database_id))
            .key("bucket", s(bucket))
            .update_expression("ADD requests :r, events_written :w")
            .expression_attribute_values(":r", n(requests))
            .expression_attribute_values(":w", n(events_written))
            .send()
            .await
            .map_err(ddb_err)?;
        Ok(())
    }

    async fn set_gauges(&self, database_id: &str, events_stored: u64, storage_bytes: u64) -> StoreResult<()> {
        self.ddb
            .update_item()
            .table_name(&self.table)
            .key("database_id", s(database_id))
            .key("bucket", s(GAUGE_BUCKET))
            .update_expression("SET events_stored = :e, storage_bytes = :b")
            .expression_attribute_values(":e", n(events_stored))
            .expression_attribute_values(":b", n(storage_bytes))
            .send()
            .await
            .map_err(ddb_err)?;
        Ok(())
    }

    async fn get(&self, database_id: &str, bucket: &str) -> StoreResult<Option<UsageItem>> {
        let r = self
            .ddb
            .get_item()
            .table_name(&self.table)
            .key("database_id", s(database_id))
            .key("bucket", s(bucket))
            .consistent_read(true)
            .send()
            .await
            .map_err(ddb_err)?;
        r.item().map(usage_of).transpose()
    }

    async fn scan(&self) -> StoreResult<Vec<UsageItem>> {
        let mut out = Vec::new();
        let mut start: Option<HashMap<String, AttributeValue>> = None;
        loop {
            let r = self
                .ddb
                .scan()
                .table_name(&self.table)
                .consistent_read(true)
                .set_exclusive_start_key(start.take())
                .send()
                .await
                .map_err(ddb_err)?;
            for it in r.items() {
                out.push(usage_of(it)?);
            }
            match r.last_evaluated_key() {
                Some(k) if !k.is_empty() => start = Some(k.clone()),
                _ => break,
            }
        }
        Ok(out)
    }

    async fn delete_gauges(&self, database_id: &str) -> StoreResult<()> {
        self.ddb
            .delete_item()
            .table_name(&self.table)
            .key("database_id", s(database_id))
            .key("bucket", s(GAUGE_BUCKET))
            .send()
            .await
            .map_err(ddb_err)?;
        Ok(())
    }
}
