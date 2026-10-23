//! # heapdb-server
//!
//! The HeapDB data plane: hosts many databases behind an HTTP/JSON Data API
//! (clients) and an Admin API (control plane). Runs as a single node on
//! local disk (`HEAPDB_CLUSTER=off`, the default) or as a cluster of nodes
//! sharing S3 and DynamoDB (`HEAPDB_CLUSTER=aws`, feature `cluster-aws`).
//! See `docs/server.md` for the API reference and architecture.

#![forbid(unsafe_code)]

pub mod catalog;
pub mod config;
pub mod database;
pub mod error;
pub mod http;
pub mod ids;
pub mod json;
pub mod manager;
pub mod metrics;
pub mod quota;
pub mod schema;
pub mod server;
pub mod state;
pub mod storage;
