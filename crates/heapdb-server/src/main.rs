//! `heapdb-server`: the HeapDB data plane binary.
//!
//! ```text
//! heapdb-server --data-dir ./data            # single node (default)
//! HEAPDB_CLUSTER=aws heapdb-server ...       # cluster on S3 + DynamoDB
//! heapdb-server keygen                       # print a new API key and its sha256
//! heapdb-server new-id                       # print a new database id
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use heapdb_server::config::{ClusterMode, Config};
use heapdb_server::ids::{new_api_key, new_database_id, sha256_hex};
use heapdb_server::server::{backends_for, Server};

#[derive(Parser, Debug)]
#[command(
    name = "heapdb-server",
    version,
    about = "HeapDB data plane (Data API + Admin API)"
)]
struct Args {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Public listener (Data API, Admin API, health).
    #[arg(long, env = "HEAPDB_LISTEN", default_value = "0.0.0.0:8080")]
    listen: SocketAddr,
    /// Internal listener (forwarding, /metrics). Keep it private.
    #[arg(long, env = "HEAPDB_INTERNAL_LISTEN", default_value = "0.0.0.0:9090")]
    internal_listen: SocketAddr,
    /// `off` (single node, local disk) or `aws` (S3 + DynamoDB).
    #[arg(long, env = "HEAPDB_CLUSTER", default_value = "off")]
    cluster: ClusterMode,
    /// This node's private ip:port of the internal listener (cluster mode;
    /// read from ECS task metadata when unset).
    #[arg(long, env = "HEAPDB_NODE_ADDR")]
    node_addr: Option<String>,
    /// Data directory (single node) or local cache (cluster).
    #[arg(long, env = "HEAPDB_DATA_DIR", default_value = "./data")]
    data_dir: PathBuf,
    /// Memory budget for open databases in MB (0 = unlimited).
    #[arg(long, env = "HEAPDB_MEMORY_BUDGET_MB", default_value_t = 1024)]
    memory_budget_mb: u64,
    /// S3 bucket for WAL and snapshots (cluster mode).
    #[arg(long, env = "HEAPDB_S3_BUCKET", default_value = "")]
    s3_bucket: String,
    /// Key prefix inside the bucket.
    #[arg(long, env = "HEAPDB_S3_PREFIX", default_value = "")]
    s3_prefix: String,
    /// DynamoDB table of leases.
    #[arg(long, env = "HEAPDB_DDB_LEASES", default_value = "heapdb-leases")]
    ddb_leases: String,
    /// DynamoDB table of the registry.
    #[arg(long, env = "HEAPDB_DDB_REGISTRY", default_value = "heapdb-registry")]
    ddb_registry: String,
    /// DynamoDB table of usage counters.
    #[arg(long, env = "HEAPDB_DDB_USAGE", default_value = "heapdb-usage")]
    ddb_usage: String,
    /// Log format: json (default) or text.
    #[arg(long, env = "HEAPDB_LOG_FORMAT", default_value = "json")]
    log_format: String,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Print a new API key and the sha256 to register with the Admin API.
    Keygen,
    /// Print a new database id.
    NewId,
    /// Create the S3 bucket and DynamoDB tables if missing (cluster mode).
    #[cfg(feature = "cluster-aws")]
    AwsInit,
}

fn init_logging(format: &str) {
    let filter = tracing_subscriber::EnvFilter::try_from_env("HEAPDB_LOG")
        .or_else(|_| tracing_subscriber::EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let b = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false);
    if format == "text" {
        b.init();
    } else {
        b.json().flatten_event(true).with_current_span(false).init();
    }
}

/// `ip:port` of this task from ECS task metadata (v4).
async fn ecs_node_addr(port: u16) -> Option<String> {
    let uri = std::env::var("ECS_CONTAINER_METADATA_URI_V4").ok()?;
    let v: serde_json::Value = reqwest::get(format!("{uri}/task"))
        .await
        .ok()?
        .text()
        .await
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())?;
    let ip = v["Containers"]
        .as_array()?
        .iter()
        .find_map(|c| c["Networks"][0]["IPv4Addresses"][0].as_str())?;
    Some(format!("{ip}:{port}"))
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    match &args.cmd {
        Some(Cmd::Keygen) => {
            let k = new_api_key();
            println!("api_key: {k}\nsha256:  {}", sha256_hex(k.as_bytes()));
            return;
        }
        Some(Cmd::NewId) => {
            println!("{}", new_database_id());
            return;
        }
        _ => {}
    }
    init_logging(&args.log_format);
    let mut cfg = Config::single_node(&args.data_dir);
    cfg.listen = args.listen;
    cfg.internal_listen = args.internal_listen;
    cfg.cluster = args.cluster;
    cfg.memory_budget_mb = args.memory_budget_mb;
    cfg.s3_bucket = args.s3_bucket;
    cfg.s3_prefix = args.s3_prefix;
    cfg.ddb_leases = args.ddb_leases;
    cfg.ddb_registry = args.ddb_registry;
    cfg.ddb_usage = args.ddb_usage;
    cfg.admin_token = std::env::var("HEAPDB_ADMIN_TOKEN").ok().filter(|t| !t.is_empty());
    cfg.node_addr = match args.node_addr {
        Some(a) => Some(a),
        None if cfg.cluster == ClusterMode::Aws => ecs_node_addr(cfg.internal_listen.port()).await,
        None => None,
    };
    if let Err(e) = cfg.apply_tuning_env() {
        eprintln!("heapdb-server: {e}");
        std::process::exit(2);
    }
    #[cfg(feature = "cluster-aws")]
    if matches!(args.cmd, Some(Cmd::AwsInit)) {
        match heapdb_server::storage::aws::ensure_resources(&cfg).await {
            Ok(()) => println!("ok: bucket {} and tables ready", cfg.s3_bucket),
            Err(e) => {
                eprintln!("heapdb-server: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    if cfg.cluster == ClusterMode::Aws && cfg.node_addr.is_none() {
        eprintln!("heapdb-server: cluster mode needs HEAPDB_NODE_ADDR (or ECS task metadata)");
        std::process::exit(2);
    }
    if cfg.admin_token.is_none() {
        tracing::warn!("HEAPDB_ADMIN_TOKEN is not set: the Admin API is disabled");
    }
    let backends = match backends_for(&cfg).await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("heapdb-server: {e}");
            std::process::exit(2);
        }
    };
    let server = match Server::start(cfg, backends).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("heapdb-server: cannot listen: {e}");
            std::process::exit(2);
        }
    };
    tracing::info!(public = %server.public_addr, internal = %server.internal_addr, "listening");
    shutdown_signal().await;
    server.shutdown().await;
}
