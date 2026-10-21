//! Server lifecycle: bind listeners, run background work, shut down
//! gracefully (stop accepting, finish in-flight requests, flush the WAL,
//! snapshot open databases, release their leases).

use std::net::SocketAddr;
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::config::{ClusterMode, Config};
use crate::database::NodeCtx;
use crate::http::{internal_router, public_router, AppState};
use crate::ids::new_node_id;
use crate::manager::Manager;
use crate::metrics::Metrics;
use crate::storage::local::LocalObjectStore;
use crate::storage::memory::{MemLeaseStore, MemRegistry, MemUsageStore};
use crate::storage::Backends;

/// A running node.
pub struct Server {
    /// Bound public address.
    pub public_addr: SocketAddr,
    /// Bound internal address.
    pub internal_addr: SocketAddr,
    /// The database manager (tests inspect it).
    pub manager: Arc<Manager>,
    http_stop: CancellationToken,
    bg_stop: CancellationToken,
    http_tasks: Vec<JoinHandle<()>>,
    bg_task: JoinHandle<()>,
}

/// Single-node backends under `cfg.data_dir`.
pub fn local_backends(cfg: &Config) -> std::io::Result<Backends> {
    let d = &cfg.data_dir;
    std::fs::create_dir_all(d)?;
    Ok(Backends {
        objects: Arc::new(LocalObjectStore::new(d.join("objects"), cfg.fsync)?),
        leases: Arc::new(MemLeaseStore::persistent(d.join("leases.json"))?),
        registry: Arc::new(MemRegistry::persistent(d.join("registry.json"))?),
        usage: Arc::new(MemUsageStore::persistent(d.join("usage.json"))?),
    })
}

/// Backends for `cfg.cluster`.
pub async fn backends_for(cfg: &Config) -> Result<Backends, String> {
    match cfg.cluster {
        ClusterMode::Off => {
            local_backends(cfg).map_err(|e| format!("data dir {}: {e}", cfg.data_dir.display()))
        }
        #[cfg(feature = "cluster-aws")]
        ClusterMode::Aws => crate::storage::aws::backends(cfg).await,
        #[cfg(not(feature = "cluster-aws"))]
        ClusterMode::Aws => Err("HEAPDB_CLUSTER=aws needs a build with the cluster-aws feature".into()),
    }
}

impl Server {
    /// Bind and start. `cfg.node_addr` defaults to the internal listener's
    /// address (loopback if it is bound to all interfaces).
    pub async fn start(cfg: Config, backends: Backends) -> std::io::Result<Server> {
        let public = TcpListener::bind(cfg.listen).await?;
        let internal = TcpListener::bind(cfg.internal_listen).await?;
        let public_addr = public.local_addr()?;
        let internal_addr = internal.local_addr()?;
        let node_addr = cfg.node_addr.clone().unwrap_or_else(|| {
            if internal_addr.ip().is_unspecified() {
                format!("127.0.0.1:{}", internal_addr.port())
            } else {
                internal_addr.to_string()
            }
        });
        let node_id = new_node_id();
        tracing::info!(
            node_id = %node_id,
            node_addr = %node_addr,
            public = %public_addr,
            internal = %internal_addr,
            cluster = ?cfg.cluster,
            "heapdb-server starting"
        );
        let ctx = Arc::new(NodeCtx {
            cfg: Arc::new(cfg),
            backends,
            node_id,
            node_addr,
            metrics: Arc::new(Metrics::default()),
        });
        let manager = Manager::new(ctx);
        let st = AppState { mgr: manager.clone() };
        let http_stop = CancellationToken::new();
        let bg_stop = CancellationToken::new();
        let mut http_tasks = Vec::new();
        for (listener, router) in [
            (public, public_router(st.clone())),
            (internal, internal_router(st)),
        ] {
            let stop = http_stop.clone();
            http_tasks.push(tokio::spawn(async move {
                let r = axum::serve(listener, router)
                    .with_graceful_shutdown(stop.cancelled_owned())
                    .await;
                if let Err(e) = r {
                    tracing::error!(error = %e, "listener failed");
                }
            }));
        }
        let bg_task = tokio::spawn(manager.clone().run_background(bg_stop.clone()));
        Ok(Server {
            public_addr,
            internal_addr,
            manager,
            http_stop,
            bg_stop,
            http_tasks,
            bg_task,
        })
    }

    /// Graceful shutdown.
    pub async fn shutdown(self) {
        tracing::info!("shutting down: draining requests");
        self.http_stop.cancel();
        for t in self.http_tasks {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(30), t).await;
        }
        tracing::info!("shutting down: closing databases");
        self.manager.shutdown().await;
        self.bg_stop.cancel();
        let _ = self.bg_task.await;
        tracing::info!("shutdown complete");
    }

    /// Stop everything at once without flushing anything (tests: crash).
    pub fn abort(self) {
        self.http_stop.cancel();
        self.bg_stop.cancel();
        for t in &self.http_tasks {
            t.abort();
        }
        self.bg_task.abort();
        for db in self.manager.open_databases() {
            db.poison("node aborted");
        }
    }
}
