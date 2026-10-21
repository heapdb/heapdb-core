//! Local-disk backends for single-node mode.
//!
//! Objects are files under `<data-dir>/objects/<key>`. A put writes a
//! temporary file, fsyncs it, hard-links it into place (which fails if the
//! key exists, giving put-if-absent) and fsyncs the directory, so a WAL
//! segment is durable before the write is acknowledged.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::{ObjectStore, StoreError, StoreResult};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_name(path: &Path) -> PathBuf {
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().map(|s| s.to_os_string()).unwrap_or_default();
    name.push(format!(".tmp.{}.{n}", std::process::id()));
    path.with_file_name(name)
}

fn fsync_dir(dir: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
    Ok(())
}

/// Write `value` as JSON to `path` atomically (temp file + rename).
pub fn save_json_atomic<T: Serialize + ?Sized>(path: &Path, value: &T, fsync: bool) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = tmp_name(path);
    {
        let mut f = File::create(&tmp)?;
        serde_json::to_writer(&mut f, value).map_err(std::io::Error::other)?;
        f.flush()?;
        if fsync {
            f.sync_all()?;
        }
    }
    fs::rename(&tmp, path)?;
    if fsync {
        if let Some(dir) = path.parent() {
            fsync_dir(dir)?;
        }
    }
    Ok(())
}

/// Read JSON from `path` (`None` if the file does not exist).
pub fn load_json<T: DeserializeOwned>(path: &Path) -> std::io::Result<Option<T>> {
    match fs::read(path) {
        Ok(b) => serde_json::from_slice(&b)
            .map(Some)
            .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Object store on the local file system.
pub struct LocalObjectStore {
    root: PathBuf,
    fsync: bool,
}

impl LocalObjectStore {
    /// Objects under `root`.
    pub fn new(root: impl Into<PathBuf>, fsync: bool) -> std::io::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;
        Ok(LocalObjectStore { root, fsync })
    }

    fn path(&self, key: &str) -> StoreResult<PathBuf> {
        if key.is_empty() || key.split('/').any(|p| p.is_empty() || p == "." || p == "..") {
            return Err(StoreError::Unavailable(format!("invalid object key {key:?}")));
        }
        Ok(self.root.join(key))
    }
}

fn unavailable(e: impl std::fmt::Display) -> StoreError {
    StoreError::Unavailable(e.to_string())
}

fn put_blocking(path: &Path, body: &[u8], fsync: bool) -> StoreResult<()> {
    let dir = path
        .parent()
        .ok_or_else(|| unavailable("object without parent"))?;
    fs::create_dir_all(dir).map_err(unavailable)?;
    let tmp = tmp_name(path);
    let res = (|| -> StoreResult<()> {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(unavailable)?;
        f.write_all(body).map_err(unavailable)?;
        if fsync {
            f.sync_all().map_err(unavailable)?;
        }
        match fs::hard_link(&tmp, path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(StoreError::Condition(format!("object {} exists", path.display())))
            }
            Err(e) => return Err(unavailable(e)),
        }
        if fsync {
            fsync_dir(dir).map_err(unavailable)?;
        }
        Ok(())
    })();
    let _ = fs::remove_file(&tmp);
    res
}

fn list_blocking(root: &Path, prefix: &str) -> StoreResult<Vec<String>> {
    // Walk the directory of the prefix (keys use '/' separators).
    let (dir_part, _) = match prefix.rfind('/') {
        Some(i) => (&prefix[..i], &prefix[i + 1..]),
        None => ("", prefix),
    };
    let start = if dir_part.is_empty() {
        root.to_path_buf()
    } else {
        root.join(dir_part)
    };
    let mut out = Vec::new();
    let mut stack = vec![start];
    while let Some(dir) = stack.pop() {
        let rd = match fs::read_dir(&dir) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(unavailable(e)),
        };
        for ent in rd {
            let ent = ent.map_err(unavailable)?;
            let p = ent.path();
            let ft = ent.file_type().map_err(unavailable)?;
            if ft.is_dir() {
                stack.push(p);
                continue;
            }
            let rel = p.strip_prefix(root).map_err(unavailable)?;
            let key: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            let key = key.join("/");
            if key.contains(".tmp.") {
                continue;
            }
            if key.starts_with(prefix) {
                out.push(key);
            }
        }
    }
    out.sort();
    Ok(out)
}

#[async_trait]
impl ObjectStore for LocalObjectStore {
    async fn put_if_absent(&self, key: &str, body: Bytes) -> StoreResult<()> {
        let path = self.path(key)?;
        let fsync = self.fsync;
        tokio::task::spawn_blocking(move || put_blocking(&path, &body, fsync))
            .await
            .map_err(unavailable)?
    }

    async fn get(&self, key: &str) -> StoreResult<Option<Bytes>> {
        let path = self.path(key)?;
        match tokio::fs::read(&path).await {
            Ok(b) => Ok(Some(Bytes::from(b))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(unavailable(e)),
        }
    }

    async fn list(&self, prefix: &str) -> StoreResult<Vec<String>> {
        let root = self.root.clone();
        let prefix = prefix.to_string();
        tokio::task::spawn_blocking(move || list_blocking(&root, &prefix))
            .await
            .map_err(unavailable)?
    }

    async fn delete(&self, key: &str) -> StoreResult<()> {
        let path = self.path(key)?;
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(unavailable(e)),
        }
    }

    async fn ping(&self) -> StoreResult<()> {
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || {
            let p = root.join(".readyz");
            fs::write(&p, b"ok").map_err(unavailable)?;
            fs::remove_file(&p).map_err(unavailable)
        })
        .await
        .map_err(unavailable)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn put_if_absent_list_delete() {
        let dir = tempfile::tempdir().unwrap();
        let s = LocalObjectStore::new(dir.path(), true).unwrap();
        s.put_if_absent("db/x/wal/1-1.jsonl", Bytes::from_static(b"a"))
            .await
            .unwrap();
        s.put_if_absent("db/x/wal/1-2.jsonl", Bytes::from_static(b"b"))
            .await
            .unwrap();
        s.put_if_absent("db/x/snap/1-2.jsonl", Bytes::from_static(b"c"))
            .await
            .unwrap();
        assert!(matches!(
            s.put_if_absent("db/x/wal/1-1.jsonl", Bytes::from_static(b"z"))
                .await,
            Err(StoreError::Condition(_))
        ));
        assert_eq!(
            s.list("db/x/wal/").await.unwrap(),
            vec!["db/x/wal/1-1.jsonl", "db/x/wal/1-2.jsonl"]
        );
        assert_eq!(s.list("db/x/").await.unwrap().len(), 3);
        assert_eq!(&s.get("db/x/wal/1-1.jsonl").await.unwrap().unwrap()[..], b"a");
        s.delete("db/x/wal/1-1.jsonl").await.unwrap();
        s.delete("db/x/wal/1-1.jsonl").await.unwrap();
        assert!(s.get("db/x/wal/1-1.jsonl").await.unwrap().is_none());
        assert!(s.put_if_absent("../evil", Bytes::new()).await.is_err());
        s.ping().await.unwrap();
    }
}
