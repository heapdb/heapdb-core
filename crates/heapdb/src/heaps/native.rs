//! [`NativeHeaps`]: `persist_to = "native:<dir>"`.
//!
//! Durable heaps for one process, in a directory:
//!
//! * `ops.jsonl`: an append-only log, one [`HeapOp`] per line with its
//!   sequence number (`{"seq": 7, "op": "push", "args": {...}}`). Every
//!   mutation is applied in memory (which validates it), then appended and
//!   fsynced before the call returns.
//! * `checkpoint.jsonl`: the full state at some `seq` (a header line
//!   `{"kind":"heapdb_heaps","format":1,"seq":n}` followed by
//!   [`MemoryHeaps::to_records`]). Written every
//!   [`DEFAULT_CHECKPOINT_EVERY`] ops (and by [`NativeHeaps::checkpoint`])
//!   to a temporary file that is fsynced and renamed into place; then the
//!   log is truncated.
//!
//! Opening loads the checkpoint and replays the log records with a higher
//! `seq`. A last line without its trailing newline is a torn write that was
//! never acknowledged: it is ignored and cut off. Any other unreadable line
//! is corruption and an error. A crash between the checkpoint rename and
//! the log truncation is harmless: the replay skips records the checkpoint
//! already covers.
//!
//! One process at a time: concurrent writers to the same directory are not
//! detected.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};

use super::filter::Filter;
use super::memory::{AttrsArgs, BaseArgs, MemoryHeaps, PushArgs};
use super::ops::HeapOp;
use super::{check_finite, HeapBackend, HeapInfo, HeapItem, HeapSpec, Ranked};
use crate::error::{Error, Result};
use crate::event::Attrs;

/// Ops between automatic checkpoints.
pub const DEFAULT_CHECKPOINT_EVERY: u64 = 10_000;

const LOG: &str = "ops.jsonl";
const CHECKPOINT: &str = "checkpoint.jsonl";
const CHECKPOINT_TMP: &str = "checkpoint.jsonl.tmp";
const KIND: &str = "heapdb_heaps";

/// Heaps in a directory: op log + checkpoints (see the module docs).
#[derive(Debug)]
pub struct NativeHeaps {
    dir: PathBuf,
    mem: MemoryHeaps,
    log: File,
    /// Last sequence number written (or replayed).
    seq: u64,
    /// `seq` covered by the checkpoint on disk.
    checkpoint_seq: u64,
    checkpoint_every: u64,
}

fn corrupt(path: &Path, line: usize, msg: impl std::fmt::Display) -> Error {
    Error::Snapshot(format!("{}:{line}: {msg}", path.display()))
}

fn sync_dir(dir: &Path) -> Result<()> {
    // Directory fsync makes renames and new files durable (POSIX); not
    // supported on Windows, where the rename itself is enough.
    #[cfg(unix)]
    File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Load `checkpoint.jsonl`: `(state, seq)`.
fn load_checkpoint(path: &Path) -> Result<(MemoryHeaps, u64)> {
    let mut mem = MemoryHeaps::new();
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((mem, 0)),
        Err(e) => return Err(e.into()),
    };
    let mut seq = None;
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line).map_err(|e| corrupt(path, i + 1, e))?;
        let Value::Object(rec) = v else {
            return Err(corrupt(path, i + 1, "record must be an object"));
        };
        if seq.is_none() {
            if rec.get("kind").and_then(Value::as_str) != Some(KIND) || rec.get("format") != Some(&json!(1)) {
                return Err(corrupt(path, i + 1, "not a heapdb heaps checkpoint (format 1)"));
            }
            seq = Some(
                rec.get("seq")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| corrupt(path, i + 1, "bad seq"))?,
            );
            continue;
        }
        if !mem.load_record(&rec).map_err(|e| corrupt(path, i + 1, e))? {
            return Err(corrupt(path, i + 1, "unknown record kind"));
        }
    }
    Ok((mem, seq.ok_or_else(|| corrupt(path, 1, "empty checkpoint"))?))
}

/// Replay `ops.jsonl` onto `mem`; returns the last seq and the length of
/// the valid prefix (a torn last line is excluded).
fn replay(path: &Path, mem: &mut MemoryHeaps, after: u64) -> Result<(u64, u64)> {
    let mut bytes = Vec::new();
    match File::open(path) {
        Ok(mut f) => {
            f.read_to_end(&mut bytes)?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((after, 0)),
        Err(e) => return Err(e.into()),
    }
    let mut seq = after;
    let mut valid = 0u64;
    let mut start = 0usize;
    let mut lineno = 0usize;
    while start < bytes.len() {
        lineno += 1;
        let Some(nl) = bytes[start..].iter().position(|&b| b == b'\n') else {
            break; // torn last write: never acknowledged
        };
        let line = &bytes[start..start + nl];
        start += nl + 1;
        if line.iter().all(u8::is_ascii_whitespace) {
            valid = start as u64;
            continue;
        }
        let rec: Map<String, Value> = serde_json::from_slice(line).map_err(|e| corrupt(path, lineno, e))?;
        let s = rec
            .get("seq")
            .and_then(Value::as_u64)
            .ok_or_else(|| corrupt(path, lineno, "missing seq"))?;
        if s > after {
            if s != seq + 1 {
                return Err(corrupt(
                    path,
                    lineno,
                    format!("expected seq {}, found {s}", seq + 1),
                ));
            }
            let mut body = rec.clone();
            body.remove("seq");
            let op: HeapOp =
                serde_json::from_value(Value::Object(body)).map_err(|e| corrupt(path, lineno, e))?;
            mem.apply(&op)
                .map_err(|e| corrupt(path, lineno, format!("replaying seq {s}: {e}")))?;
            seq = s;
        }
        valid = start as u64;
    }
    Ok((seq, valid))
}

impl NativeHeaps {
    /// Open (creating the directory if needed) and recover the state.
    pub fn open(dir: impl AsRef<Path>) -> Result<NativeHeaps> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        let (mut mem, checkpoint_seq) = load_checkpoint(&dir.join(CHECKPOINT))?;
        let log_path = dir.join(LOG);
        let (seq, valid) = replay(&log_path, &mut mem, checkpoint_seq)?;
        let log = OpenOptions::new().create(true).append(true).open(&log_path)?;
        if log.metadata()?.len() != valid {
            log.set_len(valid)?; // cut off a torn last line
            log.sync_all()?;
        }
        sync_dir(&dir)?;
        Ok(NativeHeaps {
            dir,
            mem,
            log,
            seq,
            checkpoint_seq,
            checkpoint_every: DEFAULT_CHECKPOINT_EVERY,
        })
    }

    /// Checkpoint every `n` ops (at least 1).
    pub fn with_checkpoint_every(mut self, n: u64) -> Self {
        self.checkpoint_every = n.max(1);
        self
    }

    /// The directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The in-memory state (read-only).
    pub fn state(&self) -> &MemoryHeaps {
        &self.mem
    }

    /// Last sequence number.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Apply an op and make it durable. On error nothing changed (if the
    /// write itself fails, the in-memory state is reloaded from disk).
    pub fn apply(&mut self, op: &HeapOp) -> Result<Value> {
        self.commit(op, |mem| mem.apply(op))
    }

    /// Run `f` (which applies `op`) on the in-memory state, then log `op`.
    fn commit<T>(&mut self, op: &HeapOp, f: impl FnOnce(&mut MemoryHeaps) -> Result<T>) -> Result<T> {
        let out = f(&mut self.mem)?;
        if let Err(e) = self.append(op) {
            // The op may or may not be on disk; disk is the truth.
            if let Ok(fresh) = NativeHeaps::open(&self.dir) {
                let every = self.checkpoint_every;
                *self = fresh.with_checkpoint_every(every);
            }
            return Err(e);
        }
        if self.seq - self.checkpoint_seq >= self.checkpoint_every {
            self.checkpoint()?;
        }
        Ok(out)
    }

    fn append(&mut self, op: &HeapOp) -> Result<()> {
        let mut rec = match serde_json::to_value(op).map_err(|e| Error::Backend(e.to_string()))? {
            Value::Object(m) => m,
            _ => unreachable!("ops serialise to objects"),
        };
        rec.insert("seq".into(), (self.seq + 1).into());
        let mut line = serde_json::to_vec(&rec).map_err(|e| Error::Backend(e.to_string()))?;
        line.push(b'\n');
        self.log.write_all(&line)?;
        self.log.sync_data()?;
        self.seq += 1;
        Ok(())
    }

    /// Write a checkpoint of the current state and truncate the log.
    pub fn checkpoint(&mut self) -> Result<()> {
        let tmp = self.dir.join(CHECKPOINT_TMP);
        {
            let mut w = std::io::BufWriter::new(File::create(&tmp)?);
            let header = json!({"kind": KIND, "format": 1, "seq": self.seq});
            writeln!(w, "{header}")?;
            for r in self.mem.to_records() {
                writeln!(w, "{r}")?;
            }
            w.flush()?;
            w.get_ref().sync_all()?;
        }
        fs::rename(&tmp, self.dir.join(CHECKPOINT))?;
        sync_dir(&self.dir)?;
        self.checkpoint_seq = self.seq;
        self.log.set_len(0)?;
        self.log.sync_all()?;
        Ok(())
    }

    /// Atomic batch of pushes.
    pub fn push_many(&mut self, heap: &str, items: &[PushArgs]) -> Result<Vec<HeapItem>> {
        let op = HeapOp::Push {
            heap: heap.into(),
            items: items.to_vec(),
        };
        self.apply_items(&op)
    }

    /// Atomic batch of `set_base`.
    pub fn set_base_many(&mut self, heap: &str, items: &[BaseArgs]) -> Result<Vec<HeapItem>> {
        let op = HeapOp::SetBase {
            heap: heap.into(),
            items: items.to_vec(),
        };
        self.apply_items(&op)
    }

    /// Atomic batch of `set_attrs`.
    pub fn set_attrs_many(&mut self, heap: &str, items: &[AttrsArgs]) -> Result<Vec<HeapItem>> {
        let op = HeapOp::SetAttrs {
            heap: heap.into(),
            items: items.to_vec(),
        };
        self.apply_items(&op)
    }

    fn apply_items(&mut self, op: &HeapOp) -> Result<Vec<HeapItem>> {
        self.commit(op, |mem| match op {
            HeapOp::Push { heap, items } => mem.push_many(heap, items),
            HeapOp::SetBase { heap, items } => mem.set_base_many(heap, items),
            HeapOp::SetAttrs { heap, items } => mem.set_attrs_many(heap, items),
            _ => unreachable!("item batches only"),
        })
    }
}

impl HeapBackend for NativeHeaps {
    fn create_heap(&mut self, spec: &HeapSpec, now: f64) -> Result<HeapInfo> {
        spec.validate()?;
        if let Some(have) = self.mem.heap(&spec.name)? {
            spec.check_matches(&have)?;
            return Ok(have);
        }
        let landmark = match spec.landmark {
            Some(l) => l,
            None => {
                check_finite("now", now)?;
                now
            }
        };
        self.apply(&HeapOp::CreateHeap {
            name: spec.name.clone(),
            half_life: spec.half_life,
            landmark: Some(landmark),
            now: landmark,
            meta: spec.meta.clone(),
        })?;
        Ok(self.mem.heap(&spec.name)?.expect("just created"))
    }

    fn drop_heap(&mut self, name: &str) -> Result<bool> {
        if self.mem.heap(name)?.is_none() {
            return Ok(false);
        }
        self.apply(&HeapOp::DropHeap { name: name.into() })?;
        Ok(true)
    }

    fn heap(&self, name: &str) -> Result<Option<HeapInfo>> {
        self.mem.heap(name)
    }

    fn heaps(&self) -> Result<Vec<HeapInfo>> {
        self.mem.heaps()
    }

    fn push(&mut self, heap: &str, id: &str, weight: f64, t: f64, attrs: Option<&Attrs>) -> Result<HeapItem> {
        let args = [PushArgs {
            id: id.into(),
            weight,
            time: t,
            attrs: attrs.cloned(),
        }];
        Ok(self.push_many(heap, &args)?.remove(0))
    }

    fn set_base(&mut self, heap: &str, id: &str, base: f64) -> Result<HeapItem> {
        let args = [BaseArgs { id: id.into(), base }];
        Ok(self.set_base_many(heap, &args)?.remove(0))
    }

    fn set_attrs(&mut self, heap: &str, id: &str, attrs: &Attrs, merge: bool) -> Result<HeapItem> {
        let args = [AttrsArgs {
            id: id.into(),
            attrs: attrs.clone(),
            merge,
        }];
        Ok(self.set_attrs_many(heap, &args)?.remove(0))
    }

    fn remove(&mut self, heap: &str, id: &str) -> Result<bool> {
        if self.mem.get(heap, id, 0.0)?.is_none() {
            return Ok(false);
        }
        self.apply(&HeapOp::Remove {
            heap: heap.into(),
            id: id.into(),
        })?;
        Ok(true)
    }

    fn get(&self, heap: &str, id: &str, now: f64) -> Result<Option<Ranked>> {
        self.mem.get(heap, id, now)
    }

    fn top_k(&self, heap: &str, k: usize, now: f64, filter: Option<&Filter>) -> Result<Vec<Ranked>> {
        self.mem.top_k(heap, k, now, filter)
    }

    fn pop(&mut self, heap: &str, now: f64, filter: Option<&Filter>) -> Result<Option<Ranked>> {
        let Some(top) = self.mem.top_k(heap, 1, now, filter)?.pop() else {
            return Ok(None);
        };
        // Log the removal of the chosen id: replay must not depend on
        // re-evaluating the filter.
        self.apply(&HeapOp::Remove {
            heap: heap.into(),
            id: top.id.clone(),
        })?;
        Ok(Some(top))
    }

    fn rebase(&mut self, heap: &str, now: f64, half_life: Option<f64>) -> Result<HeapInfo> {
        self.apply(&HeapOp::Rebase {
            heap: heap.into(),
            now,
            half_life,
        })?;
        Ok(self.mem.heap(heap)?.expect("rebased heap exists"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "heapdb-native-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn reopen_checkpoint_and_torn_line() {
        let dir = tmp();
        let mut h = NativeHeaps::open(&dir).unwrap().with_checkpoint_every(3);
        h.create_heap(&HeapSpec::new("q", 60.0).landmark(0.0), 0.0)
            .unwrap();
        for i in 0..5 {
            h.push("q", &format!("i{i}"), 1.0 + i as f64, i as f64, None)
                .unwrap();
        }
        assert!(h.pop("q", 10.0, None).unwrap().is_some());
        let want = h.state().to_records();
        assert!(dir.join(CHECKPOINT).exists());
        drop(h);
        // A torn write at the end of the log is ignored and cut off.
        let mut f = OpenOptions::new().append(true).open(dir.join(LOG)).unwrap();
        f.write_all(br#"{"seq":99,"op":"pu"#).unwrap();
        drop(f);
        let h = NativeHeaps::open(&dir).unwrap();
        assert_eq!(h.state().to_records(), want);
        assert!(!fs::read_to_string(dir.join(LOG)).unwrap().contains("\"pu"));
        // A corrupt complete line is an error.
        drop(h);
        let mut f = OpenOptions::new().append(true).open(dir.join(LOG)).unwrap();
        f.write_all(b"garbage\n").unwrap();
        drop(f);
        assert!(matches!(NativeHeaps::open(&dir), Err(Error::Snapshot(_))));
        fs::remove_dir_all(&dir).unwrap();
    }
}
