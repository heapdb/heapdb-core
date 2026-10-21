//! [`HeapOp`]: a heap mutation with every input resolved (no clock reads),
//! so replaying the same ops on the same state gives the same state. Used
//! by the [`super::NativeHeaps`] op log and by heapdb-server's WAL.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::filter::Filter;
use super::memory::{AttrsArgs, BaseArgs, MemoryHeaps, PushArgs};
use super::{HeapBackend, HeapSpec};
use crate::error::Result;
use crate::event::Attrs;

/// A heap mutation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", content = "args", rename_all = "snake_case")]
pub enum HeapOp {
    /// `create_heap` (the landmark is resolved by the caller).
    CreateHeap {
        /// Name.
        name: String,
        /// Half-life (seconds).
        half_life: f64,
        /// Landmark.
        landmark: f64,
        /// Metadata (`None`: matches anything / `{}` for a new heap).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        meta: Option<Attrs>,
    },
    /// `drop_heap`.
    DropHeap {
        /// Name.
        name: String,
    },
    /// Batched `push` (atomic).
    Push {
        /// Heap.
        heap: String,
        /// Pushes, applied in order.
        items: Vec<PushArgs>,
    },
    /// Batched `set_base` (atomic).
    SetBase {
        /// Heap.
        heap: String,
        /// Items.
        items: Vec<BaseArgs>,
    },
    /// Batched `set_attrs` (atomic).
    SetAttrs {
        /// Heap.
        heap: String,
        /// Items.
        items: Vec<AttrsArgs>,
    },
    /// `remove`.
    Remove {
        /// Heap.
        heap: String,
        /// Item id.
        id: String,
    },
    /// `pop` at `now` with an optional filter.
    Pop {
        /// Heap.
        heap: String,
        /// Clock.
        now: f64,
        /// Filter.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filter: Option<Filter>,
    },
    /// `rebase`.
    Rebase {
        /// Heap.
        heap: String,
        /// New landmark.
        now: f64,
        /// New half-life.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        half_life: Option<f64>,
    },
}

impl HeapOp {
    /// The heap this op touches.
    pub fn heap(&self) -> &str {
        match self {
            HeapOp::CreateHeap { name, .. } | HeapOp::DropHeap { name } => name,
            HeapOp::Push { heap, .. }
            | HeapOp::SetBase { heap, .. }
            | HeapOp::SetAttrs { heap, .. }
            | HeapOp::Remove { heap, .. }
            | HeapOp::Pop { heap, .. }
            | HeapOp::Rebase { heap, .. } => heap,
        }
    }
}

fn items(v: Vec<super::HeapItem>) -> Value {
    json!({"items": v.iter().map(super::HeapItem::to_json).collect::<Vec<_>>()})
}

impl MemoryHeaps {
    /// Apply an op; on error nothing changed. Results (JSON):
    /// `create_heap` / `rebase`: HeapInfo; `push` / `set_base` / `set_attrs`:
    /// `{items: [HeapItem]}`; `remove`: `{removed}`; `drop_heap`:
    /// `{dropped}`; `pop`: `{item: Ranked | null}`.
    pub fn apply(&mut self, op: &HeapOp) -> Result<Value> {
        Ok(match op {
            HeapOp::CreateHeap {
                name,
                half_life,
                landmark,
                meta,
            } => {
                let spec = HeapSpec {
                    name: name.clone(),
                    half_life: *half_life,
                    landmark: Some(*landmark),
                    meta: meta.clone(),
                };
                self.create_heap(&spec, *landmark)?.to_json()
            }
            HeapOp::DropHeap { name } => json!({"dropped": self.drop_heap(name)?}),
            HeapOp::Push { heap, items: it } => items(self.push_many(heap, it)?),
            HeapOp::SetBase { heap, items: it } => items(self.set_base_many(heap, it)?),
            HeapOp::SetAttrs { heap, items: it } => items(self.set_attrs_many(heap, it)?),
            HeapOp::Remove { heap, id } => json!({"removed": self.remove(heap, id)?}),
            HeapOp::Pop { heap, now, filter } => {
                let r = self.pop(heap, *now, filter.as_ref())?;
                json!({"item": r.as_ref().map(super::Ranked::to_json)})
            }
            HeapOp::Rebase { heap, now, half_life } => self.rebase(heap, *now, *half_life)?.to_json(),
        })
    }
}
