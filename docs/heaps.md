# Keyed heaps and `persist_to`

Status: proposal implemented on branch `keyed-heaps`. Version 0.6.

HeapDB's `ForwardDecayHeap` orders items by a forward-decay key so the order is right at every
instant without re-scoring. Today it is an in-memory library structure: each `push` is a separate
entry with a handle, there are no item attributes or filters, it is not persisted, and the server
and query language do not expose it. Work queues (the first user is pr0, a multi-tenant work
queue for humans and coding agents) need a *named, durable heap of keyed items* that many
processes read and write. This document specifies that heap and where it can live.

## Concepts

**Heap.** A named priority queue with a half-life and a landmark time `L` (Unix seconds).
`λ = ln 2 / half_life`.

**Item.** Identified by a string `id` within a heap. It carries

| Field | Meaning |
| --- | --- |
| `base` | slow-moving priority weight, `> 0`, default `1` |
| `activity` | log-sum of decayed pushes: `ln Σ wᵢ·exp(λ(tᵢ − L))`; `null` until the first push |
| `key` | `ln(base) + activity`; `null` when `activity` is `null` (not ranked) |
| `attrs` | JSON object used by filters (kind, labels, required skills, risk, …) |
| `version` | increments on every change |

Score at `now`: `score = exp(key − λ(now − L)) = base · heat(now)`, where
`heat(now) = Σ wᵢ·exp(−λ(now − tᵢ))`. Ordering by `key` (descending, ties by `id` ascending)
equals ordering by score at every instant.

### Operations

| Operation | Effect |
| --- | --- |
| `create_heap(name, half_life, landmark = now, meta = {})` | idempotent when the spec matches; error otherwise |
| `push(heap, id, weight, t = now, attrs?)` | `activity ← logaddexp(activity, ln weight + λ(t − L))`; creates the item (base 1) if missing; `attrs` replaces when given |
| `set_base(heap, id, base)` | sets `base`; creates the item unranked if missing |
| `set_attrs(heap, id, attrs, merge = false)` | replace or shallow-merge attributes |
| `remove(heap, id)` | deletes; returns whether it existed |
| `get(heap, id, now)` | the item with `score` and `heat` at `now` |
| `top_k(heap, k, now, filter?)` | ranked items matching `filter`, highest key first |
| `pop(heap, now, filter?)` | `top_k(…, 1)` then `remove`, atomically |
| `rebase(heap, now, half_life?)` | moves the landmark to `now` (and optionally changes the half-life) keeping every item's current heat: `activity ← activity − λ_old·(now − L_old)` |
| `drop_heap(name)` | deletes the heap and its items |

Rules: `weight`, `base` and `half_life` must be finite and `> 0`; times finite; ids and heap names
1–200 UTF-8 bytes. A push with `t` far in the past is allowed (it simply contributes little).
Keys stay numerically safe for decades with a day-scale half-life; `rebase` exists for longer.

### Filters

A filter is a JSON tree, evaluated against `attrs`:

```json
{"all": [
  {"attr": "kind", "op": "in", "value": ["bug", "task"]},
  {"attr": "required_skills", "op": "subset_of", "value": ["ts", "react"]},
  {"not": {"attr": "labels", "op": "contains", "value": "needs-human"}},
  {"attr": "blocked", "op": "ne", "value": true}
]}
```

Nodes: `{"all": [...]}`, `{"any": [...]}`, `{"not": node}`, and leaves
`{"attr", "op", "value"}`. Leaf ops:

| op | true when |
| --- | --- |
| `eq`, `ne` | JSON equality (numbers compared numerically) |
| `lt`, `lte`, `gt`, `gte` | both numbers, or both strings (lexicographic); mixed types never match |
| `in`, `not_in` | `value` is an array containing / not containing the attribute |
| `contains` | array attribute contains `value`, or string attribute contains the substring |
| `contains_all` | array attribute contains every element of array `value` |
| `subset_of` | every element of the array attribute is in array `value` (a missing attribute counts as `[]`) |
| `exists`, `missing` | attribute present (non-null) / absent or null |

A missing attribute matches only `ne`, `not_in`, `missing` and `subset_of`.

## `persist_to`: where heaps live

The heap semantics above are fixed; where the state lives is pluggable, the way a Postgres table
access method separates the storage engine from the table interface. Every backend implements
the same trait and passes the same conformance suite.

```rust
pub trait HeapBackend {
    fn create_heap(&mut self, spec: &HeapSpec) -> Result<HeapInfo>;
    fn drop_heap(&mut self, name: &str) -> Result<bool>;
    fn heap(&self, name: &str) -> Result<Option<HeapInfo>>;
    fn push(&mut self, heap: &str, id: &str, weight: f64, t: f64, attrs: Option<&Attrs>) -> Result<HeapItem>;
    fn set_base(&mut self, heap: &str, id: &str, base: f64) -> Result<HeapItem>;
    fn set_attrs(&mut self, heap: &str, id: &str, attrs: &Attrs, merge: bool) -> Result<HeapItem>;
    fn remove(&mut self, heap: &str, id: &str) -> Result<bool>;
    fn get(&self, heap: &str, id: &str, now: f64) -> Result<Option<Ranked>>;
    fn top_k(&self, heap: &str, k: usize, now: f64, filter: Option<&Filter>) -> Result<Vec<Ranked>>;
    fn pop(&mut self, heap: &str, now: f64, filter: Option<&Filter>) -> Result<Option<Ranked>>;
    fn rebase(&mut self, heap: &str, now: f64, half_life: Option<f64>) -> Result<HeapInfo>;
}
```

| `persist_to` | Backend | Where | Use |
| --- | --- | --- | --- |
| `memory` (default) | `MemoryHeaps` | process memory | tests, embedded, ephemeral |
| `native:<dir>` | `NativeHeaps` | append-only JSONL op log + periodic checkpoint in `<dir>` | embedded durable, single process |
| server | heapdb-server | the database's WAL + snapshots (local disk or S3) | hosted / cluster |
| `postgres://…` | `PostgresHeaps` (crate `heapdb-postgres`) | tables + functions in schema `heapdb` | apps that already run Postgres |

### The Postgres engine

`sql/postgres/heapdb.sql` installs the heap natively in Postgres: tables `heapdb.heaps` and
`heapdb.items` (with a stored generated `key` column and an index on `(heap, key DESC, id)`), and
SQL functions `heapdb.create_heap`, `push`, `set_base`, `set_attrs`, `remove`, `get`, `top_k`,
`pop`, `rebase`, `drop_heap` and `heapdb.matches(attrs, filter)`. The functions are `SECURITY
INVOKER`, so an application's row-level security policies on `heapdb.items` apply. `pop` uses
`FOR UPDATE SKIP LOCKED`, so concurrent poppers never get the same item.

Because the engine is plain SQL, any language can use it directly (`SELECT * FROM
heapdb.top_k('acme/checkout', 20, extract(epoch from now()), '{"…"}')`). `PostgresHeaps` in Rust,
the Python binding and the TypeScript client call these functions; the SQL is the single
implementation for that backend, and the conformance suite checks it against `MemoryHeaps`.

Version: `heapdb.schema_version` holds the installed version; the file is idempotent and
upgrades in place.

### Server

heapdb-server stores heaps in each database's engine state, so heap mutations are WAL records
and heaps are part of snapshots (snapshot header `format` 2; format 1 snapshots still load).

| Method & path | Scope | Body → response |
| --- | --- | --- |
| `PUT /v1/databases/{id}/heaps/{heap}` | admin | `{half_life, landmark?, meta?}` → HeapInfo |
| `GET /v1/databases/{id}/heaps` / `GET …/heaps/{heap}` | read | HeapInfo list / one |
| `DELETE …/heaps/{heap}` | admin | `{dropped}` |
| `POST …/heaps/{heap}/push` | write | `{items: [{id, weight, time?, attrs?}]}` (≤ 1000, atomic) → `{items: [HeapItem]}` |
| `POST …/heaps/{heap}/base` | write | `{items: [{id, base}]}` → `{items}` |
| `POST …/heaps/{heap}/attrs` | write | `{items: [{id, attrs, merge?}]}` → `{items}` |
| `DELETE …/heaps/{heap}/items/{item}` | write | `{removed}` |
| `GET …/heaps/{heap}/items/{item}?now=` | read | Ranked |
| `POST …/heaps/{heap}/top` | read | `{k, now?, filter?}` → `{items: [Ranked]}` |
| `POST …/heaps/{heap}/pop` | write | `{now?, filter?}` → `{item: Ranked \| null}` |
| `POST …/heaps/{heap}/rebase` | admin | `{now?, half_life?}` → HeapInfo |

Heap names in paths are percent-encoded. `Ranked` = `{rank, id, key, score, heat, base, attrs,
version}`.

### Query language

```sql
CREATE HEAP work HALF LIFE 72h;
PUSH 'CHK-12' INTO work WEIGHT 1.5 ATTRS {"kind": "bug", "required_skills": ["ts"]};
SET BASE 'CHK-12' IN work TO 4;
PEEK 5 FROM work WHERE kind IN ('bug', 'task') AND required_skills SUBSET OF ('ts', 'react');
POP FROM work WHERE NOT labels CONTAINS 'needs-human';
REMOVE 'CHK-12' FROM work;
REBASE work;
```

### Bindings and clients

- Python: `heapdb.Heaps(persist_to=None | "native:/path" | "postgres://…")` with the operations
  above; results are `Ranked` objects.
- TypeScript (`@heapdb/client`): `heaps(target)` where `target` is a server connection
  (`connect(url)`) or a Postgres client with a `query(text, params)` method; same methods.

## Not in this change

Leases (pop with a visibility timeout and heartbeat), change streams, and cross-heap merges.
Applications keep ownership in their own store; `pop` is enough for simple queues.
