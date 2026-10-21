# Keyed heaps and `persist_to`

Status: **implemented** in 0.7.0 (branch `keyed-heaps`). The decisions made
while implementing, where the proposal left room or was changed, are listed
in [Implementation notes](#implementation-notes) at the end.

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
| `create_heap(name, half_life, landmark = now, meta = {})` | idempotent when the spec matches (an omitted landmark / meta matches anything); error otherwise |
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
    fn create_heap(&mut self, spec: &HeapSpec, now: f64) -> Result<HeapInfo>;
    fn drop_heap(&mut self, name: &str) -> Result<bool>;
    fn heap(&self, name: &str) -> Result<Option<HeapInfo>>;
    fn heaps(&self) -> Result<Vec<HeapInfo>>;
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

## Implementation notes

Decisions made while implementing, beyond or different from the proposal
above.

**Engine (`crates/heapdb/src/heaps`).**

- `HeapBackend::create_heap` takes `now` (the landmark of a new heap
  without one: the engine never reads a clock), and the trait has
  `heaps()` to list heaps. `HeapSpec.landmark` and `.meta` are optional; an
  omitted value matches an existing heap, so a repeated `create_heap(name,
  half_life)` is idempotent.
- Numerics are shared by every backend: `logaddexp(a, b) = hi + ln(1 +
  exp(-(hi - lo)))`, skipping the `exp` when `hi - lo > 40` (the term is
  exactly 0 there in `f64`; Postgres would raise an underflow error);
  `exp` of the score is 0 below -745 and infinity above `ln(f64::MAX)`; `-0.0`
  keys are normalised to `0.0` (Postgres compares them equal). Keys and
  scores agree across backends within 1e-9 relative (in practice to the
  last bit or one ulp).
- A push whose `ln w + λ(t − L)` is not finite is rejected ("rebase the
  heap"); a finite key whose score at `now` overflows gives `score =
  Infinity` (JSON: `"Infinity"`).
- `rebase` does not change item versions; every other change bumps them.
  `set_attrs` on a missing item creates it (unranked, base 1), like
  `set_base`.
- `get` returns `rank` 0; unranked items (never pushed) have `key = None`,
  `score = heat = 0`, are returned by `get` and counted in `size`, but never
  listed by `top_k` / `pop`.
- Ids and names: 1-200 UTF-8 bytes and no NUL character (Postgres `text`
  cannot hold one); attrs, meta and filter values must not contain `\u0000`
  (Postgres `jsonb` rejects it).
- Filters: equality compares numbers numerically at any depth; strings are
  ordered byte-wise (Postgres: `COLLATE "C"`); `{"all": []}` is true and
  `{"any": []}` false; a present attribute of the wrong shape (a scalar for
  `contains_all` / `subset_of`) does not match. Filters are validated
  (unknown keys and ops, value shapes, depth ≤ 32, ≤ 1000 nodes) with errors
  naming the node (`filter.all[1].not: unknown op 'like' ...`).
- `MemoryHeaps` keeps a `BTreeSet<(key desc, id)>` index plus a map by id;
  `top_k` walks the index and stops after `k` matches. Batched `push_many`,
  `set_base_many`, `set_attrs_many` are atomic (used by the server's batch
  routes).
- `NativeHeaps`: one `HeapOp` per line in `ops.jsonl` with a sequence
  number, `fsync`ed before returning; `checkpoint.jsonl` every 10,000 ops
  (temp file, fsync, rename, directory fsync), then the log is truncated.
  On open a torn last line (no trailing newline) is cut off; any other bad
  line is an error. `pop` is logged as a `remove` of the chosen id. One
  process per directory (not detected).
- Query language: also `SET ATTRS <id> IN <heap> TO {json} [MERGE]`, `SHOW
  HEAPS`, `CREATE HEAP ... [LANDMARK <time>] [META {json}]`, `PEEK/POP ...
  [AT <time>]` and `REBASE <heap> [AT <time>] [HALF LIFE <d>]`; `!=` and `<>`.
  Heap statements run through `query::execute_with(db, heaps, ...)`; plain
  `execute` reports that it needs a heap backend. The `heapdb` binary takes
  `--heaps memory|native:<dir>`.
- `PersistTo::parse` accepts `memory` (or empty), `native:<dir>`,
  `postgres://` and `postgresql://`; `PersistTo::open` on a Postgres URL
  returns an error pointing to `heapdb_postgres::open`. Displayed URLs hide
  the password.

**Postgres (`sql/postgres/heapdb.sql`, crate `heapdb-postgres`).**

- Function signatures: `create_heap(name, half_life, landmark, meta, now)`,
  `push(heap, id, weight, t, attrs)`, `set_base(heap, id, base)`,
  `set_attrs(heap, id, attrs, merge)`, `remove(heap, id)`, `get(heap, id,
  now)`, `top_k(heap, k, now, filter)`, `pop(heap, now, filter)`,
  `rebase(heap, now, half_life)`, `drop_heap(name)`, plus `heap_info(name)`,
  `list_heaps()`, `matches(attrs, filter)` and `check_filter(filter)`.
  `now` / `t` default to the transaction time. The item-returning functions
  return rows shaped like HeapItem / Ranked / HeapInfo.
- Errors use SQLSTATE `HD400` (invalid argument), `HD404` (unknown heap) and
  `HD409` (conflicting `create_heap`); clients map them to their error types.
- Table and column names are `COLLATE "C"`, so ties by id order the same as
  in Rust. `key` is `GENERATED ALWAYS AS (ln(base) + activity + 0.0) STORED`
  with a partial index on `(heap, key DESC, id) WHERE key IS NOT NULL`.
  Numeric helpers are plpgsql so the planner cannot constant-fold an `exp`
  that would raise.
- The install is idempotent and serialised with an advisory lock; it
  records the version in `heapdb.schema_version`. `heapdb_postgres::open`
  runs it when the installed version differs.
- TLS: rustls (`tokio-postgres-rustls`, `ring`) rather than
  `postgres-native-tls`, so nothing links against OpenSSL and the abi3 wheels
  build the same on Linux, macOS and Windows. `sslmode` follows libpq:
  `prefer` (default) and `require` encrypt without verifying the
  certificate, `verify-ca` / `verify-full` verify against the platform roots
  plus `sslrootcert`.
- Under row-level security, `pop` and `remove` only see the caller's rows;
  writes that violate a policy fail with the policy error.

**Server.** Heap ops are WAL records with every input resolved (default
weight 1 and time `now` are filled in before logging; a pop records its
clock and filter, and replays to the same item). Snapshots carry the heaps
between the header and the engine section (format 2). Extra error codes:
`unknown_item` (404) and `heap_conflict` (409). Heap names in paths are
single percent-encoded segments, so `/` is `%2F`.

**Clients.** Python: `heapdb.Heaps(persist_to)`, filters as dicts or
condition strings (`"kind IN ('bug') AND NOT blocked = TRUE"`, compiled with
the query-language parser), `heapdb.heap_filter()` to normalise one,
`Ranked` / `HeapItem` / `HeapInfo` (`to_dict`, `from_dict`), and
`heapdb.connect(...).heaps` (`RemoteHeaps`, also `push_many` etc.). The
Postgres backend is behind the extension's `postgres` cargo feature, on in
`pyproject.toml` and the release wheels (`heapdb._core.HAS_POSTGRES`).
TypeScript: `heaps(target)` returns the same API for a server connection or
a `query(text, params)` client; `pushMany` over Postgres is one statement
(atomic, works with a pool); `HEAPDB_SQL` is generated from the SQL file at
build time and `typecheck` fails if it is stale.

## Not in this change

Leases (pop with a visibility timeout and heartbeat), change streams, and cross-heap merges.
Applications keep ownership in their own store; `pop` is enough for simple queues.
