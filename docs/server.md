# HeapDB server

`heapdb-server` is the HeapDB data plane: one Rust binary (crate
[`crates/heapdb-server`](../crates/heapdb-server)) that hosts many HeapDB
databases behind an HTTP/JSON API. It powers the hosted service at
`api.heapdb.com` and runs self-hosted as a single node on local disk.

* **Data API** (`/v1/databases/{id}/...`): what clients call. Every SDK takes a
  connection URL `heapdb://<api_key>@api.heapdb.com/<database_id>`.
* **Admin API** (`/admin/v1/...`): creates databases, sets plans and limits,
  manages API keys, reports usage. Called by a control plane (or by you, when
  self-hosting) with a shared admin token.

```python
import heapdb
db = heapdb.connect("heapdb://hdb_live_...@api.heapdb.com/db_7Kq2...")  # or HEAPDB_URL
db.register_effect("bloating", heapdb.LogNormalKernel(median=2 * 3600, sigma=0.6))
db.insert("me", t, "meal", {"beans"})
db.top_causes("me", "bloating", at=now, k=3, explain=True)
```

Contents: [Identifiers and credentials](#identifiers-and-credentials) ·
[Data API](#data-api) · [Errors](#errors) · [Quotas and metering](#quotas-and-metering) ·
[Admin API](#admin-api) · [Architecture](#architecture) ·
[Self-hosting](#self-hosting) · [Configuration](#configuration) ·
[Operations](#operations) · [Clients](#clients)

## Identifiers and credentials

| Thing | Format |
| --- | --- |
| Database id | `db_` + 20 base62 characters |
| Org id | `org_` + 1-64 of `[A-Za-z0-9_-]` |
| Key id | `key_` + 1-64 of `[A-Za-z0-9_-]` |
| API key | `hdb_live_` + 43 base62 characters (256 bits); shown once |

The server never sees or stores API keys in clear: the Admin API registers
`sha256(key)` (lowercase hex) with the key's scopes and optional expiry. A
request's `Authorization: Bearer <api_key>` is hashed and looked up.

**Scopes.** `read` (queries, reports, heap reads), `write` (inserts and
learning: events, feedback, observe, settle; heap item changes and pops),
`admin` (schema calls: effects, event types, rules, `forget_entity`; creating,
dropping and rebasing heaps). Scopes are independent; a full-access key has all three.
`GET /v1/databases/{id}` works with any valid key of the database.

| Route | Scope |
| --- | --- |
| `GET /v1/databases/{id}` | any |
| `POST /effects`, `POST /event-types`, `POST /rules`, `DELETE /entities/{entity}` | `admin` |
| `POST /events`, `POST /feedback`, `POST /observe`, `POST /settle` | `write` |
| `POST /query/top-causes`, `POST /query/risk`, `GET /links`, `GET /calibration` | `read` |
| `PUT /heaps/{heap}`, `DELETE /heaps/{heap}`, `POST /heaps/{heap}/rebase` | `admin` |
| `POST /heaps/{heap}/push`, `/base`, `/attrs`, `/pop`, `DELETE /heaps/{heap}/items/{item}` | `write` |
| `GET /heaps`, `GET /heaps/{heap}`, `POST /heaps/{heap}/top`, `GET /heaps/{heap}/items/{item}` | `read` |
| `POST /hql` | `read` for TOP / RISK / LINKS / CALIBRATION / PEEK / SHOW HEAPS, `write` for INSERT / FEEDBACK / OBSERVE / SETTLE / PUSH / SET / POP / REMOVE, `admin` for CREATE HEAP / DROP HEAP / REBASE |

The admin token (`HEAPDB_ADMIN_TOKEN`) is compared in constant time. Expose
`/admin/*` only to the control plane's network (security group / listener rule).

## Data API

Base `https://api.heapdb.com/v1/databases/{database_id}`. JSON request and
response bodies, `Authorization: Bearer <api_key>`. Times are Unix seconds
(floats). Field names are snake_case and stable; new fields may be added.
The machine-readable description is served at `GET /v1/openapi.json` (OpenAPI 3.1).

Conventions:

* Every response carries `x-request-id` (the client's, if it sent a sane one).
* Errors are always `{"error": {"code": "...", "message": "..."}}` (see [Errors](#errors)).
* Non-finite numbers (e.g. `leak: Infinity`) are the strings `"Infinity"`,
  `"-Infinity"`, `"NaN"`; numeric request fields that may be non-finite accept them.
* CORS: `/v1/*` answers preflights with `Access-Control-Allow-Origin: *`,
  allows `Authorization, Content-Type, Accept, User-Agent, X-Request-Id` and
  exposes `x-request-id, retry-after` (auth is a bearer key, no cookies).
* Results with a human-readable form also carry `text` (the engine's
  `Display`, as printed by the Python library).
* Request bodies reject unknown fields (`400 bad_request`). Maximum body size 8 MiB.

### Objects

**Event**: `{id, entity, time, type, features, weight, attrs}`. `features` is
sorted and unique; `weight` is the dose (> 0).

**Explanation**: `{dt, kernel_value, weight, link_strength, features:[{feature,
strength, alpha, beta}], warnings:[str], rules:[{name, multiplier}],
joint_causes:[{features, name, strength, joint: BetaCounts|null, inseparable}],
entity, prospective, rule_multiplier, score, inseparable, text}` where
`score = kernel_value x weight x link_strength x rule_multiplier`.

**BetaCounts**: `{alpha, beta, strength, prior_alpha, prior_beta, pos, neg, global_pos, global_neg}`.

### `GET /v1/databases/{id}`

Usage and limits (not counted, never rate limited, works when the monthly
quota is used up):

```json
{"database_id": "db_...", "org_id": "org_...", "plan": "free", "suspended": false,
 "usage": {"requests_this_period": 1234, "period_start": 1790812800, "period_end": 1793491200,
           "events_stored": 5000, "storage_bytes": 1048576},
 "limits": {"max_events": null, "max_requests_per_minute": 600, "max_storage_bytes": 104857600,
            "max_requests_per_month": 100000, "billing_period_start": 1790812800}}
```

`null` = unlimited. A trailing slash (`/v1/databases/{id}/`) is accepted.

### `POST /effects` (admin)

```json
{"name": "bloating",
 "kernel": {"type": "lognormal", "params": {"median": 7200, "sigma": 0.6}},
 "priors": {"beans": [3, 1]}, "default_prior": [1, 4], "cause_types": ["meal"],
 "credit": "proportional", "global_pseudocount": 5, "hierarchical": true,
 "leak": 0.05, "w_neg": 1, "repeatable": true}
```

→ `{"ok": true}`. Kernel `type`: `exponential` (alias `exp`; params
`half_life`, `horizon_half_lives`), `window` (`min_lag`, `max_lag`),
`lognormal` (`median`, `sigma`, `tail_sigmas`), `gamma` (`shape`, `scale`,
`tail_stds`). `credit`: `proportional` | `uniform`. Only `name` and `kernel`
are required. Re-registering an effect replaces it (and drops its rules), as
in the library.

### `POST /event-types` (admin)

`{"type": "deploy", "weight": {"kind": "log1p", "attr": "lines_changed"}}` → `{"ok": true}`.

Weight functions are declarative (no code is uploaded):

| `kind` | Weight | Python equivalent |
| --- | --- | --- |
| `attr` | `attrs[attr]` | `lambda a: a.get(attr, value)` |
| `log1p` | `ln(1 + attrs[attr])` | `lambda a: math.log1p(a[attr])` |
| `const` | `value` | `lambda a: value` |

For `attr`/`log1p`, the optional `value` is used when the attribute is
missing; without it, inserting such an event fails with `400`. `weight` may be
omitted (the type is just declared). An explicit event `weight` always wins.

### `POST /rules` (admin)

```json
{"effect": "fraud", "name": "small transaction", "multiplier": 0.3,
 "predicate": {"attr": "amount", "op": "lt", "value": 10, "default": 0}}
```

→ `{"ok": true}`. `multiplier` in `[0, 1]`; optional `types` restricts the rule
to event types. Without `predicate` the rule matches every event (of `types`):
a type-level rule, which also tightens the pruning bounds. Predicate `op`:
`eq`, `ne`, `lt`, `lte`, `gt`, `gte` (also `==`, `!=`, `<`, `<=`, `>`, `>=`, `le`,
`ge`), `in` / `not_in` (value is a list), `contains` (string or list
attribute), `exists`, `missing`. A missing (or null) attribute takes `default`
if given; otherwise only `ne`, `not_in` and `missing` match. Numbers compare
numerically, strings lexicographically, mixed types never match.

### `POST /events` (write)

```json
{"events": [{"entity": "me", "time": 1760000000, "type": "meal",
             "features": ["beans", "onion"], "weight": 1.5, "attrs": {"portion": 2},
             "external_id": "meal-8812"}],
 "return_events": false}
```

→ `{"ids": [17]}`, same order and length as `events` (at most 1000, else
`413 batch_too_large`). The batch is atomic: it is validated completely before
anything is stored. With `return_events: true` the stored events (with derived
weights) are returned too.

**Idempotency.** An event with an `external_id` that is already stored is not
inserted again: its existing id is returned (also within one batch). Retrying
a request whose outcome is unknown (timeout, `503`) is therefore safe for
events that carry an `external_id`. The mapping is dropped when the event is
removed by `forget_entity`.

### `POST /query/top-causes` (read)

`{"entity": "me" | "entities": ["a", "b"], "effect": "bloating", "at": 1760007200, "k": 3, "within": 43200, "explain": true}`

→ `{"entity", "entities", "effect", "at", "causes": [{"rank", "event", "score", "explanation"?}], "stats": {"segments_in_window", "segments_scanned", "segments_pruned", "events_scanned", "union_fallbacks"}, "text"}`.
`k` defaults to 3. `entities` runs `top_causes_group`.

### `POST /query/risk` (read)

Single candidate (`feature_risk`):
`{"effect", "features": [...], "weight"?, "entity"?, "type"?, "attrs"?, "explain"?}` →
`{"effect", "entity", "features", "type", "score", "explanation"?}`. Without
`weight`, the weight comes from the type's weight function, else 1.

Several candidates (`rank_candidates`):
`{"effect", "candidates": [{"id", "features", "weight"?, "type"?, "attrs"?, "entity"?}], "k": 10, "entity"?, "explain"?}` →
`{"candidates": [{"rank", "index", "candidate": {...echo...}, "score", "entity", "explanation"?}]}`.

### `POST /feedback` (write)

`{"entity", "effect", "at", "confirmed_event_id"?, "rejected_ids"?: [...], "credit"?: "proportional"|"uniform"}`
→ `{"confirmed", "rejected", "credit": {feature: share}}`.

### `POST /observe` (write)

`{"entity", "effect", "at", "leak"?}` → `{"entity", "effect", "at", "leak", "leak_responsibility", "total_credit", "credits": [{"event_id", "score", "responsibility", "features": {...}}], "text"}`.

### `POST /settle` (write)

`{"until", "effects"?: str | [str], "entity"?, "partial"?: false}` →
`{"until", "events_settled", "events_rejected", "rejected_weight", "skipped_credited", "skipped_observed", "per_effect": {...}, "text"}`.

### `GET /links?effect=&entity=&min_support=&level=&baseline=&features=` (read)

`baseline`: `mean` (default), `prior` or a number; `features`: comma-separated.
→ `{"effect", "entity", "baseline", "level", "rows": [{"feature", "strength", "alpha", "beta", "n", "lo", "hi", "lift"}], "text"}`.
Rows are associations under the model, not causal effects.

### `GET /calibration?effect=&bins=` (read)

→ `{"effect", "n", "brier", "bins": [{"lo", "hi", "count", "mean_predicted", "observed_rate"}], "text"}`.

### `DELETE /entities/{entity}?scrub_global=false` (admin)

`entity` is percent-encoded. → `{"counts": {"events", "segments", "links", "calibration_records"}}`.

### Keyed heaps

Named priority queues of keyed items ordered by forward decay (semantics,
filters and numerics: [heaps.md](heaps.md)). `{heap}` and `{item}` are one
percent-encoded path segment each (`acme%2Fcheckout`); names and ids are
1-200 UTF-8 bytes. Times default to the server clock. Heap requests are
admitted and counted like every other request; writes check the storage
quota.

| Method & path | Scope | Body → response |
| --- | --- | --- |
| `PUT /heaps/{heap}` | admin | `{half_life, landmark?, meta?}` → HeapInfo. Idempotent when the spec matches (an omitted `landmark` / `meta` matches anything); `409 heap_conflict` otherwise |
| `GET /heaps` | read | `{heaps: [HeapInfo]}` |
| `GET /heaps/{heap}` | read | HeapInfo, or `404 unknown_heap` |
| `DELETE /heaps/{heap}` | admin | `{dropped}` |
| `POST /heaps/{heap}/push` | write | `{items: [{id, weight? (1), time? (now), attrs?}]}` → `{items: [HeapItem]}` |
| `POST /heaps/{heap}/base` | write | `{items: [{id, base}]}` → `{items: [HeapItem]}` |
| `POST /heaps/{heap}/attrs` | write | `{items: [{id, attrs, merge? (false)}]}` → `{items: [HeapItem]}` |
| `DELETE /heaps/{heap}/items/{item}` | write | `{removed}` |
| `GET /heaps/{heap}/items/{item}?now=` | read | Ranked (`rank` 0), or `404 unknown_item` |
| `POST /heaps/{heap}/top` | read | `{k? (10), now?, filter?}` → `{items: [Ranked]}` |
| `POST /heaps/{heap}/pop` | write | `{now?, filter?}` (body optional) → `{item: Ranked \| null}` |
| `POST /heaps/{heap}/rebase` | admin | `{now?, half_life?}` (body optional) → HeapInfo |

**HeapInfo** `{name, half_life, landmark, meta, size}`; **HeapItem** `{id,
base, activity, key, attrs, version}` (`activity` / `key` are `null` until
the first push); **Ranked** `{rank, id, key, score, heat, base, attrs,
version}` (`score` / `heat` may be `"Infinity"` for a push far in the
future).

Batches (`push`, `base`, `attrs`) hold 1-1000 items (`413 batch_too_large`
above), are applied in order (repeated ids accumulate) and are atomic: an
invalid item fails the whole request with `400 invalid_argument` naming it
(`items[3]: weight must be finite and > 0`) and nothing is stored. Bad
filters are `400 invalid_argument` naming the node (`filter.all[1].not:
unknown op 'like' ...`). Other errors: `404 unknown_heap`, `404
unknown_item`.

Heap mutations are WAL records like every other write (a pop is logged
with its clock and filter, so replay picks the same item), and heaps are
part of snapshots.

### `POST /hql`

`{"query": "TOP 3 CAUSES OF bloating FOR me AT now-1h EXPLAIN", "now"?: 1760000000}` →
`{"kind": "top_causes"|"calibration"|"risk"|"links"|"observe"|"settle"|"message"|"heap_items"|"heap"|"heaps", "result": ..., "text"}`.
One statement of the [query language](../README.md#query-language), including
heap statements: `PEEK` / `POP` give `heap_items` (`{heap, now, popped,
items: [Ranked]}`), `CREATE HEAP` / `REBASE` give `heap` (HeapInfo), `SHOW
HEAPS` gives `heaps`. `now` defaults to the server clock.

## Errors

| Status | `code` | Meaning |
| --- | --- | --- |
| 400 | `bad_request` | malformed JSON, unknown field, bad parameter |
| 400 | `invalid_argument` | rejected by the engine (bad kernel, weight <= 0, multiplier > 1, ...) |
| 400 | `query_error` | `/hql` statement does not parse |
| 401 | `unauthorized` | no bearer token (or bad admin token) |
| 401 | `invalid_api_key` / `key_expired` | unknown or revoked key / expired key |
| 403 | `insufficient_scope` | key lacks the route's scope |
| 403 | `key_not_for_database` | key belongs to another database |
| 403 | `database_suspended` | suspended (e.g. non-payment) |
| 403 | `admin_disabled` | `HEAPDB_ADMIN_TOKEN` not set |
| 404 | `database_not_found`, `unknown_effect`, `unknown_event`, `unknown_heap`, `unknown_item`, `not_found` | |
| 405 | `method_not_allowed` | |
| 409 | `database_exists`, `key_exists` | Admin API conflicts |
| 409 | `heap_conflict` | `PUT /heaps/{heap}` with a spec that differs from the existing heap |
| 413 | `payload_too_large`, `batch_too_large` | body > 8 MiB / > 1000 events or heap items |
| 413 | `events_quota_exceeded`, `storage_quota_exceeded` | quota reached; not retryable |
| 429 | `rate_limited` | per-minute limit; `Retry-After` |
| 429 | `monthly_quota_exceeded` | monthly quota; `Retry-After` = seconds to the period end; do not retry earlier |
| 503 | `lease_lost`, `database_moving`, `database_unavailable`, `not_owner`, `owner_unreachable`, `storage_unavailable`, `shutting_down` | transient; `Retry-After: 1` |
| 500 | `internal` | bug |

A `503` on a write means the outcome is unknown (the write was not
acknowledged, but may have been applied). Retry with `external_id`s for
exactly-once inserts.

## Quotas and metering

Per database, enforced on the node that owns it, in this order:

1. **Suspension**: `403 database_suspended`.
2. **Monthly request quota** (`max_requests_per_month`; `null` = unlimited).
   Data API requests are counted per billing period: calendar months from
   `billing_period_start` (Unix seconds; default the first day of the current
   UTC month; day-of-month clamps, e.g. Jan 31 → Feb 28). At the limit, every
   Data API request except `GET /v1/databases/{id}` returns
   `429 monthly_quota_exceeded` with `Retry-After` = seconds until the period
   ends. The count resets when the period rolls over. Changing
   `billing_period_start` starts counting the period it defines.
3. **Rate limit** (`max_requests_per_minute`): a token bucket of that size,
   refilled continuously; `429 rate_limited` with `Retry-After`.
4. **Storage / events** (`max_storage_bytes`, `max_events`) on writes:
   `413 storage_quota_exceeded` / `413 events_quota_exceeded`.

Counted requests: every admitted Data API request except the info endpoint
(requests rejected by auth, suspension or quotas are not counted). A request
forwarded between nodes is counted once, by the owner. Counters are flushed to
the usage store every second (and when a database closes) as atomic
additions, so they survive failover; a crash can lose at most the last flush
interval of request counts, never acknowledged writes.

## Admin API

Base `/admin/v1`, `Authorization: Bearer <HEAPDB_ADMIN_TOKEN>`.

| Method & path | Body → response |
| --- | --- |
| `POST /databases` | `{database_id, org_id, plan?, limits?}` → `201` record; `200` with the existing record if it exists in the same org (idempotent); `409 database_exists` otherwise |
| `GET /databases/{id}` | → record |
| `PATCH /databases/{id}` | `{plan?, limits?, suspended?}` → record. `limits` is merged: given keys replace, `null` removes (= unlimited) |
| `DELETE /databases/{id}` | → `{"deleted": bool}`. Keys, registry entry, open state and all stored data (WAL, snapshots) are removed; object-store versions expire per your retention |
| `PUT /databases/{id}/keys/{key_id}` | `{sha256, scopes, expires_at?}` → key record. Replaces a key with the same id |
| `DELETE /databases/{id}/keys/{key_id}` | → `{"revoked": bool}`, effective immediately on every node |
| `GET /usage?since=<unix seconds>` | → `{since, generated_at, granularity_seconds: 3600, databases: [{database_id, requests, events_written, events_stored, storage_bytes, requests_this_period, period_start, period_end}]}` |
| `GET /health` | → `{status, version, node_id, node_addr, cluster, open_databases}` |

Record: `{database_id, org_id, plan, limits, suspended, created_at, key_ids}`.
`limits`: `{max_events, max_requests_per_minute, max_storage_bytes,
max_requests_per_month, billing_period_start, max_databases_per_org}` (the
last is stored but only meaningful to a control plane).

Usage: `requests` and `events_written` are sums of hourly buckets starting at
or after `since` (rounded down to the hour); `events_stored` and
`storage_bytes` are current gauges; `requests_this_period` is the current
billing period's count.

## Architecture

### One database, one writer

Each database is open on at most one node at a time, its **owner**, which
holds a **lease**: `{database_id, owner_node, owner_addr, epoch, expires_at}`.
Leases last 30 s and are renewed every 10 s with a write conditional on the
`(owner, epoch)`. `epoch` is a monotonic fencing token. A node receiving a
request for database `d`:

1. owns `d` → serves it;
2. a live lease of another node → **forwards** the request to `owner_addr`
   (the internal listener) with `X-HeapDB-Forwarded: 1`; a forwarded request
   is never forwarded again (`503 not_owner` instead);
3. no live lease → acquires it (conditional put, `epoch + 1`; exactly one
   node wins a race, the others forward), recovers `d` from storage and serves it.

Every node authenticates every request (registry lookups are cached for at
most 30 s); admin changes invalidate the local cache and the owner's cache, so
revocation and suspension are immediate.

Single-node mode is the same machinery with local stores: one node always
wins the lease.

### Write path and durability

A mutation is applied to the engine under the database's write lock, given
the next sequence number `seq`, and handed to the database's WAL writer (in
lock order). The writer **group-commits** records arriving within a short
window (5 ms, at most 10 ms) into one object
`db/<id>/wal/<epoch>-<last seq>.jsonl` and, for each group:

1. checks the lease locally (more than the safety margin left);
2. writes the object (put-if-absent; on S3 `If-None-Match: *`), which returns
   only once it is durable (single node: file fsync + directory fsync);
3. re-reads the lease (strongly consistent) and requires the same owner and epoch;
4. only then acknowledges the requests.

Reads wait until everything they observed is durable, so a client never sees
state a crash could take back. If any step fails, pending writes get a
retryable `503` and the database is dropped from memory; the next request
recovers it from storage under a new epoch.

**Snapshots** are written every 5 minutes (if anything changed) or every
10,000 WAL records, and when a database closes: `db/<id>/snap/<epoch>-<seq>.jsonl`
= one header line (`{"kind":"heapdb_server", "format":2, "database_id",
"epoch", "seq", "schema", "external_ids", "heaps"}`: declarative weight
functions and rules, the idempotency index, the number of heap records),
then that many keyed-heap records (`{"kind":"heap",...}` / `{"kind":"item",...}`),
then the engine's JSONL snapshot (format v3, as `HeapDB.save`). Format 1
snapshots (0.5, no heap section) still load. Afterwards the WAL objects it covers and all but the
two newest snapshots are deleted.

**Recovery** loads the newest valid snapshot and replays WAL records with a
higher `seq` in order. Each new owner first writes an `epoch_start` record;
epoch `e` starts at the seq of that record. A record or snapshot `(e, s)` is
**fenced** if some later epoch `e' > e` started at `s' <= s`: a write that a
stale owner managed to put after losing its lease is ignored by every later
recovery (and was never acknowledged, because step 3 failed).

A node that loses its lease (renewal rejected, or no renewal before the
margin) stops serving the database immediately; its in-flight writes fail
with `503 lease_lost`.

### Memory

Databases idle for 10 minutes are closed (final snapshot, lease released).
Above `HEAPDB_MEMORY_BUDGET_MB` (estimated as 3x the serialized size) the
least recently used databases are closed first.

### Storage backends

| | Single node (`HEAPDB_CLUSTER=off`) | Cluster (`HEAPDB_CLUSTER=aws`) |
| --- | --- | --- |
| WAL + snapshots | `<data-dir>/objects/db/<id>/...` (fsync) | S3 `s3://<bucket>/<prefix>db/<id>/...` |
| Leases | `<data-dir>/leases.json` | DynamoDB `heapdb-leases` |
| Registry | `<data-dir>/registry.json` | DynamoDB `heapdb-registry` |
| Usage | `<data-dir>/usage.json` | DynamoDB `heapdb-usage` |

DynamoDB tables (on-demand; enable PITR):

* `heapdb-leases`: partition key `database_id` (S); `owner_node` (S),
  `owner_addr` (S), `epoch` (N), `expires_at` (N, Unix ms).
* `heapdb-registry`: partition key `pk` (S): `db#<id>` and `key#<sha256>`
  items with a JSON `record` (S).
* `heapdb-usage`: partition key `database_id` (S), sort key `bucket` (S):
  `h#<hour start>` and `p#<period start>` counters (`requests`,
  `events_written`, updated with atomic `ADD`) and `gauge` (`events_stored`,
  `storage_bytes`).

Enable S3 versioning and a lifecycle rule for backups (e.g. 7 days free, 30
days paid). Node clocks must be roughly in sync (lease expiry is wall-clock).

The crate's storage traits (`ObjectStore`, `LeaseStore`, `Registry`,
`UsageStore`) also have in-memory implementations with the same conditional
semantics; the cluster tests run on them and, optionally, on an AWS emulator.

## Self-hosting

### Binary

```sh
cargo build --release -p heapdb-server            # add --features cluster-aws for S3/DynamoDB
export HEAPDB_ADMIN_TOKEN=$(openssl rand -hex 32)
./target/release/heapdb-server --data-dir ./data  # Data + Admin API on :8080, internal on :9090
```

### Docker

```sh
docker run -d --name heapdb -p 8080:8080 -v heapdb-data:/data \
  -e HEAPDB_ADMIN_TOKEN=change-me ghcr.io/heapdb/heapdb-server:latest
```

Images: `ghcr.io/heapdb/heapdb-server:<version>` (tags `server-v<version>`),
`:latest`, and `:edge` (main), for `linux/amd64` and `linux/arm64`. The image
runs as a non-root user, stores data in `/data` and has a `HEALTHCHECK`.
Keep port 9090 private.

### Create a database and a key

```sh
heapdb-server keygen      # prints api_key and its sha256 (keep the key, register the hash)
heapdb-server new-id      # prints a database id
A="Authorization: Bearer $HEAPDB_ADMIN_TOKEN"
curl -X POST localhost:8080/admin/v1/databases -H "$A" -H 'Content-Type: application/json' \
  -d '{"database_id": "db_...", "org_id": "org_me", "plan": "self-hosted"}'
curl -X PUT localhost:8080/admin/v1/databases/db_.../keys/key_1 -H "$A" -H 'Content-Type: application/json' \
  -d '{"sha256": "<sha256>", "scopes": ["read", "write", "admin"]}'
```

Connect with `heapdb+http://<api_key>@localhost:8080/<database_id>` (plain
HTTP) or put TLS in front (e.g. Caddy) and use `heapdb://...`.

### A cluster

Run N nodes behind a load balancer with `HEAPDB_CLUSTER=aws`, an S3 bucket,
the three DynamoDB tables (`heapdb-server aws-init` creates them, mainly for
development) and each node's private `HEAPDB_NODE_ADDR=ip:9090` (read from
ECS task metadata when unset). The internal port must be reachable between
nodes only. Health checks: `/readyz`.

## Configuration

| Variable (flag) | Default | |
| --- | --- | --- |
| `HEAPDB_LISTEN` (`--listen`) | `0.0.0.0:8080` | public listener |
| `HEAPDB_INTERNAL_LISTEN` (`--internal-listen`) | `0.0.0.0:9090` | forwarding, `/metrics` |
| `HEAPDB_CLUSTER` (`--cluster`) | `off` | `off` or `aws` |
| `HEAPDB_NODE_ADDR` (`--node-addr`) | ECS metadata | private `ip:port` of the internal listener |
| `HEAPDB_DATA_DIR` (`--data-dir`) | `./data` | single-node storage |
| `HEAPDB_ADMIN_TOKEN` | unset | Admin API disabled when unset |
| `HEAPDB_MEMORY_BUDGET_MB` | `1024` | `0` = unlimited |
| `HEAPDB_S3_BUCKET`, `HEAPDB_S3_PREFIX` | | cluster |
| `HEAPDB_DDB_LEASES`, `HEAPDB_DDB_REGISTRY`, `HEAPDB_DDB_USAGE` | `heapdb-leases`, `heapdb-registry`, `heapdb-usage` | cluster |
| `AWS_REGION`, `AWS_ENDPOINT_URL`, credentials | standard AWS | cluster |
| `HEAPDB_S3_FORCE_PATH_STYLE` | `0` | `1` for S3-compatible emulators |
| `HEAPDB_LOG` / `HEAPDB_LOG_FORMAT` | `info` / `json` | tracing filter / `json` or `text` |

Tuning (mostly for tests): `HEAPDB_LEASE_TTL_MS` (30000),
`HEAPDB_LEASE_RENEW_MS` (10000), `HEAPDB_GROUP_COMMIT_MS` (5, max 10),
`HEAPDB_SNAPSHOT_INTERVAL_MS` (300000), `HEAPDB_SNAPSHOT_RECORDS` (10000),
`HEAPDB_IDLE_CLOSE_MS` (600000), `HEAPDB_AUTH_CACHE_TTL_MS` (30000, max),
`HEAPDB_USAGE_FLUSH_MS` (1000), `HEAPDB_MAX_BODY_BYTES` (8 MiB),
`HEAPDB_FSYNC` (1).

## Operations

* `GET /healthz`: process up. `GET /readyz`: storage, leases and registry
  reachable (503 otherwise). Both on both listeners.
* `GET /metrics` (internal listener only): Prometheus text: responses by
  surface and status, WAL commits/records/failures, snapshots, forwards,
  lease acquisitions/losses, opens/closes/evictions, open databases and bytes.
* Logs: one JSON line per request (`request_id`, method, path, status,
  latency) plus lifecycle events (lease acquired/lost, snapshot written, ...).
* Graceful shutdown (SIGTERM/SIGINT): stop accepting, finish in-flight
  requests, flush the WAL, snapshot open databases, release their leases, so
  another node takes over at once. A crash is safe too: the lease expires
  after at most 30 s and the next owner recovers every acknowledged write.

## Clients

* Python: `heapdb.connect(url_or_env=None, *, timeout=30, retries=3)` in the
  `heapdb` package (pure standard library) returns a `RemoteHeapDB` with the
  methods and result attributes of `heapdb.HeapDB`. Weight functions and
  predicates are declarative: `heapdb.remote.weight_attr("portion", 1.0)`,
  `weight_log1p("lines_changed")`, `weight_const(2)`,
  `where("amount", "<", 10, default=0)`. Errors are `heapdb.HeapDBError`
  subclasses with `.code`, `.status`, `.retry_after`. `db.heaps` is the
  keyed-heap API of `heapdb.Heaps` over the heap routes.
* TypeScript: `@heapdb/client` (`connect(process.env.HEAPDB_URL!)`;
  `heaps(db)` for keyed heaps).

Both retry `429 rate_limited` and `503` with backoff honouring `Retry-After`,
and never retry quota errors.
