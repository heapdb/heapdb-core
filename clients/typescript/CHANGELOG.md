# Changelog

All notable changes to `@heapdb/client` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package uses
[Semantic Versioning](https://semver.org/).

## [0.7.0] - Unreleased

The client's version now follows HeapDB's (0.7.0).

### Added

- **Keyed heaps** (`docs/heaps.md`): `heaps(target)` returns the same API for two
  transports: a server connection (`connect(url)`, the new `/heaps` routes) or any
  Postgres client with `query(text, params)` (node-postgres `Client` / `Pool`),
  which calls the SQL functions of schema `heapdb` directly. Methods:
  `createHeap`, `dropHeap`, `heap`, `listHeaps`, `push`, `pushMany` (atomic),
  `setBase`, `setAttrs`, `remove`, `get`, `topK`, `pop`, `rebase`; typed results
  `HeapInfo`, `HeapItem`, `Ranked` and typed `HeapFilter` trees.
- `HEAPDB_SQL` / `HEAPDB_SQL_VERSION` (generated at build time from
  `sql/postgres/heapdb.sql`) and `installHeapSql(client)`.
- No new runtime dependencies (`pg` is a devDependency for the tests).

## [0.1.0] - Unreleased

First preview of the TypeScript client for the HeapDB Data API.

### Added

- `connect(url?, opts?)`: parses `heapdb://<api_key>@<host>/<database_id>` (or
  `HEAPDB_URL`), with `timeoutMs`, `retries`, `retryBaseDelayMs`, `retryMaxDelayMs`,
  `userAgent`, `baseUrl` (self-hosted), `fetch` and `headers` options.
- Methods for every Data API endpoint: `registerEffect`, `registerEventType`,
  `addRule`, `insert`, `insertMany` (auto-batched to 1000 events per request),
  `topCauses` (single entity or group), `featureRisk`, `rankCandidates`, `feedback`,
  `observe`, `settle`, `linkReport`, `calibrationReport`, `forgetEntity`, `hql`, `info`.
- Typed requests and responses (camelCase), `kernels` helpers for `{type, params}` specs.
- `HeapDBError` with `code`, `status`, `retryAfter`, `requestId`, `insertedIds`.
- Retries with exponential backoff and full jitter on 429 / 503 / network errors and
  timeouts, honouring `Retry-After`; `monthly_quota_exceeded` is never retried.
- Per-call `AbortSignal` and timeout.
- API key redaction in errors, `toString()`, `JSON.stringify` and `util.inspect`.
- ESM + CommonJS builds, zero runtime dependencies, Node >= 18 / Bun / Deno / edge / browsers.
