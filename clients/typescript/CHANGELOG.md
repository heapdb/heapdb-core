# Changelog

All notable changes to `@heapdb/client` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the package uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First preview of the TypeScript client for the HeapDB Data API.

### Added

- `registerEffect(..., { combine: "logistic", learningRate, l2 })`: HeapDB's
  logistic combine mode. Logistic explanations decode `combine`, `baseLogOdds`,
  `baseOdds`, `logOdds`, `odds` and per-feature `logOdds` / `oddsMultiplier`;
  link-report rows decode `logOdds`. All optional, absent for noisy-OR effects.
- Catalog: `effects({limit, cursor})`, `entities({prefix, limit, cursor})`,
  `eventTypes({features, limit, cursor})` and `occurrences({effect, entity, limit, cursor})`
  (server `GET /effects`, `/entities`, `/event-types`, `/occurrences`; read scope). Each
  returns one page with `nextCursor` (`null` on the last page); effects include
  `combine` and, for logistic effects, `logistic: {learningRate, l2, baseLogOdds}`.
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
