# @heapdb/client

TypeScript client for the [HeapDB](https://github.com/heapdb/heapdb-core) Data API:
hosted HeapDB Cloud or a self-hosted `heapdb-server`.

> **Preview (0.1.x).** The hosted Data API is new; request and response shapes may
> still change before 1.0.

- Zero runtime dependencies. Uses the global `fetch` (Node >= 18, Bun, Deno,
  Cloudflare Workers / Vercel Edge, browsers), or one you inject.
- ESM and CommonJS builds with full type definitions.
- camelCase TypeScript API, snake_case on the wire.
- Retries with exponential backoff and jitter on 429 / 503 / network errors,
  honouring `Retry-After`. The monthly quota error is never retried.
- The API key is never logged and is redacted from errors and `toString()`.

## Install

```bash
npm install @heapdb/client
# or: pnpm add @heapdb/client / yarn add @heapdb/client / bun add @heapdb/client
```

Deno: `import { connect } from "npm:@heapdb/client";`

## Connect

Create a database and an API key in the console. It gives you a connection URL:

```
heapdb://<api_key>@api.heapdb.com/<database_id>
```

```ts
import { connect } from "@heapdb/client";

const db = connect(process.env.HEAPDB_URL!); // or connect() to read HEAPDB_URL itself
```

`connect()` makes no request; the first call does. Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `timeoutMs` | `30000` | Per-attempt timeout (0 disables). |
| `retries` | `3` | Retries after the first attempt on 429, 503, network errors and timeouts. |
| `retryBaseDelayMs` | `250` | First backoff; doubles per retry, full jitter. |
| `retryMaxDelayMs` | `30000` | Cap per wait. A `Retry-After` longer than this is not waited for: the error is thrown. |
| `userAgent` | | Appended to `heapdb-ts/<version>`. |
| `baseUrl` | `https://<host>` | Server origin for self-hosted servers, e.g. `http://localhost:8080`. |
| `fetch` | global `fetch` | Custom fetch (proxies, tracing, tests). |
| `headers` | | Extra headers on every request. |

Every method also takes `{ signal, timeoutMs }` (an `AbortSignal` cancels the call
and any pending retry).

## Quickstart

The same flow as the Python quickstart, against a hosted database. Times are Unix
seconds (float) or `Date` objects.

```ts
import { connect, kernels } from "@heapdb/client";

const H = 3600;
const db = connect(); // HEAPDB_URL

await db.registerEffect("bloating", {
  kernel: kernels.lognormal(2 * H, { sigma: 0.6 }), // peaks ~1.4h, long tail
  priors: { beans: [3, 1], dairy: [2, 1] },          // Beta(alpha, beta)
});

await db.insert({ entity: "me", time: 8 * H, type: "meal", features: ["oats", "dairy"] });
const lunch = await db.insert({
  entity: "me", time: 13 * H, type: "meal", features: ["beans", "onion"],
  weight: 1.5,                                        // weight = dose (portion)
});

const res = await db.topCauses({ entity: "me", effect: "bloating", at: 15 * H, k: 3, explain: true });
for (const c of res.causes) {
  console.log(c.rank, c.event.id, c.event.features, c.score.toFixed(3));
  console.log("   lag", c.explanation?.dt, "kernel", c.explanation?.kernelValue);
}
console.log(res.stats); // { segmentsScanned, segmentsPruned, ... }

// Tell HeapDB what really happened.
await db.feedback({ entity: "me", effect: "bloating", at: 15 * H, confirmedEventId: lunch });
```

Kernels are plain `{type, params}` objects, so you can also write them literally:

```ts
await db.registerEffect("incident", {
  kernel: { type: "exponential", params: { half_life: 3600 } },
  causeTypes: ["pr", "deploy"],
});
```

Helpers: `kernels.exponential(halfLife)`, `kernels.window(minLag, maxLag)`,
`kernels.lognormal(median, { sigma })`, `kernels.gamma(shape, scale)`.

## API

| Method | Endpoint |
| --- | --- |
| `registerEffect(name, { kernel, priors?, defaultPrior?, causeTypes?, credit?, globalPseudocount?, leak? })` | `POST /effects` |
| `registerEventType(type, weight?)` with `{kind: "log1p" \| "attr", attr}` or `{kind: "const", value}` | `POST /event-types` |
| `addRule(effect, name, { multiplier, types?, predicate?: {attr, op, value} })` | `POST /rules` |
| `insert(event)` → id | `POST /events` |
| `insertMany(events, { batchSize? })` → ids (split into requests of <= 1000) | `POST /events` |
| `topCauses({ entity \| entities, effect, at, k?, within?, explain? })` | `POST /query/top-causes` |
| `featureRisk({ effect, features, weight?, entity?, type?, attrs?, explain? })` | `POST /query/risk` |
| `rankCandidates({ effect, candidates, k?, entity?, explain? })` | `POST /query/risk` |
| `feedback({ entity, effect, at, confirmedEventId?, rejectedIds?, credit? })` | `POST /feedback` |
| `observe({ entity, effect, at, leak? })` | `POST /observe` |
| `settle({ until, effects?, entity? })` | `POST /settle` |
| `linkReport({ effect, entity?, minSupport? })` | `GET /links` |
| `calibrationReport({ effect, bins? })` | `GET /calibration` |
| `forgetEntity(entity, { scrubGlobal? })` | `DELETE /entities/{entity}` |
| `hql(query)` | `POST /hql` |
| `info()` → `{ databaseId, plan, usage, limits }` | `GET /` |

Schema calls (`registerEffect`, `registerEventType`, `addRule`, `forgetEntity`)
need a key with the `admin` scope. Weights and rule predicates are declarative:
no code is uploaded.

## Errors

Every failure is a `HeapDBError` with `code`, `status` (0 if no response),
`retryAfter` (seconds, or `null`) and `requestId`:

```ts
import { HeapDBError } from "@heapdb/client";

try {
  await db.insert({ entity: "me", time: Date.now() / 1000, type: "meal" });
} catch (e) {
  if (e instanceof HeapDBError && e.code === "monthly_quota_exceeded") {
    // Free plan limit reached; resets in e.retryAfter seconds. Upgrade in the console.
  }
  throw e;
}
```

Client-side codes: `invalid_url`, `invalid_argument`, `network_error`, `timeout`,
`aborted`, `invalid_response`, `no_fetch`. Server codes pass through unchanged.
If `insertMany` fails part-way, `error.insertedIds` lists the ids of the batches
that were already committed.

Retries re-send the same request. A `POST /events` that timed out may have been
applied; set `externalId` on events if you need to detect duplicates.

## Agents and edge runtimes

The client holds no connection state, so one instance can be shared across
requests, and it runs anywhere `fetch` does.

```ts
// Cloudflare Worker / Vercel Edge: an agent tool that answers "what caused this?"
import { connect } from "@heapdb/client";

export default {
  async fetch(req: Request, env: { HEAPDB_URL: string }) {
    const db = connect(env.HEAPDB_URL, { timeoutMs: 5000, retries: 1 });
    const { service, at } = (await req.json()) as { service: string; at: number };
    const res = await db.topCauses({ entity: service, effect: "incident", at, k: 3, explain: true });
    return Response.json(
      res.causes.map((c) => ({ id: c.event.id, type: c.event.type, score: c.score })),
    );
  },
};
```

Pre-merge review queue (rank open PRs by learned risk):

```ts
const queue = await db.rankCandidates({
  effect: "incident",
  candidates: openPrs.map((pr) => ({
    id: pr.number,
    type: "pr",
    features: pr.labels,
    attrs: { lines_changed: pr.additions + pr.deletions },
  })),
  k: 5,
});
```

In browsers, do not ship a key with `write` or `admin` scope to end users; call
HeapDB from your server or an edge function.

## Self-hosted

Point `baseUrl` at your server; the database path is appended
(`http://localhost:8080/v1/databases/<db>`). The host in the URL is then ignored.

```ts
const db = connect("heapdb://<key>@localhost:8080/my_db", { baseUrl: "http://localhost:8080" });
```

## Development

```bash
cd clients/typescript
npm ci
npm run typecheck && npm run lint && npm test && npm run build
HEAPDB_TEST_URL=heapdb://<key>@localhost:8080/<db> HEAPDB_TEST_BASE_URL=http://localhost:8080 npm test
```

The integration test runs only when `HEAPDB_TEST_URL` is set; use a throwaway database.
Regenerate `package-lock.json` with npm >= 11 (`npx npm@11 install`): npm 10 fails
to resolve vitest's optional peer set. `npm ci` works with either.

## License

Apache-2.0
