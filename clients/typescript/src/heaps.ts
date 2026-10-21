/**
 * Keyed heaps: named priority queues ordered by forward decay (docs/heaps.md).
 *
 * `heaps(target)` works on two transports with the same methods and results:
 *
 * - a server connection (`connect(url)`): the `/heaps` routes of the Data API;
 * - a Postgres client with `query(text, params) => Promise<{ rows }>` (node-postgres
 *   `Client` / `Pool` and compatible drivers): the SQL functions of schema `heapdb`
 *   (install them once with {@link installHeapSql}).
 *
 * Times are Unix seconds (or `Date`); `now` / `t` default to the server's or the
 * database's clock. Filters are the JSON trees of docs/heaps.md.
 */
import { HeapDB, transportOf } from "./client.js";
import { HeapDBError } from "./errors.js";
import type { Transport } from "./http.js";
import { HEAPDB_SQL, HEAPDB_SQL_VERSION } from "./sql.generated.js";
import type { Attrs, AttrValue, RequestOptions, Time } from "./types.js";
import { compact, toSeconds } from "./wire.js";

export { HEAPDB_SQL, HEAPDB_SQL_VERSION };

// -- types --------------------------------------------------------------------

/** Leaf operators of a heap filter. */
export type HeapFilterOp =
  | "eq"
  | "ne"
  | "lt"
  | "lte"
  | "gt"
  | "gte"
  | "in"
  | "not_in"
  | "contains"
  | "contains_all"
  | "subset_of"
  | "exists"
  | "missing";

/** `{all: [...]}`, `{any: [...]}`, `{not: f}` or a leaf `{attr, op, value}`. */
export type HeapFilter =
  | { all: HeapFilter[] }
  | { any: HeapFilter[] }
  | { not: HeapFilter }
  | { attr: string; op: "exists" | "missing" }
  | { attr: string; op: Exclude<HeapFilterOp, "exists" | "missing">; value: AttrValue };

/** A heap's settings and size. */
export interface HeapInfo {
  name: string;
  /** Seconds. */
  halfLife: number;
  /** Landmark time `L` (Unix seconds). */
  landmark: number;
  meta: Attrs;
  /** Items (ranked or not). */
  size: number;
}

/** An item as stored. */
export interface HeapItem {
  id: string;
  base: number;
  /** `ln sum w_i exp(lambda (t_i - L))`; `null` until the first push. */
  activity: number | null;
  /** `ln(base) + activity`; `null` while unranked. */
  key: number | null;
  attrs: Attrs;
  version: number;
}

/** An item with its score and heat at `now`. */
export interface Ranked {
  /** 1-based (0 from `get`). */
  rank: number;
  id: string;
  key: number | null;
  /** `base * heat` at `now` (may be `Infinity` for pushes far in the future). */
  score: number;
  heat: number;
  base: number;
  attrs: Attrs;
  version: number;
}

export interface CreateHeapOptions extends RequestOptions {
  /** Default: now (a new heap); omitted matches an existing heap's landmark. */
  landmark?: Time;
  meta?: Attrs;
}

export interface PushInput {
  id: string;
  /** Default 1. */
  weight?: number;
  /** Default: now. */
  t?: Time;
  /** Replaces the attributes when given. */
  attrs?: Attrs;
}

export interface TopKOptions extends RequestOptions {
  /** Default 10. */
  k?: number;
  now?: Time;
  filter?: HeapFilter;
}

export interface PopOptions extends RequestOptions {
  now?: Time;
  filter?: HeapFilter;
}

export interface RebaseOptions extends RequestOptions {
  now?: Time;
  halfLife?: number;
}

/** What node-postgres (and compatible drivers) provide. */
export interface SqlClient {
  // biome-ignore lint/suspicious/noExplicitAny: driver-defined row type
  query(text: string, params?: unknown[]): Promise<{ rows: any[] }>;
}

/** The heap API, identical over both transports. */
export interface Heaps {
  createHeap(name: string, halfLife: number, opts?: CreateHeapOptions): Promise<HeapInfo>;
  dropHeap(name: string, opts?: RequestOptions): Promise<boolean>;
  heap(name: string, opts?: RequestOptions): Promise<HeapInfo | null>;
  listHeaps(opts?: RequestOptions): Promise<HeapInfo[]>;
  push(
    heap: string,
    id: string,
    weight?: number,
    opts?: { t?: Time; attrs?: Attrs } & RequestOptions,
  ): Promise<HeapItem>;
  /** Atomic batch (at most 1000 over the server). */
  pushMany(heap: string, items: PushInput[], opts?: RequestOptions): Promise<HeapItem[]>;
  setBase(heap: string, id: string, base: number, opts?: RequestOptions): Promise<HeapItem>;
  setAttrs(
    heap: string,
    id: string,
    attrs: Attrs,
    opts?: { merge?: boolean } & RequestOptions,
  ): Promise<HeapItem>;
  remove(heap: string, id: string, opts?: RequestOptions): Promise<boolean>;
  get(heap: string, id: string, opts?: { now?: Time } & RequestOptions): Promise<Ranked | null>;
  topK(heap: string, opts?: TopKOptions): Promise<Ranked[]>;
  pop(heap: string, opts?: PopOptions): Promise<Ranked | null>;
  rebase(heap: string, opts?: RebaseOptions): Promise<HeapInfo>;
}

// -- helpers ------------------------------------------------------------------

function badArg(message: string): HeapDBError {
  return new HeapDBError({ code: "invalid_argument", message });
}

function name(v: unknown, what: string): string {
  if (typeof v !== "string" || v === "") throw badArg(`${what} must be a non-empty string`);
  return v;
}

function optTime(t: Time | undefined, field: string): number | undefined {
  return t === undefined ? undefined : toSeconds(t, field);
}

/** A number from JSON or SQL (`"Infinity"`, bigint strings). */
function num(v: unknown): number {
  if (typeof v === "number") return v;
  if (typeof v === "string") {
    if (v === "Infinity") return Number.POSITIVE_INFINITY;
    if (v === "-Infinity") return Number.NEGATIVE_INFINITY;
    return Number(v);
  }
  throw new HeapDBError({
    code: "invalid_response",
    message: `expected a number, got ${typeof v}`,
  });
}

function optNum(v: unknown): number | null {
  return v === null || v === undefined ? null : num(v);
}

function attrsOf(v: unknown): Attrs {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Attrs) : {};
}

// biome-ignore lint/suspicious/noExplicitAny: decoded JSON / SQL rows
type Row = any;

/** Decoders shared by both transports (snake_case on the wire and in SQL). */
export function decodeHeapInfo(r: Row): HeapInfo {
  return {
    name: String(r.name),
    halfLife: num(r.half_life),
    landmark: num(r.landmark),
    meta: attrsOf(r.meta),
    size: num(r.size),
  };
}

export function decodeHeapItem(r: Row): HeapItem {
  return {
    id: String(r.id),
    base: num(r.base),
    activity: optNum(r.activity),
    key: optNum(r.key),
    attrs: attrsOf(r.attrs),
    version: num(r.version),
  };
}

export function decodeRanked(r: Row): Ranked {
  return {
    rank: num(r.rank),
    id: String(r.id),
    key: optNum(r.key),
    score: num(r.score),
    heat: num(r.heat),
    base: num(r.base),
    attrs: attrsOf(r.attrs),
    version: num(r.version),
  };
}

// -- server transport ---------------------------------------------------------

const seg = (s: string) => encodeURIComponent(s);

function split(o: RequestOptions | undefined): RequestOptions {
  const r: RequestOptions = {};
  if (o?.signal !== undefined) r.signal = o.signal;
  if (o?.timeoutMs !== undefined) r.timeoutMs = o.timeoutMs;
  return r;
}

class ServerHeaps implements Heaps {
  constructor(private readonly http: Transport) {}

  private path(heap: string, ...rest: string[]): string {
    return ["/heaps", seg(name(heap, "heap")), ...rest].join("/");
  }

  async createHeap(
    heap: string,
    halfLife: number,
    opts: CreateHeapOptions = {},
  ): Promise<HeapInfo> {
    const body = compact({
      half_life: halfLife,
      landmark: optTime(opts.landmark, "landmark"),
      meta: opts.meta,
    });
    return decodeHeapInfo(
      await this.http.request("PUT", this.path(heap), body, undefined, split(opts)),
    );
  }

  async dropHeap(heap: string, opts?: RequestOptions): Promise<boolean> {
    const r = await this.http.request<Row>(
      "DELETE",
      this.path(heap),
      undefined,
      undefined,
      split(opts),
    );
    return Boolean(r.dropped);
  }

  async heap(heap: string, opts?: RequestOptions): Promise<HeapInfo | null> {
    try {
      return decodeHeapInfo(
        await this.http.request("GET", this.path(heap), undefined, undefined, split(opts)),
      );
    } catch (e) {
      if (e instanceof HeapDBError && e.code === "unknown_heap") return null;
      throw e;
    }
  }

  async listHeaps(opts?: RequestOptions): Promise<HeapInfo[]> {
    const r = await this.http.request<Row>("GET", "/heaps", undefined, undefined, split(opts));
    return (r.heaps ?? []).map(decodeHeapInfo);
  }

  async push(
    heap: string,
    id: string,
    weight = 1,
    opts: { t?: Time; attrs?: Attrs } & RequestOptions = {},
  ): Promise<HeapItem> {
    const item: PushInput = { id, weight };
    if (opts.t !== undefined) item.t = opts.t;
    if (opts.attrs !== undefined) item.attrs = opts.attrs;
    const [one] = await this.pushMany(heap, [item], opts);
    return one!;
  }

  async pushMany(heap: string, items: PushInput[], opts?: RequestOptions): Promise<HeapItem[]> {
    const wire = items.map((i) =>
      compact({ id: name(i.id, "id"), weight: i.weight, time: optTime(i.t, "t"), attrs: i.attrs }),
    );
    const r = await this.http.request<Row>(
      "POST",
      this.path(heap, "push"),
      { items: wire },
      undefined,
      split(opts),
    );
    return r.items.map(decodeHeapItem);
  }

  async setBase(heap: string, id: string, base: number, opts?: RequestOptions): Promise<HeapItem> {
    const r = await this.http.request<Row>(
      "POST",
      this.path(heap, "base"),
      { items: [{ id: name(id, "id"), base }] },
      undefined,
      split(opts),
    );
    return decodeHeapItem(r.items[0]);
  }

  async setAttrs(
    heap: string,
    id: string,
    attrs: Attrs,
    opts: { merge?: boolean } & RequestOptions = {},
  ): Promise<HeapItem> {
    const r = await this.http.request<Row>(
      "POST",
      this.path(heap, "attrs"),
      { items: [{ id: name(id, "id"), attrs, merge: opts.merge ?? false }] },
      undefined,
      split(opts),
    );
    return decodeHeapItem(r.items[0]);
  }

  async remove(heap: string, id: string, opts?: RequestOptions): Promise<boolean> {
    const r = await this.http.request<Row>(
      "DELETE",
      this.path(heap, "items", seg(name(id, "id"))),
      undefined,
      undefined,
      split(opts),
    );
    return Boolean(r.removed);
  }

  async get(
    heap: string,
    id: string,
    opts: { now?: Time } & RequestOptions = {},
  ): Promise<Ranked | null> {
    try {
      const r = await this.http.request(
        "GET",
        this.path(heap, "items", seg(name(id, "id"))),
        undefined,
        { now: optTime(opts.now, "now") },
        split(opts),
      );
      return decodeRanked(r);
    } catch (e) {
      if (e instanceof HeapDBError && e.code === "unknown_item") return null;
      throw e;
    }
  }

  async topK(heap: string, opts: TopKOptions = {}): Promise<Ranked[]> {
    const body = compact({ k: opts.k ?? 10, now: optTime(opts.now, "now"), filter: opts.filter });
    const r = await this.http.request<Row>(
      "POST",
      this.path(heap, "top"),
      body,
      undefined,
      split(opts),
    );
    return r.items.map(decodeRanked);
  }

  async pop(heap: string, opts: PopOptions = {}): Promise<Ranked | null> {
    const body = compact({ now: optTime(opts.now, "now"), filter: opts.filter });
    const r = await this.http.request<Row>(
      "POST",
      this.path(heap, "pop"),
      body,
      undefined,
      split(opts),
    );
    return r.item ? decodeRanked(r.item) : null;
  }

  async rebase(heap: string, opts: RebaseOptions = {}): Promise<HeapInfo> {
    const body = compact({ now: optTime(opts.now, "now"), half_life: opts.halfLife });
    return decodeHeapInfo(
      await this.http.request("POST", this.path(heap, "rebase"), body, undefined, split(opts)),
    );
  }
}

// -- postgres transport -------------------------------------------------------

/** SQLSTATEs raised by heapdb.sql. */
const SQL_ERRORS: Record<string, { code: string; status: number }> = {
  HD400: { code: "invalid_argument", status: 400 },
  HD404: { code: "unknown_heap", status: 404 },
  HD409: { code: "heap_conflict", status: 409 },
};

/** `now` defaults to the database clock. */
const NOW = (i: number) => `coalesce($${i}::float8, extract(epoch from now())::float8)`;

const json = (v: unknown) => (v === undefined || v === null ? null : JSON.stringify(v));

class SqlHeaps implements Heaps {
  constructor(private readonly db: SqlClient) {}

  private async q(text: string, params: unknown[]): Promise<Row[]> {
    try {
      return (await this.db.query(text, params)).rows;
    } catch (e) {
      const code = (e as { code?: unknown })?.code;
      const mapped = typeof code === "string" ? SQL_ERRORS[code] : undefined;
      if (mapped) {
        throw new HeapDBError({
          code: mapped.code,
          status: mapped.status,
          message: e instanceof Error ? e.message : String(e),
          cause: e,
        });
      }
      throw new HeapDBError({
        code: "database_error",
        message: `postgres: ${e instanceof Error ? e.message : String(e)}`,
        cause: e,
      });
    }
  }

  async createHeap(
    heap: string,
    halfLife: number,
    opts: CreateHeapOptions = {},
  ): Promise<HeapInfo> {
    const rows = await this.q(
      `SELECT * FROM heapdb.create_heap($1, $2, $3::float8, $4::jsonb, ${NOW(5)})`,
      [
        name(heap, "heap"),
        halfLife,
        optTime(opts.landmark, "landmark") ?? null,
        json(opts.meta),
        null,
      ],
    );
    return decodeHeapInfo(rows[0]);
  }

  async dropHeap(heap: string): Promise<boolean> {
    return Boolean((await this.q("SELECT heapdb.drop_heap($1) AS dropped", [heap]))[0].dropped);
  }

  async heap(heap: string): Promise<HeapInfo | null> {
    const rows = await this.q("SELECT * FROM heapdb.heap_info($1)", [heap]);
    return rows[0] ? decodeHeapInfo(rows[0]) : null;
  }

  async listHeaps(): Promise<HeapInfo[]> {
    return (await this.q("SELECT * FROM heapdb.list_heaps()", [])).map(decodeHeapInfo);
  }

  async push(
    heap: string,
    id: string,
    weight = 1,
    opts: { t?: Time; attrs?: Attrs } = {},
  ): Promise<HeapItem> {
    const rows = await this.q(`SELECT * FROM heapdb.push($1, $2, $3, ${NOW(4)}, $5::jsonb)`, [
      name(heap, "heap"),
      name(id, "id"),
      weight,
      optTime(opts.t, "t") ?? null,
      json(opts.attrs),
    ]);
    return decodeHeapItem(rows[0]);
  }

  /** One statement, so the batch is atomic even on a pool. */
  async pushMany(heap: string, items: PushInput[]): Promise<HeapItem[]> {
    if (items.length === 0) return [];
    const wire = items.map((i) => ({
      id: name(i.id, "id"),
      weight: i.weight ?? 1,
      t: optTime(i.t, "t") ?? null,
      attrs: i.attrs ?? null,
    }));
    const rows = await this.q(
      `SELECT p.* FROM jsonb_array_elements($2::jsonb) WITH ORDINALITY AS e(x, n),
         LATERAL heapdb.push($1, e.x->>'id', (e.x->>'weight')::float8,
           coalesce((e.x->>'t')::float8, extract(epoch from now())::float8),
           CASE WHEN jsonb_typeof(e.x->'attrs') = 'object' THEN e.x->'attrs' END) AS p
       ORDER BY e.n`,
      [name(heap, "heap"), JSON.stringify(wire)],
    );
    return rows.map(decodeHeapItem);
  }

  async setBase(heap: string, id: string, base: number): Promise<HeapItem> {
    const rows = await this.q("SELECT * FROM heapdb.set_base($1, $2, $3)", [
      heap,
      name(id, "id"),
      base,
    ]);
    return decodeHeapItem(rows[0]);
  }

  async setAttrs(
    heap: string,
    id: string,
    attrs: Attrs,
    opts: { merge?: boolean } = {},
  ): Promise<HeapItem> {
    const rows = await this.q("SELECT * FROM heapdb.set_attrs($1, $2, $3::jsonb, $4)", [
      heap,
      name(id, "id"),
      JSON.stringify(attrs),
      opts.merge ?? false,
    ]);
    return decodeHeapItem(rows[0]);
  }

  async remove(heap: string, id: string): Promise<boolean> {
    return Boolean(
      (await this.q("SELECT heapdb.remove($1, $2) AS removed", [heap, id]))[0].removed,
    );
  }

  async get(heap: string, id: string, opts: { now?: Time } = {}): Promise<Ranked | null> {
    const rows = await this.q(`SELECT * FROM heapdb.get($1, $2, ${NOW(3)})`, [
      heap,
      id,
      optTime(opts.now, "now") ?? null,
    ]);
    return rows[0] ? decodeRanked(rows[0]) : null;
  }

  async topK(heap: string, opts: TopKOptions = {}): Promise<Ranked[]> {
    const rows = await this.q(`SELECT * FROM heapdb.top_k($1, $2, ${NOW(3)}, $4::jsonb)`, [
      heap,
      opts.k ?? 10,
      optTime(opts.now, "now") ?? null,
      json(opts.filter),
    ]);
    return rows.map(decodeRanked);
  }

  async pop(heap: string, opts: PopOptions = {}): Promise<Ranked | null> {
    const rows = await this.q(`SELECT * FROM heapdb.pop($1, ${NOW(2)}, $3::jsonb)`, [
      heap,
      optTime(opts.now, "now") ?? null,
      json(opts.filter),
    ]);
    return rows[0] ? decodeRanked(rows[0]) : null;
  }

  async rebase(heap: string, opts: RebaseOptions = {}): Promise<HeapInfo> {
    const rows = await this.q(`SELECT * FROM heapdb.rebase($1, ${NOW(2)}, $3::float8)`, [
      heap,
      optTime(opts.now, "now") ?? null,
      opts.halfLife ?? null,
    ]);
    return decodeHeapInfo(rows[0]);
  }
}

// -- entry points -------------------------------------------------------------

function isSqlClient(t: unknown): t is SqlClient {
  return typeof (t as { query?: unknown })?.query === "function";
}

/**
 * The heap API over a server connection (`connect(url)`) or a Postgres client
 * (`new pg.Client(...)`, `new pg.Pool(...)`, or anything with `query(text, params)`).
 */
export function heaps(target: HeapDB | SqlClient): Heaps {
  if (target instanceof HeapDB) return new ServerHeaps(transportOf(target));
  if (isSqlClient(target)) return new SqlHeaps(target);
  throw badArg(
    "heaps(target): target must be a HeapDB connection or a client with query(text, params)",
  );
}

/**
 * Install (or upgrade) the HeapDB Postgres engine (schema `heapdb`) with
 * `HEAPDB_SQL`. Idempotent; it runs its own transaction, so do not call it inside one.
 * Returns the installed version.
 */
export async function installHeapSql(client: SqlClient): Promise<string> {
  await client.query(HEAPDB_SQL);
  const r = await client.query("SELECT version FROM heapdb.schema_version");
  return String(r.rows[0]?.version ?? HEAPDB_SQL_VERSION);
}
