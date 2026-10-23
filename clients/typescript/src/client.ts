import { HeapDBError, maskKey } from "./errors.js";
import { Transport } from "./http.js";
import type {
  AddRuleOptions,
  CalibrationReport,
  CalibrationReportOptions,
  ConnectOptions,
  DatabaseInfo,
  EffectPage,
  EffectsOptions,
  EntitiesOptions,
  EntityPage,
  EventInput,
  EventTypeCatalog,
  EventTypesOptions,
  FeatureRiskOptions,
  FeedbackOptions,
  FeedbackReport,
  FetchLike,
  ForgetEntityOptions,
  InsertManyOptions,
  LinkReport,
  LinkReportOptions,
  ObserveOptions,
  ObserveReport,
  OccurrencePage,
  OccurrencesOptions,
  QueryResult,
  RankCandidatesOptions,
  RankedCandidate,
  RegisterEffectOptions,
  RequestOptions,
  RiskResult,
  SettleOptions,
  SettleReport,
  TopCausesOptions,
  WeightSpec,
} from "./types.js";
import { dataApiBase, envUrl, type ParsedUrl, parseConnectionUrl } from "./url.js";
import { VERSION } from "./version.js";
import {
  compact,
  decodeCalibration,
  decodeCounts,
  decodeEffects,
  decodeEntities,
  decodeEventTypes,
  decodeFeedback,
  decodeInfo,
  decodeLinkReport,
  decodeObserve,
  decodeOccurrences,
  decodeQueryResult,
  decodeRanked,
  decodeRisk,
  decodeSettle,
  encodeCandidate,
  encodeEvent,
  featureList,
  toSeconds,
} from "./wire.js";

/** Maximum events per `POST /events` request (contract §4). */
export const MAX_BATCH = 1000;

function badArg(message: string): HeapDBError {
  return new HeapDBError({ code: "invalid_argument", message });
}

function requireString(v: unknown, name: string): string {
  if (typeof v !== "string" || v === "") throw badArg(`${name} is required`);
  return v;
}

function optList(v: Iterable<string> | undefined): string[] | undefined {
  return v === undefined ? undefined : Array.from(v);
}

function split<T extends RequestOptions>(o: T): RequestOptions {
  const r: RequestOptions = {};
  if (o.signal !== undefined) r.signal = o.signal;
  if (o.timeoutMs !== undefined) r.timeoutMs = o.timeoutMs;
  return r;
}

/**
 * A connection to one HeapDB database over the Data API.
 * Create with {@link connect}. Instances are stateless and safe to share.
 */
export class HeapDB {
  readonly #http: Transport;
  readonly #display: string;
  /** Database id from the connection URL. */
  readonly database: string;
  /** Data API base URL, e.g. `https://api.heapdb.com/v1/databases/db_...`. */
  readonly baseUrl: string;

  /** @internal use {@link connect}. */
  constructor(parsed: ParsedUrl, opts: ConnectOptions = {}) {
    const f: FetchLike | undefined =
      opts.fetch ?? (typeof fetch === "function" ? fetch.bind(globalThis) : undefined);
    if (!f) {
      throw new HeapDBError({
        code: "no_fetch",
        message: "no global fetch: use Node >= 18 or pass { fetch } to connect()",
      });
    }
    const retries = opts.retries ?? 3;
    if (!Number.isInteger(retries) || retries < 0) throw badArg("retries must be an integer >= 0");
    const timeoutMs = opts.timeoutMs ?? 30_000;
    if (!(timeoutMs >= 0)) throw badArg("timeoutMs must be >= 0");
    this.database = parsed.database;
    this.baseUrl = dataApiBase(parsed, opts.baseUrl);
    this.#display = `heapdb://${maskKey(parsed.apiKey)}@${parsed.host}/${parsed.database}`;
    this.#http = new Transport({
      base: this.baseUrl,
      apiKey: parsed.apiKey,
      fetch: f,
      timeoutMs,
      retries,
      retryBaseDelayMs: opts.retryBaseDelayMs ?? 250,
      retryMaxDelayMs: opts.retryMaxDelayMs ?? 30_000,
      userAgent: `heapdb-ts/${VERSION}${opts.userAgent ? ` ${opts.userAgent}` : ""}`,
      headers: opts.headers ?? {},
    });
  }

  toString(): string {
    return `HeapDB(${this.#display})`;
  }

  toJSON(): { database: string; baseUrl: string } {
    return { database: this.database, baseUrl: this.baseUrl };
  }

  /** Node's util.inspect / console.log. */
  [Symbol.for("nodejs.util.inspect.custom")](): string {
    return this.toString();
  }

  // -- schema (admin scope) -------------------------------------------------

  /** Register (or replace) an effect type with its lag kernel and link priors. */
  async registerEffect(name: string, opts: RegisterEffectOptions & RequestOptions): Promise<void> {
    requireString(name, "name");
    if (!opts?.kernel || typeof opts.kernel.type !== "string") throw badArg("kernel is required");
    await this.#http.request(
      "POST",
      "/effects",
      compact({
        name,
        kernel: opts.kernel,
        priors: opts.priors,
        default_prior: opts.defaultPrior,
        cause_types: optList(opts.causeTypes),
        credit: opts.credit,
        global_pseudocount: opts.globalPseudocount,
        leak: opts.leak,
        combine: opts.combine,
        learning_rate: opts.learningRate,
        l2: opts.l2,
      }),
      undefined,
      split(opts),
    );
  }

  /** Register an event type with a declarative weight function. */
  async registerEventType(
    type: string,
    weight?: WeightSpec | null,
    opts: RequestOptions = {},
  ): Promise<void> {
    requireString(type, "type");
    await this.#http.request(
      "POST",
      "/event-types",
      compact({ type, weight: weight ?? undefined }),
      undefined,
      opts,
    );
  }

  /** Add a soft plausibility rule (score multiplier) to an effect. */
  async addRule(
    effect: string,
    name: string,
    opts: AddRuleOptions & RequestOptions,
  ): Promise<void> {
    requireString(effect, "effect");
    requireString(name, "name");
    if (typeof opts?.multiplier !== "number") throw badArg("multiplier is required");
    await this.#http.request(
      "POST",
      "/rules",
      compact({
        effect,
        name,
        multiplier: opts.multiplier,
        types: optList(opts.types),
        predicate: opts.predicate ?? undefined,
      }),
      undefined,
      split(opts),
    );
  }

  // -- events -----------------------------------------------------------------

  /** Append one event; returns its id. */
  async insert(event: EventInput, opts: RequestOptions = {}): Promise<number> {
    const res = await this.#http.request<{ ids?: number[] }>(
      "POST",
      "/events",
      { events: [encodeEvent(event)] },
      undefined,
      opts,
    );
    const id = res.ids?.[0];
    if (typeof id !== "number") {
      throw new HeapDBError({ code: "invalid_response", message: "POST /events: no id returned" });
    }
    return id;
  }

  /**
   * Append many events, split into requests of at most 1000 events. Batches are
   * sent sequentially; returns ids in input order. If a batch fails, the error's
   * `insertedIds` holds the ids of the batches already committed.
   */
  async insertMany(events: Iterable<EventInput>, opts: InsertManyOptions = {}): Promise<number[]> {
    const size = opts.batchSize ?? MAX_BATCH;
    if (!Number.isInteger(size) || size < 1 || size > MAX_BATCH) {
      throw badArg(`batchSize must be an integer in 1..${MAX_BATCH}`);
    }
    const encoded = Array.from(events, encodeEvent);
    const ids: number[] = [];
    for (let i = 0; i < encoded.length; i += size) {
      const batch = encoded.slice(i, i + size);
      let res: { ids?: number[] };
      try {
        res = await this.#http.request(
          "POST",
          "/events",
          { events: batch },
          undefined,
          split(opts),
        );
      } catch (e) {
        if (e instanceof HeapDBError && ids.length > 0) {
          throw new HeapDBError({
            code: e.code,
            status: e.status,
            message: `${e.message} (after ${ids.length} of ${encoded.length} events were inserted)`,
            retryAfter: e.retryAfter,
            requestId: e.requestId,
            insertedIds: ids.slice(),
            cause: e,
          });
        }
        throw e;
      }
      if (!Array.isArray(res.ids) || res.ids.length !== batch.length) {
        throw new HeapDBError({
          code: "invalid_response",
          message: `POST /events: expected ${batch.length} ids`,
          insertedIds: ids.slice(),
        });
      }
      ids.push(...res.ids);
    }
    return ids;
  }

  // -- queries ----------------------------------------------------------------

  /**
   * Rank the most likely causes of `effect` at time `at` for one `entity`, or
   * across a group with `entities`.
   */
  async topCauses(opts: TopCausesOptions): Promise<QueryResult> {
    const hasEntity = typeof opts.entity === "string";
    const hasGroup = Array.isArray(opts.entities);
    if (hasEntity === hasGroup) throw badArg("pass exactly one of entity or entities");
    if (hasGroup && opts.entities!.length === 0) throw badArg("entities must not be empty");
    const res = await this.#http.request(
      "POST",
      "/query/top-causes",
      compact({
        entity: opts.entity,
        entities: opts.entities,
        effect: opts.effect,
        at: toSeconds(opts.at, "at"),
        k: opts.k,
        within: opts.within,
        explain: opts.explain,
      }),
      undefined,
      split(opts),
    );
    return decodeQueryResult(res);
  }

  /** Pre-event risk of a hypothetical event with these features (no kernel term). */
  async featureRisk(opts: FeatureRiskOptions): Promise<RiskResult> {
    requireString(opts.effect, "effect");
    const res = await this.#http.request(
      "POST",
      "/query/risk",
      compact({
        effect: opts.effect,
        features: featureList(opts.features),
        weight: opts.weight,
        entity: opts.entity,
        type: opts.type,
        attrs: opts.attrs,
        explain: opts.explain,
      }),
      undefined,
      split(opts),
    );
    return decodeRisk(res);
  }

  /** Rank candidate (not yet happened) events by risk; returns the top `k` (default 10). */
  async rankCandidates(opts: RankCandidatesOptions): Promise<RankedCandidate[]> {
    requireString(opts.effect, "effect");
    if (!Array.isArray(opts.candidates)) throw badArg("candidates must be an array");
    const res = await this.#http.request(
      "POST",
      "/query/risk",
      compact({
        effect: opts.effect,
        candidates: opts.candidates.map(encodeCandidate),
        k: opts.k ?? 10,
        entity: opts.entity,
        explain: opts.explain,
      }),
      undefined,
      split(opts),
    );
    return decodeRanked(res);
  }

  // -- learning ---------------------------------------------------------------

  /** Tell HeapDB which event really caused an effect (and which did not). */
  async feedback(opts: FeedbackOptions): Promise<FeedbackReport> {
    requireString(opts.entity, "entity");
    requireString(opts.effect, "effect");
    const res = await this.#http.request(
      "POST",
      "/feedback",
      compact({
        entity: opts.entity,
        effect: opts.effect,
        at: toSeconds(opts.at, "at"),
        confirmed_event_id: opts.confirmedEventId,
        rejected_ids: opts.rejectedIds,
        credit: opts.credit,
      }),
      undefined,
      split(opts),
    );
    return decodeFeedback(res);
  }

  /** Record an effect occurrence for observational (unlabelled) learning. */
  async observe(opts: ObserveOptions): Promise<ObserveReport> {
    requireString(opts.entity, "entity");
    requireString(opts.effect, "effect");
    const res = await this.#http.request(
      "POST",
      "/observe",
      compact({
        entity: opts.entity,
        effect: opts.effect,
        at: toSeconds(opts.at, "at"),
        leak: opts.leak,
      }),
      undefined,
      split(opts),
    );
    return decodeObserve(res);
  }

  /** Settle events whose effect window closed before `until` without an effect. */
  async settle(opts: SettleOptions): Promise<SettleReport> {
    const effects = typeof opts.effects === "string" ? [opts.effects] : opts.effects;
    const res = await this.#http.request(
      "POST",
      "/settle",
      compact({ until: toSeconds(opts.until, "until"), effects, entity: opts.entity }),
      undefined,
      split(opts),
    );
    return decodeSettle(res);
  }

  // -- reports ----------------------------------------------------------------

  /** Learned link strengths per feature for an effect (optionally one entity). */
  async linkReport(opts: LinkReportOptions): Promise<LinkReport> {
    requireString(opts.effect, "effect");
    const res = await this.#http.request(
      "GET",
      "/links",
      undefined,
      { effect: opts.effect, entity: opts.entity, min_support: opts.minSupport },
      split(opts),
    );
    return decodeLinkReport(res, opts.effect, opts.entity);
  }

  /** Calibration of past predictions against feedback. */
  async calibrationReport(opts: CalibrationReportOptions): Promise<CalibrationReport> {
    requireString(opts.effect, "effect");
    const res = await this.#http.request(
      "GET",
      "/calibration",
      undefined,
      { effect: opts.effect, bins: opts.bins },
      split(opts),
    );
    return decodeCalibration(res, opts.effect);
  }

  // -- catalog ----------------------------------------------------------------

  /** Registered effects with kernels, priors and counts (registration order); paginate with `cursor`. */
  async effects(opts: EffectsOptions = {}): Promise<EffectPage> {
    const res = await this.#http.request(
      "GET",
      "/effects",
      undefined,
      { limit: opts.limit, cursor: opts.cursor || undefined },
      split(opts),
    );
    return decodeEffects(res);
  }

  /** Entities by most recent activity, optionally by name prefix; paginate with `cursor`. */
  async entities(opts: EntitiesOptions = {}): Promise<EntityPage> {
    const res = await this.#http.request(
      "GET",
      "/entities",
      undefined,
      { prefix: opts.prefix || undefined, limit: opts.limit, cursor: opts.cursor || undefined },
      split(opts),
    );
    return decodeEntities(res);
  }

  /** Event types with counts and their most common features (most events first); paginate with `cursor`. */
  async eventTypes(opts: EventTypesOptions = {}): Promise<EventTypeCatalog> {
    const res = await this.#http.request(
      "GET",
      "/event-types",
      undefined,
      { features: opts.features, limit: opts.limit, cursor: opts.cursor || undefined },
      split(opts),
    );
    return decodeEventTypes(res);
  }

  /** Effect occurrences (from observe and feedback), newest first; `cursor` goes further back. */
  async occurrences(opts: OccurrencesOptions = {}): Promise<OccurrencePage> {
    const res = await this.#http.request(
      "GET",
      "/occurrences",
      undefined,
      {
        effect: opts.effect,
        entity: opts.entity,
        limit: opts.limit,
        cursor: opts.cursor || undefined,
      },
      split(opts),
    );
    return decodeOccurrences(res);
  }

  // -- admin / misc -----------------------------------------------------------

  /** Delete everything stored about an entity (right to erasure). Returns removal counts. */
  async forgetEntity(
    entity: string,
    opts: ForgetEntityOptions = {},
  ): Promise<Record<string, number>> {
    requireString(entity, "entity");
    const res = await this.#http.request(
      "DELETE",
      `/entities/${encodeURIComponent(entity)}`,
      undefined,
      opts.scrubGlobal === undefined ? undefined : { scrub_global: opts.scrubGlobal },
      split(opts),
    );
    return decodeCounts(res);
  }

  /** Run an HQL statement, e.g. `TOP 3 CAUSES OF bloating FOR me AT now EXPLAIN`. */
  async hql<T = unknown>(query: string, opts: RequestOptions = {}): Promise<T> {
    requireString(query, "query");
    return this.#http.request<T>("POST", "/hql", { query }, undefined, opts);
  }

  /** Database id, plan, usage and limits. Works even when the monthly quota is exhausted. */
  async info(opts: RequestOptions = {}): Promise<DatabaseInfo> {
    const res = await this.#http.request("GET", "", undefined, undefined, opts);
    return decodeInfo(res);
  }
}

/**
 * Connect to a HeapDB database.
 *
 * `url` is `heapdb://<api_key>@<host>/<database_id>`; when omitted, `HEAPDB_URL`
 * from the environment is used (where a `process.env` exists). No request is
 * made until the first call.
 */
export function connect(url?: string, opts?: ConnectOptions): HeapDB;
export function connect(opts: ConnectOptions): HeapDB;
export function connect(urlOrOpts?: string | ConnectOptions, maybeOpts?: ConnectOptions): HeapDB {
  let url: string | undefined;
  let opts: ConnectOptions;
  if (typeof urlOrOpts === "object" && urlOrOpts !== null) {
    opts = urlOrOpts;
  } else {
    url = urlOrOpts;
    opts = maybeOpts ?? {};
  }
  return new HeapDB(parseConnectionUrl(url ?? envUrl() ?? ""), opts);
}
