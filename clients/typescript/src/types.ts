/**
 * Public request/response types. The TypeScript API is camelCase; the wire
 * format (JSON over HTTP) is snake_case and is translated in `wire.ts`.
 *
 * User-supplied maps (`attrs`, `priors`, feature names, ...) are never renamed.
 */

/** Unix time in seconds (float), or a `Date`. */
export type Time = number | Date;

/** A feature set: an array, a `Set`, or a single feature string. */
export type Features = Iterable<string> | string;

/** JSON-compatible attribute values. */
export type AttrValue = string | number | boolean | null | AttrValue[] | { [k: string]: AttrValue };
export type Attrs = Record<string, AttrValue>;

/** Beta(alpha, beta) prior. */
export type BetaPrior = readonly [alpha: number, beta: number];

// -- schema -----------------------------------------------------------------

/**
 * Kernel kinds. The contract names the exponential kernel `"exponential"`;
 * the engine's own short name `"exp"` is accepted too.
 */
export type KernelType = "exponential" | "exp" | "window" | "lognormal" | "gamma";

/** Lag kernel, as sent on the wire: `{type, params}` with snake_case params. */
export type KernelSpec =
  | { type: "exponential" | "exp"; params: { half_life: number; horizon_half_lives?: number } }
  | { type: "window"; params: { min_lag?: number; max_lag?: number } }
  | { type: "lognormal"; params: { median: number; sigma?: number; tail_sigmas?: number } }
  | { type: "gamma"; params: { shape: number; scale: number; tail_stds?: number } };

export type CreditMode = "proportional" | "uniform";

export interface RegisterEffectOptions {
  kernel: KernelSpec;
  /** Per-feature Beta priors, e.g. `{ beans: [3, 1] }`. */
  priors?: Record<string, BetaPrior>;
  defaultPrior?: BetaPrior;
  /** Only these event types can be causes of this effect. */
  causeTypes?: Iterable<string>;
  credit?: CreditMode;
  globalPseudocount?: number;
  /** Probability that the effect happened with no recorded cause. */
  leak?: number;
}

/**
 * Declarative event weight (no code upload):
 * - `log1p`: `log1p(attrs[attr])`
 * - `attr`: `attrs[attr]`
 * - `const`: a constant `value`
 */
export type WeightSpec =
  | { kind: "log1p"; attr: string }
  | { kind: "attr"; attr: string }
  | { kind: "const"; value: number };

/** Comparison operators for declarative rule predicates. */
export type PredicateOp =
  | "eq"
  | "ne"
  | "lt"
  | "lte"
  | "gt"
  | "gte"
  | "in"
  | "not_in"
  | "exists"
  | "contains";

/** Declarative rule predicate on an event attribute: `attrs[attr] <op> value`. */
export interface Predicate {
  attr: string;
  op: PredicateOp;
  value?: AttrValue;
}

export interface AddRuleOptions {
  /** Score multiplier applied when the rule matches (e.g. 0.3, or 0 to exclude). */
  multiplier: number;
  /** Only events of these types are tested. */
  types?: Iterable<string>;
  /** Omit to match every event (of `types`). */
  predicate?: Predicate | null;
}

// -- events -----------------------------------------------------------------

export interface EventInput {
  entity: string;
  time: Time;
  type: string;
  features?: Features;
  /** Dose / exposure magnitude (default from the event type's weight, else 1.0). */
  weight?: number;
  attrs?: Attrs;
  /** Optional idempotency / correlation id. */
  externalId?: string;
}

export interface Event {
  id: number;
  entity: string;
  time: number;
  type: string;
  features: string[];
  weight: number;
  attrs: Attrs;
  externalId?: string;
}

export interface InsertManyOptions extends RequestOptions {
  /** Events per request (1..1000, default 1000). */
  batchSize?: number;
}

// -- results ----------------------------------------------------------------

export interface FeatureContribution {
  feature: string;
  strength: number;
  alpha: number;
  beta: number;
}

export interface AppliedRule {
  name: string;
  multiplier: number;
}

export interface BetaCounts {
  priorAlpha: number;
  priorBeta: number;
  pos: number;
  neg: number;
  globalPos: number;
  globalNeg: number;
  alpha?: number;
  beta?: number;
  strength?: number;
  observations?: number;
}

export interface JointCause {
  features: string[];
  strength: number;
  joint: BetaCounts | null;
  inseparable: boolean;
  name?: string;
}

export interface Explanation {
  /** Lag between cause and effect, seconds. */
  dt: number;
  kernelValue: number;
  weight: number;
  linkStrength: number;
  features: FeatureContribution[];
  warnings: string[];
  rules: AppliedRule[];
  jointCauses: JointCause[];
  entity: string;
  prospective: boolean;
  ruleMultiplier?: number;
  score?: number;
  inseparable?: boolean;
}

export interface Cause {
  rank: number;
  event: Event;
  score: number;
  explanation: Explanation | null;
}

export interface QueryStats {
  segmentsInWindow: number;
  segmentsScanned: number;
  segmentsPruned: number;
  eventsScanned: number;
  unionFallbacks: number;
}

export interface QueryResult {
  causes: Cause[];
  stats: QueryStats;
  entity?: string;
  effect?: string;
  at?: number;
  entities?: string[];
}

interface TopCausesBase extends RequestOptions {
  effect: string | string[];
  at: Time;
  k?: number;
  /** Look-back window in seconds (default: the kernel's support). */
  within?: number;
  explain?: boolean;
}
export interface TopCausesSingle extends TopCausesBase {
  entity: string;
  entities?: never;
}
export interface TopCausesGroup extends TopCausesBase {
  entities: string[];
  entity?: never;
}
export type TopCausesOptions = TopCausesSingle | TopCausesGroup;

export interface FeatureRiskOptions extends RequestOptions {
  effect: string;
  features: Features;
  weight?: number | null;
  entity?: string;
  type?: string;
  attrs?: Attrs;
  explain?: boolean;
}

export interface RiskResult {
  score: number;
  explanation: Explanation | null;
  effect?: string;
  entity?: string | null;
  features?: string[];
  type?: string | null;
}

export interface CandidateInput {
  id: string | number;
  features?: Features;
  weight?: number | null;
  type?: string;
  attrs?: Attrs;
  entity?: string;
}

export interface Candidate {
  id: string | number;
  features: string[];
  weight: number | null;
  type: string | null;
  attrs: Attrs | null;
  entity: string | null;
}

export interface RankCandidatesOptions extends RequestOptions {
  effect: string;
  candidates: CandidateInput[];
  k?: number;
  entity?: string;
  explain?: boolean;
}

export interface RankedCandidate {
  rank: number;
  candidate: Candidate;
  score: number;
  entity: string | null;
  explanation: Explanation | null;
}

export interface FeedbackOptions extends RequestOptions {
  entity: string;
  effect: string;
  at: Time;
  confirmedEventId?: number | null;
  rejectedIds?: number[];
  credit?: CreditMode;
}

export interface FeedbackReport {
  confirmed: number | null;
  rejected: number[];
  credit: Record<string, number>;
}

export interface ObserveOptions extends RequestOptions {
  entity: string;
  effect: string;
  at: Time;
  leak?: number;
}

export interface EventCredit {
  eventId: number;
  score: number;
  responsibility: number;
  features: Record<string, number>;
}

export interface ObserveReport {
  entity: string;
  effect: string;
  at: number;
  leak: number;
  leakResponsibility: number;
  credits: EventCredit[];
  totalCredit?: number;
}

export interface SettleOptions extends RequestOptions {
  until: Time;
  effects?: string | string[];
  entity?: string;
}

export interface SettleReport {
  until?: number;
  eventsSettled: number;
  eventsRejected?: number;
  rejectedWeight?: number;
  skippedCredited?: number;
  skippedObserved?: number;
  perEffect?: Record<string, number>;
}

export interface LinkReportOptions extends RequestOptions {
  effect: string;
  entity?: string;
  minSupport?: number;
}

export interface LinkRow {
  feature: string;
  strength: number;
  alpha: number;
  beta: number;
  n: number;
  lo: number;
  hi: number;
  lift: number;
}

export interface LinkReport {
  effect: string;
  entity: string | null;
  baseline: number | null;
  level: number | null;
  rows: LinkRow[];
}

export interface CalibrationReportOptions extends RequestOptions {
  effect: string;
  bins?: number;
}

export interface CalibrationBin {
  lo: number;
  hi: number;
  count: number;
  meanPredicted: number | null;
  observedRate: number | null;
}

export interface CalibrationReport {
  effect: string;
  n: number;
  brier: number | null;
  bins: CalibrationBin[];
}

export interface ForgetEntityOptions extends RequestOptions {
  /** Also remove this entity's contribution to the global link layer. */
  scrubGlobal?: boolean;
}

export interface DatabaseInfo {
  databaseId: string;
  plan: string;
  /** Usage counters (keys camelCased, e.g. `requestsThisPeriod`). */
  usage: Record<string, unknown>;
  /** Plan limits (keys camelCased, e.g. `maxRequestsPerMonth`). `null` = unlimited. */
  limits: Record<string, unknown>;
}

// -- transport --------------------------------------------------------------

/** Per-call options. */
export interface RequestOptions {
  /** Abort the call (including pending retries). */
  signal?: AbortSignal;
  /** Override the client's per-attempt timeout. */
  timeoutMs?: number;
}

export type FetchLike = (input: string, init: RequestInit) => Promise<Response>;

export interface ConnectOptions {
  /** Per-attempt timeout in ms (default 30000; 0 disables). */
  timeoutMs?: number;
  /** Retries after the first attempt on 429 / 503 / network errors (default 3). */
  retries?: number;
  /** First backoff delay in ms (default 250); doubles each retry, with full jitter. */
  retryBaseDelayMs?: number;
  /** Cap for a single backoff or Retry-After wait in ms (default 30000). */
  retryMaxDelayMs?: number;
  /** Appended to the default `heapdb-ts/<version>` user agent. */
  userAgent?: string;
  /**
   * Server origin for self-hosted or local servers, e.g. `http://localhost:8080`.
   * The database path `/v1/databases/<db>` is appended. Default `https://<host>`.
   */
  baseUrl?: string;
  /** Custom fetch (tests, proxies, instrumented fetch). Default: global `fetch`. */
  fetch?: FetchLike;
  /** Extra headers sent with every request. */
  headers?: Record<string, string>;
}
