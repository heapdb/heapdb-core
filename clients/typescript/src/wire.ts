/**
 * Translation between the camelCase TypeScript API and the snake_case wire format.
 * Decoders are tolerant: unknown fields are ignored, missing optional fields
 * become `null`/defaults, so additive server changes never break the client.
 */
import { HeapDBError } from "./errors.js";
import type {
  AppliedRule,
  Attrs,
  BetaCounts,
  CalibrationBin,
  CalibrationReport,
  Candidate,
  CandidateInput,
  Cause,
  DatabaseInfo,
  Event,
  EventCredit,
  EventInput,
  Explanation,
  FeatureContribution,
  Features,
  FeedbackReport,
  JointCause,
  LinkReport,
  LinkRow,
  ObserveReport,
  QueryResult,
  QueryStats,
  RankedCandidate,
  RiskResult,
  SettleReport,
  Time,
} from "./types.js";

// biome-ignore lint/suspicious/noExplicitAny: decoded JSON
type J = any;

export function toSeconds(t: Time, field = "time"): number {
  const v = t instanceof Date ? t.getTime() / 1000 : t;
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new HeapDBError({
      code: "invalid_argument",
      message: `${field} must be Unix seconds (finite number) or a valid Date`,
    });
  }
  return v;
}

export function featureList(f: Features | undefined): string[] {
  if (f === undefined) return [];
  if (typeof f === "string") return [f];
  return Array.from(new Set(f));
}

/** Drop `undefined` values so they are not sent. */
export function compact<T extends Record<string, unknown>>(o: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(o)) if (v !== undefined) out[k] = v;
  return out as Partial<T>;
}

export function camelKey(k: string): string {
  return k.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase());
}

/** Shallow camelCase of server-defined keys (not user data). */
function camelKeys(o: J): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (o && typeof o === "object") for (const [k, v] of Object.entries(o)) out[camelKey(k)] = v;
  return out;
}

// -- encoders ---------------------------------------------------------------

export function encodeEvent(e: EventInput): Record<string, unknown> {
  if (!e || typeof e.entity !== "string" || e.entity === "") {
    throw new HeapDBError({ code: "invalid_argument", message: "event.entity is required" });
  }
  if (typeof e.type !== "string" || e.type === "") {
    throw new HeapDBError({ code: "invalid_argument", message: "event.type is required" });
  }
  return compact({
    entity: e.entity,
    time: toSeconds(e.time, "event.time"),
    type: e.type,
    features: featureList(e.features),
    weight: e.weight,
    attrs: e.attrs,
    external_id: e.externalId,
  });
}

export function encodeCandidate(c: CandidateInput): Record<string, unknown> {
  return compact({
    id: c.id,
    features: featureList(c.features),
    weight: c.weight,
    type: c.type,
    attrs: c.attrs,
    entity: c.entity,
  });
}

// -- decoders ---------------------------------------------------------------

const num = (v: J, d = 0): number => (typeof v === "number" ? v : d);
const numOrNull = (v: J): number | null => (typeof v === "number" ? v : null);
const str = (v: J, d = ""): string => (typeof v === "string" ? v : d);
const strOrNull = (v: J): string | null => (typeof v === "string" ? v : null);
const arr = (v: J): J[] => (Array.isArray(v) ? v : []);
const rec = <T>(v: J): Record<string, T> =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, T>) : {};

export function decodeEvent(e: J): Event {
  const out: Event = {
    id: num(e?.id),
    entity: str(e?.entity),
    time: num(e?.time),
    type: str(e?.type),
    features: arr(e?.features).map(String),
    weight: num(e?.weight, 1),
    attrs: rec(e?.attrs) as Attrs,
  };
  if (typeof e?.external_id === "string") out.externalId = e.external_id;
  return out;
}

function decodeBetaCounts(b: J): BetaCounts | null {
  if (!b || typeof b !== "object") return null;
  const out: BetaCounts = {
    priorAlpha: num(b.prior_alpha),
    priorBeta: num(b.prior_beta),
    pos: num(b.pos),
    neg: num(b.neg),
    globalPos: num(b.global_pos),
    globalNeg: num(b.global_neg),
  };
  if (typeof b.alpha === "number") out.alpha = b.alpha;
  if (typeof b.beta === "number") out.beta = b.beta;
  if (typeof b.strength === "number") out.strength = b.strength;
  if (typeof b.observations === "number") out.observations = b.observations;
  return out;
}

export function decodeExplanation(x: J): Explanation | null {
  if (!x || typeof x !== "object") return null;
  const out: Explanation = {
    dt: num(x.dt),
    kernelValue: num(x.kernel_value),
    weight: num(x.weight, 1),
    linkStrength: num(x.link_strength),
    features: arr(x.features).map((f: J): FeatureContribution => {
      const fc: FeatureContribution = {
        feature: str(f?.feature),
        strength: num(f?.strength),
        alpha: num(f?.alpha),
        beta: num(f?.beta),
      };
      if (typeof f?.log_odds === "number") fc.logOdds = f.log_odds;
      if (typeof f?.odds_multiplier === "number") fc.oddsMultiplier = f.odds_multiplier;
      return fc;
    }),
    warnings: arr(x.warnings).map(String),
    rules: arr(x.rules).map(
      (r: J): AppliedRule => ({ name: str(r?.name), multiplier: num(r?.multiplier, 1) }),
    ),
    jointCauses: arr(x.joint_causes).map((j: J): JointCause => {
      const jc: JointCause = {
        features: arr(j?.features).map(String),
        strength: num(j?.strength),
        joint: decodeBetaCounts(j?.joint),
        inseparable: j?.inseparable === true,
      };
      if (typeof j?.name === "string") jc.name = j.name;
      return jc;
    }),
    entity: str(x.entity),
    prospective: x.prospective === true,
  };
  if (typeof x.rule_multiplier === "number") out.ruleMultiplier = x.rule_multiplier;
  if (typeof x.score === "number") out.score = x.score;
  if (typeof x.inseparable === "boolean") out.inseparable = x.inseparable;
  if (x.combine === "logistic" || x.combine === "noisy_or") out.combine = x.combine;
  if (typeof x.base_log_odds === "number") out.baseLogOdds = x.base_log_odds;
  if (typeof x.base_odds === "number") out.baseOdds = x.base_odds;
  if (typeof x.log_odds === "number") out.logOdds = x.log_odds;
  if (typeof x.odds === "number") out.odds = x.odds;
  return out;
}

function decodeStats(s: J): QueryStats {
  return {
    segmentsInWindow: num(s?.segments_in_window),
    segmentsScanned: num(s?.segments_scanned),
    segmentsPruned: num(s?.segments_pruned),
    eventsScanned: num(s?.events_scanned),
    unionFallbacks: num(s?.union_fallbacks),
  };
}

export function decodeQueryResult(r: J): QueryResult {
  const out: QueryResult = {
    causes: arr(r?.causes).map(
      (c: J, i: number): Cause => ({
        rank: num(c?.rank, i + 1),
        event: decodeEvent(c?.event),
        score: num(c?.score),
        explanation: decodeExplanation(c?.explanation),
      }),
    ),
    stats: decodeStats(r?.stats),
  };
  if (typeof r?.entity === "string") out.entity = r.entity;
  if (typeof r?.effect === "string") out.effect = r.effect;
  if (typeof r?.at === "number") out.at = r.at;
  if (Array.isArray(r?.entities)) out.entities = r.entities.map(String);
  return out;
}

export function decodeRisk(r: J): RiskResult {
  // Tolerate a bare number (the engine's non-explain return type).
  if (typeof r === "number") return { score: r, explanation: null };
  const out: RiskResult = { score: num(r?.score), explanation: decodeExplanation(r?.explanation) };
  if (typeof r?.effect === "string") out.effect = r.effect;
  if (r && "entity" in r) out.entity = strOrNull(r.entity);
  if (Array.isArray(r?.features)) out.features = r.features.map(String);
  if (r && "type" in r) out.type = strOrNull(r.type);
  return out;
}

function decodeCandidate(c: J): Candidate {
  return {
    id: typeof c?.id === "number" || typeof c?.id === "string" ? c.id : String(c?.id),
    features: arr(c?.features).map(String),
    weight: numOrNull(c?.weight),
    type: strOrNull(c?.type),
    attrs: c?.attrs && typeof c.attrs === "object" ? (c.attrs as Attrs) : null,
    entity: strOrNull(c?.entity),
  };
}

export function decodeRanked(r: J): RankedCandidate[] {
  const list = Array.isArray(r) ? r : arr(r?.candidates ?? r?.ranked ?? r?.results);
  return list.map(
    (x: J, i: number): RankedCandidate => ({
      rank: num(x?.rank, i + 1),
      candidate: decodeCandidate(x?.candidate),
      score: num(x?.score),
      entity: strOrNull(x?.entity),
      explanation: decodeExplanation(x?.explanation),
    }),
  );
}

export function decodeFeedback(r: J): FeedbackReport {
  return {
    confirmed: numOrNull(r?.confirmed),
    rejected: arr(r?.rejected).map(Number),
    credit: rec<number>(r?.credit),
  };
}

export function decodeObserve(r: J): ObserveReport {
  const out: ObserveReport = {
    entity: str(r?.entity),
    effect: str(r?.effect),
    at: num(r?.at),
    leak: num(r?.leak),
    leakResponsibility: num(r?.leak_responsibility),
    credits: arr(r?.credits).map(
      (c: J): EventCredit => ({
        eventId: num(c?.event_id),
        score: num(c?.score),
        responsibility: num(c?.responsibility),
        features: rec<number>(c?.features),
      }),
    ),
  };
  if (typeof r?.total_credit === "number") out.totalCredit = r.total_credit;
  return out;
}

export function decodeSettle(r: J): SettleReport {
  // Contract: `{settled}`. Accept `settled` as a count or as a full SettleReport,
  // or the SettleReport fields at top level.
  const s = r?.settled;
  if (typeof s === "number") return { eventsSettled: s };
  const src = s && typeof s === "object" ? s : r;
  const out: SettleReport = { eventsSettled: num(src?.events_settled) };
  if (typeof src?.until === "number") out.until = src.until;
  if (typeof src?.events_rejected === "number") out.eventsRejected = src.events_rejected;
  if (typeof src?.rejected_weight === "number") out.rejectedWeight = src.rejected_weight;
  if (typeof src?.skipped_credited === "number") out.skippedCredited = src.skipped_credited;
  if (typeof src?.skipped_observed === "number") out.skippedObserved = src.skipped_observed;
  if (src?.per_effect && typeof src.per_effect === "object") out.perEffect = rec(src.per_effect);
  return out;
}

export function decodeLinkReport(r: J, effect: string, entity: string | undefined): LinkReport {
  const rows = Array.isArray(r) ? r : arr(r?.rows);
  return {
    effect: str(r?.effect, effect),
    entity: r && !Array.isArray(r) && "entity" in r ? strOrNull(r.entity) : (entity ?? null),
    baseline: numOrNull(r?.baseline),
    level: numOrNull(r?.level),
    rows: rows.map((x: J): LinkRow => {
      const row: LinkRow = {
        feature: str(x?.feature),
        strength: num(x?.strength),
        alpha: num(x?.alpha),
        beta: num(x?.beta),
        n: num(x?.n),
        lo: num(x?.lo),
        hi: num(x?.hi),
        lift: num(x?.lift),
      };
      if (typeof x?.log_odds === "number") row.logOdds = x.log_odds;
      return row;
    }),
  };
}

export function decodeCalibration(r: J, effect: string): CalibrationReport {
  return {
    effect: str(r?.effect, effect),
    n: num(r?.n),
    brier: numOrNull(r?.brier),
    bins: arr(r?.bins).map(
      (b: J): CalibrationBin => ({
        lo: num(b?.lo),
        hi: num(b?.hi),
        count: num(b?.count),
        meanPredicted: numOrNull(b?.mean_predicted),
        observedRate: numOrNull(b?.observed_rate),
      }),
    ),
  };
}

export function decodeCounts(r: J): Record<string, number> {
  const src = r?.counts && typeof r.counts === "object" ? r.counts : r;
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(rec(src))) if (typeof v === "number") out[camelKey(k)] = v;
  return out;
}

export function decodeInfo(r: J): DatabaseInfo {
  return {
    databaseId: str(r?.database_id),
    plan: str(r?.plan),
    usage: camelKeys(r?.usage),
    limits: camelKeys(r?.limits),
  };
}
