//! One database's state: the engine plus the declarative schema (weight
//! functions, rule predicates), the `external_id` index and the WAL
//! position. Every mutation is a [`WalOp`]; applying the same ops in the
//! same order to the same snapshot gives a bit-identical engine, which is
//! what recovery relies on.
//!
//! Keyed heaps (`docs/heaps.md`) live next to the engine in a
//! [`MemoryHeaps`]; their mutations are [`WalOp::Heap`] records (or
//! [`WalOp::Hql`] for heap statements).
//!
//! Snapshot format 2: one header line
//! `{"kind":"heapdb_server","format":2,"database_id","epoch","seq","schema","external_ids","heaps":N}`,
//! then N heap records (`{"kind":"heap",...}` / `{"kind":"item",...}`,
//! see [`MemoryHeaps::to_records`]), then the engine's JSONL snapshot
//! (format v3). Format 1 (no heap section) still loads, with no heaps.

use std::collections::BTreeMap;
use std::io::BufRead;

use heapdb::heaps::hql;
use heapdb::heaps::{AttrsArgs, BaseArgs, HeapBackend, HeapOp, MemoryHeaps, PushArgs};
use heapdb::query::{Output, Statement};
use heapdb::{Candidate, Credit, Filter, HeapDB, LinkReportOptions, NewEvent, RiskResult, TopK};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::error::{ApiError, ApiResult};
use crate::json::{self, Num};
use crate::schema::{EffectReq, PredicateSpec, RuleSpec, WeightSpec};

/// Engine bucket width of hosted databases.
pub const BUCKET_SECONDS: f64 = 900.0;
/// Engine set cap per segment.
pub const MAX_SETS: usize = 64;
/// Maximum events per `POST /events` (and items per heap batch).
pub const MAX_BATCH: usize = 1000;
/// Snapshot header format written by this version.
pub const SNAPSHOT_FORMAT: u32 = 2;

// ---------------------------------------------------------------------------
// Requests (also WAL payloads)
// ---------------------------------------------------------------------------

/// One event of `POST /events`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventIn {
    /// Entity.
    pub entity: String,
    /// Unix seconds.
    pub time: f64,
    /// Event type.
    #[serde(rename = "type")]
    pub event_type: String,
    /// Feature tokens.
    #[serde(default)]
    pub features: Vec<String>,
    /// Dose (`None`: from the event type's weight function, else 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    /// Attributes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attrs: Option<Map<String, Value>>,
    /// Idempotency key: a second insert with the same id returns the first id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
}

/// Body of `POST /events`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsertReq {
    /// Up to 1000 events.
    pub events: Vec<EventIn>,
    /// Also return the stored events (with derived weights).
    #[serde(default)]
    pub return_events: bool,
}

/// Body of `POST /event-types`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventTypeReq {
    /// Event type.
    #[serde(rename = "type")]
    pub event_type: String,
    /// Declarative weight function.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight: Option<WeightSpec>,
}

/// Body of `POST /rules`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleReq {
    /// Effect.
    pub effect: String,
    /// Rule name.
    pub name: String,
    /// Multiplier in `[0, 1]`.
    pub multiplier: f64,
    /// Event types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub types: Option<Vec<String>>,
    /// Attribute predicate (omitted: all events of `types`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<PredicateSpec>,
}

/// Body of `POST /feedback`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeedbackReq {
    /// Entity.
    pub entity: String,
    /// Effect.
    pub effect: String,
    /// Effect time.
    pub at: f64,
    /// Confirmed cause.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirmed_event_id: Option<u64>,
    /// Rejected candidates.
    #[serde(default)]
    pub rejected_ids: Vec<u64>,
    /// `proportional` or `uniform`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credit: Option<String>,
}

/// Body of `POST /observe`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObserveReq {
    /// Entity.
    pub entity: String,
    /// Effect.
    pub effect: String,
    /// Occurrence time.
    pub at: f64,
    /// Leak override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub leak: Option<Num>,
}

/// One name or a list.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany {
    /// One.
    One(String),
    /// Many.
    Many(Vec<String>),
}

impl OneOrMany {
    /// As a list.
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            OneOrMany::One(s) => vec![s.clone()],
            OneOrMany::Many(v) => v.clone(),
        }
    }
}

/// Body of `POST /settle`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SettleReq {
    /// Horizon.
    pub until: f64,
    /// Effects (default: all).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effects: Option<OneOrMany>,
    /// Entity (default: all).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity: Option<String>,
    /// Partial evidence for windows still open.
    #[serde(default)]
    pub partial: bool,
}

/// Body of `POST /query/top-causes`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopCausesReq {
    /// One entity ...
    #[serde(default)]
    pub entity: Option<String>,
    /// ... or several (group query).
    #[serde(default)]
    pub entities: Option<Vec<String>>,
    /// Effect.
    pub effect: String,
    /// Effect time.
    pub at: f64,
    /// k (default 3).
    #[serde(default)]
    pub k: Option<usize>,
    /// Lookback cap (seconds).
    #[serde(default)]
    pub within: Option<f64>,
    /// Attach explanations.
    #[serde(default)]
    pub explain: bool,
}

/// One candidate of `POST /query/risk` (candidates mode).
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateIn {
    /// Caller's id (echoed back).
    #[serde(default)]
    pub id: Value,
    /// Features.
    #[serde(default)]
    pub features: Vec<String>,
    /// Dose.
    #[serde(default)]
    pub weight: Option<f64>,
    /// Event type.
    #[serde(default, rename = "type")]
    pub event_type: Option<String>,
    /// Attributes.
    #[serde(default)]
    pub attrs: Option<Map<String, Value>>,
    /// Entity whose links to use.
    #[serde(default)]
    pub entity: Option<String>,
}

/// Body of `POST /query/risk`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskReq {
    /// Effect.
    pub effect: String,
    /// Single mode: features.
    #[serde(default)]
    pub features: Option<Vec<String>>,
    /// Single mode: dose (omitted: from type/attrs, else 1).
    #[serde(default)]
    pub weight: Option<f64>,
    /// Entity view (both modes).
    #[serde(default)]
    pub entity: Option<String>,
    /// Single mode: event type.
    #[serde(default, rename = "type")]
    pub event_type: Option<String>,
    /// Single mode: attributes.
    #[serde(default)]
    pub attrs: Option<Map<String, Value>>,
    /// Attach explanations.
    #[serde(default)]
    pub explain: bool,
    /// Candidates mode.
    #[serde(default)]
    pub candidates: Option<Vec<CandidateIn>>,
    /// Candidates mode: k (default 10).
    #[serde(default)]
    pub k: Option<usize>,
}

/// Body of `POST /hql`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HqlReq {
    /// One statement of the query language.
    pub query: String,
    /// Clock for `now` (default: server time).
    #[serde(default)]
    pub now: Option<f64>,
}

/// Body of `PUT /heaps/{heap}`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateHeapReq {
    /// Half-life (seconds).
    pub half_life: f64,
    /// Landmark (default: now).
    #[serde(default)]
    pub landmark: Option<f64>,
    /// Metadata.
    #[serde(default)]
    pub meta: Option<Map<String, Value>>,
}

/// One item of `POST /heaps/{heap}/push`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PushIn {
    /// Item id.
    pub id: String,
    /// Weight (default 1).
    #[serde(default)]
    pub weight: Option<f64>,
    /// Time (default: now).
    #[serde(default)]
    pub time: Option<f64>,
    /// Replacement attributes.
    #[serde(default)]
    pub attrs: Option<Map<String, Value>>,
}

/// Body of `POST /heaps/{heap}/push`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PushReq {
    /// At most 1000 pushes, applied atomically in order.
    pub items: Vec<PushIn>,
}

/// Body of `POST /heaps/{heap}/base`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaseReq {
    /// At most 1000 items.
    pub items: Vec<BaseArgs>,
}

/// Body of `POST /heaps/{heap}/attrs`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttrsReq {
    /// At most 1000 items.
    pub items: Vec<AttrsArgs>,
}

/// Body of `POST /heaps/{heap}/top`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopReq {
    /// How many (default 10).
    #[serde(default)]
    pub k: Option<usize>,
    /// Clock (default: now).
    #[serde(default)]
    pub now: Option<f64>,
    /// Filter (`docs/heaps.md`).
    #[serde(default)]
    pub filter: Option<Value>,
}

/// Body of `POST /heaps/{heap}/pop`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PopReq {
    /// Clock (default: now).
    #[serde(default)]
    pub now: Option<f64>,
    /// Filter.
    #[serde(default)]
    pub filter: Option<Value>,
}

/// Body of `POST /heaps/{heap}/rebase`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RebaseReq {
    /// New landmark (default: now).
    #[serde(default)]
    pub now: Option<f64>,
    /// New half-life.
    #[serde(default)]
    pub half_life: Option<f64>,
}

/// A filter from a request body (validated; `bad_request` naming the node).
pub fn parse_filter(v: Option<&Value>) -> ApiResult<Option<Filter>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(v) => Filter::from_json(v)
            .map(Some)
            .map_err(|e| ApiError::invalid(e.to_string())),
    }
}

/// A mutation, as stored in the WAL.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", content = "args", rename_all = "snake_case")]
pub enum WalOp {
    /// First record of a new owner epoch (no-op; fences older epochs).
    EpochStart,
    /// `register_effect`.
    RegisterEffect(EffectReq),
    /// `register_event_type`.
    RegisterEventType(EventTypeReq),
    /// `add_rule`.
    AddRule(RuleReq),
    /// `insert` (batched).
    Insert(Vec<EventIn>),
    /// `feedback`.
    Feedback(FeedbackReq),
    /// `observe`.
    Observe(ObserveReq),
    /// `settle`.
    Settle(SettleReq),
    /// `forget_entity`.
    Forget {
        /// Entity.
        entity: String,
        /// Also subtract from the global layer.
        scrub_global: bool,
    },
    /// A keyed-heap mutation (all inputs resolved: no clock reads).
    Heap(HeapOp),
    /// A mutating query-language statement, with the clock it ran at.
    Hql {
        /// Statement.
        query: String,
        /// `now`.
        now: f64,
    },
}

/// A WAL record: one line of a WAL object.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WalRecord {
    /// Owner epoch that wrote it.
    pub epoch: u64,
    /// Database-wide sequence number.
    pub seq: u64,
    /// Wall clock (informational).
    pub ts: f64,
    /// The mutation.
    #[serde(flatten)]
    pub op: WalOp,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Declarative schema mirrored next to the engine (the engine cannot
/// serialise callables).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Schema {
    /// Event types in registration order.
    pub event_types: IndexMap<String, Option<WeightSpec>>,
    /// Rules per effect, in engine order.
    pub rules: IndexMap<String, Vec<RuleSpec>>,
}

/// Header line of a server snapshot.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SnapshotHeader {
    /// `heapdb_server`.
    pub kind: String,
    /// 1.
    pub format: u32,
    /// Database id.
    pub database_id: String,
    /// Epoch of the writer.
    pub epoch: u64,
    /// Last WAL seq included.
    pub seq: u64,
    /// Wall clock.
    pub created_at: f64,
    /// Schema.
    pub schema: Schema,
    /// external_id -> event id.
    pub external_ids: BTreeMap<String, u64>,
    /// Number of heap records that follow the header (format 2).
    #[serde(default)]
    pub heaps: u64,
}

/// Engine + schema + idempotency index + WAL position.
pub struct DbState {
    /// The engine.
    pub db: HeapDB,
    /// Declarative schema.
    pub schema: Schema,
    /// external_id -> event id.
    pub external_ids: BTreeMap<String, u64>,
    /// Keyed heaps.
    pub heaps: MemoryHeaps,
    /// Last applied WAL seq.
    pub seq: u64,
}

impl Default for DbState {
    fn default() -> Self {
        Self::new()
    }
}

fn ok() -> Value {
    json!({"ok": true})
}

impl DbState {
    /// Empty database.
    pub fn new() -> Self {
        DbState {
            db: HeapDB::new(BUCKET_SECONDS, MAX_SETS).expect("valid constants"),
            schema: Schema::default(),
            external_ids: BTreeMap::new(),
            heaps: MemoryHeaps::new(),
            seq: 0,
        }
    }

    /// Validate and pre-compute an insert batch: per event, `Some(existing
    /// id)` for a duplicate `external_id`, else `None`. Nothing changes.
    fn plan_insert(&self, events: &[EventIn]) -> ApiResult<Vec<Option<u64>>> {
        let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
        let mut out = Vec::with_capacity(events.len());
        for (i, e) in events.iter().enumerate() {
            if !e.time.is_finite() {
                return Err(ApiError::invalid(format!("events[{i}]: time must be finite")));
            }
            let attrs = e.attrs.clone().unwrap_or_default();
            let w = match e.weight {
                Some(w) => w,
                None => match self.schema.event_types.get(&e.event_type) {
                    Some(Some(spec)) => spec
                        .eval(&attrs)
                        .map_err(|m| ApiError::invalid(format!("events[{i}]: weight function: {m}")))?,
                    _ => 1.0,
                },
            };
            heapdb::event::validate_weight(w)
                .map_err(|err| ApiError::invalid(format!("events[{i}]: {err}")))?;
            let dup = match &e.external_id {
                Some(x) => {
                    if let Some(&id) = self.external_ids.get(x).filter(|&&id| self.db.contains(id)) {
                        Some(Some(id))
                    } else if seen.contains_key(x.as_str()) {
                        Some(None) // resolved after insertion
                    } else {
                        seen.insert(x, i);
                        None
                    }
                }
                None => None,
            };
            out.push(dup.flatten());
        }
        Ok(out)
    }

    /// Apply a mutation. Deterministic; on error nothing changed (except
    /// the documented case of a failure inside an insert batch, reported
    /// as `internal`, after which the caller must reload the database).
    pub fn apply(&mut self, op: &WalOp) -> ApiResult<Value> {
        match op {
            WalOp::EpochStart => Ok(ok()),
            WalOp::RegisterEffect(req) => {
                let opts = req.options()?;
                self.db.register_effect(&req.name, opts)?;
                self.schema.rules.shift_remove(&req.name);
                Ok(ok())
            }
            WalOp::RegisterEventType(req) => {
                if let Some(w) = &req.weight {
                    w.validate()?;
                }
                self.db
                    .register_event_type(&req.event_type, req.weight.as_ref().map(WeightSpec::weight_fn));
                self.schema
                    .event_types
                    .insert(req.event_type.clone(), req.weight.clone());
                Ok(ok())
            }
            WalOp::AddRule(req) => {
                if let Some(p) = &req.predicate {
                    p.validate()?;
                }
                let spec = RuleSpec {
                    name: req.name.clone(),
                    multiplier: req.multiplier,
                    types: req.types.clone(),
                    predicate: req.predicate.clone(),
                };
                self.db.add_rule(
                    &req.effect,
                    &req.name,
                    spec.predicate_fn(),
                    req.multiplier,
                    req.types.clone(),
                )?;
                let rules = self.schema.rules.entry(req.effect.clone()).or_default();
                rules.retain(|r| r.name != req.name);
                rules.push(spec);
                Ok(ok())
            }
            WalOp::Insert(events) => {
                let plan = self.plan_insert(events)?;
                let mut ids = Vec::with_capacity(events.len());
                let mut stored = Vec::with_capacity(events.len());
                for (e, dup) in events.iter().zip(plan) {
                    if let Some(id) = dup {
                        ids.push(id);
                        stored.push(json::event(&self.db.get(id)?));
                        continue;
                    }
                    if let Some(id) = e
                        .external_id
                        .as_ref()
                        .and_then(|x| self.external_ids.get(x))
                        .copied()
                        .filter(|&id| self.db.contains(id))
                    {
                        // Duplicate inside this batch.
                        ids.push(id);
                        stored.push(json::event(&self.db.get(id)?));
                        continue;
                    }
                    let ev = self
                        .db
                        .insert(NewEvent {
                            entity: e.entity.clone(),
                            time: e.time,
                            event_type: e.event_type.clone(),
                            features: e.features.clone(),
                            weight: e.weight,
                            attrs: e.attrs.clone().unwrap_or_default(),
                        })
                        .map_err(|err| {
                            ApiError::internal(format!("insert failed after validation: {err}"))
                        })?;
                    if let Some(x) = &e.external_id {
                        self.external_ids.insert(x.clone(), ev.id);
                    }
                    ids.push(ev.id);
                    stored.push(json::event(&ev));
                }
                Ok(json!({"ids": ids, "events": stored}))
            }
            WalOp::Feedback(r) => {
                let credit = r.credit.as_deref().map(Credit::parse).transpose()?;
                let rep = self.db.feedback(
                    &r.entity,
                    &r.effect,
                    r.at,
                    r.confirmed_event_id,
                    &r.rejected_ids,
                    credit,
                )?;
                Ok(json::feedback(&rep))
            }
            WalOp::Observe(r) => {
                let rep = self.db.observe(&r.entity, &r.effect, r.at, r.leak.map(|l| l.0))?;
                Ok(json::observe(&rep))
            }
            WalOp::Settle(r) => {
                let effs = r.effects.as_ref().map(OneOrMany::to_vec);
                let refs: Option<Vec<&str>> = effs.as_ref().map(|v| v.iter().map(String::as_str).collect());
                for e in refs.iter().flatten() {
                    self.db.effect(e)?;
                }
                let rep = self
                    .db
                    .settle(r.until, refs.as_deref(), r.entity.as_deref(), r.partial)?;
                Ok(json::settle(&rep))
            }
            WalOp::Forget { entity, scrub_global } => {
                let rep = self.db.forget_entity(entity, *scrub_global);
                let db = &self.db;
                self.external_ids.retain(|_, id| db.contains(*id));
                Ok(json!({"counts": json::forget(&rep)}))
            }
            WalOp::Heap(op) => Ok(self.heaps.apply(op)?),
            WalOp::Hql { query, now } => {
                let stmt = heapdb::parse(query)?;
                let out = heapdb::query::execute_with(&mut self.db, &mut self.heaps, &stmt, *now)?;
                Ok(json::output(&out))
            }
        }
    }

    // -- reads -----------------------------------------------------------------

    /// `POST /query/top-causes`.
    pub fn top_causes(&self, r: &TopCausesReq) -> ApiResult<Value> {
        let opts = TopK::new(r.k.unwrap_or(3)).explain(r.explain).within(r.within);
        let res = match (&r.entity, &r.entities) {
            (Some(e), None) => self.db.top_causes(e, &r.effect, r.at, opts)?,
            (None, Some(es)) => {
                let refs: Vec<&str> = es.iter().map(String::as_str).collect();
                self.db.top_causes_group(&refs, &r.effect, r.at, opts)?
            }
            _ => return Err(ApiError::bad_request("give exactly one of entity or entities")),
        };
        Ok(json::query_result(&res))
    }

    /// `POST /query/risk`.
    pub fn risk(&self, r: &RiskReq) -> ApiResult<Value> {
        match (&r.candidates, &r.features) {
            (Some(cands), None) => {
                let cs: Vec<Candidate> = cands
                    .iter()
                    .map(|c| Candidate {
                        features: c.features.clone(),
                        weight: c.weight,
                        event_type: c.event_type.clone(),
                        attrs: c.attrs.clone().unwrap_or_default(),
                        entity: c.entity.clone(),
                    })
                    .collect();
                let ranked = self.db.rank_candidates(
                    &r.effect,
                    &cs,
                    r.k.unwrap_or(10),
                    r.entity.as_deref(),
                    r.explain,
                )?;
                let rows: Vec<Value> = ranked
                    .iter()
                    .map(|rc| {
                        let c = &cands[rc.index];
                        let echo = json!({
                            "id": c.id,
                            "features": c.features,
                            "weight": c.weight,
                            "type": c.event_type,
                            "attrs": c.attrs,
                            "entity": c.entity,
                        });
                        json::ranked_candidate(rc, &echo)
                    })
                    .collect();
                Ok(json!({"candidates": rows}))
            }
            (None, Some(features)) => {
                let c = Candidate {
                    features: features.clone(),
                    weight: r.weight,
                    event_type: r.event_type.clone(),
                    attrs: r.attrs.clone().unwrap_or_default(),
                    entity: r.entity.clone(),
                };
                let (score, explanation) = self.db.feature_risk(&r.effect, &c, r.explain)?;
                let mut feats = features.clone();
                feats.sort();
                feats.dedup();
                Ok(json::risk_result(&RiskResult {
                    effect: r.effect.clone(),
                    entity: r.entity.clone(),
                    features: feats,
                    event_type: r.event_type.clone(),
                    score,
                    explanation,
                }))
            }
            _ => Err(ApiError::bad_request(
                "give exactly one of features or candidates",
            )),
        }
    }

    /// `GET /links`.
    pub fn links(&self, effect: &str, opts: &LinkReportOptions) -> ApiResult<Value> {
        Ok(json::link_report(&self.db.link_report(effect, opts)?))
    }

    /// `GET /calibration`.
    pub fn calibration(&self, effect: &str, bins: usize) -> ApiResult<Value> {
        Ok(json::calibration(&self.db.calibration_report(effect, bins)?))
    }

    /// A read-only query-language statement.
    pub fn hql_read(&self, stmt: &Statement, now: f64) -> ApiResult<Value> {
        let out = match stmt {
            Statement::Top(q) => {
                let opts = TopK::new(q.k).explain(q.explain).within(q.within);
                let at = q.at.resolve(now);
                if q.is_group() {
                    let ents: Vec<&str> = q.entities.iter().map(String::as_str).collect();
                    Output::Query(self.db.top_causes_group(&ents, &q.effect, at, opts)?)
                } else {
                    Output::Query(self.db.top_causes(&q.entity, &q.effect, at, opts)?)
                }
            }
            Statement::Calibration(q) => Output::Calibration(self.db.calibration_report(&q.effect, q.bins)?),
            Statement::Risk(q) => {
                let c = Candidate {
                    features: q.features.iter().cloned().collect(),
                    weight: q.weight,
                    event_type: q.event_type.clone(),
                    attrs: q.attrs.iter().cloned().collect(),
                    entity: Some(q.entity.clone()),
                };
                let (score, explanation) = self.db.feature_risk(&q.effect, &c, q.explain)?;
                Output::Risk(RiskResult {
                    effect: q.effect.clone(),
                    entity: Some(q.entity.clone()),
                    features: q.features.iter().cloned().collect(),
                    event_type: q.event_type.clone(),
                    score,
                    explanation,
                })
            }
            Statement::Links(q) => Output::Links(self.db.link_report(
                &q.effect,
                &LinkReportOptions {
                    entity: q.entity.clone(),
                    min_support: q.min_support,
                    ..Default::default()
                },
            )?),
            Statement::Heap(h) if !h.is_mutating() => {
                // PEEK / SHOW HEAPS only read; run them on a shared borrow.
                Output::Heap(read_heap_stmt(&self.heaps, h, now)?)
            }
            _ => return Err(ApiError::internal("hql_read called with a mutating statement")),
        };
        Ok(json::output(&out))
    }

    // -- heaps -----------------------------------------------------------------

    /// `GET /heaps/{heap}`.
    pub fn heap_info(&self, name: &str) -> ApiResult<Value> {
        match self.heaps.heap(name)? {
            Some(i) => Ok(i.to_json()),
            None => Err(heapdb::Error::UnknownHeap(name.to_string()).into()),
        }
    }

    /// `GET /heaps`.
    pub fn heap_list(&self) -> ApiResult<Value> {
        let v: Vec<Value> = self
            .heaps
            .heaps()?
            .iter()
            .map(heapdb::HeapInfo::to_json)
            .collect();
        Ok(json!({"heaps": v}))
    }

    /// `GET /heaps/{heap}/items/{item}`.
    pub fn heap_get(&self, heap: &str, id: &str, now: f64) -> ApiResult<Value> {
        match self.heaps.get(heap, id, now)? {
            Some(r) => Ok(r.to_json()),
            None => Err(ApiError::new(
                axum::http::StatusCode::NOT_FOUND,
                "unknown_item",
                format!(
                    "no item {} in heap {}",
                    heapdb::pyfmt::repr_str(id),
                    heapdb::pyfmt::repr_str(heap)
                ),
            )),
        }
    }

    /// `POST /heaps/{heap}/top`.
    pub fn heap_top(&self, heap: &str, k: usize, now: f64, filter: Option<&Filter>) -> ApiResult<Value> {
        let v: Vec<Value> = self
            .heaps
            .top_k(heap, k, now, filter)?
            .iter()
            .map(heapdb::Ranked::to_json)
            .collect();
        Ok(json!({"items": v}))
    }

    // -- snapshots ---------------------------------------------------------------

    /// Serialise (header line + heap records + engine snapshot).
    pub fn snapshot(&self, database_id: &str, epoch: u64, now: f64) -> Vec<u8> {
        let heap_records = self.heaps.to_records();
        let header = SnapshotHeader {
            kind: "heapdb_server".into(),
            format: SNAPSHOT_FORMAT,
            database_id: database_id.to_string(),
            epoch,
            seq: self.seq,
            created_at: now,
            schema: self.schema.clone(),
            external_ids: self.external_ids.clone(),
            heaps: heap_records.len() as u64,
        };
        let mut out = serde_json::to_vec(&header).expect("header serialises");
        out.push(b'\n');
        for r in &heap_records {
            serde_json::to_writer(&mut out, r).expect("heap records serialise");
            out.push(b'\n');
        }
        self.db.save_to(&mut out).expect("writing to memory cannot fail");
        out
    }

    /// Inverse of [`DbState::snapshot`]; re-registers weight functions and
    /// rules (in their original order).
    pub fn from_snapshot(bytes: &[u8]) -> Result<(DbState, SnapshotHeader), String> {
        let mut r = std::io::Cursor::new(bytes);
        let mut first = String::new();
        r.read_line(&mut first).map_err(|e| e.to_string())?;
        let header: SnapshotHeader =
            serde_json::from_str(first.trim()).map_err(|e| format!("snapshot header: {e}"))?;
        if header.kind != "heapdb_server" || !(1..=SNAPSHOT_FORMAT).contains(&header.format) {
            return Err(format!("unsupported snapshot {}/{}", header.kind, header.format));
        }
        let mut heaps = MemoryHeaps::new();
        for i in 0..header.heaps {
            let mut line = String::new();
            r.read_line(&mut line).map_err(|e| e.to_string())?;
            let rec: Map<String, Value> =
                serde_json::from_str(line.trim()).map_err(|e| format!("snapshot heap record {i}: {e}"))?;
            if !heaps.load_record(&rec).map_err(|e| e.to_string())? {
                return Err(format!("snapshot heap record {i}: unknown kind"));
            }
        }
        let mut db = HeapDB::load_from(r, "snapshot").map_err(|e| e.to_string())?;
        for (t, w) in &header.schema.event_types {
            db.register_event_type(t, w.as_ref().map(WeightSpec::weight_fn));
        }
        for (effect, rules) in &header.schema.rules {
            if db.effect(effect).is_err() {
                continue;
            }
            for r in rules {
                db.add_rule(effect, &r.name, r.predicate_fn(), r.multiplier, r.types.clone())
                    .map_err(|e| e.to_string())?;
            }
        }
        let state = DbState {
            db,
            schema: header.schema.clone(),
            external_ids: header.external_ids.clone(),
            heaps,
            seq: header.seq,
        };
        Ok((state, header))
    }
}

fn read_heap_stmt(h: &MemoryHeaps, stmt: &hql::HeapStmt, now: f64) -> ApiResult<hql::HeapOutput> {
    use hql::{HeapOutput, HeapStmt};
    Ok(match stmt {
        HeapStmt::Peek { heap, k, at, filter } => {
            let t = at.resolve(now);
            HeapOutput::Top(heapdb::heaps::HeapTop {
                heap: heap.clone(),
                now: t,
                popped: false,
                items: h.top_k(heap, *k, t, filter.as_ref())?,
            })
        }
        HeapStmt::Show => HeapOutput::List(h.heaps()?),
        _ => {
            return Err(ApiError::internal(
                "read_heap_stmt called with a mutating statement",
            ))
        }
    })
}

/// True for statements that change the database.
pub fn is_mutating(stmt: &Statement) -> bool {
    match stmt {
        Statement::Heap(h) => h.is_mutating(),
        _ => matches!(
            stmt,
            Statement::Insert(_) | Statement::Feedback(_) | Statement::Observe(_) | Statement::Settle(_)
        ),
    }
}

/// The API scope a statement needs: read for queries (TOP, RISK, LINKS,
/// CALIBRATION, PEEK, SHOW HEAPS), write for INSERT / FEEDBACK / OBSERVE /
/// SETTLE / PUSH / SET / POP / REMOVE, admin for CREATE / DROP / REBASE.
pub fn hql_scope(stmt: &Statement) -> crate::storage::Scope {
    use crate::storage::Scope;
    match stmt {
        Statement::Heap(h) => match h.access() {
            hql::Access::Read => Scope::Read,
            hql::Access::Write => Scope::Write,
            hql::Access::Admin => Scope::Admin,
        },
        s if is_mutating(s) => Scope::Write,
        _ => Scope::Read,
    }
}

/// Build the WAL op of a heap push batch, resolving defaults (weight 1,
/// time `now`).
pub fn push_op(heap: &str, items: Vec<PushIn>, now: f64) -> WalOp {
    WalOp::Heap(HeapOp::Push {
        heap: heap.to_string(),
        items: items
            .into_iter()
            .map(|p| PushArgs {
                id: p.id,
                weight: p.weight.unwrap_or(1.0),
                time: p.time.unwrap_or(now),
                attrs: p.attrs,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ops() -> Vec<WalOp> {
        let v = |x: Value| -> WalOp { serde_json::from_value(x).unwrap() };
        vec![
            v(
                json!({"op": "register_effect", "args": {"name": "bloating", "kernel": {"type": "lognormal", "params": {"median": 7200, "sigma": 0.6}}, "priors": {"beans": [3, 1]}}}),
            ),
            v(
                json!({"op": "register_event_type", "args": {"type": "meal", "weight": {"kind": "attr", "attr": "portion", "value": 1.0}}}),
            ),
            v(
                json!({"op": "add_rule", "args": {"effect": "bloating", "name": "small", "multiplier": 0.5, "predicate": {"attr": "portion", "op": "lt", "value": 0.5}}}),
            ),
            v(
                json!({"op": "add_rule", "args": {"effect": "bloating", "name": "snacks", "multiplier": 0.8, "types": ["snack"]}}),
            ),
            v(json!({"op": "insert", "args": [
                {"entity": "me", "time": 0, "type": "meal", "features": ["beans", "onion"], "attrs": {"portion": 2}, "external_id": "a"},
                {"entity": "me", "time": 3600, "type": "meal", "features": ["dairy"], "attrs": {"portion": 0.25}},
                {"entity": "me", "time": 3700, "type": "snack", "features": ["chips"]},
                {"entity": "me", "time": 0, "type": "meal", "features": ["beans"], "external_id": "a"}
            ]})),
            v(
                json!({"op": "feedback", "args": {"entity": "me", "effect": "bloating", "at": 7200, "confirmed_event_id": 1, "rejected_ids": [2]}}),
            ),
            v(json!({"op": "observe", "args": {"entity": "me", "effect": "bloating", "at": 9000}})),
            v(
                json!({"op": "hql", "args": {"query": "INSERT meal FOR you AT now-1h FEATURES beans", "now": 10000}}),
            ),
        ]
    }

    #[test]
    fn apply_snapshot_round_trip() {
        let mut s = DbState::new();
        for (i, op) in ops().iter().enumerate() {
            let r = s.apply(op).unwrap();
            s.seq = i as u64 + 1;
            if let WalOp::Insert(_) = op {
                assert_eq!(
                    r["ids"],
                    json!([1, 2, 3, 1]),
                    "external_id dedupes inside a batch"
                );
            }
        }
        // Dedupe across requests too.
        let again: WalOp = serde_json::from_value(json!({"op": "insert", "args": [
            {"entity": "me", "time": 0, "type": "meal", "external_id": "a"}]}))
        .unwrap();
        assert_eq!(s.apply(&again).unwrap()["ids"], json!([1]));
        let q = TopCausesReq {
            entity: Some("me".into()),
            entities: None,
            effect: "bloating".into(),
            at: 7300.0,
            k: Some(3),
            within: None,
            explain: true,
        };
        let before = s.top_causes(&q).unwrap();
        let bytes = s.snapshot("db_x", 1, 0.0);
        let (s2, h) = DbState::from_snapshot(&bytes).unwrap();
        assert_eq!(h.seq, s.seq);
        assert_eq!(s2.top_causes(&q).unwrap(), before);
        assert_eq!(s2.schema, s.schema);
        assert_eq!(s2.external_ids, s.external_ids);
        // Replaying the same ops gives the same snapshot.
        let mut s3 = DbState::new();
        for (i, op) in ops().iter().enumerate() {
            s3.apply(op).unwrap();
            s3.seq = i as u64 + 1;
        }
        s3.apply(&again).unwrap();
        assert_eq!(s3.snapshot("db_x", 1, 0.0), bytes);
    }

    fn heap_ops() -> Vec<WalOp> {
        let v = |x: Value| -> WalOp { serde_json::from_value(x).unwrap() };
        vec![
            v(
                json!({"op": "heap", "args": {"op": "create_heap", "args": {"name": "acme/q", "half_life": 3600, "landmark": 1000, "now": 5}}}),
            ),
            v(
                json!({"op": "heap", "args": {"op": "push", "args": {"heap": "acme/q", "items": [
                    {"id": "a", "weight": 2, "time": 1000, "attrs": {"kind": "bug"}},
                    {"id": "b", "weight": 1, "time": 1500},
                    {"id": "a", "weight": 1, "time": 2000}
                ]}}}),
            ),
            v(
                json!({"op": "heap", "args": {"op": "set_base", "args": {"heap": "acme/q", "items": [{"id": "c", "base": 3}]}}}),
            ),
            v(
                json!({"op": "heap", "args": {"op": "set_attrs", "args": {"heap": "acme/q", "items": [{"id": "b", "attrs": {"kind": "task"}, "merge": true}]}}}),
            ),
            v(
                json!({"op": "hql", "args": {"query": "PUSH d INTO 'acme/q' WEIGHT 5 AT now-1h ATTRS {\"kind\": \"bug\"}", "now": 9000}}),
            ),
            v(
                json!({"op": "heap", "args": {"op": "pop", "args": {"heap": "acme/q", "now": 9000, "filter": {"attr": "kind", "op": "eq", "value": "bug"}}}}),
            ),
            v(json!({"op": "heap", "args": {"op": "rebase", "args": {"heap": "acme/q", "now": 9500}}})),
        ]
    }

    #[test]
    fn heaps_replay_and_snapshot() {
        let mut s = DbState::new();
        for (i, op) in ops().iter().chain(heap_ops().iter()).enumerate() {
            s.apply(op).unwrap();
            s.seq = i as u64 + 1;
        }
        let top = s.heap_top("acme/q", 10, 9600.0, None).unwrap();
        let ids: Vec<&str> = top["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["a", "b"], "d was popped (best bug), c is unranked");
        let bytes = s.snapshot("db_x", 1, 0.0);
        let text = String::from_utf8(bytes.clone()).unwrap();
        assert!(text.starts_with("{\"kind\":\"heapdb_server\",\"format\":2"));
        let (s2, h) = DbState::from_snapshot(&bytes).unwrap();
        assert_eq!(h.format, 2);
        assert_eq!(s2.heaps.to_records(), s.heaps.to_records());
        assert_eq!(s2.heap_top("acme/q", 10, 9600.0, None).unwrap(), top);
        // A replay gives the identical snapshot.
        let mut s3 = DbState::new();
        for (i, op) in ops().iter().chain(heap_ops().iter()).enumerate() {
            s3.apply(op).unwrap();
            s3.seq = i as u64 + 1;
        }
        assert_eq!(s3.snapshot("db_x", 1, 0.0), bytes);
        // A failing heap batch changes nothing (and is not a WAL record).
        let bad: WalOp = serde_json::from_value(
            json!({"op": "heap", "args": {"op": "push", "args": {"heap": "acme/q", "items": [
            {"id": "x", "weight": 1, "time": 0}, {"id": "y", "weight": 0, "time": 0}]}}}),
        )
        .unwrap();
        let e = s.apply(&bad).unwrap_err();
        assert_eq!(e.code, "invalid_argument");
        assert!(e.message.contains("items[1]"), "{}", e.message);
        assert_eq!(s.heaps.to_records(), s2.heaps.to_records());
    }

    #[test]
    fn format_1_snapshots_still_load() {
        // Written by heapdb-server 0.5 (DbState::snapshot before heaps).
        let bytes = include_bytes!("../tests/fixtures/snapshot_format1.jsonl");
        assert!(bytes.starts_with(b"{\"kind\":\"heapdb_server\",\"format\":1,"));
        let (s, h) = DbState::from_snapshot(bytes).unwrap();
        assert_eq!((h.format, h.seq, h.heaps), (1, 5, 0));
        assert!(s.heaps.is_empty());
        assert_eq!(s.db.len(), 2);
        assert_eq!(s.external_ids.get("a"), Some(&1));
        assert_eq!(s.schema.rules["bloating"][0].name, "small");
        let q = TopCausesReq {
            entity: Some("me".into()),
            entities: None,
            effect: "bloating".into(),
            at: 7300.0,
            k: Some(3),
            within: None,
            explain: true,
        };
        let r = s.top_causes(&q).unwrap();
        assert_eq!(r["causes"][0]["event"]["id"], json!(1));
        assert_eq!(r["causes"][1]["explanation"]["rules"][0]["name"], json!("small"));
        // Re-saved, it is format 2 and loads again.
        let again = s.snapshot("db_fixture0000000000000", 3, 0.0);
        let (s2, h2) = DbState::from_snapshot(&again).unwrap();
        assert_eq!(h2.format, 2);
        assert_eq!(s2.top_causes(&q).unwrap(), r);
    }

    #[test]
    fn hql_scopes() {
        use crate::storage::Scope;
        for (q, want) in [
            ("TOP 3 CAUSES OF a FOR b", Scope::Read),
            ("INSERT meal FOR me", Scope::Write),
            ("PEEK 3 FROM w", Scope::Read),
            ("SHOW HEAPS", Scope::Read),
            ("PUSH a INTO w WEIGHT 1", Scope::Write),
            ("SET BASE a IN w TO 2", Scope::Write),
            ("POP FROM w", Scope::Write),
            ("REMOVE a FROM w", Scope::Write),
            ("CREATE HEAP w HALF LIFE 1h", Scope::Admin),
            ("DROP HEAP w", Scope::Admin),
            ("REBASE w", Scope::Admin),
        ] {
            assert_eq!(hql_scope(&heapdb::parse(q).unwrap()), want, "{q}");
        }
    }

    #[test]
    fn insert_batch_is_atomic() {
        let mut s = DbState::new();
        let op: WalOp = serde_json::from_value(json!({"op": "insert", "args": [
            {"entity": "me", "time": 0, "type": "meal"},
            {"entity": "me", "time": 1, "type": "meal", "weight": -1}
        ]}))
        .unwrap();
        let e = s.apply(&op).unwrap_err();
        assert_eq!(e.code, "invalid_argument");
        assert_eq!(s.db.len(), 0);
    }
}
