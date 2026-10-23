//! [`HeapDB`]: the store, the top-k attribution query, feedback and
//! calibration.
//!
//! # Scoring model
//!
//! For an effect `E` observed for entity `x` at time `T` and a candidate
//! event `e` of `x` (time `t`, weight `w`, features `F`, type `y`):
//!
//! ```text
//! score(e | E) = kernel_E(T - t) * w * link_E(F) * rules_E(e)
//! ```
//!
//! `link_E(F)` is `noisy_or_{f in F}(strength_x(f, E))` (default) or, for an
//! effect registered with `combine = logistic`, `sigmoid(b + sum_f w_f)`
//! (see [`crate::logistic`]).
//!
//! # Top-k query (threshold-algorithm style)
//!
//! 1. Only segments overlapping `[T - support, T - min_lag]` can contribute.
//! 2. Each segment gets an upper bound on the score of *any* of its events,
//!    from its summaries and the *current* links: the kernel maximum over
//!    the segment's lag range times the max over its distinct
//!    `(type, feature set)` keys of `max_weight * link * rule_bound`
//!    (or the looser union bound when the segment tracks too many keys:
//!    noisy-OR of all its features, or `sigmoid(b + sum max(0, w_f))`).
//! 3. Segments are visited best-bound-first, scoring events into a bounded
//!    min-heap of size `k`; as soon as the heap is full and its minimum is
//!    `>=` the next bound, the scan stops.
//!
//! The result is exactly the brute-force top-k (ties broken by the later
//! event id).

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::Arc;

use indexmap::IndexMap;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::error::{Error, Result};
use crate::event::{validate_weight, Attrs, Event, EventRef, Features, EMPTY_ATTRS};
use crate::intern::Vocab;
use crate::kernels::Kernel;
use crate::links::{BetaCounts, Combine, Credit, EntityId, EntityRef, FeatureId, LinkTable, StrengthCache};
use crate::logistic::LogisticParams;
use crate::pyfmt::{repr_f64, repr_str, repr_str_list};
use crate::results::*;
use crate::rules::{Predicate, Rule, WeightFn};
use crate::segments::{EntityLog, Segment, StoredEvent, TypeId, DEFAULT_MAX_SETS};

/// Relative slack applied to segment bounds before comparing them with
/// real scores (guards against last-bit floating point differences).
pub const BOUND_SLACK: f64 = 1e-9;

/// Default background (leak) score of an effect for `observe`.
pub const DEFAULT_LEAK: f64 = 0.05;

/// Snapshot format version written by [`HeapDB::save`].
pub const SNAPSHOT_VERSION: u32 = 3;

/// Names of the observational-learning counters, in report order.
pub const OBS_STAT_KEYS: [&str; 10] = [
    "occurrences",
    "leak_responsibility",
    "credited_mass",
    "events_settled",
    "events_rejected",
    "rejected_weight",
    "skipped_credited",
    "skipped_observed",
    "skipped_terminal",
    "rejections_undone",
];

/// Per-effect state of observational learning.
///
/// * `occurrences`: entity -> sorted times the effect was observed.
/// * `credited`: event id -> total soft responsibility from `observe`.
/// * `rejected`: event id -> negative evidence added by `settle`.
/// * `cursors`: entity -> settled event-time watermark; `late`: entity ->
///   ids of events inserted behind the cursor.
/// * `stats`: running counters.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationState {
    /// Entity -> sorted occurrence times.
    pub occurrences: BTreeMap<String, Vec<f64>>,
    /// Event id -> soft responsibility received.
    pub credited: FxHashMap<u64, f64>,
    /// Event id -> negative evidence added.
    pub rejected: FxHashMap<u64, f64>,
    /// Entity -> settle cursor.
    pub cursors: BTreeMap<String, f64>,
    /// Entity -> late event ids.
    pub late: BTreeMap<String, Vec<u64>>,
    /// Running counters (see [`OBS_STAT_KEYS`]).
    pub stats: IndexMap<String, f64>,
}

impl Default for ObservationState {
    fn default() -> Self {
        ObservationState {
            occurrences: BTreeMap::new(),
            credited: FxHashMap::default(),
            rejected: FxHashMap::default(),
            cursors: BTreeMap::new(),
            late: BTreeMap::new(),
            stats: OBS_STAT_KEYS.iter().map(|k| (k.to_string(), 0.0)).collect(),
        }
    }
}

impl ObservationState {
    pub(crate) fn bump(&mut self, key: &str, by: f64) {
        *self.stats.entry(key.to_string()).or_insert(0.0) += by;
    }
}

/// Everything HeapDB knows about one effect type.
#[derive(Clone, Debug)]
pub struct EffectSpec {
    pub(crate) name: String,
    pub(crate) kernel: Kernel,
    pub(crate) links: LinkTable,
    pub(crate) cause_types: Option<BTreeSet<String>>,
    pub(crate) cause_type_ids: Option<FxHashSet<TypeId>>,
    pub(crate) rules: Vec<Rule>,
    pub(crate) credit: Credit,
    pub(crate) pending_rules: Vec<String>,
    pub(crate) leak: f64,
    pub(crate) w_neg: f64,
    pub(crate) repeatable: bool,
    pub(crate) obs: ObservationState,
}

impl EffectSpec {
    /// Effect name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Lag kernel.
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }
    /// Replace the lag kernel, keeping learned links.
    pub fn set_kernel(&mut self, kernel: Kernel) {
        self.kernel = kernel;
    }
    /// Learned links.
    pub fn links(&self) -> &LinkTable {
        &self.links
    }
    /// Allowed cause types (`None`: any type).
    pub fn cause_types(&self) -> Option<&BTreeSet<String>> {
        self.cause_types.as_ref()
    }
    /// Plausibility rules, in application order.
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }
    /// Credit mode for confirmations.
    pub fn credit(&self) -> Credit {
        self.credit
    }
    /// How features are combined into the link.
    pub fn combine(&self) -> Combine {
        self.links.combine()
    }
    /// Logistic hyperparameters (`None` under `noisy_or`).
    pub fn logistic_params(&self) -> Option<LogisticParams> {
        self.links.logistic().map(|l| l.params())
    }
    /// Change the credit mode.
    pub fn set_credit(&mut self, c: Credit) {
        self.credit = c;
    }
    /// Predicate rules named in a snapshot that must be re-registered.
    pub fn pending_rules(&self) -> &[String] {
        &self.pending_rules
    }
    /// Background score for `observe`.
    pub fn leak(&self) -> f64 {
        self.leak
    }
    /// Set the leak (> 0).
    pub fn set_leak(&mut self, leak: f64) -> Result<()> {
        self.leak = check_leak(leak)?;
        Ok(())
    }
    /// Negative evidence per feature added by `settle`.
    pub fn w_neg(&self) -> f64 {
        self.w_neg
    }
    /// Set `w_neg` (finite, >= 0).
    pub fn set_w_neg(&mut self, w: f64) -> Result<()> {
        self.w_neg = check_w_neg(w)?;
        Ok(())
    }
    /// False for one-shot effects.
    pub fn repeatable(&self) -> bool {
        self.repeatable
    }
    /// Mark the effect repeatable or one-shot.
    pub fn set_repeatable(&mut self, r: bool) {
        self.repeatable = r;
    }
    /// Observational-learning state.
    pub fn observations(&self) -> &ObservationState {
        &self.obs
    }

    /// True if events of this (interned) type may be causes.
    pub(crate) fn type_allowed_id(&self, ty: Option<TypeId>) -> bool {
        match &self.cause_type_ids {
            None => true,
            Some(s) => ty.is_some_and(|t| s.contains(&t)),
        }
    }

    /// True if events of type `name` may be causes.
    pub fn type_allowed(&self, name: &str) -> bool {
        self.cause_types.as_ref().is_none_or(|s| s.contains(name))
    }

    /// Upper bound of the rule product for any event of type `ty`.
    pub(crate) fn rule_bound_id(&self, ty: Option<TypeId>) -> f64 {
        let mut b = 1.0;
        for r in &self.rules {
            if r.predicate.is_none() && r.applies_to_type_id(ty) {
                b *= r.multiplier;
            }
        }
        b
    }

    /// Upper bound of the rule product for any event of type `name`.
    pub fn rule_bound(&self, name: &str) -> f64 {
        let mut b = 1.0;
        for r in &self.rules {
            if r.predicate.is_none() && r.applies_to_type(name) {
                b *= r.multiplier;
            }
        }
        b
    }

    /// Product of the multipliers of the rules matching an event.
    pub(crate) fn rule_multiplier_ref(&self, ty: Option<TypeId>, ev: &EventRef<'_>) -> Result<f64> {
        let mut m = 1.0;
        for r in &self.rules {
            if m == 0.0 {
                break;
            }
            if r.matches(ty, ev).map_err(Error::Callback)? {
                m *= r.multiplier;
            }
        }
        Ok(m)
    }
}

/// Options for [`HeapDB::register_effect`].
#[derive(Clone, Debug)]
pub struct EffectOptions {
    /// Lag kernel.
    pub kernel: Kernel,
    /// Feature -> Beta(alpha, beta) prior.
    pub priors: Vec<(String, (f64, f64))>,
    /// Prior of features without one (default Beta(1, 4)).
    pub default_prior: (f64, f64),
    /// Restrict causes to these event types (`None` or empty: any).
    pub cause_types: Option<Vec<String>>,
    /// Credit mode for confirmations (`None`: the combine mode's default,
    /// [`Combine::default_credit`]).
    pub credit: Option<Credit>,
    /// How features are combined into the link (default `noisy_or`).
    pub combine: Combine,
    /// Hyperparameters of the `logistic` combine mode (ignored otherwise).
    pub logistic: LogisticParams,
    /// Cap on pseudo-observations borrowed from the global layer.
    pub global_pseudocount: f64,
    /// Borrow from the global layer at all.
    pub hierarchical: bool,
    /// Background score for `observe` (> 0; `inf` disables learning).
    pub leak: f64,
    /// Negative evidence per feature added by `settle` (>= 0).
    pub w_neg: f64,
    /// False for one-shot effects (churn, first conversion).
    pub repeatable: bool,
}

impl EffectOptions {
    /// Defaults for a kernel.
    pub fn new(kernel: Kernel) -> Self {
        EffectOptions {
            kernel,
            priors: Vec::new(),
            default_prior: (1.0, 4.0),
            cause_types: None,
            credit: None,
            combine: Combine::NoisyOr,
            logistic: LogisticParams::default(),
            global_pseudocount: 5.0,
            hierarchical: true,
            leak: DEFAULT_LEAK,
            w_neg: 1.0,
            repeatable: true,
        }
    }
    /// Add a feature prior.
    pub fn prior(mut self, feature: impl Into<String>, alpha: f64, beta: f64) -> Self {
        self.priors.push((feature.into(), (alpha, beta)));
        self
    }
    /// Set the default prior.
    pub fn default_prior(mut self, alpha: f64, beta: f64) -> Self {
        self.default_prior = (alpha, beta);
        self
    }
    /// Restrict cause types.
    pub fn cause_types<I: IntoIterator<Item = S>, S: Into<String>>(mut self, types: I) -> Self {
        self.cause_types = Some(types.into_iter().map(Into::into).collect());
        self
    }
    /// Set the credit mode.
    pub fn credit(mut self, c: Credit) -> Self {
        self.credit = Some(c);
        self
    }
    /// Set the combine mode.
    pub fn combine(mut self, c: Combine) -> Self {
        self.combine = c;
        self
    }
    /// Set the logistic learning rate (FTRL `alpha`, > 0).
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.logistic.learning_rate = lr;
        self
    }
    /// Set the logistic L2 penalty towards the seeds (>= 0).
    pub fn l2(mut self, l2: f64) -> Self {
        self.logistic.l2 = l2;
        self
    }
    /// Set the global pseudocount.
    pub fn global_pseudocount(mut self, c: f64) -> Self {
        self.global_pseudocount = c;
        self
    }
    /// Enable/disable borrowing.
    pub fn hierarchical(mut self, h: bool) -> Self {
        self.hierarchical = h;
        self
    }
    /// Set the leak.
    pub fn leak(mut self, l: f64) -> Self {
        self.leak = l;
        self
    }
    /// Set `w_neg`.
    pub fn w_neg(mut self, w: f64) -> Self {
        self.w_neg = w;
        self
    }
    /// Mark one-shot.
    pub fn repeatable(mut self, r: bool) -> Self {
        self.repeatable = r;
        self
    }
}

/// An event to insert (see [`HeapDB::insert`]).
#[derive(Clone, Debug)]
pub struct NewEvent {
    /// Entity.
    pub entity: String,
    /// Time in seconds.
    pub time: f64,
    /// Event type.
    pub event_type: String,
    /// Features (any order; duplicates collapse).
    pub features: Vec<String>,
    /// Dose; `None` derives it from `attrs` with the type's weight function
    /// (else 1.0).
    pub weight: Option<f64>,
    /// Attributes.
    pub attrs: Attrs,
}

impl NewEvent {
    /// An event without features, weight or attributes.
    pub fn new(entity: impl Into<String>, time: f64, event_type: impl Into<String>) -> Self {
        NewEvent {
            entity: entity.into(),
            time,
            event_type: event_type.into(),
            features: Vec::new(),
            weight: None,
            attrs: Attrs::new(),
        }
    }
    /// Set the features.
    pub fn features<I: IntoIterator<Item = S>, S: Into<String>>(mut self, f: I) -> Self {
        self.features = f.into_iter().map(Into::into).collect();
        self
    }
    /// Set the dose.
    pub fn weight(mut self, w: f64) -> Self {
        self.weight = Some(w);
        self
    }
    /// Set the attributes.
    pub fn attrs(mut self, a: Attrs) -> Self {
        self.attrs = a;
        self
    }
}

/// Options for [`HeapDB::top_causes`].
#[derive(Clone, Copy, Debug)]
pub struct TopK {
    /// Number of causes (>= 1).
    pub k: usize,
    /// Attach explanations.
    pub explain: bool,
    /// Extra lookback cap in seconds.
    pub within: Option<f64>,
    /// Use the per-set bound (false: v0.1 union bound; for experiments).
    pub use_sets: bool,
}

impl TopK {
    /// Top `k`, no explanations.
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            explain: false,
            within: None,
            use_sets: true,
        }
    }
    /// Attach explanations.
    pub fn explain(mut self, e: bool) -> Self {
        self.explain = e;
        self
    }
    /// Cap the lookback.
    pub fn within(mut self, w: Option<f64>) -> Self {
        self.within = w;
        self
    }
}

/// A candidate that has *not happened yet* (for [`HeapDB::feature_risk`]
/// and [`HeapDB::rank_candidates`]).
#[derive(Clone, Debug, Default)]
pub struct Candidate {
    /// Features.
    pub features: Vec<String>,
    /// Dose (`None`: derived from `attrs` via the type's weight function,
    /// else 1.0).
    pub weight: Option<f64>,
    /// Event type (enables `cause_types` and type-scoped rules).
    pub event_type: Option<String>,
    /// Attributes visible to predicates and the weight function.
    pub attrs: Attrs,
    /// Whose links to use (`None`: the global view, or the default entity
    /// passed to `rank_candidates`).
    pub entity: Option<String>,
}

impl Candidate {
    /// A candidate with these features and dose 1.
    pub fn new<I: IntoIterator<Item = S>, S: Into<String>>(features: I) -> Self {
        Candidate {
            features: features.into_iter().map(Into::into).collect(),
            weight: Some(1.0),
            ..Default::default()
        }
    }
}

/// Resolved summaries of one segment (see [`HeapDB::segment_info`]).
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentInfo {
    /// Bucket number.
    pub bucket: i64,
    /// Bucket start.
    pub start: f64,
    /// Bucket width.
    pub width: f64,
    /// Earliest event time.
    pub min_time: f64,
    /// Latest event time.
    pub max_time: f64,
    /// Largest weight.
    pub max_weight: f64,
    /// Event ids in insertion order.
    pub event_ids: Vec<u64>,
    /// Feature -> number of events with it (sorted by feature).
    pub feature_counts: Vec<(String, u32)>,
    /// Event types present (first-seen order).
    pub types: Vec<String>,
    /// `(type, sorted features, max weight)` keys, or `None` on overflow.
    pub sets: Option<Vec<(String, Vec<String>, f64)>>,
    /// Key cap.
    pub max_sets: usize,
}

/// What [`HeapDB::forget_entity`] removed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ForgetReport {
    /// Events removed.
    pub events: usize,
    /// Segments removed.
    pub segments: usize,
    /// Entity-level link entries removed (summed over effects).
    pub links: usize,
    /// Calibration records removed.
    pub calibration_records: usize,
}

/// An event being scored that need not be stored (a user-supplied event or
/// a not-yet-happened candidate).
pub(crate) struct Probe<'a> {
    pub id: u64,
    pub entity: &'a str,
    pub time: f64,
    pub ty: Option<TypeId>,
    pub ty_name: &'a str,
    /// Known features sorted by id, then unknown (`None`) ones.
    pub feats: Vec<Option<FeatureId>>,
    /// All feature names, sorted.
    pub names: &'a [String],
    pub weight: f64,
    pub attrs: &'a Attrs,
}

impl<'a> Probe<'a> {
    fn evref(&self) -> EventRef<'_> {
        EventRef {
            id: self.id,
            entity: self.entity,
            time: self.time,
            event_type: self.ty_name,
            features: Features::Names(self.names),
            weight: self.weight,
            attrs: self.attrs,
        }
    }

    fn known_ids(&self) -> impl Iterator<Item = FeatureId> + '_ {
        self.feats.iter().filter_map(|f| *f)
    }
}

/// An inseparable feature group of an event: member ids (sorted by id,
/// unknown last), member names (sorted) and the number of events shared.
pub(crate) struct Group {
    ids: Vec<Option<FeatureId>>,
    names: Vec<String>,
    n: usize,
}

#[derive(Clone, Copy)]
struct HeapItem {
    score: f64,
    id: u64,
    ord: u32,
}

impl PartialEq for HeapItem {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score.total_cmp(&o.score).then(self.id.cmp(&o.id))
    }
}

/// An append-only event store answering "what most likely caused this?".
pub struct HeapDB {
    pub(crate) bucket_seconds: f64,
    pub(crate) max_sets: usize,
    pub(crate) vocab: Vocab,
    pub(crate) logs: IndexMap<EntityId, EntityLog, FxBuildHasher>,
    /// `(event id, entity, ordinal)`, sorted by id.
    pub(crate) id_index: Vec<(u64, EntityId, u32)>,
    pub(crate) effects: IndexMap<String, EffectSpec>,
    pub(crate) event_types: BTreeMap<String, Option<Arc<dyn WeightFn>>>,
    pub(crate) calibration: IndexMap<String, Vec<CalibrationRecord>>,
    pub(crate) pending_event_types: BTreeSet<String>,
    pub(crate) next_id: u64,
}

impl std::fmt::Debug for HeapDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.repr())
    }
}

impl HeapDB {
    /// A new database. `bucket_seconds` is the width of the time buckets
    /// each entity's log is split into (smaller: tighter bounds, more
    /// segments); `max_sets_per_segment` caps the distinct
    /// `(type, feature set)` keys a segment tracks (0: always the union
    /// bound).
    pub fn new(bucket_seconds: f64, max_sets_per_segment: usize) -> Result<Self> {
        if bucket_seconds.is_nan() || bucket_seconds <= 0.0 {
            return Err(Error::invalid("bucket_seconds must be > 0"));
        }
        Ok(HeapDB {
            bucket_seconds,
            max_sets: max_sets_per_segment,
            vocab: Vocab::default(),
            logs: IndexMap::default(),
            id_index: Vec::new(),
            effects: IndexMap::new(),
            event_types: BTreeMap::new(),
            calibration: IndexMap::new(),
            pending_event_types: BTreeSet::new(),
            next_id: 1,
        })
    }

    /// Bucket width.
    pub fn bucket_seconds(&self) -> f64 {
        self.bucket_seconds
    }

    /// Key cap per segment.
    pub fn max_sets_per_segment(&self) -> usize {
        self.max_sets
    }

    /// Next event id.
    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    /// The interned vocabularies.
    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    // -- schema -----------------------------------------------------------

    /// Declare (or replace) an effect type, its lag kernel and link priors.
    pub fn register_effect(&mut self, name: &str, opts: EffectOptions) -> Result<&EffectSpec> {
        let leak = check_leak(opts.leak)?;
        let w_neg = check_w_neg(opts.w_neg)?;
        let mut links = LinkTable::new(opts.default_prior, opts.global_pseudocount, opts.hierarchical)?;
        if opts.combine == Combine::Logistic {
            links.set_combine(Combine::Logistic, Some(opts.logistic))?;
        }
        for (f, (a, b)) in &opts.priors {
            let id = self.vocab.features.intern(f);
            links.set_prior(id, f, *a, *b)?;
        }
        let cause_types: Option<BTreeSet<String>> = opts
            .cause_types
            .filter(|c| !c.is_empty())
            .map(|c| c.into_iter().collect());
        let cause_type_ids = cause_types
            .as_ref()
            .map(|c| c.iter().map(|t| self.vocab.types.intern(t)).collect());
        let spec = EffectSpec {
            name: name.to_string(),
            kernel: opts.kernel,
            links,
            cause_types,
            cause_type_ids,
            rules: Vec::new(),
            credit: opts.credit.unwrap_or(opts.combine.default_credit()),
            pending_rules: Vec::new(),
            leak,
            w_neg,
            repeatable: opts.repeatable,
            obs: ObservationState::default(),
        };
        self.effects.insert(name.to_string(), spec);
        Ok(&self.effects[name])
    }

    /// The spec of an effect.
    pub fn effect(&self, name: &str) -> Result<&EffectSpec> {
        self.effects
            .get(name)
            .ok_or_else(|| Error::UnknownEffect(name.to_string()))
    }

    /// The spec of an effect, mutably.
    pub fn effect_mut(&mut self, name: &str) -> Result<&mut EffectSpec> {
        self.effects
            .get_mut(name)
            .ok_or_else(|| Error::UnknownEffect(name.to_string()))
    }

    /// The link table of an effect plus the vocabulary (for low-level
    /// link manipulation).
    pub fn links_mut(&mut self, effect: &str) -> Result<(&mut LinkTable, &mut Vocab)> {
        let spec = self
            .effects
            .get_mut(effect)
            .ok_or_else(|| Error::UnknownEffect(effect.to_string()))?;
        Ok((&mut spec.links, &mut self.vocab))
    }

    /// Registered effect names, sorted.
    pub fn effects(&self) -> Vec<String> {
        let mut v: Vec<String> = self.effects.keys().cloned().collect();
        v.sort();
        v
    }

    /// Effect names in registration order.
    pub fn effects_in_order(&self) -> impl Iterator<Item = &str> {
        self.effects.keys().map(String::as_str)
    }

    /// Declare an event type, optionally deriving weight from `attrs`.
    pub fn register_event_type(&mut self, event_type: &str, weight: Option<Arc<dyn WeightFn>>) {
        self.event_types.insert(event_type.to_string(), weight);
        self.pending_event_types.remove(event_type);
    }

    /// Registered event types and their weight functions.
    pub fn event_types(&self) -> &BTreeMap<String, Option<Arc<dyn WeightFn>>> {
        &self.event_types
    }

    /// Event types with weight functions named in a loaded snapshot that
    /// have not been re-registered.
    pub fn pending_event_types(&self) -> &BTreeSet<String> {
        &self.pending_event_types
    }

    /// Add (or replace, by name) a plausibility rule. `multiplier` must be
    /// in `[0, 1]`. `predicate = None` makes a type-level rule.
    pub fn add_rule(
        &mut self,
        effect: &str,
        name: &str,
        predicate: Option<Arc<dyn Predicate>>,
        multiplier: f64,
        types: Option<Vec<String>>,
    ) -> Result<&Rule> {
        if !self.effects.contains_key(effect) {
            return Err(Error::UnknownEffect(effect.to_string()));
        }
        if !(0.0..=1.0).contains(&multiplier) {
            return Err(Error::invalid(format!(
                "rule multiplier must be in [0, 1] (bounds rely on it), got {}",
                repr_f64(multiplier)
            )));
        }
        let types: Option<BTreeSet<String>> = types.map(|t| t.into_iter().collect());
        let type_ids = types
            .as_ref()
            .map(|t| t.iter().map(|x| self.vocab.types.intern(x)).collect());
        let rule = Rule {
            name: name.to_string(),
            predicate,
            multiplier,
            types,
            type_ids,
        };
        let spec = self.effects.get_mut(effect).expect("checked");
        spec.rules.retain(|r| r.name != name);
        spec.rules.push(rule);
        spec.pending_rules.retain(|n| n != name);
        Ok(spec.rules.last().expect("just pushed"))
    }

    /// Remove a rule by name.
    pub fn remove_rule(&mut self, effect: &str, name: &str) -> Result<()> {
        self.effect_mut(effect)?.rules.retain(|r| r.name != name);
        Ok(())
    }

    // -- writes -------------------------------------------------------------

    /// Append an event and return it (with its assigned id).
    pub fn insert(&mut self, ev: NewEvent) -> Result<Event> {
        if !ev.time.is_finite() {
            return Err(Error::invalid(format!(
                "event time must be finite, got {}",
                repr_f64(ev.time)
            )));
        }
        let weight = match ev.weight {
            Some(w) => w,
            None => self.derive_weight(Some(&ev.event_type), &ev.attrs)?,
        };
        let weight = validate_weight(weight)?;
        let id = self.next_id;
        let mut names = ev.features;
        names.sort();
        names.dedup();
        let event = Event {
            id,
            entity: ev.entity,
            time: ev.time,
            event_type: ev.event_type,
            features: names,
            weight,
            attrs: ev.attrs,
        };
        self.next_id += 1;
        self.append(&event, true);
        Ok(event)
    }

    pub(crate) fn derive_weight(&self, event_type: Option<&str>, attrs: &Attrs) -> Result<f64> {
        let f = event_type
            .and_then(|t| self.event_types.get(t))
            .and_then(Option::as_ref);
        match f {
            Some(f) => f.weight(attrs).map_err(Error::Callback),
            None => Ok(1.0),
        }
    }

    /// Store an event with an already assigned id (inserts and snapshot
    /// loads). `queue_late`: queue it for `settle` if it lands behind a
    /// settle cursor (not for snapshot loads: their `late` lists are
    /// restored as saved).
    pub(crate) fn append(&mut self, ev: &Event, queue_late: bool) {
        let eid = self.vocab.entities.intern(&ev.entity);
        let ty = self.vocab.types.intern(&ev.event_type);
        let mut feats: Vec<FeatureId> = ev
            .features
            .iter()
            .map(|f| self.vocab.features.intern(f))
            .collect();
        feats.sort_unstable();
        feats.dedup();
        let (bucket, max_sets) = (self.bucket_seconds, self.max_sets);
        let log = self
            .logs
            .entry(eid)
            .or_insert_with(|| EntityLog::new(bucket, max_sets));
        let attrs = (!ev.attrs.is_empty()).then(|| Box::new(ev.attrs.clone()));
        let ord = log.add(ev.id, ev.time, ty, &feats, ev.weight, attrs);
        match self.id_index.last() {
            Some(&(last, _, _)) if last >= ev.id => {
                let pos = self.id_index.partition_point(|&(i, _, _)| i < ev.id);
                self.id_index.insert(pos, (ev.id, eid, ord));
            }
            _ => self.id_index.push((ev.id, eid, ord)),
        }
        if !queue_late {
            return;
        }
        for spec in self.effects.values_mut() {
            if let Some(&cur) = spec.obs.cursors.get(&ev.entity) {
                if ev.time <= cur {
                    spec.obs.late.entry(ev.entity.clone()).or_default().push(ev.id);
                }
            }
        }
    }

    // -- reads ------------------------------------------------------------

    pub(crate) fn locate(&self, id: u64) -> Option<(EntityId, u32)> {
        let i = self.id_index.binary_search_by_key(&id, |&(x, _, _)| x).ok()?;
        let (_, e, o) = self.id_index[i];
        Some((e, o))
    }

    pub(crate) fn stored(&self, eid: EntityId, ord: u32) -> (&EntityLog, &StoredEvent) {
        let log = &self.logs[&eid];
        (log, &log.events[ord as usize])
    }

    pub(crate) fn to_event(&self, eid: EntityId, log: &EntityLog, ev: &StoredEvent) -> Event {
        let mut features: Vec<String> = log
            .feats(ev)
            .iter()
            .map(|&f| self.vocab.features.name(f).to_string())
            .collect();
        features.sort();
        Event {
            id: ev.id,
            entity: self.vocab.entities.name(eid).to_string(),
            time: ev.time,
            event_type: self.vocab.types.name(ev.ty).to_string(),
            features,
            weight: ev.weight,
            attrs: ev.attrs.as_deref().cloned().unwrap_or_default(),
        }
    }

    /// The event with this id.
    pub fn get(&self, id: u64) -> Result<Event> {
        let (e, o) = self.locate(id).ok_or(Error::UnknownEvent(id))?;
        let (log, ev) = self.stored(e, o);
        Ok(self.to_event(e, log, ev))
    }

    /// True if an event with this id is stored.
    pub fn contains(&self, id: u64) -> bool {
        self.locate(id).is_some()
    }

    /// Event ids: of one entity (time-bucket order, then insertion
    /// order), or of all entities (id order).
    pub fn event_ids(&self, entity: Option<&str>) -> Vec<u64> {
        match entity {
            None => self.id_index.iter().map(|&(i, _, _)| i).collect(),
            Some(e) => match self.entity_log(e) {
                None => Vec::new(),
                Some(log) => log.ords_in_order().map(|o| log.events[o as usize].id).collect(),
            },
        }
    }

    /// Events (see [`HeapDB::event_ids`] for the order).
    pub fn events(&self, entity: Option<&str>) -> impl Iterator<Item = Event> + '_ {
        self.event_ids(entity)
            .into_iter()
            .filter_map(move |i| self.get(i).ok())
    }

    /// Entities with a log, sorted.
    pub fn entities(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .logs
            .keys()
            .map(|&e| self.vocab.entities.name(e).to_string())
            .collect();
        v.sort();
        v
    }

    /// Number of entities with a log.
    pub fn num_entities(&self) -> usize {
        self.logs.len()
    }

    /// The log of an entity.
    pub fn entity_log(&self, entity: &str) -> Option<&EntityLog> {
        self.vocab.entities.get(entity).and_then(|e| self.logs.get(&e))
    }

    /// Number of stored events.
    pub fn len(&self) -> usize {
        self.id_index.len()
    }

    /// True if no events are stored.
    pub fn is_empty(&self) -> bool {
        self.id_index.is_empty()
    }

    /// Effective link belief for `feature` (global view if `entity` is None).
    pub fn link(&self, effect: &str, feature: &str, entity: Option<&str>) -> Result<BetaCounts> {
        let spec = self.effect(effect)?;
        Ok(spec
            .links
            .get(self.vocab.features.get(feature), self.entity_ref(entity)))
    }

    pub(crate) fn entity_ref(&self, entity: Option<&str>) -> EntityRef {
        match entity {
            None => EntityRef::Global,
            Some(e) => EntityRef::Entity(self.vocab.entities.get(e)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn probe<'a>(
        &self,
        id: u64,
        entity: &'a str,
        time: f64,
        ty_name: &'a str,
        names: &'a [String],
        weight: f64,
        attrs: &'a Attrs,
    ) -> Probe<'a> {
        let mut known: Vec<FeatureId> = names.iter().filter_map(|n| self.vocab.features.get(n)).collect();
        known.sort_unstable();
        known.dedup();
        let unknown = names.len() - known.len();
        let mut feats: Vec<Option<FeatureId>> = known.into_iter().map(Some).collect();
        feats.extend(std::iter::repeat_n(None, unknown));
        Probe {
            id,
            entity,
            time,
            ty: self.vocab.types.get(ty_name),
            ty_name,
            feats,
            names,
            weight,
            attrs,
        }
    }

    fn event_probe<'a>(&self, ev: &'a Event) -> Probe<'a> {
        self.probe(
            ev.id,
            &ev.entity,
            ev.time,
            &ev.event_type,
            &ev.features,
            ev.weight,
            &ev.attrs,
        )
    }

    /// Score of a stored event (fast path).
    fn score_stored(
        &self,
        spec: &EffectSpec,
        eid: EntityId,
        log: &EntityLog,
        ev: &StoredEvent,
        at: f64,
        cache: &mut StrengthCache,
    ) -> Result<f64> {
        if !spec.type_allowed_id(Some(ev.ty)) {
            return Ok(0.0);
        }
        let dt = at - ev.time;
        if dt < 0.0 {
            return Ok(0.0);
        }
        let k = spec.kernel.value(dt);
        if k == 0.0 {
            return Ok(0.0);
        }
        let feats = log.feats(ev);
        let mut s = k * ev.weight * spec.links.link_cached(feats, cache);
        if spec.rules.is_empty() {
            return Ok(s);
        }
        let evref = EventRef {
            id: ev.id,
            entity: self.vocab.entities.name(eid),
            time: ev.time,
            event_type: self.vocab.types.name(ev.ty),
            features: Features::Interned(feats, &self.vocab.features),
            weight: ev.weight,
            attrs: ev.attrs.as_deref().unwrap_or(&EMPTY_ATTRS),
        };
        for r in &spec.rules {
            if s == 0.0 {
                break;
            }
            if r.matches(Some(ev.ty), &evref).map_err(Error::Callback)? {
                s *= r.multiplier;
            }
        }
        Ok(s)
    }

    /// The link of a probe under the effect's combine mode.
    fn link_probe(&self, spec: &EffectSpec, p: &Probe<'_>, ent: EntityRef) -> f64 {
        spec.links.link_opt(&p.feats, ent)
    }

    /// Score of an event for an effect at time `at` (the event need not be
    /// stored; its entity's links are used).
    pub fn score(&self, event: &Event, effect: &str, at: f64) -> Result<f64> {
        let spec = self.effect(effect)?;
        let p = self.event_probe(event);
        if !spec.type_allowed_id(p.ty) {
            return Ok(0.0);
        }
        let dt = at - p.time;
        if dt < 0.0 {
            return Ok(0.0);
        }
        let k = spec.kernel.value(dt);
        if k == 0.0 {
            return Ok(0.0);
        }
        let ent = EntityRef::Entity(self.vocab.entities.get(&event.entity));
        let mut s = k * p.weight * self.link_probe(spec, &p, ent);
        let evref = p.evref();
        for r in &spec.rules {
            if s == 0.0 {
                break;
            }
            if r.matches(p.ty, &evref).map_err(Error::Callback)? {
                s *= r.multiplier;
            }
        }
        Ok(s)
    }

    /// Break an event's score into timing, weight, links and rules.
    pub fn explain(&self, event: &Event, effect: &str, at: f64) -> Result<Explanation> {
        let spec = self.effect(effect)?;
        let p = self.event_probe(event);
        self.explain_probe(spec, &p, at)
    }

    fn contributions(&self, spec: &EffectSpec, p: &Probe<'_>, ent: EntityRef) -> Vec<FeatureContribution> {
        let mut out: Vec<FeatureContribution> = p
            .names
            .iter()
            .map(|n| {
                let id = self.vocab.features.get(n);
                let c = spec.links.get(id, ent);
                FeatureContribution {
                    feature: n.clone(),
                    strength: c.strength(),
                    alpha: c.alpha(),
                    beta: c.beta(),
                    log_odds: spec.links.weight(id),
                }
            })
            .collect();
        // Strongest first: by link strength (noisy-OR) or by the log-odds
        // multiplier (logistic), then by name.
        let key = |c: &FeatureContribution| c.log_odds.unwrap_or(c.strength);
        out.sort_by(|a, b| {
            key(b)
                .partial_cmp(&key(a))
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.feature.cmp(&b.feature))
        });
        out
    }

    fn applied_rules(&self, spec: &EffectSpec, p: &Probe<'_>) -> Result<Vec<AppliedRule>> {
        let evref = p.evref();
        let mut out = Vec::new();
        for r in &spec.rules {
            if r.matches(p.ty, &evref).map_err(Error::Callback)? {
                out.push(AppliedRule {
                    name: r.name.clone(),
                    multiplier: r.multiplier,
                });
            }
        }
        Ok(out)
    }

    fn explain_probe(&self, spec: &EffectSpec, p: &Probe<'_>, at: f64) -> Result<Explanation> {
        let dt = at - p.time;
        let ent = EntityRef::Entity(self.vocab.entities.get(p.entity));
        let groups = self.inseparable_groups(p);
        Ok(Explanation {
            dt,
            kernel_value: if dt >= 0.0 { spec.kernel.value(dt) } else { 0.0 },
            weight: p.weight,
            link_strength: self.link_probe(spec, p, ent),
            features: self.contributions(spec, p, ent),
            base_log_odds: spec.links.base_log_odds(),
            warnings: self.confounding_warnings(spec, p, &groups),
            rules: self.applied_rules(spec, p)?,
            joint_causes: self.joint_causes(spec, p, &groups),
            entity: p.entity.to_string(),
            prospective: false,
        })
    }

    // -- co-occurrence / joint causes --------------------------------------

    /// Groups (size >= 2) of the event's features that have always
    /// co-occurred for its entity (identical posting lists).
    pub(crate) fn inseparable_groups(&self, p: &Probe<'_>) -> Vec<Group> {
        let Some(log) = self.entity_log(p.entity) else {
            return Vec::new();
        };
        if p.names.len() < 2 {
            return Vec::new();
        }
        #[allow(clippy::type_complexity)]
        let mut groups: Vec<(&[u32], Vec<String>, Vec<Option<FeatureId>>)> = Vec::new();
        for n in p.names {
            let id = self.vocab.features.get(n);
            let post = log.posting(id);
            match groups.iter_mut().find(|g| g.0 == post) {
                Some(g) => {
                    g.1.push(n.clone());
                    g.2.push(id);
                }
                None => groups.push((post, vec![n.clone()], vec![id])),
            }
        }
        groups
            .into_iter()
            .filter(|g| g.1.len() >= 2)
            .map(|(post, names, mut ids)| {
                ids.sort_by_key(|f| (f.is_none(), *f));
                Group {
                    ids,
                    names,
                    n: post.len(),
                }
            })
            .collect()
    }

    fn has_feedback(&self, spec: &EffectSpec, entity: &str, ids: &[Option<FeatureId>]) -> bool {
        let Some(e) = self.vocab.entities.get(entity) else {
            return false;
        };
        ids.iter().flatten().any(|&f| spec.links.has_entity_entry(e, f))
    }

    fn joint_causes(&self, spec: &EffectSpec, p: &Probe<'_>, groups: &[Group]) -> Vec<JointCause> {
        let eid = self.vocab.entities.get(p.entity);
        let ent = EntityRef::Entity(eid);
        let mut out = Vec::new();
        let mut seen: Vec<Vec<FeatureId>> = Vec::new();
        for g in groups {
            let all_known: Option<Vec<FeatureId>> = g.ids.iter().copied().collect();
            let joint = match (eid, &all_known) {
                (Some(e), Some(ids)) => spec.links.joint(e, ids),
                _ => None,
            };
            if joint.is_none() && !self.has_feedback(spec, p.entity, &g.ids) {
                continue;
            }
            if let Some(ids) = all_known {
                seen.push(ids);
            }
            out.push(JointCause {
                features: g.names.clone(),
                strength: spec.links.link_opt(&g.ids, ent),
                joint,
                inseparable: true,
            });
        }
        if let Some(e) = eid {
            let known: FxHashSet<FeatureId> = p.known_ids().collect();
            for fs in spec.links.joint_sets(e) {
                if seen.iter().any(|s| s.as_slice() == fs) || !fs.iter().all(|f| known.contains(f)) {
                    continue;
                }
                let mut names: Vec<String> = fs
                    .iter()
                    .map(|&f| self.vocab.features.name(f).to_string())
                    .collect();
                names.sort();
                let opt: Vec<Option<FeatureId>> = fs.iter().copied().map(Some).collect();
                out.push(JointCause {
                    features: names,
                    strength: spec.links.link_opt(&opt, ent),
                    joint: spec.links.joint(e, fs),
                    inseparable: false,
                });
            }
        }
        out.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.features.cmp(&b.features))
        });
        out
    }

    fn confounding_warnings(&self, spec: &EffectSpec, p: &Probe<'_>, groups: &[Group]) -> Vec<String> {
        let mut out = Vec::new();
        for g in groups {
            if g.n < 2 || !self.has_feedback(spec, p.entity, &g.ids) {
                continue;
            }
            let names: Vec<String> = g.names.iter().map(|f| format!("'{f}'")).collect();
            out.push(format!(
                "{} always co-occur ({} events): feedback cannot separate their effects yet; reported as joint cause {}",
                names.join(", "),
                g.n,
                g.names.join("+")
            ));
        }
        out
    }

    // -- the main query ---------------------------------------------------

    /// Event-time interval that can possibly contribute to the effect.
    pub fn window(&self, effect: &str, at: f64, within: Option<f64>) -> Result<(f64, f64)> {
        Ok(window(self.effect(effect)?, at, within))
    }

    /// Upper bound on the score of any event of segment `bucket` of
    /// `log_entity`'s log, using the links of `link_entity`, and whether the
    /// loose union bound was used.
    #[allow(clippy::too_many_arguments)]
    pub fn segment_bound(
        &self,
        effect: &str,
        log_entity: &str,
        bucket: i64,
        link_entity: &str,
        at: f64,
        t_lo: f64,
        t_hi: f64,
        use_sets: bool,
    ) -> Result<(f64, bool)> {
        let spec = self.effect(effect)?;
        let log = self
            .entity_log(log_entity)
            .ok_or_else(|| Error::invalid(format!("no log for entity {}", repr_str(log_entity))))?;
        let seg = log.segment(bucket).ok_or_else(|| {
            Error::invalid(format!("no segment {bucket} for entity {}", repr_str(log_entity)))
        })?;
        let mut cache = StrengthCache::new(self.entity_ref(Some(link_entity)));
        Ok(bound(spec, log, seg, at, t_lo, t_hi, use_sets, &mut cache))
    }

    /// Bucket numbers of an entity's segments overlapping `[t_lo, t_hi]`
    /// (all segments if `range` is `None`), in time order.
    pub fn segment_buckets(&self, entity: &str, range: Option<(f64, f64)>) -> Vec<i64> {
        let Some(log) = self.entity_log(entity) else {
            return Vec::new();
        };
        match range {
            None => log.segments().map(Segment::bucket).collect(),
            Some((lo, hi)) => log.segments_between(lo, hi).map(Segment::bucket).collect(),
        }
    }

    /// Resolved summaries of one segment.
    pub fn segment_info(&self, entity: &str, bucket: i64) -> Option<SegmentInfo> {
        let log = self.entity_log(entity)?;
        let seg = log.segment(bucket)?;
        let fname = |f: FeatureId| self.vocab.features.name(f).to_string();
        let mut feature_counts: Vec<(String, u32)> =
            seg.feature_counts.iter().map(|&(f, c)| (fname(f), c)).collect();
        feature_counts.sort();
        Some(SegmentInfo {
            bucket,
            start: seg.start,
            width: seg.width,
            min_time: seg.min_time,
            max_time: seg.max_time,
            max_weight: seg.max_weight,
            event_ids: seg.ords.iter().map(|&o| log.events[o as usize].id).collect(),
            feature_counts,
            types: seg
                .types
                .iter()
                .map(|&t| self.vocab.types.name(t).to_string())
                .collect(),
            sets: seg.sets.as_ref().map(|sets| {
                sets.iter()
                    .map(|k| {
                        let mut names: Vec<String> = log.key_feats(k).iter().map(|&f| fname(f)).collect();
                        names.sort();
                        (self.vocab.types.name(k.ty).to_string(), names, k.max_weight)
                    })
                    .collect()
            }),
            max_sets: log.max_sets,
        })
    }

    /// Ids of an entity's events with a feature (insertion order).
    pub fn event_ids_with_feature(&self, entity: &str, feature: &str) -> Vec<u64> {
        let Some(log) = self.entity_log(entity) else {
            return Vec::new();
        };
        log.posting(self.vocab.features.get(feature))
            .iter()
            .map(|&o| log.events[o as usize].id)
            .collect()
    }

    /// The `k` most likely causes of `effect` for `entity` at `at`.
    pub fn top_causes(&self, entity: &str, effect: &str, at: f64, opts: TopK) -> Result<QueryResult> {
        if opts.k == 0 {
            return Err(Error::invalid("k must be >= 1"));
        }
        let spec = self.effect(effect)?;
        let mut result = QueryResult {
            entity: entity.to_string(),
            effect: effect.to_string(),
            at,
            causes: Vec::new(),
            stats: QueryStats::default(),
            entities: vec![entity.to_string()],
        };
        let Some(eid) = self.vocab.entities.get(entity) else {
            return Ok(result);
        };
        let Some(log) = self.logs.get(&eid) else {
            return Ok(result);
        };
        let (t_lo, t_hi) = window(spec, at, opts.within);
        let mut cache = StrengthCache::new(EntityRef::Entity(Some(eid)));
        let stats = &mut result.stats;

        // Step 1: bound every segment from its summaries + current links.
        let mut bounded: Vec<(f64, &Segment)> = Vec::new();
        for seg in log.segments_between(t_lo, t_hi) {
            let (b, union) = bound(spec, log, seg, at, t_lo, t_hi, opts.use_sets, &mut cache);
            stats.union_fallbacks += union as usize;
            bounded.push((b * (1.0 + BOUND_SLACK), seg));
        }
        stats.segments_in_window = bounded.len();

        // Step 2: best bound first (ties by start: deterministic).
        bounded.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.start.partial_cmp(&b.1.start).unwrap_or(Ordering::Equal))
        });

        // Bounded min-heap of (score, id); the root is the k-th best so far.
        let k = opts.k;
        let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::with_capacity(k + 1);
        for (i, &(b, seg)) in bounded.iter().enumerate() {
            if b <= 0.0 || (heap.len() == k && heap.peek().is_some_and(|m| m.0.score >= b)) {
                stats.segments_pruned += bounded.len() - i;
                break;
            }
            stats.segments_scanned += 1;
            for &ord in &seg.ords {
                let ev = &log.events[ord as usize];
                if !(t_lo <= ev.time && ev.time <= t_hi) {
                    continue;
                }
                stats.events_scanned += 1;
                let s = self.score_stored(spec, eid, log, ev, at, &mut cache)?;
                if s <= 0.0 {
                    continue;
                }
                let item = HeapItem {
                    score: s,
                    id: ev.id,
                    ord,
                };
                if heap.len() < k {
                    heap.push(Reverse(item));
                } else if heap.peek().is_some_and(|m| item > m.0) {
                    heap.pop();
                    heap.push(Reverse(item));
                }
            }
        }
        let mut ranked: Vec<HeapItem> = heap.into_iter().map(|r| r.0).collect();
        ranked.sort_by(|a, b| b.cmp(a));
        result.causes = ranked
            .into_iter()
            .enumerate()
            .map(|(i, it)| self.make_cause(spec, eid, log, it.ord, i + 1, it.score, at, opts.explain))
            .collect::<Result<_>>()?;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn make_cause(
        &self,
        spec: &EffectSpec,
        eid: EntityId,
        log: &EntityLog,
        ord: u32,
        rank: usize,
        score: f64,
        at: f64,
        explain: bool,
    ) -> Result<Cause> {
        let event = self.to_event(eid, log, &log.events[ord as usize]);
        let explanation = if explain {
            Some(self.explain_probe(spec, &self.event_probe(&event), at)?)
        } else {
            None
        };
        Ok(Cause {
            rank,
            event,
            score,
            explanation,
        })
    }

    /// Top-k causes of one effect across several entities: per-entity top-k
    /// merged with a k-way merge. Exact w.r.t. brute force over the union.
    ///
    /// Scores are only comparable across entities to the extent they are
    /// calibrated (see [`HeapDB::calibration_report`]).
    pub fn top_causes_group(
        &self,
        entities: &[&str],
        effect: &str,
        at: f64,
        opts: TopK,
    ) -> Result<QueryResult> {
        let mut ents: Vec<&str> = Vec::new();
        for &e in entities {
            if !ents.contains(&e) {
                ents.push(e);
            }
        }
        if ents.is_empty() {
            return Err(Error::invalid("need at least one entity"));
        }
        let per: Vec<QueryResult> = ents
            .iter()
            .map(|e| self.top_causes(e, effect, at, opts))
            .collect::<Result<_>>()?;
        let mut result = QueryResult {
            entity: ents.join(","),
            effect: effect.to_string(),
            at,
            causes: Vec::new(),
            stats: QueryStats::default(),
            entities: ents.iter().map(|s| s.to_string()).collect(),
        };
        for r in &per {
            result.stats.add(&r.stats);
        }
        // k-way merge on (-score, -id), earlier lists first on ties.
        #[derive(PartialEq)]
        struct Head(f64, u64, usize, usize);
        impl Eq for Head {}
        impl PartialOrd for Head {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Head {
            // Max-heap: "greater" = should come out first.
            fn cmp(&self, o: &Self) -> Ordering {
                self.0
                    .total_cmp(&o.0)
                    .then(self.1.cmp(&o.1))
                    .then(o.2.cmp(&self.2))
            }
        }
        let mut lists: Vec<std::vec::IntoIter<Cause>> =
            per.into_iter().map(|r| r.causes.into_iter()).collect();
        let mut fronts: Vec<Option<Cause>> = lists.iter_mut().map(Iterator::next).collect();
        let mut heap = BinaryHeap::new();
        for (i, c) in fronts.iter().enumerate() {
            if let Some(c) = c {
                heap.push(Head(c.score, c.event.id, i, 0));
            }
        }
        while result.causes.len() < opts.k {
            let Some(Head(_, _, i, _)) = heap.pop() else { break };
            let mut c = fronts[i].take().expect("head present");
            c.rank = result.causes.len() + 1;
            result.causes.push(c);
            fronts[i] = lists[i].next();
            if let Some(n) = &fronts[i] {
                heap.push(Head(n.score, n.event.id, i, 0));
            }
        }
        Ok(result)
    }

    /// Reference brute force: score every event of the entity in the
    /// window (no bounds, no pruning), sort, cut. For testing and as a
    /// benchmark baseline.
    pub fn top_causes_naive(
        &self,
        entity: &str,
        effect: &str,
        at: f64,
        k: usize,
        within: Option<f64>,
    ) -> Result<Vec<(f64, Event)>> {
        let spec = self.effect(effect)?;
        let (t_lo, t_hi) = window(spec, at, within);
        let Some(eid) = self.vocab.entities.get(entity) else {
            return Ok(Vec::new());
        };
        let Some(log) = self.logs.get(&eid) else {
            return Ok(Vec::new());
        };
        let mut cache = StrengthCache::new(EntityRef::Entity(Some(eid)));
        let mut scored = Vec::new();
        let ords = log
            .segments_between(t_lo, t_hi)
            .flat_map(|s| s.ords.iter().copied());
        for ord in ords {
            let ev = &log.events[ord as usize];
            if !(t_lo <= ev.time && ev.time <= t_hi) {
                continue;
            }
            let s = self.score_stored(spec, eid, log, ev, at, &mut cache)?;
            if s > 0.0 {
                scored.push((s, ev.id, ord));
            }
        }
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
                .then(b.1.cmp(&a.1))
        });
        scored.truncate(k);
        Ok(scored
            .into_iter()
            .map(|(s, _, ord)| (s, self.to_event(eid, log, &log.events[ord as usize])))
            .collect())
    }

    /// Pop the top: the single most likely cause right now (explained).
    pub fn most_likely_cause(&self, entity: &str, effect: &str, at: f64) -> Result<Option<Cause>> {
        let r = self.top_causes(entity, effect, at, TopK::new(1).explain(true))?;
        Ok(r.causes.into_iter().next())
    }

    // -- prospective risk ---------------------------------------------------

    fn transient_weight(&self, c: &Candidate) -> Result<f64> {
        let w = match c.weight {
            Some(w) => w,
            None => self.derive_weight(c.event_type.as_deref(), &c.attrs)?,
        };
        validate_weight(w)
    }

    fn risk(&self, spec: &EffectSpec, p: &Probe<'_>, ent: EntityRef, type_given: bool) -> Result<f64> {
        if type_given && !spec.type_allowed_id(p.ty) {
            return Ok(0.0);
        }
        let mut s = p.weight * self.link_probe(spec, p, ent);
        let evref = p.evref();
        for r in &spec.rules {
            if s == 0.0 {
                break;
            }
            if r.matches(p.ty, &evref).map_err(Error::Callback)? {
                s *= r.multiplier;
            }
        }
        Ok(s)
    }

    fn risk_explanation(
        &self,
        spec: &EffectSpec,
        p: &Probe<'_>,
        entity: Option<&str>,
        type_given: bool,
    ) -> Result<Explanation> {
        let ent = self.entity_ref(entity);
        let mut rules = self.applied_rules(spec, p)?;
        if type_given && !spec.type_allowed_id(p.ty) {
            rules.insert(
                0,
                AppliedRule {
                    name: format!("type {} not in cause_types", repr_str(p.ty_name)),
                    multiplier: 0.0,
                },
            );
        }
        let (mut warnings, mut joint) = (Vec::new(), Vec::new());
        if entity.is_some() {
            let groups: Vec<Group> = self
                .inseparable_groups(p)
                .into_iter()
                .filter(|g| g.n > 0)
                .collect();
            warnings = self.confounding_warnings(spec, p, &groups);
            joint = self.joint_causes(spec, p, &groups);
        }
        Ok(Explanation {
            dt: 0.0,
            kernel_value: 1.0,
            weight: p.weight,
            link_strength: self.link_probe(spec, p, ent),
            features: self.contributions(spec, p, ent),
            base_log_odds: spec.links.base_log_odds(),
            warnings,
            rules,
            joint_causes: joint,
            entity: p.entity.to_string(),
            prospective: true,
        })
    }

    /// Risk that a candidate which has **not happened yet** causes `effect`:
    /// `weight * noisy_OR(strength_entity(f)) * rules` (no kernel term).
    ///
    /// Rules see a transient, unstored event (`id = 0`, `time = NaN`). With
    /// `explain`, also returns a prospective [`Explanation`].
    pub fn feature_risk(
        &self,
        effect: &str,
        c: &Candidate,
        explain: bool,
    ) -> Result<(f64, Option<Explanation>)> {
        let spec = self.effect(effect)?;
        let weight = self.transient_weight(c)?;
        let mut names = c.features.clone();
        names.sort();
        names.dedup();
        let ent_name = c.entity.as_deref().unwrap_or("");
        let ty_name = c.event_type.as_deref().unwrap_or("");
        let p = self.probe(0, ent_name, f64::NAN, ty_name, &names, weight, &c.attrs);
        let ent = self.entity_ref(c.entity.as_deref());
        let type_given = c.event_type.is_some();
        let s = self.risk(spec, &p, ent, type_given)?;
        let ex = if explain {
            Some(self.risk_explanation(spec, &p, c.entity.as_deref(), type_given)?)
        } else {
            None
        };
        Ok((s, ex))
    }

    /// Top-`k` not-yet-happened candidates by [`HeapDB::feature_risk`]. A
    /// candidate's own `entity` overrides `entity`. Zero scores are dropped;
    /// ties keep the earlier candidate.
    pub fn rank_candidates(
        &self,
        effect: &str,
        candidates: &[Candidate],
        k: usize,
        entity: Option<&str>,
        explain: bool,
    ) -> Result<Vec<RankedCandidate>> {
        if k == 0 {
            return Err(Error::invalid("k must be >= 1"));
        }
        let spec = self.effect(effect)?;
        // Min-heap on (score, -index).
        #[derive(PartialEq)]
        struct It(f64, usize);
        impl Eq for It {}
        impl PartialOrd for It {
            fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for It {
            fn cmp(&self, o: &Self) -> Ordering {
                self.0.total_cmp(&o.0).then(o.1.cmp(&self.1))
            }
        }
        let mut heap: BinaryHeap<Reverse<It>> = BinaryHeap::new();
        for (i, c) in candidates.iter().enumerate() {
            let mut c2 = c.clone();
            if c2.entity.is_none() {
                c2.entity = entity.map(str::to_string);
            }
            let (s, _) = self.feature_risk(effect, &c2, false)?;
            if s <= 0.0 {
                continue;
            }
            let it = It(s, i);
            if heap.len() < k {
                heap.push(Reverse(it));
            } else if heap.peek().is_some_and(|m| it > m.0) {
                heap.pop();
                heap.push(Reverse(it));
            }
        }
        let mut ranked: Vec<It> = heap.into_iter().map(|r| r.0).collect();
        ranked.sort_by(|a, b| b.cmp(a));
        let _ = spec;
        ranked
            .into_iter()
            .enumerate()
            .map(|(r, It(s, i))| {
                let mut c = candidates[i].clone();
                if c.entity.is_none() {
                    c.entity = entity.map(str::to_string);
                }
                let explanation = if explain {
                    self.feature_risk(effect, &c, true)?.1
                } else {
                    None
                };
                Ok(RankedCandidate {
                    rank: r + 1,
                    index: i,
                    score: s,
                    entity: c.entity,
                    explanation,
                })
            })
            .collect()
    }

    // -- feedback ---------------------------------------------------------------

    /// Teach the database which candidate really was the cause.
    ///
    /// Confirmed event: Beta credit per `credit` (default: the effect's credit
    /// mode: proportional / EM under noisy-OR, uniform under logistic);
    /// rejected events: `beta += 1` per feature. Updates go to the entity and
    /// the global layer; inseparable groups get a joint link; the pre-update
    /// scores are recorded for calibration. Under `combine = logistic` the
    /// weights also take one example per judged event: the confirmed one
    /// first (label 1), then the rejected ones in the given order (label 0);
    /// see [`crate::logistic`]. Everything is validated before anything
    /// changes.
    pub fn feedback(
        &mut self,
        entity: &str,
        effect: &str,
        at: f64,
        confirmed: Option<u64>,
        rejected: &[u64],
        credit: Option<Credit>,
    ) -> Result<FeedbackReport> {
        let spec = self.effect(effect)?;
        let mode = credit.unwrap_or(spec.credit);
        let checked = |id: u64| -> Result<Event> {
            let ev = self.get(id)?;
            if ev.entity != entity {
                return Err(Error::invalid(format!(
                    "event {id} belongs to {}, not {}",
                    repr_str(&ev.entity),
                    repr_str(entity)
                )));
            }
            if ev.time > at {
                return Err(Error::invalid(format!("event {id} happened after the effect")));
            }
            Ok(ev)
        };
        let conf = confirmed.map(checked).transpose()?;
        let rej: Vec<Event> = rejected.iter().map(|&i| checked(i)).collect::<Result<_>>()?;
        if let Some(c) = &conf {
            if rej.iter().any(|r| r.id == c.id) {
                return Err(Error::invalid(format!(
                    "event {} cannot be both confirmed and rejected",
                    c.id
                )));
            }
        }
        // Calibration: the scores as shown, before learning.
        let mut records = Vec::new();
        for (ev, ok) in conf
            .iter()
            .map(|e| (e, true))
            .chain(rej.iter().map(|e| (e, false)))
        {
            records.push(CalibrationRecord {
                entity: entity.to_string(),
                event_id: ev.id,
                at,
                score: self.score(ev, effect, at)?,
                confirmed: ok,
            });
        }
        // Everything that depends on the pre-update state.
        let eid = self.vocab.entities.get(entity).expect("validated events exist");
        let ent = EntityRef::Entity(Some(eid));
        let ids_by_name = |db: &HeapDB, ev: &Event| -> Vec<FeatureId> {
            ev.features
                .iter()
                .map(|f| db.vocab.features.get(f).expect("stored feature"))
                .collect()
        };
        let sorted_ids = |db: &HeapDB, names: &[String]| -> Vec<FeatureId> {
            let mut v: Vec<FeatureId> = names
                .iter()
                .map(|f| db.vocab.features.get(f).expect("stored"))
                .collect();
            v.sort_unstable();
            v
        };
        let mut joint_updates: Vec<(Vec<FeatureId>, f64, f64)> = Vec::new();
        let mut confirm_ids: Option<Vec<FeatureId>> = None;
        if let Some(c) = &conf {
            let p = self.event_probe(c);
            let groups = self.inseparable_groups(&p);
            let p_all = self.link_probe(spec, &p, ent);
            for g in &groups {
                let gp = spec.links.link_opt(&g.ids, ent);
                let r = if mode == Credit::Uniform || p_all <= 0.0 {
                    1.0
                } else {
                    (gp / p_all).min(1.0)
                };
                joint_updates.push((sorted_ids(self, &g.names), r, 1.0 - r));
            }
            confirm_ids = Some(ids_by_name(self, c));
        }
        let logistic_confirm: Option<Vec<FeatureId>> = confirm_ids.as_ref().map(|ids| {
            let mut v = ids.clone();
            v.sort_unstable();
            v
        });
        let mut reject_plan: Vec<(Vec<Vec<FeatureId>>, Vec<FeatureId>)> = Vec::new();
        for r in &rej {
            let p = self.event_probe(r);
            let gs = self
                .inseparable_groups(&p)
                .iter()
                .map(|g| sorted_ids(self, &g.names))
                .collect();
            reject_plan.push((gs, ids_by_name(self, r)));
        }
        // Mutations.
        self.calibration
            .entry(effect.to_string())
            .or_default()
            .extend(records);
        let spec = self.effects.get_mut(effect).expect("checked");
        let links = &mut spec.links;
        let mut credit_map = Vec::new();
        if let Some(ids) = confirm_ids {
            let resp = links.confirm(&ids, Some(eid), mode);
            credit_map = ids
                .iter()
                .zip(resp)
                .map(|(&f, r)| (self.vocab.features.name(f).to_string(), r))
                .collect();
            for (fs, p, n) in &joint_updates {
                links.update_joint(eid, fs, *p, *n);
            }
        }
        for (groups, ids) in &reject_plan {
            for fs in groups {
                links.update_joint(eid, fs, 0.0, 1.0);
            }
            links.reject(ids, Some(eid));
        }
        // Logistic weights: one example per judged event, the confirmed one
        // first (label 1), then the rejected ones in the given order.
        if links.combine() == Combine::Logistic {
            if let Some(ids) = &logistic_confirm {
                links.logistic_step(ids, 1.0, 1.0, Some(eid));
            }
            for (_, ids) in &reject_plan {
                let mut ids = ids.clone();
                ids.sort_unstable();
                links.logistic_step(&ids, 0.0, 1.0, Some(eid));
            }
        }
        Ok(FeedbackReport {
            confirmed: conf.map(|c| c.id),
            rejected: rej.iter().map(|r| r.id).collect(),
            credit: credit_map,
        })
    }

    // -- calibration ----------------------------------------------------------

    /// The (score, outcome) records of an effect.
    pub fn calibration_records(&self, effect: &str) -> Result<&[CalibrationRecord]> {
        self.effect(effect)?;
        Ok(self.calibration.get(effect).map(Vec::as_slice).unwrap_or(&[]))
    }

    /// Mean predicted score vs observed confirm rate per equal-width bin on
    /// `[0, 1]` (scores clipped), plus the Brier score.
    pub fn calibration_report(&self, effect: &str, bins: usize) -> Result<CalibrationReport> {
        if bins < 1 {
            return Err(Error::invalid("bins must be >= 1"));
        }
        let recs = self.calibration_records(effect)?;
        let mut acc = vec![(0usize, 0.0f64, 0usize); bins];
        let mut sq = 0.0;
        for r in recs {
            let p = r.predicted();
            let i = ((p * bins as f64) as usize).min(bins - 1);
            acc[i].0 += 1;
            acc[i].1 += p;
            acc[i].2 += r.confirmed as usize;
            let y = if r.confirmed { 1.0 } else { 0.0 };
            sq += (p - y).powi(2);
        }
        let b = bins as f64;
        let out = acc
            .into_iter()
            .enumerate()
            .map(|(i, (n, s, c))| CalibrationBin {
                lo: i as f64 / b,
                hi: (i + 1) as f64 / b,
                count: n,
                mean_predicted: (n > 0).then(|| s / n as f64),
                observed_rate: (n > 0).then(|| c as f64 / n as f64),
            })
            .collect();
        Ok(CalibrationReport {
            effect: effect.to_string(),
            n: recs.len(),
            brier: (!recs.is_empty()).then(|| sq / recs.len() as f64),
            bins: out,
        })
    }

    // -- privacy ---------------------------------------------------------------

    /// Remove everything stored about `entity`: events, segments,
    /// entity-level and joint links, observational state and calibration
    /// records. With `scrub_global` its contributions to the global layer
    /// are subtracted too (exactly: the entity layer records them).
    pub fn forget_entity(&mut self, entity: &str, scrub_global: bool) -> ForgetReport {
        let mut out = ForgetReport::default();
        let eid = self.vocab.entities.get(entity);
        let log = eid.and_then(|e| self.logs.shift_remove(&e));
        let mut ids: Vec<u64> = Vec::new();
        if let (Some(e), Some(log)) = (eid, &log) {
            out.segments = log.num_segments();
            ids = log.events.iter().map(|ev| ev.id).collect();
            out.events = ids.len();
            self.id_index.retain(|&(_, x, _)| x != e);
        }
        for spec in self.effects.values_mut() {
            if let Some(e) = eid {
                out.links += spec.links.forget_entity(e, scrub_global);
            }
            let obs = &mut spec.obs;
            obs.occurrences.remove(entity);
            obs.cursors.remove(entity);
            obs.late.remove(entity);
            for id in &ids {
                obs.credited.remove(id);
                obs.rejected.remove(id);
            }
        }
        for recs in self.calibration.values_mut() {
            let before = recs.len();
            recs.retain(|r| r.entity != entity);
            out.calibration_records += before - recs.len();
        }
        out
    }

    /// `HeapDB(events=..., entities=..., effects=[...], bucket_seconds=...)`.
    pub fn repr(&self) -> String {
        let effs = self.effects();
        format!(
            "HeapDB(events={}, entities={}, effects={}, bucket_seconds={})",
            self.len(),
            self.logs.len(),
            repr_str_list(effs.iter().map(String::as_str)),
            crate::pyfmt::fmt_g6(self.bucket_seconds)
        )
    }
}

impl Default for HeapDB {
    fn default() -> Self {
        HeapDB::new(900.0, DEFAULT_MAX_SETS).expect("valid defaults")
    }
}

pub(crate) fn window(spec: &EffectSpec, at: f64, within: Option<f64>) -> (f64, f64) {
    let support = spec.kernel.support();
    let lookback = match within {
        None => support,
        Some(w) => w.min(support),
    };
    (at - lookback, at - spec.kernel.min_lag().max(0.0))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bound(
    spec: &EffectSpec,
    log: &EntityLog,
    seg: &Segment,
    at: f64,
    t_lo: f64,
    t_hi: f64,
    use_sets: bool,
    cache: &mut StrengthCache,
) -> (f64, bool) {
    if let Some(ct) = &spec.cause_type_ids {
        if !seg.types.iter().any(|t| ct.contains(t)) {
            return (0.0, false);
        }
    }
    let dt_lo = at - seg.max_time.min(t_hi);
    let dt_hi = at - seg.min_time.max(t_lo);
    let kb = if dt_lo <= dt_hi {
        spec.kernel.max_over(dt_lo, dt_hi)
    } else {
        0.0
    };
    if kb <= 0.0 {
        return (0.0, false);
    }
    if use_sets {
        if let Some(sets) = &seg.sets {
            let mut best = 0.0_f64;
            for key in sets {
                if !spec.type_allowed_id(Some(key.ty)) {
                    continue;
                }
                let rb = spec.rule_bound_id(Some(key.ty));
                let w = key.max_weight;
                if w * rb <= best {
                    continue;
                }
                let lk = spec.links.link_cached(log.key_feats(key), cache);
                let v = w * lk * rb;
                if v > best {
                    best = v;
                }
            }
            return (kb * best, false);
        }
    }
    let mut rb = 0.0_f64;
    let mut any = false;
    for &t in &seg.types {
        if spec.type_allowed_id(Some(t)) {
            let b = spec.rule_bound_id(Some(t));
            if !any || b > rb {
                rb = b;
            }
            any = true;
        }
    }
    let lk = spec
        .links
        .link_upper_bound(seg.feature_counts.iter().map(|&(f, _)| f), cache);
    (kb * seg.max_weight * rb * lk, true)
}

pub(crate) fn check_leak(leak: f64) -> Result<f64> {
    if leak.is_nan() || leak <= 0.0 {
        return Err(Error::invalid(format!(
            "leak must be > 0 (inf disables observational learning), got {}",
            repr_f64(leak)
        )));
    }
    Ok(leak)
}

pub(crate) fn check_w_neg(w: f64) -> Result<f64> {
    if !(w >= 0.0 && w.is_finite()) {
        return Err(Error::invalid(format!(
            "w_neg must be a finite number >= 0, got {}",
            repr_f64(w)
        )));
    }
    Ok(w)
}

/// Convenience: decay rate for a given half-life (seconds).
pub fn half_life_to_lambda(half_life: f64) -> f64 {
    std::f64::consts::LN_2 / half_life
}

#[allow(dead_code)]
fn _assert_send_sync() {
    fn f<T: Send + Sync>() {}
    f::<HeapDB>();
}
