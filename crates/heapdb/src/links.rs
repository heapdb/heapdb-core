//! Learned link strengths between features and effect types.
//!
//! For every `(effect, feature)` pair HeapDB keeps a Beta(alpha, beta)
//! belief about "how likely is this feature to produce this effect"; the
//! point estimate used for ranking is the posterior mean.
//!
//! Three layers (hierarchical priors):
//!
//! 1. **Registered prior** (or the effect's `default_prior`).
//! 2. **Global layer**: evidence pooled over all entities.
//! 3. **Entity layer**: the entity's own evidence.
//!
//! ```text
//! alpha = prior_alpha + c * G_pos + own_pos
//! beta  = prior_beta  + c * G_neg + own_neg
//! c     = min(1, global_pseudocount / (G_pos + G_neg))
//! ```
//!
//! where `G` is the global evidence *excluding this entity's own
//! contributions* (leave-one-out). Every update goes to both the entity and
//! the global layer, so the entity layer is also the ledger of what the
//! entity contributed (this makes `forget_entity(scrub_global)` exact).
//!
//! Features are combined into an event's *link* by the effect's
//! [`Combine`] mode:
//!
//! * `noisy_or` (default): `P = 1 - prod(1 - s_f)` over the posterior means.
//!   Confirmations use the EM responsibilities `r_f = s_f / P` (clamped to
//!   `[0, 1]`): `alpha_f += r_f`, `beta_f += 1 - r_f`; rejections add
//!   `beta_f += 1` to every feature.
//! * `logistic`: `P = sigmoid(b + sum_f w_f)`, an online logistic
//!   regression seeded from the Beta priors (see [`crate::logistic`]). The
//!   Beta links are still updated (by default with uniform credit: the
//!   confirmed event counts as a positive for every feature) and reported,
//!   but scoring uses the weights.
//!
//! Joint links (per entity, per inseparable feature set that was judged)
//! are kept for reporting only.

use indexmap::IndexMap;
use rustc_hash::{FxBuildHasher, FxHashMap};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::intern::Vocab;
use crate::logistic::{beta_logit, sigmoid, Logistic, LogisticParams};
use crate::pyfmt::repr_str;
use crate::snapshot::{as_f64, num, pair};

/// Interned feature id.
pub type FeatureId = u32;
/// Interned entity id.
pub type EntityId = u32;

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;
type JointMap = FxIndexMap<Box<[FeatureId]>, [f64; 2]>;

/// Pseudo-observation weight of a joint link's prior.
pub const JOINT_PRIOR_STRENGTH: f64 = 2.0;

/// Credit assignment for confirmations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Credit {
    /// EM credit for noisy-OR: `r_f = s_f / noisyOR(F)`.
    #[default]
    Proportional,
    /// v0.1 behaviour: `alpha_f += 1` for every feature.
    Uniform,
}

impl Credit {
    /// `"proportional"` or `"uniform"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Credit::Proportional => "proportional",
            Credit::Uniform => "uniform",
        }
    }

    /// Parse `"proportional"` / `"uniform"`.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "proportional" => Ok(Credit::Proportional),
            "uniform" => Ok(Credit::Uniform),
            _ => Err(Error::invalid(format!(
                "credit must be one of ('proportional', 'uniform'), got {}",
                repr_str(s)
            ))),
        }
    }
}

/// How an effect combines the evidence of an event's features into the
/// event's link.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Combine {
    /// Noisy-OR of the Beta link means: `1 - prod(1 - s_f)`. Independent
    /// causes; evidence adds up (in probability).
    #[default]
    NoisyOr,
    /// Logistic: `sigmoid(b + sum_f w_f)`; each feature multiplies the
    /// odds by `exp(w_f)`. Weights are learned online (FTRL-Proximal,
    /// L2 towards the Beta-prior seeds).
    Logistic,
}

impl Combine {
    /// `"noisy_or"` or `"logistic"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Combine::NoisyOr => "noisy_or",
            Combine::Logistic => "logistic",
        }
    }

    /// Parse `"noisy_or"` / `"logistic"`.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "noisy_or" => Ok(Combine::NoisyOr),
            "logistic" => Ok(Combine::Logistic),
            _ => Err(Error::invalid(format!(
                "combine must be one of ('noisy_or', 'logistic'), got {}",
                repr_str(s)
            ))),
        }
    }

    /// The credit mode used when none is given: `proportional` (the EM
    /// credit derived for noisy-OR) or `uniform` for logistic (each feature
    /// of a confirmed event counts one positive observation, the count a
    /// multiplicative model wants; the logistic gradient is likewise the
    /// same for every feature present).
    pub fn default_credit(self) -> Credit {
        match self {
            Combine::NoisyOr => Credit::Proportional,
            Combine::Logistic => Credit::Uniform,
        }
    }
}

/// Whose view of a link: the population (global) view, or an entity's
/// (`None` = an entity without evidence of its own).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EntityRef {
    /// The global (population) view: prior + full pooled evidence.
    Global,
    /// An entity's view: prior + shrunk leave-one-out global + own evidence.
    Entity(Option<EntityId>),
}

/// A Beta(alpha, beta) belief, split into where the evidence came from.
///
/// `alpha = prior_alpha + global_pos + pos` (likewise for beta). For an
/// entity view `global_*` is the shrunk evidence actually borrowed; for the
/// global view `global_*` holds the full pooled counts and `pos`/`neg` are 0.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BetaCounts {
    /// Prior alpha.
    pub prior_alpha: f64,
    /// Prior beta.
    pub prior_beta: f64,
    /// Own positive evidence.
    pub pos: f64,
    /// Own negative evidence.
    pub neg: f64,
    /// Borrowed (or pooled) positive evidence.
    pub global_pos: f64,
    /// Borrowed (or pooled) negative evidence.
    pub global_neg: f64,
}

impl BetaCounts {
    /// Posterior alpha.
    pub fn alpha(&self) -> f64 {
        self.prior_alpha + self.global_pos + self.pos
    }

    /// Posterior beta.
    pub fn beta(&self) -> f64 {
        self.prior_beta + self.global_neg + self.neg
    }

    /// Posterior mean.
    pub fn strength(&self) -> f64 {
        let a = self.alpha();
        a / (a + self.beta())
    }

    /// Own + borrowed evidence (excludes the prior).
    pub fn observations(&self) -> f64 {
        self.pos + self.neg + self.global_pos + self.global_neg
    }
}

/// `1 - prod(1 - s)`; 0.0 for an empty iterator.
///
/// The product is taken in a canonical order (ascending strength), so the
/// result depends only on the multiset of strengths: events whose features
/// have equal strengths get bit-identical scores and tie exactly (ties are
/// then broken by event id, deterministically).
pub fn noisy_or(strengths: impl IntoIterator<Item = f64>) -> f64 {
    let mut buf = SmallF64::new();
    for s in strengths {
        buf.push(s);
    }
    buf.noisy_or()
}

/// A tiny stack buffer for the strengths of one feature set.
pub(crate) struct SmallF64 {
    inline: [f64; 8],
    len: usize,
    spill: Vec<f64>,
}

impl SmallF64 {
    pub(crate) fn new() -> Self {
        SmallF64 {
            inline: [0.0; 8],
            len: 0,
            spill: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, x: f64) {
        if self.len < 8 {
            self.inline[self.len] = x;
        } else {
            if self.spill.is_empty() {
                self.spill.extend_from_slice(&self.inline);
            }
            self.spill.push(x);
        }
        self.len += 1;
    }

    /// `1 - prod(1 - s)` over the values in ascending order.
    pub(crate) fn noisy_or(&mut self) -> f64 {
        let v: &mut [f64] = if self.len <= 8 {
            &mut self.inline[..self.len]
        } else {
            &mut self.spill
        };
        v.sort_unstable_by(f64::total_cmp);
        let mut miss = 1.0;
        for &s in v.iter() {
            miss *= 1.0 - s;
        }
        1.0 - miss
    }
}

/// EM responsibilities of each feature for a confirmed noisy-OR effect:
/// `r_f = s_f / (1 - prod(1 - s_g))`, clamped to `[0, 1]`; uniform if the
/// noisy-OR is 0.
pub fn responsibilities(strengths: &[f64]) -> Vec<f64> {
    if strengths.is_empty() {
        return Vec::new();
    }
    // Product in the given order (callers pass features sorted by name), as
    // the reference implementation does, so learned counts match it exactly.
    let mut miss = 1.0;
    for &s in strengths {
        miss *= 1.0 - s;
    }
    let p = 1.0 - miss;
    if p <= 0.0 {
        let share = 1.0 / strengths.len() as f64;
        return vec![share; strengths.len()];
    }
    strengths.iter().map(|&s| (s / p).clamp(0.0, 1.0)).collect()
}

fn pymax0(x: f64) -> f64 {
    // Python's max(0.0, x): the first argument unless x is strictly larger.
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Per-query memo of feature strengths for one [`EntityRef`].
#[derive(Debug)]
///
/// Under `logistic` it also memoises each feature's weight `w_f` and the
/// intercept (which do not depend on the view).
pub struct StrengthCache {
    ent: EntityRef,
    map: FxHashMap<FeatureId, f64>,
    weights: FxHashMap<FeatureId, f64>,
    bias: Option<f64>,
}

impl StrengthCache {
    /// An empty cache for `ent`.
    pub fn new(ent: EntityRef) -> Self {
        StrengthCache {
            ent,
            map: FxHashMap::default(),
            weights: FxHashMap::default(),
            bias: None,
        }
    }

    /// The view this cache is for.
    pub fn entity(&self) -> EntityRef {
        self.ent
    }
}

/// Hierarchical Beta link beliefs for one effect.
#[derive(Clone, Debug)]
pub struct LinkTable {
    default_prior: (f64, f64),
    global_pseudocount: f64,
    hierarchical: bool,
    priors: FxHashMap<FeatureId, (f64, f64)>,
    global: FxHashMap<FeatureId, [f64; 2]>,
    entity: FxHashMap<EntityId, FxIndexMap<FeatureId, [f64; 2]>>,
    joint: FxHashMap<EntityId, JointMap>,
    logistic: Option<Logistic>,
}

impl LinkTable {
    /// A new table. `default_prior` applies to features without a
    /// registered prior; `global_pseudocount` caps how much an entity
    /// borrows from the global layer; `hierarchical = false` disables
    /// borrowing.
    pub fn new(default_prior: (f64, f64), global_pseudocount: f64, hierarchical: bool) -> Result<Self> {
        if !(default_prior.0 > 0.0 && default_prior.1 > 0.0) {
            return Err(Error::invalid("Beta prior parameters must be > 0"));
        }
        if global_pseudocount < 0.0 {
            return Err(Error::invalid("global_pseudocount must be >= 0"));
        }
        Ok(LinkTable {
            default_prior,
            global_pseudocount,
            hierarchical,
            priors: FxHashMap::default(),
            global: FxHashMap::default(),
            entity: FxHashMap::default(),
            joint: FxHashMap::default(),
            logistic: None,
        })
    }

    // -- combine mode ------------------------------------------------------

    /// The combine mode.
    pub fn combine(&self) -> Combine {
        if self.logistic.is_some() {
            Combine::Logistic
        } else {
            Combine::NoisyOr
        }
    }

    /// Switch the combine mode. Switching to `logistic` starts every weight
    /// at its seed (`params` default if `None`); switching to `noisy_or`
    /// drops the weights. The Beta links are kept either way.
    pub fn set_combine(&mut self, c: Combine, params: Option<LogisticParams>) -> Result<()> {
        match c {
            Combine::NoisyOr => self.logistic = None,
            Combine::Logistic => {
                let p = params.unwrap_or_default();
                match &mut self.logistic {
                    Some(l) => l.params = p.validate()?,
                    None => self.logistic = Some(Logistic::new(p)?),
                }
            }
        }
        Ok(())
    }

    /// The logistic state (`None` under `noisy_or`).
    pub fn logistic(&self) -> Option<&Logistic> {
        self.logistic.as_ref()
    }

    pub(crate) fn set_logistic_state(&mut self, l: Logistic) {
        self.logistic = Some(l);
    }

    /// Seed of the intercept: `logit(mean of default_prior)`.
    pub fn bias_seed(&self) -> f64 {
        beta_logit(self.default_prior.0, self.default_prior.1)
    }

    /// Seed of a feature's weight: `logit(prior mean) - logit(default prior
    /// mean)` (0 for features without a registered prior).
    pub fn weight_seed(&self, f: Option<FeatureId>) -> f64 {
        match f.and_then(|f| self.priors.get(&f)) {
            None => 0.0,
            Some(&(a, b)) => beta_logit(a, b) - self.bias_seed(),
        }
    }

    /// Logistic intercept `b` (base log-odds); `None` under `noisy_or`.
    pub fn base_log_odds(&self) -> Option<f64> {
        self.logistic.as_ref().map(|l| self.bias_seed() + l.bias_delta())
    }

    /// Logistic weight `w_f` (log-odds multiplier); `None` under `noisy_or`.
    /// Unknown features (`None`) are at their seed, 0.
    pub fn weight(&self, f: Option<FeatureId>) -> Option<f64> {
        let l = self.logistic.as_ref()?;
        Some(match f {
            None => 0.0,
            Some(f) => self.weight_seed(Some(f)) + l.feature_delta(f),
        })
    }

    fn weight_cached(&self, f: FeatureId, cache: &mut StrengthCache) -> f64 {
        if let Some(&w) = cache.weights.get(&f) {
            return w;
        }
        let w = self.weight(Some(f)).unwrap_or(0.0);
        cache.weights.insert(f, w);
        w
    }

    fn bias_cached(&self, cache: &mut StrengthCache) -> f64 {
        if let Some(b) = cache.bias {
            return b;
        }
        let b = self.base_log_odds().unwrap_or(0.0);
        cache.bias = Some(b);
        b
    }

    /// The link of a feature set under the combine mode (ids in iteration
    /// order). Under `noisy_or` this is exactly
    /// [`LinkTable::noisy_or_cached`].
    pub fn link_cached(&self, fs: &[FeatureId], cache: &mut StrengthCache) -> f64 {
        if self.logistic.is_none() {
            return self.noisy_or_cached(fs, cache);
        }
        let mut z = self.bias_cached(cache);
        for &f in fs {
            z += self.weight_cached(f, cache);
        }
        sigmoid(z)
    }

    /// [`LinkTable::link_cached`] for features that may be unknown (`None`).
    pub fn link_opt(&self, fs: &[Option<FeatureId>], ent: EntityRef) -> f64 {
        if self.logistic.is_none() {
            return self.noisy_or_opt(fs, ent);
        }
        sigmoid(self.log_odds_opt(fs))
    }

    /// Logistic log-odds `b + sum_f w_f` (0 contribution for unknown
    /// features); 0.0 under `noisy_or`.
    pub fn log_odds_opt(&self, fs: &[Option<FeatureId>]) -> f64 {
        let mut z = self.base_log_odds().unwrap_or(0.0);
        for &f in fs {
            z += self.weight(f).unwrap_or(0.0);
        }
        z
    }

    /// An upper bound on the link of *any subset* of `fs` (for segment
    /// bounds): noisy-OR of all of them (noisy-OR only grows with more
    /// features), or `sigmoid(b + sum_f max(0, w_f))` under `logistic`.
    pub fn link_upper_bound(
        &self,
        fs: impl IntoIterator<Item = FeatureId>,
        cache: &mut StrengthCache,
    ) -> f64 {
        if self.logistic.is_none() {
            let mut buf = SmallF64::new();
            for f in fs {
                buf.push(self.strength_cached(f, cache));
            }
            return buf.noisy_or();
        }
        let mut z = self.bias_cached(cache);
        for f in fs {
            let w = self.weight_cached(f, cache);
            if w > 0.0 {
                z += w;
            }
        }
        sigmoid(z)
    }

    /// One logistic training example (no-op under `noisy_or`): the event
    /// with features `fs` gets label `y` with example weight `c`.
    pub fn logistic_step(&mut self, fs: &[FeatureId], y: f64, c: f64, e: Option<EntityId>) {
        if self.logistic.is_none() {
            return;
        }
        let opt: Vec<Option<FeatureId>> = fs.iter().copied().map(Some).collect();
        let p = sigmoid(self.log_odds_opt(&opt));
        if let Some(l) = &mut self.logistic {
            l.step(fs, p, y, c, e);
        }
    }

    /// Default prior of features without a registered prior.
    pub fn default_prior(&self) -> (f64, f64) {
        self.default_prior
    }

    /// Cap on borrowed pseudo-observations.
    pub fn global_pseudocount(&self) -> f64 {
        self.global_pseudocount
    }

    /// Whether entities borrow from the global layer.
    pub fn hierarchical(&self) -> bool {
        self.hierarchical
    }

    /// The prior of a feature (`None`: a feature never seen).
    pub fn prior(&self, f: Option<FeatureId>) -> (f64, f64) {
        f.and_then(|f| self.priors.get(&f).copied())
            .unwrap_or(self.default_prior)
    }

    /// Effective belief for `f` in view `ent`.
    pub fn get(&self, f: Option<FeatureId>, ent: EntityRef) -> BetaCounts {
        let (pa, pb) = self.prior(f);
        let [gp, gn] = f.and_then(|f| self.global.get(&f).copied()).unwrap_or([0.0, 0.0]);
        let e = match ent {
            EntityRef::Global => {
                return BetaCounts {
                    prior_alpha: pa,
                    prior_beta: pb,
                    pos: 0.0,
                    neg: 0.0,
                    global_pos: gp,
                    global_neg: gn,
                }
            }
            EntityRef::Entity(e) => e,
        };
        let [op, on] = match (e, f) {
            (Some(e), Some(f)) => self
                .entity
                .get(&e)
                .and_then(|m| m.get(&f).copied())
                .unwrap_or([0.0, 0.0]),
            _ => [0.0, 0.0],
        };
        let (mut bp, mut bn) = (0.0, 0.0);
        if self.hierarchical && self.global_pseudocount > 0.0 {
            bp = pymax0(gp - op);
            bn = pymax0(gn - on);
            let total = bp + bn;
            if total > self.global_pseudocount {
                let c = self.global_pseudocount / total;
                bp *= c;
                bn *= c;
            }
        }
        BetaCounts {
            prior_alpha: pa,
            prior_beta: pb,
            pos: op,
            neg: on,
            global_pos: bp,
            global_neg: bn,
        }
    }

    /// Posterior mean of `f` in view `ent`.
    pub fn strength(&self, f: Option<FeatureId>, ent: EntityRef) -> f64 {
        self.get(f, ent).strength()
    }

    /// [`LinkTable::strength`] memoised in a per-query cache.
    pub fn strength_cached(&self, f: FeatureId, cache: &mut StrengthCache) -> f64 {
        if let Some(&s) = cache.map.get(&f) {
            return s;
        }
        let s = self.strength(Some(f), cache.ent);
        cache.map.insert(f, s);
        s
    }

    /// Noisy-OR of a feature set (ids in iteration order).
    pub fn noisy_or_cached(&self, fs: &[FeatureId], cache: &mut StrengthCache) -> f64 {
        let mut buf = SmallF64::new();
        for &f in fs {
            buf.push(self.strength_cached(f, cache));
        }
        buf.noisy_or()
    }

    /// Noisy-OR of features that may be unknown (`None`).
    pub fn noisy_or_opt(&self, fs: &[Option<FeatureId>], ent: EntityRef) -> f64 {
        noisy_or(fs.iter().map(|&f| self.strength(f, ent)))
    }

    /// Feature ids with a registered prior or global evidence.
    pub fn known_features(&self) -> impl Iterator<Item = FeatureId> + '_ {
        self.priors.keys().copied().chain(
            self.global
                .keys()
                .copied()
                .filter(|f| !self.priors.contains_key(f)),
        )
    }

    /// Registered feature priors `(feature id, (alpha, beta))`, in no
    /// particular order.
    pub fn priors(&self) -> impl Iterator<Item = (FeatureId, (f64, f64))> + '_ {
        self.priors.iter().map(|(&f, &p)| (f, p))
    }

    /// Number of features with global evidence.
    pub fn num_learned(&self) -> usize {
        self.global.len()
    }

    /// Entities with own evidence.
    pub fn entity_ids(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.entity.keys().copied()
    }

    /// The entity's own `(pos, neg)` counts (its contributions to the
    /// global layer), in insertion order.
    pub fn entity_counts(&self, e: EntityId) -> Vec<(FeatureId, f64, f64)> {
        self.entity
            .get(&e)
            .map(|m| m.iter().map(|(&f, &[p, n])| (f, p, n)).collect())
            .unwrap_or_default()
    }

    /// True if the entity has an entity-layer entry for `f`.
    pub fn has_entity_entry(&self, e: EntityId, f: FeatureId) -> bool {
        self.entity.get(&e).is_some_and(|m| m.contains_key(&f))
    }

    // -- joint links -------------------------------------------------------

    /// Prior of a joint link: noisy-OR of the members' prior means, worth
    /// [`JOINT_PRIOR_STRENGTH`] pseudo-observations.
    pub fn joint_prior(&self, fs: &[Option<FeatureId>]) -> (f64, f64) {
        let p = noisy_or(fs.iter().map(|&f| {
            let (a, b) = self.prior(f);
            a / (a + b)
        }));
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        (JOINT_PRIOR_STRENGTH * p, JOINT_PRIOR_STRENGTH * (1.0 - p))
    }

    /// Joint link of an exact feature set (sorted ids), if ever judged.
    pub fn joint(&self, e: EntityId, fs: &[FeatureId]) -> Option<BetaCounts> {
        let [p, n] = *self.joint.get(&e)?.get(fs)?;
        let opt: Vec<Option<FeatureId>> = fs.iter().copied().map(Some).collect();
        let (pa, pb) = self.joint_prior(&opt);
        Some(BetaCounts {
            prior_alpha: pa,
            prior_beta: pb,
            pos: p,
            neg: n,
            global_pos: 0.0,
            global_neg: 0.0,
        })
    }

    /// Feature sets with a joint link for `e`, in creation order.
    pub fn joint_sets(&self, e: EntityId) -> Vec<&[FeatureId]> {
        self.joint
            .get(&e)
            .map(|m| m.keys().map(|k| &**k).collect())
            .unwrap_or_default()
    }

    /// Add evidence to a joint link (`fs` sorted ids).
    pub fn update_joint(&mut self, e: EntityId, fs: &[FeatureId], pos: f64, neg: f64) {
        let c = self
            .joint
            .entry(e)
            .or_default()
            .entry(fs.into())
            .or_insert([0.0, 0.0]);
        c[0] += pos;
        c[1] += neg;
    }

    // -- writes ------------------------------------------------------------

    /// Set (or replace) the prior of one feature, keeping its counts.
    /// `name` is used in the error message.
    pub fn set_prior(&mut self, f: FeatureId, name: &str, alpha: f64, beta: f64) -> Result<()> {
        if !(alpha > 0.0 && beta > 0.0) {
            return Err(Error::invalid(format!(
                "Beta prior for {} must be > 0",
                repr_str(name)
            )));
        }
        self.priors.insert(f, (alpha, beta));
        Ok(())
    }

    /// Add evidence to the global layer and (if given) the entity layer.
    pub fn update(&mut self, f: FeatureId, pos: f64, neg: f64, e: Option<EntityId>) {
        let g = self.global.entry(f).or_insert([0.0, 0.0]);
        g[0] += pos;
        g[1] += neg;
        if let Some(e) = e {
            let c = self.entity.entry(e).or_default().entry(f).or_insert([0.0, 0.0]);
            c[0] += pos;
            c[1] += neg;
        }
    }

    /// Record a confirmed cause. `fs` must be sorted by feature *name* and
    /// unique. Returns each feature's credit `r_f` (same order).
    pub fn confirm(&mut self, fs: &[FeatureId], e: Option<EntityId>, credit: Credit) -> Vec<f64> {
        let resp = match credit {
            Credit::Uniform => vec![1.0; fs.len()],
            Credit::Proportional => {
                let ent = EntityRef::Entity(e);
                let s: Vec<f64> = fs.iter().map(|&f| self.strength(Some(f), ent)).collect();
                responsibilities(&s)
            }
        };
        for (&f, &r) in fs.iter().zip(&resp) {
            self.update(f, r, 1.0 - r, e);
        }
        resp
    }

    /// Record a rejected candidate: every feature independently failed.
    pub fn reject(&mut self, fs: &[FeatureId], e: Option<EntityId>) {
        for &f in fs {
            self.update(f, 0.0, 1.0, e);
        }
    }

    /// Drop an entity's links and joint links; with `scrub_global` also
    /// subtract its contributions from the global layer. Returns the
    /// number of features dropped.
    pub fn forget_entity(&mut self, e: EntityId, scrub_global: bool) -> usize {
        let own = self.entity.remove(&e).unwrap_or_default();
        self.joint.remove(&e);
        if let Some(l) = &mut self.logistic {
            l.forget_entity(e, scrub_global);
        }
        if scrub_global {
            for (f, [p, n]) in &own {
                if let Some(g) = self.global.get_mut(f) {
                    g[0] = pymax0(g[0] - p);
                    g[1] = pymax0(g[1] - n);
                    if g[0] < 1e-12 && g[1] < 1e-12 {
                        self.global.remove(f);
                    }
                }
            }
        }
        own.len()
    }

    // -- (de)serialisation -------------------------------------------------

    /// Snapshot form (names sorted, as in the reference implementation).
    pub fn to_json(&self, vocab: &Vocab) -> Map<String, Value> {
        let fname = |f: &FeatureId| vocab.features.name(*f).to_string();
        let ename = |e: &EntityId| vocab.entities.name(*e).to_string();
        let mut m = Map::new();
        m.insert(
            "default_prior".into(),
            pair(self.default_prior.0, self.default_prior.1),
        );
        m.insert("global_pseudocount".into(), num(self.global_pseudocount));
        m.insert("hierarchical".into(), Value::Bool(self.hierarchical));
        let mut priors: Vec<(String, (f64, f64))> = self.priors.iter().map(|(f, &p)| (fname(f), p)).collect();
        priors.sort_by(|a, b| a.0.cmp(&b.0));
        m.insert(
            "priors".into(),
            Value::Object(priors.into_iter().map(|(f, (a, b))| (f, pair(a, b))).collect()),
        );
        let mut global: Vec<(String, [f64; 2])> = self.global.iter().map(|(f, &c)| (fname(f), c)).collect();
        global.sort_by(|a, b| a.0.cmp(&b.0));
        m.insert(
            "global".into(),
            Value::Object(global.into_iter().map(|(f, [p, n])| (f, pair(p, n))).collect()),
        );
        let mut ents: Vec<(String, &FxIndexMap<FeatureId, [f64; 2]>)> =
            self.entity.iter().map(|(e, fs)| (ename(e), fs)).collect();
        ents.sort_by(|a, b| a.0.cmp(&b.0));
        let ents: Map<String, Value> = ents
            .into_iter()
            .map(|(e, fs)| {
                let mut rows: Vec<(String, [f64; 2])> = fs.iter().map(|(f, &c)| (fname(f), c)).collect();
                rows.sort_by(|a, b| a.0.cmp(&b.0));
                (
                    e,
                    Value::Object(rows.into_iter().map(|(f, [p, n])| (f, pair(p, n))).collect()),
                )
            })
            .collect();
        m.insert("entities".into(), Value::Object(ents));
        let mut joint: Vec<(String, &JointMap)> = self.joint.iter().map(|(e, js)| (ename(e), js)).collect();
        joint.sort_by(|a, b| a.0.cmp(&b.0));
        let joint: Map<String, Value> = joint
            .into_iter()
            .map(|(e, js)| {
                let mut rows: Vec<(Vec<String>, [f64; 2])> = js
                    .iter()
                    .map(|(fs, &c)| {
                        let mut names: Vec<String> = fs.iter().map(fname).collect();
                        names.sort();
                        (names, c)
                    })
                    .collect();
                rows.sort_by(|a, b| a.0.cmp(&b.0));
                let arr = rows
                    .into_iter()
                    .map(|(names, [p, n])| {
                        Value::Array(vec![
                            Value::Array(names.into_iter().map(Value::from).collect()),
                            num(p),
                            num(n),
                        ])
                    })
                    .collect();
                (e, Value::Array(arr))
            })
            .collect();
        m.insert("joint".into(), Value::Object(joint));
        m
    }

    /// Inverse of [`LinkTable::to_json`]; also reads v0.1 tables
    /// (`{"default_prior", "links": {f: [pa, pb, pos, neg]}}`).
    pub fn from_json(d: &Map<String, Value>, vocab: &mut Vocab) -> Result<Self> {
        let bad = |what: &str| Error::Snapshot(format!("bad link table: {what}"));
        let dp = d
            .get("default_prior")
            .and_then(two)
            .ok_or_else(|| bad("default_prior"))?;
        let pc = d.get("global_pseudocount").and_then(as_f64).unwrap_or(5.0);
        let hier = d.get("hierarchical").and_then(Value::as_bool).unwrap_or(true);
        let mut t = LinkTable::new(dp, pc, hier)?;
        if let Some(Value::Object(links)) = d.get("links") {
            for (f, v) in links {
                let arr = v.as_array().ok_or_else(|| bad("links"))?;
                let x: Vec<f64> = arr.iter().filter_map(as_f64).collect();
                if x.len() != 4 {
                    return Err(bad("links"));
                }
                let id = vocab.features.intern(f);
                t.priors.insert(id, (x[0], x[1]));
                if x[2] != 0.0 || x[3] != 0.0 {
                    t.global.insert(id, [x[2], x[3]]);
                }
            }
            return Ok(t);
        }
        if let Some(Value::Object(p)) = d.get("priors") {
            for (f, v) in p {
                let (a, b) = two(v).ok_or_else(|| bad("priors"))?;
                t.priors.insert(vocab.features.intern(f), (a, b));
            }
        }
        if let Some(Value::Object(g)) = d.get("global") {
            for (f, v) in g {
                let (a, b) = two(v).ok_or_else(|| bad("global"))?;
                t.global.insert(vocab.features.intern(f), [a, b]);
            }
        }
        if let Some(Value::Object(es)) = d.get("entities") {
            for (e, fs) in es {
                let eid = vocab.entities.intern(e);
                let fs = fs.as_object().ok_or_else(|| bad("entities"))?;
                let m = t.entity.entry(eid).or_default();
                for (f, v) in fs {
                    let (a, b) = two(v).ok_or_else(|| bad("entities"))?;
                    m.insert(vocab.features.intern(f), [a, b]);
                }
            }
        }
        if let Some(Value::Object(js)) = d.get("joint") {
            for (e, rows) in js {
                let eid = vocab.entities.intern(e);
                let rows = rows.as_array().ok_or_else(|| bad("joint"))?;
                let m = t.joint.entry(eid).or_default();
                for r in rows {
                    let r = r
                        .as_array()
                        .filter(|r| r.len() == 3)
                        .ok_or_else(|| bad("joint"))?;
                    let names = r[0].as_array().ok_or_else(|| bad("joint"))?;
                    let mut ids = Vec::with_capacity(names.len());
                    for n in names {
                        ids.push(vocab.features.intern(n.as_str().ok_or_else(|| bad("joint"))?));
                    }
                    ids.sort_unstable();
                    ids.dedup();
                    let p = as_f64(&r[1]).ok_or_else(|| bad("joint"))?;
                    let n = as_f64(&r[2]).ok_or_else(|| bad("joint"))?;
                    m.insert(ids.into_boxed_slice(), [p, n]);
                }
            }
        }
        Ok(t)
    }
}

fn two(v: &Value) -> Option<(f64, f64)> {
    let a = v.as_array()?;
    if a.len() != 2 {
        return None;
    }
    Some((as_f64(&a[0])?, as_f64(&a[1])?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn responsibilities_formula() {
        let r = responsibilities(&[0.8, 0.2]);
        let p = 1.0 - 0.2 * 0.8;
        assert!((r[0] - 0.8 / p).abs() < 1e-12 && (r[1] - 0.2 / p).abs() < 1e-12);
        assert_eq!(responsibilities(&[0.0, 0.0]), vec![0.5, 0.5]);
        assert_eq!(responsibilities(&[1.0, 1.0]), vec![1.0, 1.0]);
        assert!(responsibilities(&[]).is_empty());
    }

    #[test]
    fn hierarchical_borrowing_is_leave_one_out_and_capped() {
        let mut t = LinkTable::new((1.0, 4.0), 5.0, true).unwrap();
        for _ in 0..40 {
            t.update(0, 1.0, 0.0, Some(1));
        }
        let fresh = t.get(Some(0), EntityRef::Entity(None));
        assert!((fresh.global_pos + fresh.global_neg - 5.0).abs() < 1e-12);
        assert!((fresh.strength() - 0.6).abs() < 1e-12);
        let own = t.get(Some(0), EntityRef::Entity(Some(1)));
        assert_eq!((own.global_pos, own.pos), (0.0, 40.0));
        assert!((t.strength(Some(0), EntityRef::Global) - 41.0 / 45.0).abs() < 1e-12);
        assert_eq!(t.forget_entity(1, true), 1);
        assert_eq!(t.get(Some(0), EntityRef::Global).observations(), 0.0);
    }

    #[test]
    fn json_roundtrip() {
        let mut v = Vocab::default();
        let mut t = LinkTable::new((1.0, 4.0), 3.0, false).unwrap();
        let (a, b) = (v.features.intern("a"), v.features.intern("b"));
        let e = v.entities.intern("me");
        t.set_prior(a, "a", 3.0, 1.0).unwrap();
        t.update(b, 0.25, 0.75, Some(e));
        t.update_joint(e, &[a, b], 1.0, 0.0);
        let j = t.to_json(&v);
        let mut v2 = Vocab::default();
        let t2 = LinkTable::from_json(&j, &mut v2).unwrap();
        assert_eq!(t2.to_json(&v2), j);
    }
}
