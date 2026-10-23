//! The `logistic` combine mode: an online logistic regression over an
//! event's (binary) features.
//!
//! # Model
//!
//! ```text
//! link(e) = sigmoid(b + sum_{f in F(e)} w_f)
//! score(e | E) = kernel(lag) x weight x link(e) x rules
//! ```
//!
//! `b` is the effect's intercept (its *base log-odds*), `w_f` a feature's
//! log-odds multiplier: a feature multiplies the odds that the event caused
//! the effect by `exp(w_f)`, independently of the event's other features.
//! That is the multiplicative-evidence counterpart of noisy-OR (which adds
//! probabilities of independent causes, `1 - prod(1 - s_f)`).
//!
//! **Seeds (from the Beta priors).** Before any feedback,
//! `b = logit(mean of default_prior)` and
//! `w_f = logit(mean of prior_f) - logit(mean of default_prior)`: 0 for a
//! feature without a registered prior, and for a registered one the
//! log-odds ratio of its prior against a typical feature. So before
//! learning, single-feature events rank exactly as under noisy-OR.
//!
//! **Learning.** Every labelled event is one training example of an
//! FTRL-Proximal step (McMahan et al. 2013) per coordinate (the intercept
//! and the event's features) on the log-loss, with an L2 penalty that
//! pulls each weight towards its seed:
//!
//! ```text
//! p   = link(e)                     (current weights)
//! g   = c * (p - y)                 (y = 1 confirmed, 0 rejected; c = example weight)
//! n_k += g^2;  z_k += g - sigma_k * (w_k - seed_k),  sigma_k = (sqrt(n_k) - sqrt(n_k - g^2)) / lr
//! w_k = seed_k - z_k / ((FTRL_BETA + sqrt(n_k)) / lr + l2)
//! ```
//!
//! `feedback` steps on the confirmed event (y = 1), then on each rejected
//! event in the given order (y = 0). `observe` steps on each event in the
//! window with y = 1 and c = its responsibility; `settle` steps with y = 0
//! and c = the negative evidence it adds (a later `observe` that undoes a
//! settle steps with c < 0).
//!
//! Weights are **pooled across entities** (one weight vector per effect);
//! the Beta links keep their hierarchical, per-entity evidence and are what
//! `link_report` / `LINKS OF` show next to the weights. Each entity's
//! contributions to `z` and `n` are recorded so that
//! `forget_entity(scrub_global = true)` can subtract them. That removes the
//! entity's own gradient terms, but it is not exact unlearning: later steps
//! by other entities were taken at weights the entity had influenced.

use indexmap::IndexMap;
use rustc_hash::{FxBuildHasher, FxHashMap};
use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::intern::Vocab;
use crate::links::{EntityId, FeatureId};
use crate::pyfmt::repr_f64;
use crate::snapshot::{as_f64, num, pair};

type FxIndexMap<K, V> = IndexMap<K, V, FxBuildHasher>;

/// FTRL-Proximal `beta` (the per-coordinate learning rate starts at
/// `learning_rate / FTRL_BETA`).
pub const FTRL_BETA: f64 = 1.0;
/// Default learning rate (FTRL `alpha`).
pub const DEFAULT_LEARNING_RATE: f64 = 0.1;
/// Default L2 penalty towards the seeds.
pub const DEFAULT_L2: f64 = 1.0;

/// Hyperparameters of the `logistic` combine mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogisticParams {
    /// FTRL learning rate `alpha` (> 0).
    pub learning_rate: f64,
    /// L2 penalty pulling weights towards their seeds (>= 0).
    pub l2: f64,
}

impl Default for LogisticParams {
    fn default() -> Self {
        LogisticParams {
            learning_rate: DEFAULT_LEARNING_RATE,
            l2: DEFAULT_L2,
        }
    }
}

impl LogisticParams {
    /// Validate (`learning_rate > 0`, `l2 >= 0`, both finite).
    pub fn validate(self) -> Result<Self> {
        if !(self.learning_rate > 0.0 && self.learning_rate.is_finite()) {
            return Err(Error::invalid(format!(
                "learning_rate must be a finite number > 0, got {}",
                repr_f64(self.learning_rate)
            )));
        }
        if !(self.l2 >= 0.0 && self.l2.is_finite()) {
            return Err(Error::invalid(format!(
                "l2 must be a finite number >= 0, got {}",
                repr_f64(self.l2)
            )));
        }
        Ok(self)
    }
}

/// `1 / (1 + exp(-x))`, without overflow for large `|x|`.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `ln(a / b)`: the log-odds of a Beta(a, b) mean, `logit(a / (a + b))`,
/// without forming `1 - mean`.
pub fn beta_logit(a: f64, b: f64) -> f64 {
    a.ln() - b.ln()
}

/// One FTRL coordinate: `[z, n]`.
pub(crate) type Coord = [f64; 2];

/// What one entity contributed to the coordinates (for `forget_entity`).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Ledger {
    pub bias: Coord,
    pub coords: FxIndexMap<FeatureId, Coord>,
}

/// FTRL state of the logistic combine mode (one per effect).
#[derive(Clone, Debug, PartialEq)]
pub struct Logistic {
    pub(crate) params: LogisticParams,
    pub(crate) bias: Coord,
    pub(crate) coords: FxHashMap<FeatureId, Coord>,
    pub(crate) ledger: FxHashMap<EntityId, Ledger>,
}

impl Logistic {
    /// Fresh state (every weight at its seed).
    pub fn new(params: LogisticParams) -> Result<Self> {
        Ok(Logistic {
            params: params.validate()?,
            bias: [0.0, 0.0],
            coords: FxHashMap::default(),
            ledger: FxHashMap::default(),
        })
    }

    /// Hyperparameters.
    pub fn params(&self) -> LogisticParams {
        self.params
    }

    /// The learned offset from the seed, `-z / ((beta + sqrt(n)) / lr + l2)`.
    pub(crate) fn delta(&self, c: Coord) -> f64 {
        let [z, n] = c;
        if z == 0.0 {
            return 0.0;
        }
        -z / ((FTRL_BETA + n.sqrt()) / self.params.learning_rate + self.params.l2)
    }

    /// Learned offset of the intercept.
    pub fn bias_delta(&self) -> f64 {
        self.delta(self.bias)
    }

    /// Learned offset of a feature (0 if never stepped).
    pub fn feature_delta(&self, f: FeatureId) -> f64 {
        self.coords.get(&f).map_or(0.0, |&c| self.delta(c))
    }

    /// Number of features with learned state.
    pub fn num_features(&self) -> usize {
        self.coords.len()
    }

    /// One FTRL step on the coordinate `k` (`None`: the intercept) with
    /// gradient `g`; records the contribution in `e`'s ledger.
    fn step_coord(&mut self, k: Option<FeatureId>, g: f64, e: Option<EntityId>) {
        let lr = self.params.learning_rate;
        let cur = match k {
            None => self.bias,
            Some(f) => self.coords.get(&f).copied().unwrap_or([0.0, 0.0]),
        };
        let d = self.delta(cur);
        let n_new = cur[1] + g * g;
        let sigma = (n_new.sqrt() - cur[1].sqrt()) / lr;
        let dz = g - sigma * d;
        let dn = g * g;
        match k {
            None => {
                self.bias[0] += dz;
                self.bias[1] = n_new;
            }
            Some(f) => {
                let c = self.coords.entry(f).or_insert([0.0, 0.0]);
                c[0] += dz;
                c[1] = n_new;
            }
        }
        if let Some(e) = e {
            let l = self.ledger.entry(e).or_default();
            let c = match k {
                None => &mut l.bias,
                Some(f) => l.coords.entry(f).or_insert([0.0, 0.0]),
            };
            c[0] += dz;
            c[1] += dn;
        }
    }

    /// One training example: features `fs` (unique ids), soft label `y` in
    /// `[0, 1]`, example weight `c` (may be negative to take back earlier
    /// negative evidence), `p` = the current link of the example.
    pub(crate) fn step(&mut self, fs: &[FeatureId], p: f64, y: f64, c: f64, e: Option<EntityId>) {
        let g = c * (p - y);
        if g == 0.0 || !g.is_finite() {
            return;
        }
        self.step_coord(None, g, e);
        for &f in fs {
            self.step_coord(Some(f), g, e);
        }
    }

    /// Drop an entity's ledger; with `scrub_global` also subtract its
    /// contributions from the coordinates (see the module docs).
    pub(crate) fn forget_entity(&mut self, e: EntityId, scrub_global: bool) {
        let Some(l) = self.ledger.remove(&e) else {
            return;
        };
        if !scrub_global {
            return;
        }
        self.bias[0] -= l.bias[0];
        self.bias[1] = (self.bias[1] - l.bias[1]).max(0.0);
        for (f, [dz, dn]) in l.coords {
            if let Some(c) = self.coords.get_mut(&f) {
                c[0] -= dz;
                c[1] = (c[1] - dn).max(0.0);
                if c[1] < 1e-24 && c[0].abs() < 1e-12 {
                    self.coords.remove(&f);
                }
            }
        }
    }

    // -- (de)serialisation -------------------------------------------------

    /// Snapshot form (names sorted).
    pub(crate) fn to_json(&self, vocab: &Vocab) -> Map<String, Value> {
        let fname = |f: &FeatureId| vocab.features.name(*f).to_string();
        let ename = |e: &EntityId| vocab.entities.name(*e).to_string();
        let coords = |m: &mut dyn Iterator<Item = (String, Coord)>| -> Value {
            let mut rows: Vec<(String, Coord)> = m.collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(rows.into_iter().map(|(f, [z, n])| (f, pair(z, n))).collect())
        };
        let mut m = Map::new();
        m.insert("learning_rate".into(), num(self.params.learning_rate));
        m.insert("l2".into(), num(self.params.l2));
        m.insert("bias".into(), pair(self.bias[0], self.bias[1]));
        m.insert(
            "weights".into(),
            coords(&mut self.coords.iter().map(|(f, &c)| (fname(f), c))),
        );
        let mut ents: Vec<(String, &Ledger)> = self.ledger.iter().map(|(e, l)| (ename(e), l)).collect();
        ents.sort_by(|a, b| a.0.cmp(&b.0));
        let ledger: Map<String, Value> = ents
            .into_iter()
            .map(|(e, l)| {
                let mut x = Map::new();
                x.insert("bias".into(), pair(l.bias[0], l.bias[1]));
                x.insert(
                    "weights".into(),
                    coords(&mut l.coords.iter().map(|(f, &c)| (fname(f), c))),
                );
                (e, Value::Object(x))
            })
            .collect();
        m.insert("ledger".into(), Value::Object(ledger));
        m
    }

    /// Inverse of [`Logistic::to_json`].
    pub(crate) fn from_json(d: &Map<String, Value>, vocab: &mut Vocab) -> Result<Self> {
        let bad = |what: &str| Error::Snapshot(format!("bad logistic state: {what}"));
        let two = |v: &Value| -> Option<Coord> {
            let a = v.as_array()?;
            (a.len() == 2).then_some([as_f64(&a[0])?, as_f64(&a[1])?])
        };
        let params = LogisticParams {
            learning_rate: d
                .get("learning_rate")
                .and_then(as_f64)
                .unwrap_or(DEFAULT_LEARNING_RATE),
            l2: d.get("l2").and_then(as_f64).unwrap_or(DEFAULT_L2),
        };
        let mut s = Logistic::new(params)?;
        if let Some(b) = d.get("bias") {
            s.bias = two(b).ok_or_else(|| bad("bias"))?;
        }
        if let Some(Value::Object(w)) = d.get("weights") {
            for (f, v) in w {
                s.coords
                    .insert(vocab.features.intern(f), two(v).ok_or_else(|| bad("weights"))?);
            }
        }
        if let Some(Value::Object(l)) = d.get("ledger") {
            for (e, v) in l {
                let v = v.as_object().ok_or_else(|| bad("ledger"))?;
                let mut led = Ledger::default();
                if let Some(b) = v.get("bias") {
                    led.bias = two(b).ok_or_else(|| bad("ledger"))?;
                }
                if let Some(Value::Object(w)) = v.get("weights") {
                    for (f, c) in w {
                        led.coords
                            .insert(vocab.features.intern(f), two(c).ok_or_else(|| bad("ledger"))?);
                    }
                }
                s.ledger.insert(vocab.entities.intern(e), led);
            }
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_is_stable_and_bounded() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(800.0) == 1.0 && sigmoid(-800.0) == 0.0);
        assert!(sigmoid(-30.0) > 0.0 && sigmoid(30.0) < 1.0);
        for x in [-20.0, -3.0, -0.1, 0.1, 3.0, 20.0] {
            assert!((sigmoid(x) + sigmoid(-x) - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn beta_logit_matches_logit_of_mean() {
        for (a, b) in [(1.0, 4.0), (3.0, 1.0), (1e-3, 10.0), (1e6, 1e-6)] {
            let m: f64 = a / (a + b);
            let l = (m / (1.0 - m)).ln();
            assert!((beta_logit(a, b) - l).abs() < 1e-6 * (1.0 + l.abs()), "{a} {b}");
        }
        // Extreme counts stay finite (1 - mean would round to 0).
        assert!(beta_logit(1e300, 1e-300).is_finite());
    }

    #[test]
    fn params_are_validated() {
        for (lr, l2) in [
            (0.0, 1.0),
            (-1.0, 1.0),
            (f64::NAN, 1.0),
            (f64::INFINITY, 1.0),
            (0.1, -1.0),
            (0.1, f64::NAN),
        ] {
            assert!(
                Logistic::new(LogisticParams {
                    learning_rate: lr,
                    l2
                })
                .is_err(),
                "{lr} {l2}"
            );
        }
        assert!(Logistic::new(LogisticParams::default()).is_ok());
    }

    #[test]
    fn a_positive_step_raises_and_a_negative_step_lowers() {
        let mut s = Logistic::new(LogisticParams::default()).unwrap();
        s.step(&[1, 2], 0.5, 1.0, 1.0, Some(7));
        assert!(s.feature_delta(1) > 0.0 && s.feature_delta(2) > 0.0 && s.bias_delta() > 0.0);
        // First step from 0: w = lr * |g| / (beta + |g|) / (1 + l2 * lr / (beta + |g|)).
        let g: f64 = 0.5;
        let want = g / ((FTRL_BETA + g) / 0.1 + 1.0);
        assert!((s.feature_delta(1) - want).abs() < 1e-15);
        let mut t = Logistic::new(LogisticParams::default()).unwrap();
        t.step(&[1], 0.5, 0.0, 1.0, None);
        assert!(t.feature_delta(1) < 0.0);
        // Zero gradient (already right) and zero weight are no-ops.
        t.step(&[3], 1.0, 1.0, 1.0, None);
        t.step(&[3], 0.2, 1.0, 0.0, None);
        assert_eq!(t.feature_delta(3), 0.0);
        assert!(!t.coords.contains_key(&3));
    }

    #[test]
    fn scrub_subtracts_the_ledger() {
        let mut s = Logistic::new(LogisticParams::default()).unwrap();
        s.step(&[1], 0.5, 1.0, 1.0, Some(1));
        let before = s.clone();
        s.step(&[1, 2], 0.4, 0.0, 1.0, Some(2));
        s.forget_entity(2, true);
        assert!((s.feature_delta(1) - before.feature_delta(1)).abs() < 1e-12);
        assert_eq!(s.feature_delta(2), 0.0);
        s.forget_entity(1, false);
        assert!(s.ledger.is_empty() && s.feature_delta(1) > 0.0);
    }

    #[test]
    fn json_roundtrip() {
        let mut v = Vocab::default();
        let (a, b) = (v.features.intern("a"), v.features.intern("b"));
        let e = v.entities.intern("me");
        let mut s = Logistic::new(LogisticParams {
            learning_rate: 0.5,
            l2: 2.0,
        })
        .unwrap();
        s.step(&[a, b], 0.3, 1.0, 1.0, Some(e));
        s.step(&[b], 0.6, 0.0, 1.0, None);
        let j = s.to_json(&v);
        let mut v2 = Vocab::default();
        let s2 = Logistic::from_json(&j, &mut v2).unwrap();
        assert_eq!(s2.to_json(&v2), j);
        assert_eq!(s2.params, s.params);
    }
}
