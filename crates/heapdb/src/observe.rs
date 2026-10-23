//! Observational learning: `observe`, `settle`, `link_report`.
//!
//! * `observe(entity, effect, at)`: the effect happened, cause unknown. One
//!   soft-EM step: every event `e` in the kernel window gets
//!   `r_e = score_e / (leak + sum score)` and each feature proportional
//!   credit scaled by `r_e`.
//! * `settle(until)`: events whose kernel window passed without the effect
//!   get negative evidence `beta_f += w_neg * rules(e)`; incremental through
//!   per-(entity, effect) cursors.
//! * Under `combine = logistic` the same evidence also trains the weights:
//!   `observe` is a positive example per event weighted by `r_e`, `settle`
//!   a negative one weighted by the evidence added (see
//!   [`crate::logistic`]).
//! * `link_report(effect)`: features ranked by learned strength, with Beta
//!   credible intervals and lift. **Associations under the model, not
//!   causal effects.**

use indexmap::IndexMap;

use crate::error::{Error, Result};
use crate::event::{EventRef, Features, EMPTY_ATTRS};
use crate::links::{responsibilities, Credit, EntityRef, FeatureId, StrengthCache};
use crate::pyfmt::py_sum;
use crate::results::{EventCredit, LinkReport, LinkRow, ObserveReport, SettleReport};
use crate::segments::EntityLog;
use crate::stats::beta_interval;
use crate::store::{check_leak, window, EffectSpec, HeapDB};

/// Baseline for `lift` in a [`LinkReport`].
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Baseline {
    /// Mean strength of the reported features.
    #[default]
    Mean,
    /// The effect's default prior mean.
    Prior,
    /// A fixed number.
    Value(f64),
}

/// Options for [`HeapDB::link_report`].
#[derive(Clone, Debug)]
pub struct LinkReportOptions {
    /// Entity view (`None`: global).
    pub entity: Option<String>,
    /// Only features with at least this much evidence.
    pub min_support: f64,
    /// Credible level of the interval.
    pub level: f64,
    /// Baseline for lift.
    pub baseline: Baseline,
    /// Report exactly these features (default: all with a prior or evidence).
    pub features: Option<Vec<String>>,
}

impl Default for LinkReportOptions {
    fn default() -> Self {
        LinkReportOptions {
            entity: None,
            min_support: 1.0,
            level: 0.9,
            baseline: Baseline::Mean,
            features: None,
        }
    }
}

struct Scored {
    id: u64,
    feats: Vec<FeatureId>,
    by_name: Vec<(FeatureId, f64)>,
    score: f64,
}

fn shares(
    spec: &EffectSpec,
    db: &HeapDB,
    feats: &[FeatureId],
    cache: &mut StrengthCache,
) -> Vec<(FeatureId, f64)> {
    let mut by_name = feats.to_vec();
    db.vocab.sort_by_name(&mut by_name);
    match spec.credit {
        Credit::Uniform => by_name.into_iter().map(|f| (f, 1.0)).collect(),
        Credit::Proportional => {
            let s: Vec<f64> = by_name
                .iter()
                .map(|&f| spec.links.strength_cached(f, cache))
                .collect();
            by_name.into_iter().zip(responsibilities(&s)).collect()
        }
    }
}

fn evref<'a>(db: &'a HeapDB, eid: u32, log: &'a EntityLog, ord: u32) -> EventRef<'a> {
    let ev = &log.events[ord as usize];
    EventRef {
        id: ev.id,
        entity: db.vocab.entities.name(eid),
        time: ev.time,
        event_type: db.vocab.types.name(ev.ty),
        features: Features::Interned(log.feats(ev), &db.vocab.features),
        weight: ev.weight,
        attrs: ev.attrs.as_deref().unwrap_or(&EMPTY_ATTRS),
    }
}

impl HeapDB {
    /// Record an occurrence of `effect` with **no labelled cause** and learn
    /// from it by one soft-EM step (see the module docs). `leak` overrides
    /// the effect's background score for this call.
    pub fn observe(
        &mut self,
        entity: &str,
        effect: &str,
        at: f64,
        leak: Option<f64>,
    ) -> Result<ObserveReport> {
        let spec = self.effect(effect)?;
        let lk = match leak {
            Some(l) => check_leak(l)?,
            None => spec.leak,
        };
        let eid = self.vocab.entities.get(entity);
        let mut scored: Vec<Scored> = Vec::new();
        if let Some(log) = eid.and_then(|e| self.logs.get(&e)) {
            let e = eid.expect("log implies id");
            let (t_lo, t_hi) = window(spec, at, None);
            let mut cache = StrengthCache::new(EntityRef::Entity(eid));
            for seg in log.segments_between(t_lo, t_hi) {
                for &ord in &seg.ords {
                    let ev = &log.events[ord as usize];
                    if !(t_lo <= ev.time && ev.time <= t_hi) {
                        continue;
                    }
                    let s = self.score_event_ord(spec, e, log, ord, at, &mut cache)?;
                    if s > 0.0 {
                        let feats = log.feats(ev).to_vec();
                        let by_name = shares(spec, self, &feats, &mut cache);
                        scored.push(Scored {
                            id: ev.id,
                            feats,
                            by_name,
                            score: s,
                        });
                    }
                }
            }
        }
        let total = lk + py_sum(scored.iter().map(|s| s.score));
        if lk.is_infinite() {
            scored.clear();
        }
        let names: Vec<Vec<String>> = scored
            .iter()
            .map(|s| {
                s.by_name
                    .iter()
                    .map(|&(f, _)| self.vocab.features.name(f).to_string())
                    .collect()
            })
            .collect();
        let spec = self.effects.get_mut(effect).expect("checked");
        let mut credits = Vec::with_capacity(scored.len());
        for (sc, names) in scored.iter().zip(names) {
            let r = sc.score / total;
            let undo = spec.obs.rejected.remove(&sc.id).unwrap_or(0.0);
            let mut sorted = sc.feats.clone();
            sorted.sort_unstable();
            if undo != 0.0 {
                for &f in &sc.feats {
                    spec.links.update(f, 0.0, -undo, eid);
                }
                // Logistic: take back the negative example `settle` added.
                spec.links.logistic_step(&sorted, 0.0, -undo, eid);
                spec.obs.bump("rejections_undone", 1.0);
            }
            // Logistic: a positive example weighted by its responsibility.
            spec.links.logistic_step(&sorted, 1.0, r, eid);
            let mut added = Vec::with_capacity(names.len());
            for (&(f, sh), name) in sc.by_name.iter().zip(names) {
                spec.links.update(f, r * sh, r * (1.0 - sh), eid);
                added.push((name, r * sh));
            }
            *spec.obs.credited.entry(sc.id).or_insert(0.0) += r;
            credits.push(EventCredit {
                event_id: sc.id,
                score: sc.score,
                responsibility: r,
                features: added,
            });
        }
        credits.sort_by(|a, b| {
            b.responsibility
                .partial_cmp(&a.responsibility)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.event_id.cmp(&a.event_id))
        });
        let occ = spec.obs.occurrences.entry(entity.to_string()).or_default();
        let pos = occ.partition_point(|&t| t <= at);
        occ.insert(pos, at);
        let leak_r = if lk.is_infinite() { 1.0 } else { lk / total };
        spec.obs.bump("occurrences", 1.0);
        spec.obs.bump("leak_responsibility", leak_r);
        spec.obs.bump("credited_mass", 1.0 - leak_r);
        Ok(ObserveReport {
            entity: entity.to_string(),
            effect: effect.to_string(),
            at,
            leak: lk,
            leak_responsibility: leak_r,
            credits,
        })
    }

    pub(crate) fn score_event_ord(
        &self,
        spec: &EffectSpec,
        eid: u32,
        log: &EntityLog,
        ord: u32,
        at: f64,
        cache: &mut StrengthCache,
    ) -> Result<f64> {
        let ev = &log.events[ord as usize];
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
        let mut s = k * ev.weight * spec.links.link_cached(log.feats(ev), cache);
        if spec.rules.is_empty() {
            return Ok(s);
        }
        let r = evref(self, eid, log, ord);
        for rule in &spec.rules {
            if s == 0.0 {
                break;
            }
            if rule.matches(Some(ev.ty), &r).map_err(Error::Callback)? {
                s *= rule.multiplier;
            }
        }
        Ok(s)
    }

    /// Add negative evidence for events whose window passed without the
    /// effect (see the module docs). `effects = None`: all effects;
    /// `entity = None`: all entities. With `partial`, events whose window
    /// has only partly elapsed get evidence scaled by the elapsed kernel
    /// mass (topped up later, undone by a later `observe`).
    pub fn settle(
        &mut self,
        until: f64,
        effects: Option<&[&str]>,
        entity: Option<&str>,
        partial: bool,
    ) -> Result<SettleReport> {
        let names: Vec<String> = match effects {
            None => self.effects(),
            Some(e) => e.iter().map(|s| s.to_string()).collect(),
        };
        let ents: Vec<String> = match entity {
            Some(e) => vec![e.to_string()],
            None => self
                .logs
                .keys()
                .map(|&e| self.vocab.entities.name(e).to_string())
                .collect(),
        };
        let (mut n_settled, mut n_rej, mut n_cred, mut n_obs) = (0usize, 0usize, 0usize, 0usize);
        let mut w_total = 0.0;
        let mut per: IndexMap<String, usize> = IndexMap::new();
        for name in &names {
            // Split borrows: the spec mutably, logs and vocab immutably.
            let HeapDB {
                effects: specs,
                logs,
                vocab,
                id_index,
                ..
            } = self;
            let spec = specs
                .get_mut(name.as_str())
                .ok_or_else(|| Error::UnknownEffect(name.clone()))?;
            per.insert(name.clone(), 0);
            let support = spec.kernel.support();
            for ent in &ents {
                let Some(eid) = vocab.entities.get(ent) else {
                    continue;
                };
                let Some(log) = logs.get(&eid) else { continue };
                let cur = spec.obs.cursors.get(ent).copied().unwrap_or(f64::NEG_INFINITY);
                let cand = until - support;
                let new_cur = if cand > cur { cand } else { cur };
                let mut todo: Vec<(u32, f64)> = Vec::new();
                for id in spec.obs.late.remove(ent).unwrap_or_default() {
                    if let Ok(i) = id_index.binary_search_by_key(&id, |&(x, _, _)| x) {
                        let (_, e, o) = id_index[i];
                        if e == eid {
                            todo.push((o, 1.0));
                        }
                    }
                }
                if new_cur > cur {
                    if cur.is_finite() {
                        for seg in log.segments_between(cur, new_cur) {
                            for &o in &seg.ords {
                                let t = log.events[o as usize].time;
                                if cur < t && t <= new_cur {
                                    todo.push((o, 1.0));
                                }
                            }
                        }
                    } else {
                        for seg in log.segments() {
                            for &o in &seg.ords {
                                let t = log.events[o as usize].time;
                                if t <= new_cur {
                                    todo.push((o, 1.0));
                                }
                            }
                        }
                    }
                }
                if partial {
                    for seg in log.segments_between(new_cur, until) {
                        for &o in &seg.ords {
                            let t = log.events[o as usize].time;
                            if new_cur < t && t <= until {
                                todo.push((o, spec.kernel.mass_fraction(until - t)));
                            }
                        }
                    }
                }
                spec.obs.cursors.insert(ent.clone(), new_cur);
                for (ord, frac) in todo {
                    let ev = &log.events[ord as usize];
                    if !spec.type_allowed_id(Some(ev.ty)) {
                        continue;
                    }
                    let r = EventRef {
                        id: ev.id,
                        entity: ent,
                        time: ev.time,
                        event_type: vocab.types.name(ev.ty),
                        features: Features::Interned(log.feats(ev), &vocab.features),
                        weight: ev.weight,
                        attrs: ev.attrs.as_deref().unwrap_or(&EMPTY_ATTRS),
                    };
                    let m = spec.rule_multiplier_ref(Some(ev.ty), &r)?;
                    if m <= 0.0 {
                        continue;
                    }
                    let full = frac >= 1.0;
                    if full {
                        n_settled += 1;
                        spec.obs.bump("events_settled", 1.0);
                    }
                    if spec.obs.credited.contains_key(&ev.id) {
                        if full {
                            n_cred += 1;
                            spec.obs.bump("skipped_credited", 1.0);
                        }
                        continue;
                    }
                    if !spec.repeatable {
                        if let Some(first) = spec.obs.occurrences.get(ent).and_then(|o| o.first()) {
                            if ev.time >= *first {
                                if full {
                                    n_obs += 1;
                                    spec.obs.bump("skipped_terminal", 1.0);
                                }
                                continue;
                            }
                        }
                    }
                    if occurred_in_window(spec, ent, ev.time) {
                        if full {
                            n_obs += 1;
                            spec.obs.bump("skipped_observed", 1.0);
                        }
                        continue;
                    }
                    let target = spec.w_neg * m * frac;
                    let w = target - spec.obs.rejected.get(&ev.id).copied().unwrap_or(0.0);
                    if w <= 1e-12 {
                        continue;
                    }
                    for &f in log.feats(ev) {
                        spec.links.update(f, 0.0, w, Some(eid));
                    }
                    let mut sorted = log.feats(ev).to_vec();
                    sorted.sort_unstable();
                    spec.links.logistic_step(&sorted, 0.0, w, Some(eid));
                    spec.obs.rejected.insert(ev.id, target);
                    w_total += w;
                    spec.obs.bump("rejected_weight", w);
                    if full {
                        n_rej += 1;
                        *per.get_mut(name.as_str()).expect("inserted") += 1;
                        spec.obs.bump("events_rejected", 1.0);
                    }
                }
            }
        }
        Ok(SettleReport {
            until,
            events_settled: n_settled,
            events_rejected: n_rej,
            rejected_weight: w_total,
            skipped_credited: n_cred,
            skipped_observed: n_obs,
            per_effect: per.into_iter().collect(),
        })
    }

    /// Running counters of observational learning for `effect`, plus
    /// `mean_leak_responsibility` (NaN before the first occurrence).
    pub fn observation_stats(&self, effect: &str) -> Result<IndexMap<String, f64>> {
        let mut st = self.effect(effect)?.obs.stats.clone();
        let n = st.get("occurrences").copied().unwrap_or(0.0);
        let lr = st.get("leak_responsibility").copied().unwrap_or(0.0);
        st.insert(
            "mean_leak_responsibility".into(),
            if n != 0.0 { lr / n } else { f64::NAN },
        );
        Ok(st)
    }

    /// Features ranked by mean link strength, with credible intervals and
    /// lift. **Associations under the model, not causal effects.**
    pub fn link_report(&self, effect: &str, opts: &LinkReportOptions) -> Result<LinkReport> {
        let spec = self.effect(effect)?;
        let links = &spec.links;
        let ent = self.entity_ref(opts.entity.as_deref());
        let mut feats: Vec<String> = match &opts.features {
            Some(f) => f.clone(),
            None => {
                let mut v: Vec<String> = links
                    .known_features()
                    .map(|f| self.vocab.features.name(f).to_string())
                    .collect();
                if let Some(e) = opts.entity.as_deref().and_then(|e| self.vocab.entities.get(e)) {
                    v.extend(
                        links
                            .entity_counts(e)
                            .into_iter()
                            .map(|(f, _, _)| self.vocab.features.name(f).to_string()),
                    );
                }
                v
            }
        };
        feats.sort();
        feats.dedup();
        let mut rows: Vec<(String, f64, f64, f64, f64, f64, f64)> = Vec::new();
        for f in feats {
            let b = links.get(self.vocab.features.get(&f), ent);
            let n = b.observations();
            if n < opts.min_support {
                continue;
            }
            let (lo, hi) = beta_interval(b.alpha(), b.beta(), opts.level)?;
            rows.push((f, b.strength(), b.alpha(), b.beta(), n, lo, hi));
        }
        let base = match opts.baseline {
            Baseline::Mean => {
                if rows.is_empty() {
                    f64::NAN
                } else {
                    py_sum(rows.iter().map(|r| r.1)) / rows.len() as f64
                }
            }
            Baseline::Prior => {
                let (a, b) = links.default_prior();
                a / (a + b)
            }
            Baseline::Value(v) => v,
        };
        rows.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        let lift_ok = base != 0.0 && base > 0.0;
        Ok(LinkReport {
            effect: effect.to_string(),
            entity: opts.entity.clone(),
            baseline: base,
            level: opts.level,
            rows: rows
                .into_iter()
                .map(|(feature, strength, alpha, beta, n, lo, hi)| LinkRow {
                    lift: if lift_ok { strength / base } else { f64::NAN },
                    log_odds: links.weight(self.vocab.features.get(&feature)),
                    feature,
                    strength,
                    alpha,
                    beta,
                    n,
                    lo,
                    hi,
                })
                .collect(),
        })
    }
}

fn occurred_in_window(spec: &EffectSpec, entity: &str, t: f64) -> bool {
    let Some(occ) = spec.obs.occurrences.get(entity) else {
        return false;
    };
    if occ.is_empty() {
        return false;
    }
    let lo = t + spec.kernel.min_lag().max(0.0);
    let hi = t + spec.kernel.support();
    let i = occ.partition_point(|&x| x < lo);
    i < occ.len() && occ[i] <= hi
}
