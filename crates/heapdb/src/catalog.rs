//! Catalog: what is in a database, for people (and autocomplete) deciding
//! what to ask. Effects with their kernels and priors, entities by recent
//! activity, event types with their common features, and recent effect
//! occurrences (from `observe` and `feedback`).
//!
//! Every call is bounded and paged: scans stop at a cap and say so
//! (`truncated`), and each call returns at most `limit` rows plus what to
//! pass back for the next page, so asking about a huge database costs at
//! most a fixed amount of work.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

use rustc_hash::FxHashMap;

use crate::error::Result;
use crate::kernels::Kernel;
use crate::links::{Combine, Credit, FeatureId};
use crate::logistic::LogisticParams;
use crate::segments::{EntityLog, TypeId};
use crate::store::HeapDB;

/// Entities examined at most by one catalog call.
pub const MAX_SCAN_ENTITIES: usize = 100_000;
/// Events examined at most by [`HeapDB::event_type_summaries`].
pub const MAX_SCAN_EVENTS: usize = 100_000;
/// Feedback records examined at most (per effect) by
/// [`HeapDB::recent_occurrences`].
pub const MAX_SCAN_RECORDS: usize = 100_000;

/// A registered feature prior.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorSummary {
    /// Feature.
    pub feature: String,
    /// Beta alpha.
    pub alpha: f64,
    /// Beta beta.
    pub beta: f64,
}

impl PriorSummary {
    /// Prior mean `alpha / (alpha + beta)`.
    pub fn strength(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }
}

/// One registered effect.
#[derive(Clone, Debug)]
pub struct EffectSummary {
    /// Name.
    pub name: String,
    /// Lag kernel.
    pub kernel: Kernel,
    /// Credit mode for confirmations.
    pub credit: Credit,
    /// How features combine into the link.
    pub combine: Combine,
    /// Hyperparameters of the `logistic` combine mode (`None`: `noisy_or`).
    pub logistic: Option<LogisticParams>,
    /// Learned logistic intercept, the base log-odds (`None`: `noisy_or`).
    pub base_log_odds: Option<f64>,
    /// Allowed cause types (`None`: any).
    pub cause_types: Option<Vec<String>>,
    /// Prior of features without one.
    pub default_prior: (f64, f64),
    /// Cap on pseudo-observations borrowed from the global layer.
    pub global_pseudocount: f64,
    /// Entities borrow from the global layer.
    pub hierarchical: bool,
    /// Background score for `observe`.
    pub leak: f64,
    /// Negative evidence per feature added by `settle`.
    pub w_neg: f64,
    /// False for one-shot effects.
    pub repeatable: bool,
    /// Number of registered priors.
    pub num_priors: usize,
    /// The strongest priors (at most `top_priors`), strongest first.
    pub priors: Vec<PriorSummary>,
    /// Rule names, in application order.
    pub rules: Vec<String>,
    /// Occurrences recorded by `observe`.
    pub occurrences: usize,
    /// Entities with at least one observed occurrence.
    pub occurrence_entities: usize,
    /// Feedback records (one per confirmed or rejected event).
    pub feedback_records: usize,
    /// Of which confirmations.
    pub feedback_confirmed: usize,
    /// Features with learned (global) evidence.
    pub learned_features: usize,
}

/// A page of [`HeapDB::effect_summaries`].
#[derive(Clone, Debug)]
pub struct EffectPage {
    /// Effects, in registration order.
    pub effects: Vec<EffectSummary>,
    /// Pass back as `after` for the next page (`None`: last page).
    pub next: Option<String>,
    /// Registered effects in total.
    pub total: usize,
}

/// One entity with activity.
#[derive(Clone, Debug, PartialEq)]
pub struct EntitySummary {
    /// Name.
    pub name: String,
    /// Stored events.
    pub events: usize,
    /// Earliest event time.
    pub first_time: f64,
    /// Latest event time.
    pub last_time: f64,
}

/// A page of [`HeapDB::entity_summaries`].
#[derive(Clone, Debug, PartialEq)]
pub struct EntityPage {
    /// Entities, most recent activity first (ties by name).
    pub entities: Vec<EntitySummary>,
    /// Pass back as `after` for the next page (`None`: last page).
    pub next: Option<(f64, String)>,
    /// Entities examined.
    pub scanned: usize,
    /// Entities matching the prefix among those examined.
    pub matched: usize,
    /// Entities with a log in total.
    pub total: usize,
    /// True if the scan stopped at the cap (the most recently created
    /// entities are examined first).
    pub truncated: bool,
}

/// One event type.
#[derive(Clone, Debug, PartialEq)]
pub struct EventTypeSummary {
    /// Type name.
    pub name: String,
    /// Events of this type among those examined.
    pub events: usize,
    /// Latest time among those examined (`None`: no events examined).
    pub last_time: Option<f64>,
    /// Registered with `register_event_type`.
    pub registered: bool,
    /// Most common features (most frequent first, ties by name).
    pub features: Vec<(String, usize)>,
}

/// Result of [`HeapDB::event_type_summaries`].
#[derive(Clone, Debug, PartialEq)]
pub struct EventTypeCatalog {
    /// Types, most events first (ties by name); registered types without
    /// examined events last.
    pub types: Vec<EventTypeSummary>,
    /// Events examined.
    pub events_scanned: usize,
    /// Events stored.
    pub total_events: usize,
    /// True if not every event was examined (counts are then those of
    /// the most recent events).
    pub truncated: bool,
}

/// One effect occurrence: a time an effect was recorded for an entity.
#[derive(Clone, Debug, PartialEq)]
pub struct Occurrence {
    /// Effect.
    pub effect: String,
    /// Entity.
    pub entity: String,
    /// Effect time.
    pub at: f64,
    /// Recorded by `observe`.
    pub observed: bool,
    /// Recorded by `feedback`.
    pub feedback: bool,
    /// The confirmed cause, if feedback confirmed one.
    pub confirmed_event_id: Option<u64>,
    /// Candidates rejected by feedback.
    pub rejected: usize,
}

/// Position of an occurrence in the order of
/// [`HeapDB::recent_occurrences`]: `(at, entity, effect)`.
pub type OccurrenceKey = (f64, String, String);

/// Result of [`HeapDB::recent_occurrences`].
#[derive(Clone, Debug, PartialEq)]
pub struct OccurrencePage {
    /// Most recent first (ties by entity, then effect).
    pub occurrences: Vec<Occurrence>,
    /// Pass back as `after` for the next page (`None`: last page).
    pub next: Option<OccurrenceKey>,
    /// True if a scan stopped at a cap.
    pub truncated: bool,
}

/// `(last_time desc, name asc)`: the order of [`HeapDB::entity_summaries`].
fn recency_cmp(a: (f64, &str), b: (f64, &str)) -> Ordering {
    b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1))
}

fn log_times(log: &EntityLog) -> Option<(f64, f64)> {
    let first = log.segments.values().next()?.min_time;
    let last = log.segments.values().next_back()?.max_time;
    Some((first, last))
}

/// `(at, entity)` ranked like [`HeapDB::recent_occurrences`] within one
/// effect: greater is newer (ties: the smaller entity name is greater).
#[derive(PartialEq)]
struct Rank<'a>(f64, &'a str);
impl Eq for Rank<'_> {}
impl PartialOrd for Rank<'_> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Rank<'_> {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0).then_with(|| o.1.cmp(self.1))
    }
}

/// Min-heap of the best `cap` ranks seen (its top is the worst kept).
type RankHeap<'a> = BinaryHeap<Reverse<Rank<'a>>>;

/// Offer an entity's occurrence times (sorted ascending, duplicates
/// allowed) that pass `keep` to a heap of the best `cap` overall.
fn newest_into<'a>(
    heap: &mut RankHeap<'a>,
    cap: usize,
    ent: &'a str,
    times: &[f64],
    keep: impl Fn(f64) -> bool,
) {
    let mut prev = None;
    for &t in times.iter().rev() {
        if prev == Some(t.to_bits()) {
            continue; // the same occurrence time recorded twice
        }
        prev = Some(t.to_bits());
        if !keep(t) {
            continue; // newer than the cursor: comes first, skip on
        }
        let r = Rank(t, ent);
        if heap.len() < cap {
            heap.push(Reverse(r));
        } else if heap.peek().is_some_and(|w| r > w.0) {
            heap.pop();
            heap.push(Reverse(r));
        } else {
            break; // sorted: older times cannot enter
        }
    }
}

/// `(at desc, entity asc, effect asc)`: the order of
/// [`HeapDB::recent_occurrences`].
fn occurrence_cmp(a: (f64, &str, &str), b: (f64, &str, &str)) -> Ordering {
    recency_cmp((a.0, a.1), (b.0, b.1)).then_with(|| a.2.cmp(b.2))
}

impl HeapDB {
    /// Registered effects in registration order, at most `limit` (>= 1)
    /// continuing after the effect `after` (the previous page's `next`),
    /// with at most `top_priors` priors each. Fails for an unknown `after`.
    pub fn effect_summaries(
        &self,
        after: Option<&str>,
        limit: usize,
        top_priors: usize,
    ) -> Result<EffectPage> {
        let limit = limit.max(1);
        let start = match after {
            Some(name) => {
                self.effect(name)?;
                self.effects.get_index_of(name).map_or(0, |i| i + 1)
            }
            None => 0,
        };
        let total = self.effects.len();
        let end = start.saturating_add(limit).min(total);
        let next = (end < total)
            .then(|| self.effects.get_index(end - 1).map(|(k, _)| k.clone()))
            .flatten();
        let effects = self
            .effects
            .values()
            .skip(start)
            .take(end.saturating_sub(start))
            .map(|spec| {
                let links = spec.links();
                let mut priors: Vec<PriorSummary> = links
                    .priors()
                    .map(|(f, (alpha, beta))| PriorSummary {
                        feature: self.vocab.features.name(f).to_string(),
                        alpha,
                        beta,
                    })
                    .collect();
                let num_priors = priors.len();
                priors.sort_by(|a, b| {
                    b.strength()
                        .total_cmp(&a.strength())
                        .then_with(|| a.feature.cmp(&b.feature))
                });
                priors.truncate(top_priors);
                let obs = spec.observations();
                let recs = self.calibration.get(&spec.name).map(Vec::as_slice).unwrap_or(&[]);
                EffectSummary {
                    name: spec.name.clone(),
                    kernel: spec.kernel.clone(),
                    credit: spec.credit,
                    combine: links.combine(),
                    logistic: links.logistic().map(|l| l.params()),
                    base_log_odds: links.base_log_odds(),
                    cause_types: spec.cause_types.as_ref().map(|c| c.iter().cloned().collect()),
                    default_prior: links.default_prior(),
                    global_pseudocount: links.global_pseudocount(),
                    hierarchical: links.hierarchical(),
                    leak: spec.leak,
                    w_neg: spec.w_neg,
                    repeatable: spec.repeatable,
                    num_priors,
                    priors,
                    rules: spec.rules.iter().map(|r| r.name.clone()).collect(),
                    occurrences: obs.occurrences.values().map(Vec::len).sum(),
                    occurrence_entities: obs.occurrences.values().filter(|v| !v.is_empty()).count(),
                    feedback_records: recs.len(),
                    feedback_confirmed: recs.iter().filter(|r| r.confirmed).count(),
                    learned_features: links.num_learned(),
                }
            })
            .collect();
        Ok(EffectPage { effects, next, total })
    }

    /// Entities whose name starts with `prefix`, most recent activity
    /// first, at most `limit` (>= 1), continuing after `after` (the
    /// previous page's `next`). Examines at most `max_scan` entities, the
    /// most recently created first.
    pub fn entity_summaries(
        &self,
        prefix: Option<&str>,
        limit: usize,
        after: Option<(f64, &str)>,
        max_scan: usize,
    ) -> EntityPage {
        let limit = limit.max(1);
        let prefix = prefix.unwrap_or("");
        let total = self.logs.len();
        let mut scanned = 0usize;
        let mut matched = 0usize;
        let mut rows: Vec<(f64, &str, &EntityLog)> = Vec::new();
        for (&eid, log) in self.logs.iter().rev() {
            if scanned >= max_scan {
                break;
            }
            scanned += 1;
            let name = self.vocab.entities.name(eid);
            if !name.starts_with(prefix) {
                continue;
            }
            let Some((_, last)) = log_times(log) else {
                continue;
            };
            matched += 1;
            if let Some(a) = after {
                if recency_cmp((last, name), a) != Ordering::Greater {
                    continue;
                }
            }
            rows.push((last, name, log));
        }
        let cmp =
            |a: &(f64, &str, &EntityLog), b: &(f64, &str, &EntityLog)| recency_cmp((a.0, a.1), (b.0, b.1));
        if rows.len() > limit + 1 {
            rows.select_nth_unstable_by(limit, cmp);
            rows.truncate(limit + 1);
        }
        rows.sort_by(cmp);
        let more = rows.len() > limit;
        rows.truncate(limit);
        let next = if more {
            rows.last().map(|&(t, n, _)| (t, n.to_string()))
        } else {
            None
        };
        let entities = rows
            .into_iter()
            .map(|(last, name, log)| EntitySummary {
                name: name.to_string(),
                events: log.len(),
                first_time: log_times(log).map_or(last, |t| t.0),
                last_time: last,
            })
            .collect();
        EntityPage {
            entities,
            next,
            scanned,
            matched,
            total,
            truncated: scanned < total,
        }
    }

    /// Event types with counts and their `top_features` most common
    /// features. Examines at most `max_scan` events: the most recent ones
    /// of the most recently active entities.
    pub fn event_type_summaries(&self, top_features: usize, max_scan: usize) -> EventTypeCatalog {
        let total_events = self.len();
        // Most recently active entities first (bounded like entity_summaries).
        let mut logs: Vec<(f64, &EntityLog)> = self
            .logs
            .values()
            .rev()
            .take(MAX_SCAN_ENTITIES)
            .filter_map(|log| log_times(log).map(|(_, last)| (last, log)))
            .collect();
        logs.sort_by(|a, b| b.0.total_cmp(&a.0));
        struct Acc {
            events: usize,
            last: f64,
            features: FxHashMap<FeatureId, usize>,
        }
        let mut acc: FxHashMap<TypeId, Acc> = FxHashMap::default();
        let mut scanned = 0usize;
        'outer: for (_, log) in &logs {
            for seg in log.segments.values().rev() {
                for &ord in seg.ords.iter().rev() {
                    if scanned >= max_scan {
                        break 'outer;
                    }
                    scanned += 1;
                    let ev = &log.events[ord as usize];
                    let a = acc.entry(ev.ty).or_insert_with(|| Acc {
                        events: 0,
                        last: f64::NEG_INFINITY,
                        features: FxHashMap::default(),
                    });
                    a.events += 1;
                    a.last = a.last.max(ev.time);
                    for &f in log.feats(ev) {
                        *a.features.entry(f).or_insert(0) += 1;
                    }
                }
            }
        }
        let mut types: Vec<EventTypeSummary> = acc
            .into_iter()
            .map(|(ty, a)| {
                let name = self.vocab.types.name(ty).to_string();
                let mut features: Vec<(String, usize)> = a
                    .features
                    .into_iter()
                    .map(|(f, n)| (self.vocab.features.name(f).to_string(), n))
                    .collect();
                features.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
                features.truncate(top_features);
                EventTypeSummary {
                    registered: self.event_types.contains_key(&name),
                    name,
                    events: a.events,
                    last_time: Some(a.last),
                    features,
                }
            })
            .collect();
        types.sort_by(|a, b| b.events.cmp(&a.events).then_with(|| a.name.cmp(&b.name)));
        let mut idle: Vec<EventTypeSummary> = self
            .event_types
            .keys()
            .filter(|t| !types.iter().any(|s| &s.name == *t))
            .map(|t| EventTypeSummary {
                name: t.clone(),
                events: 0,
                last_time: None,
                registered: true,
                features: Vec::new(),
            })
            .collect();
        types.append(&mut idle);
        EventTypeCatalog {
            types,
            events_scanned: scanned,
            total_events,
            truncated: scanned < total_events,
        }
    }

    /// The most recent `limit` (>= 1) occurrences of `effect` (all
    /// effects when `None`), optionally of one entity, continuing after
    /// `after` (the previous page's `next`): times recorded by `observe`
    /// merged with the effect times of `feedback`, newest first (ties by
    /// entity, then effect). Fails for an unknown effect. Examines at most
    /// `max_scan` entities and feedback records per effect.
    pub fn recent_occurrences(
        &self,
        effect: Option<&str>,
        entity: Option<&str>,
        limit: usize,
        after: Option<(f64, &str, &str)>,
        max_scan: usize,
    ) -> Result<OccurrencePage> {
        let limit = limit.max(1);
        // One more than a page, to know whether there is a next one.
        let cap = limit + 1;
        let specs: Vec<&str> = match effect {
            Some(e) => vec![self.effect(e)?.name()],
            None => self.effects.keys().map(String::as_str).collect(),
        };
        let mut truncated = false;
        // (effect, entity, at bits) -> occurrence
        let mut merged: HashMap<(&str, &str, u64), Occurrence> = HashMap::new();
        let blank = |eff: &str, ent: &str, at: f64| Occurrence {
            effect: eff.to_string(),
            entity: ent.to_string(),
            at,
            observed: false,
            feedback: false,
            confirmed_event_id: None,
            rejected: 0,
        };
        for eff in specs {
            let spec = &self.effects[eff];
            // Strictly after the cursor in the page order.
            let keep = |t: f64, ent: &str| {
                after.is_none_or(|a| occurrence_cmp((t, ent, eff), a) == Ordering::Greater)
            };
            // Observed: a bounded min-heap of the newest `cap` times.
            let mut heap: RankHeap<'_> = BinaryHeap::new();
            let occ = &spec.obs.occurrences;
            match entity {
                Some(e) => {
                    if let Some((k, v)) = occ.get_key_value(e) {
                        newest_into(&mut heap, cap, k.as_str(), v, |t| keep(t, k));
                    }
                }
                None => {
                    for (i, (k, v)) in occ.iter().enumerate() {
                        if i >= max_scan {
                            truncated = true;
                            break;
                        }
                        newest_into(&mut heap, cap, k.as_str(), v, |t| keep(t, k));
                    }
                }
            }
            for Reverse(Rank(at, ent)) in heap {
                merged
                    .entry((eff, ent, at.to_bits()))
                    .or_insert_with(|| blank(eff, ent, at))
                    .observed = true;
            }
            // Feedback: newest records last; group by (entity, at).
            let recs = self.calibration.get(eff).map(Vec::as_slice).unwrap_or(&[]);
            if recs.len() > max_scan {
                truncated = true;
            }
            let mut fb: Vec<(f64, &str)> = Vec::new();
            let mut groups: HashMap<(&str, u64), (Option<u64>, usize)> = HashMap::new();
            for r in recs.iter().rev().take(max_scan) {
                if entity.is_some_and(|e| e != r.entity) || !keep(r.at, &r.entity) {
                    continue;
                }
                let g = groups
                    .entry((r.entity.as_str(), r.at.to_bits()))
                    .or_insert_with(|| {
                        fb.push((r.at, r.entity.as_str()));
                        (None, 0)
                    });
                if r.confirmed {
                    g.0 = Some(r.event_id);
                } else {
                    g.1 += 1;
                }
            }
            fb.sort_by(|a, b| recency_cmp((a.0, a.1), (b.0, b.1)));
            fb.truncate(cap);
            for (at, ent) in fb {
                let (confirmed, rejected) = groups[&(ent, at.to_bits())];
                let o = merged
                    .entry((eff, ent, at.to_bits()))
                    .or_insert_with(|| blank(eff, ent, at));
                o.feedback = true;
                o.confirmed_event_id = confirmed;
                o.rejected = rejected;
            }
        }
        let mut occurrences: Vec<Occurrence> = merged.into_values().collect();
        occurrences
            .sort_by(|a, b| occurrence_cmp((a.at, &a.entity, &a.effect), (b.at, &b.entity, &b.effect)));
        let more = occurrences.len() > limit;
        occurrences.truncate(limit);
        let next = if more {
            occurrences
                .last()
                .map(|o| (o.at, o.entity.clone(), o.effect.clone()))
        } else {
            None
        };
        Ok(OccurrencePage {
            occurrences,
            next,
            truncated,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{EffectOptions, NewEvent};

    fn db() -> HeapDB {
        let mut db = HeapDB::new(900.0, 64).unwrap();
        db.register_effect(
            "failing_main_check",
            EffectOptions::new(Kernel::lognormal(4.0 * 3600.0, 0.8).unwrap())
                .prior("migration", 3.0, 1.0)
                .prior("tests:none", 2.0, 2.0)
                .cause_types(["merge"]),
        )
        .unwrap();
        db.register_effect(
            "incident",
            EffectOptions::new(Kernel::exponential(86400.0).unwrap()),
        )
        .unwrap();
        for (i, ent) in ["acme/web", "acme/api", "checkout", "acme/worker"]
            .iter()
            .enumerate()
        {
            for j in 0..=i {
                let t = 1000.0 * (i as f64) + 10.0 * j as f64;
                db.insert(
                    NewEvent::new(*ent, t, if j % 2 == 0 { "merge" } else { "deploy" })
                        .features(["size:large", if j == 0 { "migration" } else { "config" }]),
                )
                .unwrap();
            }
        }
        db
    }

    #[test]
    fn empty_database() {
        let db = HeapDB::new(900.0, 64).unwrap();
        let e = db.effect_summaries(None, 10, 5).unwrap();
        assert!(e.effects.is_empty() && e.next.is_none() && e.total == 0);
        let p = db.entity_summaries(None, 10, None, MAX_SCAN_ENTITIES);
        assert!(p.entities.is_empty() && p.next.is_none() && !p.truncated);
        assert_eq!((p.scanned, p.matched, p.total), (0, 0, 0));
        let c = db.event_type_summaries(5, MAX_SCAN_EVENTS);
        assert!(c.types.is_empty() && !c.truncated);
        assert_eq!((c.events_scanned, c.total_events), (0, 0));
        let o = db
            .recent_occurrences(None, None, 10, None, MAX_SCAN_RECORDS)
            .unwrap();
        assert!(o.occurrences.is_empty() && o.next.is_none() && !o.truncated);
        assert!(db.recent_occurrences(Some("x"), None, 10, None, 10).is_err());
    }

    #[test]
    fn effects_are_summarised() {
        let db = db();
        let page = db.effect_summaries(None, 10, 1).unwrap();
        assert_eq!((page.total, page.next.as_deref()), (2, None));
        let s = page.effects;
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].name, "failing_main_check");
        assert_eq!(s[0].num_priors, 2);
        assert_eq!(s[0].priors.len(), 1);
        assert_eq!(s[0].priors[0].feature, "migration");
        assert_eq!(s[0].cause_types.as_deref(), Some(&["merge".to_string()][..]));
        assert_eq!(s[1].cause_types, None);
        assert_eq!(s[1].kernel.name(), "exp");
        assert_eq!((s[1].occurrences, s[1].feedback_records), (0, 0));
        assert_eq!(s[0].combine, Combine::NoisyOr);
        assert_eq!((s[0].logistic, s[0].base_log_odds), (None, None));

        // Pages in registration order.
        let p1 = db.effect_summaries(None, 1, 1).unwrap();
        assert_eq!(p1.effects[0].name, "failing_main_check");
        assert_eq!(p1.next.as_deref(), Some("failing_main_check"));
        let p2 = db.effect_summaries(p1.next.as_deref(), 1, 1).unwrap();
        assert_eq!(p2.effects[0].name, "incident");
        assert_eq!(p2.next, None);
        assert!(db.effect_summaries(Some("nope"), 1, 1).is_err());
    }

    #[test]
    fn logistic_effects_report_their_mode() {
        let mut db = db();
        db.register_effect(
            "churn",
            EffectOptions::new(Kernel::exponential(86400.0).unwrap())
                .combine(Combine::Logistic)
                .learning_rate(0.5),
        )
        .unwrap();
        let s = db.effect_summaries(Some("incident"), 5, 5).unwrap().effects;
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].combine, Combine::Logistic);
        assert_eq!(s[0].logistic.unwrap().learning_rate, 0.5);
        assert!(s[0].base_log_odds.unwrap().is_finite());
    }

    #[test]
    fn entities_by_recency_with_prefix_and_pages() {
        let db = db();
        let all = db.entity_summaries(None, 50, None, MAX_SCAN_ENTITIES);
        let names: Vec<&str> = all.entities.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["acme/worker", "checkout", "acme/api", "acme/web"]);
        assert_eq!(all.entities[0].events, 4);
        assert_eq!(all.entities[0].first_time, 3000.0);
        assert_eq!(all.entities[0].last_time, 3030.0);
        assert!(!all.truncated && all.next.is_none());

        let p1 = db.entity_summaries(Some("acme/"), 2, None, MAX_SCAN_ENTITIES);
        assert_eq!(p1.matched, 3);
        assert_eq!(p1.entities.len(), 2);
        let (t, n) = p1.next.clone().unwrap();
        let p2 = db.entity_summaries(Some("acme/"), 2, Some((t, &n)), MAX_SCAN_ENTITIES);
        assert_eq!(p2.entities.len(), 1);
        assert_eq!(p2.entities[0].name, "acme/web");
        assert!(p2.next.is_none());

        let capped = db.entity_summaries(None, 50, None, 2);
        assert!(capped.truncated);
        assert_eq!(capped.scanned, 2);
        assert_eq!(capped.entities.len(), 2);
        assert_eq!(db.entity_summaries(Some("nope"), 5, None, 10).matched, 0);
    }

    #[test]
    fn entity_pages_with_tied_times_cover_everything_once() {
        let mut db = HeapDB::new(900.0, 64).unwrap();
        for ent in ["e3", "e1", "e5", "e2", "e4"] {
            db.insert(NewEvent::new(ent, 10.0, "x")).unwrap();
        }
        db.insert(NewEvent::new("e0", 5.0, "x")).unwrap();
        let mut seen = Vec::new();
        let mut after: Option<(f64, String)> = None;
        loop {
            let p = db.entity_summaries(
                None,
                2,
                after.as_ref().map(|(t, n)| (*t, n.as_str())),
                MAX_SCAN_ENTITIES,
            );
            assert!(p.entities.len() <= 2);
            seen.extend(p.entities.into_iter().map(|e| e.name));
            match p.next {
                Some(n) => after = Some(n),
                None => break,
            }
        }
        assert_eq!(seen, ["e1", "e2", "e3", "e4", "e5", "e0"]);
    }

    #[test]
    fn event_types_with_top_features_and_cap() {
        let mut db = db();
        db.register_event_type("page_view", None);
        let c = db.event_type_summaries(20, MAX_SCAN_EVENTS);
        assert_eq!(c.total_events, 10);
        assert_eq!(c.events_scanned, 10);
        assert!(!c.truncated);
        let merge = c.types.iter().find(|t| t.name == "merge").unwrap();
        assert_eq!(merge.events, 6);
        assert_eq!(merge.features[0], ("size:large".to_string(), 6));
        let last = c.types.last().unwrap();
        assert_eq!(
            (last.name.as_str(), last.events, last.registered),
            ("page_view", 0, true)
        );
        let capped = db.event_type_summaries(1, 3);
        assert!(capped.truncated);
        assert_eq!(capped.events_scanned, 3);
        assert!(capped.types.iter().all(|t| t.features.len() <= 1));
    }

    #[test]
    fn occurrences_merge_observe_and_feedback() {
        let mut db = db();
        db.observe("acme/web", "failing_main_check", 500.0, None).unwrap();
        db.observe("acme/api", "failing_main_check", 1500.0, None)
            .unwrap();
        db.observe("acme/api", "failing_main_check", 1600.0, None)
            .unwrap();
        let ids = db.event_ids(Some("acme/api"));
        db.feedback(
            "acme/api",
            "failing_main_check",
            1600.0,
            Some(ids[0]),
            &ids[1..],
            None,
        )
        .unwrap();
        let p = db
            .recent_occurrences(Some("failing_main_check"), None, 20, None, MAX_SCAN_RECORDS)
            .unwrap();
        assert_eq!(p.next, None);
        let got: Vec<(&str, f64, bool, bool)> = p
            .occurrences
            .iter()
            .map(|o| (o.entity.as_str(), o.at, o.observed, o.feedback))
            .collect();
        assert_eq!(
            got,
            [
                ("acme/api", 1600.0, true, true),
                ("acme/api", 1500.0, true, false),
                ("acme/web", 500.0, true, false)
            ]
        );
        assert_eq!(p.occurrences[0].confirmed_event_id, Some(ids[0]));
        assert_eq!(p.occurrences[0].rejected, 1);
        let one = db
            .recent_occurrences(None, Some("acme/web"), 1, None, 10)
            .unwrap();
        assert_eq!(one.occurrences.len(), 1);
        assert_eq!(one.occurrences[0].at, 500.0);
        let lim = db.recent_occurrences(None, None, 2, None, 10).unwrap();
        assert_eq!(lim.occurrences.len(), 2);
        assert!(db.recent_occurrences(Some("nope"), None, 5, None, 10).is_err());
        let capped = db.recent_occurrences(None, None, 20, None, 1).unwrap();
        assert!(capped.truncated);
    }

    /// Paging one row at a time visits every occurrence once, in order,
    /// including ties in time across entities and effects and a time
    /// observed twice.
    #[test]
    fn occurrence_pages_cover_everything_once() {
        let mut db = db();
        for (ent, eff, at) in [
            ("b", "incident", 100.0),
            ("a", "incident", 100.0),
            ("a", "failing_main_check", 100.0),
            ("c", "incident", 100.0),
            ("a", "incident", 50.0),
            ("a", "incident", 50.0),
            ("b", "failing_main_check", 200.0),
        ] {
            db.observe(ent, eff, at, None).unwrap();
        }
        let want = [
            (200.0, "b", "failing_main_check"),
            (100.0, "a", "failing_main_check"),
            (100.0, "a", "incident"),
            (100.0, "b", "incident"),
            (100.0, "c", "incident"),
            (50.0, "a", "incident"),
        ];
        let all = db
            .recent_occurrences(None, None, 50, None, MAX_SCAN_RECORDS)
            .unwrap();
        let got: Vec<(f64, &str, &str)> = all
            .occurrences
            .iter()
            .map(|o| (o.at, o.entity.as_str(), o.effect.as_str()))
            .collect();
        assert_eq!(got, want);
        for size in [1, 2, 4] {
            let mut seen = Vec::new();
            let mut after: Option<OccurrenceKey> = None;
            loop {
                let p = db
                    .recent_occurrences(
                        None,
                        None,
                        size,
                        after.as_ref().map(|(t, n, e)| (*t, n.as_str(), e.as_str())),
                        MAX_SCAN_RECORDS,
                    )
                    .unwrap();
                assert!(p.occurrences.len() <= size);
                seen.extend(p.occurrences.into_iter().map(|o| (o.at, o.entity, o.effect)));
                match p.next {
                    Some(n) => after = Some(n),
                    None => break,
                }
            }
            let seen: Vec<(f64, &str, &str)> = seen
                .iter()
                .map(|(t, n, e)| (*t, n.as_str(), e.as_str()))
                .collect();
            assert_eq!(seen, want, "page size {size}");
        }
        // Within one effect, ties by entity name.
        let p = db
            .recent_occurrences(Some("incident"), None, 2, None, MAX_SCAN_RECORDS)
            .unwrap();
        let ents: Vec<&str> = p.occurrences.iter().map(|o| o.entity.as_str()).collect();
        assert_eq!(ents, ["a", "b"]);
        assert_eq!(p.next, Some((100.0, "b".to_string(), "incident".to_string())));
    }
}
