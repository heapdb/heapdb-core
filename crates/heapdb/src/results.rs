//! Query results and reports, with the reference implementation's text
//! formatting (`Display`).

use std::collections::HashSet;
use std::fmt;

use crate::event::Event;
use crate::links::BetaCounts;
use crate::pyfmt::{fmt_f, fmt_g, fmt_g6, fmt_pct, format_time};

/// Counts are floats; print integers without noise (`3`, else `2.75`).
pub fn fmt_count(x: f64) -> String {
    if (x - x.round_ties_even()).abs() < 1e-9 {
        fmt_g6(x)
    } else {
        fmt_f(x, 2)
    }
}

/// How one feature of a candidate contributes to its score.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureContribution {
    /// Feature token.
    pub feature: String,
    /// Posterior mean.
    pub strength: f64,
    /// Posterior alpha.
    pub alpha: f64,
    /// Posterior beta.
    pub beta: f64,
    /// Logistic weight `w_f`: the feature adds `w_f` to the event's
    /// log-odds, i.e. multiplies its odds by `exp(w_f)`. `None` under
    /// noisy-OR.
    pub log_odds: Option<f64>,
}

impl FeatureContribution {
    /// `exp(log_odds)`: the factor this feature multiplies the odds by
    /// (`None` under noisy-OR).
    pub fn odds_multiplier(&self) -> Option<f64> {
        self.log_odds.map(f64::exp)
    }
}

impl fmt::Display for FeatureContribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(m) = self.odds_multiplier() {
            return write!(
                f,
                "{} x{} (a={}, b={})",
                self.feature,
                fmt_g(m, 3),
                fmt_count(self.alpha),
                fmt_count(self.beta)
            );
        }
        write!(
            f,
            "{}={} (a={}, b={})",
            self.feature,
            fmt_f(self.strength, 2),
            fmt_count(self.alpha),
            fmt_count(self.beta)
        )
    }
}

/// A rule that matched, with its multiplier.
#[derive(Clone, Debug, PartialEq)]
pub struct AppliedRule {
    /// Rule name.
    pub name: String,
    /// Multiplier in `[0, 1]`.
    pub multiplier: f64,
}

impl fmt::Display for AppliedRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} x{}", self.name, fmt_g6(self.multiplier))
    }
}

/// A set of features reported as one cause (e.g. `beans+onion`).
///
/// `strength` is what scoring uses for the members alone (their noisy-OR,
/// or under logistic `sigmoid(b + sum of their weights)`). `joint` is the
/// joint Beta link learned for exactly this set, if it has been judged.
/// `inseparable` is true while the members have always co-occurred.
#[derive(Clone, Debug, PartialEq)]
pub struct JointCause {
    /// Member features, sorted.
    pub features: Vec<String>,
    /// Link of the members alone (noisy-OR or logistic).
    pub strength: f64,
    /// The joint link, if judged.
    pub joint: Option<BetaCounts>,
    /// True while the members have always co-occurred.
    pub inseparable: bool,
}

impl JointCause {
    /// `a+b+c`.
    pub fn name(&self) -> String {
        self.features.join("+")
    }
}

impl fmt::Display for JointCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.name(), fmt_f(self.strength, 2))?;
        if let Some(j) = &self.joint {
            write!(
                f,
                " (joint link {}, a={}, b={})",
                fmt_f(j.strength(), 2),
                fmt_count(j.alpha()),
                fmt_count(j.beta())
            )?;
        }
        if self.inseparable {
            f.write_str(" inseparable")?;
        }
        Ok(())
    }
}

/// Why an event got its score: `kernel x weight x link x rules`.
#[derive(Clone, Debug, PartialEq)]
pub struct Explanation {
    /// Seconds between cause and effect.
    pub dt: f64,
    /// Timing plausibility in `[0, 1]`.
    pub kernel_value: f64,
    /// Dose.
    pub weight: f64,
    /// The link: noisy-OR of the feature strengths, or under logistic
    /// `sigmoid(base_log_odds + sum of the features' log_odds)`.
    pub link_strength: f64,
    /// Per-feature contributions, strongest first.
    pub features: Vec<FeatureContribution>,
    /// Logistic intercept `b` (the effect's base log-odds); `None` under
    /// noisy-OR.
    pub base_log_odds: Option<f64>,
    /// Confounding warnings.
    pub warnings: Vec<String>,
    /// Applied rules, in rule order.
    pub rules: Vec<AppliedRule>,
    /// Joint causes, strongest first.
    pub joint_causes: Vec<JointCause>,
    /// Entity of the event.
    pub entity: String,
    /// True for `feature_risk` explanations (no lag yet, kernel 1).
    pub prospective: bool,
}

impl Explanation {
    /// Product of the applied rules' multipliers.
    pub fn rule_multiplier(&self) -> f64 {
        let mut m = 1.0;
        for r in &self.rules {
            m *= r.multiplier;
        }
        m
    }

    /// `kernel_value * weight * link_strength * rule_multiplier`.
    pub fn score(&self) -> f64 {
        self.kernel_value * self.weight * self.link_strength * self.rule_multiplier()
    }

    /// True if some features of this event are an inseparable joint cause.
    pub fn inseparable(&self) -> bool {
        self.joint_causes.iter().any(|j| j.inseparable)
    }

    /// `"noisy_or"` or `"logistic"`.
    pub fn combine(&self) -> &'static str {
        if self.base_log_odds.is_some() {
            "logistic"
        } else {
            "noisy_or"
        }
    }

    /// Logistic: `base_log_odds + sum of the features' log_odds`, so that
    /// `link_strength = sigmoid(log_odds)`. `None` under noisy-OR.
    pub fn log_odds(&self) -> Option<f64> {
        let mut z = self.base_log_odds?;
        for c in &self.features {
            z += c.log_odds.unwrap_or(0.0);
        }
        Some(z)
    }

    /// Logistic: `exp(log_odds)` = base odds x every feature's multiplier.
    pub fn odds(&self) -> Option<f64> {
        self.log_odds().map(f64::exp)
    }

    /// Logistic: `exp(base_log_odds)`.
    pub fn base_odds(&self) -> Option<f64> {
        self.base_log_odds.map(f64::exp)
    }
}

impl fmt::Display for Explanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let joint: HashSet<&str> = self
            .joint_causes
            .iter()
            .filter(|j| j.inseparable)
            .flat_map(|j| j.features.iter().map(String::as_str))
            .collect();
        let mut parts: Vec<String> = self
            .joint_causes
            .iter()
            .filter(|j| j.inseparable)
            .map(ToString::to_string)
            .collect();
        parts.extend(
            self.features
                .iter()
                .filter(|c| !joint.contains(c.feature.as_str()))
                .map(ToString::to_string),
        );
        if self.prospective {
            write!(
                f,
                "no lag (pre-event) -> weight {} x link {}",
                fmt_g(self.weight, 3),
                fmt_f(self.link_strength, 3)
            )?;
        } else {
            write!(
                f,
                "lag {}h -> kernel {} x weight {} x link {}",
                fmt_f(self.dt / 3600.0, 2),
                fmt_f(self.kernel_value, 3),
                fmt_g(self.weight, 3),
                fmt_f(self.link_strength, 3)
            )?;
        }
        if !self.rules.is_empty() {
            let rs: Vec<String> = self.rules.iter().map(ToString::to_string).collect();
            write!(f, " x rules {}", rs.join(", "))?;
        }
        let p = if parts.is_empty() {
            "-".to_string()
        } else {
            parts.join(", ")
        };
        match (self.base_odds(), self.odds()) {
            (Some(b), Some(o)) => {
                // Every feature as a multiplier (joint causes are reported
                // below, they do not change the product).
                let ms: Vec<String> = self.features.iter().map(ToString::to_string).collect();
                let ms = if ms.is_empty() {
                    String::new()
                } else {
                    format!(" x {}", ms.join(" x "))
                };
                write!(
                    f,
                    " [logistic: base odds {}{ms} = odds {} -> link {}]",
                    fmt_g(b, 3),
                    fmt_g(o, 3),
                    fmt_g(self.link_strength, 3)
                )?;
                for j in self.joint_causes.iter().filter(|j| j.inseparable) {
                    write!(f, "\n      joint cause: {j}")?;
                }
            }
            _ => write!(f, " [noisy-OR of {p}]")?,
        }
        for j in self.joint_causes.iter().filter(|j| !j.inseparable) {
            write!(f, "\n      joint link (now separable): {j}")?;
        }
        for w in &self.warnings {
            write!(f, "\n      warning: {w}")?;
        }
        Ok(())
    }
}

/// One ranked candidate cause.
#[derive(Clone, Debug, PartialEq)]
pub struct Cause {
    /// 1-based rank.
    pub rank: usize,
    /// The event.
    pub event: Event,
    /// Its score.
    pub score: f64,
    /// Explanation, if requested.
    pub explanation: Option<Explanation>,
}

/// Work counters, so you can see the pruning happen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QueryStats {
    /// Segments overlapping the kernel support.
    pub segments_in_window: usize,
    /// Segments whose events were scored.
    pub segments_scanned: usize,
    /// Segments skipped thanks to the bound (or a zero bound).
    pub segments_pruned: usize,
    /// Events actually scored.
    pub events_scanned: usize,
    /// Segments bounded with the loose union bound.
    pub union_fallbacks: usize,
}

impl QueryStats {
    /// Add another query's counters.
    pub fn add(&mut self, o: &QueryStats) {
        self.segments_in_window += o.segments_in_window;
        self.segments_scanned += o.segments_scanned;
        self.segments_pruned += o.segments_pruned;
        self.events_scanned += o.events_scanned;
        self.union_fallbacks += o.union_fallbacks;
    }
}

impl fmt::Display for QueryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "segments: {} scanned / {} pruned / {} in window; events scored: {}",
            self.segments_scanned, self.segments_pruned, self.segments_in_window, self.events_scanned
        )
    }
}

/// Ranked causes (best first) plus execution statistics.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryResult {
    /// Entity (comma-joined for group queries).
    pub entity: String,
    /// Effect.
    pub effect: String,
    /// Effect time.
    pub at: f64,
    /// Causes, best first.
    pub causes: Vec<Cause>,
    /// Work counters.
    pub stats: QueryStats,
    /// Individual entities.
    pub entities: Vec<String>,
}

impl QueryResult {
    /// The single most likely cause ("pop the top").
    pub fn top(&self) -> Option<&Cause> {
        self.causes.first()
    }
    /// True for a group (multi-entity) query.
    pub fn is_group(&self) -> bool {
        self.entities.len() > 1
    }
}

impl fmt::Display for QueryResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let who = if self.is_group() {
            format!("entities {}", self.entity)
        } else {
            self.entity.clone()
        };
        write!(
            f,
            "Top causes of '{}' for {who} at t={}:",
            self.effect,
            format_time(self.at)
        )?;
        if self.causes.is_empty() {
            f.write_str("\n  (no candidate causes in window)")?;
        }
        for c in &self.causes {
            let ent = if self.is_group() {
                format!("{}: ", c.event.entity)
            } else {
                String::new()
            };
            write!(
                f,
                "\n  {}. {ent}{:<44} score={}",
                c.rank,
                c.event.short(),
                fmt_f(c.score, 4)
            )?;
            if let Some(e) = &c.explanation {
                write!(f, "\n      {e}")?;
            }
        }
        write!(f, "\n  [{}]", self.stats)
    }
}

/// One row of `rank_candidates` (best first).
#[derive(Clone, Debug, PartialEq)]
pub struct RankedCandidate {
    /// 1-based rank.
    pub rank: usize,
    /// Index of the candidate in the input.
    pub index: usize,
    /// Risk score.
    pub score: f64,
    /// Entity whose links were used.
    pub entity: Option<String>,
    /// Explanation, if requested.
    pub explanation: Option<Explanation>,
}

/// Result of a `RISK OF ...` statement.
#[derive(Clone, Debug, PartialEq)]
pub struct RiskResult {
    /// Effect.
    pub effect: String,
    /// Entity (None: global view).
    pub entity: Option<String>,
    /// Features, sorted.
    pub features: Vec<String>,
    /// Event type, if given.
    pub event_type: Option<String>,
    /// Risk score.
    pub score: f64,
    /// Explanation, if requested.
    pub explanation: Option<Explanation>,
}

impl fmt::Display for RiskResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let who = self.entity.as_deref().unwrap_or("(global)");
        let typ = match &self.event_type {
            Some(t) if !t.is_empty() => format!(" type={t}"),
            _ => String::new(),
        };
        write!(
            f,
            "Risk of '{}' for {who}{typ} [{}]: {}",
            self.effect,
            self.features.join(","),
            fmt_f(self.score, 4)
        )?;
        if let Some(e) = &self.explanation {
            write!(f, "\n      {e}")?;
        }
        Ok(())
    }
}

/// A judged event recorded for calibration.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationRecord {
    /// Entity.
    pub entity: String,
    /// Event id.
    pub event_id: u64,
    /// Effect time.
    pub at: f64,
    /// Score shown at feedback time, before links were updated.
    pub score: f64,
    /// Confirmed (true) or rejected.
    pub confirmed: bool,
}

impl CalibrationRecord {
    /// The score clipped to `[0, 1]` (scores are not probabilities).
    pub fn predicted(&self) -> f64 {
        self.score.clamp(0.0, 1.0)
    }
}

/// One bin of a calibration report.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationBin {
    /// Lower edge.
    pub lo: f64,
    /// Upper edge.
    pub hi: f64,
    /// Records in the bin.
    pub count: usize,
    /// Mean clipped score (None if empty).
    pub mean_predicted: Option<f64>,
    /// Confirm rate (None if empty).
    pub observed_rate: Option<f64>,
}

/// Reliability table + Brier score for one effect.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationReport {
    /// Effect.
    pub effect: String,
    /// Number of judged events.
    pub n: usize,
    /// Mean squared error of the clipped score (None without feedback).
    pub brier: Option<f64>,
    /// Equal-width bins on `[0, 1]`.
    pub bins: Vec<CalibrationBin>,
}

impl fmt::Display for CalibrationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let brier = self.brier.map_or("-".to_string(), |b| fmt_f(b, 4));
        write!(f, "Calibration of '{}': n={}, Brier={brier}", self.effect, self.n)?;
        f.write_str("\n  bin           count  mean_pred  observed")?;
        for b in self.bins.iter().filter(|b| b.count > 0) {
            write!(
                f,
                "\n  [{}, {})  {:>5}  {:>9}  {:>8}",
                fmt_f(b.lo, 2),
                fmt_f(b.hi, 2),
                b.count,
                fmt_f(b.mean_predicted.unwrap_or(f64::NAN), 3),
                fmt_f(b.observed_rate.unwrap_or(f64::NAN), 3)
            )?;
        }
        Ok(())
    }
}

/// What a feedback call did.
#[derive(Clone, Debug, PartialEq)]
pub struct FeedbackReport {
    /// Confirmed event id.
    pub confirmed: Option<u64>,
    /// Rejected event ids.
    pub rejected: Vec<u64>,
    /// Per-feature credit of the confirmed event (sorted by feature).
    pub credit: Vec<(String, f64)>,
}

/// Soft credit one event received from `observe`.
#[derive(Clone, Debug, PartialEq)]
pub struct EventCredit {
    /// Event id.
    pub event_id: u64,
    /// Its score.
    pub score: f64,
    /// `score / (leak + sum of scores)`.
    pub responsibility: f64,
    /// Feature -> alpha added (sorted by feature).
    pub features: Vec<(String, f64)>,
}

impl fmt::Display for EventCredit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut fs: Vec<&(String, f64)> = self.features.iter().collect();
        fs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let fs: Vec<String> = fs.iter().map(|(n, a)| format!("{n}+{}", fmt_f(*a, 3))).collect();
        write!(
            f,
            "#{} score={} r={} [{}]",
            self.event_id,
            fmt_f(self.score, 4),
            fmt_f(self.responsibility, 3),
            fs.join(", ")
        )
    }
}

/// What `observe` did for one effect occurrence.
#[derive(Clone, Debug, PartialEq)]
pub struct ObserveReport {
    /// Entity.
    pub entity: String,
    /// Effect.
    pub effect: String,
    /// Occurrence time.
    pub at: f64,
    /// Leak used.
    pub leak: f64,
    /// `leak / (leak + sum of scores)`.
    pub leak_responsibility: f64,
    /// Credits, by responsibility (desc).
    pub credits: Vec<EventCredit>,
}

impl ObserveReport {
    /// Sum of the events' responsibilities.
    pub fn total_credit(&self) -> f64 {
        crate::pyfmt::py_sum(self.credits.iter().map(|c| c.responsibility))
    }
}

impl fmt::Display for ObserveReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Observed '{}' for {} at t={}: {} candidate(s), leak {} took {}",
            self.effect,
            self.entity,
            fmt_g6(self.at),
            self.credits.len(),
            fmt_g6(self.leak),
            fmt_pct(self.leak_responsibility, 1)
        )?;
        for c in &self.credits {
            write!(f, "\n  {c}")?;
        }
        Ok(())
    }
}

/// What `settle` did (summed over effects and entities).
#[derive(Clone, Debug, PartialEq)]
pub struct SettleReport {
    /// Settle horizon.
    pub until: f64,
    /// Events whose full window was examined.
    pub events_settled: usize,
    /// Events that received negative evidence.
    pub events_rejected: usize,
    /// Total negative evidence added (per event).
    pub rejected_weight: f64,
    /// Skipped: had soft credit.
    pub skipped_credited: usize,
    /// Skipped: effect observed in the window (or after a one-shot effect).
    pub skipped_observed: usize,
    /// Effect -> events rejected.
    pub per_effect: Vec<(String, usize)>,
}

impl fmt::Display for SettleReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Settled until t={}: {} event(s) examined, {} rejected (weight {}), {} skipped (had soft credit), {} skipped (effect observed in window)",
            fmt_g6(self.until),
            self.events_settled,
            self.events_rejected,
            fmt_f(self.rejected_weight, 2),
            self.skipped_credited,
            self.skipped_observed
        )
    }
}

/// One feature of a [`LinkReport`].
#[derive(Clone, Debug, PartialEq)]
pub struct LinkRow {
    /// Feature.
    pub feature: String,
    /// Posterior mean.
    pub strength: f64,
    /// Posterior alpha.
    pub alpha: f64,
    /// Posterior beta.
    pub beta: f64,
    /// Effective evidence (excludes the prior).
    pub n: f64,
    /// Credible interval lower bound.
    pub lo: f64,
    /// Credible interval upper bound.
    pub hi: f64,
    /// `strength / baseline`.
    pub lift: f64,
    /// Logistic weight of the feature (log-odds multiplier); `None` under
    /// noisy-OR.
    pub log_odds: Option<f64>,
}

/// Features ranked by learned link strength for one effect.
///
/// These are **associations under the model**, not causal effects.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkReport {
    /// Effect.
    pub effect: String,
    /// Entity (None: global view).
    pub entity: Option<String>,
    /// Baseline for lift.
    pub baseline: f64,
    /// Credible level.
    pub level: f64,
    /// Rows, strongest first.
    pub rows: Vec<LinkRow>,
}

impl LinkReport {
    /// The row of a feature.
    pub fn row(&self, feature: &str) -> Option<&LinkRow> {
        self.rows.iter().find(|r| r.feature == feature)
    }
}

impl fmt::Display for LinkReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let who = self.entity.as_deref().unwrap_or("(global)");
        let pct = format!("{} CI", fmt_pct(self.level, 0));
        write!(
            f,
            "Links of '{}' for {who} (baseline {}; associations under the model, not causal effects):",
            self.effect,
            fmt_f(self.baseline, 3)
        )?;
        write!(
            f,
            "\n  {:<36} {:>6} {:>15} {:>6} {:>8} {:>8} {:>8}",
            "feature", "mean", pct, "lift", "n", "alpha", "beta"
        )?;
        for r in &self.rows {
            write!(
                f,
                "\n  {:<36} {:>6} [{}, {}] {:>6} {:>8} {:>8} {:>8}",
                r.feature,
                fmt_f(r.strength, 3),
                fmt_f(r.lo, 3),
                fmt_f(r.hi, 3),
                fmt_f(r.lift, 2),
                fmt_f(r.n, 1),
                fmt_f(r.alpha, 2),
                fmt_f(r.beta, 2)
            )?;
        }
        if self.rows.is_empty() {
            f.write_str("\n  (no features with enough support)")?;
        }
        Ok(())
    }
}
