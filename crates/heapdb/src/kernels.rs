//! Lag kernels: how plausible is a cause that happened `dt` seconds ago?
//!
//! A kernel maps a non-negative lag `dt = effect_time - event_time` to a
//! value in `[0, 1]`. Every kernel is *peak-normalised* (its maximum is 1.0)
//! and *truncated* at a finite [`Kernel::support`]: events older than the
//! support window score 0 and are never looked at by the query engine.
//!
//! For pruning, the engine needs an upper bound of the kernel over a range
//! of lags ([`Kernel::max_over`]). All kernels here are unimodal, so the bound
//! is exact: the peak if the mode lies in the range, otherwise the larger of
//! the two endpoint values.

use std::sync::OnceLock;

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::pyfmt::repr_f64;

/// The shape (and parameters) of a [`Kernel`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KernelShape {
    /// `0.5 ** (dt / half_life)`, truncated at `horizon_half_lives` half-lives.
    Exponential {
        /// Half-life in seconds (> 0).
        half_life: f64,
        /// Support in half-lives (default 10).
        horizon_half_lives: f64,
    },
    /// Flat: 1 for `min_lag <= dt <= max_lag`.
    Window {
        /// Minimum lag in seconds.
        min_lag: f64,
        /// Maximum lag in seconds.
        max_lag: f64,
    },
    /// Log-normal shaped delay, peak-normalised, truncated at
    /// `median * exp(tail_sigmas * sigma)`.
    LogNormal {
        /// Median lag in seconds (> 0).
        median: f64,
        /// Log-space standard deviation (> 0).
        sigma: f64,
        /// Truncation in sigmas (default 4).
        tail_sigmas: f64,
    },
    /// Gamma shaped delay `dt^(shape-1) exp(-dt/scale)`, peak-normalised,
    /// truncated at `mean + tail_stds * std`.
    Gamma {
        /// Shape (>= 1, so the peak is finite).
        shape: f64,
        /// Scale in seconds (> 0).
        scale: f64,
        /// Truncation in standard deviations (default 6).
        tail_stds: f64,
    },
}

/// A unimodal, peak-normalised, truncated lag kernel.
#[derive(Debug)]
pub struct Kernel {
    shape: KernelShape,
    support: f64,
    mu: f64,
    mass: OnceLock<MassTable>,
}

#[derive(Debug, Clone)]
struct MassTable {
    lo: f64,
    step: f64,
    cum: Vec<f64>,
}

impl Clone for Kernel {
    fn clone(&self) -> Self {
        Kernel {
            shape: self.shape,
            support: self.support,
            mu: self.mu,
            mass: self.mass.clone(),
        }
    }
}

impl PartialEq for Kernel {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
}

/// Grid size of the numeric cumulative-mass table.
const MASS_GRID: usize = 2048;

impl Kernel {
    fn build(shape: KernelShape) -> Self {
        let (support, mu) = match shape {
            KernelShape::Exponential {
                half_life,
                horizon_half_lives,
            } => (half_life * horizon_half_lives, 0.0),
            KernelShape::Window { max_lag, .. } => (max_lag, 0.0),
            KernelShape::LogNormal {
                median,
                sigma,
                tail_sigmas,
            } => (median * (tail_sigmas * sigma).exp(), median.ln()),
            KernelShape::Gamma {
                shape,
                scale,
                tail_stds,
            } => {
                let mean = shape * scale;
                let std = shape.sqrt() * scale;
                (mean + tail_stds * std, 0.0)
            }
        };
        Kernel {
            shape,
            support,
            mu,
            mass: OnceLock::new(),
        }
    }

    /// Exponential decay with the default horizon of 10 half-lives.
    pub fn exponential(half_life: f64) -> Result<Self> {
        Self::exponential_with_horizon(half_life, 10.0)
    }

    /// Exponential decay truncated at `horizon_half_lives` half-lives.
    pub fn exponential_with_horizon(half_life: f64, horizon_half_lives: f64) -> Result<Self> {
        if half_life <= 0.0 {
            return Err(Error::invalid("half_life must be > 0"));
        }
        Ok(Self::build(KernelShape::Exponential {
            half_life,
            horizon_half_lives,
        }))
    }

    /// Flat window `[min_lag, max_lag]`.
    pub fn window(min_lag: f64, max_lag: f64) -> Result<Self> {
        if !(0.0 <= min_lag && min_lag <= max_lag) {
            return Err(Error::invalid("need 0 <= min_lag <= max_lag"));
        }
        Ok(Self::build(KernelShape::Window { min_lag, max_lag }))
    }

    /// Log-normal kernel with the default truncation of 4 sigmas.
    pub fn lognormal(median: f64, sigma: f64) -> Result<Self> {
        Self::lognormal_with_tail(median, sigma, 4.0)
    }

    /// Log-normal kernel truncated at `median * exp(tail_sigmas * sigma)`.
    pub fn lognormal_with_tail(median: f64, sigma: f64, tail_sigmas: f64) -> Result<Self> {
        if median <= 0.0 || sigma <= 0.0 {
            return Err(Error::invalid("median and sigma must be > 0"));
        }
        Ok(Self::build(KernelShape::LogNormal {
            median,
            sigma,
            tail_sigmas,
        }))
    }

    /// Gamma kernel with the default truncation of 6 standard deviations.
    pub fn gamma(shape: f64, scale: f64) -> Result<Self> {
        Self::gamma_with_tail(shape, scale, 6.0)
    }

    /// Gamma kernel truncated at `mean + tail_stds * std`.
    pub fn gamma_with_tail(shape: f64, scale: f64, tail_stds: f64) -> Result<Self> {
        if shape < 1.0 || scale <= 0.0 {
            return Err(Error::invalid("need shape >= 1 and scale > 0"));
        }
        Ok(Self::build(KernelShape::Gamma {
            shape,
            scale,
            tail_stds,
        }))
    }

    /// The kernel's shape and parameters.
    pub fn shape(&self) -> &KernelShape {
        &self.shape
    }

    /// Short name used in snapshots and the REPL (`exp`, `window`,
    /// `lognormal`, `gamma`).
    pub fn name(&self) -> &'static str {
        match self.shape {
            KernelShape::Exponential { .. } => "exp",
            KernelShape::Window { .. } => "window",
            KernelShape::LogNormal { .. } => "lognormal",
            KernelShape::Gamma { .. } => "gamma",
        }
    }

    /// Name of the corresponding Python class.
    pub fn class_name(&self) -> &'static str {
        match self.shape {
            KernelShape::Exponential { .. } => "ExponentialDecay",
            KernelShape::Window { .. } => "WindowKernel",
            KernelShape::LogNormal { .. } => "LogNormalKernel",
            KernelShape::Gamma { .. } => "GammaKernel",
        }
    }

    /// Maximum lookback in seconds: `value(dt) == 0` for `dt > support`.
    pub fn support(&self) -> f64 {
        self.support
    }

    /// Minimum lag in seconds: `value(dt) == 0` for `dt < min_lag`.
    pub fn min_lag(&self) -> f64 {
        match self.shape {
            KernelShape::Window { min_lag, .. } => min_lag,
            _ => 0.0,
        }
    }

    /// Lag at which the kernel peaks.
    pub fn mode(&self) -> f64 {
        match self.shape {
            KernelShape::Exponential { .. } => 0.0,
            KernelShape::Window { min_lag, .. } => min_lag,
            KernelShape::LogNormal { sigma, .. } => (self.mu - sigma.powi(2)).exp(),
            KernelShape::Gamma { shape, scale, .. } => (shape - 1.0) * scale,
        }
    }

    /// `ln(median)` of a log-normal kernel (0 otherwise).
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Peak value: always 1.0 (kernels are peak-normalised).
    pub fn max_value(&self) -> f64 {
        1.0
    }

    /// The un-normalised shape (for `min_lag <= dt <= support`).
    pub fn raw(&self, dt: f64) -> f64 {
        match self.shape {
            KernelShape::Exponential { half_life, .. } => 0.5_f64.powf(dt / half_life),
            KernelShape::Window { .. } => 1.0,
            KernelShape::LogNormal { sigma, .. } => {
                if dt <= 0.0 {
                    return 0.0;
                }
                let z = (dt.ln() - self.mu) / sigma;
                (-0.5 * z * z).exp() / dt
            }
            KernelShape::Gamma { shape, scale, .. } => {
                if dt < 0.0 {
                    return 0.0;
                }
                if dt == 0.0 {
                    return if shape == 1.0 { 1.0 } else { 0.0 };
                }
                ((shape - 1.0) * dt.ln() - dt / scale).exp()
            }
        }
    }

    /// Kernel value for a cause that happened `dt` seconds before the
    /// effect. Causes in the future (`dt < 0`) and outside
    /// `[min_lag, support]` score 0.
    pub fn value(&self, dt: f64) -> f64 {
        if dt < self.min_lag() || dt > self.support || dt < 0.0 {
            return 0.0;
        }
        let peak = self.raw(self.mode());
        if peak > 0.0 {
            (self.raw(dt) / peak).min(1.0)
        } else {
            0.0
        }
    }

    /// Upper bound (here: exact maximum) of [`Kernel::value`] over
    /// `[dt_lo, dt_hi]`, used to bound every event of a segment at once.
    pub fn max_over(&self, dt_lo: f64, dt_hi: f64) -> f64 {
        if let KernelShape::Window { min_lag, max_lag } = self.shape {
            return if dt_lo.max(min_lag) <= dt_hi.min(max_lag) {
                1.0
            } else {
                0.0
            };
        }
        let lo = dt_lo.max(self.min_lag()).max(0.0);
        let hi = dt_hi.min(self.support);
        if lo > hi {
            return 0.0;
        }
        let mode = self.mode();
        if lo <= mode && mode <= hi {
            return self.max_value();
        }
        self.value(lo).max(self.value(hi))
    }

    /// Fraction of the kernel's total mass at lags `<= dt` (0 below
    /// `min_lag`, 1 at or beyond `support`). Trapezoidal integration on a
    /// 2048-point grid, computed once per kernel and linearly interpolated.
    pub fn mass_fraction(&self, dt: f64) -> f64 {
        let (lo, hi) = (self.min_lag().max(0.0), self.support);
        if dt <= lo {
            return 0.0;
        }
        if dt >= hi {
            return 1.0;
        }
        let t = self.mass.get_or_init(|| {
            let n = MASS_GRID;
            let step = (hi - lo) / n as f64;
            let vals: Vec<f64> = (0..=n).map(|i| self.value(lo + i as f64 * step)).collect();
            let mut cum = Vec::with_capacity(n + 1);
            cum.push(0.0);
            for i in 0..n {
                let last = cum[i];
                cum.push(last + 0.5 * (vals[i] + vals[i + 1]) * step);
            }
            let total = if cum[n] != 0.0 { cum[n] } else { 1.0 };
            MassTable {
                lo,
                step,
                cum: cum.into_iter().map(|c| c / total).collect(),
            }
        });
        let x = (dt - t.lo) / t.step;
        let i = (x as usize).min(t.cum.len() - 2);
        let frac = x - i as f64;
        t.cum[i] + frac * (t.cum[i + 1] - t.cum[i])
    }

    /// Parameters in snapshot order (excluding `kind`).
    pub fn params(&self) -> Vec<(&'static str, f64)> {
        match self.shape {
            KernelShape::Exponential {
                half_life,
                horizon_half_lives,
            } => vec![
                ("half_life", half_life),
                ("horizon_half_lives", horizon_half_lives),
            ],
            KernelShape::Window { min_lag, max_lag } => {
                vec![("min_lag", min_lag), ("max_lag", max_lag)]
            }
            KernelShape::LogNormal {
                median,
                sigma,
                tail_sigmas,
            } => vec![("median", median), ("sigma", sigma), ("tail_sigmas", tail_sigmas)],
            KernelShape::Gamma {
                shape,
                scale,
                tail_stds,
            } => vec![("shape", shape), ("scale", scale), ("tail_stds", tail_stds)],
        }
    }

    /// Snapshot form: `{"kind": ..., <params>}`.
    pub fn to_json(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("kind".into(), Value::from(self.name()));
        for (k, v) in self.params() {
            m.insert(k.into(), Value::from(v));
        }
        m
    }

    /// Inverse of [`Kernel::to_json`].
    pub fn from_json(d: &Map<String, Value>) -> Result<Self> {
        let kind = d
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Snapshot("kernel record has no 'kind'".into()))?;
        let allowed: &[&str] = match kind {
            "exp" => &["half_life", "horizon_half_lives"],
            "window" => &["min_lag", "max_lag"],
            "lognormal" => &["median", "sigma", "tail_sigmas"],
            "gamma" => &["shape", "scale", "tail_stds"],
            _ => {
                return Err(Error::invalid(format!(
                    "unknown kernel kind {}",
                    crate::pyfmt::repr_str(kind)
                )))
            }
        };
        for k in d.keys() {
            if k != "kind" && !allowed.contains(&k.as_str()) {
                return Err(Error::Type(format!(
                    "kernel {kind} got an unexpected keyword argument {}",
                    crate::pyfmt::repr_str(k)
                )));
            }
        }
        let num = |k: &str, default: Option<f64>| -> Result<f64> {
            match d.get(k) {
                Some(v) => v
                    .as_f64()
                    .ok_or_else(|| Error::Type(format!("kernel parameter {k} must be a number"))),
                None => default.ok_or_else(|| Error::Type(format!("kernel {kind} missing argument {k}"))),
            }
        };
        match kind {
            "exp" => Self::exponential_with_horizon(
                num("half_life", None)?,
                num("horizon_half_lives", Some(10.0))?,
            ),
            "window" => Self::window(num("min_lag", Some(0.0))?, num("max_lag", Some(3600.0))?),
            "lognormal" => Self::lognormal_with_tail(
                num("median", None)?,
                num("sigma", Some(0.5))?,
                num("tail_sigmas", Some(4.0))?,
            ),
            _ => Self::gamma_with_tail(
                num("shape", None)?,
                num("scale", None)?,
                num("tail_stds", Some(6.0))?,
            ),
        }
    }

    /// Python-style repr, e.g. `LogNormalKernel(median=7200.0, sigma=0.6, tail_sigmas=4.0)`.
    pub fn repr(&self) -> String {
        let args: Vec<String> = self
            .params()
            .into_iter()
            .map(|(k, v)| format!("{k}={}", repr_f64(v)))
            .collect();
        format!("{}({})", self.class_name(), args.join(", "))
    }
}

impl std::fmt::Display for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.repr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<Kernel> {
        vec![
            Kernel::exponential(600.0).unwrap(),
            Kernel::window(300.0, 3600.0).unwrap(),
            Kernel::lognormal(7200.0, 0.6).unwrap(),
            Kernel::gamma(3.0, 1800.0).unwrap(),
            Kernel::gamma(1.0, 900.0).unwrap(),
        ]
    }

    #[test]
    fn exponential_half_life() {
        let k = Kernel::exponential(600.0).unwrap();
        assert!((k.value(600.0) - 0.5).abs() < 1e-12);
        assert_eq!(k.support(), 6000.0);
        assert_eq!(k.value(6001.0), 0.0);
    }

    #[test]
    fn peaks_and_negative_lags() {
        for k in all() {
            assert!((k.value(k.mode()) - 1.0).abs() < 1e-12 || k.mode() == 0.0 && k.name() == "lognormal");
            assert_eq!(k.value(-1.0), 0.0);
            assert_eq!(k.max_over(-k.support(), -1e-9), 0.0);
        }
    }

    #[test]
    fn max_over_bounds_value() {
        for k in all() {
            let s = k.support();
            for i in 0..200 {
                let a = (i as f64 / 200.0 * 1.3 - 0.1) * s;
                let b = a + 0.17 * s;
                let bound = k.max_over(a, b);
                for j in 0..=20 {
                    let dt = a + (b - a) * j as f64 / 20.0;
                    assert!(k.value(dt) <= bound + 1e-12);
                }
            }
        }
    }

    #[test]
    fn json_roundtrip_and_repr() {
        for k in all() {
            let k2 = Kernel::from_json(&k.to_json()).unwrap();
            assert_eq!(k, k2);
        }
        assert_eq!(
            Kernel::exponential(600.0).unwrap().repr(),
            "ExponentialDecay(half_life=600.0, horizon_half_lives=10.0)"
        );
        let mut bad = Map::new();
        bad.insert("kind".into(), "nope".into());
        assert!(Kernel::from_json(&bad).is_err());
    }

    #[test]
    fn mass_fraction_monotone() {
        let k = Kernel::window(0.0, 100.0).unwrap();
        assert!((k.mass_fraction(25.0) - 0.25).abs() < 1e-3);
        let ln = Kernel::lognormal(100.0, 0.5).unwrap();
        let mut prev = 0.0;
        for t in 0..800 {
            let v = ln.mass_fraction(t as f64);
            assert!(v >= prev);
            prev = v;
        }
        assert_eq!(prev, 1.0);
    }
}
