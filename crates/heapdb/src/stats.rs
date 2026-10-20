//! Small statistics helpers.
//!
//! * [`betainc`]: the regularized incomplete beta function `I_x(a, b)` (the
//!   Beta CDF), by the Lentz continued fraction (Numerical Recipes `betacf`)
//!   with the usual symmetry switch.
//! * [`beta_ppf`]: Beta quantiles by bisection on [`betainc`].
//! * [`beta_interval`]: equal-tailed credible interval of a Beta belief.
//! * [`spearman`]: Spearman rank correlation (average ranks for ties).
//! * [`lgamma`]: `ln |Γ(x)|`, a port of CPython's `math.lgamma` (Lanczos
//!   approximation) so results match the reference implementation bit for
//!   bit.

use crate::error::{Error, Result};
use crate::pyfmt::py_sum;

const EPS: f64 = 1e-15;
const TINY: f64 = 1e-300;

const LANCZOS_N: usize = 13;
#[allow(clippy::excessive_precision)]
const LANCZOS_G: f64 = 6.024_680_040_776_729_583_740_234_375;
#[allow(clippy::excessive_precision)]
const LANCZOS_NUM: [f64; LANCZOS_N] = [
    23531376880.410759688572007674451636754734846804940,
    42919803642.649098768957899047001988850926355848959,
    35711959237.355668049440185451547166705960488635843,
    17921034426.037209699919755754458931112671403265390,
    6039542586.3520280050642916443072979210699388420708,
    1439720407.3117216736632230727949123939715485786772,
    248874557.86205415651146038641322942321632125127801,
    31426415.585400194380614231628318205362874684987640,
    2876370.6289353724412254090516208496135991145378768,
    186056.26539522349504029498971604569928220784236328,
    8071.6720023658162106380029022722506138218516325024,
    210.82427775157934587250973392071336271166969580291,
    2.5066282746310002701649081771338373386264310793408,
];
const LANCZOS_DEN: [f64; LANCZOS_N] = [
    0.0,
    39916800.0,
    120543840.0,
    150917976.0,
    105258076.0,
    45995730.0,
    13339535.0,
    2637558.0,
    357423.0,
    32670.0,
    1925.0,
    66.0,
    1.0,
];

fn lanczos_sum(x: f64) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    if x < 5.0 {
        for i in (0..LANCZOS_N).rev() {
            num = num * x + LANCZOS_NUM[i];
            den = den * x + LANCZOS_DEN[i];
        }
    } else {
        for i in 0..LANCZOS_N {
            num = num / x + LANCZOS_NUM[i];
            den = den / x + LANCZOS_DEN[i];
        }
    }
    num / den
}

/// `ln |Γ(x)|` for `x > 0` (CPython's `m_lgamma`). Returns `+inf` for
/// non-positive integers and NaN for NaN.
pub fn lgamma(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_nan() { x } else { f64::INFINITY };
    }
    if x == x.floor() && x <= 2.0 {
        return if x <= 0.0 { f64::INFINITY } else { 0.0 };
    }
    let absx = x.abs();
    if absx < 1e-20 {
        return -absx.ln();
    }
    let mut r = lanczos_sum(absx).ln() - LANCZOS_G;
    r += (absx - 0.5) * ((absx + LANCZOS_G - 0.5).ln() - 1.0);
    if x < 0.0 {
        // Reflection; never needed by HeapDB (Beta parameters are > 0).
        let pi = std::f64::consts::PI;
        r = pi.ln() - (pi * absx).sin().abs().ln() - absx.ln() - r;
    }
    r
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=500 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)` = CDF of Beta(a, b) at `x`.
pub fn betainc(a: f64, b: f64, x: f64) -> Result<f64> {
    if !(a > 0.0 && b > 0.0) {
        return Err(Error::invalid("a and b must be > 0"));
    }
    if x <= 0.0 {
        return Ok(0.0);
    }
    if x >= 1.0 {
        return Ok(1.0);
    }
    let ln_front = lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (-x).ln_1p();
    let front = ln_front.exp();
    let v = if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    };
    Ok(v.clamp(0.0, 1.0))
}

/// Quantile of Beta(a, b): the `x` with `I_x(a, b) = q` (bisection).
pub fn beta_ppf(q: f64, a: f64, b: f64) -> Result<f64> {
    beta_ppf_tol(q, a, b, 1e-10)
}

/// [`beta_ppf`] with an explicit bisection tolerance.
pub fn beta_ppf_tol(q: f64, a: f64, b: f64, tol: f64) -> Result<f64> {
    if !(0.0..=1.0).contains(&q) {
        return Err(Error::invalid("q must be in [0, 1]"));
    }
    if q == 0.0 {
        return Ok(0.0);
    }
    if q == 1.0 {
        return Ok(1.0);
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if betainc(a, b, mid)? < q {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < tol {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Equal-tailed `level` credible interval of Beta(a, b).
pub fn beta_interval(a: f64, b: f64, level: f64) -> Result<(f64, f64)> {
    if !(level > 0.0 && level < 1.0) {
        return Err(Error::invalid("level must be in (0, 1)"));
    }
    let tail = (1.0 - level) / 2.0;
    Ok((beta_ppf(tail, a, b)?, beta_ppf(1.0 - tail, a, b)?))
}

fn ranks(xs: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&i, &j| xs[i].partial_cmp(&xs[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0; xs.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && xs[order[j + 1]] == xs[order[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for &k in &order[i..=j] {
            out[k] = r;
        }
        i = j + 1;
    }
    out
}

/// Spearman rank correlation of two equal-length sequences.
pub fn spearman(xs: &[f64], ys: &[f64]) -> Result<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return Err(Error::invalid("need two sequences of equal length >= 2"));
    }
    let (rx, ry) = (ranks(xs), ranks(ys));
    let n = rx.len() as f64;
    let mx = py_sum(rx.iter().copied()) / n;
    let my = py_sum(ry.iter().copied()) / n;
    let cov = py_sum(rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)));
    let vx = py_sum(rx.iter().map(|a| (a - mx).powi(2)));
    let vy = py_sum(ry.iter().map(|b| (b - my).powi(2)));
    if vx == 0.0 || vy == 0.0 {
        return Ok(0.0);
    }
    Ok(cov / (vx * vy).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binom_tail(n: u64, k: u64, x: f64) -> f64 {
        let mut s = 0.0;
        for j in k..=n {
            let mut c = 1.0;
            for i in 0..j {
                c *= (n - i) as f64 / (i + 1) as f64;
            }
            s += c * x.powi(j as i32) * (1.0 - x).powi((n - j) as i32);
        }
        s
    }

    #[test]
    fn lgamma_known_values() {
        assert_eq!(lgamma(1.0), 0.0);
        assert_eq!(lgamma(2.0), 0.0);
        assert!((lgamma(5.0) - 24f64.ln()).abs() < 1e-13);
        assert!((lgamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-14);
        assert_eq!(lgamma(0.0), f64::INFINITY);
    }

    #[test]
    fn betainc_binomial_identity() {
        for (a, b, x) in [
            (2u64, 3u64, 0.5),
            (5, 7, 0.3),
            (1, 9, 0.05),
            (20, 30, 0.42),
            (50, 2, 0.97),
        ] {
            let got = betainc(a as f64, b as f64, x).unwrap();
            let want = binom_tail(a + b - 1, a, x);
            assert!((got - want).abs() < 1e-10, "{a} {b} {x}: {got} vs {want}");
        }
        assert!(betainc(0.0, 1.0, 0.5).is_err());
        assert_eq!(betainc(3.0, 4.0, 1.0).unwrap(), 1.0);
    }

    #[test]
    fn ppf_inverts_cdf() {
        for (a, b, q) in [(2.0, 8.0, 0.05), (40.0, 160.0, 0.1), (0.7, 3.3, 0.9)] {
            let x = beta_ppf(q, a, b).unwrap();
            assert!((betainc(a, b, x).unwrap() - q).abs() < 1e-8);
        }
        let (lo, hi) = beta_interval(1.0, 1.0, 0.9).unwrap();
        assert!((lo - 0.05).abs() < 1e-8 && (hi - 0.95).abs() < 1e-8);
        assert!(beta_interval(1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn spearman_basic() {
        assert!((spearman(&[1.0, 2.0, 3.0, 4.0], &[4.0, 3.0, 2.0, 1.0]).unwrap() + 1.0).abs() < 1e-12);
        assert!((spearman(&[1.0, 1.0, 2.0], &[5.0, 5.0, 9.0]).unwrap() - 1.0).abs() < 1e-12);
        assert!(spearman(&[1.0], &[1.0]).is_err());
    }
}
