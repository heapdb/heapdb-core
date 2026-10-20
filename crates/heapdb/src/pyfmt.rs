//! Python-compatible number and string formatting.
//!
//! HeapDB's human-readable output (explanations, reports, the REPL) was
//! defined by the original Python implementation, and the Python bindings
//! promise the same `str()` output. Rust's `{:e}`/`{:?}` differ from
//! Python's `format()` mini-language in exponent style, `g` handling and
//! non-finite values, so this module reimplements the few presentation
//! types HeapDB uses: `f`, `e`, `g`, `%`, `repr()` of floats and strings,
//! and Python's float floor division (used for time buckets).
//!
//! Fixed-point formatting delegates to Rust's `{:.N}`, which, like CPython,
//! rounds the exact binary value correctly (ties to even on exact halves).

/// `format(x, ".{prec}f")`.
pub fn fmt_f(x: f64, prec: usize) -> String {
    if let Some(s) = non_finite(x) {
        return s.to_string();
    }
    format!("{x:.prec$}")
}

/// `format(x, ".{prec}%")`: `x * 100` in fixed point, then `%`.
pub fn fmt_pct(x: f64, prec: usize) -> String {
    let mut s = fmt_f(x * 100.0, prec);
    s.push('%');
    s
}

/// `format(x, ".{prec}e")`.
pub fn fmt_e(x: f64, prec: usize) -> String {
    if let Some(s) = non_finite(x) {
        return s.to_string();
    }
    let s = format!("{x:.prec$e}");
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    format!("{mant}{}", py_exponent(exp))
}

/// `format(x, "g")` (precision 6).
pub fn fmt_g6(x: f64) -> String {
    fmt_g(x, 6)
}

/// `format(x, ".{prec}g")`.
pub fn fmt_g(x: f64, prec: usize) -> String {
    if let Some(s) = non_finite(x) {
        return s.to_string();
    }
    let p = prec.max(1);
    // Exponent after rounding to p significant digits.
    let sci = format!("{:.*e}", p - 1, x);
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if exp >= -4 && exp < p as i32 {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        strip_zeros(format!("{x:.decimals$}"))
    } else {
        format!("{}{}", strip_zeros(mant.to_string()), py_exponent(exp))
    }
}

/// `repr(x)` for a Python float (shortest round-trip digits).
pub fn repr_f64(x: f64) -> String {
    if let Some(s) = non_finite(x) {
        return s.to_string();
    }
    // Rust's `{:e}` gives the shortest round-trip digits in scientific form.
    let sci = format!("{x:e}");
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let decpt = exp + 1; // value = 0.DIGITS * 10^decpt
    let sign = if neg { "-" } else { "" };
    if decpt <= -4 || decpt > 16 {
        let m = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        return format!("{sign}{m}{}", py_exponent(exp));
    }
    let n = digits.len() as i32;
    let body = if decpt <= 0 {
        format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt >= n {
        format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
    } else {
        format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    };
    format!("{sign}{body}")
}

/// `repr(s)` for a Python `str`.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `repr(list_of_str)`, e.g. `['a', 'b']`.
pub fn repr_str_list<'a>(items: impl IntoIterator<Item = &'a str>) -> String {
    let parts: Vec<String> = items.into_iter().map(repr_str).collect();
    format!("[{}]", parts.join(", "))
}

/// Python's float floor division `x // y` (CPython `float_floor_div`).
///
/// Differs from `(x / y).floor()` near bucket boundaries; HeapDB's time
/// buckets must match the reference implementation exactly.
pub fn floordiv(vx: f64, wx: f64) -> f64 {
    let mut m = vx % wx;
    let mut div = (vx - m) / wx;
    if m != 0.0 {
        if (wx < 0.0) != (m < 0.0) {
            m += wx;
            div -= 1.0;
        }
    } else {
        m = 0.0_f64.copysign(wx);
    }
    let _ = m;
    if div != 0.0 {
        let mut floordiv = div.floor();
        if div - floordiv > 0.5 {
            floordiv += 1.0;
        }
        floordiv
    } else {
        0.0_f64.copysign(vx / wx)
    }
}

/// Python's builtin `sum()` over floats (CPython >= 3.12: Neumaier
/// compensated summation, starting from the integer 0).
pub fn py_sum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut it = xs.into_iter();
    let Some(first) = it.next() else {
        return 0.0;
    };
    // int 0 + float x
    let mut f = 0.0 + first;
    let mut c = 0.0;
    for x in it {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

fn strip_zeros(s: String) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0').trim_end_matches('.');
        t.to_string()
    } else {
        s
    }
}

fn py_exponent(exp: i32) -> String {
    let sign = if exp < 0 { '-' } else { '+' };
    format!("e{sign}{:02}", exp.abs())
}

fn non_finite(x: f64) -> Option<&'static str> {
    if x.is_nan() {
        Some("nan")
    } else if x == f64::INFINITY {
        Some("inf")
    } else if x == f64::NEG_INFINITY {
        Some("-inf")
    } else {
        None
    }
}

/// Unix seconds -> `YYYY-MM-DDTHH:MM:SSZ` (UTC), matching
/// `datetime.fromtimestamp(t, tz=timezone.utc).strftime(...)`.
pub fn utc_timestamp(t: f64) -> String {
    let mut secs = t.floor();
    let us = ((t - secs) * 1e6).round_ties_even();
    if us >= 1e6 {
        secs += 1.0;
    }
    let secs = secs as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Howard Hinnant's days-from-civil inverse.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `heapdb.events.format_time`: small (simulated) times as plain seconds,
/// epoch-like times as UTC.
pub fn format_time(t: f64) -> String {
    if t.abs() < 1e8 {
        fmt_g6(t)
    } else {
        utc_timestamp(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_format_matches_python() {
        let cases = [
            (7200.0, "7200"),
            (0.125, "0.125"),
            (1e16, "1e+16"),
            (1e-5, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (-0.0, "-0"),
            (0.0001, "0.0001"),
            (123456789.0, "1.23457e+08"),
            (100000.0, "100000"),
            (1000000.0, "1e+06"),
            (2.5, "2.5"),
            (f64::NAN, "nan"),
        ];
        for (x, want) in cases {
            assert_eq!(fmt_g6(x), want, "{x}");
        }
        assert_eq!(fmt_g(2.675, 2), "2.7");
        assert_eq!(fmt_g(0.30000000000000004, 3), "0.3");
        assert_eq!(fmt_g(1.5, 3), "1.5");
    }

    #[test]
    fn repr_matches_python() {
        let cases = [
            (7200.0, "7200.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1.2345678901234568e17, "1.2345678901234568e+17"),
            (-2.5, "-2.5"),
            (0.0, "0.0"),
            (5e-324, "5e-324"),
        ];
        for (x, want) in cases {
            assert_eq!(repr_f64(x), want);
        }
    }

    #[test]
    fn str_repr_and_time() {
        assert_eq!(repr_str("me"), "'me'");
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str_list(["a", "b"]), "['a', 'b']");
        assert_eq!(utc_timestamp(1714521600.0), "2024-05-01T00:00:00Z");
        assert_eq!(format_time(484200.0), "484200");
        assert_eq!(fmt_pct(0.0512, 1), "5.1%");
        assert_eq!(fmt_e(1234.5, 2), "1.23e+03");
    }

    #[test]
    fn floordiv_and_sum() {
        assert_eq!(floordiv(7.0, 2.0), 3.0);
        assert_eq!(floordiv(-7.0, 2.0), -4.0);
        assert_eq!(floordiv(0.3, 0.1), 2.0); // Python: 0.3 // 0.1 == 2.0
        assert_eq!(py_sum([0.1; 10]), 1.0); // compensated, like Python 3.12+
        assert_eq!(py_sum(Vec::<f64>::new()), 0.0);
    }
}
