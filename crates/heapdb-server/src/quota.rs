//! Rate limiting (per-database token bucket) and monthly billing periods.

use std::time::Instant;

use chrono::{DateTime, Datelike, Months, TimeZone, Utc};

/// Token bucket: `capacity = rpm`, refilled at `rpm / 60` tokens per second.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    rpm: u64,
    tokens: f64,
    last: Instant,
}

impl TokenBucket {
    /// A full bucket.
    pub fn new(rpm: u64) -> Self {
        TokenBucket {
            rpm,
            tokens: rpm as f64,
            last: Instant::now(),
        }
    }

    /// Requests per minute.
    pub fn rpm(&self) -> u64 {
        self.rpm
    }

    /// Take one token, or return the seconds until one is available.
    pub fn take(&mut self, now: Instant) -> Result<(), u64> {
        let rate = self.rpm as f64 / 60.0;
        let dt = now.saturating_duration_since(self.last).as_secs_f64();
        self.last = now;
        self.tokens = (self.tokens + dt * rate).min(self.rpm as f64);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else if rate <= 0.0 {
            Err(60)
        } else {
            Err(((1.0 - self.tokens) / rate).ceil().max(1.0) as u64)
        }
    }
}

/// First instant of the UTC month containing `now`.
pub fn month_start(now: i64) -> i64 {
    let dt = Utc.timestamp_opt(now, 0).single().unwrap_or_default();
    Utc.with_ymd_and_hms(dt.year(), dt.month(), 1, 0, 0, 0)
        .single()
        .map_or(now, |d| d.timestamp())
}

fn add_months(anchor: DateTime<Utc>, k: i64) -> DateTime<Utc> {
    if k >= 0 {
        anchor.checked_add_months(Months::new(k as u32)).unwrap_or(anchor)
    } else {
        anchor
            .checked_sub_months(Months::new((-k) as u32))
            .unwrap_or(anchor)
    }
}

/// The billing period `[start, end)` containing `now`: calendar months
/// counted from `anchor` (Unix seconds; default the first day of the
/// current UTC month). Month arithmetic clamps the day (Jan 31 -> Feb 28).
pub fn billing_period(anchor: Option<i64>, now: i64) -> (i64, i64) {
    let anchor = anchor.unwrap_or_else(|| month_start(now));
    let Some(a) = Utc.timestamp_opt(anchor, 0).single() else {
        return (now, now + 30 * 86400);
    };
    let n = Utc.timestamp_opt(now, 0).single().unwrap_or(a);
    let mut k = (n.year() as i64 - a.year() as i64) * 12 + (n.month() as i64 - a.month() as i64);
    // Adjust so start <= now < end.
    for _ in 0..4 {
        if add_months(a, k).timestamp() > now {
            k -= 1;
        } else if add_months(a, k + 1).timestamp() <= now {
            k += 1;
        } else {
            break;
        }
    }
    (add_months(a, k).timestamp(), add_months(a, k + 1).timestamp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn ts(y: i32, m: u32, d: u32) -> i64 {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap().timestamp()
    }

    #[test]
    fn periods() {
        assert_eq!(
            billing_period(None, ts(2026, 10, 20)),
            (ts(2026, 10, 1), ts(2026, 11, 1))
        );
        let a = ts(2026, 1, 15);
        assert_eq!(
            billing_period(Some(a), ts(2026, 3, 14)),
            (ts(2026, 2, 15), ts(2026, 3, 15))
        );
        assert_eq!(
            billing_period(Some(a), ts(2026, 3, 15)),
            (ts(2026, 3, 15), ts(2026, 4, 15))
        );
        assert_eq!(billing_period(Some(a), a), (a, ts(2026, 2, 15)));
        // Before the anchor: the month before it.
        assert_eq!(billing_period(Some(a), a - 1), (ts(2025, 12, 15), a));
        // Clamped end-of-month anchors.
        let b = ts(2026, 1, 31);
        let (s, e) = billing_period(Some(b), ts(2026, 2, 28) + 10);
        assert_eq!((s, e), (ts(2026, 2, 28), ts(2026, 3, 31)));
    }

    #[test]
    fn bucket() {
        let mut b = TokenBucket::new(2);
        let t = Instant::now();
        assert!(b.take(t).is_ok());
        assert!(b.take(t).is_ok());
        let wait = b.take(t).unwrap_err();
        assert!((1..=30).contains(&wait), "{wait}");
        assert!(b.take(t + Duration::from_secs(31)).is_ok());
    }
}
