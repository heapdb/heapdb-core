//! [`ForwardDecayHeap`]: a priority queue of exponentially decaying scores.
//!
//! We want "pop the item with the highest *current* score", where
//! `score(now) = w * exp(-lam * (now - t))`. In log space
//!
//! ```text
//! log score(now) = [log w + lam * t] - lam * now
//! ```
//!
//! the bracket (the *key*) is fixed at insert time and the second term is
//! the same for every item, so a heap ordered by key is correctly ordered at
//! every instant (forward decay, Cormode et al. 2009). Deletion uses
//! tombstones: `remove` is O(1) and dead entries are dropped lazily.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::{Error, Result};

struct Entry<T> {
    key: f64,
    handle: u64,
    item: T,
}

impl<T> PartialEq for Entry<T> {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl<T> Eq for Entry<T> {}
impl<T> PartialOrd for Entry<T> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl<T> Ord for Entry<T> {
    // Max-heap: higher key first, then lower handle (insertion order).
    fn cmp(&self, o: &Self) -> Ordering {
        self.key.total_cmp(&o.key).then(o.handle.cmp(&self.handle))
    }
}

/// An item with its score at the `now` passed to `pop`/`peek_top`/`top_k`.
#[derive(Clone, Debug, PartialEq)]
pub struct Popped<T> {
    /// The item.
    pub item: T,
    /// Score at `now`.
    pub score: f64,
    /// Handle returned by `push`.
    pub handle: u64,
}

/// Max-heap of items whose scores decay as `w * exp(-lam * age)`.
pub struct ForwardDecayHeap<T> {
    lam: f64,
    heap: BinaryHeap<Entry<T>>,
    dead: FxHashSet<u64>,
    live: FxHashMap<u64, f64>,
    next: u64,
}

impl<T> ForwardDecayHeap<T> {
    /// A heap with decay rate `lam` per second (`ln 2 / half_life`).
    pub fn new(lam: f64) -> Result<Self> {
        if lam < 0.0 {
            return Err(Error::invalid("lam must be >= 0"));
        }
        Ok(ForwardDecayHeap {
            lam,
            heap: BinaryHeap::new(),
            dead: FxHashSet::default(),
            live: FxHashMap::default(),
            next: 0,
        })
    }

    /// A heap with the given half-life (seconds).
    pub fn from_half_life(half_life: f64) -> Result<Self> {
        Self::new(std::f64::consts::LN_2 / half_life)
    }

    /// Decay rate.
    pub fn lam(&self) -> f64 {
        self.lam
    }

    /// Time-invariant priority `log(w) + lam * t`.
    pub fn key(&self, weight: f64, t: f64) -> Result<f64> {
        if weight <= 0.0 {
            return Err(Error::invalid("weight must be > 0 (log of 0 is undefined)"));
        }
        Ok(weight.ln() + self.lam * t)
    }

    /// Score at `now` of a key.
    pub fn score_at(&self, key: f64, now: f64) -> f64 {
        (key - self.lam * now).exp()
    }

    /// Insert an item observed at `t` with weight `w`; returns its handle.
    pub fn push(&mut self, item: T, weight: f64, t: f64) -> Result<u64> {
        let key = self.key(weight, t)?;
        let handle = self.next;
        self.next += 1;
        self.heap.push(Entry { key, handle, item });
        self.live.insert(handle, key);
        Ok(handle)
    }

    /// Tombstone an entry (O(1)). False if not live.
    pub fn remove(&mut self, handle: u64) -> bool {
        if self.live.remove(&handle).is_some() {
            self.dead.insert(handle);
            true
        } else {
            false
        }
    }

    fn skip_dead(&mut self) {
        while let Some(top) = self.heap.peek() {
            if !self.dead.contains(&top.handle) {
                break;
            }
            let e = self.heap.pop().expect("peeked");
            self.dead.remove(&e.handle);
        }
    }

    /// Highest-scoring live item at `now`, without removing it.
    pub fn peek_top(&mut self, now: f64) -> Option<Popped<&T>> {
        self.skip_dead();
        let lam = self.lam;
        self.heap.peek().map(|e| Popped {
            item: &e.item,
            score: (e.key - lam * now).exp(),
            handle: e.handle,
        })
    }

    /// Remove and return the highest-scoring live item at `now`.
    pub fn pop(&mut self, now: f64) -> Option<Popped<T>> {
        self.skip_dead();
        let e = self.heap.pop()?;
        self.live.remove(&e.handle);
        Some(Popped {
            score: self.score_at(e.key, now),
            item: e.item,
            handle: e.handle,
        })
    }

    /// The `k` best live items at `now`, best first, without removing.
    pub fn top_k(&self, k: usize, now: f64) -> Vec<Popped<&T>> {
        let mut v: Vec<&Entry<T>> = self
            .heap
            .iter()
            .filter(|e| !self.dead.contains(&e.handle))
            .collect();
        v.sort_by(|a, b| b.cmp(a));
        v.truncate(k);
        v.into_iter()
            .map(|e| Popped {
                item: &e.item,
                score: self.score_at(e.key, now),
                handle: e.handle,
            })
            .collect()
    }

    /// Physically drop all tombstoned entries (O(n)).
    pub fn compact(&mut self) {
        let dead = std::mem::take(&mut self.dead);
        let old = std::mem::take(&mut self.heap);
        self.heap = old.into_iter().filter(|e| !dead.contains(&e.handle)).collect();
    }

    /// Number of live items.
    pub fn len(&self) -> usize {
        self.live.len()
    }

    /// True if no live items.
    pub fn is_empty(&self) -> bool {
        self.live.is_empty()
    }

    /// Live items in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let mut v: Vec<&Entry<T>> = self
            .heap
            .iter()
            .filter(|e| !self.dead.contains(&e.handle))
            .collect();
        v.sort_by_key(|e| e.handle);
        v.into_iter().map(|e| &e.item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordering_is_time_invariant() {
        let lam = std::f64::consts::LN_2 / 300.0;
        let mut h = ForwardDecayHeap::new(lam).unwrap();
        let mut items = Vec::new();
        let mut x = 0.12345_f64;
        for _ in 0..200 {
            x = (x * 9301.0 + 49297.0) % 233280.0;
            let (w, t) = (0.1 + x / 233280.0 * 10.0, x / 233280.0 * 10_000.0);
            let hd = h.push((), w, t).unwrap();
            items.push((hd, w, t));
        }
        for now in [10_000.0, 20_000.0, 1e6] {
            let mut want: Vec<(f64, u64)> = items
                .iter()
                .map(|&(hd, w, t)| (w * (-lam * (now - t)).exp(), hd))
                .collect();
            want.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let got: Vec<u64> = h.top_k(items.len(), now).iter().map(|p| p.handle).collect();
            if now < 1e5 {
                assert_eq!(got, want.iter().map(|w| w.1).collect::<Vec<_>>());
            }
            assert_eq!(got.len(), items.len());
        }
    }

    #[test]
    fn tombstones() {
        let mut h = ForwardDecayHeap::from_half_life(60.0).unwrap();
        let a = h.push("a", 1.0, 0.0).unwrap();
        h.push("b", 1.0, 60.0).unwrap();
        let c = h.push("c", 5.0, 0.0).unwrap();
        assert!(h.remove(c) && !h.remove(c));
        assert_eq!(h.len(), 2);
        assert_eq!(*h.peek_top(100.0).unwrap().item, "b");
        assert_eq!(h.pop(100.0).unwrap().item, "b");
        assert_eq!(h.pop(100.0).unwrap().handle, a);
        assert!(h.pop(100.0).is_none());
        assert!(h.push("x", 0.0, 0.0).is_err());
    }
}
