//! [`MemoryHeaps`]: heaps in process memory.
//!
//! Per heap, items live in a map by id plus an ordered index of the ranked
//! ones on `(key descending, id ascending)`. `top_k` and `pop` walk the
//! index from the top and stop after `k` matches, so an unfiltered (or
//! mostly matching) query costs `O(k log n)` regardless of heap size.
//!
//! Batched mutations (`push_many`, ...) are atomic: every item is computed
//! first and nothing changes unless all of them succeed.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::filter::Filter;
use super::{
    check_attrs, check_finite, check_name, check_positive, item_key, lambda, logaddexp, push_activity,
    too_far, HeapBackend, HeapInfo, HeapItem, HeapSpec, Ranked,
};
use crate::error::{Error, Result};
use crate::event::Attrs;

/// One item of a batched push.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PushArgs {
    /// Item id.
    pub id: String,
    /// Weight of the push (finite, > 0).
    pub weight: f64,
    /// Time of the push (Unix seconds).
    pub time: f64,
    /// Replaces the attributes when given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attrs: Option<Attrs>,
}

/// One item of a batched `set_base`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaseArgs {
    /// Item id.
    pub id: String,
    /// New base (finite, > 0).
    pub base: f64,
}

/// One item of a batched `set_attrs`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttrsArgs {
    /// Item id.
    pub id: String,
    /// Attributes.
    pub attrs: Attrs,
    /// Shallow-merge into the existing attributes instead of replacing them.
    #[serde(default)]
    pub merge: bool,
}

/// An `f64` key ordered descending (`total_cmp`), for the index.
#[derive(Clone, Copy, Debug)]
struct Desc(f64);

impl PartialEq for Desc {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Desc {}
impl PartialOrd for Desc {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Desc {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0)
    }
}

/// One heap.
#[derive(Clone, Debug, Default)]
struct Heap {
    half_life: f64,
    landmark: f64,
    meta: Attrs,
    items: HashMap<String, HeapItem>,
    /// Ranked items: `(key desc, id asc)`.
    index: BTreeSet<(Desc, String)>,
}

impl Heap {
    fn info(&self, name: &str) -> HeapInfo {
        HeapInfo {
            name: name.to_string(),
            half_life: self.half_life,
            landmark: self.landmark,
            meta: self.meta.clone(),
            size: self.items.len() as u64,
        }
    }

    fn lam(&self) -> f64 {
        lambda(self.half_life)
    }

    /// Insert or replace an item, keeping the index in sync.
    fn put(&mut self, item: HeapItem) {
        if let Some(old) = self.items.get(&item.id) {
            if let Some(k) = old.key {
                self.index.remove(&(Desc(k), item.id.clone()));
            }
        }
        if let Some(k) = item.key {
            self.index.insert((Desc(k), item.id.clone()));
        }
        self.items.insert(item.id.clone(), item);
    }

    fn take(&mut self, id: &str) -> Option<HeapItem> {
        let item = self.items.remove(id)?;
        if let Some(k) = item.key {
            self.index.remove(&(Desc(k), item.id.clone()));
        }
        Some(item)
    }

    /// Ranked items matching `filter`, best first, at most `k`.
    fn top(&self, k: usize, filter: Option<&Filter>) -> Vec<&HeapItem> {
        self.index
            .iter()
            .map(|(_, id)| &self.items[id])
            .filter(|it| filter.is_none_or(|f| f.matches(&it.attrs)))
            .take(k)
            .collect()
    }

    /// Apply `f` to each id in order on a scratch copy of the affected items
    /// (later entries see earlier ones), then commit all of them.
    fn batch<T>(
        &mut self,
        args: &[T],
        id_of: impl Fn(&T) -> &str,
        mut f: impl FnMut(&Heap, Option<&HeapItem>, &T) -> Result<HeapItem>,
    ) -> Result<Vec<HeapItem>> {
        let mut scratch: HashMap<&str, HeapItem> = HashMap::new();
        let mut out = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let id = id_of(a);
            check_name("item id", id).map_err(|e| batch_err(args.len(), i, e))?;
            let cur = scratch.get(id).or_else(|| self.items.get(id));
            let next = f(self, cur, a).map_err(|e| batch_err(args.len(), i, e))?;
            scratch.insert(id, next.clone());
            out.push(next);
        }
        for (_, item) in scratch {
            self.put(item);
        }
        Ok(out)
    }
}

/// Prefix errors with the item position in batches of more than one.
fn batch_err(n: usize, i: usize, e: Error) -> Error {
    match e {
        Error::Invalid(m) if n > 1 => Error::Invalid(format!("items[{i}]: {m}")),
        other => other,
    }
}

fn new_item(id: &str) -> HeapItem {
    HeapItem {
        id: id.to_string(),
        base: 1.0,
        activity: None,
        key: None,
        attrs: Attrs::new(),
        version: 0,
    }
}

/// Fill in `key` and bump the version; reject a non-finite key.
fn finish(mut item: HeapItem) -> Result<HeapItem> {
    item.key = item_key(item.base, item.activity);
    if item.key.is_some_and(|k| !k.is_finite()) {
        return Err(too_far("the push time"));
    }
    item.version += 1;
    Ok(item)
}

/// Heaps in process memory (the default `persist_to`).
#[derive(Clone, Debug, Default)]
pub struct MemoryHeaps {
    heaps: BTreeMap<String, Heap>,
}

impl MemoryHeaps {
    /// No heaps.
    pub fn new() -> Self {
        Self::default()
    }

    /// True if there are no heaps.
    pub fn is_empty(&self) -> bool {
        self.heaps.is_empty()
    }

    fn get_heap(&self, name: &str) -> Result<&Heap> {
        self.heaps
            .get(name)
            .ok_or_else(|| Error::UnknownHeap(name.to_string()))
    }

    fn get_heap_mut(&mut self, name: &str) -> Result<&mut Heap> {
        self.heaps
            .get_mut(name)
            .ok_or_else(|| Error::UnknownHeap(name.to_string()))
    }

    /// Atomic batch of pushes (see [`HeapBackend::push`]).
    pub fn push_many(&mut self, heap: &str, items: &[PushArgs]) -> Result<Vec<HeapItem>> {
        for (i, a) in items.iter().enumerate() {
            let pre = || -> Result<()> {
                check_positive("weight", a.weight)?;
                check_finite("time", a.time)?;
                a.attrs.as_ref().map_or(Ok(()), check_attrs)
            };
            pre().map_err(|e| batch_err(items.len(), i, e))?;
        }
        let h = self.get_heap_mut(heap)?;
        let (lam, landmark) = (h.lam(), h.landmark);
        h.batch(
            items,
            |a| &a.id,
            |_, cur, a| {
                let mut it = cur.cloned().unwrap_or_else(|| new_item(&a.id));
                let x = push_activity(a.weight, a.time, lam, landmark);
                if !x.is_finite() {
                    return Err(too_far("the push time"));
                }
                it.activity = Some(it.activity.map_or(x, |act| logaddexp(act, x)));
                if let Some(attrs) = &a.attrs {
                    it.attrs = attrs.clone();
                }
                finish(it)
            },
        )
    }

    /// Atomic batch of `set_base`.
    pub fn set_base_many(&mut self, heap: &str, items: &[BaseArgs]) -> Result<Vec<HeapItem>> {
        for (i, a) in items.iter().enumerate() {
            check_positive("base", a.base).map_err(|e| batch_err(items.len(), i, e))?;
        }
        self.get_heap_mut(heap)?.batch(
            items,
            |a| &a.id,
            |_, cur, a| {
                let mut it = cur.cloned().unwrap_or_else(|| new_item(&a.id));
                it.base = a.base;
                finish(it)
            },
        )
    }

    /// Atomic batch of `set_attrs`.
    pub fn set_attrs_many(&mut self, heap: &str, items: &[AttrsArgs]) -> Result<Vec<HeapItem>> {
        for (i, a) in items.iter().enumerate() {
            check_attrs(&a.attrs).map_err(|e| batch_err(items.len(), i, e))?;
        }
        self.get_heap_mut(heap)?.batch(
            items,
            |a| &a.id,
            |_, cur, a| {
                let mut it = cur.cloned().unwrap_or_else(|| new_item(&a.id));
                if a.merge {
                    for (k, v) in &a.attrs {
                        it.attrs.insert(k.clone(), v.clone());
                    }
                } else {
                    it.attrs = a.attrs.clone();
                }
                finish(it)
            },
        )
    }

    /// Total number of items over all heaps.
    pub fn item_count(&self) -> u64 {
        self.heaps.values().map(|h| h.items.len() as u64).sum()
    }

    // -- serialisation (NativeHeaps checkpoints, server snapshots) -----------

    /// The full state as JSON records: per heap
    /// `{"kind":"heap", name, half_life, landmark, meta}` followed by its
    /// items `{"kind":"item", heap, id, base, activity, attrs, version}`
    /// (ordered by id, so the output is deterministic).
    pub fn to_records(&self) -> Vec<Value> {
        let mut out = Vec::new();
        for (name, h) in &self.heaps {
            let mut m = Map::new();
            m.insert("kind".into(), "heap".into());
            m.insert("name".into(), name.clone().into());
            m.insert("half_life".into(), h.half_life.into());
            m.insert("landmark".into(), h.landmark.into());
            m.insert("meta".into(), Value::Object(h.meta.clone()));
            out.push(Value::Object(m));
            let mut ids: Vec<&String> = h.items.keys().collect();
            ids.sort();
            for id in ids {
                let it = &h.items[id];
                let mut m = Map::new();
                m.insert("kind".into(), "item".into());
                m.insert("heap".into(), name.clone().into());
                m.insert("id".into(), it.id.clone().into());
                m.insert("base".into(), it.base.into());
                m.insert("activity".into(), it.activity.into());
                m.insert("attrs".into(), Value::Object(it.attrs.clone()));
                m.insert("version".into(), it.version.into());
                out.push(Value::Object(m));
            }
        }
        out
    }

    /// Add one record written by [`MemoryHeaps::to_records`]. Returns false
    /// for a record of another kind (left to the caller).
    pub fn load_record(&mut self, rec: &Map<String, Value>) -> Result<bool> {
        let bad = |what: &str| Error::Snapshot(format!("heap record: missing or invalid {what}"));
        let s = |k: &str| rec.get(k).and_then(Value::as_str).ok_or_else(|| bad(k));
        let f = |k: &str| rec.get(k).and_then(Value::as_f64).ok_or_else(|| bad(k));
        let obj = |k: &str| match rec.get(k) {
            Some(Value::Object(m)) => Ok(m.clone()),
            None => Ok(Attrs::new()),
            _ => Err(bad(k)),
        };
        match rec.get("kind").and_then(Value::as_str) {
            Some("heap") => {
                let h = Heap {
                    half_life: f("half_life")?,
                    landmark: f("landmark")?,
                    meta: obj("meta")?,
                    ..Heap::default()
                };
                self.heaps.insert(s("name")?.to_string(), h);
                Ok(true)
            }
            Some("item") => {
                let activity = match rec.get("activity") {
                    None | Some(Value::Null) => None,
                    Some(v) => Some(v.as_f64().ok_or_else(|| bad("activity"))?),
                };
                let base = f("base")?;
                let item = HeapItem {
                    id: s("id")?.to_string(),
                    base,
                    activity,
                    key: item_key(base, activity),
                    attrs: obj("attrs")?,
                    version: rec
                        .get("version")
                        .and_then(Value::as_u64)
                        .ok_or_else(|| bad("version"))?,
                };
                let heap = s("heap")?;
                self.heaps
                    .get_mut(heap)
                    .ok_or_else(|| Error::Snapshot(format!("item of unknown heap {heap:?}")))?
                    .put(item);
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

impl HeapBackend for MemoryHeaps {
    fn create_heap(&mut self, spec: &HeapSpec, now: f64) -> Result<HeapInfo> {
        spec.validate()?;
        if let Some(h) = self.heaps.get(&spec.name) {
            let info = h.info(&spec.name);
            spec.check_matches(&info)?;
            return Ok(info);
        }
        let landmark = match spec.landmark {
            Some(l) => l,
            None => {
                check_finite("now", now)?;
                now
            }
        };
        let h = Heap {
            half_life: spec.half_life,
            landmark,
            meta: spec.meta.clone().unwrap_or_default(),
            ..Heap::default()
        };
        let info = h.info(&spec.name);
        self.heaps.insert(spec.name.clone(), h);
        Ok(info)
    }

    fn drop_heap(&mut self, name: &str) -> Result<bool> {
        Ok(self.heaps.remove(name).is_some())
    }

    fn heap(&self, name: &str) -> Result<Option<HeapInfo>> {
        Ok(self.heaps.get(name).map(|h| h.info(name)))
    }

    fn heaps(&self) -> Result<Vec<HeapInfo>> {
        Ok(self.heaps.iter().map(|(n, h)| h.info(n)).collect())
    }

    fn push(&mut self, heap: &str, id: &str, weight: f64, t: f64, attrs: Option<&Attrs>) -> Result<HeapItem> {
        let args = [PushArgs {
            id: id.to_string(),
            weight,
            time: t,
            attrs: attrs.cloned(),
        }];
        Ok(self.push_many(heap, &args)?.remove(0))
    }

    fn set_base(&mut self, heap: &str, id: &str, base: f64) -> Result<HeapItem> {
        let args = [BaseArgs {
            id: id.to_string(),
            base,
        }];
        Ok(self.set_base_many(heap, &args)?.remove(0))
    }

    fn set_attrs(&mut self, heap: &str, id: &str, attrs: &Attrs, merge: bool) -> Result<HeapItem> {
        let args = [AttrsArgs {
            id: id.to_string(),
            attrs: attrs.clone(),
            merge,
        }];
        Ok(self.set_attrs_many(heap, &args)?.remove(0))
    }

    fn remove(&mut self, heap: &str, id: &str) -> Result<bool> {
        Ok(self.get_heap_mut(heap)?.take(id).is_some())
    }

    fn get(&self, heap: &str, id: &str, now: f64) -> Result<Option<Ranked>> {
        check_finite("now", now)?;
        let h = self.get_heap(heap)?;
        Ok(h.items
            .get(id)
            .map(|it| Ranked::of(0, it, h.lam(), h.landmark, now)))
    }

    fn top_k(&self, heap: &str, k: usize, now: f64, filter: Option<&Filter>) -> Result<Vec<Ranked>> {
        check_finite("now", now)?;
        let h = self.get_heap(heap)?;
        let (lam, landmark) = (h.lam(), h.landmark);
        Ok(h.top(k, filter)
            .into_iter()
            .enumerate()
            .map(|(i, it)| Ranked::of(i + 1, it, lam, landmark, now))
            .collect())
    }

    fn pop(&mut self, heap: &str, now: f64, filter: Option<&Filter>) -> Result<Option<Ranked>> {
        let Some(top) = self.top_k(heap, 1, now, filter)?.pop() else {
            return Ok(None);
        };
        self.get_heap_mut(heap)?.take(&top.id);
        Ok(Some(top))
    }

    fn rebase(&mut self, heap: &str, now: f64, half_life: Option<f64>) -> Result<HeapInfo> {
        check_finite("now", now)?;
        if let Some(hl) = half_life {
            check_positive("half_life", hl)?;
        }
        let h = self.get_heap_mut(heap)?;
        let shift = h.lam() * (now - h.landmark);
        if !shift.is_finite() {
            return Err(too_far("now"));
        }
        let mut items: Vec<HeapItem> = h.items.values().cloned().collect();
        for it in &mut items {
            if let Some(a) = it.activity {
                let a2 = a - shift;
                if !a2.is_finite() || !item_key(it.base, Some(a2)).is_some_and(f64::is_finite) {
                    return Err(too_far("now"));
                }
                it.activity = Some(a2);
                it.key = item_key(it.base, it.activity);
            }
        }
        h.landmark = now;
        if let Some(hl) = half_life {
            h.half_life = hl;
        }
        h.index.clear();
        for it in items {
            h.put(it);
        }
        Ok(h.info(heap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn attrs(v: Value) -> Attrs {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn basics() {
        let mut m = MemoryHeaps::new();
        let info = m.create_heap(&HeapSpec::new("q", 3600.0), 1000.0).unwrap();
        assert_eq!((info.landmark, info.size), (1000.0, 0));
        // Idempotent; conflicting specs are rejected.
        assert_eq!(m.create_heap(&HeapSpec::new("q", 3600.0), 5.0).unwrap(), info);
        assert!(matches!(
            m.create_heap(&HeapSpec::new("q", 60.0), 5.0),
            Err(Error::HeapConflict(_))
        ));
        assert!(matches!(
            m.push("nope", "a", 1.0, 0.0, None),
            Err(Error::UnknownHeap(_))
        ));
        let a = m
            .push("q", "a", 2.0, 1000.0, Some(&attrs(json!({"kind": "bug"}))))
            .unwrap();
        assert_eq!((a.version, a.key), (1, Some(2f64.ln())));
        m.push("q", "a", 2.0, 1000.0, None).unwrap();
        let b = m.set_base("q", "b", 4.0).unwrap();
        assert_eq!(b.key, None);
        let top = m.top_k("q", 10, 1000.0 + 3600.0, None).unwrap();
        assert_eq!(top.len(), 1, "unranked items are not listed");
        assert!((top[0].score - 2.0).abs() < 1e-12, "{}", top[0].score);
        assert!((top[0].heat - 2.0).abs() < 1e-12);
        assert_eq!(top[0].attrs, attrs(json!({"kind": "bug"})));
        m.push("q", "b", 1.0, 1000.0, None).unwrap();
        let ids: Vec<String> = m
            .top_k("q", 10, 0.0, None)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, ["a", "b"], "equal keys tie by id");
        let f = Filter::from_json(&json!({"attr": "kind", "op": "eq", "value": "bug"})).unwrap();
        assert_eq!(m.pop("q", 0.0, Some(&f)).unwrap().unwrap().id, "a");
        assert_eq!(m.pop("q", 0.0, Some(&f)).unwrap(), None);
        assert!(m.remove("q", "b").unwrap());
        assert!(!m.remove("q", "b").unwrap());
        assert!(m.drop_heap("q").unwrap() && !m.drop_heap("q").unwrap());
    }

    #[test]
    fn batches_are_atomic() {
        let mut m = MemoryHeaps::new();
        m.create_heap(&HeapSpec::new("q", 60.0).landmark(0.0), 0.0)
            .unwrap();
        let bad = [
            PushArgs {
                id: "a".into(),
                weight: 1.0,
                time: 0.0,
                attrs: None,
            },
            PushArgs {
                id: "b".into(),
                weight: -1.0,
                time: 0.0,
                attrs: None,
            },
        ];
        let e = m.push_many("q", &bad).unwrap_err().to_string();
        assert!(e.starts_with("items[1]: weight"), "{e}");
        assert_eq!(m.heap("q").unwrap().unwrap().size, 0);
        // A push so far away that lambda * (t - L) overflows: rejected, nothing stored.
        m.create_heap(&HeapSpec::new("fast", 1e-6).landmark(0.0), 0.0)
            .unwrap();
        let far = [
            bad[0].clone(),
            PushArgs {
                time: 1e305,
                ..bad[0].clone()
            },
        ];
        assert!(m
            .push_many("fast", &far)
            .unwrap_err()
            .to_string()
            .contains("rebase"));
        assert_eq!(m.heap("fast").unwrap().unwrap().size, 0);
        // Far but finite: the key is fine, the score saturates at infinity.
        let it = m.push("q", "far", 1.0, 1e9, None).unwrap();
        assert!(it.key.unwrap().is_finite());
        assert_eq!(m.get("q", "far", 0.0).unwrap().unwrap().score, f64::INFINITY);
        m.remove("q", "far").unwrap();
        // Repeated ids within a batch accumulate in order.
        let two = [bad[0].clone(), bad[0].clone()];
        let out = m.push_many("q", &two).unwrap();
        assert_eq!((out[0].version, out[1].version), (1, 2));
        assert_eq!(out[1].activity, Some(2f64.ln()));
    }

    #[test]
    fn merge_and_records() {
        let mut m = MemoryHeaps::new();
        m.create_heap(
            &HeapSpec::new("q", 60.0)
                .landmark(0.0)
                .meta(attrs(json!({"o": 1}))),
            0.0,
        )
        .unwrap();
        m.set_attrs("q", "a", &attrs(json!({"x": 1, "y": 2})), false)
            .unwrap();
        let it = m
            .set_attrs("q", "a", &attrs(json!({"y": 3, "z": 4})), true)
            .unwrap();
        assert_eq!(it.attrs, attrs(json!({"x": 1, "y": 3, "z": 4})));
        assert_eq!(it.version, 2);
        m.push("q", "b", 0.3, 17.25, None).unwrap();
        let recs = m.to_records();
        let mut m2 = MemoryHeaps::new();
        for r in &recs {
            let text = serde_json::to_string(r).unwrap();
            let back: Value = serde_json::from_str(&text).unwrap();
            assert!(m2.load_record(back.as_object().unwrap()).unwrap());
        }
        assert_eq!(m2.to_records(), recs);
        assert_eq!(
            m2.top_k("q", 5, 99.0, None).unwrap(),
            m.top_k("q", 5, 99.0, None).unwrap()
        );
    }
}
