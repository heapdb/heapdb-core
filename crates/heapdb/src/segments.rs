//! Time-bucketed segments: the unit of storage *and* of pruning.
//!
//! Each entity's log is cut into fixed-width time buckets. A [`Segment`]
//! holds the events of one bucket plus summaries that let the query engine
//! bound *every* event in the segment without reading any of them:
//!
//! * `min_time` / `max_time`: the actual event-time range;
//! * `max_weight`: the largest event weight;
//! * `feature_counts`: the union of features (with multiplicities), for
//!   the (loose) union noisy-OR bound;
//! * `sets`: the distinct `(event type, feature set)` keys with the max
//!   weight of each, for the tight per-set bound. Above `max_sets` keys it
//!   becomes `None` and the engine falls back to the union bound;
//! * `types`: the event types present.
//!
//! Summaries only grow (the log is append-only) and contain no link
//! strengths, so they stay valid as links are learned.

use std::collections::BTreeMap;
use std::ops::Bound;

use rustc_hash::FxHashMap;

use crate::event::Attrs;
use crate::links::FeatureId;
use crate::pyfmt::floordiv;

/// Interned event type id.
pub type TypeId = u32;

/// Default cap on distinct `(type, feature set)` keys tracked per segment.
pub const DEFAULT_MAX_SETS: usize = 64;

/// An event as stored in an entity log (features live in the log's pool).
#[derive(Clone, Debug)]
pub(crate) struct StoredEvent {
    pub id: u64,
    pub time: f64,
    pub weight: f64,
    pub ty: TypeId,
    pub feat_off: u32,
    pub feat_len: u32,
    pub attrs: Option<Box<Attrs>>,
}

/// One `(type, feature set)` key of a segment with its max weight.
#[derive(Clone, Debug)]
pub(crate) struct SetKey {
    pub ty: TypeId,
    pub off: u32,
    pub len: u32,
    pub max_weight: f64,
}

/// All events of one entity whose time falls in `[start, start + width)`.
#[derive(Clone, Debug)]
pub struct Segment {
    pub(crate) bucket: i64,
    pub(crate) start: f64,
    pub(crate) width: f64,
    pub(crate) ords: Vec<u32>,
    pub(crate) min_time: f64,
    pub(crate) max_time: f64,
    pub(crate) max_weight: f64,
    /// Sorted by feature id.
    pub(crate) feature_counts: Vec<(FeatureId, u32)>,
    pub(crate) types: Vec<TypeId>,
    pub(crate) sets: Option<Vec<SetKey>>,
}

impl Segment {
    fn new(bucket: i64, width: f64) -> Self {
        Segment {
            bucket,
            start: bucket as f64 * width,
            width,
            ords: Vec::new(),
            min_time: f64::INFINITY,
            max_time: f64::NEG_INFINITY,
            max_weight: 0.0,
            feature_counts: Vec::new(),
            types: Vec::new(),
            sets: Some(Vec::new()),
        }
    }

    /// Bucket number (`floor(start / width)`).
    pub fn bucket(&self) -> i64 {
        self.bucket
    }
    /// Bucket start time.
    pub fn start(&self) -> f64 {
        self.start
    }
    /// Bucket width.
    pub fn width(&self) -> f64 {
        self.width
    }
    /// Bucket end time.
    pub fn end(&self) -> f64 {
        self.start + self.width
    }
    /// Number of events.
    pub fn len(&self) -> usize {
        self.ords.len()
    }
    /// True if the segment holds no events.
    pub fn is_empty(&self) -> bool {
        self.ords.is_empty()
    }
    /// Earliest event time.
    pub fn min_time(&self) -> f64 {
        self.min_time
    }
    /// Latest event time.
    pub fn max_time(&self) -> f64 {
        self.max_time
    }
    /// Largest event weight.
    pub fn max_weight(&self) -> f64 {
        self.max_weight
    }
    /// Number of distinct features.
    pub fn num_features(&self) -> usize {
        self.feature_counts.len()
    }
    /// Number of tracked `(type, feature set)` keys (`None` on overflow).
    pub fn num_sets(&self) -> Option<usize> {
        self.sets.as_ref().map(Vec::len)
    }
}

/// The segmented log of a single entity.
#[derive(Clone, Debug)]
pub struct EntityLog {
    pub(crate) bucket_seconds: f64,
    pub(crate) max_sets: usize,
    pub(crate) events: Vec<StoredEvent>,
    pub(crate) segments: BTreeMap<i64, Segment>,
    pub(crate) pool: Vec<FeatureId>,
    /// Entity-wide posting lists: feature -> event ordinals, in insertion order.
    pub(crate) postings: FxHashMap<FeatureId, Vec<u32>>,
}

/// Bucket number of time `t` (Python `int(t // width)`).
pub(crate) fn bucket_of(t: f64, width: f64) -> i64 {
    if t == f64::INFINITY {
        return i64::MAX;
    }
    if t == f64::NEG_INFINITY {
        return i64::MIN;
    }
    let b = floordiv(t, width);
    if b.is_nan() {
        0
    } else {
        b as i64 // saturating
    }
}

impl EntityLog {
    pub(crate) fn new(bucket_seconds: f64, max_sets: usize) -> Self {
        EntityLog {
            bucket_seconds,
            max_sets,
            events: Vec::new(),
            segments: BTreeMap::new(),
            pool: Vec::new(),
            postings: FxHashMap::default(),
        }
    }

    /// Append an event; `feats` sorted and unique. Returns its ordinal.
    pub(crate) fn add(
        &mut self,
        id: u64,
        time: f64,
        ty: TypeId,
        feats: &[FeatureId],
        weight: f64,
        attrs: Option<Box<Attrs>>,
    ) -> u32 {
        let ord = u32::try_from(self.events.len()).expect("more than 2^32 events for one entity");
        let off = u32::try_from(self.pool.len()).expect("feature pool overflow");
        self.pool.extend_from_slice(feats);
        let len = feats.len() as u32;
        self.events.push(StoredEvent {
            id,
            time,
            weight,
            ty,
            feat_off: off,
            feat_len: len,
            attrs,
        });
        for &f in feats {
            self.postings.entry(f).or_default().push(ord);
        }
        let b = bucket_of(time, self.bucket_seconds);
        let width = self.bucket_seconds;
        let max_sets = self.max_sets;
        let pool = &self.pool;
        let seg = self.segments.entry(b).or_insert_with(|| Segment::new(b, width));
        seg.ords.push(ord);
        seg.min_time = seg.min_time.min(time);
        seg.max_time = seg.max_time.max(time);
        seg.max_weight = seg.max_weight.max(weight);
        for &f in feats {
            match seg.feature_counts.binary_search_by_key(&f, |&(g, _)| g) {
                Ok(i) => seg.feature_counts[i].1 += 1,
                Err(i) => seg.feature_counts.insert(i, (f, 1)),
            }
        }
        if !seg.types.contains(&ty) {
            seg.types.push(ty);
        }
        if let Some(sets) = &mut seg.sets {
            let found = sets
                .iter()
                .position(|k| k.ty == ty && &pool[k.off as usize..(k.off + k.len) as usize] == feats);
            match found {
                Some(i) => {
                    if weight > sets[i].max_weight {
                        sets[i].max_weight = weight;
                    }
                }
                None if sets.len() >= max_sets => seg.sets = None,
                None => sets.push(SetKey {
                    ty,
                    off,
                    len,
                    max_weight: weight,
                }),
            }
        }
        ord
    }

    /// Features of a stored event (sorted ids).
    pub(crate) fn feats(&self, ev: &StoredEvent) -> &[FeatureId] {
        &self.pool[ev.feat_off as usize..(ev.feat_off + ev.feat_len) as usize]
    }

    /// Features of a segment key.
    pub(crate) fn key_feats(&self, k: &SetKey) -> &[FeatureId] {
        &self.pool[k.off as usize..(k.off + k.len) as usize]
    }

    /// Segments whose bucket overlaps `[t_lo, t_hi]`, in time order.
    pub fn segments_between(&self, t_lo: f64, t_hi: f64) -> impl Iterator<Item = &Segment> + '_ {
        let range = if t_lo > t_hi || t_lo.is_nan() || t_hi.is_nan() {
            None
        } else {
            Some((
                bucket_of(t_lo, self.bucket_seconds),
                bucket_of(t_hi, self.bucket_seconds),
            ))
        };
        let bounds = match range {
            Some((lo, hi)) if lo <= hi => (Bound::Included(lo), Bound::Included(hi)),
            _ => (Bound::Included(0), Bound::Excluded(0)),
        };
        self.segments.range(bounds).map(|(_, s)| s)
    }

    /// All segments in time order.
    pub fn segments(&self) -> impl Iterator<Item = &Segment> + '_ {
        self.segments.values()
    }

    /// The segment of a bucket number.
    pub fn segment(&self, bucket: i64) -> Option<&Segment> {
        self.segments.get(&bucket)
    }

    /// Number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Number of segments.
    pub fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Width of the time buckets.
    pub fn bucket_seconds(&self) -> f64 {
        self.bucket_seconds
    }

    /// Cap on tracked keys per segment.
    pub fn max_sets(&self) -> usize {
        self.max_sets
    }

    /// Ordinals of the entity's events in segment order (time buckets,
    /// then insertion order), as the reference implementation iterates.
    pub(crate) fn ords_in_order(&self) -> impl Iterator<Item = u32> + '_ {
        self.segments.values().flat_map(|s| s.ords.iter().copied())
    }

    /// Posting list of a feature (event ordinals).
    pub(crate) fn posting(&self, f: Option<FeatureId>) -> &[u32] {
        f.and_then(|f| self.postings.get(&f))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}
