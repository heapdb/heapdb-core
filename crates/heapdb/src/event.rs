//! Events, attributes and the borrowed view rule predicates see.

use std::sync::LazyLock;

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::intern::Interner;
use crate::pyfmt::{fmt_g, format_time, repr_f64};

/// Free-form event attributes (JSON values, insertion-ordered).
pub type Attrs = Map<String, Value>;

pub(crate) static EMPTY_ATTRS: LazyLock<Attrs> = LazyLock::new(Map::new);

/// A single append-only event, as returned by queries.
///
/// * `weight` is the **dose / exposure** (> 0): portion size, deploy size,
///   amount. Propensity belongs in the learned links, not here.
/// * `features` are the unit of learning; they are kept sorted and unique.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// Monotonic id assigned by the store (unique per database).
    pub id: u64,
    /// Partition key, e.g. a user, a service, a device.
    pub entity: String,
    /// Event time in seconds (any consistent epoch).
    pub time: f64,
    /// Free-form event type, e.g. `meal` or `deploy`.
    pub event_type: String,
    /// Feature tokens, sorted and unique.
    pub features: Vec<String>,
    /// Dose / exposure magnitude (> 0).
    pub weight: f64,
    /// Optional attributes, used by weight functions and rule predicates.
    pub attrs: Attrs,
}

impl Event {
    /// Compact human-readable form: `#3 meal@7200 [beans,onion] w=1.5`.
    pub fn short(&self) -> String {
        short(
            self.id,
            &self.event_type,
            self.time,
            self.features.iter().map(String::as_str),
            self.weight,
        )
    }

    /// Snapshot / `to_dict` form (without the `kind` key).
    pub fn to_json(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("id".into(), Value::from(self.id));
        m.insert("entity".into(), Value::from(self.entity.clone()));
        m.insert("time".into(), crate::snapshot::num(self.time));
        m.insert("type".into(), Value::from(self.event_type.clone()));
        m.insert(
            "features".into(),
            Value::Array(self.features.iter().cloned().map(Value::from).collect()),
        );
        m.insert("weight".into(), crate::snapshot::num(self.weight));
        if !self.attrs.is_empty() {
            m.insert("attrs".into(), Value::Object(self.attrs.clone()));
        }
        m
    }

    /// Inverse of [`Event::to_json`].
    pub fn from_json(d: &Map<String, Value>) -> Result<Self> {
        use crate::snapshot::{get_f64, get_str, get_u64};
        let mut features: Vec<String> = match d.get("features") {
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Error::Snapshot("event features must be strings".into()))?,
            None | Some(Value::Null) => Vec::new(),
            _ => return Err(Error::Snapshot("event features must be a list".into())),
        };
        features.sort();
        features.dedup();
        let weight = match d.get("weight") {
            Some(v) => crate::snapshot::as_f64(v)
                .ok_or_else(|| Error::Snapshot("event weight must be a number".into()))?,
            None => 1.0,
        };
        Ok(Event {
            id: get_u64(d, "id")?,
            entity: get_str(d, "entity")?.to_string(),
            time: get_f64(d, "time")?,
            event_type: get_str(d, "type")?.to_string(),
            features,
            weight: validate_weight(weight)?,
            attrs: match d.get("attrs") {
                Some(Value::Object(m)) => m.clone(),
                _ => Map::new(),
            },
        })
    }
}

pub(crate) fn short<'a>(
    id: u64,
    event_type: &str,
    time: f64,
    features: impl Iterator<Item = &'a str>,
    weight: f64,
) -> String {
    let feats: Vec<&str> = features.collect();
    format!(
        "#{id} {event_type}@{} [{}] w={}",
        format_time(time),
        feats.join(","),
        fmt_g(weight, 3)
    )
}

/// Weights are doses: finite and strictly positive.
pub fn validate_weight(weight: f64) -> Result<f64> {
    if !weight.is_finite() || weight <= 0.0 {
        return Err(Error::invalid(format!(
            "event weight must be a finite number > 0, got {}",
            repr_f64(weight)
        )));
    }
    Ok(weight)
}

/// Split a comma-separated feature string (`"beans, onion"`), dropping
/// empty items, like `heapdb.events.as_feature_set`.
pub fn split_features(s: &str) -> Vec<String> {
    let mut v: Vec<String> = s
        .split(',')
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .map(str::to_string)
        .collect();
    v.sort();
    v.dedup();
    v
}

/// The features of an [`EventRef`].
#[derive(Clone, Copy)]
pub enum Features<'a> {
    /// Interned ids (sorted by id) resolved through an interner.
    Interned(&'a [u32], &'a Interner),
    /// Plain names (sorted).
    Names(&'a [String]),
}

impl<'a> Features<'a> {
    /// Number of features.
    pub fn len(&self) -> usize {
        match self {
            Features::Interned(ids, _) => ids.len(),
            Features::Names(n) => n.len(),
        }
    }

    /// True if there are no features.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate the feature names (in no particular order).
    pub fn iter(&self) -> impl Iterator<Item = &'a str> + 'a {
        let (ids, names, strs): (&'a [u32], Option<&'a Interner>, &'a [String]) = match *self {
            Features::Interned(ids, i) => (ids, Some(i), &[]),
            Features::Names(n) => (&[], None, n),
        };
        ids.iter()
            .map(move |&id| names.map(|i| i.name(id)).unwrap_or(""))
            .chain(strs.iter().map(String::as_str))
    }

    /// True if `name` is one of the features.
    pub fn contains(&self, name: &str) -> bool {
        match self {
            Features::Interned(ids, i) => i.get(name).is_some_and(|id| ids.contains(&id)),
            Features::Names(n) => n.iter().any(|f| f == name),
        }
    }

    /// Owned, sorted feature names.
    pub fn to_sorted_vec(&self) -> Vec<String> {
        let mut v: Vec<String> = self.iter().map(str::to_string).collect();
        v.sort();
        v
    }
}

/// A borrowed view of an event, handed to rule predicates.
///
/// Transient candidates (see `HeapDB::feature_risk`) have `id == 0` and a
/// NaN time, so predicates that look at time do not match them.
#[derive(Clone, Copy)]
pub struct EventRef<'a> {
    /// Event id (0 for a transient candidate).
    pub id: u64,
    /// Entity name (`""` for a candidate without entity).
    pub entity: &'a str,
    /// Event time (NaN for a transient candidate).
    pub time: f64,
    /// Event type (`""` for an untyped candidate).
    pub event_type: &'a str,
    /// Features.
    pub features: Features<'a>,
    /// Dose.
    pub weight: f64,
    /// Attributes.
    pub attrs: &'a Attrs,
}

impl EventRef<'_> {
    /// An owned copy.
    pub fn to_event(&self) -> Event {
        Event {
            id: self.id,
            entity: self.entity.to_string(),
            time: self.time,
            event_type: self.event_type.to_string(),
            features: self.features.to_sorted_vec(),
            weight: self.weight,
            attrs: self.attrs.clone(),
        }
    }
}

/// Encode a float as an attribute / snapshot JSON value. Non-finite values
/// use internal markers that snapshots write as `NaN` / `Infinity`.
pub fn json_number(x: f64) -> serde_json::Value {
    crate::snapshot::num(x)
}

/// Decode a JSON number (accepting the non-finite markers).
pub fn json_as_f64(v: &serde_json::Value) -> Option<f64> {
    crate::snapshot::as_f64(v)
}
