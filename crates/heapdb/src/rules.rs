//! Plausibility rules and event-type weight functions.
//!
//! A [`Rule`] multiplies the score of matching events by a factor in
//! `[0, 1]` (0 = hard block; values above 1 are rejected because they would
//! break the segment upper bounds). A rule without predicate matches every
//! event of its `types` (or every event): such *type-level* rules tighten
//! the bounds and survive snapshots. A rule with a [`Predicate`] is opaque
//! to the bounds (factor 1) and must be re-registered after loading.

use std::any::Any;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use rustc_hash::FxHashSet;

use crate::error::BoxError;
use crate::event::{Attrs, EventRef};
use crate::segments::TypeId;

/// A rule predicate: does the rule apply to this event?
///
/// Implemented for every `Fn(&EventRef) -> bool + Send + Sync`. Language
/// bindings implement it to call back into their runtime; an `Err` aborts
/// the current operation and is returned as `Error::Callback`.
pub trait Predicate: Send + Sync {
    /// Evaluate the predicate.
    fn matches(&self, event: &EventRef<'_>) -> Result<bool, BoxError>;
    /// For downcasting (bindings recover their native callable).
    fn as_any(&self) -> &dyn Any;
}

impl<F> Predicate for F
where
    F: Fn(&EventRef<'_>) -> bool + Send + Sync + 'static,
{
    fn matches(&self, event: &EventRef<'_>) -> Result<bool, BoxError> {
        Ok(self(event))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Derives an event's weight (dose) from its attributes.
///
/// Implemented for every `Fn(&Attrs) -> f64 + Send + Sync`.
pub trait WeightFn: Send + Sync {
    /// Compute the weight. Must be finite and > 0 (checked by the caller).
    fn weight(&self, attrs: &Attrs) -> Result<f64, BoxError>;
    /// For downcasting.
    fn as_any(&self) -> &dyn Any;
}

impl<F> WeightFn for F
where
    F: Fn(&Attrs) -> f64 + Send + Sync + 'static,
{
    fn weight(&self, attrs: &Attrs) -> Result<f64, BoxError> {
        Ok(self(attrs))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A soft plausibility rule: matching events get `score *= multiplier`.
#[derive(Clone)]
pub struct Rule {
    pub(crate) name: String,
    pub(crate) predicate: Option<Arc<dyn Predicate>>,
    pub(crate) multiplier: f64,
    pub(crate) types: Option<BTreeSet<String>>,
    pub(crate) type_ids: Option<FxHashSet<TypeId>>,
}

impl fmt::Debug for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rule")
            .field("name", &self.name)
            .field("predicate", &self.predicate.is_some())
            .field("multiplier", &self.multiplier)
            .field("types", &self.types)
            .finish()
    }
}

impl Rule {
    /// Rule name (unique per effect).
    pub fn name(&self) -> &str {
        &self.name
    }
    /// The predicate, if any.
    pub fn predicate(&self) -> Option<&Arc<dyn Predicate>> {
        self.predicate.as_ref()
    }
    /// Multiplier in `[0, 1]`.
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }
    /// Event types the rule is restricted to (`None`: all types).
    pub fn types(&self) -> Option<&BTreeSet<String>> {
        self.types.as_ref()
    }
    /// True for a type-level rule (no predicate).
    pub fn is_type_level(&self) -> bool {
        self.predicate.is_none()
    }

    /// Does the rule apply to events of this (interned) type?
    pub(crate) fn applies_to_type_id(&self, ty: Option<TypeId>) -> bool {
        match &self.type_ids {
            None => true,
            Some(set) => ty.is_some_and(|t| set.contains(&t)),
        }
    }

    /// Does the rule apply to events of type `name`?
    pub fn applies_to_type(&self, name: &str) -> bool {
        self.types.as_ref().is_none_or(|t| t.contains(name))
    }

    /// Full match: type and predicate.
    pub(crate) fn matches(&self, ty: Option<TypeId>, ev: &EventRef<'_>) -> Result<bool, BoxError> {
        if !self.applies_to_type_id(ty) {
            return Ok(false);
        }
        match &self.predicate {
            None => Ok(true),
            Some(p) => p.matches(ev),
        }
    }
}
