//! # HeapDB: pop the most likely cause, right now.
//!
//! An append-only event store whose primary operation is not `SELECT` but
//! *"given what just happened, what most likely caused it?"*. Append events;
//! when an effect shows up, ask for the top-k candidate causes, ranked and
//! explained:
//!
//! ```text
//! score(event | effect) = kernel(dt) x weight x link(features, effect) x rules
//! ```
//!
//! where the link combines the features' evidence with noisy-OR
//! (`1 - prod(1 - strength_f)`, the default) or, per effect, logistically
//! (`sigmoid(b + sum_f w_f)`, see [`logistic`]).
//!
//! * [`Kernel`]: how plausible is a cause `dt` seconds before the effect.
//! * weight: the dose / exposure of the event.
//! * links: Beta beliefs per `(feature, effect)` with hierarchical
//!   (global + entity) layers, learned from [`HeapDB::feedback`] or
//!   observationally from [`HeapDB::observe`] / [`HeapDB::settle`].
//! * [`Rule`]s: soft multipliers in `[0, 1]`.
//!
//! [`HeapDB::top_causes`] prunes whole time segments with upper bounds and
//! stops early; the answer equals brute force.
//!
//! ```
//! use heapdb::{EffectOptions, HeapDB, Kernel, NewEvent, TopK};
//!
//! let mut db = HeapDB::new(900.0, 64)?;
//! db.register_effect(
//!     "bloating",
//!     EffectOptions::new(Kernel::lognormal(7200.0, 0.6)?).prior("beans", 3.0, 1.0),
//! )?;
//! db.insert(NewEvent::new("me", 0.0, "meal").features(["beans", "onion"]).weight(1.5))?;
//! let res = db.top_causes("me", "bloating", 7200.0, TopK::new(1).explain(true))?;
//! assert_eq!(res.top().unwrap().event.features, ["beans", "onion"]);
//! println!("{res}");
//! # Ok::<(), heapdb::Error>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod catalog;
pub mod error;
pub mod event;
pub mod forward_decay;
pub mod intern;
pub mod kernels;
pub mod links;
pub mod logistic;
pub mod observe;
pub mod pyfmt;
pub mod query;
pub mod repl;
pub mod results;
pub mod rules;
pub mod segments;
mod snapshot;
pub mod stats;
pub mod store;

pub use catalog::{
    EffectPage, EffectSummary, EntityPage, EntitySummary, EventTypeCatalog, EventTypeSummary, Occurrence,
    OccurrenceKey, OccurrencePage, PriorSummary,
};
pub use error::{BoxError, Error, ErrorKind, Result};
pub use event::{Attrs, Event, EventRef, Features};
pub use forward_decay::{ForwardDecayHeap, Popped};
pub use kernels::{Kernel, KernelShape};
pub use links::{noisy_or, responsibilities, BetaCounts, Combine, Credit, EntityRef, LinkTable};
pub use logistic::{sigmoid, LogisticParams};
pub use observe::{Baseline, LinkReportOptions};
pub use query::{execute, parse, Output, Statement};
pub use results::*;
pub use rules::{Predicate, Rule, WeightFn};
pub use segments::{EntityLog, Segment, DEFAULT_MAX_SETS};
pub use stats::{beta_interval, beta_ppf, betainc, spearman};
pub use store::{
    Candidate, EffectOptions, EffectSpec, ForgetReport, HeapDB, NewEvent, ObservationState, SegmentInfo,
    TopK, DEFAULT_LEAK, SNAPSHOT_VERSION,
};
