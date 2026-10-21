//! Error type of the engine.

use std::error::Error as StdError;

/// A boxed error raised by user-supplied code (rule predicates, weight
/// functions). Language bindings use it to carry their native exception
/// through the engine unchanged.
pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

/// Everything that can go wrong in HeapDB.
///
/// The variants map onto the exception types of the original Python API
/// (see [`Error::kind`]), which the Python bindings rely on.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An effect name that was never registered.
    #[error("unknown effect {}; call register_effect first", crate::pyfmt::repr_str(.0))]
    UnknownEffect(String),
    /// An event id that is not (or no longer) stored.
    #[error("unknown event id {0}")]
    UnknownEvent(u64),
    /// A bad argument value (Python: `ValueError`).
    #[error("{0}")]
    Invalid(String),
    /// A bad argument type (Python: `TypeError`).
    #[error("{0}")]
    Type(String),
    /// A query-language statement that could not be parsed or run.
    #[error("{0}")]
    Query(String),
    /// A missing positional argument in a REPL command (Python: `IndexError`).
    #[error("list index out of range")]
    MissingArgument,
    /// A malformed snapshot.
    #[error("{0}")]
    Snapshot(String),
    /// File I/O.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A rule predicate or weight function failed.
    #[error("{0}")]
    Callback(BoxError),
    /// A keyed heap that does not exist (see [`crate::heaps`]).
    #[error("unknown heap {}; call create_heap first", crate::pyfmt::repr_str(.0))]
    UnknownHeap(String),
    /// `create_heap` with a spec that differs from the existing heap's.
    #[error("{0}")]
    HeapConflict(String),
    /// A heap storage backend failed (I/O, a database connection, ...).
    #[error("{0}")]
    Backend(String),
}

/// Coarse classification of an [`Error`], mirroring Python exception types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// `KeyError`
    Key,
    /// `ValueError`
    Value,
    /// `TypeError`
    Type,
    /// `heapdb.QueryError` (a `ValueError`)
    Query,
    /// `IndexError`
    Index,
    /// `OSError`
    Io,
    /// Raised by user code.
    Callback,
}

impl Error {
    /// Shorthand for [`Error::Invalid`].
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }

    /// Shorthand for [`Error::Query`].
    pub fn query(msg: impl Into<String>) -> Self {
        Error::Query(msg.into())
    }

    /// The Python exception type this error corresponds to.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Error::UnknownEffect(_) | Error::UnknownEvent(_) | Error::UnknownHeap(_) => ErrorKind::Key,
            Error::Invalid(_) | Error::Snapshot(_) | Error::HeapConflict(_) => ErrorKind::Value,
            Error::Type(_) => ErrorKind::Type,
            Error::Query(_) => ErrorKind::Query,
            Error::MissingArgument => ErrorKind::Index,
            Error::Io(_) | Error::Backend(_) => ErrorKind::Io,
            Error::Callback(_) => ErrorKind::Callback,
        }
    }
}

/// `Result` with [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;
