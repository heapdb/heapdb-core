//! The query language: AST classes, `parse`, `execute`, `format_result`,
//! and the REPL entry point used by `heapdb.cli`.

use pyo3::exceptions::PyEOFError;
use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyString, PyTuple};

use heapdb::query::{self as q, Output, Statement, TimeExpr};

use crate::conv::{err, from_json, frozenset, str_list, to_json};
use crate::db::PyHeapDB;
use crate::results::*;

fn attr_pairs<'py>(py: Python<'py>, a: &q::AttrPairs) -> PyResult<Bound<'py, PyTuple>> {
    let items: Vec<Bound<'py, PyTuple>> = a
        .iter()
        .map(|(k, v)| PyTuple::new(py, [k.clone().into_pyobject(py)?.into_any(), from_json(py, v)?]))
        .collect::<PyResult<_>>()?;
    PyTuple::new(py, items)
}

fn attr_pairs_from(obj: Option<&Bound<'_, PyAny>>) -> PyResult<q::AttrPairs> {
    let Some(obj) = obj else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        out.push((k, to_json(&v)?));
    }
    Ok(out)
}

fn feature_set(obj: Option<&Bound<'_, PyAny>>) -> PyResult<std::collections::BTreeSet<String>> {
    Ok(match obj {
        None => Default::default(),
        Some(o) => str_list(o)?.into_iter().collect(),
    })
}

fn fs_py<'py>(py: Python<'py>, s: &std::collections::BTreeSet<String>) -> PyResult<Bound<'py, PyFrozenSet>> {
    let v: Vec<String> = s.iter().cloned().collect();
    frozenset(py, &v)
}

fn time_of(t: Option<PyRef<'_, PyTimeExpr>>) -> TimeExpr {
    t.map_or(TimeExpr::NOW, |t| t.inner)
}

fn opt_repr(s: &Option<String>) -> String {
    s.as_deref().map_or("None".into(), heapdb::pyfmt::repr_str)
}

fn opt_f(x: Option<f64>) -> String {
    x.map_or("None".into(), heapdb::pyfmt::repr_f64)
}

/// Either an absolute time or ``now - offset``; resolved at execution.
#[pyclass(frozen, module = "heapdb.query", name = "TimeExpr")]
pub struct PyTimeExpr {
    pub inner: TimeExpr,
}

#[pymethods]
impl PyTimeExpr {
    #[new]
    #[pyo3(signature = (absolute = None, now_offset = 0.0))]
    fn new(absolute: Option<f64>, now_offset: f64) -> Self {
        PyTimeExpr {
            inner: TimeExpr { absolute, now_offset },
        }
    }
    #[getter]
    fn absolute(&self) -> Option<f64> {
        self.inner.absolute
    }
    #[getter]
    fn now_offset(&self) -> f64 {
        self.inner.now_offset
    }
    fn resolve(&self, now: f64) -> f64 {
        self.inner.resolve(now)
    }
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyTimeExpr>().is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        (self.inner.absolute, self.inner.now_offset)
            .into_pyobject(py)?
            .hash()
    }
    fn __repr__(&self) -> String {
        format!(
            "TimeExpr(absolute={}, now_offset={})",
            opt_f(self.inner.absolute),
            heapdb::pyfmt::repr_f64(self.inner.now_offset)
        )
    }
}

fn time_py(py: Python<'_>, t: TimeExpr) -> PyResult<Py<PyTimeExpr>> {
    Py::new(py, PyTimeExpr { inner: t })
}

/// ``TOP <k> CAUSES OF ...``.
#[pyclass(frozen, module = "heapdb.query", name = "TopCausesQuery")]
pub struct PyTopCausesQuery {
    pub inner: q::TopCausesQuery,
}

#[pymethods]
impl PyTopCausesQuery {
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyTopCausesQuery>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    #[new]
    #[pyo3(signature = (k, effect, entity, at = None, within = None, explain = false, entities = None))]
    fn new(
        k: usize,
        effect: String,
        entity: String,
        at: Option<PyRef<'_, PyTimeExpr>>,
        within: Option<f64>,
        explain: bool,
        entities: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(PyTopCausesQuery {
            inner: q::TopCausesQuery {
                k,
                effect,
                entity,
                at: time_of(at),
                within,
                explain,
                entities: entities.map(str_list).transpose()?.unwrap_or_default(),
            },
        })
    }
    #[getter]
    fn k(&self) -> usize {
        self.inner.k
    }
    #[getter]
    fn effect(&self) -> &str {
        &self.inner.effect
    }
    #[getter]
    fn entity(&self) -> &str {
        &self.inner.entity
    }
    #[getter]
    fn at(&self, py: Python<'_>) -> PyResult<Py<PyTimeExpr>> {
        time_py(py, self.inner.at)
    }
    #[getter]
    fn within(&self) -> Option<f64> {
        self.inner.within
    }
    #[getter]
    fn explain(&self) -> bool {
        self.inner.explain
    }
    #[getter]
    fn entities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.entities)
    }
    #[getter]
    fn is_group(&self) -> bool {
        self.inner.is_group()
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let i = &self.inner;
        Ok(format!(
            "TopCausesQuery(k={}, effect={}, entity={}, at={}, within={}, explain={}, entities={})",
            i.k,
            heapdb::pyfmt::repr_str(&i.effect),
            heapdb::pyfmt::repr_str(&i.entity),
            time_py(py, i.at)?.bind(py).repr()?,
            opt_f(i.within),
            if i.explain { "True" } else { "False" },
            self.entities(py)?.repr()?
        ))
    }
}

/// ``CALIBRATION OF <effect> [BINS <n>]``.
#[pyclass(frozen, module = "heapdb.query", name = "CalibrationQuery")]
pub struct PyCalibrationQuery {
    pub inner: q::CalibrationQuery,
}

#[pymethods]
impl PyCalibrationQuery {
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyCalibrationQuery>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    #[new]
    #[pyo3(signature = (effect, bins = 10))]
    fn new(effect: String, bins: usize) -> Self {
        PyCalibrationQuery {
            inner: q::CalibrationQuery { effect, bins },
        }
    }
    #[getter]
    fn effect(&self) -> &str {
        &self.inner.effect
    }
    #[getter]
    fn bins(&self) -> usize {
        self.inner.bins
    }
    fn __repr__(&self) -> String {
        format!(
            "CalibrationQuery(effect={}, bins={})",
            heapdb::pyfmt::repr_str(&self.inner.effect),
            self.inner.bins
        )
    }
}

/// ``RISK OF <effect> FOR <entity> FEATURES ...``.
#[pyclass(frozen, module = "heapdb.query", name = "RiskQuery")]
pub struct PyRiskQuery {
    pub inner: q::RiskQuery,
}

#[pymethods]
impl PyRiskQuery {
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyRiskQuery>().is_ok_and(|o| o.get().inner == self.inner)
    }
    #[new]
    #[pyo3(signature = (effect, entity, features, weight = None, r#type = None, attrs = None, explain = false))]
    fn new(
        effect: String,
        entity: String,
        features: &Bound<'_, PyAny>,
        weight: Option<f64>,
        r#type: Option<String>,
        attrs: Option<&Bound<'_, PyAny>>,
        explain: bool,
    ) -> PyResult<Self> {
        Ok(PyRiskQuery {
            inner: q::RiskQuery {
                effect,
                entity,
                features: feature_set(Some(features))?,
                weight,
                event_type: r#type,
                attrs: attr_pairs_from(attrs)?,
                explain,
            },
        })
    }
    #[getter]
    fn effect(&self) -> &str {
        &self.inner.effect
    }
    #[getter]
    fn entity(&self) -> &str {
        &self.inner.entity
    }
    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        fs_py(py, &self.inner.features)
    }
    #[getter]
    fn weight(&self) -> Option<f64> {
        self.inner.weight
    }
    #[getter]
    fn r#type(&self) -> Option<&str> {
        self.inner.event_type.as_deref()
    }
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        attr_pairs(py, &self.inner.attrs)
    }
    #[getter]
    fn explain(&self) -> bool {
        self.inner.explain
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let i = &self.inner;
        Ok(format!(
            "RiskQuery(effect={}, entity={}, features={}, weight={}, type={}, attrs={}, explain={})",
            heapdb::pyfmt::repr_str(&i.effect),
            heapdb::pyfmt::repr_str(&i.entity),
            self.features(py)?.repr()?,
            opt_f(i.weight),
            opt_repr(&i.event_type),
            self.attrs(py)?.repr()?,
            if i.explain { "True" } else { "False" }
        ))
    }
}

/// ``FEEDBACK <effect> FOR <entity> AT <time> CONFIRMED <id> [REJECTED ...]``.
#[pyclass(frozen, module = "heapdb.query", name = "FeedbackStmt")]
pub struct PyFeedbackStmt {
    pub inner: q::FeedbackStmt,
}

#[pymethods]
impl PyFeedbackStmt {
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyFeedbackStmt>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    #[new]
    #[pyo3(signature = (effect, entity, at, confirmed, rejected = None))]
    fn new(
        effect: String,
        entity: String,
        at: PyRef<'_, PyTimeExpr>,
        confirmed: u64,
        rejected: Option<Vec<u64>>,
    ) -> Self {
        PyFeedbackStmt {
            inner: q::FeedbackStmt {
                effect,
                entity,
                at: at.inner,
                confirmed,
                rejected: rejected.unwrap_or_default(),
            },
        }
    }
    #[getter]
    fn effect(&self) -> &str {
        &self.inner.effect
    }
    #[getter]
    fn entity(&self) -> &str {
        &self.inner.entity
    }
    #[getter]
    fn at(&self, py: Python<'_>) -> PyResult<Py<PyTimeExpr>> {
        time_py(py, self.inner.at)
    }
    #[getter]
    fn confirmed(&self) -> u64 {
        self.inner.confirmed
    }
    #[getter]
    fn rejected<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.rejected)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let i = &self.inner;
        Ok(format!(
            "FeedbackStmt(effect={}, entity={}, at={}, confirmed={}, rejected={})",
            heapdb::pyfmt::repr_str(&i.effect),
            heapdb::pyfmt::repr_str(&i.entity),
            time_py(py, i.at)?.bind(py).repr()?,
            i.confirmed,
            self.rejected(py)?.repr()?
        ))
    }
}

/// ``INSERT <type> FOR <entity> ...``.
#[pyclass(frozen, module = "heapdb.query", name = "InsertStmt")]
pub struct PyInsertStmt {
    pub inner: q::InsertStmt,
}

#[pymethods]
impl PyInsertStmt {
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyInsertStmt>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    #[new]
    #[pyo3(signature = (r#type, entity, at = None, features = None, weight = None, attrs = None))]
    fn new(
        r#type: String,
        entity: String,
        at: Option<PyRef<'_, PyTimeExpr>>,
        features: Option<&Bound<'_, PyAny>>,
        weight: Option<f64>,
        attrs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(PyInsertStmt {
            inner: q::InsertStmt {
                event_type: r#type,
                entity,
                at: time_of(at),
                features: feature_set(features)?,
                weight,
                attrs: attr_pairs_from(attrs)?,
            },
        })
    }
    #[getter]
    fn r#type(&self) -> &str {
        &self.inner.event_type
    }
    #[getter]
    fn entity(&self) -> &str {
        &self.inner.entity
    }
    #[getter]
    fn at(&self, py: Python<'_>) -> PyResult<Py<PyTimeExpr>> {
        time_py(py, self.inner.at)
    }
    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        fs_py(py, &self.inner.features)
    }
    #[getter]
    fn weight(&self) -> Option<f64> {
        self.inner.weight
    }
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        attr_pairs(py, &self.inner.attrs)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let i = &self.inner;
        Ok(format!(
            "InsertStmt(type={}, entity={}, at={}, features={}, weight={}, attrs={})",
            heapdb::pyfmt::repr_str(&i.event_type),
            heapdb::pyfmt::repr_str(&i.entity),
            time_py(py, i.at)?.bind(py).repr()?,
            self.features(py)?.repr()?,
            opt_f(i.weight),
            self.attrs(py)?.repr()?
        ))
    }
}

/// ``OBSERVE <effect> FOR <entity> [AT <time>] [LEAK <x>]``.
#[pyclass(frozen, module = "heapdb.query", name = "ObserveStmt")]
pub struct PyObserveStmt {
    pub inner: q::ObserveStmt,
}

#[pymethods]
impl PyObserveStmt {
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyObserveStmt>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    #[new]
    #[pyo3(signature = (effect, entity, at = None, leak = None))]
    fn new(effect: String, entity: String, at: Option<PyRef<'_, PyTimeExpr>>, leak: Option<f64>) -> Self {
        PyObserveStmt {
            inner: q::ObserveStmt {
                effect,
                entity,
                at: time_of(at),
                leak,
            },
        }
    }
    #[getter]
    fn effect(&self) -> &str {
        &self.inner.effect
    }
    #[getter]
    fn entity(&self) -> &str {
        &self.inner.entity
    }
    #[getter]
    fn at(&self, py: Python<'_>) -> PyResult<Py<PyTimeExpr>> {
        time_py(py, self.inner.at)
    }
    #[getter]
    fn leak(&self) -> Option<f64> {
        self.inner.leak
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let i = &self.inner;
        Ok(format!(
            "ObserveStmt(effect={}, entity={}, at={}, leak={})",
            heapdb::pyfmt::repr_str(&i.effect),
            heapdb::pyfmt::repr_str(&i.entity),
            time_py(py, i.at)?.bind(py).repr()?,
            opt_f(i.leak)
        ))
    }
}

/// ``SETTLE [<effect>] UNTIL <time> [FOR <entity>]``.
#[pyclass(frozen, module = "heapdb.query", name = "SettleStmt")]
pub struct PySettleStmt {
    pub inner: q::SettleStmt,
}

#[pymethods]
impl PySettleStmt {
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PySettleStmt>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    #[new]
    #[pyo3(signature = (until, effect = None, entity = None))]
    fn new(until: PyRef<'_, PyTimeExpr>, effect: Option<String>, entity: Option<String>) -> Self {
        PySettleStmt {
            inner: q::SettleStmt {
                until: until.inner,
                effect,
                entity,
            },
        }
    }
    #[getter]
    fn until(&self, py: Python<'_>) -> PyResult<Py<PyTimeExpr>> {
        time_py(py, self.inner.until)
    }
    #[getter]
    fn effect(&self) -> Option<&str> {
        self.inner.effect.as_deref()
    }
    #[getter]
    fn entity(&self) -> Option<&str> {
        self.inner.entity.as_deref()
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "SettleStmt(until={}, effect={}, entity={})",
            time_py(py, self.inner.until)?.bind(py).repr()?,
            opt_repr(&self.inner.effect),
            opt_repr(&self.inner.entity)
        ))
    }
}

/// ``LINKS OF <effect> [FOR <entity>] [MIN <n>]``.
#[pyclass(frozen, module = "heapdb.query", name = "LinksQuery")]
pub struct PyLinksQuery {
    pub inner: q::LinksQuery,
}

#[pymethods]
impl PyLinksQuery {
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyLinksQuery>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    #[new]
    #[pyo3(signature = (effect, entity = None, min_support = 1.0))]
    fn new(effect: String, entity: Option<String>, min_support: f64) -> Self {
        PyLinksQuery {
            inner: q::LinksQuery {
                effect,
                entity,
                min_support,
            },
        }
    }
    #[getter]
    fn effect(&self) -> &str {
        &self.inner.effect
    }
    #[getter]
    fn entity(&self) -> Option<&str> {
        self.inner.entity.as_deref()
    }
    #[getter]
    fn min_support(&self) -> f64 {
        self.inner.min_support
    }
    fn __repr__(&self) -> String {
        format!(
            "LinksQuery(effect={}, entity={}, min_support={})",
            heapdb::pyfmt::repr_str(&self.inner.effect),
            opt_repr(&self.inner.entity),
            heapdb::pyfmt::repr_f64(self.inner.min_support)
        )
    }
}

fn stmt_to_py(py: Python<'_>, s: Statement) -> PyResult<Py<PyAny>> {
    Ok(match s {
        Statement::Top(inner) => Py::new(py, PyTopCausesQuery { inner })?.into_any(),
        Statement::Feedback(inner) => Py::new(py, PyFeedbackStmt { inner })?.into_any(),
        Statement::Insert(inner) => Py::new(py, PyInsertStmt { inner })?.into_any(),
        Statement::Calibration(inner) => Py::new(py, PyCalibrationQuery { inner })?.into_any(),
        Statement::Risk(inner) => Py::new(py, PyRiskQuery { inner })?.into_any(),
        Statement::Observe(inner) => Py::new(py, PyObserveStmt { inner })?.into_any(),
        Statement::Settle(inner) => Py::new(py, PySettleStmt { inner })?.into_any(),
        Statement::Links(inner) => Py::new(py, PyLinksQuery { inner })?.into_any(),
        // Heap statements have no AST class; they stay text.
        Statement::Heap(_) => {
            return Err(crate::conv::QueryError::new_err(
                "heap statements (CREATE HEAP, PUSH, PEEK, ...) run on heapdb.Heaps: use Heaps.execute(...)",
            ))
        }
    })
}

fn py_to_stmt(o: &Bound<'_, PyAny>) -> PyResult<Statement> {
    if let Ok(s) = o.cast::<PyString>() {
        return q::parse(s.to_str()?).map_err(err);
    }
    macro_rules! try_cast {
        ($t:ty, $v:ident) => {
            if let Ok(x) = o.cast::<$t>() {
                return Ok(Statement::$v(x.get().inner.clone()));
            }
        };
    }
    try_cast!(PyTopCausesQuery, Top);
    try_cast!(PyFeedbackStmt, Feedback);
    try_cast!(PyInsertStmt, Insert);
    try_cast!(PyCalibrationQuery, Calibration);
    try_cast!(PyRiskQuery, Risk);
    try_cast!(PyObserveStmt, Observe);
    try_cast!(PySettleStmt, Settle);
    try_cast!(PyLinksQuery, Links);
    Err(crate::conv::QueryError::new_err(format!(
        "cannot execute {}",
        o.repr()?
    )))
}

fn output_to_py(py: Python<'_>, out: Output) -> PyResult<Py<PyAny>> {
    Ok(match out {
        Output::Query(r) => PyQueryResult::new_py(py, r)?.into_any(),
        Output::Calibration(inner) => Py::new(py, PyCalibrationReport { inner })?.into_any(),
        Output::Risk(inner) => Py::new(py, PyRiskResult { inner })?.into_any(),
        Output::Links(inner) => Py::new(py, PyLinkReport { inner })?.into_any(),
        Output::Observe(inner) => Py::new(py, PyObserveReport { inner })?.into_any(),
        Output::Settle(inner) => Py::new(py, PySettleReport { inner })?.into_any(),
        Output::Message(m) => m.into_pyobject(py)?.into_any().unbind(),
        Output::Heap(h) => crate::heaps::heap_output_to_py(py, h)?,
    })
}

/// Parse one statement. Raises ``QueryError`` on bad input.
#[pyfunction]
pub fn parse(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    stmt_to_py(py, q::parse(text).map_err(err)?)
}

fn now_of(py: Python<'_>, clock: Option<&Bound<'_, PyAny>>) -> PyResult<f64> {
    match clock {
        Some(c) if !c.is_none() => c.call0()?.extract(),
        _ => py.import("time")?.getattr("time")?.call0()?.extract(),
    }
}

/// Run a statement (or a string). ``clock`` (default ``time.time``) resolves
/// ``now``. Queries return result objects; INSERT/FEEDBACK return a message.
#[pyfunction]
#[pyo3(signature = (db, stmt, clock = None))]
pub fn execute(
    py: Python<'_>,
    db: &Bound<'_, PyHeapDB>,
    stmt: &Bound<'_, PyAny>,
    clock: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let s = py_to_stmt(stmt)?;
    let now = now_of(py, clock)?;
    let out = {
        let mut b = db.borrow_mut();
        q::execute(&mut b.db, &s, now).map_err(err)?
    };
    output_to_py(py, out)
}

/// Pretty-print a QueryResult or any of the report types.
#[pyfunction]
pub fn format_result(res: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(r) = res.cast::<PyQueryResult>() {
        return Ok(r.get().inner.to_string());
    }
    res.str()?.extract()
}

/// ``"12h"`` -> 43200.0. A bare number means seconds.
#[pyfunction]
pub fn parse_duration(text: &str) -> PyResult<f64> {
    q::parse_duration(text).map_err(err)
}

/// Parse a ``<time>``: seconds, ``now``, ``now-<duration>`` or ISO-8601.
#[pyfunction]
pub fn parse_time(text: &str) -> PyResult<PyTimeExpr> {
    Ok(PyTimeExpr {
        inner: q::parse_time(text).map_err(err)?,
    })
}

/// Process one REPL line. Returns the (possibly replaced) db and output.
#[pyfunction]
pub fn handle_line(
    py: Python<'_>,
    db: Py<PyHeapDB>,
    line: &str,
    clock: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyHeapDB>, Option<String>)> {
    let mut clock_fn = || -> heapdb::Result<f64> {
        clock
            .call0()
            .and_then(|v| v.extract::<f64>())
            .map_err(|e| heapdb::Error::Callback(Box::new(e)))
    };
    let outcome = {
        let mut b = db.bind(py).borrow_mut();
        heapdb::repl::handle_line(&mut b.db, line, &mut clock_fn).map_err(err)?
    };
    match outcome {
        heapdb::repl::Outcome::Quit => Err(PyEOFError::new_err(())),
        heapdb::repl::Outcome::Output(o) => Ok((db, o)),
        heapdb::repl::Outcome::Replace(new, msg) => Ok((Py::new(py, PyHeapDB { db: *new })?, Some(msg))),
    }
}

/// Preload a small bloating example relative to ``now``.
#[pyfunction]
pub fn load_demo(db: &Bound<'_, PyHeapDB>, now: f64) -> PyResult<()> {
    heapdb::repl::load_demo(&mut db.borrow_mut().db, now).map_err(err)
}
