//! `heapdb.Heaps`: keyed heaps with `persist_to` = memory, `native:<dir>`
//! or (with the `postgres` cargo feature) `postgres://...`; and the result
//! classes `Ranked`, `HeapItem`, `HeapInfo`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use heapdb::heaps::hql::{self, HeapOutput};
use heapdb::heaps::{HeapBackend, PersistTo};
use heapdb::{Filter, HeapSpec};

use crate::conv::{err, from_json, map_to_dict, to_json};

type Backend = Box<dyn HeapBackend + Send>;

fn attrs_arg(obj: Option<&Bound<'_, PyAny>>, what: &str) -> PyResult<Option<heapdb::event::Attrs>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => match to_json(o)? {
            serde_json::Value::Object(m) => Ok(Some(m)),
            _ => Err(PyValueError::new_err(format!("{what} must be a dict"))),
        },
    }
}

/// A filter given as a dict (the JSON tree of docs/heaps.md) or as a
/// condition string (`"kind IN ('bug') AND NOT blocked = TRUE"`).
fn filter_arg(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Filter>> {
    let Some(o) = obj.filter(|o| !o.is_none()) else {
        return Ok(None);
    };
    if let Ok(s) = o.cast::<PyString>() {
        return hql::parse_condition(s.to_str()?).map(Some).map_err(err);
    }
    Filter::from_json(&to_json(o)?).map(Some).map_err(err)
}

/// Validate a filter (a dict, or a condition string such as
/// ``"kind IN ('bug') AND NOT blocked = TRUE"``) and return its canonical
/// dict form; ``None`` stays ``None``.
#[pyfunction]
#[pyo3(signature = (filter = None))]
pub fn heap_filter<'py>(
    py: Python<'py>,
    filter: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    filter_arg(filter)?
        .map(|f| from_json(py, &f.to_json()))
        .transpose()
}

fn now_or_clock(py: Python<'_>, now: Option<f64>) -> PyResult<f64> {
    match now {
        Some(n) => Ok(n),
        None => py.import("time")?.getattr("time")?.call0()?.extract(),
    }
}

fn repr_attrs(py: Python<'_>, a: &heapdb::event::Attrs) -> PyResult<String> {
    map_to_dict(py, a)?.repr()?.extract()
}

fn opt_f(x: Option<f64>) -> String {
    x.map_or("None".into(), heapdb::pyfmt::repr_f64)
}

/// An item with its score and heat at ``now`` (from ``top_k``, ``pop``, ``get``).
#[pyclass(frozen, module = "heapdb", name = "Ranked")]
pub struct PyRanked {
    pub inner: heapdb::Ranked,
}

#[pymethods]
impl PyRanked {
    /// Build from the JSON form (``to_dict()``, or a heapdb-server response).
    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyAny>) -> PyResult<Self> {
        match to_json(d)? {
            serde_json::Value::Object(m) => Ok(PyRanked {
                inner: heapdb::Ranked::from_json(&m).map_err(err)?,
            }),
            _ => Err(PyValueError::new_err("expected a dict")),
        }
    }
    /// 1-based position (0 from ``get``).
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank
    }
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }
    /// ``ln(base) + activity``; ``None`` for an unranked item (``get`` only).
    #[getter]
    fn key(&self) -> Option<f64> {
        self.inner.key
    }
    /// ``base * heat`` at ``now``.
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }
    /// ``sum w_i * exp(-lambda (now - t_i))``.
    #[getter]
    fn heat(&self) -> f64 {
        self.inner.heat
    }
    #[getter]
    fn base(&self) -> f64 {
        self.inner.base
    }
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_dict(py, &self.inner.attrs)
    }
    #[getter]
    fn version(&self) -> u64 {
        self.inner.version
    }
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        from_json(py, &self.inner.to_json())
    }
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyRanked>().is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let r = &self.inner;
        Ok(format!(
            "Ranked(rank={}, id={}, score={}, heat={}, base={}, key={}, version={}, attrs={})",
            r.rank,
            heapdb::pyfmt::repr_str(&r.id),
            heapdb::pyfmt::repr_f64(r.score),
            heapdb::pyfmt::repr_f64(r.heat),
            heapdb::pyfmt::repr_f64(r.base),
            opt_f(r.key),
            r.version,
            repr_attrs(py, &r.attrs)?
        ))
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// An item as stored (from ``push``, ``set_base``, ``set_attrs``).
#[pyclass(frozen, module = "heapdb", name = "HeapItem")]
pub struct PyHeapItem {
    pub inner: heapdb::HeapItem,
}

#[pymethods]
impl PyHeapItem {
    /// Build from the JSON form (``to_dict()``, or a heapdb-server response).
    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyAny>) -> PyResult<Self> {
        match to_json(d)? {
            serde_json::Value::Object(m) => Ok(PyHeapItem {
                inner: heapdb::HeapItem::from_json(&m).map_err(err)?,
            }),
            _ => Err(PyValueError::new_err("expected a dict")),
        }
    }
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }
    #[getter]
    fn base(&self) -> f64 {
        self.inner.base
    }
    /// ``ln sum w_i * exp(lambda (t_i - L))``; ``None`` until the first push.
    #[getter]
    fn activity(&self) -> Option<f64> {
        self.inner.activity
    }
    #[getter]
    fn key(&self) -> Option<f64> {
        self.inner.key
    }
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_dict(py, &self.inner.attrs)
    }
    #[getter]
    fn version(&self) -> u64 {
        self.inner.version
    }
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        from_json(py, &self.inner.to_json())
    }
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyHeapItem>().is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let r = &self.inner;
        Ok(format!(
            "HeapItem(id={}, base={}, activity={}, key={}, version={}, attrs={})",
            heapdb::pyfmt::repr_str(&r.id),
            heapdb::pyfmt::repr_f64(r.base),
            opt_f(r.activity),
            opt_f(r.key),
            r.version,
            repr_attrs(py, &r.attrs)?
        ))
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// A heap's settings and size.
#[pyclass(frozen, module = "heapdb", name = "HeapInfo")]
pub struct PyHeapInfo {
    pub inner: heapdb::HeapInfo,
}

#[pymethods]
impl PyHeapInfo {
    /// Build from the JSON form (``to_dict()``, or a heapdb-server response).
    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyAny>) -> PyResult<Self> {
        match to_json(d)? {
            serde_json::Value::Object(m) => Ok(PyHeapInfo {
                inner: heapdb::HeapInfo::from_json(&m).map_err(err)?,
            }),
            _ => Err(PyValueError::new_err("expected a dict")),
        }
    }
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn half_life(&self) -> f64 {
        self.inner.half_life
    }
    #[getter]
    fn landmark(&self) -> f64 {
        self.inner.landmark
    }
    #[getter]
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_dict(py, &self.inner.meta)
    }
    #[getter]
    fn size(&self) -> u64 {
        self.inner.size
    }
    /// ``ln 2 / half_life``.
    #[getter]
    fn lam(&self) -> f64 {
        self.inner.lambda()
    }
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        from_json(py, &self.inner.to_json())
    }
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyHeapInfo>().is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let r = &self.inner;
        Ok(format!(
            "HeapInfo(name={}, half_life={}, landmark={}, size={}, meta={})",
            heapdb::pyfmt::repr_str(&r.name),
            heapdb::pyfmt::repr_f64(r.half_life),
            heapdb::pyfmt::repr_f64(r.landmark),
            r.size,
            repr_attrs(py, &r.meta)?
        ))
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

fn info(py: Python<'_>, i: heapdb::HeapInfo) -> PyResult<Py<PyHeapInfo>> {
    Py::new(py, PyHeapInfo { inner: i })
}

fn ranked(py: Python<'_>, r: heapdb::Ranked) -> PyResult<Py<PyRanked>> {
    Py::new(py, PyRanked { inner: r })
}

fn item(py: Python<'_>, r: heapdb::HeapItem) -> PyResult<Py<PyHeapItem>> {
    Py::new(py, PyHeapItem { inner: r })
}

/// Heap statement result -> Python.
pub fn heap_output_to_py(py: Python<'_>, h: HeapOutput) -> PyResult<Py<PyAny>> {
    Ok(match h {
        HeapOutput::Top(t) => {
            let v: Vec<Py<PyRanked>> = t
                .items
                .into_iter()
                .map(|r| ranked(py, r))
                .collect::<PyResult<_>>()?;
            PyList::new(py, v)?.into_any().unbind()
        }
        HeapOutput::Info(i) => info(py, i)?.into_any(),
        HeapOutput::List(v) => {
            let v: Vec<Py<PyHeapInfo>> = v.into_iter().map(|i| info(py, i)).collect::<PyResult<_>>()?;
            PyList::new(py, v)?.into_any().unbind()
        }
        HeapOutput::Message(m) => m.into_pyobject(py)?.into_any().unbind(),
    })
}

fn open(persist_to: &PersistTo) -> PyResult<Backend> {
    match persist_to {
        #[cfg(feature = "postgres")]
        PersistTo::Postgres(url) => heapdb_postgres::open(url)
            .map(|b| Box::new(b) as Backend)
            .map_err(err),
        #[cfg(not(feature = "postgres"))]
        PersistTo::Postgres(_) => Err(PyValueError::new_err(
            "this build of heapdb has no Postgres support (it is enabled in the release wheels; \
             build from source with `maturin build --features postgres`)",
        )),
        other => other.open().map_err(err),
    }
}

/// Keyed heaps (named priority queues ordered by forward decay).
///
/// ``persist_to``: ``None`` / ``"memory"`` (process memory), ``"native:/dir"``
/// (op log + checkpoints in a directory) or ``"postgres://..."`` (the
/// ``heapdb`` schema in Postgres; installed on first use). Times are Unix
/// seconds; ``now``/``t`` default to ``time.time()``.
#[pyclass(module = "heapdb", name = "Heaps")]
pub struct PyHeaps {
    // A Mutex makes the pyclass Sync (needed to release the GIL around
    // backend calls); `&mut self` methods mean it is never contended.
    backend: std::sync::Mutex<Option<Backend>>,
    persist_to: PersistTo,
}

impl PyHeaps {
    /// Run `f` on the backend (the GIL stays held: used by the REPL).
    pub(crate) fn with_backend<T>(
        &mut self,
        f: impl FnOnce(&mut dyn HeapBackend) -> heapdb::Result<T>,
    ) -> PyResult<T> {
        let slot = self.backend.get_mut().unwrap_or_else(|p| p.into_inner());
        let b = slot
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Heaps is closed"))?;
        f(b.as_mut()).map_err(err)
    }
}

/// Run `$body` on the backend with the GIL released (Postgres round trips).
macro_rules! with_backend {
    ($self:ident, $py:ident, |$b:ident| $body:expr) => {{
        let slot = $self.backend.get_mut().unwrap_or_else(|p| p.into_inner());
        let $b = slot
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("Heaps is closed"))?;
        $py.detach(move || $body).map_err(err)
    }};
}

#[pymethods]
impl PyHeaps {
    #[new]
    #[pyo3(signature = (persist_to = None))]
    fn new(py: Python<'_>, persist_to: Option<&str>) -> PyResult<Self> {
        let p = PersistTo::parse(persist_to.unwrap_or("memory")).map_err(err)?;
        let backend = py.detach(|| open(&p))?;
        Ok(PyHeaps {
            backend: std::sync::Mutex::new(Some(backend)),
            persist_to: p,
        })
    }

    /// ``"memory"``, ``"native:<dir>"`` or the Postgres URL (password redacted).
    #[getter]
    fn persist_to(&self) -> String {
        self.persist_to.to_string()
    }

    /// Create a heap (idempotent when the spec matches; ``ValueError`` otherwise).
    #[pyo3(signature = (name, half_life, landmark = None, meta = None))]
    fn create_heap(
        &mut self,
        py: Python<'_>,
        name: String,
        half_life: f64,
        landmark: Option<f64>,
        meta: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyHeapInfo>> {
        let spec = HeapSpec {
            name,
            half_life,
            landmark,
            meta: attrs_arg(meta, "meta")?,
        };
        let now = now_or_clock(py, None)?;
        let r = with_backend!(self, py, |b| b.create_heap(&spec, now))?;
        info(py, r)
    }

    /// Delete a heap and its items; ``False`` if it did not exist.
    fn drop_heap(&mut self, py: Python<'_>, name: String) -> PyResult<bool> {
        with_backend!(self, py, |b| b.drop_heap(&name))
    }

    /// One heap, or ``None``.
    fn heap(&mut self, py: Python<'_>, name: String) -> PyResult<Option<Py<PyHeapInfo>>> {
        with_backend!(self, py, |b| b.heap(&name))?
            .map(|i| info(py, i))
            .transpose()
    }

    /// All heaps, by name.
    fn heaps(&mut self, py: Python<'_>) -> PyResult<Vec<Py<PyHeapInfo>>> {
        with_backend!(self, py, |b| b.heaps())?
            .into_iter()
            .map(|i| info(py, i))
            .collect()
    }

    /// Add ``weight`` of activity at time ``t`` (default now); creates the
    /// item (base 1) if missing; ``attrs`` replaces the attributes when given.
    #[pyo3(signature = (heap, id, weight = 1.0, t = None, attrs = None))]
    fn push(
        &mut self,
        py: Python<'_>,
        heap: String,
        id: String,
        weight: f64,
        t: Option<f64>,
        attrs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyHeapItem>> {
        let t = now_or_clock(py, t)?;
        let a = attrs_arg(attrs, "attrs")?;
        let r = with_backend!(self, py, |b| b.push(&heap, &id, weight, t, a.as_ref()))?;
        item(py, r)
    }

    /// Set the priority weight; creates the item (unranked) if missing.
    fn set_base(&mut self, py: Python<'_>, heap: String, id: String, base: f64) -> PyResult<Py<PyHeapItem>> {
        let r = with_backend!(self, py, |b| b.set_base(&heap, &id, base))?;
        item(py, r)
    }

    /// Replace (``merge=False``) or shallow-merge the attributes.
    #[pyo3(signature = (heap, id, attrs, merge = false))]
    fn set_attrs(
        &mut self,
        py: Python<'_>,
        heap: String,
        id: String,
        attrs: &Bound<'_, PyAny>,
        merge: bool,
    ) -> PyResult<Py<PyHeapItem>> {
        let a = attrs_arg(Some(attrs), "attrs")?.unwrap_or_default();
        let r = with_backend!(self, py, |b| b.set_attrs(&heap, &id, &a, merge))?;
        item(py, r)
    }

    /// Delete an item; ``False`` if it did not exist.
    fn remove(&mut self, py: Python<'_>, heap: String, id: String) -> PyResult<bool> {
        with_backend!(self, py, |b| b.remove(&heap, &id))
    }

    /// One item with its score at ``now``, or ``None``.
    #[pyo3(signature = (heap, id, now = None))]
    fn get(
        &mut self,
        py: Python<'_>,
        heap: String,
        id: String,
        now: Option<f64>,
    ) -> PyResult<Option<Py<PyRanked>>> {
        let now = now_or_clock(py, now)?;
        with_backend!(self, py, |b| b.get(&heap, &id, now))?
            .map(|r| ranked(py, r))
            .transpose()
    }

    /// The ``k`` best ranked items matching ``filter`` (a dict, or a
    /// condition string as in ``PEEK ... WHERE``), best first.
    #[pyo3(signature = (heap, k = 10, now = None, filter = None))]
    fn top_k(
        &mut self,
        py: Python<'_>,
        heap: String,
        k: usize,
        now: Option<f64>,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Py<PyRanked>>> {
        let now = now_or_clock(py, now)?;
        let f = filter_arg(filter)?;
        with_backend!(self, py, |b| b.top_k(&heap, k, now, f.as_ref()))?
            .into_iter()
            .map(|r| ranked(py, r))
            .collect()
    }

    /// Remove and return the best item matching ``filter`` (``None`` if none).
    #[pyo3(signature = (heap, now = None, filter = None))]
    fn pop(
        &mut self,
        py: Python<'_>,
        heap: String,
        now: Option<f64>,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Option<Py<PyRanked>>> {
        let now = now_or_clock(py, now)?;
        let f = filter_arg(filter)?;
        with_backend!(self, py, |b| b.pop(&heap, now, f.as_ref()))?
            .map(|r| ranked(py, r))
            .transpose()
    }

    /// Move the landmark to ``now`` (optionally changing the half-life),
    /// keeping every item's current heat.
    #[pyo3(signature = (heap, now = None, half_life = None))]
    fn rebase(
        &mut self,
        py: Python<'_>,
        heap: String,
        now: Option<f64>,
        half_life: Option<f64>,
    ) -> PyResult<Py<PyHeapInfo>> {
        let now = now_or_clock(py, now)?;
        let r = with_backend!(self, py, |b| b.rebase(&heap, now, half_life))?;
        info(py, r)
    }

    /// Run one heap statement (``CREATE HEAP``, ``PUSH``, ``PEEK``, ``POP``,
    /// ...). ``PEEK``/``POP`` return a list of ``Ranked``, ``CREATE``/``REBASE``
    /// a ``HeapInfo``, others a message string.
    #[pyo3(signature = (query, now = None))]
    fn execute(&mut self, py: Python<'_>, query: &str, now: Option<f64>) -> PyResult<Py<PyAny>> {
        let stmt = hql::parse(query).map_err(err)?;
        let now = now_or_clock(py, now)?;
        let out = with_backend!(self, py, |b| hql::execute(b.as_mut(), &stmt, now))?;
        heap_output_to_py(py, out)
    }

    /// Release the backend (closes the Postgres connection / log file).
    fn close(&mut self) {
        *self.backend.get_mut().unwrap_or_else(|p| p.into_inner()) = None;
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _a: Option<&Bound<'_, PyAny>>,
        _b: Option<&Bound<'_, PyAny>>,
        _c: Option<&Bound<'_, PyAny>>,
    ) {
        self.close();
    }

    fn __repr__(&self) -> String {
        format!(
            "Heaps(persist_to={})",
            heapdb::pyfmt::repr_str(&self.persist_to.to_string())
        )
    }
}

/// Run SQL on a Postgres URL (no results). Private: used by the test suite
/// to create and drop scratch databases without a Python driver.
#[cfg(feature = "postgres")]
#[pyfunction]
pub fn _pg_execute(py: Python<'_>, url: &str, sql: &str) -> PyResult<()> {
    py.detach(|| {
        let mut c = heapdb_postgres::connect(url)?;
        c.batch_execute(sql)
            .map_err(|e| heapdb::Error::Backend(format!("postgres: {e}")))
    })
    .map_err(err)
}
