//! `heapdb.Event`.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFrozenSet, PyTuple, PyType};

use heapdb::event::{validate_weight, Event};
use heapdb::pyfmt::format_time;

use crate::conv::{attrs_from, err, frozenset, map_to_dict, to_json, weight_number};

/// A single append-only event: "entity X had an event of type T at time t,
/// with these features and this weight (dose)". Immutable.
#[pyclass(frozen, module = "heapdb", name = "Event")]
pub struct PyEvent {
    pub inner: Event,
    features: PyOnceLock<Py<PyFrozenSet>>,
    attrs: PyOnceLock<Py<PyAny>>,
}

impl PyEvent {
    pub fn wrap(inner: Event) -> Self {
        PyEvent {
            inner,
            features: PyOnceLock::new(),
            attrs: PyOnceLock::new(),
        }
    }

    pub fn new_py(py: Python<'_>, inner: Event) -> PyResult<Py<PyEvent>> {
        Py::new(py, PyEvent::wrap(inner))
    }

    fn fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let f = self.features(py)?;
        PyTuple::new(
            py,
            [
                self.inner.id.into_pyobject(py)?.into_any(),
                self.inner.entity.clone().into_pyobject(py)?.into_any(),
                self.inner.time.into_pyobject(py)?.into_any(),
                self.inner.event_type.clone().into_pyobject(py)?.into_any(),
                f.into_any(),
                self.inner.weight.into_pyobject(py)?.into_any(),
            ],
        )
    }
}

#[pymethods]
impl PyEvent {
    #[new]
    #[pyo3(signature = (id, entity, time, r#type, features = None, weight = None, attrs = None))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        py: Python<'_>,
        id: u64,
        entity: String,
        time: f64,
        r#type: String,
        features: Option<&Bound<'_, PyAny>>,
        weight: Option<&Bound<'_, PyAny>>,
        attrs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // dataclass semantics: frozenset(features) (a string becomes its chars).
        let fs: Bound<'_, PyFrozenSet> = match features {
            None => PyFrozenSet::empty(py)?,
            Some(f) => py
                .get_type::<PyFrozenSet>()
                .call1((f,))?
                .cast_into::<PyFrozenSet>()?,
        };
        let mut names: Vec<String> = Vec::new();
        for x in fs.iter() {
            names.push(x.extract()?);
        }
        names.sort();
        let weight = match weight {
            None => 1.0,
            Some(w) => weight_number(w)?,
        };
        validate_weight(weight).map_err(err)?;
        let inner = Event {
            id,
            entity,
            time,
            event_type: r#type,
            features: names,
            weight,
            attrs: attrs_from(attrs)?,
        };
        let ev = PyEvent::wrap(inner);
        let _ = ev.features.set(py, fs.unbind());
        Ok(ev)
    }

    #[getter]
    fn id(&self) -> u64 {
        self.inner.id
    }
    #[getter]
    fn entity(&self) -> &str {
        &self.inner.entity
    }
    #[getter]
    fn time(&self) -> f64 {
        self.inner.time
    }
    #[getter]
    fn r#type(&self) -> &str {
        &self.inner.event_type
    }
    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }

    /// Feature tokens (frozenset).
    #[getter]
    pub fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        let f = self
            .features
            .get_or_try_init(py, || frozenset(py, &self.inner.features).map(Bound::unbind))?;
        Ok(f.bind(py).clone())
    }

    /// Read-only attributes (``types.MappingProxyType``).
    #[getter]
    fn attrs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let a = self.attrs.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
            let d = map_to_dict(py, &self.inner.attrs)?;
            let mp = py.import("types")?.getattr("MappingProxyType")?.call1((d,))?;
            Ok(mp.unbind())
        })?;
        Ok(a.bind(py).clone())
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_dict(py, &self.inner.to_json())
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, d: &Bound<'_, PyAny>) -> PyResult<Self> {
        let v = to_json(d)?;
        let serde_json::Value::Object(m) = v else {
            return Err(pyo3::exceptions::PyTypeError::new_err("from_dict expects a dict"));
        };
        Ok(PyEvent::wrap(Event::from_json(&m).map_err(err)?))
    }

    /// Compact human-readable form used by examples and the CLI.
    fn short(&self) -> String {
        self.inner.short()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.cast::<PyEvent>() {
            Ok(o) => self.inner == o.get().inner,
            Err(_) => false,
        }
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.fields(py)?.hash()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Event(id={}, entity={}, time={}, type={}, features={}, weight={}, attrs={})",
            self.inner.id,
            heapdb::pyfmt::repr_str(&self.inner.entity),
            heapdb::pyfmt::repr_f64(self.inner.time),
            heapdb::pyfmt::repr_str(&self.inner.event_type),
            self.features(py)?.repr()?,
            heapdb::pyfmt::repr_f64(self.inner.weight),
            self.attrs(py)?.repr()?
        ))
    }
}

/// `heapdb.events.format_time`.
#[pyfunction(name = "format_time")]
pub fn py_format_time(t: f64) -> String {
    format_time(t)
}

/// `heapdb.events.validate_weight`.
#[pyfunction(name = "validate_weight")]
pub fn py_validate_weight(weight: &Bound<'_, PyAny>) -> PyResult<f64> {
    let w = weight_number(weight)?;
    validate_weight(w).map_err(err)
}

/// `heapdb.events.as_feature_set`.
#[pyfunction(name = "as_feature_set")]
#[pyo3(signature = (features = None))]
pub fn py_as_feature_set<'py>(
    py: Python<'py>,
    features: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyFrozenSet>> {
    frozenset(py, &crate::conv::feature_list(features)?)
}
