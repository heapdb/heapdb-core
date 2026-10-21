//! Conversions between Python objects and engine values, and error mapping.

use pyo3::exceptions::{PyIndexError, PyKeyError, PyOSError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Value};

use heapdb::event::{json_as_f64, json_number, split_features};
use heapdb::Error;

pyo3::create_exception!(
    heapdb,
    QueryError,
    PyValueError,
    "Raised for anything the query parser does not understand."
);

/// Engine error -> Python exception (callback errors are re-raised as is).
pub fn err(e: Error) -> PyErr {
    let msg = e.to_string();
    match e {
        Error::UnknownEffect(_) | Error::UnknownEvent(_) | Error::UnknownHeap(_) => PyKeyError::new_err(msg),
        Error::Invalid(_) | Error::Snapshot(_) | Error::HeapConflict(_) => PyValueError::new_err(msg),
        Error::Backend(_) => PyOSError::new_err(msg),
        Error::Type(_) => PyTypeError::new_err(msg),
        Error::Query(_) => QueryError::new_err(msg),
        Error::MissingArgument => PyIndexError::new_err(msg),
        Error::Io(io) => PyErr::from(io),
        Error::Callback(b) => match b.downcast::<PyErr>() {
            Ok(pe) => *pe,
            Err(other) => PyRuntimeError::new_err(other.to_string()),
        },
        #[allow(unreachable_patterns)]
        _ => PyOSError::new_err(msg),
    }
}

/// `Result<T, heapdb::Error>` -> `PyResult<T>`.
pub trait IntoPy<T> {
    fn py(self) -> PyResult<T>;
}

impl<T> IntoPy<T> for Result<T, Error> {
    fn py(self) -> PyResult<T> {
        self.map_err(err)
    }
}

/// Python value -> JSON (attrs).
pub fn to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if obj.is_instance_of::<PyBool>() {
        return Ok(Value::Bool(obj.extract()?));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(Value::from(i));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Ok(Value::from(u));
        }
        return Ok(json_number(obj.extract::<f64>()?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(json_number(obj.extract()?));
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(Value::String(obj.extract()?));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        return Ok(Value::Object(dict_to_map(d)?));
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let mut v = Vec::new();
        for x in obj.try_iter()? {
            v.push(to_json(&x?)?);
        }
        return Ok(Value::Array(v));
    }
    // Other mappings (e.g. MappingProxyType).
    if obj.hasattr("keys")? && obj.hasattr("__getitem__")? {
        let d = PyDict::new(obj.py());
        d.call_method1("update", (obj,))?;
        return Ok(Value::Object(dict_to_map(&d)?));
    }
    Err(PyTypeError::new_err(format!(
        "attrs values must be JSON-serialisable (None, bool, int, float, str, list, dict), got {}",
        obj.get_type().name()?
    )))
}

fn key_string(k: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = k.extract::<String>() {
        return Ok(s);
    }
    // json.dumps turns int / float / bool / None keys into strings.
    if k.is_none() {
        return Ok("null".into());
    }
    if k.is_instance_of::<PyBool>() {
        return Ok(if k.extract::<bool>()? { "true" } else { "false" }.into());
    }
    if k.is_instance_of::<PyInt>() || k.is_instance_of::<PyFloat>() {
        return k.str()?.extract();
    }
    Err(PyTypeError::new_err(format!(
        "attrs keys must be str, int, float, bool or None, not {}",
        k.get_type().name()?
    )))
}

/// Python mapping -> JSON object (insertion order kept).
pub fn dict_to_map(d: &Bound<'_, PyDict>) -> PyResult<Map<String, Value>> {
    let mut m = Map::new();
    for (k, v) in d.iter() {
        m.insert(key_string(&k)?, to_json(&v)?);
    }
    Ok(m)
}

/// Optional attrs mapping -> JSON object.
pub fn attrs_from(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Map<String, Value>> {
    match obj {
        None => Ok(Map::new()),
        Some(o) if o.is_none() => Ok(Map::new()),
        Some(o) => match to_json(o)? {
            Value::Object(m) => Ok(m),
            _ => Err(PyTypeError::new_err("attrs must be a mapping")),
        },
    }
}

/// JSON -> Python.
pub fn from_json<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                PyFloat::new(py, n.as_f64().unwrap_or(f64::NAN)).into_any()
            }
        }
        Value::String(s) => match json_as_f64(v) {
            Some(x) => PyFloat::new(py, x).into_any(),
            None => PyString::new(py, s).into_any(),
        },
        Value::Array(a) => {
            let l = PyList::empty(py);
            for x in a {
                l.append(from_json(py, x)?)?;
            }
            l.into_any()
        }
        Value::Object(m) => map_to_dict(py, m)?.into_any(),
    })
}

/// JSON object -> Python dict.
pub fn map_to_dict<'py>(py: Python<'py>, m: &Map<String, Value>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in m {
        d.set_item(k, from_json(py, v)?)?;
    }
    Ok(d)
}

/// `heapdb.events.as_feature_set`: None, a comma-separated string, or an
/// iterable of strings. Returns sorted unique names.
pub fn feature_list(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let Some(obj) = obj else { return Ok(Vec::new()) };
    if obj.is_none() {
        return Ok(Vec::new());
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(split_features(s.to_str()?));
    }
    let mut v = Vec::new();
    for x in obj.try_iter()? {
        let x = x?;
        let s: String = x.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "features must be strings, got {}",
                x.repr().map(|r| r.to_string()).unwrap_or_default()
            ))
        })?;
        v.push(s);
    }
    v.sort();
    v.dedup();
    Ok(v)
}

/// Iterable of strings (not split) -> Vec.
pub fn str_list(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(vec![s.to_str()?.to_owned()]);
    }
    let mut v = Vec::new();
    for x in obj.try_iter()? {
        v.push(x?.extract::<String>()?);
    }
    Ok(v)
}

/// A frozenset of strings.
pub fn frozenset<'py>(py: Python<'py>, items: &[String]) -> PyResult<Bound<'py, PyFrozenSet>> {
    PyFrozenSet::new(py, items)
}

/// A dataclass-like repr: `Name(a=..., b=...)`.
pub fn dc_repr(py: Python<'_>, name: &str, fields: Vec<(&str, Bound<'_, PyAny>)>) -> PyResult<String> {
    let _ = py;
    let mut parts = Vec::with_capacity(fields.len());
    for (k, v) in fields {
        parts.push(format!("{k}={}", v.repr()?));
    }
    Ok(format!("{name}({})", parts.join(", ")))
}

/// Accept int or float (bool included) as a number, like the reference
/// `validate_weight`; anything else -> ValueError.
pub fn weight_number(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>() {
        return obj.extract::<f64>();
    }
    Err(PyValueError::new_err(format!(
        "event weight must be a finite number > 0, got {}",
        obj.repr()?
    )))
}

/// `float(x)`.
pub fn py_float(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    let b = obj.py().get_type::<PyFloat>().call1((obj,))?;
    b.extract()
}
