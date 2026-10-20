//! `heapdb.ForwardDecayHeap` and `heapdb.forward_decay.Popped`.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyType};

use heapdb::ForwardDecayHeap;

use crate::conv::err;

/// An item with its score at the ``now`` passed to pop/peek_top/top_k.
#[pyclass(frozen, module = "heapdb.forward_decay", name = "Popped")]
pub struct PyPopped {
    #[pyo3(get)]
    item: Py<PyAny>,
    #[pyo3(get)]
    score: f64,
    #[pyo3(get)]
    handle: u64,
}

#[pymethods]
impl PyPopped {
    #[new]
    fn new(item: Py<PyAny>, score: f64, handle: u64) -> Self {
        PyPopped { item, score, handle }
    }
    fn __eq__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<bool> {
        match o.cast::<PyPopped>() {
            Ok(o) => {
                let o = o.get();
                Ok(self.score == o.score
                    && self.handle == o.handle
                    && self.item.bind(py).eq(o.item.bind(py))?)
            }
            Err(_) => Ok(false),
        }
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Popped(item={}, score={}, handle={})",
            self.item.bind(py).repr()?,
            heapdb::pyfmt::repr_f64(self.score),
            self.handle
        ))
    }
    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        item: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cls.py()
            .import("types")?
            .getattr("GenericAlias")?
            .call1((cls, item))
    }
}

/// Max-heap of items whose scores decay as ``w * exp(-lam * age)``; the
/// ordering is time-invariant (forward decay).
#[pyclass(module = "heapdb", name = "ForwardDecayHeap")]
pub struct PyForwardDecayHeap {
    inner: ForwardDecayHeap<Py<PyAny>>,
}

#[pymethods]
impl PyForwardDecayHeap {
    #[new]
    fn new(lam: f64) -> PyResult<Self> {
        Ok(PyForwardDecayHeap {
            inner: ForwardDecayHeap::new(lam).map_err(err)?,
        })
    }
    #[classmethod]
    fn from_half_life(_cls: &Bound<'_, PyType>, half_life: f64) -> PyResult<Self> {
        Ok(PyForwardDecayHeap {
            inner: ForwardDecayHeap::from_half_life(half_life).map_err(err)?,
        })
    }
    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        item: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        cls.py()
            .import("types")?
            .getattr("GenericAlias")?
            .call1((cls, item))
    }
    #[getter]
    fn lam(&self) -> f64 {
        self.inner.lam()
    }
    /// Time-invariant priority: ``log(w) + lam * t``.
    fn key(&self, weight: f64, t: f64) -> PyResult<f64> {
        self.inner.key(weight, t).map_err(err)
    }
    fn score_at(&self, key: f64, now: f64) -> f64 {
        self.inner.score_at(key, now)
    }
    /// Insert ``item`` observed at time ``t`` with weight ``w``; returns a handle.
    fn push(&mut self, item: Py<PyAny>, weight: f64, t: f64) -> PyResult<u64> {
        self.inner.push(item, weight, t).map_err(err)
    }
    /// Tombstone an entry. O(1).
    fn remove(&mut self, handle: u64) -> bool {
        self.inner.remove(handle)
    }
    fn peek_top(&mut self, py: Python<'_>, now: f64) -> Option<PyPopped> {
        self.inner.peek_top(now).map(|p| PyPopped {
            item: p.item.clone_ref(py),
            score: p.score,
            handle: p.handle,
        })
    }
    fn pop(&mut self, now: f64) -> Option<PyPopped> {
        self.inner.pop(now).map(|p| PyPopped {
            item: p.item,
            score: p.score,
            handle: p.handle,
        })
    }
    /// The ``k`` best live items at ``now``, best first, without removing.
    fn top_k(&self, py: Python<'_>, k: usize, now: f64) -> Vec<PyPopped> {
        self.inner
            .top_k(k, now)
            .into_iter()
            .map(|p| PyPopped {
                item: p.item.clone_ref(py),
                score: p.score,
                handle: p.handle,
            })
            .collect()
    }
    /// Physically drop all tombstoned entries (O(n)).
    fn compact(&mut self) {
        self.inner.compact();
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<Py<PyAny>> = self.inner.iter().map(|i| i.clone_ref(py)).collect();
        PyList::new(py, items)?.try_iter().map(Bound::into_any)
    }
}
