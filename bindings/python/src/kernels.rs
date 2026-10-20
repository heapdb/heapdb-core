//! Lag kernels: `Kernel` (base) and its four subclasses.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use heapdb::kernels::{Kernel, KernelShape};

use crate::conv::{err, map_to_dict, to_json};

/// Base class for unimodal, peak-normalised, truncated lag kernels.
#[pyclass(frozen, subclass, module = "heapdb", name = "Kernel")]
pub struct PyKernel {
    pub inner: Kernel,
}

/// Wrap an engine kernel into the matching Python subclass.
pub fn kernel_to_py(py: Python<'_>, k: Kernel) -> PyResult<Py<PyAny>> {
    let shape = *k.shape();
    let base = PyClassInitializer::from(PyKernel { inner: k });
    Ok(match shape {
        KernelShape::Exponential { .. } => Py::new(py, base.add_subclass(PyExponentialDecay))?.into_any(),
        KernelShape::Window { .. } => Py::new(py, base.add_subclass(PyWindowKernel))?.into_any(),
        KernelShape::LogNormal { .. } => Py::new(py, base.add_subclass(PyLogNormalKernel))?.into_any(),
        KernelShape::Gamma { .. } => Py::new(py, base.add_subclass(PyGammaKernel))?.into_any(),
    })
}

#[pymethods]
impl PyKernel {
    #[classattr]
    fn name() -> &'static str {
        "kernel"
    }
    /// Maximum lookback in seconds.
    #[getter]
    fn support(&self) -> f64 {
        self.inner.support()
    }
    /// Minimum lag in seconds.
    #[getter]
    fn min_lag(&self) -> f64 {
        self.inner.min_lag()
    }
    /// Lag at which the kernel peaks.
    #[getter]
    fn mode(&self) -> f64 {
        self.inner.mode()
    }
    /// Peak value (always 1.0).
    #[getter]
    fn max_value(&self) -> f64 {
        1.0
    }
    /// Kernel value for a cause `dt` seconds before the effect.
    fn value(&self, dt: f64) -> f64 {
        self.inner.value(dt)
    }
    /// Exact maximum of `value` over `[dt_lo, dt_hi]`.
    fn max_over(&self, dt_lo: f64, dt_hi: f64) -> f64 {
        self.inner.max_over(dt_lo, dt_hi)
    }
    /// Fraction of the kernel's mass at lags `<= dt`.
    fn mass_fraction(&self, dt: f64) -> f64 {
        self.inner.mass_fraction(dt)
    }
    /// Un-normalised shape.
    fn _raw(&self, dt: f64) -> f64 {
        self.inner.raw(dt)
    }
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        map_to_dict(py, &self.inner.to_json())
    }
    fn __repr__(&self) -> String {
        self.inner.repr()
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyKernel>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __hash__(&self) -> isize {
        let mut h: u64 = 1469598103934665603;
        for (_, v) in self.inner.params() {
            h = (h ^ v.to_bits()).wrapping_mul(1099511628211);
        }
        h as isize
    }
}

macro_rules! param {
    ($slf:ident, $pat:pat => $v:expr) => {
        match *$slf.as_super().inner.shape() {
            $pat => $v,
            _ => unreachable!("kernel subclass/shape mismatch"),
        }
    };
}

/// `value(dt) = 0.5 ** (dt / half_life)`, truncated at `horizon_half_lives`.
#[pyclass(frozen, extends = PyKernel, module = "heapdb", name = "ExponentialDecay")]
pub struct PyExponentialDecay;

#[pymethods]
impl PyExponentialDecay {
    #[new]
    #[pyo3(signature = (half_life, horizon_half_lives = 10.0))]
    fn new(half_life: f64, horizon_half_lives: f64) -> PyResult<PyClassInitializer<Self>> {
        let k = Kernel::exponential_with_horizon(half_life, horizon_half_lives).map_err(err)?;
        Ok(PyClassInitializer::from(PyKernel { inner: k }).add_subclass(PyExponentialDecay))
    }
    #[classattr]
    fn name() -> &'static str {
        "exp"
    }
    #[getter]
    fn half_life(slf: PyRef<'_, Self>) -> f64 {
        param!(slf, KernelShape::Exponential { half_life, .. } => half_life)
    }
    #[getter]
    fn horizon_half_lives(slf: PyRef<'_, Self>) -> f64 {
        param!(slf, KernelShape::Exponential { horizon_half_lives, .. } => horizon_half_lives)
    }
}

/// Flat kernel: 1.0 for `min_lag <= dt <= max_lag`, else 0.
#[pyclass(frozen, extends = PyKernel, module = "heapdb", name = "WindowKernel")]
pub struct PyWindowKernel;

#[pymethods]
impl PyWindowKernel {
    #[new]
    #[pyo3(signature = (min_lag = 0.0, max_lag = 3600.0))]
    fn new(min_lag: f64, max_lag: f64) -> PyResult<PyClassInitializer<Self>> {
        let k = Kernel::window(min_lag, max_lag).map_err(err)?;
        Ok(PyClassInitializer::from(PyKernel { inner: k }).add_subclass(PyWindowKernel))
    }
    #[classattr]
    fn name() -> &'static str {
        "window"
    }
    #[getter]
    fn max_lag(slf: PyRef<'_, Self>) -> f64 {
        param!(slf, KernelShape::Window { max_lag, .. } => max_lag)
    }
    #[getter]
    fn _min_lag(slf: PyRef<'_, Self>) -> f64 {
        param!(slf, KernelShape::Window { min_lag, .. } => min_lag)
    }
}

/// Log-normal shaped delay (peak-normalised), truncated at
/// `median * exp(tail_sigmas * sigma)`.
#[pyclass(frozen, extends = PyKernel, module = "heapdb", name = "LogNormalKernel")]
pub struct PyLogNormalKernel;

#[pymethods]
impl PyLogNormalKernel {
    #[new]
    #[pyo3(signature = (median, sigma = 0.5, tail_sigmas = 4.0))]
    fn new(median: f64, sigma: f64, tail_sigmas: f64) -> PyResult<PyClassInitializer<Self>> {
        let k = Kernel::lognormal_with_tail(median, sigma, tail_sigmas).map_err(err)?;
        Ok(PyClassInitializer::from(PyKernel { inner: k }).add_subclass(PyLogNormalKernel))
    }
    #[classattr]
    fn name() -> &'static str {
        "lognormal"
    }
    #[getter]
    fn median(slf: PyRef<'_, Self>) -> f64 {
        param!(slf, KernelShape::LogNormal { median, .. } => median)
    }
    #[getter]
    fn sigma(slf: PyRef<'_, Self>) -> f64 {
        param!(slf, KernelShape::LogNormal { sigma, .. } => sigma)
    }
    #[getter]
    fn tail_sigmas(slf: PyRef<'_, Self>) -> f64 {
        param!(slf, KernelShape::LogNormal { tail_sigmas, .. } => tail_sigmas)
    }
    #[getter]
    fn _mu(slf: PyRef<'_, Self>) -> f64 {
        slf.as_super().inner.mu()
    }
}

/// Gamma shaped delay `dt**(shape-1) * exp(-dt/scale)`, peak-normalised.
#[pyclass(frozen, extends = PyKernel, module = "heapdb", name = "GammaKernel")]
pub struct PyGammaKernel;

#[pymethods]
impl PyGammaKernel {
    #[new]
    #[pyo3(signature = (shape, scale, tail_stds = 6.0))]
    fn new(shape: f64, scale: f64, tail_stds: f64) -> PyResult<PyClassInitializer<Self>> {
        let k = Kernel::gamma_with_tail(shape, scale, tail_stds).map_err(err)?;
        Ok(PyClassInitializer::from(PyKernel { inner: k }).add_subclass(PyGammaKernel))
    }
    #[classattr]
    fn name() -> &'static str {
        "gamma"
    }
    #[getter]
    fn shape(slf: PyRef<'_, Self>) -> f64 {
        param!(slf, KernelShape::Gamma { shape, .. } => shape)
    }
    #[getter]
    fn scale(slf: PyRef<'_, Self>) -> f64 {
        param!(slf, KernelShape::Gamma { scale, .. } => scale)
    }
    #[getter]
    fn tail_stds(slf: PyRef<'_, Self>) -> f64 {
        param!(slf, KernelShape::Gamma { tail_stds, .. } => tail_stds)
    }
}

/// Inverse of `Kernel.to_dict` (used by snapshots).
#[pyfunction]
pub fn kernel_from_dict(py: Python<'_>, d: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let serde_json::Value::Object(m) = to_json(d)? else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "kernel_from_dict expects a dict",
        ));
    };
    if !m.contains_key("kind") {
        return Err(pyo3::exceptions::PyKeyError::new_err("kind"));
    }
    kernel_to_py(py, Kernel::from_json(&m).map_err(err)?)
}
