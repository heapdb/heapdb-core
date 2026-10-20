//! Python bindings for the HeapDB engine (module `heapdb._core`).
//!
//! The pure-Python package `heapdb` re-exports everything from here, so
//! `import heapdb` keeps the API of the reference implementation.

#![forbid(unsafe_code)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

mod conv;
mod db;
mod event;
mod fdh;
mod kernels;
mod query;
mod results;

use conv::{err, py_float, str_list};

/// Regularized incomplete beta ``I_x(a, b)`` = CDF of Beta(a, b) at ``x``.
#[pyfunction]
fn betainc(a: f64, b: f64, x: f64) -> PyResult<f64> {
    heapdb::betainc(a, b, x).map_err(err)
}

/// Quantile of Beta(a, b) (bisection on ``betainc``).
#[pyfunction]
#[pyo3(signature = (q, a, b, tol = 1e-10))]
fn beta_ppf(q: f64, a: f64, b: f64, tol: f64) -> PyResult<f64> {
    heapdb::stats::beta_ppf_tol(q, a, b, tol).map_err(err)
}

/// Equal-tailed ``level`` credible interval of Beta(a, b).
#[pyfunction]
#[pyo3(signature = (a, b, level = 0.9))]
fn beta_interval(a: f64, b: f64, level: f64) -> PyResult<(f64, f64)> {
    heapdb::beta_interval(a, b, level).map_err(err)
}

/// Spearman rank correlation of two equal-length sequences.
#[pyfunction]
fn spearman(xs: &Bound<'_, PyAny>, ys: &Bound<'_, PyAny>) -> PyResult<f64> {
    let to = |o: &Bound<'_, PyAny>| -> PyResult<Vec<f64>> { o.try_iter()?.map(|x| py_float(&x?)).collect() };
    heapdb::spearman(&to(xs)?, &to(ys)?).map_err(err)
}

/// ``1 - prod(1 - s)``; 0.0 for an empty iterable.
#[pyfunction]
fn noisy_or(strengths: &Bound<'_, PyAny>) -> PyResult<f64> {
    let v: Vec<f64> = strengths
        .try_iter()?
        .map(|x| py_float(&x?))
        .collect::<PyResult<_>>()?;
    Ok(heapdb::noisy_or(v))
}

/// EM responsibilities of each feature for a confirmed noisy-OR effect.
#[pyfunction]
fn responsibilities<'py>(py: Python<'py>, strengths: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let mut keys = Vec::new();
    let mut vals = Vec::new();
    for (k, v) in strengths.iter() {
        keys.push(k);
        vals.push(py_float(&v)?);
    }
    let d = PyDict::new(py);
    for (k, r) in keys.into_iter().zip(heapdb::responsibilities(&vals)) {
        d.set_item(k, r)?;
    }
    Ok(d)
}

/// Decay rate for a given half-life (seconds).
#[pyfunction]
fn half_life_to_lambda(half_life: f64) -> f64 {
    heapdb::store::half_life_to_lambda(half_life)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("QueryError", py.get_type::<conv::QueryError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("DEFAULT_LEAK", heapdb::DEFAULT_LEAK)?;
    m.add("SNAPSHOT_VERSION", heapdb::SNAPSHOT_VERSION)?;
    m.add("DEFAULT_MAX_SETS", heapdb::DEFAULT_MAX_SETS)?;
    m.add("JOINT_PRIOR_STRENGTH", heapdb::links::JOINT_PRIOR_STRENGTH)?;
    m.add("CREDIT_MODES", PyTuple::new(py, ["proportional", "uniform"])?)?;
    m.add("BOUND_SLACK", heapdb::store::BOUND_SLACK)?;
    m.add("HELP", heapdb::repl::HELP)?;

    m.add_class::<event::PyEvent>()?;
    m.add_function(wrap_pyfunction!(event::py_format_time, m)?)?;
    m.add_function(wrap_pyfunction!(event::py_validate_weight, m)?)?;
    m.add_function(wrap_pyfunction!(event::py_as_feature_set, m)?)?;

    m.add_class::<kernels::PyKernel>()?;
    m.add_class::<kernels::PyExponentialDecay>()?;
    m.add_class::<kernels::PyWindowKernel>()?;
    m.add_class::<kernels::PyLogNormalKernel>()?;
    m.add_class::<kernels::PyGammaKernel>()?;
    m.add_function(wrap_pyfunction!(kernels::kernel_from_dict, m)?)?;

    m.add_class::<results::PyBetaCounts>()?;
    m.add_class::<results::PyFeatureContribution>()?;
    m.add_class::<results::PyAppliedRule>()?;
    m.add_class::<results::PyJointCause>()?;
    m.add_class::<results::PyExplanation>()?;
    m.add_class::<results::PyCause>()?;
    m.add_class::<results::PyQueryStats>()?;
    m.add_class::<results::PyQueryResult>()?;
    m.add_class::<results::PyRankedCandidate>()?;
    m.add_class::<results::PyRiskResult>()?;
    m.add_class::<results::PyCalibrationRecord>()?;
    m.add_class::<results::PyCalibrationBin>()?;
    m.add_class::<results::PyCalibrationReport>()?;
    m.add_class::<results::PyFeedbackReport>()?;
    m.add_class::<results::PyEventCredit>()?;
    m.add_class::<results::PyObserveReport>()?;
    m.add_class::<results::PySettleReport>()?;
    m.add_class::<results::PyLinkRow>()?;
    m.add_class::<results::PyLinkReport>()?;

    m.add_class::<db::PyHeapDB>()?;
    m.add_class::<db::PyEffectSpec>()?;
    m.add_class::<db::PyRule>()?;
    m.add_class::<db::PyCandidate>()?;
    m.add_class::<db::PyObservationState>()?;
    m.add_class::<db::PyLinkTable>()?;
    m.add_class::<db::PyEntityLog>()?;
    m.add_class::<db::PySegment>()?;

    m.add_class::<fdh::PyForwardDecayHeap>()?;
    m.add_class::<fdh::PyPopped>()?;

    m.add_class::<query::PyTimeExpr>()?;
    m.add_class::<query::PyTopCausesQuery>()?;
    m.add_class::<query::PyCalibrationQuery>()?;
    m.add_class::<query::PyRiskQuery>()?;
    m.add_class::<query::PyFeedbackStmt>()?;
    m.add_class::<query::PyInsertStmt>()?;
    m.add_class::<query::PyObserveStmt>()?;
    m.add_class::<query::PySettleStmt>()?;
    m.add_class::<query::PyLinksQuery>()?;
    m.add(
        "NOW",
        Py::new(
            py,
            query::PyTimeExpr {
                inner: heapdb::query::TimeExpr::NOW,
            },
        )?,
    )?;
    m.add_function(wrap_pyfunction!(query::parse, m)?)?;
    m.add_function(wrap_pyfunction!(query::execute, m)?)?;
    m.add_function(wrap_pyfunction!(query::format_result, m)?)?;
    m.add_function(wrap_pyfunction!(query::parse_duration, m)?)?;
    m.add_function(wrap_pyfunction!(query::parse_time, m)?)?;
    m.add_function(wrap_pyfunction!(query::handle_line, m)?)?;
    m.add_function(wrap_pyfunction!(query::load_demo, m)?)?;

    m.add_function(wrap_pyfunction!(betainc, m)?)?;
    m.add_function(wrap_pyfunction!(beta_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(beta_interval, m)?)?;
    m.add_function(wrap_pyfunction!(spearman, m)?)?;
    m.add_function(wrap_pyfunction!(noisy_or, m)?)?;
    m.add_function(wrap_pyfunction!(responsibilities, m)?)?;
    m.add_function(wrap_pyfunction!(half_life_to_lambda, m)?)?;
    let _ = str_list;
    Ok(())
}
