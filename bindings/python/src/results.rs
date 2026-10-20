//! Result and report classes.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyTuple};

use heapdb::results as r;
use heapdb::BetaCounts;

use crate::conv::{dc_repr, frozenset};
use crate::event::PyEvent;

fn tuple<'py, T: pyo3::PyClass + Into<PyClassInitializer<T>>>(
    py: Python<'py>,
    items: impl IntoIterator<Item = T>,
) -> PyResult<Bound<'py, PyTuple>> {
    let objs: Vec<Py<T>> = items
        .into_iter()
        .map(|x| Py::new(py, x))
        .collect::<PyResult<_>>()?;
    PyTuple::new(py, objs)
}

fn pairs_dict<'py>(py: Python<'py>, items: &[(String, f64)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in items {
        d.set_item(k, v)?;
    }
    Ok(d)
}

// -- BetaCounts -------------------------------------------------------------

/// A Beta(alpha, beta) belief, split into where the evidence came from.
#[pyclass(frozen, module = "heapdb", name = "BetaCounts")]
pub struct PyBetaCounts {
    pub inner: BetaCounts,
}

#[pymethods]
impl PyBetaCounts {
    #[new]
    #[pyo3(signature = (prior_alpha, prior_beta, pos = 0.0, neg = 0.0, global_pos = 0.0, global_neg = 0.0))]
    fn new(prior_alpha: f64, prior_beta: f64, pos: f64, neg: f64, global_pos: f64, global_neg: f64) -> Self {
        PyBetaCounts {
            inner: BetaCounts {
                prior_alpha,
                prior_beta,
                pos,
                neg,
                global_pos,
                global_neg,
            },
        }
    }
    #[getter]
    fn prior_alpha(&self) -> f64 {
        self.inner.prior_alpha
    }
    #[getter]
    fn prior_beta(&self) -> f64 {
        self.inner.prior_beta
    }
    #[getter]
    fn pos(&self) -> f64 {
        self.inner.pos
    }
    #[getter]
    fn neg(&self) -> f64 {
        self.inner.neg
    }
    #[getter]
    fn global_pos(&self) -> f64 {
        self.inner.global_pos
    }
    #[getter]
    fn global_neg(&self) -> f64 {
        self.inner.global_neg
    }
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha()
    }
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta()
    }
    /// Posterior mean.
    #[getter]
    fn strength(&self) -> f64 {
        self.inner.strength()
    }
    /// Own + borrowed evidence (excludes the prior).
    #[getter]
    fn observations(&self) -> f64 {
        self.inner.observations()
    }
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyBetaCounts>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        let b = &self.inner;
        PyTuple::new(
            py,
            [
                b.prior_alpha,
                b.prior_beta,
                b.pos,
                b.neg,
                b.global_pos,
                b.global_neg,
            ],
        )?
        .hash()
    }
    fn __repr__(&self) -> String {
        use heapdb::pyfmt::repr_f64 as f;
        let b = &self.inner;
        format!(
            "BetaCounts(prior_alpha={}, prior_beta={}, pos={}, neg={}, global_pos={}, global_neg={})",
            f(b.prior_alpha),
            f(b.prior_beta),
            f(b.pos),
            f(b.neg),
            f(b.global_pos),
            f(b.global_neg)
        )
    }
}

pub fn beta_py(py: Python<'_>, b: BetaCounts) -> PyResult<Py<PyBetaCounts>> {
    Py::new(py, PyBetaCounts { inner: b })
}

// -- explanation pieces ----------------------------------------------------------

/// How one feature of a candidate event contributes to its score.
#[pyclass(frozen, module = "heapdb", name = "FeatureContribution")]
pub struct PyFeatureContribution {
    pub inner: r::FeatureContribution,
}

#[pymethods]
impl PyFeatureContribution {
    #[new]
    fn new(feature: String, strength: f64, alpha: f64, beta: f64) -> Self {
        PyFeatureContribution {
            inner: r::FeatureContribution {
                feature,
                strength,
                alpha,
                beta,
            },
        }
    }
    #[getter]
    fn feature(&self) -> &str {
        &self.inner.feature
    }
    #[getter]
    fn strength(&self) -> f64 {
        self.inner.strength
    }
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyFeatureContribution>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let i = &self.inner;
        dc_repr(
            py,
            "FeatureContribution",
            vec![
                ("feature", i.feature.clone().into_pyobject(py)?.into_any()),
                ("strength", i.strength.into_pyobject(py)?.into_any()),
                ("alpha", i.alpha.into_pyobject(py)?.into_any()),
                ("beta", i.beta.into_pyobject(py)?.into_any()),
            ],
        )
    }
}

/// A rule that matched an event, with its multiplier.
#[pyclass(frozen, module = "heapdb", name = "AppliedRule")]
pub struct PyAppliedRule {
    pub inner: r::AppliedRule,
}

#[pymethods]
impl PyAppliedRule {
    #[new]
    fn new(name: String, multiplier: f64) -> Self {
        PyAppliedRule {
            inner: r::AppliedRule { name, multiplier },
        }
    }
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn multiplier(&self) -> f64 {
        self.inner.multiplier
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyAppliedRule>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __repr__(&self) -> String {
        format!(
            "AppliedRule(name={}, multiplier={})",
            heapdb::pyfmt::repr_str(&self.inner.name),
            heapdb::pyfmt::repr_f64(self.inner.multiplier)
        )
    }
}

/// A set of features reported as one cause (e.g. ``beans+onion``).
#[pyclass(frozen, module = "heapdb", name = "JointCause")]
pub struct PyJointCause {
    pub inner: r::JointCause,
}

#[pymethods]
impl PyJointCause {
    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.features)
    }
    #[getter]
    fn strength(&self) -> f64 {
        self.inner.strength
    }
    #[getter]
    fn joint(&self, py: Python<'_>) -> PyResult<Option<Py<PyBetaCounts>>> {
        self.inner.joint.map(|b| beta_py(py, b)).transpose()
    }
    #[getter]
    fn inseparable(&self) -> bool {
        self.inner.inseparable
    }
    #[getter]
    fn name(&self) -> String {
        self.inner.name()
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyJointCause>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __repr__(&self) -> String {
        format!("JointCause({})", self.inner)
    }
}

/// Why an event got its score: ``kernel * weight * link * rules``.
#[pyclass(frozen, module = "heapdb", name = "Explanation")]
pub struct PyExplanation {
    pub inner: r::Explanation,
}

pub fn expl_py(py: Python<'_>, e: Option<r::Explanation>) -> PyResult<Option<Py<PyExplanation>>> {
    e.map(|inner| Py::new(py, PyExplanation { inner })).transpose()
}

#[pymethods]
impl PyExplanation {
    #[getter]
    fn dt(&self) -> f64 {
        self.inner.dt
    }
    #[getter]
    fn kernel_value(&self) -> f64 {
        self.inner.kernel_value
    }
    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }
    #[getter]
    fn link_strength(&self) -> f64 {
        self.inner.link_strength
    }
    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        tuple(
            py,
            self.inner
                .features
                .iter()
                .cloned()
                .map(|inner| PyFeatureContribution { inner }),
        )
    }
    #[getter]
    fn warnings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.warnings)
    }
    #[getter]
    fn rules<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        tuple(
            py,
            self.inner
                .rules
                .iter()
                .cloned()
                .map(|inner| PyAppliedRule { inner }),
        )
    }
    #[getter]
    fn joint_causes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        tuple(
            py,
            self.inner
                .joint_causes
                .iter()
                .cloned()
                .map(|inner| PyJointCause { inner }),
        )
    }
    #[getter]
    fn entity(&self) -> &str {
        &self.inner.entity
    }
    #[getter]
    fn prospective(&self) -> bool {
        self.inner.prospective
    }
    #[getter]
    fn rule_multiplier(&self) -> f64 {
        self.inner.rule_multiplier()
    }
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score()
    }
    /// True if some features of this event are an inseparable joint cause.
    #[getter]
    fn inseparable(&self) -> bool {
        self.inner.inseparable()
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyExplanation>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __repr__(&self) -> String {
        format!("Explanation({})", self.inner)
    }
}

// -- causes and query results ------------------------------------------------------

/// One ranked candidate cause.
#[pyclass(frozen, module = "heapdb", name = "Cause")]
pub struct PyCause {
    #[pyo3(get)]
    rank: usize,
    #[pyo3(get)]
    event: Py<PyEvent>,
    #[pyo3(get)]
    score: f64,
    #[pyo3(get)]
    explanation: Option<Py<PyExplanation>>,
}

impl PyCause {
    pub fn from_cause(py: Python<'_>, c: r::Cause) -> PyResult<Self> {
        Ok(PyCause {
            rank: c.rank,
            event: PyEvent::new_py(py, c.event)?,
            score: c.score,
            explanation: expl_py(py, c.explanation)?,
        })
    }
}

#[pymethods]
impl PyCause {
    #[new]
    #[pyo3(signature = (rank, event, score, explanation = None))]
    fn new(rank: usize, event: Py<PyEvent>, score: f64, explanation: Option<Py<PyExplanation>>) -> Self {
        PyCause {
            rank,
            event,
            score,
            explanation,
        }
    }
    #[getter]
    fn entity(&self) -> String {
        self.event.get().inner.entity.clone()
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        dc_repr(
            py,
            "Cause",
            vec![
                ("rank", self.rank.into_pyobject(py)?.into_any()),
                ("event", self.event.bind(py).clone().into_any()),
                ("score", self.score.into_pyobject(py)?.into_any()),
                (
                    "explanation",
                    match &self.explanation {
                        Some(e) => e.bind(py).clone().into_any(),
                        None => py.None().into_bound(py),
                    },
                ),
            ],
        )
    }
}

/// Work counters, so you can see the pruning happen.
#[pyclass(module = "heapdb", name = "QueryStats")]
pub struct PyQueryStats {
    pub inner: r::QueryStats,
}

#[pymethods]
impl PyQueryStats {
    #[new]
    #[pyo3(signature = (segments_in_window = 0, segments_scanned = 0, segments_pruned = 0, events_scanned = 0, union_fallbacks = 0))]
    fn new(
        segments_in_window: usize,
        segments_scanned: usize,
        segments_pruned: usize,
        events_scanned: usize,
        union_fallbacks: usize,
    ) -> Self {
        PyQueryStats {
            inner: r::QueryStats {
                segments_in_window,
                segments_scanned,
                segments_pruned,
                events_scanned,
                union_fallbacks,
            },
        }
    }
    #[getter]
    fn segments_in_window(&self) -> usize {
        self.inner.segments_in_window
    }
    #[getter]
    fn segments_scanned(&self) -> usize {
        self.inner.segments_scanned
    }
    #[getter]
    fn segments_pruned(&self) -> usize {
        self.inner.segments_pruned
    }
    #[getter]
    fn events_scanned(&self) -> usize {
        self.inner.events_scanned
    }
    #[getter]
    fn union_fallbacks(&self) -> usize {
        self.inner.union_fallbacks
    }
    fn add(&mut self, other: PyRef<'_, PyQueryStats>) {
        self.inner.add(&other.inner);
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __repr__(&self) -> String {
        let s = &self.inner;
        format!(
            "QueryStats(segments_in_window={}, segments_scanned={}, segments_pruned={}, events_scanned={}, union_fallbacks={})",
            s.segments_in_window, s.segments_scanned, s.segments_pruned, s.events_scanned, s.union_fallbacks
        )
    }
}

/// Ranked causes (best first) plus execution statistics. Iterable.
#[pyclass(frozen, module = "heapdb", name = "QueryResult")]
pub struct PyQueryResult {
    pub inner: r::QueryResult,
    causes: Py<PyList>,
    stats: Py<PyQueryStats>,
}

impl PyQueryResult {
    pub fn new_py(py: Python<'_>, inner: r::QueryResult) -> PyResult<Py<Self>> {
        let causes = PyList::empty(py);
        for c in inner.causes.clone() {
            causes.append(Py::new(py, PyCause::from_cause(py, c)?)?)?;
        }
        // The engine copy is kept for formatting (format_result).
        let stats = Py::new(py, PyQueryStats { inner: inner.stats })?;
        Py::new(
            py,
            PyQueryResult {
                inner,
                causes: causes.unbind(),
                stats,
            },
        )
    }
}

#[pymethods]
impl PyQueryResult {
    #[getter]
    fn entity(&self) -> &str {
        &self.inner.entity
    }
    #[getter]
    fn effect(&self) -> &str {
        &self.inner.effect
    }
    #[getter]
    fn at(&self) -> f64 {
        self.inner.at
    }
    #[getter]
    fn causes(&self, py: Python<'_>) -> Py<PyList> {
        self.causes.clone_ref(py)
    }
    #[getter]
    fn stats(&self, py: Python<'_>) -> Py<PyQueryStats> {
        self.stats.clone_ref(py)
    }
    #[getter]
    fn entities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.entities)
    }
    /// The single most likely cause ("pop the top"), or None.
    #[getter]
    fn top<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let l = self.causes.bind(py);
        if l.is_empty() {
            Ok(None)
        } else {
            Ok(Some(l.get_item(0)?))
        }
    }
    #[getter]
    fn is_group(&self) -> bool {
        self.inner.is_group()
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.causes.bind(py).try_iter().map(Bound::into_any)
    }
    fn __len__(&self, py: Python<'_>) -> usize {
        self.causes.bind(py).len()
    }
    fn __getitem__<'py>(&self, py: Python<'py>, i: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.causes.bind(py).as_any().get_item(i)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "QueryResult(entity={}, effect={}, at={}, causes={}, stats={}, entities={})",
            heapdb::pyfmt::repr_str(&self.inner.entity),
            heapdb::pyfmt::repr_str(&self.inner.effect),
            heapdb::pyfmt::repr_f64(self.inner.at),
            self.causes.bind(py).repr()?,
            self.stats.bind(py).repr()?,
            self.entities(py)?.repr()?
        ))
    }
}

/// One row of ``HeapDB.rank_candidates`` (best first).
#[pyclass(frozen, module = "heapdb", name = "RankedCandidate")]
pub struct PyRankedCandidate {
    #[pyo3(get)]
    pub rank: usize,
    #[pyo3(get)]
    pub candidate: Py<PyAny>,
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub entity: Option<String>,
    #[pyo3(get)]
    pub explanation: Option<Py<PyExplanation>>,
}

#[pymethods]
impl PyRankedCandidate {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "RankedCandidate(rank={}, candidate={}, score={}, entity={}, explanation={})",
            self.rank,
            self.candidate.bind(py).repr()?,
            heapdb::pyfmt::repr_f64(self.score),
            self.entity
                .as_deref()
                .map_or("None".into(), heapdb::pyfmt::repr_str),
            match &self.explanation {
                Some(e) => e.bind(py).repr()?.to_string(),
                None => "None".into(),
            }
        ))
    }
}

/// Result of a ``RISK OF ...`` statement.
#[pyclass(frozen, module = "heapdb", name = "RiskResult")]
pub struct PyRiskResult {
    pub inner: r::RiskResult,
}

#[pymethods]
impl PyRiskResult {
    #[new]
    #[pyo3(signature = (effect, entity, features, r#type, score, explanation = None))]
    fn new(
        effect: String,
        entity: Option<String>,
        features: &Bound<'_, PyAny>,
        r#type: Option<String>,
        score: f64,
        explanation: Option<PyRef<'_, PyExplanation>>,
    ) -> PyResult<Self> {
        Ok(PyRiskResult {
            inner: r::RiskResult {
                effect,
                entity,
                features: crate::conv::feature_list(Some(features))?,
                event_type: r#type,
                score,
                explanation: explanation.map(|e| e.inner.clone()),
            },
        })
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
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        frozenset(py, &self.inner.features)
    }
    #[getter]
    fn r#type(&self) -> Option<&str> {
        self.inner.event_type.as_deref()
    }
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }
    #[getter]
    fn explanation(&self, py: Python<'_>) -> PyResult<Option<Py<PyExplanation>>> {
        expl_py(py, self.inner.explanation.clone())
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

// -- calibration -------------------------------------------------------------------

/// A judged event recorded for calibration.
#[pyclass(frozen, module = "heapdb", name = "CalibrationRecord")]
pub struct PyCalibrationRecord {
    pub inner: r::CalibrationRecord,
}

#[pymethods]
impl PyCalibrationRecord {
    #[new]
    fn new(entity: String, event_id: u64, at: f64, score: f64, confirmed: bool) -> Self {
        PyCalibrationRecord {
            inner: r::CalibrationRecord {
                entity,
                event_id,
                at,
                score,
                confirmed,
            },
        }
    }
    #[getter]
    fn entity(&self) -> &str {
        &self.inner.entity
    }
    #[getter]
    fn event_id(&self) -> u64 {
        self.inner.event_id
    }
    #[getter]
    fn at(&self) -> f64 {
        self.inner.at
    }
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }
    #[getter]
    fn confirmed(&self) -> bool {
        self.inner.confirmed
    }
    /// Scores are not probabilities; clip to [0, 1].
    #[getter]
    fn predicted(&self) -> f64 {
        self.inner.predicted()
    }
    fn __eq__(&self, o: &Bound<'_, PyAny>) -> bool {
        o.cast::<PyCalibrationRecord>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }
    fn __repr__(&self) -> String {
        let i = &self.inner;
        format!(
            "CalibrationRecord(entity={}, event_id={}, at={}, score={}, confirmed={})",
            heapdb::pyfmt::repr_str(&i.entity),
            i.event_id,
            heapdb::pyfmt::repr_f64(i.at),
            heapdb::pyfmt::repr_f64(i.score),
            if i.confirmed { "True" } else { "False" }
        )
    }
}

/// One bin of a calibration report.
#[pyclass(frozen, module = "heapdb", name = "CalibrationBin")]
pub struct PyCalibrationBin {
    pub inner: r::CalibrationBin,
}

#[pymethods]
impl PyCalibrationBin {
    #[getter]
    fn lo(&self) -> f64 {
        self.inner.lo
    }
    #[getter]
    fn hi(&self) -> f64 {
        self.inner.hi
    }
    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }
    #[getter]
    fn mean_predicted(&self) -> Option<f64> {
        self.inner.mean_predicted
    }
    #[getter]
    fn observed_rate(&self) -> Option<f64> {
        self.inner.observed_rate
    }
    fn __repr__(&self) -> String {
        format!("CalibrationBin({:?})", self.inner)
    }
}

/// Reliability table + Brier score for one effect.
#[pyclass(frozen, module = "heapdb", name = "CalibrationReport")]
pub struct PyCalibrationReport {
    pub inner: r::CalibrationReport,
}

#[pymethods]
impl PyCalibrationReport {
    #[getter]
    fn effect(&self) -> &str {
        &self.inner.effect
    }
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }
    #[getter]
    fn brier(&self) -> Option<f64> {
        self.inner.brier
    }
    #[getter]
    fn bins<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        tuple(
            py,
            self.inner
                .bins
                .iter()
                .cloned()
                .map(|inner| PyCalibrationBin { inner }),
        )
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// What a feedback call did: per-feature credit of the confirmed event.
#[pyclass(frozen, module = "heapdb", name = "FeedbackReport")]
pub struct PyFeedbackReport {
    pub inner: r::FeedbackReport,
}

#[pymethods]
impl PyFeedbackReport {
    #[getter]
    fn confirmed(&self) -> Option<u64> {
        self.inner.confirmed
    }
    #[getter]
    fn rejected<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.rejected)
    }
    #[getter]
    fn credit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs_dict(py, &self.inner.credit)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "FeedbackReport(confirmed={}, rejected={}, credit={})",
            self.inner.confirmed.map_or("None".into(), |c| c.to_string()),
            self.rejected(py)?.repr()?,
            self.credit(py)?.repr()?
        ))
    }
}

// -- observational learning ----------------------------------------------------------

/// Soft credit one candidate event received from ``HeapDB.observe``.
#[pyclass(frozen, module = "heapdb", name = "EventCredit")]
pub struct PyEventCredit {
    pub inner: r::EventCredit,
}

#[pymethods]
impl PyEventCredit {
    #[getter]
    fn event_id(&self) -> u64 {
        self.inner.event_id
    }
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }
    #[getter]
    fn responsibility(&self) -> f64 {
        self.inner.responsibility
    }
    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        pairs_dict(py, &self.inner.features)
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// What ``HeapDB.observe`` did for one effect occurrence.
#[pyclass(frozen, module = "heapdb", name = "ObserveReport")]
pub struct PyObserveReport {
    pub inner: r::ObserveReport,
}

#[pymethods]
impl PyObserveReport {
    #[getter]
    fn entity(&self) -> &str {
        &self.inner.entity
    }
    #[getter]
    fn effect(&self) -> &str {
        &self.inner.effect
    }
    #[getter]
    fn at(&self) -> f64 {
        self.inner.at
    }
    #[getter]
    fn leak(&self) -> f64 {
        self.inner.leak
    }
    #[getter]
    fn leak_responsibility(&self) -> f64 {
        self.inner.leak_responsibility
    }
    #[getter]
    fn credits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        tuple(
            py,
            self.inner
                .credits
                .iter()
                .cloned()
                .map(|inner| PyEventCredit { inner }),
        )
    }
    /// Sum of the events' responsibilities (= 1 - leak responsibility).
    #[getter]
    fn total_credit(&self) -> f64 {
        self.inner.total_credit()
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// What ``HeapDB.settle`` did (summed over effects and entities).
#[pyclass(frozen, module = "heapdb", name = "SettleReport")]
pub struct PySettleReport {
    pub inner: r::SettleReport,
}

#[pymethods]
impl PySettleReport {
    #[getter]
    fn until(&self) -> f64 {
        self.inner.until
    }
    #[getter]
    fn events_settled(&self) -> usize {
        self.inner.events_settled
    }
    #[getter]
    fn events_rejected(&self) -> usize {
        self.inner.events_rejected
    }
    #[getter]
    fn rejected_weight(&self) -> f64 {
        self.inner.rejected_weight
    }
    #[getter]
    fn skipped_credited(&self) -> usize {
        self.inner.skipped_credited
    }
    #[getter]
    fn skipped_observed(&self) -> usize {
        self.inner.skipped_observed
    }
    #[getter]
    fn per_effect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.inner.per_effect {
            d.set_item(k, v)?;
        }
        Ok(d)
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

/// One feature in a ``LinkReport``.
#[pyclass(frozen, module = "heapdb", name = "LinkRow")]
pub struct PyLinkRow {
    pub inner: r::LinkRow,
}

#[pymethods]
impl PyLinkRow {
    #[getter]
    fn feature(&self) -> &str {
        &self.inner.feature
    }
    #[getter]
    fn strength(&self) -> f64 {
        self.inner.strength
    }
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }
    #[getter]
    fn n(&self) -> f64 {
        self.inner.n
    }
    #[getter]
    fn lo(&self) -> f64 {
        self.inner.lo
    }
    #[getter]
    fn hi(&self) -> f64 {
        self.inner.hi
    }
    #[getter]
    fn lift(&self) -> f64 {
        self.inner.lift
    }
    fn __repr__(&self) -> String {
        use heapdb::pyfmt::repr_f64 as f;
        let i = &self.inner;
        format!(
            "LinkRow(feature={}, strength={}, alpha={}, beta={}, n={}, lo={}, hi={}, lift={})",
            heapdb::pyfmt::repr_str(&i.feature),
            f(i.strength),
            f(i.alpha),
            f(i.beta),
            f(i.n),
            f(i.lo),
            f(i.hi),
            f(i.lift)
        )
    }
}

/// Features ranked by learned link strength for one effect (associations
/// under the model, not causal effects).
#[pyclass(frozen, module = "heapdb", name = "LinkReport")]
pub struct PyLinkReport {
    pub inner: r::LinkReport,
}

#[pymethods]
impl PyLinkReport {
    #[getter]
    fn effect(&self) -> &str {
        &self.inner.effect
    }
    #[getter]
    fn entity(&self) -> Option<&str> {
        self.inner.entity.as_deref()
    }
    #[getter]
    fn baseline(&self) -> f64 {
        self.inner.baseline
    }
    #[getter]
    fn level(&self) -> f64 {
        self.inner.level
    }
    #[getter]
    fn rows<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        tuple(
            py,
            self.inner.rows.iter().cloned().map(|inner| PyLinkRow { inner }),
        )
    }
    fn row(&self, feature: &str) -> Option<PyLinkRow> {
        self.inner.row(feature).cloned().map(|inner| PyLinkRow { inner })
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.rows(py)?.try_iter().map(Bound::into_any)
    }
    fn __len__(&self) -> usize {
        self.inner.rows.len()
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}
