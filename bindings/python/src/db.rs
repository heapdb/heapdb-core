//! `heapdb.HeapDB` and its live views (EffectSpec, LinkTable, EntityLog,
//! Segment), rules, candidates and Python callbacks.

use std::any::Any;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple, PyType};

use heapdb::event::{Attrs, EventRef};
use heapdb::intern::Vocab;
use heapdb::links::{Credit, EntityRef, LinkTable};
use heapdb::observe::{Baseline, LinkReportOptions};
use heapdb::{
    BoxError, Candidate, EffectOptions, HeapDB, NewEvent, ObservationState, Predicate, TopK, WeightFn,
};

use crate::conv::{
    attrs_from, err, feature_list, frozenset, map_to_dict, py_float, str_list, to_json, weight_number, IntoPy,
};
use crate::event::PyEvent;
use crate::kernels::{kernel_to_py, PyKernel};
use crate::results::*;

// ---------------------------------------------------------------------------
// Python callbacks
// ---------------------------------------------------------------------------

/// A Python callable used as a rule predicate (`predicate(event) -> bool`).
pub struct PyPredicate(pub Py<PyAny>);

impl Predicate for PyPredicate {
    fn matches(&self, ev: &EventRef<'_>) -> Result<bool, BoxError> {
        Python::attach(|py| -> PyResult<bool> {
            let e = Py::new(py, PyEvent::wrap(ev.to_event()))?;
            self.0.call1(py, (e,))?.bind(py).is_truthy()
        })
        .map_err(|e| Box::new(e) as BoxError)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A Python callable deriving an event's weight from its attrs.
pub struct PyWeightFn(pub Py<PyAny>);

impl WeightFn for PyWeightFn {
    fn weight(&self, attrs: &Attrs) -> Result<f64, BoxError> {
        Python::attach(|py| -> PyResult<f64> {
            let d = map_to_dict(py, attrs)?;
            let w = self.0.call1(py, (d,))?;
            weight_number(w.bind(py))
        })
        .map_err(|e| Box::new(e) as BoxError)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn predicate_object(py: Python<'_>, r: &heapdb::Rule) -> Py<PyAny> {
    match r.predicate() {
        None => py.None(),
        Some(p) => match p.as_any().downcast_ref::<PyPredicate>() {
            Some(pp) => pp.0.clone_ref(py),
            None => py.None(),
        },
    }
}

fn credit_arg(c: &str) -> PyResult<Credit> {
    Credit::parse(c).map_err(err)
}

fn opt_str_set(py: Python<'_>, s: Option<&BTreeSet<String>>) -> PyResult<Py<PyAny>> {
    Ok(match s {
        None => py.None(),
        Some(s) => {
            let v: Vec<String> = s.iter().cloned().collect();
            frozenset(py, &v)?.into_any().unbind()
        }
    })
}

// ---------------------------------------------------------------------------
// Rule
// ---------------------------------------------------------------------------

/// A soft plausibility rule: matching events get ``score *= multiplier``.
#[pyclass(frozen, module = "heapdb", name = "Rule")]
pub struct PyRule {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    predicate: Py<PyAny>,
    #[pyo3(get)]
    multiplier: f64,
    types: Option<BTreeSet<String>>,
}

impl PyRule {
    fn from_rule(py: Python<'_>, r: &heapdb::Rule) -> Self {
        PyRule {
            name: r.name().to_string(),
            predicate: predicate_object(py, r),
            multiplier: r.multiplier(),
            types: r.types().cloned(),
        }
    }
}

#[pymethods]
impl PyRule {
    #[new]
    #[pyo3(signature = (name, predicate, multiplier, types = None))]
    fn new(
        name: String,
        predicate: Py<PyAny>,
        multiplier: f64,
        types: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(PyRule {
            name,
            predicate,
            multiplier,
            types: types.map(str_list).transpose()?.map(|v| v.into_iter().collect()),
        })
    }
    #[getter]
    fn types(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        opt_str_set(py, self.types.as_ref())
    }
    fn applies_to_type(&self, type_: &str) -> bool {
        self.types.as_ref().is_none_or(|t| t.contains(type_))
    }
    /// Does the rule apply to this event (type and predicate)?
    fn matches(&self, py: Python<'_>, event: &Bound<'_, PyAny>) -> PyResult<bool> {
        let t: String = event.getattr("type")?.extract()?;
        if !self.applies_to_type(&t) {
            return Ok(false);
        }
        if self.predicate.is_none(py) {
            return Ok(true);
        }
        self.predicate.call1(py, (event,))?.bind(py).is_truthy()
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Rule(name={}, predicate={}, multiplier={}, types={})",
            heapdb::pyfmt::repr_str(&self.name),
            self.predicate.bind(py).repr()?,
            heapdb::pyfmt::repr_f64(self.multiplier),
            self.types(py)?.bind(py).repr()?
        ))
    }
}

// ---------------------------------------------------------------------------
// Candidate
// ---------------------------------------------------------------------------

/// Something that has *not* happened yet and could cause an effect (e.g. an
/// open pull request). ``weight=None`` derives the dose from ``attrs``.
#[pyclass(frozen, module = "heapdb", name = "Candidate")]
pub struct PyCandidate {
    #[pyo3(get)]
    id: Py<PyAny>,
    names: Vec<String>,
    #[pyo3(get)]
    weight: Py<PyAny>,
    #[pyo3(get, name = "type")]
    type_: Option<String>,
    #[pyo3(get)]
    attrs: Py<PyAny>,
    #[pyo3(get)]
    entity: Option<String>,
}

impl PyCandidate {
    fn key<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                self.id.bind(py).clone(),
                frozenset(py, &self.names)?.into_any(),
                self.weight.bind(py).clone(),
                self.type_.clone().into_pyobject(py)?.into_any(),
                self.entity.clone().into_pyobject(py)?.into_any(),
            ],
        )
    }

    pub fn to_engine(&self, py: Python<'_>) -> PyResult<Candidate> {
        let w = self.weight.bind(py);
        Ok(Candidate {
            features: self.names.clone(),
            weight: if w.is_none() {
                None
            } else {
                Some(weight_number(w)?)
            },
            event_type: self.type_.clone(),
            attrs: attrs_from(Some(self.attrs.bind(py)))?,
            entity: self.entity.clone(),
        })
    }

    /// `Candidate.from_any`.
    pub fn coerce(py: Python<'_>, c: &Bound<'_, PyAny>, index: usize) -> PyResult<Py<PyCandidate>> {
        if let Ok(x) = c.cast::<PyCandidate>() {
            return Ok(x.clone().unbind());
        }
        let mapping = py.import("collections.abc")?.getattr("Mapping")?;
        if c.is_instance(&mapping)? {
            let allowed = ["id", "features", "weight", "type", "attrs", "entity"];
            let mut unknown: Vec<String> = Vec::new();
            for k in c.try_iter()? {
                let k = k?;
                let ks: String = k.str()?.extract()?;
                if !(k.is_instance_of::<PyString>() && allowed.contains(&ks.as_str())) {
                    unknown.push(ks);
                }
            }
            if !unknown.is_empty() {
                unknown.sort();
                return Err(PyValueError::new_err(format!(
                    "unknown candidate keys {}",
                    heapdb::pyfmt::repr_str_list(unknown.iter().map(String::as_str))
                )));
            }
            let get = |k: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
                let v = c.call_method1("get", (k,))?;
                Ok(if v.is_none() { None } else { Some(v) })
            };
            let id = match get("id")? {
                Some(v) => v.unbind(),
                None => index.into_pyobject(py)?.into_any().unbind(),
            };
            let feats = get("features")?.filter(|f| f.is_truthy().unwrap_or(true));
            let cand = PyCandidate {
                id,
                names: feature_list(feats.as_ref())?,
                weight: get("weight")?.map_or_else(|| py.None(), Bound::unbind),
                type_: get("type")?.map(|t| t.extract()).transpose()?,
                attrs: get("attrs")?.map_or_else(|| py.None(), Bound::unbind),
                entity: get("entity")?.map(|t| t.extract()).transpose()?,
            };
            return Py::new(py, cand);
        }
        Err(PyTypeError::new_err(format!(
            "candidate must be a Candidate or a dict, got {}",
            c.get_type().name()?
        )))
    }
}

#[pymethods]
impl PyCandidate {
    #[new]
    #[pyo3(signature = (id, features = None, weight = None, r#type = None, attrs = None, entity = None))]
    fn new(
        py: Python<'_>,
        id: Py<PyAny>,
        features: Option<&Bound<'_, PyAny>>,
        weight: Option<Py<PyAny>>,
        r#type: Option<String>,
        attrs: Option<Py<PyAny>>,
        entity: Option<String>,
    ) -> PyResult<Self> {
        Ok(PyCandidate {
            id,
            names: feature_list(features)?,
            weight: weight.unwrap_or_else(|| py.None()),
            type_: r#type,
            attrs: attrs.unwrap_or_else(|| py.None()),
            entity,
        })
    }
    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        frozenset(py, &self.names)
    }
    #[classmethod]
    fn from_any(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        c: &Bound<'_, PyAny>,
        index: usize,
    ) -> PyResult<Py<PyCandidate>> {
        PyCandidate::coerce(py, c, index)
    }
    fn __eq__(&self, py: Python<'_>, o: &Bound<'_, PyAny>) -> PyResult<bool> {
        match o.cast::<PyCandidate>() {
            Ok(o) => self.key(py)?.eq(o.get().key(py)?),
            Err(_) => Ok(false),
        }
    }
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.key(py)?.hash()
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Candidate(id={}, features={}, weight={}, type={}, attrs={}, entity={})",
            self.id.bind(py).repr()?,
            self.features(py)?.repr()?,
            self.weight.bind(py).repr()?,
            self.type_
                .as_deref()
                .map_or("None".into(), heapdb::pyfmt::repr_str),
            self.attrs.bind(py).repr()?,
            self.entity
                .as_deref()
                .map_or("None".into(), heapdb::pyfmt::repr_str)
        ))
    }
}

// ---------------------------------------------------------------------------
// ObservationState
// ---------------------------------------------------------------------------

/// Per-effect state of observational learning (a snapshot copy).
#[pyclass(frozen, module = "heapdb", name = "ObservationState")]
pub struct PyObservationState {
    inner: ObservationState,
}

#[pymethods]
impl PyObservationState {
    #[new]
    fn new() -> Self {
        PyObservationState {
            inner: ObservationState::default(),
        }
    }
    #[getter]
    fn occurrences<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (e, ts) in &self.inner.occurrences {
            d.set_item(e, PyList::new(py, ts)?)?;
        }
        Ok(d)
    }
    #[getter]
    fn credited<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut v: Vec<(&u64, &f64)> = self.inner.credited.iter().collect();
        v.sort_by_key(|(k, _)| **k);
        let d = PyDict::new(py);
        for (k, x) in v {
            d.set_item(k, x)?;
        }
        Ok(d)
    }
    #[getter]
    fn rejected<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mut v: Vec<(&u64, &f64)> = self.inner.rejected.iter().collect();
        v.sort_by_key(|(k, _)| **k);
        let d = PyDict::new(py);
        for (k, x) in v {
            d.set_item(k, x)?;
        }
        Ok(d)
    }
    #[getter]
    fn cursors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, x) in &self.inner.cursors {
            d.set_item(k, x)?;
        }
        Ok(d)
    }
    #[getter]
    fn late<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, x) in &self.inner.late {
            d.set_item(k, PyList::new(py, x)?)?;
        }
        Ok(d)
    }
    #[getter]
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, x) in &self.inner.stats {
            d.set_item(k, x)?;
        }
        Ok(d)
    }
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("occurrences", self.occurrences(py)?)?;
        let cr = PyDict::new(py);
        for (k, v) in self.credited(py)?.iter() {
            cr.set_item(k.str()?, v)?;
        }
        d.set_item("credited", cr)?;
        let rj = PyDict::new(py);
        for (k, v) in self.rejected(py)?.iter() {
            rj.set_item(k.str()?, v)?;
        }
        d.set_item("rejected", rj)?;
        d.set_item("cursors", self.cursors(py)?)?;
        d.set_item("late", self.late(py)?)?;
        d.set_item("stats", self.stats(py)?)?;
        Ok(d)
    }
    fn __repr__(&self) -> String {
        format!(
            "ObservationState(occurrences={} entities, credited={}, rejected={}, cursors={})",
            self.inner.occurrences.len(),
            self.inner.credited.len(),
            self.inner.rejected.len(),
            self.inner.cursors.len()
        )
    }
}

// ---------------------------------------------------------------------------
// LinkTable (live view of an effect's links, or a standalone table)
// ---------------------------------------------------------------------------

enum LinkSrc {
    View(Py<PyHeapDB>, String),
    Owned(Box<(Vocab, LinkTable)>),
}

/// Hierarchical Beta link beliefs for one effect.
#[pyclass(module = "heapdb", name = "LinkTable")]
pub struct PyLinkTable {
    src: LinkSrc,
}

fn ent_ref(v: &Vocab, entity: Option<&str>) -> EntityRef {
    match entity {
        None => EntityRef::Global,
        Some(e) => EntityRef::Entity(v.entities.get(e)),
    }
}

fn sorted_names(features: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut v = str_list(features)?;
    v.sort();
    v.dedup();
    Ok(v)
}

impl PyLinkTable {
    fn read<R>(&self, py: Python<'_>, f: impl FnOnce(&Vocab, &LinkTable) -> PyResult<R>) -> PyResult<R> {
        match &self.src {
            LinkSrc::Owned(b) => f(&b.0, &b.1),
            LinkSrc::View(db, name) => {
                let db = db.bind(py).borrow();
                let spec = db.db.effect(name).py()?;
                f(db.db.vocab(), spec.links())
            }
        }
    }
    fn write<R>(
        &mut self,
        py: Python<'_>,
        f: impl FnOnce(&mut Vocab, &mut LinkTable) -> PyResult<R>,
    ) -> PyResult<R> {
        match &mut self.src {
            LinkSrc::Owned(b) => {
                let (v, t) = &mut **b;
                f(v, t)
            }
            LinkSrc::View(db, name) => {
                let mut db = db.bind(py).borrow_mut();
                let (t, v) = db.db.links_mut(name).py()?;
                f(v, t)
            }
        }
    }
}

#[pymethods]
impl PyLinkTable {
    #[new]
    #[pyo3(signature = (default_prior = (1.0, 4.0), priors = None, global_pseudocount = 5.0, hierarchical = true))]
    fn new(
        default_prior: (f64, f64),
        priors: Option<&Bound<'_, PyDict>>,
        global_pseudocount: f64,
        hierarchical: bool,
    ) -> PyResult<Self> {
        let mut t = LinkTable::new(default_prior, global_pseudocount, hierarchical).py()?;
        let mut v = Vocab::default();
        if let Some(p) = priors {
            for (f, ab) in p.iter() {
                let f: String = f.extract()?;
                let (a, b): (f64, f64) = ab.extract()?;
                let id = v.features.intern(&f);
                t.set_prior(id, &f, a, b).py()?;
            }
        }
        Ok(PyLinkTable {
            src: LinkSrc::Owned(Box::new((v, t))),
        })
    }

    #[getter]
    fn default_prior(&self, py: Python<'_>) -> PyResult<(f64, f64)> {
        self.read(py, |_, t| Ok(t.default_prior()))
    }
    #[getter]
    fn global_pseudocount(&self, py: Python<'_>) -> PyResult<f64> {
        self.read(py, |_, t| Ok(t.global_pseudocount()))
    }
    #[getter]
    fn hierarchical(&self, py: Python<'_>) -> PyResult<bool> {
        self.read(py, |_, t| Ok(t.hierarchical()))
    }

    fn prior(&self, py: Python<'_>, feature: &str) -> PyResult<(f64, f64)> {
        self.read(py, |v, t| Ok(t.prior(v.features.get(feature))))
    }

    /// Effective belief for ``feature``: global view, or for ``entity``.
    #[pyo3(signature = (feature, entity = None))]
    fn get(&self, py: Python<'_>, feature: &str, entity: Option<&str>) -> PyResult<PyBetaCounts> {
        self.read(py, |v, t| {
            Ok(PyBetaCounts {
                inner: t.get(v.features.get(feature), ent_ref(v, entity)),
            })
        })
    }

    #[pyo3(signature = (feature, entity = None))]
    fn strength(&self, py: Python<'_>, feature: &str, entity: Option<&str>) -> PyResult<f64> {
        self.read(py, |v, t| {
            Ok(t.strength(v.features.get(feature), ent_ref(v, entity)))
        })
    }

    #[pyo3(signature = (features, entity = None))]
    fn strengths<'py>(
        &self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
        entity: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let names = str_list(features)?;
        let vals = self.read(py, |v, t| {
            Ok(names
                .iter()
                .map(|n| t.strength(v.features.get(n), ent_ref(v, entity)))
                .collect::<Vec<f64>>())
        })?;
        let d = PyDict::new(py);
        for (n, s) in names.iter().zip(vals) {
            d.set_item(n, s)?;
        }
        Ok(d)
    }

    /// Aggregate (noisy-OR) strength of a feature set.
    #[pyo3(signature = (features, entity = None))]
    fn noisy_or(&self, py: Python<'_>, features: &Bound<'_, PyAny>, entity: Option<&str>) -> PyResult<f64> {
        let names = str_list(features)?;
        self.read(py, |v, t| {
            Ok(heapdb::noisy_or(
                names
                    .iter()
                    .map(|n| t.strength(v.features.get(n), ent_ref(v, entity))),
            ))
        })
    }

    /// Global view of every feature with a prior or global evidence (sorted).
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let rows = self.read(py, |v, t| {
            let mut fs: Vec<(String, heapdb::BetaCounts)> = t
                .known_features()
                .map(|f| (v.features.name(f).to_string(), t.get(Some(f), EntityRef::Global)))
                .collect();
            fs.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(fs)
        })?;
        let l = PyList::empty(py);
        for (f, b) in rows {
            l.append((f, PyBetaCounts { inner: b }))?;
        }
        Ok(l)
    }

    fn entities(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.read(py, |v, t| {
            let mut e: Vec<String> = t.entity_ids().map(|e| v.entities.name(e).to_string()).collect();
            e.sort();
            Ok(e)
        })
    }

    /// The entity's own (pos, neg) counts = its contributions to global.
    fn entity_counts<'py>(&self, py: Python<'py>, entity: &str) -> PyResult<Bound<'py, PyDict>> {
        let rows = self.read(py, |v, t| {
            Ok(match v.entities.get(entity) {
                None => Vec::new(),
                Some(e) => t
                    .entity_counts(e)
                    .into_iter()
                    .map(|(f, p, n)| (v.features.name(f).to_string(), p, n))
                    .collect(),
            })
        })?;
        let d = PyDict::new(py);
        for (f, p, n) in rows {
            d.set_item(f, (p, n))?;
        }
        Ok(d)
    }

    fn joint_prior(&self, py: Python<'_>, features: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
        let names = sorted_names(features)?;
        self.read(py, |v, t| {
            let ids: Vec<_> = names.iter().map(|n| v.features.get(n)).collect();
            Ok(t.joint_prior(&ids))
        })
    }

    /// Joint link for an exact feature set, or None if never judged.
    fn joint(
        &self,
        py: Python<'_>,
        entity: &str,
        features: &Bound<'_, PyAny>,
    ) -> PyResult<Option<PyBetaCounts>> {
        let names = sorted_names(features)?;
        self.read(py, |v, t| {
            let Some(e) = v.entities.get(entity) else {
                return Ok(None);
            };
            let ids: Option<Vec<u32>> = names.iter().map(|n| v.features.get(n)).collect();
            let Some(mut ids) = ids else { return Ok(None) };
            ids.sort_unstable();
            Ok(t.joint(e, &ids).map(|inner| PyBetaCounts { inner }))
        })
    }

    fn joint_sets<'py>(&self, py: Python<'py>, entity: &str) -> PyResult<Bound<'py, PyList>> {
        let sets = self.read(py, |v, t| {
            Ok(match v.entities.get(entity) {
                None => Vec::new(),
                Some(e) => t
                    .joint_sets(e)
                    .into_iter()
                    .map(|fs| {
                        fs.iter()
                            .map(|&f| v.features.name(f).to_string())
                            .collect::<Vec<_>>()
                    })
                    .collect(),
            })
        })?;
        let l = PyList::empty(py);
        for s in sets {
            l.append(frozenset(py, &s)?)?;
        }
        Ok(l)
    }

    fn update_joint(
        &mut self,
        py: Python<'_>,
        entity: &str,
        features: &Bound<'_, PyAny>,
        pos: f64,
        neg: f64,
    ) -> PyResult<()> {
        let names = sorted_names(features)?;
        self.write(py, |v, t| {
            let e = v.entities.intern(entity);
            let mut ids: Vec<u32> = names.iter().map(|n| v.features.intern(n)).collect();
            ids.sort_unstable();
            t.update_joint(e, &ids, pos, neg);
            Ok(())
        })
    }

    /// Set (or replace) the prior for one feature, keeping observed counts.
    fn set_prior(&mut self, py: Python<'_>, feature: &str, alpha: f64, beta: f64) -> PyResult<()> {
        self.write(py, |v, t| {
            let f = v.features.intern(feature);
            t.set_prior(f, feature, alpha, beta).py()
        })
    }

    /// Add evidence to the global layer and (if given) the entity layer.
    #[pyo3(signature = (feature, pos, neg, entity = None))]
    fn update(
        &mut self,
        py: Python<'_>,
        feature: &str,
        pos: f64,
        neg: f64,
        entity: Option<&str>,
    ) -> PyResult<()> {
        self.write(py, |v, t| {
            let f = v.features.intern(feature);
            let e = entity.map(|e| v.entities.intern(e));
            t.update(f, pos, neg, e);
            Ok(())
        })
    }

    /// Record a confirmed cause. Returns each feature's credit ``r_f``.
    #[pyo3(signature = (features, entity = None, credit = "proportional"))]
    fn confirm<'py>(
        &mut self,
        py: Python<'py>,
        features: &Bound<'py, PyAny>,
        entity: Option<&str>,
        credit: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mode = credit_arg(credit)?;
        let names = sorted_names(features)?;
        let resp = self.write(py, |v, t| {
            let ids: Vec<u32> = names.iter().map(|n| v.features.intern(n)).collect();
            let e = entity.map(|e| v.entities.intern(e));
            Ok(t.confirm(&ids, e, mode))
        })?;
        let d = PyDict::new(py);
        for (n, r) in names.iter().zip(resp) {
            d.set_item(n, r)?;
        }
        Ok(d)
    }

    /// Record a rejected candidate: every feature independently failed.
    #[pyo3(signature = (features, entity = None))]
    fn reject(&mut self, py: Python<'_>, features: &Bound<'_, PyAny>, entity: Option<&str>) -> PyResult<()> {
        let names = sorted_names(features)?;
        self.write(py, |v, t| {
            let ids: Vec<u32> = names.iter().map(|n| v.features.intern(n)).collect();
            let e = entity.map(|e| v.entities.intern(e));
            t.reject(&ids, e);
            Ok(())
        })
    }

    #[pyo3(signature = (entity, scrub_global = false))]
    fn forget_entity(&mut self, py: Python<'_>, entity: &str, scrub_global: bool) -> PyResult<usize> {
        self.write(py, |v, t| {
            Ok(v.entities
                .get(entity)
                .map_or(0, |e| t.forget_entity(e, scrub_global)))
        })
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = self.read(py, |v, t| Ok(t.to_json(v)))?;
        map_to_dict(py, &m)
    }

    #[classmethod]
    fn from_dict(_cls: &Bound<'_, PyType>, d: &Bound<'_, PyAny>) -> PyResult<Self> {
        let serde_json::Value::Object(m) = to_json(d)? else {
            return Err(PyTypeError::new_err("from_dict expects a dict"));
        };
        let mut v = Vocab::default();
        let t = LinkTable::from_json(&m, &mut v).py()?;
        Ok(PyLinkTable {
            src: LinkSrc::Owned(Box::new((v, t))),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.read(py, |_, t| {
            Ok(format!(
                "LinkTable(default_prior={:?}, global_pseudocount={}, hierarchical={})",
                t.default_prior(),
                t.global_pseudocount(),
                t.hierarchical()
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// EffectSpec (live view)
// ---------------------------------------------------------------------------

/// Everything HeapDB knows about one effect type (a live view).
#[pyclass(frozen, module = "heapdb", name = "EffectSpec")]
pub struct PyEffectSpec {
    db: Py<PyHeapDB>,
    #[pyo3(get)]
    name: String,
}

impl PyEffectSpec {
    fn with<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&HeapDB, &heapdb::EffectSpec) -> PyResult<R>,
    ) -> PyResult<R> {
        let db = self.db.bind(py).borrow();
        let spec = db.db.effect(&self.name).py()?;
        f(&db.db, spec)
    }
    fn with_mut<R>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut heapdb::EffectSpec) -> PyResult<R>,
    ) -> PyResult<R> {
        let mut db = self.db.bind(py).borrow_mut();
        let spec = db.db.effect_mut(&self.name).py()?;
        f(spec)
    }
}

#[pymethods]
impl PyEffectSpec {
    #[getter]
    fn kernel(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let k = self.with(py, |_, s| Ok(s.kernel().clone()))?;
        kernel_to_py(py, k)
    }
    #[setter]
    fn set_kernel(&self, py: Python<'_>, k: PyRef<'_, PyKernel>) -> PyResult<()> {
        let k = k.inner.clone();
        self.with_mut(py, |s| {
            s.set_kernel(k);
            Ok(())
        })
    }
    #[getter]
    fn links(&self, py: Python<'_>) -> PyResult<PyLinkTable> {
        self.with(py, |_, _| Ok(()))?;
        Ok(PyLinkTable {
            src: LinkSrc::View(self.db.clone_ref(py), self.name.clone()),
        })
    }
    #[getter]
    fn cause_types(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let c = self.with(py, |_, s| Ok(s.cause_types().cloned()))?;
        opt_str_set(py, c.as_ref())
    }
    #[getter]
    fn rules(&self, py: Python<'_>) -> PyResult<Vec<PyRule>> {
        self.with(py, |_, s| {
            Ok(s.rules().iter().map(|r| PyRule::from_rule(py, r)).collect())
        })
    }
    #[getter]
    fn credit(&self, py: Python<'_>) -> PyResult<&'static str> {
        self.with(py, |_, s| Ok(s.credit().as_str()))
    }
    #[setter]
    fn set_credit(&self, py: Python<'_>, c: &str) -> PyResult<()> {
        let c = credit_arg(c)?;
        self.with_mut(py, |s| {
            s.set_credit(c);
            Ok(())
        })
    }
    #[getter]
    fn pending_rules(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.with(py, |_, s| Ok(s.pending_rules().to_vec()))
    }
    #[getter]
    fn leak(&self, py: Python<'_>) -> PyResult<f64> {
        self.with(py, |_, s| Ok(s.leak()))
    }
    #[setter]
    fn set_leak(&self, py: Python<'_>, v: f64) -> PyResult<()> {
        self.with_mut(py, |s| s.set_leak(v).py())
    }
    #[getter]
    fn w_neg(&self, py: Python<'_>) -> PyResult<f64> {
        self.with(py, |_, s| Ok(s.w_neg()))
    }
    #[setter]
    fn set_w_neg(&self, py: Python<'_>, v: f64) -> PyResult<()> {
        self.with_mut(py, |s| s.set_w_neg(v).py())
    }
    #[getter]
    fn repeatable(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |_, s| Ok(s.repeatable()))
    }
    #[setter]
    fn set_repeatable(&self, py: Python<'_>, v: bool) -> PyResult<()> {
        self.with_mut(py, |s| {
            s.set_repeatable(v);
            Ok(())
        })
    }
    /// Observational bookkeeping (a copy).
    #[getter]
    fn obs(&self, py: Python<'_>) -> PyResult<PyObservationState> {
        self.with(py, |_, s| {
            Ok(PyObservationState {
                inner: s.observations().clone(),
            })
        })
    }
    /// Product of the multipliers of the rules matching ``event``.
    fn rule_multiplier(&self, py: Python<'_>, event: &Bound<'_, PyAny>) -> PyResult<f64> {
        let rules = self.rules(py)?;
        let mut m = 1.0;
        for r in rules {
            if m == 0.0 {
                break;
            }
            if r.matches(py, event)? {
                m *= r.multiplier;
            }
        }
        Ok(m)
    }
    /// Upper bound of the rule product for any event of ``type_``.
    fn rule_bound(&self, py: Python<'_>, type_: &str) -> PyResult<f64> {
        self.with(py, |_, s| Ok(s.rule_bound(type_)))
    }
    fn type_allowed(&self, py: Python<'_>, type_: &str) -> PyResult<bool> {
        self.with(py, |_, s| Ok(s.type_allowed(type_)))
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.with(py, |_, s| {
            Ok(format!(
                "EffectSpec(name={}, kernel={}, credit={}, rules={}, leak={}, w_neg={}, repeatable={})",
                heapdb::pyfmt::repr_str(s.name()),
                s.kernel().repr(),
                heapdb::pyfmt::repr_str(s.credit().as_str()),
                s.rules().len(),
                heapdb::pyfmt::repr_f64(s.leak()),
                heapdb::pyfmt::repr_f64(s.w_neg()),
                if s.repeatable() { "True" } else { "False" }
            ))
        })
    }
}

fn spec_name(spec: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = spec.cast::<PyEffectSpec>() {
        return Ok(s.get().name.clone());
    }
    spec.extract::<String>()
        .map_err(|_| PyTypeError::new_err("expected an EffectSpec or an effect name"))
}

// ---------------------------------------------------------------------------
// EntityLog / Segment (live views)
// ---------------------------------------------------------------------------

/// The segmented log of a single entity (a live view).
#[pyclass(frozen, module = "heapdb", name = "EntityLog")]
pub struct PyEntityLog {
    db: Py<PyHeapDB>,
    entity: String,
}

impl PyEntityLog {
    fn segs(&self, py: Python<'_>, range: Option<(f64, f64)>) -> Vec<PySegment> {
        let db = self.db.bind(py).borrow();
        db.db
            .segment_buckets(&self.entity, range)
            .into_iter()
            .map(|b| PySegment {
                db: self.db.clone_ref(py),
                entity: self.entity.clone(),
                bucket: b,
            })
            .collect()
    }
}

#[pymethods]
impl PyEntityLog {
    #[getter]
    fn bucket_seconds(&self, py: Python<'_>) -> f64 {
        self.db.bind(py).borrow().db.bucket_seconds()
    }
    #[getter]
    fn max_sets(&self, py: Python<'_>) -> usize {
        self.db.bind(py).borrow().db.max_sets_per_segment()
    }
    /// Segments whose bucket overlaps ``[t_lo, t_hi]``.
    fn segments_between(&self, py: Python<'_>, t_lo: f64, t_hi: f64) -> Vec<PySegment> {
        self.segs(py, Some((t_lo, t_hi)))
    }
    fn segments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyList::new(py, self.segs(py, None))?
            .try_iter()
            .map(Bound::into_any)
    }
    fn events(&self, py: Python<'_>) -> PyResult<PyEventIter> {
        let ids = self.db.bind(py).borrow().db.event_ids(Some(&self.entity));
        Ok(PyEventIter {
            db: self.db.clone_ref(py),
            ids,
            pos: 0,
        })
    }
    fn event_ids_with_feature(&self, py: Python<'_>, feature: &str) -> Vec<u64> {
        self.db
            .bind(py)
            .borrow()
            .db
            .event_ids_with_feature(&self.entity, feature)
    }
    fn __len__(&self, py: Python<'_>) -> usize {
        self.db
            .bind(py)
            .borrow()
            .db
            .entity_log(&self.entity)
            .map_or(0, |l| l.len())
    }
    #[getter]
    fn num_segments(&self, py: Python<'_>) -> usize {
        self.db
            .bind(py)
            .borrow()
            .db
            .entity_log(&self.entity)
            .map_or(0, |l| l.num_segments())
    }
}

/// All events of one entity in one time bucket, plus summaries (a view).
#[pyclass(frozen, module = "heapdb", name = "Segment")]
pub struct PySegment {
    db: Py<PyHeapDB>,
    entity: String,
    bucket: i64,
}

impl PySegment {
    fn info(&self, py: Python<'_>) -> PyResult<heapdb::SegmentInfo> {
        self.db
            .bind(py)
            .borrow()
            .db
            .segment_info(&self.entity, self.bucket)
            .ok_or_else(|| PyValueError::new_err("segment no longer exists"))
    }
}

#[pymethods]
impl PySegment {
    #[getter]
    fn start(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.info(py)?.start)
    }
    #[getter]
    fn width(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.info(py)?.width)
    }
    #[getter]
    fn end(&self, py: Python<'_>) -> PyResult<f64> {
        let i = self.info(py)?;
        Ok(i.start + i.width)
    }
    #[getter]
    fn min_time(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.info(py)?.min_time)
    }
    #[getter]
    fn max_time(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.info(py)?.max_time)
    }
    #[getter]
    fn max_weight(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.info(py)?.max_weight)
    }
    #[getter]
    fn max_sets(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.info(py)?.max_sets)
    }
    #[getter]
    fn events(&self, py: Python<'_>) -> PyResult<Vec<Py<PyEvent>>> {
        let ids = self.info(py)?.event_ids;
        let db = self.db.bind(py).borrow();
        ids.into_iter()
            .map(|i| PyEvent::new_py(py, db.db.get(i).py()?))
            .collect()
    }
    #[getter]
    fn feature_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (f, c) in self.info(py)?.feature_counts {
            d.set_item(f, c)?;
        }
        Ok(d)
    }
    #[getter]
    fn features<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let names: Vec<String> = self
            .info(py)?
            .feature_counts
            .into_iter()
            .map(|(f, _)| f)
            .collect();
        PySet::new(py, names)
    }
    #[getter]
    fn types<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, self.info(py)?.types)
    }
    #[getter]
    fn sets<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(sets) = self.info(py)?.sets else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        for (t, fs, w) in sets {
            d.set_item((t, frozenset(py, &fs)?), w)?;
        }
        Ok(Some(d))
    }
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let ids = self.info(py)?.event_ids;
        let db = self.db.bind(py).borrow();
        let mut idx: indexmap::IndexMap<String, Vec<u64>> = indexmap::IndexMap::new();
        for i in ids {
            let ev = db.db.get(i).py()?;
            for f in ev.features {
                idx.entry(f).or_default().push(i);
            }
        }
        for (f, v) in idx {
            d.set_item(f, v)?;
        }
        Ok(d)
    }
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(self.info(py)?.event_ids.len())
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        use heapdb::pyfmt::fmt_g6;
        let i = self.info(py)?;
        let nsets = i
            .sets
            .as_ref()
            .map_or("overflow".to_string(), |s| s.len().to_string());
        Ok(format!(
            "Segment(start={}, n={}, max_w={}, feats={}, sets={nsets})",
            fmt_g6(i.start),
            i.event_ids.len(),
            fmt_g6(i.max_weight),
            i.feature_counts.len()
        ))
    }
}

/// Lazy iterator over events by id.
#[pyclass(module = "heapdb")]
pub struct PyEventIter {
    db: Py<PyHeapDB>,
    ids: Vec<u64>,
    pos: usize,
}

#[pymethods]
impl PyEventIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyEvent>>> {
        let db = self.db.bind(py).borrow();
        while self.pos < self.ids.len() {
            let id = self.ids[self.pos];
            self.pos += 1;
            if let Ok(ev) = db.db.get(id) {
                return Ok(Some(PyEvent::new_py(py, ev)?));
            }
        }
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// HeapDB
// ---------------------------------------------------------------------------

/// An append-only event store answering "what most likely caused this?".
///
/// Args:
///     bucket_seconds: Width of the time buckets each entity's log is split
///         into (smaller: tighter bounds, more segments to bound).
///     max_sets_per_segment: Distinct (type, feature set) keys a segment
///         tracks for tight bounds before falling back to the union bound.
#[pyclass(dict, weakref, module = "heapdb", name = "HeapDB")]
pub struct PyHeapDB {
    pub db: HeapDB,
}

fn effects_arg(effects: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    match effects {
        None => Ok(None),
        Some(e) if e.is_none() => Ok(None),
        Some(e) => str_list(e).map(Some),
    }
}

/// A dose argument: a number, or None ("derive from attrs").
pub struct OptWeight(pub Option<f64>);

impl<'a, 'py> FromPyObject<'a, 'py> for OptWeight {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_none() {
            Ok(OptWeight(None))
        } else {
            Ok(OptWeight(Some(weight_number(&obj)?)))
        }
    }
}

fn entity_or_none(entity: Option<&str>) -> Option<String> {
    entity.map(str::to_string)
}

#[pymethods]
impl PyHeapDB {
    #[new]
    #[pyo3(signature = (bucket_seconds = 900.0, max_sets_per_segment = 64))]
    fn new(bucket_seconds: f64, max_sets_per_segment: i64) -> PyResult<Self> {
        if bucket_seconds.is_nan() || bucket_seconds <= 0.0 {
            return Err(PyValueError::new_err("bucket_seconds must be > 0"));
        }
        if max_sets_per_segment < 0 {
            return Err(PyValueError::new_err("max_sets_per_segment must be >= 0"));
        }
        Ok(PyHeapDB {
            db: HeapDB::new(bucket_seconds, max_sets_per_segment as usize).py()?,
        })
    }

    #[getter]
    fn bucket_seconds(&self) -> f64 {
        self.db.bucket_seconds()
    }
    #[getter]
    fn max_sets_per_segment(&self) -> usize {
        self.db.max_sets_per_segment()
    }
    /// Event types with weight functions named in a loaded snapshot that
    /// have not been re-registered yet.
    #[getter]
    fn pending_event_types<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        PySet::new(py, self.db.pending_event_types())
    }

    // -- schema -------------------------------------------------------------

    /// Declare an effect type, its lag kernel and its link priors.
    #[pyo3(signature = (name, kernel, priors = None, default_prior = (1.0, 4.0), cause_types = None,
                        credit = "proportional", global_pseudocount = 5.0, hierarchical = None,
                        leak = None, w_neg = None, repeatable = None))]
    #[allow(clippy::too_many_arguments)]
    fn register_effect(
        slf: &Bound<'_, Self>,
        name: String,
        kernel: PyRef<'_, PyKernel>,
        priors: Option<&Bound<'_, PyAny>>,
        default_prior: (f64, f64),
        cause_types: Option<&Bound<'_, PyAny>>,
        credit: &str,
        global_pseudocount: f64,
        hierarchical: Option<&Bound<'_, PyAny>>,
        leak: Option<&Bound<'_, PyAny>>,
        w_neg: Option<&Bound<'_, PyAny>>,
        repeatable: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyEffectSpec> {
        let py = slf.py();
        let mut opts = EffectOptions::new(kernel.inner.clone());
        opts.credit = credit_arg(credit)?;
        if let Some(l) = leak {
            opts.leak = py_float(l)?;
        }
        if let Some(w) = w_neg {
            opts.w_neg = py_float(w)?;
        }
        opts.default_prior = default_prior;
        opts.global_pseudocount = global_pseudocount;
        if let Some(h) = hierarchical {
            opts.hierarchical = h.is_truthy()?;
        }
        if let Some(r) = repeatable {
            opts.repeatable = r.is_truthy()?;
        }
        if let Some(p) = priors.filter(|p| !p.is_none()) {
            let d = PyDict::new(py);
            d.call_method1("update", (p,))?;
            for (f, ab) in d.iter() {
                let (a, b): (f64, f64) = ab.extract()?;
                opts.priors.push((f.extract()?, (a, b)));
            }
        }
        if let Some(c) = cause_types.filter(|c| !c.is_none()) {
            opts.cause_types = Some(str_list(c)?);
        }
        slf.borrow_mut().db.register_effect(&name, opts).py()?;
        Ok(PyEffectSpec {
            db: slf.clone().unbind(),
            name,
        })
    }

    fn effect(slf: &Bound<'_, Self>, name: String) -> PyResult<PyEffectSpec> {
        slf.borrow().db.effect(&name).py()?;
        Ok(PyEffectSpec {
            db: slf.clone().unbind(),
            name,
        })
    }

    #[getter]
    fn effects(&self) -> Vec<String> {
        self.db.effects()
    }

    /// Declare an event type, optionally deriving weight from ``attrs``.
    #[pyo3(signature = (r#type, weight = None))]
    fn register_event_type(&mut self, py: Python<'_>, r#type: &str, weight: Option<Py<PyAny>>) {
        let f: Option<Arc<dyn WeightFn>> = weight
            .filter(|w| !w.is_none(py))
            .map(|w| Arc::new(PyWeightFn(w)) as Arc<dyn WeightFn>);
        self.db.register_event_type(r#type, f);
    }

    /// Add a soft plausibility rule to ``effect`` (multiplier in [0, 1]).
    #[pyo3(signature = (effect, name, predicate, multiplier, types = None))]
    fn add_rule(
        &mut self,
        py: Python<'_>,
        effect: &str,
        name: &str,
        predicate: Py<PyAny>,
        multiplier: &Bound<'_, PyAny>,
        types: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRule> {
        self.db.effect(effect).py()?;
        let m = py_float(multiplier)?;
        if !(0.0..=1.0).contains(&m) {
            return Err(PyValueError::new_err(format!(
                "rule multiplier must be in [0, 1] (bounds rely on it), got {}",
                multiplier.repr()?
            )));
        }
        let pb = predicate.bind(py);
        if !pb.is_none() && !pb.is_callable() {
            return Err(PyTypeError::new_err("predicate must be callable or None"));
        }
        let pred: Option<Arc<dyn Predicate>> = if pb.is_none() {
            None
        } else {
            Some(Arc::new(PyPredicate(predicate.clone_ref(py))))
        };
        let types = types.filter(|t| !t.is_none()).map(str_list).transpose()?;
        let r = self.db.add_rule(effect, name, pred, m, types).py()?;
        Ok(PyRule::from_rule(py, r))
    }

    fn remove_rule(&mut self, effect: &str, name: &str) -> PyResult<()> {
        self.db.remove_rule(effect, name).py()
    }

    // -- writes ---------------------------------------------------------------

    /// Append an event and return it (with its assigned id).
    #[pyo3(signature = (entity, time, r#type, features = None, weight = None, attrs = None))]
    #[allow(clippy::too_many_arguments)]
    fn insert(
        &mut self,
        py: Python<'_>,
        entity: String,
        time: &Bound<'_, PyAny>,
        r#type: String,
        features: Option<&Bound<'_, PyAny>>,
        weight: Option<&Bound<'_, PyAny>>,
        attrs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyEvent>> {
        let weight = match weight {
            Some(w) if !w.is_none() => Some(weight_number(w)?),
            _ => None,
        };
        let ev = self
            .db
            .insert(NewEvent {
                entity,
                time: py_float(time)?,
                event_type: r#type,
                features: feature_list(features)?,
                weight,
                attrs: attrs_from(attrs)?,
            })
            .py()?;
        PyEvent::new_py(py, ev)
    }

    // -- reads ----------------------------------------------------------------

    fn get(&self, py: Python<'_>, event_id: u64) -> PyResult<Py<PyEvent>> {
        match self.db.get(event_id) {
            Ok(ev) => PyEvent::new_py(py, ev),
            Err(_) => Err(PyKeyError::new_err(event_id)),
        }
    }

    #[pyo3(signature = (entity = None))]
    fn events(slf: &Bound<'_, Self>, entity: Option<&str>) -> PyEventIter {
        let ids = slf.borrow().db.event_ids(entity);
        PyEventIter {
            db: slf.clone().unbind(),
            ids,
            pos: 0,
        }
    }

    #[getter]
    fn entities(&self) -> Vec<String> {
        self.db.entities()
    }

    fn entity_log(slf: &Bound<'_, Self>, entity: &str) -> Option<PyEntityLog> {
        slf.borrow().db.entity_log(entity)?;
        Some(PyEntityLog {
            db: slf.clone().unbind(),
            entity: entity.to_string(),
        })
    }

    /// Effective link belief for ``feature`` (global view if no entity).
    #[pyo3(signature = (effect, feature, entity = None))]
    fn link(&self, effect: &str, feature: &str, entity: Option<&str>) -> PyResult<PyBetaCounts> {
        Ok(PyBetaCounts {
            inner: self.db.link(effect, feature, entity).py()?,
        })
    }

    /// Score of a single event for an effect at time ``at``.
    fn score(&self, event: PyRef<'_, PyEvent>, effect: &str, at: f64) -> PyResult<f64> {
        self.db.score(&event.inner, effect, at).py()
    }

    /// Break an event's score into timing, weight, links and rules.
    fn explain(&self, event: PyRef<'_, PyEvent>, effect: &str, at: f64) -> PyResult<PyExplanation> {
        Ok(PyExplanation {
            inner: self.db.explain(&event.inner, effect, at).py()?,
        })
    }

    fn _window(&self, spec: &Bound<'_, PyAny>, at: f64, within: Option<f64>) -> PyResult<(f64, f64)> {
        self.db.window(&spec_name(spec)?, at, within).py()
    }

    /// Upper bound on the score of any event of ``seg``; and whether the
    /// loose union bound was used.
    #[pyo3(signature = (spec, seg, entity, at, t_lo, t_hi, use_sets = true, _cache = None))]
    #[allow(clippy::too_many_arguments)]
    fn segment_bound(
        &self,
        spec: &Bound<'_, PyAny>,
        seg: PyRef<'_, PySegment>,
        entity: &str,
        at: f64,
        t_lo: f64,
        t_hi: f64,
        use_sets: bool,
        _cache: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(f64, bool)> {
        self.db
            .segment_bound(
                &spec_name(spec)?,
                &seg.entity,
                seg.bucket,
                entity,
                at,
                t_lo,
                t_hi,
                use_sets,
            )
            .py()
    }

    /// Return the ``k`` most likely causes of ``effect`` for ``entity`` at ``at``.
    #[pyo3(signature = (entity, effect, at, k = 3, explain = false, within = None, _use_sets = true))]
    #[allow(clippy::too_many_arguments)]
    fn top_causes(
        &self,
        py: Python<'_>,
        entity: &str,
        effect: &str,
        at: f64,
        k: i64,
        explain: bool,
        within: Option<f64>,
        _use_sets: bool,
    ) -> PyResult<Py<PyQueryResult>> {
        if k <= 0 {
            return Err(PyValueError::new_err("k must be >= 1"));
        }
        let opts = TopK {
            k: k as usize,
            explain,
            within,
            use_sets: _use_sets,
        };
        let r = self.db.top_causes(entity, effect, at, opts).py()?;
        PyQueryResult::new_py(py, r)
    }

    /// Top-k causes of one effect across several entities (k-way merge).
    #[pyo3(signature = (entities, effect, at, k = 3, explain = false, within = None))]
    #[allow(clippy::too_many_arguments)]
    fn top_causes_group(
        &self,
        py: Python<'_>,
        entities: &Bound<'_, PyAny>,
        effect: &Bound<'_, PyAny>,
        at: f64,
        k: i64,
        explain: bool,
        within: Option<f64>,
    ) -> PyResult<Py<PyQueryResult>> {
        let effect: String = if let Ok(s) = effect.cast::<PyString>() {
            s.to_str()?.to_owned()
        } else {
            let effs = str_list(effect)?;
            let uniq: BTreeSet<&String> = effs.iter().collect();
            if uniq.len() != 1 {
                let names: Vec<&str> = uniq.iter().map(|s| s.as_str()).collect();
                return Err(PyValueError::new_err(format!(
                    "top_causes_group takes a single effect; mixing effects {} is not supported",
                    heapdb::pyfmt::repr_str_list(names)
                )));
            }
            effs[0].clone()
        };
        let ents = str_list(entities)?;
        if ents.is_empty() {
            return Err(PyValueError::new_err("need at least one entity"));
        }
        if k <= 0 {
            return Err(PyValueError::new_err("k must be >= 1"));
        }
        let refs: Vec<&str> = ents.iter().map(String::as_str).collect();
        let opts = TopK {
            k: k as usize,
            explain,
            within,
            use_sets: true,
        };
        let r = self.db.top_causes_group(&refs, &effect, at, opts).py()?;
        PyQueryResult::new_py(py, r)
    }

    /// Reference brute force: score everything, sort, cut. For testing.
    #[pyo3(signature = (entity, effect, at, k = 3, within = None))]
    fn top_causes_naive(
        &self,
        py: Python<'_>,
        entity: &str,
        effect: &str,
        at: f64,
        k: i64,
        within: Option<f64>,
    ) -> PyResult<Vec<(f64, Py<PyEvent>)>> {
        let k = k.max(0) as usize;
        self.db
            .top_causes_naive(entity, effect, at, k, within)
            .py()?
            .into_iter()
            .map(|(s, e)| Ok((s, PyEvent::new_py(py, e)?)))
            .collect()
    }

    /// Pop the top: the single most likely cause right now.
    fn most_likely_cause(
        &self,
        py: Python<'_>,
        entity: &str,
        effect: &str,
        at: f64,
    ) -> PyResult<Option<PyCause>> {
        self.db
            .most_likely_cause(entity, effect, at)
            .py()?
            .map(|c| PyCause::from_cause(py, c))
            .transpose()
    }

    // -- prospective risk -----------------------------------------------------

    /// Risk that a candidate which has **not happened yet** causes ``effect``.
    #[pyo3(signature = (effect, features, weight = OptWeight(Some(1.0)), entity = None, r#type = None, attrs = None, explain = false))]
    #[allow(clippy::too_many_arguments)]
    fn feature_risk(
        &self,
        py: Python<'_>,
        effect: &str,
        features: &Bound<'_, PyAny>,
        weight: OptWeight,
        entity: Option<&str>,
        r#type: Option<String>,
        attrs: Option<&Bound<'_, PyAny>>,
        explain: bool,
    ) -> PyResult<Py<PyAny>> {
        let w = weight.0;
        let c = Candidate {
            features: feature_list(Some(features))?,
            weight: w,
            event_type: r#type,
            attrs: attrs_from(attrs)?,
            entity: entity_or_none(entity),
        };
        let (s, ex) = self.db.feature_risk(effect, &c, explain).py()?;
        Ok(match ex {
            None => s.into_pyobject(py)?.into_any().unbind(),
            Some(inner) => (s, PyExplanation { inner })
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        })
    }

    /// Top-``k`` not-yet-happened candidates by ``feature_risk``.
    #[pyo3(signature = (effect, candidates, k = 10, entity = None, explain = false))]
    fn rank_candidates(
        &self,
        py: Python<'_>,
        effect: &str,
        candidates: &Bound<'_, PyAny>,
        k: i64,
        entity: Option<&str>,
        explain: bool,
    ) -> PyResult<Vec<PyRankedCandidate>> {
        if k <= 0 {
            return Err(PyValueError::new_err("k must be >= 1"));
        }
        self.db.effect(effect).py()?;
        let mut objs = Vec::new();
        let mut engine = Vec::new();
        for (i, c) in candidates.try_iter()?.enumerate() {
            let c = PyCandidate::coerce(py, &c?, i)?;
            engine.push(c.get().to_engine(py)?);
            objs.push(c);
        }
        let rows = self
            .db
            .rank_candidates(effect, &engine, k as usize, entity, explain)
            .py()?;
        rows.into_iter()
            .map(|r| {
                Ok(PyRankedCandidate {
                    rank: r.rank,
                    candidate: objs[r.index].clone_ref(py).into_any(),
                    score: r.score,
                    entity: r.entity,
                    explanation: expl_py(py, r.explanation)?,
                })
            })
            .collect()
    }

    // -- feedback ---------------------------------------------------------------

    /// Teach the database which candidate really was the cause.
    #[pyo3(signature = (entity, effect_type, at_time, confirmed_event_id = None, rejected_ids = None, credit = None))]
    fn feedback(
        &mut self,
        entity: &str,
        effect_type: &str,
        at_time: f64,
        confirmed_event_id: Option<&Bound<'_, PyAny>>,
        rejected_ids: Option<&Bound<'_, PyAny>>,
        credit: Option<&str>,
    ) -> PyResult<PyFeedbackReport> {
        let spec = self.db.effect(effect_type).py()?;
        let mode = match credit.filter(|c| !c.is_empty()) {
            Some(c) => credit_arg(c)?,
            None => spec.credit(),
        };
        let as_id = |o: &Bound<'_, PyAny>| -> PyResult<u64> {
            let i: i64 = o.py().get_type::<pyo3::types::PyInt>().call1((o,))?.extract()?;
            u64::try_from(i).map_err(|_| PyKeyError::new_err(format!("unknown event id {i}")))
        };
        let confirmed = match confirmed_event_id {
            Some(c) if !c.is_none() => Some(as_id(c)?),
            _ => None,
        };
        let mut rejected = Vec::new();
        if let Some(r) = rejected_ids.filter(|r| !r.is_none()) {
            for x in r.try_iter()? {
                rejected.push(as_id(&x?)?);
            }
        }
        Ok(PyFeedbackReport {
            inner: self
                .db
                .feedback(entity, effect_type, at_time, confirmed, &rejected, Some(mode))
                .py()?,
        })
    }

    // -- observational learning ---------------------------------------------------

    /// Record an occurrence of ``effect`` with **no labelled cause**.
    #[pyo3(signature = (entity, effect, at, leak = None))]
    fn observe(
        &mut self,
        entity: &str,
        effect: &str,
        at: &Bound<'_, PyAny>,
        leak: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObserveReport> {
        let leak = match leak {
            Some(l) if !l.is_none() => Some(py_float(l)?),
            _ => None,
        };
        Ok(PyObserveReport {
            inner: self.db.observe(entity, effect, py_float(at)?, leak).py()?,
        })
    }

    /// Add negative evidence for events whose window passed without the effect.
    #[pyo3(signature = (until, effects = None, entity = None, partial = false))]
    fn settle(
        &mut self,
        until: &Bound<'_, PyAny>,
        effects: Option<&Bound<'_, PyAny>>,
        entity: Option<&str>,
        partial: bool,
    ) -> PyResult<PySettleReport> {
        let effs = effects_arg(effects)?;
        let refs: Option<Vec<&str>> = effs.as_ref().map(|v| v.iter().map(String::as_str).collect());
        Ok(PySettleReport {
            inner: self
                .db
                .settle(py_float(until)?, refs.as_deref(), entity, partial)
                .py()?,
        })
    }

    /// Running counters of observational learning for ``effect``.
    fn observation_stats<'py>(&self, py: Python<'py>, effect: &str) -> PyResult<Bound<'py, PyDict>> {
        let st = self.db.observation_stats(effect).py()?;
        let d = PyDict::new(py);
        for (k, v) in st {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// Features ranked by mean link strength for ``effect`` (associations
    /// under the model, not causal effects).
    #[pyo3(signature = (effect, entity = None, min_support = 1.0, level = 0.9, baseline = None, features = None))]
    #[allow(clippy::too_many_arguments)]
    fn link_report(
        &self,
        effect: &str,
        entity: Option<&str>,
        min_support: f64,
        level: f64,
        baseline: Option<&Bound<'_, PyAny>>,
        features: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyLinkReport> {
        let baseline = match baseline {
            None => Baseline::Mean,
            Some(b) if b.is_instance_of::<PyString>() => match b.extract::<String>()?.as_str() {
                "mean" => Baseline::Mean,
                "prior" => Baseline::Prior,
                _ => {
                    return Err(PyValueError::new_err(
                        "baseline must be 'mean', 'prior' or a number",
                    ))
                }
            },
            Some(b) => Baseline::Value(py_float(b)?),
        };
        let features = features.filter(|f| !f.is_none()).map(str_list).transpose()?;
        self.db.effect(effect).py()?;
        let opts = LinkReportOptions {
            entity: entity_or_none(entity),
            min_support,
            level,
            baseline,
            features,
        };
        Ok(PyLinkReport {
            inner: self.db.link_report(effect, &opts).py()?,
        })
    }

    // -- calibration --------------------------------------------------------------

    fn calibration_records(&self, effect: &str) -> PyResult<Vec<PyCalibrationRecord>> {
        Ok(self
            .db
            .calibration_records(effect)
            .py()?
            .iter()
            .cloned()
            .map(|inner| PyCalibrationRecord { inner })
            .collect())
    }

    /// Mean predicted score vs observed confirm rate per score bin.
    #[pyo3(signature = (effect, bins = 10))]
    fn calibration_report(&self, effect: &str, bins: i64) -> PyResult<PyCalibrationReport> {
        if bins < 1 {
            return Err(PyValueError::new_err("bins must be >= 1"));
        }
        Ok(PyCalibrationReport {
            inner: self.db.calibration_report(effect, bins as usize).py()?,
        })
    }

    // -- privacy --------------------------------------------------------------------

    /// Remove everything stored about ``entity``.
    #[pyo3(signature = (entity, scrub_global = false))]
    fn forget_entity<'py>(
        &mut self,
        py: Python<'py>,
        entity: &str,
        scrub_global: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let r = self.db.forget_entity(entity, scrub_global);
        let d = PyDict::new(py);
        d.set_item("events", r.events)?;
        d.set_item("segments", r.segments)?;
        d.set_item("links", r.links)?;
        d.set_item("calibration_records", r.calibration_records)?;
        Ok(d)
    }

    // -- persistence ------------------------------------------------------------------

    /// Write a JSONL snapshot.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.db.save(path).py()
    }

    /// Rebuild a database from a snapshot (reads v1, v2 and v3).
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, path: PathBuf) -> PyResult<Self> {
        Ok(PyHeapDB {
            db: HeapDB::load(path).py()?,
        })
    }

    fn __len__(&self) -> usize {
        self.db.len()
    }

    fn __repr__(&self) -> String {
        self.db.repr()
    }
}
