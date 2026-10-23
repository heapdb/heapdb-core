//! JSONL snapshots.
//!
//! One `meta` line, one line per effect (kernel, links incl. global and
//! joint layers, rule metadata, observational state), one line per event
//! and one per calibration record. This is the format of the reference
//! implementation: versions 1, 2 and 3 are read; version 3 is written.
//!
//! Non-finite numbers (e.g. `leak = inf`) are written as the bare tokens
//! `Infinity` / `-Infinity` / `NaN`, as Python's `json` module does.
//!
//! Callables (predicate rules, weight functions) are not serialised: their
//! names are kept and they must be re-registered after loading.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::event::Event;
use crate::kernels::Kernel;
use crate::links::{Combine, Credit, LinkTable};
use crate::logistic::Logistic;
use crate::results::CalibrationRecord;
use crate::rules::Rule;
use crate::segments::DEFAULT_MAX_SETS;
use crate::store::{EffectSpec, HeapDB, ObservationState, DEFAULT_LEAK, SNAPSHOT_VERSION};

const NAN_MARK: &str = "\u{0}NaN";
const INF_MARK: &str = "\u{0}Infinity";
const NINF_MARK: &str = "\u{0}-Infinity";

/// A JSON number, or a marker for non-finite values (see module docs).
pub(crate) fn num(x: f64) -> Value {
    if x.is_finite() {
        Value::from(x)
    } else if x.is_nan() {
        Value::from(NAN_MARK)
    } else if x > 0.0 {
        Value::from(INF_MARK)
    } else {
        Value::from(NINF_MARK)
    }
}

pub(crate) fn pair(a: f64, b: f64) -> Value {
    Value::Array(vec![num(a), num(b)])
}

/// A number from JSON (accepting the non-finite markers).
pub(crate) fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) if s == NAN_MARK => Some(f64::NAN),
        Value::String(s) if s == INF_MARK => Some(f64::INFINITY),
        Value::String(s) if s == NINF_MARK => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

fn missing(k: &str) -> Error {
    Error::Snapshot(format!("missing or invalid field {}", crate::pyfmt::repr_str(k)))
}

pub(crate) fn get_f64(d: &Map<String, Value>, k: &str) -> Result<f64> {
    d.get(k).and_then(as_f64).ok_or_else(|| missing(k))
}

pub(crate) fn get_u64(d: &Map<String, Value>, k: &str) -> Result<u64> {
    let v = d.get(k).ok_or_else(|| missing(k))?;
    v.as_u64()
        .or_else(|| {
            v.as_f64()
                .filter(|x| *x >= 0.0 && x.fract() == 0.0)
                .map(|x| x as u64)
        })
        .ok_or_else(|| missing(k))
}

pub(crate) fn get_str<'a>(d: &'a Map<String, Value>, k: &str) -> Result<&'a str> {
    d.get(k).and_then(Value::as_str).ok_or_else(|| missing(k))
}

/// Serialise one record, turning markers back into bare tokens.
pub(crate) fn dump(v: &Map<String, Value>) -> String {
    let s = serde_json::to_string(v).expect("JSON maps always serialise");
    if !s.contains("\\u0000") {
        return s;
    }
    s.replace("\"\\u0000-Infinity\"", "-Infinity")
        .replace("\"\\u0000Infinity\"", "Infinity")
        .replace("\"\\u0000NaN\"", "NaN")
}

/// Parse one line, accepting bare `NaN` / `Infinity` / `-Infinity`.
pub(crate) fn parse_line(line: &str) -> std::result::Result<Value, serde_json::Error> {
    match serde_json::from_str(line) {
        Ok(v) => Ok(v),
        Err(e) => {
            let fixed = replace_bare_tokens(line);
            if fixed == line {
                Err(e)
            } else {
                serde_json::from_str(&fixed)
            }
        }
    }
}

fn replace_bare_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let mut in_str = false;
    let mut esc = false;
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() {
        let c = b[i] as char;
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_str = true;
            out.push(c);
            i += 1;
            continue;
        }
        let rest = &s[i..];
        let tok = [
            ("-Infinity", "\"\\u0000-Infinity\""),
            ("Infinity", "\"\\u0000Infinity\""),
            ("NaN", "\"\\u0000NaN\""),
        ]
        .into_iter()
        .find(|(t, _)| rest.starts_with(t));
        if let Some((t, rep)) = tok {
            out.push_str(rep);
            i += t.len();
            continue;
        }
        // Copy one UTF-8 character.
        let ch = rest.chars().next().expect("non-empty");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

impl ObservationState {
    pub(crate) fn to_json(&self) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert(
            "occurrences".into(),
            Value::Object(
                self.occurrences
                    .iter()
                    .map(|(e, ts)| (e.clone(), Value::Array(ts.iter().map(|&t| num(t)).collect())))
                    .collect(),
            ),
        );
        let sorted = |h: &rustc_hash::FxHashMap<u64, f64>| -> Value {
            let mut v: Vec<(&u64, &f64)> = h.iter().collect();
            v.sort_by_key(|(k, _)| **k);
            Value::Object(v.into_iter().map(|(k, x)| (k.to_string(), num(*x))).collect())
        };
        m.insert("credited".into(), sorted(&self.credited));
        m.insert("rejected".into(), sorted(&self.rejected));
        m.insert(
            "cursors".into(),
            Value::Object(self.cursors.iter().map(|(e, &t)| (e.clone(), num(t))).collect()),
        );
        m.insert(
            "late".into(),
            Value::Object(
                self.late
                    .iter()
                    .filter(|(_, ids)| !ids.is_empty())
                    .map(|(e, ids)| {
                        (
                            e.clone(),
                            Value::Array(ids.iter().map(|&i| Value::from(i)).collect()),
                        )
                    })
                    .collect(),
            ),
        );
        m.insert(
            "stats".into(),
            Value::Object(self.stats.iter().map(|(k, &v)| (k.clone(), num(v))).collect()),
        );
        m
    }

    pub(crate) fn from_json(d: Option<&Value>) -> Result<Self> {
        let mut st = ObservationState::default();
        let Some(Value::Object(d)) = d else {
            return Ok(st);
        };
        let bad = || Error::Snapshot("bad observation state".into());
        if let Some(Value::Object(o)) = d.get("occurrences") {
            for (e, ts) in o {
                let mut v: Vec<f64> = ts
                    .as_array()
                    .ok_or_else(bad)?
                    .iter()
                    .map(as_f64)
                    .collect::<Option<_>>()
                    .ok_or_else(bad)?;
                v.sort_by(|a, b| a.total_cmp(b));
                st.occurrences.insert(e.clone(), v);
            }
        }
        for (key, target) in [("credited", &mut st.credited), ("rejected", &mut st.rejected)] {
            if let Some(Value::Object(o)) = d.get(key) {
                for (k, v) in o {
                    let id: u64 = k.trim().parse().map_err(|_| bad())?;
                    target.insert(id, as_f64(v).ok_or_else(bad)?);
                }
            }
        }
        if let Some(Value::Object(o)) = d.get("cursors") {
            for (e, t) in o {
                st.cursors.insert(e.clone(), as_f64(t).ok_or_else(bad)?);
            }
        }
        if let Some(Value::Object(o)) = d.get("late") {
            for (e, ids) in o {
                let v: Vec<u64> = ids
                    .as_array()
                    .ok_or_else(bad)?
                    .iter()
                    .map(Value::as_u64)
                    .collect::<Option<_>>()
                    .ok_or_else(bad)?;
                st.late.insert(e.clone(), v);
            }
        }
        if let Some(Value::Object(o)) = d.get("stats") {
            for (k, v) in o {
                st.stats.insert(k.clone(), as_f64(v).ok_or_else(bad)?);
            }
        }
        Ok(st)
    }
}

impl HeapDB {
    /// Write a JSONL snapshot to `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        self.save_to(&mut w)?;
        w.flush()?;
        Ok(())
    }

    /// Write a JSONL snapshot to any writer.
    pub fn save_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let mut meta = Map::new();
        meta.insert("kind".into(), "meta".into());
        meta.insert("version".into(), SNAPSHOT_VERSION.into());
        meta.insert("bucket_seconds".into(), num(self.bucket_seconds));
        meta.insert("max_sets_per_segment".into(), self.max_sets.into());
        meta.insert("next_id".into(), self.next_id.into());
        meta.insert(
            "event_types".into(),
            Value::Object(
                self.event_types
                    .iter()
                    .map(|(t, f)| (t.clone(), Value::Bool(f.is_some())))
                    .collect(),
            ),
        );
        writeln!(w, "{}", dump(&meta))?;
        for spec in self.effects.values() {
            let mut m = Map::new();
            m.insert("kind".into(), "effect".into());
            m.insert("name".into(), spec.name.clone().into());
            m.insert("kernel".into(), Value::Object(spec.kernel.to_json()));
            m.insert("links".into(), Value::Object(spec.links.to_json(&self.vocab)));
            m.insert(
                "cause_types".into(),
                match &spec.cause_types {
                    Some(c) => Value::Array(c.iter().cloned().map(Value::from).collect()),
                    None => Value::Null,
                },
            );
            m.insert("credit".into(), spec.credit.as_str().into());
            // Only logistic effects carry these keys, so noisy-OR snapshots
            // are unchanged (and older snapshots load as noisy-OR).
            if let Some(l) = spec.links.logistic() {
                m.insert("combine".into(), Combine::Logistic.as_str().into());
                m.insert("logistic".into(), Value::Object(l.to_json(&self.vocab)));
            }
            m.insert(
                "rules".into(),
                Value::Array(
                    spec.rules
                        .iter()
                        .map(|r| {
                            let mut x = Map::new();
                            x.insert("name".into(), r.name.clone().into());
                            x.insert("multiplier".into(), num(r.multiplier));
                            x.insert(
                                "types".into(),
                                match &r.types {
                                    Some(t) => Value::Array(t.iter().cloned().map(Value::from).collect()),
                                    None => Value::Null,
                                },
                            );
                            x.insert("predicate".into(), Value::Bool(r.predicate.is_some()));
                            Value::Object(x)
                        })
                        .collect(),
                ),
            );
            m.insert(
                "pending_rules".into(),
                Value::Array(spec.pending_rules.iter().cloned().map(Value::from).collect()),
            );
            m.insert("leak".into(), num(spec.leak));
            m.insert("w_neg".into(), num(spec.w_neg));
            m.insert("repeatable".into(), Value::Bool(spec.repeatable));
            m.insert("observations".into(), Value::Object(spec.obs.to_json()));
            writeln!(w, "{}", dump(&m))?;
        }
        for &(id, e, o) in &self.id_index {
            let (log, ev) = self.stored(e, o);
            let mut m = Map::new();
            m.insert("kind".into(), "event".into());
            m.extend(self.to_event(e, log, ev).to_json());
            let _ = id;
            writeln!(w, "{}", dump(&m))?;
        }
        let mut effs: Vec<&String> = self.calibration.keys().collect();
        effs.sort();
        for eff in effs {
            for r in &self.calibration[eff] {
                let mut m = Map::new();
                m.insert("kind".into(), "calibration".into());
                m.insert("effect".into(), eff.clone().into());
                m.insert("entity".into(), r.entity.clone().into());
                m.insert("event_id".into(), r.event_id.into());
                m.insert("at".into(), num(r.at));
                m.insert("score".into(), num(r.score));
                m.insert("confirmed".into(), Value::Bool(r.confirmed));
                writeln!(w, "{}", dump(&m))?;
            }
        }
        Ok(())
    }

    /// Rebuild a database (segments, indexes, links, calibration) from a
    /// snapshot file. Reads v1, v2 and v3 snapshots.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let r = BufReader::new(File::open(path)?);
        Self::load_from(r, &path.display().to_string())
    }

    /// [`HeapDB::load`] from any reader; `name` is used in error messages.
    pub fn load_from<R: BufRead>(r: R, name: &str) -> Result<Self> {
        let mut db: Option<HeapDB> = None;
        for (i, line) in r.lines().enumerate() {
            let line = line?;
            let lineno = i + 1;
            if line.trim().is_empty() {
                continue;
            }
            let ctx = |e: Error| match e {
                Error::Snapshot(m) => Error::Snapshot(format!("{name}:{lineno}: {m}")),
                other => other,
            };
            let rec = parse_line(&line).map_err(|e| Error::Snapshot(format!("{name}:{lineno}: {e}")))?;
            let Value::Object(rec) = rec else {
                return Err(Error::Snapshot(format!(
                    "{name}:{lineno}: record must be an object"
                )));
            };
            let kind = rec.get("kind").and_then(Value::as_str);
            if kind == Some("meta") {
                let bucket = get_f64(&rec, "bucket_seconds").map_err(ctx)?;
                let ms = match rec.get("max_sets_per_segment") {
                    Some(v) => v.as_u64().ok_or_else(|| ctx(missing("max_sets_per_segment")))? as usize,
                    None => DEFAULT_MAX_SETS,
                };
                let mut d = HeapDB::new(bucket, ms)?;
                d.next_id = get_u64(&rec, "next_id").map_err(ctx)?;
                if let Some(Value::Object(ts)) = rec.get("event_types") {
                    for (t, has) in ts {
                        d.event_types.insert(t.clone(), None);
                        if has.as_bool().unwrap_or(false) {
                            d.pending_event_types.insert(t.clone());
                        }
                    }
                }
                db = Some(d);
                continue;
            }
            let Some(d) = db.as_mut() else {
                return Err(Error::Snapshot(format!(
                    "{name}:{lineno}: snapshot must start with a meta line"
                )));
            };
            match kind {
                Some("effect") => d.load_effect(&rec).map_err(ctx)?,
                Some("event") => {
                    let ev = Event::from_json(&rec).map_err(ctx)?;
                    d.append(&ev, false);
                    d.next_id = d.next_id.max(ev.id + 1);
                }
                Some("calibration") => {
                    let r = CalibrationRecord {
                        entity: get_str(&rec, "entity").map_err(ctx)?.to_string(),
                        event_id: get_u64(&rec, "event_id").map_err(ctx)?,
                        at: get_f64(&rec, "at").map_err(ctx)?,
                        score: get_f64(&rec, "score").map_err(ctx)?,
                        confirmed: rec.get("confirmed").and_then(Value::as_bool).unwrap_or(false),
                    };
                    let eff = get_str(&rec, "effect").map_err(ctx)?.to_string();
                    d.calibration.entry(eff).or_default().push(r);
                }
                other => {
                    return Err(Error::Snapshot(format!(
                        "{name}:{lineno}: unknown record kind {}",
                        other.map_or("None".to_string(), crate::pyfmt::repr_str)
                    )))
                }
            }
        }
        db.ok_or_else(|| Error::Snapshot(format!("{name}: empty snapshot")))
    }

    fn load_effect(&mut self, rec: &Map<String, Value>) -> Result<()> {
        let name = get_str(rec, "name")?.to_string();
        let kernel = match rec.get("kernel") {
            Some(Value::Object(k)) => Kernel::from_json(k)?,
            _ => return Err(missing("kernel")),
        };
        let mut links = match rec.get("links") {
            Some(Value::Object(l)) => LinkTable::from_json(l, &mut self.vocab)?,
            _ => return Err(missing("links")),
        };
        let combine = Combine::parse(rec.get("combine").and_then(Value::as_str).unwrap_or("noisy_or"))?;
        if combine == Combine::Logistic {
            let state = match rec.get("logistic") {
                Some(Value::Object(l)) => Logistic::from_json(l, &mut self.vocab)?,
                _ => Logistic::new(Default::default())?,
            };
            links.set_logistic_state(state);
        }
        let cause_types: Option<BTreeSet<String>> = match rec.get("cause_types") {
            Some(Value::Array(a)) if !a.is_empty() => Some(
                a.iter()
                    .map(|v| v.as_str().map(str::to_string))
                    .collect::<Option<_>>()
                    .ok_or_else(|| missing("cause_types"))?,
            ),
            _ => None,
        };
        let cause_type_ids = cause_types
            .as_ref()
            .map(|c| c.iter().map(|t| self.vocab.types.intern(t)).collect());
        let credit = Credit::parse(
            rec.get("credit")
                .and_then(Value::as_str)
                .unwrap_or("proportional"),
        )?;
        let mut pending: Vec<String> = match rec.get("pending_rules") {
            Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            _ => Vec::new(),
        };
        let mut rules = Vec::new();
        if let Some(Value::Array(rs)) = rec.get("rules") {
            for r in rs {
                let r = r.as_object().ok_or_else(|| missing("rules"))?;
                let rname = get_str(r, "name")?.to_string();
                if r.get("predicate").and_then(Value::as_bool).unwrap_or(false) {
                    pending.push(rname);
                    continue;
                }
                let types: Option<BTreeSet<String>> = match r.get("types") {
                    Some(Value::Array(a)) => {
                        Some(a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                    }
                    _ => None,
                };
                let type_ids = types
                    .as_ref()
                    .map(|t| t.iter().map(|x| self.vocab.types.intern(x)).collect());
                rules.push(Rule {
                    name: rname,
                    predicate: None,
                    multiplier: get_f64(r, "multiplier")?,
                    types,
                    type_ids,
                });
            }
        }
        let spec = EffectSpec {
            name: name.clone(),
            kernel,
            links,
            cause_types,
            cause_type_ids,
            rules,
            credit,
            pending_rules: pending,
            leak: rec.get("leak").and_then(as_f64).unwrap_or(DEFAULT_LEAK),
            w_neg: rec.get("w_neg").and_then(as_f64).unwrap_or(1.0),
            repeatable: rec.get("repeatable").and_then(Value::as_bool).unwrap_or(true),
            obs: ObservationState::from_json(rec.get("observations"))?,
        };
        self.effects.insert(name, spec);
        Ok(())
    }

    /// The whole snapshot as a string (convenient for tests).
    pub fn snapshot_string(&self) -> String {
        let mut buf = Vec::new();
        self.save_to(&mut buf).expect("writing to memory cannot fail");
        String::from_utf8(buf).expect("JSON is UTF-8")
    }
}
