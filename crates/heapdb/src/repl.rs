//! The REPL: statements plus `.meta` commands, shared by the `heapdb`
//! binary and the Python `heapdb.cli` module.

use crate::error::{Error, ErrorKind, Result};
use crate::heaps::HeapBackend;
use crate::kernels::Kernel;
use crate::pyfmt::{fmt_g6, repr_str, repr_str_list};
use crate::query::{execute_str, execute_str_with, parse_duration, py_float};
use crate::store::{EffectOptions, HeapDB, NewEvent};

/// Help text of the REPL.
pub const HELP: &str = r#"``python -m heapdb``: a minimal REPL over the query language.

Statements::

    TOP <k> CAUSES OF <effect> FOR <entity>[,<entity>...] [AT <time>]
        [WITHIN <duration>] [EXPLAIN]          several entities = group query
    FEEDBACK <effect> FOR <entity> AT <time> CONFIRMED <id> [REJECTED <id>,...]
    INSERT <type> FOR <entity> [AT <time>] [FEATURES f1,f2] [WEIGHT <dose>]
        [ATTRS k=v,k=v]
    CALIBRATION OF <effect> [BINS <n>]
    RISK OF <effect> FOR <entity> FEATURES f1,f2 [WEIGHT <dose>] [TYPE <t>]
        [ATTRS k=v,k=v] [EXPLAIN]      pre-event risk of a candidate that has
                                       not happened yet (e.g. an open PR)
    OBSERVE <effect> FOR <entity> [AT <time>] [LEAK <x>]
                                       an effect occurred, cause unknown:
                                       observational (soft EM) learning
    SETTLE [<effect>] UNTIL <time> [FOR <entity>]
                                       negative evidence for events whose
                                       window passed without the effect
    LINKS OF <effect> [FOR <entity>] [MIN <n>]
                                       learned links with 90% intervals and
                                       lift (associations, not causal effects)

Keyed heaps (named priority queues ordered by forward decay)::

    CREATE HEAP <heap> HALF LIFE <duration> [LANDMARK <time>] [META {json}]
    PUSH <id> INTO <heap> WEIGHT <w> [AT <time>] [ATTRS {json}]
    SET BASE <id> IN <heap> TO <base>
    SET ATTRS <id> IN <heap> TO {json} [MERGE]
    PEEK <k> FROM <heap> [AT <time>] [WHERE <condition>]
    POP FROM <heap> [AT <time>] [WHERE <condition>]
    REMOVE <id> FROM <heap>
    REBASE <heap> [AT <time>] [HALF LIFE <duration>]
    DROP HEAP <heap>
    SHOW HEAPS
                                       names and ids: bare words or 'quoted';
                                       <condition>: AND / OR / NOT / ( ),
                                       attr = != < <= > >= value,
                                       attr [NOT] IN (v, ...), attr CONTAINS v,
                                       attr CONTAINS ALL (...),
                                       attr SUBSET OF (...), attr EXISTS,
                                       attr MISSING; values: 'text', 1.5,
                                       TRUE, FALSE

``<time>``: seconds, now, now-2h, ISO-8601. ``<duration>``: 90, 30m, 12h, 2d.

Meta commands::

    .effect <name> <kernel> <args...>   register an effect, e.g.
                                        .effect bloating lognormal 2h 0.6
                                        .effect latency exp 30m
                                        .effect incident window 0 1h
    .prior <effect> <feature> <a> <b>   set a Beta prior for a feature
    .rules <effect>                     list plausibility rules (rules are
                                        Python callables: add them via the API)
    .forget <entity> [scrub]            delete an entity (scrub: also its
                                        contributions to global links)
    .events [entity]                    list stored events
    .save <path> / .load <path>         JSONL snapshot
    .help / .quit

Start with ``--load snapshot.jsonl`` to resume a saved database, and
``--demo`` to preload a small bloating example. ``--heaps native:<dir>``
keeps heaps in a directory (default: memory).
"#;

/// What a REPL line did.
#[derive(Debug)]
pub enum Outcome {
    /// Print this (or nothing).
    Output(Option<String>),
    /// `.quit` / `.exit`.
    Quit,
    /// `.load`: continue with this database.
    Replace(Box<HeapDB>, String),
}

fn arg(args: &[&str], i: usize) -> Result<String> {
    args.get(i).map(|s| s.to_string()).ok_or(Error::MissingArgument)
}

fn float_arg(s: &str) -> Result<f64> {
    py_float(s).ok_or_else(|| Error::invalid(format!("could not convert string to float: {}", repr_str(s))))
}

/// Build a kernel from REPL arguments (`exp 30m`, `window 0 1h`,
/// `lognormal 2h 0.6`, `gamma 2 30m`).
pub fn make_kernel(kind: &str, args: &[&str]) -> Result<Kernel> {
    match kind.to_lowercase().as_str() {
        "exp" => Kernel::exponential(parse_duration(&arg(args, 0)?)?),
        "window" => Kernel::window(parse_duration(&arg(args, 0)?)?, parse_duration(&arg(args, 1)?)?),
        "lognormal" => Kernel::lognormal(parse_duration(&arg(args, 0)?)?, float_arg(&arg(args, 1)?)?),
        "gamma" => Kernel::gamma(float_arg(&arg(args, 0)?)?, parse_duration(&arg(args, 1)?)?),
        k => Err(Error::query(format!(
            "unknown kernel {} (exp, window, lognormal, gamma)",
            repr_str(k)
        ))),
    }
}

/// A few meals over the last day, relative to `now`.
pub fn load_demo(db: &mut HeapDB, now: f64) -> Result<()> {
    db.register_effect(
        "bloating",
        EffectOptions::new(Kernel::lognormal(2.0 * 3600.0, 0.6)?)
            .prior("beans", 3.0, 1.0)
            .prior("dairy", 2.0, 1.0),
    )?;
    for (hours_ago, feats, w) in [
        (9.0, "oats,dairy", 1.0),
        (5.0, "bread,cheese,dairy", 1.0),
        (2.5, "beans,onion,rice", 1.5),
        (1.0, "apple", 0.5),
    ] {
        db.insert(
            NewEvent::new("me", now - hours_ago * 3600.0, "meal")
                .features(crate::event::split_features(feats))
                .weight(w),
        )?;
    }
    Ok(())
}

/// Process one REPL line. `clock` is called for statements only. Heap
/// statements fail (no heap backend): see [`handle_line_with`].
pub fn handle_line(db: &mut HeapDB, line: &str, clock: &mut dyn FnMut() -> Result<f64>) -> Result<Outcome> {
    handle(db, None, line, clock)
}

/// [`handle_line`] with heap statements running against `heaps`.
pub fn handle_line_with(
    db: &mut HeapDB,
    heaps: &mut dyn HeapBackend,
    line: &str,
    clock: &mut dyn FnMut() -> Result<f64>,
) -> Result<Outcome> {
    handle(db, Some(heaps), line, clock)
}

fn handle(
    db: &mut HeapDB,
    heaps: Option<&mut dyn HeapBackend>,
    line: &str,
    clock: &mut dyn FnMut() -> Result<f64>,
) -> Result<Outcome> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return Ok(Outcome::Output(None));
    }
    if line.starts_with('.') {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let cmd = parts[0].to_lowercase();
        let args = &parts[1..];
        let out = match cmd.as_str() {
            ".quit" | ".exit" => return Ok(Outcome::Quit),
            ".help" => HELP.to_string(),
            ".effect" => {
                if args.len() < 3 {
                    return Err(Error::query("usage: .effect <name> <kernel> <args...>"));
                }
                let k = make_kernel(args[1], &args[2..])?;
                let repr = k.repr();
                db.register_effect(args[0], EffectOptions::new(k))?;
                format!("ok: effect {} with {repr}", args[0])
            }
            ".prior" => {
                if args.len() != 4 {
                    return Err(Error::query("usage: .prior <effect> <feature> <alpha> <beta>"));
                }
                db.effect(args[0])?;
                let (a, b) = (float_arg(args[2])?, float_arg(args[3])?);
                let (links, vocab) = db.links_mut(args[0])?;
                let f = vocab.features.intern(args[1]);
                links.set_prior(f, args[1], a, b)?;
                format!("ok: prior {} ~ Beta({}, {})", args[1], args[2], args[3])
            }
            ".rules" => {
                if args.len() != 1 {
                    return Err(Error::query("usage: .rules <effect>"));
                }
                let spec = db.effect(args[0])?;
                let mut lines: Vec<String> = spec
                    .rules()
                    .iter()
                    .map(|r| {
                        let mut s = format!("{}: x{}", r.name(), fmt_g6(r.multiplier()));
                        if let Some(t) = r.types() {
                            s.push_str(&format!(
                                " for types {}",
                                repr_str_list(t.iter().map(String::as_str))
                            ));
                        }
                        if r.is_type_level() {
                            s.push_str(" (type-level)");
                        }
                        s
                    })
                    .collect();
                lines.extend(
                    spec.pending_rules()
                        .iter()
                        .map(|n| format!("{n}: (pending; re-register after load)")),
                );
                if lines.is_empty() {
                    "(no rules)".to_string()
                } else {
                    lines.join("\n")
                }
            }
            ".forget" => {
                if !(args.len() == 1 || args.len() == 2)
                    || (args.len() == 2 && args[1].to_lowercase() != "scrub")
                {
                    return Err(Error::query("usage: .forget <entity> [scrub]"));
                }
                let r = db.forget_entity(args[0], args.len() == 2);
                format!(
                    "ok: forgot {} events, {} segments, {} links, {} calibration_records",
                    r.events, r.segments, r.links, r.calibration_records
                )
            }
            ".events" => {
                let mut evs: Vec<crate::event::Event> = db.events(args.first().copied()).collect();
                evs.sort_by(|a, b| a.time.total_cmp(&b.time));
                if evs.is_empty() {
                    "(empty)".to_string()
                } else {
                    evs.iter()
                        .map(|e| format!("{}: {}", e.entity, e.short()))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
            ".save" => {
                let p = arg(args, 0)?;
                db.save(&p)?;
                format!("ok: saved {} events to {p}", db.len())
            }
            ".load" => {
                let p = arg(args, 0)?;
                let new = HeapDB::load(&p)?;
                let msg = format!("ok: loaded {}", new.repr());
                return Ok(Outcome::Replace(Box::new(new), msg));
            }
            _ => {
                return Err(Error::query(format!(
                    "unknown command {}; try .help",
                    repr_str(&cmd)
                )))
            }
        };
        return Ok(Outcome::Output(Some(out)));
    }
    let now = clock()?;
    let out = match heaps {
        Some(h) => execute_str_with(db, h, line, now)?,
        None => execute_str(db, line, now)?,
    };
    Ok(Outcome::Output(Some(out.to_string())))
}

/// How the reference REPL prints an error (`error: <str(exception)>`).
/// `None` for errors the reference REPL would not catch.
pub fn error_message(e: &Error) -> Option<String> {
    match e.kind() {
        ErrorKind::Key => Some(format!("error: {}", repr_str(&e.to_string()))),
        ErrorKind::Value | ErrorKind::Query | ErrorKind::Index | ErrorKind::Io => Some(format!("error: {e}")),
        ErrorKind::Type | ErrorKind::Callback => None,
    }
}
