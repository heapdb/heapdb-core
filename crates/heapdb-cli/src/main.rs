//! `heapdb`: a REPL over the HeapDB query language, and a runner for query
//! files.
//!
//! ```text
//! heapdb [--load SNAPSHOT] [--demo] [--bucket SECONDS] [--heaps PERSIST_TO] [FILE ...]
//! ```
//!
//! Without files it starts the REPL (statements plus `.meta` commands, see
//! `.help`). With files it executes each line of each file in order, prints
//! the outputs and exits (status 1 if any line failed). `-` reads standard
//! input as a file. Heap statements (`CREATE HEAP`, `PUSH`, `PEEK`, ...) run
//! against `--heaps` (`memory`, the default, or `native:<dir>`).

use std::fs::File;
use std::io::{self, BufRead, BufReader, IsTerminal, Write};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use heapdb::heaps::{HeapBackend, PersistTo};
use heapdb::repl::{error_message, handle_line_with, load_demo, Outcome};
use heapdb::{HeapDB, DEFAULT_MAX_SETS};

const USAGE: &str =
    "usage: heapdb [--load SNAPSHOT] [--demo] [--bucket SECONDS] [--heaps memory|native:<dir>] [FILE ...]";

fn now() -> heapdb::Result<f64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0))
}

struct Args {
    load: Option<String>,
    demo: bool,
    bucket: f64,
    heaps: PersistTo,
    files: Vec<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        load: None,
        demo: false,
        bucket: 900.0,
        heaps: PersistTo::Memory,
        files: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => return Err(format!("{USAGE}\n\n{}", heapdb::repl::HELP)),
            "--load" => a.load = Some(it.next().ok_or("--load needs a path")?),
            "--demo" => a.demo = true,
            "--bucket" => {
                let v = it.next().ok_or("--bucket needs a number")?;
                a.bucket = v.parse().map_err(|_| format!("bad --bucket {v:?}"))?;
            }
            "--heaps" => {
                let v = it.next().ok_or("--heaps needs memory or native:<dir>")?;
                a.heaps = PersistTo::parse(&v).map_err(|e| e.to_string())?;
            }
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n{USAGE}")),
            _ => a.files.push(arg),
        }
    }
    Ok(a)
}

/// Process lines; returns (keep going, number of failed lines).
fn run_lines(
    db: &mut HeapDB,
    heaps: &mut dyn HeapBackend,
    lines: impl Iterator<Item = io::Result<String>>,
    out: &mut impl Write,
    prompt: bool,
) -> io::Result<(bool, usize)> {
    let mut failed = 0;
    let mut clock = now;
    if prompt {
        write!(out, "heapdb> ")?;
        out.flush()?;
    }
    for line in lines {
        let line = line?;
        match handle_line_with(db, heaps, &line, &mut clock) {
            Ok(Outcome::Quit) => return Ok((false, failed)),
            Ok(Outcome::Output(Some(s))) => writeln!(out, "{s}")?,
            Ok(Outcome::Output(None)) => {}
            Ok(Outcome::Replace(new, msg)) => {
                *db = *new;
                writeln!(out, "{msg}")?;
            }
            Err(e) => {
                failed += 1;
                let msg = error_message(&e).unwrap_or_else(|| format!("error: {e}"));
                writeln!(out, "{msg}")?;
            }
        }
        if prompt {
            write!(out, "heapdb> ")?;
            out.flush()?;
        }
    }
    Ok((true, failed))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };
    let mut db = match &args.load {
        Some(p) => match HeapDB::load(p) {
            Ok(db) => db,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(1);
            }
        },
        None => match HeapDB::new(args.bucket, DEFAULT_MAX_SETS) {
            Ok(db) => db,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        },
    };
    if args.demo {
        if let Err(e) = load_demo(&mut db, now().unwrap_or(0.0)) {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    }
    let mut heaps = match args.heaps.open() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let heaps = heaps.as_mut();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let result = if args.files.is_empty() {
        let stdin = io::stdin();
        let interactive = stdin.is_terminal();
        if interactive {
            let _ = writeln!(out, "HeapDB REPL. Type .help for commands, .quit to exit.");
        }
        run_lines(&mut db, heaps, stdin.lock().lines(), &mut out, interactive).map(|(_, f)| f)
    } else {
        let mut failed = 0;
        let mut res = Ok(0);
        for path in &args.files {
            let r = if path == "-" {
                run_lines(&mut db, heaps, io::stdin().lock().lines(), &mut out, false)
            } else {
                match File::open(path) {
                    Ok(f) => run_lines(&mut db, heaps, BufReader::new(f).lines(), &mut out, false),
                    Err(e) => Err(io::Error::new(e.kind(), format!("{path}: {e}"))),
                }
            };
            match r {
                Ok((more, f)) => {
                    failed += f;
                    if !more {
                        break;
                    }
                }
                Err(e) => {
                    res = Err(e);
                    break;
                }
            }
        }
        res.map(|_| failed)
    };
    match result {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
