//! The README's Rust quickstart (kept compiling by `cargo test`).

use std::sync::Arc;

use heapdb::{EffectOptions, EventRef, HeapDB, Kernel, NewEvent, TopK};

fn main() -> heapdb::Result<()> {
    let (m, h) = (60.0, 3600.0);
    let mut db = HeapDB::new(900.0, 64)?; // bucket_seconds, max_sets_per_segment
    db.register_effect(
        "error_spike",
        EffectOptions::new(Kernel::lognormal(45.0 * m, 0.7)?)
            .prior("migration", 3.0, 1.0)
            .prior("deps", 2.0, 2.0),
    )?;
    // Soft rule: docs-only changes are rarely the cause (predicate = any closure).
    let docs_only = |e: &EventRef<'_>| e.features.len() == 1 && e.features.contains("docs");
    db.add_rule("error_spike", "docs only", Some(Arc::new(docs_only)), 0.1, None)?;

    // Dose = change size: log1p(lines changed), a 100-line PR weighs 1.0.
    let size = |lines: f64| lines.ln_1p() / 100f64.ln_1p();
    let deploy = |t: f64, features: &[&str], lines: f64| {
        NewEvent::new("checkout", t, "deploy")
            .features(features.iter().copied())
            .weight(size(lines))
    };
    db.insert(deploy(9.0 * h + 10.0 * m, &["docs"], 12.0))?;
    db.insert(deploy(9.0 * h + 40.0 * m, &["deps", "tests"], 180.0))?;
    db.insert(deploy(10.0 * h + 5.0 * m, &["config"], 8.0))?;
    let pr412 = db.insert(deploy(10.0 * h + 30.0 * m, &["migration", "review:fast"], 420.0))?;
    db.insert(deploy(11.0 * h, &["tests"], 60.0))?;

    let spike = 11.0 * h + 20.0 * m;
    let res = db.top_causes("checkout", "error_spike", spike, TopK::new(3).explain(true))?;
    println!("{res}"); // same text as format_result() in Python

    let rep = db.feedback("checkout", "error_spike", spike, Some(pr412.id), &[], None)?;
    println!("{:?}", rep.credit); // [("migration", 0.9375), ("review:fast", 0.25)]
    db.save("checkout.jsonl")?;
    Ok(())
}
