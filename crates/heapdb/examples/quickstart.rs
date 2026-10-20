//! The README's Rust quickstart (kept compiling by `cargo test`).

use std::sync::Arc;

use heapdb::{EffectOptions, EventRef, HeapDB, Kernel, NewEvent, TopK};

fn main() -> heapdb::Result<()> {
    let h = 3600.0;
    let mut db = HeapDB::new(900.0, 64)?; // bucket_seconds, max_sets_per_segment
    db.register_effect(
        "bloating",
        EffectOptions::new(Kernel::lognormal(2.0 * h, 0.6)?)
            .prior("beans", 3.0, 1.0)
            .prior("dairy", 2.0, 1.0),
    )?;
    // Soft rule: small portions are rarely the cause (predicate = any closure).
    let small = |e: &EventRef<'_>| e.weight < 0.5;
    db.add_rule("bloating", "small portion", Some(Arc::new(small)), 0.4, None)?;

    db.insert(NewEvent::new("me", 8.0 * h, "meal").features(["oats", "dairy"]))?;
    let lunch = db.insert(
        NewEvent::new("me", 13.0 * h, "meal")
            .features(["beans", "onion"])
            .weight(1.5),
    )?;

    let res = db.top_causes("me", "bloating", 15.0 * h, TopK::new(3).explain(true))?;
    println!("{res}"); // same text as format_result() in Python

    let rep = db.feedback("me", "bloating", 15.0 * h, Some(lunch.id), &[], None)?;
    println!("{:?}", rep.credit); // [("beans", 0.9375), ("onion", 0.25)]
    db.save("me.jsonl")?;
    Ok(())
}
