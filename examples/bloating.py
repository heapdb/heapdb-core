"""A day of meals, a bout of bloating, and learning from feedback.

Run:  python examples/bloating.py

Story: the user starts with the folk prior "beans cause bloating" and only a
mild suspicion of dairy. Each day they eat beans at lunch and have a latte in
the afternoon, then feel bloated around 16:00. HeapDB initially blames the
beans. But the user is in fact lactose intolerant: an elimination test shows
it is the latte, so they confirm the latte and reject the lunch. After a few
days of feedback the ranking flips.

Note the confounding warning at the end: 'coffee' and 'dairy' always arrive
together in a latte, so feedback alone cannot tell which of the two it is.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heapdb import HeapDB, LogNormalKernel, format_result  # noqa: E402

H = 3600.0
DAY = 24 * H
BLOATED_AT_HOUR = 16.0


def meals_for_day(db: HeapDB, day: int) -> dict[str, int]:
    """Insert one day of meals; return name -> event id."""
    t0 = day * DAY
    menu = [
        ("breakfast", 7.5, {"oats", "banana"}, 1.0),
        ("lunch", 13.0, {"beans", "onion", "rice"}, 1.5),
        ("latte", 14.5, {"coffee", "dairy"}, 0.8),
        ("dinner", 19.0, {"pasta", "tomato"}, 1.2),  # after the bloating: ignored
    ]
    return {
        name: db.insert("me", t0 + hour * H, "meal", feats, weight=w).id
        for name, hour, feats, w in menu
    }


def main() -> None:
    db = HeapDB(bucket_seconds=900)
    db.register_effect(
        "bloating",
        kernel=LogNormalKernel(median=2 * H, sigma=0.9),
        priors={"beans": (4, 1), "onion": (2, 2), "dairy": (1, 3)},
        cause_types={"meal"},
    )

    # --- Day 0: first bout of bloating ------------------------------------
    ids = meals_for_day(db, 0)
    at = BLOATED_AT_HOUR * H
    print("=== Day 0, bloated at 16:00 -- before any feedback ===")
    print(format_result(db.top_causes("me", "bloating", at=at, k=3, explain=True)))

    # Elimination test says: it's the latte, not the beans.
    print("\nfeedback: CONFIRMED latte, REJECTED lunch (and the same on days 1-4)\n")
    db.feedback("me", "bloating", at,
                confirmed_event_id=ids["latte"], rejected_ids=[ids["lunch"]])
    for day in range(1, 5):
        ids = meals_for_day(db, day)
        db.feedback("me", "bloating", day * DAY + BLOATED_AT_HOUR * H,
                    confirmed_event_id=ids["latte"], rejected_ids=[ids["lunch"]])

    # --- Day 5: same menu, bloated again ----------------------------------
    meals_for_day(db, 5)
    at = 5 * DAY + BLOATED_AT_HOUR * H
    print("=== Day 5, bloated at 16:00 -- after feedback ===")
    print(format_result(db.top_causes("me", "bloating", at=at, k=3, explain=True)))

    links = db.effect("bloating").links
    print("\nlearned link strengths:")
    for f in ("beans", "onion", "rice", "dairy", "coffee"):
        c = links.get(f)
        print(f"  {f:<7} {c.strength:.2f}   Beta({c.alpha:g}, {c.beta:g})")

    top = db.most_likely_cause("me", "bloating", at)
    assert top is not None
    print("\npop the top ->", top.event.short())


if __name__ == "__main__":
    main()
