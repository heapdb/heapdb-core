"""A day of meals, a bout of bloating, and learning from feedback.

Run:  python examples/bloating.py

Story: the user starts with the folk prior "beans cause bloating" and only a
mild suspicion of dairy. Each day they eat beans at lunch and have a latte in
the afternoon, then feel bloated around 16:00. HeapDB initially blames the
beans. But the user is in fact lactose intolerant: an elimination test shows
it is the latte, so they confirm the latte and reject the lunch. After a few
days of feedback the ranking flips.

What v0.2 adds to the story:

* Weight is the *dose*: meals carry ``attrs={"portion": ...}`` and the meal
  event type derives its weight from the portion.
* Proportional credit: confirming the latte credits ``dairy`` and ``coffee``
  in proportion to how well each already explains the bloating (EM for
  noisy-OR), instead of v0.1's "+1 to every feature". From day 5 the user
  also has a dairy-free "black coffee + biscotti" snack, which gets
  rejected. The table at the end replays the same history with v0.1's
  uniform credit: there every latte pushed ``coffee`` up exactly as hard as
  ``dairy``, so coffee ends up looking like a real cause.
* Joint cause: while coffee and dairy have only ever appeared together, the
  explanation reports them as ONE cause ``coffee+dairy`` with its own joint
  link, and flags it as inseparable.
* Calibration: every confirmation / rejection records the score the event had
  when it was judged, so we can see whether scores behave like probabilities.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heapdb import HeapDB, LogNormalKernel, format_result  # noqa: E402

H = 3600.0
DAY = 24 * H
BLOATED_AT_HOUR = 16.0

MENU = [
    ("breakfast", 7.5, {"oats", "banana"}, 1.0),
    ("lunch", 13.0, {"beans", "onion", "rice"}, 1.5),
    ("latte", 14.5, {"coffee", "dairy"}, 0.8),
    ("dinner", 19.0, {"pasta", "tomato"}, 1.2),  # after the bloating: ignored
]


def make_db(credit: str = "proportional") -> HeapDB:
    db = HeapDB(bucket_seconds=900)
    # Weight = dose. A meal's weight is its portion size (default 1.0).
    db.register_event_type("meal", weight=lambda attrs: attrs.get("portion", 1.0))
    db.register_effect(
        "bloating",
        kernel=LogNormalKernel(median=2 * H, sigma=0.9),
        priors={"beans": (4, 1), "onion": (2, 2), "dairy": (1, 3)},
        cause_types={"meal"},
        credit=credit,
    )
    return db


def meals_for_day(db: HeapDB, day: int) -> dict[str, int]:
    """Insert one day of meals; return name -> event id."""
    t0 = day * DAY
    return {
        name: db.insert("me", t0 + hour * H, "meal", feats, attrs={"portion": portion}).id
        for name, hour, feats, portion in MENU
    }


def feedback_for_day(db: HeapDB, day: int, ids: dict[str, int], verbose: bool = False) -> None:
    """Daily feedback: latte confirmed, lunch rejected. From day 5 a dairy-free
    black coffee snack is also on the menu (and rejected)."""
    rejected = [ids["lunch"]]
    if day >= 5:
        snack = db.insert("me", day * DAY + 11 * H, "meal", {"coffee", "biscotti"},
                          attrs={"portion": 0.5})
        rejected.append(snack.id)
    rep = db.feedback("me", "bloating", day * DAY + BLOATED_AT_HOUR * H,
                      confirmed_event_id=ids["latte"], rejected_ids=rejected)
    if verbose:
        credit = ", ".join(f"{f} {r:.2f}" for f, r in sorted(rep.credit.items()))
        print(f"  day {day}: confirmed latte -> credit {credit}")


def run_history(db: HeapDB, days: range, verbose: bool = False) -> None:
    for day in days:
        feedback_for_day(db, day, meals_for_day(db, day), verbose)


def main() -> None:
    db = make_db()

    # --- Day 0: first bout of bloating ------------------------------------
    ids = meals_for_day(db, 0)
    at = BLOATED_AT_HOUR * H
    print("=== Day 0, bloated at 16:00 -- before any feedback ===")
    print(format_result(db.top_causes("me", "bloating", at=at, k=3, explain=True)))

    # Elimination test says: it's the latte, not the beans.
    print("\nfeedback: CONFIRMED latte, REJECTED lunch (and the same on days 1-4)")
    print("proportional credit: each feature of the latte gets r_f = s_f / noisyOR")
    feedback_for_day(db, 0, ids, verbose=True)
    run_history(db, range(1, 5), verbose=True)

    # --- Day 5: same menu, bloated again ------------------------------------
    ids = meals_for_day(db, 5)
    at = 5 * DAY + BLOATED_AT_HOUR * H
    print("\n=== Day 5, bloated at 16:00 -- after feedback ===")
    print("(coffee and dairy have only ever arrived together: one joint cause)")
    print(format_result(db.top_causes("me", "bloating", at=at, k=3, explain=True)))

    # --- Days 5-9: a dairy-free coffee snack separates coffee from dairy ---
    print("\nfeedback days 5-9: also a black coffee + biscotti snack, REJECTED")
    feedback_for_day(db, 5, ids)
    run_history(db, range(6, 10))
    meals_for_day(db, 10)
    at = 10 * DAY + BLOATED_AT_HOUR * H
    print("\n=== Day 10 -- coffee has been seen without dairy: separable ===")
    print(format_result(db.top_causes("me", "bloating", at=at, k=2, explain=True)))

    # --- proportional vs uniform credit on the same history -----------------
    uniform = make_db(credit="uniform")
    run_history(uniform, range(0, 10))
    print("\nlearned link strengths, same feedback history (days 0-9):")
    print(f"  {'feature':<9}{'proportional':>24}{'uniform (v0.1)':>24}")
    for f in ("beans", "onion", "rice", "dairy", "coffee", "biscotti"):
        p, u = db.link("bloating", f, "me"), uniform.link("bloating", f, "me")
        print(f"  {f:<9}{p.strength:>6.2f}  Beta({p.alpha:5.2f},{p.beta:5.2f})"
              f"{u.strength:>6.2f}  Beta({u.alpha:5.2f},{u.beta:5.2f})")

    # --- calibration -----------------------------------------------------------
    print()
    print(db.calibration_report("bloating", bins=5))

    top = db.most_likely_cause("me", "bloating", at)
    assert top is not None and top.event.features == {"coffee", "dairy"}
    print("\npop the top ->", top.event.short())


if __name__ == "__main__":
    main()
