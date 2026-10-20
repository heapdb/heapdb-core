"""The HeapDB query language (parser and executor in the Rust engine).

Grammar (keywords are case-insensitive; ``[...]`` is optional)::

    TOP <k> CAUSES OF <effect> FOR <entity>[,<entity>...] [AT <time>]
        [WITHIN <duration>] [EXPLAIN]
    CALIBRATION OF <effect> [BINS <n>]
    RISK OF <effect> FOR <entity> FEATURES <f>[,<f>...] [WEIGHT <w>] [TYPE <t>]
        [ATTRS k=v[,k=v...]] [EXPLAIN]
    FEEDBACK <effect> FOR <entity> AT <time> CONFIRMED <event_id>
             [REJECTED <id>[,<id>...]]
    INSERT <type> FOR <entity> [AT <time>] [FEATURES <f>[,<f>...]] [WEIGHT <w>]
    OBSERVE <effect> FOR <entity> [AT <time>] [LEAK <x>]
    SETTLE [<effect>] UNTIL <time> [FOR <entity>]
    LINKS OF <effect> [FOR <entity>] [MIN <n>]
"""

from typing import Union

from ._core import (
    NOW,
    CalibrationQuery,
    FeedbackStmt,
    InsertStmt,
    LinksQuery,
    ObserveStmt,
    QueryError,
    RiskQuery,
    SettleStmt,
    TimeExpr,
    TopCausesQuery,
    execute,
    format_result,
    parse,
    parse_duration,
    parse_time,
)

Statement = Union[TopCausesQuery, FeedbackStmt, InsertStmt, CalibrationQuery, RiskQuery,
                  ObserveStmt, SettleStmt, LinksQuery]

__all__ = [
    "NOW", "CalibrationQuery", "FeedbackStmt", "InsertStmt", "LinksQuery", "ObserveStmt",
    "QueryError", "RiskQuery", "SettleStmt", "Statement", "TimeExpr", "TopCausesQuery",
    "execute", "format_result", "parse", "parse_duration", "parse_time",
]
