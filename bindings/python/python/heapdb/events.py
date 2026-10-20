"""Core event record (re-exported from the Rust engine)."""

from ._core import Event, as_feature_set, format_time, validate_weight

__all__ = ["Event", "as_feature_set", "format_time", "validate_weight"]
