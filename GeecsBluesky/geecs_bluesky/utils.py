"""Shared utilities for GeecsBluesky."""

from __future__ import annotations

import re


def safe_name(s: str) -> str:
    """Convert an arbitrary string to a valid Python/ophyd-async identifier.

    Strips or replaces any character that is not alphanumeric or underscore,
    collapses leading/trailing underscores, and lower-cases the result.
    Returns ``"var"`` for strings that reduce to empty.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_").lower()
    return cleaned or "var"
