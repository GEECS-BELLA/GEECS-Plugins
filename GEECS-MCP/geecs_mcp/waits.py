"""The one budget rule for the wait verbs of every domain."""

from __future__ import annotations

from typing import Any

#: Server-side cap on how long one wait call may block — pending the
#: OSPREY team's word on tool-call budgets (DESIGN.md §4.3).
MAX_WAIT_S = 600.0


def bounded_wait(timeout_s: Any) -> float:
    """``timeout_s`` as a float clamped to ``[0, MAX_WAIT_S]``; garbage is 0."""
    try:
        value = float(timeout_s)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(value, MAX_WAIT_S))
