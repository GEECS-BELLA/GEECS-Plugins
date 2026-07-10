"""Shared wire-value coercion for the GEECS transport layers."""

from __future__ import annotations

from typing import Any


def coerce_scalar(s: str) -> Any:
    """Best-effort numeric conversion; non-numeric text passes through as-is.

    Lossy for text that merely *looks* numeric (``'007'`` → ``7``) — string-typed
    variables must bypass this (the subscriber's ``text_variables`` parameter).
    """
    try:
        f = float(s)
        return int(f) if f == int(f) and "." not in s else f
    except ValueError:
        return s
