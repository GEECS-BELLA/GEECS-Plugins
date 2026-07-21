"""Shared utilities for GeecsBluesky."""

from __future__ import annotations

from geecs_ca_gateway.pv_naming import normalize_component


def safe_name(s: str) -> str:
    """Convert an arbitrary string to a valid Python/ophyd-async identifier.

    Delegates to the shared naming contract
    (:func:`geecs_ca_gateway.pv_naming.normalize_component` — runs of
    non-alphanumeric characters collapse to one underscore, lowercase), so a
    GEECS name mangles identically whether it becomes a PV component or an
    event-column component.  Returns ``"var"`` for strings that reduce to
    empty (an identifier, unlike a PV component, must be non-empty).
    """
    return normalize_component(s) or "var"
