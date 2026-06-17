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


def build_signal_attrs(variable_list: list[str]) -> list[tuple[str, str]]:
    """Map each GEECS variable to a unique ophyd-async attribute name.

    Applies :func:`safe_name` to every variable and disambiguates collisions by
    appending ``_2``, ``_3``, … in input order.  Centralises the attr-naming
    loop the detector classes share so signal creation and the legacy
    ``device variable`` header map (see each device's ``_column_headers``)
    cannot drift.

    Parameters
    ----------
    variable_list:
        GEECS variable names, in the order they should be exposed.

    Returns
    -------
    list[tuple[str, str]]
        ``(attr, variable)`` pairs, one per input variable, ``attr`` unique.
    """
    used_attrs: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for var in variable_list:
        attr = safe_name(var)
        if attr in used_attrs:
            i = 2
            while f"{attr}_{i}" in used_attrs:
                i += 1
            attr = f"{attr}_{i}"
        used_attrs.add(attr)
        pairs.append((attr, var))
    return pairs
