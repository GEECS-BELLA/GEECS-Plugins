"""The experiment's numeric settables, alias-first — the list every movable picker shows.

Post-#779 any numeric settable ``Device:Variable`` is movable and
scannable; the scan-variable catalog keeps only what has no free
equivalent (pseudo axes, ``confirm`` overlays, opt-outs).  The shorthand
comes from the DB, not a config: the per-instance ``variable.alias`` it
curates.  This module turns :meth:`GeecsDb.get_experiment_device_variables`
rows into that list — aliased variables first (alphabetical by alias), then
every remaining numeric settable by canonical name — so the web scanner,
the scan MCP and any later picker agree on one ordering and one filter
(``settable`` and :func:`effective_vartype` ``numeric``, the rule the CA
gateway types its PVs with).  The alias rides *beside* the canonical name,
never instead of it: requests store ``Device:Variable``, so a rename in the
DB breaks nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from geecs_core.db.variable_types import effective_vartype


@dataclass(frozen=True)
class NumericSettable:
    """One numeric settable as a picker lists it."""

    name: str  #: the canonical ``Device:Variable`` — what a request stores
    device: str
    variable: str
    alias: str = ""  #: the DB's curated short name; ``""`` when none
    units: str = ""
    min: Optional[float] = None
    max: Optional[float] = None


def numeric_settables(
    rows_by_device: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[NumericSettable]:
    """Keep the numeric settables of every device and order them alias-first.

    Parameters
    ----------
    rows_by_device : mapping
        ``{device: [variable metadata, ...]}`` as
        :meth:`GeecsDb.get_experiment_device_variables` returns it — each
        row a dict with ``name``, ``settable``, ``variabletype``,
        ``choices``, ``units``, ``min``, ``max``, ``alias``.

    Returns
    -------
    list of NumericSettable
        Aliased entries first, alphabetical by alias (case-insensitive),
        then the unaliased by canonical name.
    """
    out: list[NumericSettable] = []
    for device, rows in rows_by_device.items():
        for row in rows:
            if not row.get("settable"):
                continue
            if (
                effective_vartype(row.get("variabletype"), row.get("choices"))
                != "numeric"
            ):
                continue
            variable = str(row.get("name") or "").strip()
            if not variable:
                continue
            out.append(
                NumericSettable(
                    name=f"{device}:{variable}",
                    device=device,
                    variable=variable,
                    alias=str(row.get("alias") or "").strip(),
                    units=str(row.get("units") or "").strip(),
                    min=row.get("min"),
                    max=row.get("max"),
                )
            )
    out.sort(
        key=lambda s: (0, s.alias.lower(), s.name.lower())
        if s.alias
        else (1, s.name.lower(), "")
    )
    return out
