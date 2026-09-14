"""Every numeric settable of the experiment, alias-first — the movable panel's list.

The scan-variable catalog (``scan_variables.yaml``) keeps only what has no
free equivalent: pseudo axes, ``confirm`` overlays, setpoint opt-outs.  Any
numeric settable ``Device:Variable`` is movable, and the shorthand comes
from the DB, not a config: the per-instance ``variable.alias`` the DB
curates.  The list shows aliased variables first (alphabetical by alias),
then every remaining numeric settable by canonical name; each row carries
the alias *beside* the canonical ``Device:Variable``, never instead of it —
the request stores the canonical name, so a rename in the DB breaks nothing.

The DB roster is read once per process and kept: devices and their
settables change with a DB edit, and the CA gateway that serves them is
restarted for that anyway (its roster is startup-only), so the scanner
follows the same rule rather than polling the DB behind every page load.
A failed read is not cached — the next request tries again.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from geecs_scanner.service.models import SettableOut, SettablesOut

logger = logging.getLogger(__name__)


class SettablesSource(Protocol):
    """Where the list comes from: the DB in production, a fixed list in demo."""

    def settables(self) -> SettablesOut:
        """The list, alias-first, with where it came from."""


def build_settables(
    rows_by_device: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[SettableOut]:
    """Keep the numeric settables of every device and order them alias-first.

    Parameters
    ----------
    rows_by_device : mapping
        ``{device: [variable metadata, ...]}`` as
        :meth:`geecs_core.db.GeecsDb.get_experiment_device_variables`
        returns it — each row a dict with ``name``, ``settable``,
        ``variabletype``, ``choices``, ``units``, ``min``, ``max``, ``alias``.
    """
    from geecs_core.db.variable_types import effective_vartype

    out: list[SettableOut] = []
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
                SettableOut(
                    name=f"{device}:{variable}",
                    device=device,
                    variable=variable,
                    alias=str(row.get("alias") or "").strip(),
                    units=str(row.get("units") or "").strip(),
                    min=row.get("min"),
                    max=row.get("max"),
                )
            )
    # aliased first, alphabetical by alias; then the rest by canonical name
    out.sort(
        key=lambda s: (0, s.alias.lower(), s.name.lower())
        if s.alias
        else (1, s.name.lower(), "")
    )
    return out


class DbSettables:
    """The experiment's numeric settables from the GEECS DB, read once and kept."""

    def __init__(self, experiment: str) -> None:
        self._experiment = experiment
        self._cache: list[SettableOut] | None = None
        self._lock = threading.Lock()

    def settables(self) -> SettablesOut:
        """The cached list, or one DB read to fill it; a failed read is reported, not cached."""
        with self._lock:
            if self._cache is not None:
                return SettablesOut(items=self._cache, source="db")
            try:
                from geecs_core.db import GeecsDb

                rows = GeecsDb.get_experiment_device_variables(self._experiment)
                items = build_settables(rows)
            except Exception as exc:  # noqa: BLE001 — the DB is a remote service; say so, retry next time
                logger.warning("settables unavailable from the GEECS DB: %s", exc)
                return SettablesOut(items=[], source="db", detail=f"GEECS DB: {exc}")
            self._cache = items
            return SettablesOut(items=items, source="db")
