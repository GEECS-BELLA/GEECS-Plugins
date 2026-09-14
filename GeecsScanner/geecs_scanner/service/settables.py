"""Every numeric settable of the experiment, alias-first — the movable panel's list.

The filter and the order live in GEECS-Core (:func:`geecs_core.db.numeric_settables`,
one list for every picker); this module only fetches the rows from the DB
and wraps them as the API's model.

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
    """The core's alias-first numeric settables, as the API's model."""
    from dataclasses import asdict

    from geecs_core.db import numeric_settables

    return [SettableOut(**asdict(row)) for row in numeric_settables(rows_by_device)]


class DbSettables:
    """The experiment's numeric settables from the GEECS DB, read once and kept."""

    def __init__(self, experiment: str, *, db: Any = None) -> None:
        self._experiment = experiment
        self._db = db  # the GeecsDb class; tests inject a double
        self._cache: list[SettableOut] | None = None
        self._lock = threading.Lock()

    def settables(self) -> SettablesOut:
        """The cached list, or one DB read to fill it; a failed read is reported, not cached."""
        with self._lock:
            if self._cache is not None:
                return SettablesOut(items=self._cache, source="db")
            try:
                db = self._db
                if db is None:
                    from geecs_core.db import GeecsDb

                    db = GeecsDb
                rows = db.get_experiment_device_variables(self._experiment)
                items = build_settables(rows)
            except Exception as exc:  # noqa: BLE001 — the DB is a remote service; say so, retry next time
                logger.warning("settables unavailable from the GEECS DB: %s", exc)
                return SettablesOut(items=[], source="db", detail=f"GEECS DB: {exc}")
            self._cache = items
            return SettablesOut(items=items, source="db")
