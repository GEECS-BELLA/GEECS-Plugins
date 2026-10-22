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
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

from geecs_scanner.service.models import SettableOut, SettablesOut

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


class CachedDbRead(Generic[T]):
    """One GEECS DB read, kept after the first success; a failure is reported, not kept.

    Shared by every DB-backed list the page shows (the settables, the
    scan-log picker), so the caching and failure rule exist once.
    """

    def __init__(self, read: Callable[[Any], T], *, what: str, db: Any = None) -> None:
        self._read = read
        self._what = what
        self._db = db  # the GeecsDb class; tests inject a double
        self._cache: T | None = None
        self._lock = threading.Lock()

    def get(self) -> tuple[T | None, str]:
        """``(value, "")`` — or ``(None, why)`` when the DB did not answer."""
        with self._lock:
            if self._cache is not None:
                return self._cache, ""
            try:
                db = self._db
                if db is None:
                    from geecs_core.db import GeecsDb

                    db = GeecsDb
                value = self._read(db)
            except Exception as exc:  # noqa: BLE001 — the DB is a remote service; say so, retry next time
                logger.warning("%s unavailable from the GEECS DB: %s", self._what, exc)
                return None, f"GEECS DB: {exc}"
            self._cache = value
            return value, ""


class DbSettables:
    """The experiment's numeric settables from the GEECS DB, read once and kept."""

    def __init__(self, experiment: str, *, db: Any = None) -> None:
        self._read = CachedDbRead(
            lambda db: build_settables(db.get_experiment_device_variables(experiment)),
            what="settables",
            db=db,
        )

    def settables(self) -> SettablesOut:
        """The cached list, or one DB read to fill it; a failed read is reported, not cached."""
        items, detail = self._read.get()
        return SettablesOut(items=items or [], source="db", detail=detail)
