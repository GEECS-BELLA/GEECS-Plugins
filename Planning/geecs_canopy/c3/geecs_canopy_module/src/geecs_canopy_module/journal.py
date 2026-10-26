"""The command journal: one row per put attempt, whatever its outcome.

A SQLite file for the spike. In Canopy proper this would be a ``geecs``
schema in Canopy's Postgres through ``canopy.core.db`` (``Base`` with
``__table_args__ = {"schema": "geecs"}``), with its own Alembic revisions, as
Squirrel and Magpie keep theirs.
"""

from __future__ import annotations

import json
import sqlite3
import time

_COLUMNS = tuple(
    "time operator key_name device role axis value outcome canopy_code detail".split()
)

_SCHEMA = f"CREATE TABLE IF NOT EXISTS journal (id INTEGER PRIMARY KEY AUTOINCREMENT, {', '.join(_COLUMNS)})"


class Journal:
    """Append-only put journal in one SQLite file."""

    def __init__(self, path: str) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute(_SCHEMA)

    def record(self, **row: object) -> int:
        """Append one row (``time`` is filled in here) and return its id."""
        row = {**row, "time": time.time(), "value": json.dumps(row.get("value"))}
        cursor = self._db.execute(
            f"INSERT INTO journal ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
            [row.get(c) for c in _COLUMNS],
        )
        return int(cursor.lastrowid)

    def rows(self) -> list[dict[str, object]]:
        """Return every row, oldest first."""
        cursor = self._db.execute(
            f"SELECT id, {', '.join(_COLUMNS)} FROM journal ORDER BY id"
        )
        return [dict(zip(("id", *_COLUMNS), r)) for r in cursor.fetchall()]
