"""The command journal: one ``JournalEntry`` per write attempt, whatever its outcome.

SQLite in the module's state directory for push 1 (one row = the entry's
JSON, so the GEECS-Schemas model is the one contract); the Postgres path, a
``geecs`` schema with its own Alembic tree, is DEPLOYMENT.md's, not built.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from geecs_schemas import JournalEntry

_SCHEMA = "CREATE TABLE IF NOT EXISTS journal (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"


class Journal:
    """Append-only, in one SQLite file."""

    def __init__(self, path: Path) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute(_SCHEMA)
        self._lock = threading.Lock()

    def record(self, entry: JournalEntry) -> JournalEntry:
        """Append ``entry`` and return it with its row number."""
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO journal (entry) VALUES (?)",
                (entry.model_dump_json(exclude={"id"}),),
            )
        return entry.model_copy(update={"id": cursor.lastrowid})

    def last(self, entries: int) -> list[JournalEntry]:
        """The newest ``entries`` rows, oldest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, entry FROM journal ORDER BY id DESC LIMIT ?", (entries,)
            ).fetchall()
        return [
            JournalEntry.model_validate_json(text).model_copy(update={"id": row_id})
            for row_id, text in reversed(rows)
        ]
