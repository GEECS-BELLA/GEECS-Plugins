"""The put journal: every write the bridge answered, who asked, and how it ended.

One row per put (accepted, refused or denied) and per run-lock release, as
:class:`~geecs_panels.contracts.JournalEntry`. It is the experiment home's
"last commands" list (``GET /api/journal``) and the room's who-did-what,
which the LabVIEW room never had.

**SQLite, not append-only JSONL**: the id is the database's own increasing
row id (``journal_id`` in every ``PutResult``, from 1), a write is one
atomic transaction (a crash mid-write leaves no half line for a reader to
skip), and "the last N" is an indexed query rather than a file read
backwards; the logbook keeps its entries the same way (``logbook.db`` in
its ``StateDirectory``). The file is ``$STATE_DIRECTORY/journal.db`` under
the unit (the logbook's pattern) or ``--journal PATH``; without either the
journal is in memory for the life of the process (a laptop's prototype),
and the log says so.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from geecs_panels.contracts import JournalEntry, PutOutcome, Severity, Value

logger = logging.getLogger(__name__)

#: The journal's file name in the state directory.
JOURNAL_FILENAME = "journal.db"
#: The most entries one ``GET /api/journal`` returns.
MAX_ENTRIES = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS puts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time REAL NOT NULL,
    operator TEXT NOT NULL,
    device TEXT NOT NULL,
    role TEXT NOT NULL,
    axis TEXT,
    value TEXT,
    outcome TEXT NOT NULL,
    severity TEXT,
    message TEXT NOT NULL,
    snapshot TEXT
)
"""

#: Columns added after the first journals were written, added to an older file.
_ADDED_COLUMNS = {"snapshot": "TEXT"}


def default_path() -> Optional[Path]:
    """``$STATE_DIRECTORY/journal.db`` when systemd gives the unit one, else ``None`` (in memory)."""
    state = os.environ.get("STATE_DIRECTORY")
    if not state:
        return None
    return Path(state.split(":")[0]) / JOURNAL_FILENAME


class Journal:
    """The journal's store: append one put, read the last N."""

    def __init__(
        self,
        path: Optional[Path] = None,
        wall: Callable[[], float] = time.time,
        *,
        warn: bool = True,
    ) -> None:
        if path is None and warn:
            logger.warning(
                "the put journal is in memory: no --journal and no $STATE_DIRECTORY"
            )
        self.path = path
        self._wall = wall
        self._lock = threading.Lock()
        self._db = sqlite3.connect(
            ":memory:" if path is None else str(path), check_same_thread=False
        )
        with self._db:
            self._db.execute(_SCHEMA)
            have = {row[1] for row in self._db.execute("PRAGMA table_info(puts)")}
            for column, kind in _ADDED_COLUMNS.items():
                if column not in have:  # a journal from before the column
                    self._db.execute(f"ALTER TABLE puts ADD COLUMN {column} {kind}")

    def record(
        self,
        *,
        operator: str,
        device: str,
        role: str,
        axis: Optional[str],
        asked: Value,
        outcome: PutOutcome,
        severity: Optional[Severity] = None,
        message: str = "",
        snapshot: Optional[str] = None,
    ) -> JournalEntry:
        """Write one put as answered; the entry's id is the result's ``journal_id``.

        *snapshot* is the id of the snapshot a restore wrote the row from.
        """
        now = self._wall()
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO puts (time, operator, device, role, axis, value, outcome,"
                " severity, message, snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now,
                    operator,
                    device,
                    role,
                    axis,
                    None if asked is None else json.dumps(asked),
                    outcome,
                    severity,
                    message,
                    snapshot,
                ),
            )
            row_id = cursor.lastrowid
        assert row_id is not None
        return JournalEntry(
            id=row_id,
            time=now,
            operator=operator,
            device=device,
            role=role,
            axis=axis,
            value=asked,
            outcome=outcome,
            severity=severity,
            message=message,
            snapshot=snapshot,
        )

    def last(self, entries: int) -> list[JournalEntry]:
        """The last *entries* puts, newest first (at most :data:`MAX_ENTRIES`)."""
        count = max(0, min(entries, MAX_ENTRIES))
        with self._lock:
            rows = self._db.execute(
                "SELECT id, time, operator, device, role, axis, value, outcome,"
                " severity, message, snapshot FROM puts ORDER BY id DESC LIMIT ?",
                (count,),
            ).fetchall()
        return [
            JournalEntry(
                id=r[0],
                time=r[1],
                operator=r[2],
                device=r[3],
                role=r[4],
                axis=r[5],
                # The value keeps its JSON type: 1, "1" and true stay apart.
                value=None if r[6] is None else json.loads(r[6]),
                outcome=r[7],
                severity=r[8],
                message=r[9],
                snapshot=r[10],
            )
            for r in rows
        ]

    def close(self) -> None:
        """Close the database."""
        self._db.close()
