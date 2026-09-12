"""The notes store: SQLite, and the only thing here that cannot be rebuilt.

Everything else the logbook shows is derived from scan folders and can be
re-read at any time. What people wrote exists nowhere else, so this module
is the one place in the package where losing state means losing something.

Why SQLite is authoritative and the markdown is the mirror
----------------------------------------------------------
Entries are held twice: as rows here, and as markdown files beside the data
on the share (:mod:`geecs_logbook.mirror`). The database is what the page
reads and writes; the markdown is what stays legible when none of this code
runs.

The obvious alternative — files as truth, this as a rebuildable index —
is genuinely attractive, since drift becomes impossible. It was rejected
for one reason: it puts an SMB write in the save path. When the share is
slow, saving is slow; when the share is *unreachable*, saving fails, and
the sentence someone just typed is the one thing that cannot be
regenerated. Writing here first fails in the right direction, at the cost
of a reconciliation story: ``mirrored_at`` is null until the file lands, so
an entry that has not reached disk is visible and retryable rather than
silently missing.

Concurrency
-----------
Last-writer-wins is not good enough for prose — it silently eats the loser's
paragraph. Every update takes the ``version`` the writer read and fails if
it has moved, so a concurrent edit surfaces as a conflict the UI can show
instead of a sentence that quietly vanished.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from geecs_schemas.log_entry import Attachment, LogEntry

logger = logging.getLogger(__name__)

#: One table, because an entry is one document. The structured half
#: (``payload``) rides as a JSON blob rather than columns: its shape is not
#: settled, and SQLite's ``json_extract`` makes it queryable anyway. Turning
#: a payload field into a column later is additive.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    entry_id     TEXT PRIMARY KEY,
    day          TEXT NOT NULL,
    scan         INTEGER,
    after_scan   INTEGER,
    author       TEXT NOT NULL,
    kind         TEXT NOT NULL,
    status       TEXT NOT NULL,
    template     TEXT NOT NULL,
    body_md      TEXT NOT NULL,
    payload      TEXT,
    attachments  TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL,
    edited_at    TEXT,
    updated_at   TEXT NOT NULL,
    deleted_at   TEXT,
    version      INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL DEFAULT 1,
    mirrored_at  TEXT,
    mirror_attempted_at TEXT
);
"""

#: Indexes are created after the column migration below, since one of them
#: is on a column an older file will not have yet.
_INDEXES = """
CREATE INDEX IF NOT EXISTS entries_by_day ON entries (day);
CREATE INDEX IF NOT EXISTS entries_by_updated ON entries (updated_at);
CREATE INDEX IF NOT EXISTS entries_unmirrored ON entries (mirrored_at)
    WHERE mirrored_at IS NULL;
"""

#: Columns added after the table first shipped, with the expression that
#: fills them for rows that predate them. ``CREATE TABLE IF NOT EXISTS`` is
#: a no-op on an existing file, so additive columns arrive through here —
#: the whole migration story this store needs, and deliberately no more.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("updated_at", "TEXT", "COALESCE(edited_at, created_at)"),
    ("deleted_at", "TEXT", "NULL"),
    ("mirror_attempted_at", "TEXT", "NULL"),
)


class ConflictError(RuntimeError):
    """Raised when an update's expected version no longer matches.

    Carries the current entry so a caller can show what it lost to rather
    than only that it lost.
    """

    def __init__(self, current: LogEntry) -> None:
        super().__init__(
            f"entry {current.entry_id} was saved by someone else "
            f"(now at version {current.version})"
        )
        self.current = current


def _now() -> datetime:
    """Return an aware UTC timestamp.

    Aware, because entries outlive the process that wrote them and a naive
    timestamp is ambiguous the moment anyone reads it from another zone.
    """
    return datetime.now(timezone.utc)


class NotesStore:
    """Entries for one experiment, in one SQLite file.

    Parameters
    ----------
    path : Path or str
        The database file. Its parent must exist — this class will create
        the file but never a directory tree, so a mistyped path fails
        loudly instead of quietly writing somewhere nobody looks.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.parent.is_dir():
            raise NotADirectoryError(
                f"notes store directory does not exist: {self.path.parent}"
            )
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            present = {r["name"] for r in conn.execute("PRAGMA table_info(entries)")}
            for name, sql_type, fill in _ADDED_COLUMNS:
                if name in present:
                    continue
                conn.execute(f"ALTER TABLE entries ADD COLUMN {name} {sql_type}")
                conn.execute(f"UPDATE entries SET {name} = {fill}")
                logger.info("notes store: added column %s", name)
            conn.executescript(_INDEXES)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection, committing on success and closing always."""
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            # WAL lets the day view read while someone is saving; without it
            # a single writer blocks every reader on the page.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    # ---------------------------------------------------------------- read

    def get(
        self, entry_id: str, *, include_deleted: bool = False
    ) -> Optional[LogEntry]:
        """Return one entry, or ``None`` when it does not exist.

        A tombstoned entry counts as not existing unless ``include_deleted``
        is set — the default is what every write path wants, so that
        editing a deleted entry fails the same way as editing a missing one.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
        if row is None:
            return None
        entry = _from_row(row)
        if entry.is_deleted and not include_deleted:
            return None
        return entry

    def for_day(self, day: str) -> list[LogEntry]:
        """Return every live entry for one ``YYYY-MM-DD``, oldest first.

        Ordered by creation so a scan's notes read in the order they were
        written, which is how a conversation reads. Deleted entries are
        not listed.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM entries WHERE day = ? AND deleted_at IS NULL"
                " ORDER BY created_at, rowid",
                (day,),
            ).fetchall()
        return [_from_row(row) for row in rows]

    def unmirrored(self, limit: int = 100) -> list[LogEntry]:
        """Return entries whose markdown does not match the share yet.

        The reconciliation half of writing here first: a save never fails
        because the share is down, but something has to notice and retry.
        Deleted entries are included — their owed operation is removing the
        file, and :func:`geecs_logbook.mirror.sync` knows which is which.

        Never-tried entries come first, then the least recently tried: an
        entry whose day folder never appears (a note on a day with no
        scans) must not sit at the head of the queue forever and starve
        the ones behind it. Callers record each failed try with
        :meth:`mark_deferred`.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM entries WHERE mirrored_at IS NULL "
                "ORDER BY mirror_attempted_at IS NOT NULL, mirror_attempted_at,"
                " created_at LIMIT ?",
                (limit,),
            ).fetchall()
        return [_from_row(row) for row in rows]

    # --------------------------------------------------------------- write

    def create(
        self,
        *,
        day: str,
        author: str,
        body_md: str,
        scan: Optional[int] = None,
        after: Optional[int] = None,
        kind: str = "note",
        status: str = "kept",
        template: str = "blank",
        payload: Optional[dict] = None,
    ) -> LogEntry:
        """Store a new entry and return it.

        Raises
        ------
        ValueError
            If both ``scan`` and ``after`` are given — an entry is anchored
            to a scan, to the gap after one, or (neither) to the day. Also
            if an agent's entry arrives already ``kept``: only a human keeps
            an agent's draft, and that has to hold at the API, not in the
            documentation.
        """
        if scan is not None and after is not None:
            raise ValueError("an entry is anchored to a scan or after one, not both")
        if kind != "note" and status == "kept":
            raise ValueError(
                f"a {kind} entry is created as a draft; a person keeps it afterwards"
            )

        now = _now()
        entry = LogEntry(
            entry_id=uuid.uuid4().hex[:12],
            day=day,
            scan=scan,
            after=after,
            author=author,
            kind=kind,
            status=status,
            template=template,
            body_md=body_md,
            payload=payload,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO entries (entry_id, day, scan, after_scan, author, kind,"
                " status, template, body_md, payload, attachments, created_at,"
                " edited_at, updated_at, deleted_at, version, schema_version,"
                " mirrored_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                _to_params(entry),
            )
        logger.info("entry %s created for %s by %s", entry.entry_id, day, author)
        return entry

    def update(
        self, entry_id: str, *, body_md: str, author: str, expected_version: int
    ) -> LogEntry:
        """Replace an entry's body, if nobody else has saved since.

        Raises
        ------
        KeyError
            If the entry does not exist.
        ConflictError
            If ``expected_version`` is stale. The exception carries the
            current entry so the caller can show what changed.
        """
        current = self.get(entry_id)
        if current is None:
            raise KeyError(entry_id)
        if current.version != expected_version:
            raise ConflictError(current)

        with self._connect() as conn:
            # The WHERE clause repeats the version check: between the read
            # above and this write another process may have saved, and the
            # comparison has to happen where the write happens.
            now = _now().isoformat()
            cursor = conn.execute(
                "UPDATE entries SET body_md = ?, author = ?, edited_at = ?,"
                " updated_at = ?, version = version + 1, mirrored_at = NULL"
                " WHERE entry_id = ? AND version = ? AND deleted_at IS NULL",
                (body_md, author, now, now, entry_id, expected_version),
            )
            if cursor.rowcount == 0:
                raise ConflictError(self.get(entry_id) or current)
        return self.get(entry_id)  # type: ignore[return-value]

    def set_status(self, entry_id: str, status: str) -> LogEntry:
        """Promote a draft to kept, or demote it.

        Separate from :meth:`update` because promoting an agent's draft is
        a human act on unchanged text, and conflating it with an edit would
        let an agent promote its own output by rewriting it.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE entries SET status = ?, updated_at = ?, version = version + 1,"
                " mirrored_at = NULL WHERE entry_id = ? AND deleted_at IS NULL",
                (status, _now().isoformat(), entry_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(entry_id)
        return self.get(entry_id)  # type: ignore[return-value]

    def add_attachment(self, entry_id: str, attachment: Attachment) -> LogEntry:
        """Record a stored file against an entry.

        One statement, appending in SQL: two uploads landing at once must
        both reach the manifest, and a read-modify-write here would let the
        second overwrite the first — a file on disk that no manifest names.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE entries SET"
                " attachments = json_insert(attachments, '$[#]', json(?)),"
                " updated_at = ?, version = version + 1, mirrored_at = NULL"
                " WHERE entry_id = ? AND deleted_at IS NULL",
                (
                    json.dumps(attachment.model_dump(mode="json")),
                    _now().isoformat(),
                    entry_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(entry_id)
        return self.get(entry_id)  # type: ignore[return-value]

    def mark_mirrored(
        self,
        entry_id: str,
        when: Optional[datetime] = None,
        *,
        version: Optional[int] = None,
    ) -> bool:
        """Record that an entry's markdown reached the share.

        With ``version``, only if the entry is still at that version: a
        writer that mirrored what it read must not mark an edit that landed
        in between as done. Returns whether the mark was applied.
        """
        with self._connect() as conn:
            if version is None:
                cursor = conn.execute(
                    "UPDATE entries SET mirrored_at = ? WHERE entry_id = ?",
                    ((when or _now()).isoformat(), entry_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE entries SET mirrored_at = ?"
                    " WHERE entry_id = ? AND version = ?",
                    ((when or _now()).isoformat(), entry_id, version),
                )
        return cursor.rowcount > 0

    def mark_deferred(self, entry_id: str) -> None:
        """Record a failed mirror attempt, so the queue rotates past it."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE entries SET mirror_attempted_at = ? WHERE entry_id = ?",
                (_now().isoformat(), entry_id),
            )

    def delete(self, entry_id: str) -> bool:
        """Tombstone an entry. Returns whether a live one was found.

        The row stays, with ``deleted_at`` set: every listing hides it, but
        a mirror or a downstream copy can still learn that it went, and an
        accidental delete is recoverable. The markdown file is the
        caller's to clean up — this module never touches the share, so
        that a database operation can never be the thing that loses a
        file — and ``mirrored_at`` is cleared so :func:`sync` owes the
        removal until that happens.
        """
        with self._connect() as conn:
            now = _now().isoformat()
            cursor = conn.execute(
                "UPDATE entries SET deleted_at = ?, updated_at = ?,"
                " version = version + 1, mirrored_at = NULL"
                " WHERE entry_id = ? AND deleted_at IS NULL",
                (now, now, entry_id),
            )
        return cursor.rowcount > 0


def _to_params(entry: LogEntry) -> tuple:
    """Flatten an entry into INSERT parameters, in column order."""
    return (
        entry.entry_id,
        entry.day,
        entry.scan,
        entry.after,
        entry.author,
        entry.kind,
        entry.status,
        entry.template,
        entry.body_md,
        json.dumps(entry.payload.model_dump(mode="json")) if entry.payload else None,
        json.dumps([a.model_dump(mode="json") for a in entry.attachments]),
        entry.created_at.isoformat(),
        entry.edited_at.isoformat() if entry.edited_at else None,
        entry.updated_at.isoformat(),
        entry.deleted_at.isoformat() if entry.deleted_at else None,
        entry.version,
        entry.schema_version,
    )


def _from_row(row: sqlite3.Row) -> LogEntry:
    """Rebuild an entry from a database row."""
    return LogEntry.model_validate(
        {
            "schema_version": row["schema_version"],
            "entry_id": row["entry_id"],
            "day": row["day"],
            "scan": row["scan"],
            "after": row["after_scan"],
            "author": row["author"],
            "kind": row["kind"],
            "status": row["status"],
            "template": row["template"],
            "body_md": row["body_md"],
            "payload": json.loads(row["payload"]) if row["payload"] else None,
            "attachments": json.loads(row["attachments"] or "[]"),
            "created_at": row["created_at"],
            "edited_at": row["edited_at"],
            "updated_at": row["updated_at"],
            "deleted_at": row["deleted_at"],
            "version": row["version"],
        }
    )
