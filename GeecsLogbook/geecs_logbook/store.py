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

import base64
import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from geecs_schemas.log_entry import Attachment, LogEntry

from geecs_logbook.tags import parse_tags

logger = logging.getLogger(__name__)

#: Every entry a request may ask for at once. A month of a busy ops book
#: is a few hundred; this is a guard against an unbounded range, not a
#: page size.
_QUERY_CAP = 2000

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
    edited_by    TEXT,
    updated_at   TEXT NOT NULL,
    deleted_at   TEXT,
    version      INTEGER NOT NULL DEFAULT 1,
    schema_version INTEGER NOT NULL DEFAULT 1,
    mirrored_at  TEXT,
    mirror_attempted_at TEXT,
    book         TEXT NOT NULL DEFAULT 'scans',
    tags         TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS entry_history (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id     TEXT NOT NULL,
    version      INTEGER NOT NULL,
    reason       TEXT NOT NULL,
    recorded_at  TEXT NOT NULL,
    snapshot     TEXT NOT NULL
);
"""

#: Indexes are created after the column migration below, since one of them
#: is on a column an older file will not have yet.
_INDEXES = """
CREATE INDEX IF NOT EXISTS entries_by_day ON entries (day);
CREATE INDEX IF NOT EXISTS entries_by_book_day ON entries (book, day);
CREATE INDEX IF NOT EXISTS entries_by_updated ON entries (updated_at);
CREATE INDEX IF NOT EXISTS history_by_entry ON entry_history (entry_id, seq);
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
    ("edited_by", "TEXT", "NULL"),
    ("book", "TEXT NOT NULL DEFAULT 'scans'", "'scans'"),
    ("tags", "TEXT NOT NULL DEFAULT '[]'", "'[]'"),
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


@dataclass(frozen=True)
class HistoryRecord:
    """One earlier state of an entry, kept before it was changed.

    ``entry`` is the entry exactly as it was; ``reason`` is what happened
    next to it — ``edit``, ``status``, ``attach`` or ``delete``. Undo is a
    new edit using an old body; nothing here rewrites history.
    """

    version: int
    reason: str
    recorded_at: datetime
    entry: LogEntry


@dataclass(frozen=True)
class ChangePage:
    """One page of the change feed: entries in ``updated_at`` order and a cursor.

    ``next_cursor`` is ``None`` when the page was not full; otherwise pass
    it back as ``cursor`` to continue exactly where this page stopped,
    ties in ``updated_at`` included (the cursor carries the row too).
    """

    entries: list[LogEntry]
    next_cursor: Optional[str]


def _now() -> datetime:
    """Return an aware UTC timestamp.

    Aware, because entries outlive the process that wrote them and a naive
    timestamp is ambiguous the moment anyone reads it from another zone.
    """
    return datetime.now(timezone.utc)


def _stamp(when: datetime) -> str:
    """Render an aware datetime the way the store's columns hold them.

    Columns are ``datetime.isoformat()`` of a UTC-aware value, so a
    comparison against a value rendered the same way is a correct string
    comparison. A naive datetime is refused: the caller's zone is unknown.
    """
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return when.astimezone(timezone.utc).isoformat()


def _encode_cursor(stamp: str, rowid: int) -> str:
    """Render a feed cursor: opaque and URL-safe.

    The stamp carries ``+00:00``, which a raw query string turns into a
    space and thereby into a value that sorts *below* every real stamp —
    the boundary row would be re-sent. Base64 (URL alphabet, unpadded)
    keeps the cursor a token a client cannot half-encode.
    """
    raw = f"{stamp}|{rowid}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, int]:
    """Parse a cursor back into ``(stamp, rowid)``; ``ValueError`` if it is not one."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        stamp, _, row = raw.rpartition("|")
        datetime.fromisoformat(stamp)  # a real timestamp, not merely a shape
        if not row.isdigit():
            raise ValueError(row)
    except (ValueError, UnicodeDecodeError) as exc:  # binascii.Error is a ValueError
        raise ValueError(f"malformed cursor: {cursor!r}") from exc
    return stamp, int(row)


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

    def for_day(self, day: str, *, book: Optional[str] = None) -> list[LogEntry]:
        """Return every live entry for one ``YYYY-MM-DD``, oldest first.

        Ordered by creation so a scan's notes read in the order they were
        written, which is how a conversation reads. Deleted entries are
        not listed. ``book`` narrows to one book; the default is both.
        """
        sql = "SELECT * FROM entries WHERE day = ? AND deleted_at IS NULL"
        params: list[object] = [day]
        if book is not None:
            sql += " AND book = ?"
            params.append(book)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY created_at, rowid", params).fetchall()
        return [_from_row(row) for row in rows]

    def query(
        self,
        *,
        day_from: str,
        day_to: str,
        book: Optional[str] = None,
        tag: Optional[str] = None,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        author: Optional[str] = None,
        include_scan_anchored: bool = True,
        limit: int = _QUERY_CAP,
    ) -> list[LogEntry]:
        """Return live entries in a day range, oldest first, filtered.

        The one question every reader asks with different parameters: the
        month view (a book, a range, a tag), a search, and later a
        synchroniser. ``day_from``/``day_to`` are inclusive ``YYYY-MM-DD``.
        ``include_scan_anchored=False`` drops entries on or after a scan,
        which is how the month page hides the campaign record by default.
        """
        sql = "SELECT * FROM entries WHERE day BETWEEN ? AND ? AND deleted_at IS NULL"
        params: list[object] = [day_from, day_to]
        for column, value in (
            ("book", book),
            ("kind", kind),
            ("status", status),
            ("author", author),
        ):
            if value is not None:
                sql += f" AND {column} = ?"
                params.append(value)
        if tag is not None:
            sql += " AND EXISTS (SELECT 1 FROM json_each(entries.tags) WHERE value = ?)"
            params.append(tag.lower())
        if not include_scan_anchored:
            sql += " AND scan IS NULL AND after_scan IS NULL"
        if limit < 1:
            raise ValueError("limit must be at least 1")
        sql += " ORDER BY day, created_at, rowid LIMIT ?"
        params.append(min(limit, _QUERY_CAP))
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_from_row(row) for row in rows]

    def count_by_day(self, day_from: str, day_to: str) -> dict[str, dict[str, int]]:
        """Return live entry counts per day and book in an inclusive range.

        One grouped query for a whole month — what a calendar needs to
        mark the days that have notes, without loading a single body.
        The result maps ``YYYY-MM-DD`` to ``{book: count}`` and lists only
        days that have at least one live entry.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT day, book, COUNT(*) AS n FROM entries"
                " WHERE day BETWEEN ? AND ? AND deleted_at IS NULL"
                " GROUP BY day, book",
                (day_from, day_to),
            ).fetchall()
        out: dict[str, dict[str, int]] = {}
        for row in rows:
            out.setdefault(row["day"], {})[row["book"]] = row["n"]
        return out

    def changed_since(
        self,
        since: datetime,
        *,
        until: Optional[datetime] = None,
        book: Optional[str] = None,
        include_deleted: bool = True,
        cursor: Optional[str] = None,
        limit: int = _QUERY_CAP,
    ) -> ChangePage:
        """Return entries whose ``updated_at`` is after ``since``, oldest change first.

        The synchroniser's question ("everything that changed since I last
        asked"), and the one listing that includes **tombstones**: a
        deleted entry is a change a downstream copy must learn about, and
        this is the only place it can. ``include_deleted=False`` drops them.

        Timestamps are aware. Rows are ordered by ``(updated_at, rowid)``
        and a full page returns a cursor that resumes after its last row,
        so two entries sharing an ``updated_at`` across a page boundary
        cannot lose one. ``since`` is exclusive, ``until`` inclusive.
        """
        if limit < 1:
            raise ValueError("limit must be at least 1")
        sql = "SELECT rowid AS _rowid, * FROM entries WHERE "
        params: list[object] = []
        if cursor is not None:
            stamp, row = _decode_cursor(cursor)
            sql += "(updated_at > ? OR (updated_at = ? AND rowid > ?))"
            params += [stamp, stamp, row]
        else:
            sql += "updated_at > ?"
            params.append(_stamp(since))
        if until is not None:
            sql += " AND updated_at <= ?"
            params.append(_stamp(until))
        if book is not None:
            sql += " AND book = ?"
            params.append(book)
        if not include_deleted:
            sql += " AND deleted_at IS NULL"
        limit = min(limit, _QUERY_CAP)
        sql += " ORDER BY updated_at, rowid LIMIT ?"
        params.append(limit + 1)  # one extra tells us whether a page follows
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        more = len(rows) > limit
        rows = rows[:limit]
        entries = [_from_row(row) for row in rows]
        next_cursor = (
            _encode_cursor(rows[-1]["updated_at"], rows[-1]["_rowid"]) if more else None
        )
        return ChangePage(entries=entries, next_cursor=next_cursor)

    def history(self, entry_id: str) -> list[HistoryRecord]:
        """Return an entry's earlier states, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM entry_history WHERE entry_id = ? ORDER BY seq",
                (entry_id,),
            ).fetchall()
        return [
            HistoryRecord(
                version=row["version"],
                reason=row["reason"],
                recorded_at=datetime.fromisoformat(row["recorded_at"]),
                entry=LogEntry.model_validate(json.loads(row["snapshot"])),
            )
            for row in rows
        ]

    def _snapshot(self, conn: sqlite3.Connection, entry_id: str, reason: str) -> bool:
        """Keep the entry's current state before a change. Same connection."""
        row = conn.execute(
            "SELECT * FROM entries WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return False
        entry = _from_row(row)
        conn.execute(
            "INSERT INTO entry_history (entry_id, version, reason, recorded_at, snapshot)"
            " VALUES (?,?,?,?,?)",
            (
                entry_id,
                entry.version,
                reason,
                _now().isoformat(),
                json.dumps(entry.model_dump(mode="json")),
            ),
        )
        return True

    def unmirrored(self, limit: int = 100) -> list[LogEntry]:
        """Return entries whose markdown does not match the share yet.

        The reconciliation half of writing here first: a save never fails
        because the share is down, but something has to notice and retry.
        Deleted entries are included — their owed operation is removing the
        file, and :func:`geecs_logbook.mirror.sync` knows which is which.

        Never-tried entries come first, then the least recently tried: an
        entry the share keeps refusing must not sit at the head of the
        queue and starve the ones behind it. Callers record each failed
        try with :meth:`mark_deferred`.
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
        book: str = "scans",
        kind: str = "note",
        status: str = "kept",
        template: str = "blank",
        payload: Optional[dict] = None,
    ) -> LogEntry:
        """Store a new entry and return it.

        Tags are parsed from the body here — a caller cannot set them
        directly, because the body is the truth for them.

        Raises
        ------
        ValueError
            If both ``scan`` and ``after`` are given — an entry is anchored
            to a scan, to the gap after one, or (neither) to the day; if an
            ``ops`` entry carries an anchor at all; or if an agent's entry
            arrives already ``kept``: only a human keeps an agent's draft,
            and that has to hold at the API, not in the documentation.
        """
        if scan is not None and after is not None:
            raise ValueError("an entry is anchored to a scan or after one, not both")
        if book == "ops" and (scan is not None or after is not None):
            raise ValueError("an ops entry is about the day, not a scan")
        if kind != "note" and status == "kept":
            raise ValueError(
                f"a {kind} entry is created as a draft; a person keeps it afterwards"
            )

        # The stamp is taken under the write lock, as every other writer
        # takes its own: taken before it, two concurrent creates could
        # commit out of stamp order and the change feed's high-water mark
        # would skip one for good.
        with self._connect() as conn, _transaction(conn):
            now = _now()
            entry = LogEntry(
                entry_id=uuid.uuid4().hex[:12],
                day=day,
                book=book,
                scan=scan,
                after=after,
                author=author,
                kind=kind,
                status=status,
                template=template,
                body_md=body_md,
                tags=parse_tags(body_md),
                payload=payload,
                created_at=now,
                updated_at=now,
            )
            conn.execute(
                "INSERT INTO entries (entry_id, day, scan, after_scan, author, kind,"
                " status, template, body_md, payload, attachments, created_at,"
                " edited_at, edited_by, updated_at, deleted_at, version,"
                " schema_version, book, tags, mirrored_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                _to_params(entry),
            )
        logger.info("entry %s created for %s by %s", entry.entry_id, day, author)
        return entry

    def update(
        self, entry_id: str, *, body_md: str, editor: str, expected_version: int
    ) -> LogEntry:
        """Replace an entry's body, if nobody else has saved since.

        ``author`` is never touched: it is who wrote the entry and part of
        the mirror file's stable name. The editor is recorded as
        ``edited_by``.

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

        with self._connect() as conn, _transaction(conn):
            # The WHERE clause repeats the version check: between the read
            # above and this write another process may have saved, and the
            # comparison has to happen where the write happens. The
            # snapshot and the update are one transaction, so history can
            # never hold a state that was not the one replaced.
            self._snapshot(conn, entry_id, "edit")
            now = _now().isoformat()
            cursor = conn.execute(
                "UPDATE entries SET body_md = ?, tags = ?, edited_by = ?,"
                " edited_at = ?, updated_at = ?, version = version + 1,"
                " mirrored_at = NULL"
                " WHERE entry_id = ? AND version = ? AND deleted_at IS NULL",
                (
                    body_md,
                    json.dumps(parse_tags(body_md)),
                    editor,
                    now,
                    now,
                    entry_id,
                    expected_version,
                ),
            )
            if cursor.rowcount == 0:
                # Either someone saved first, or the entry was deleted
                # between the read above and here; the two deserve
                # different answers.
                latest = self.get(entry_id, include_deleted=True)
                if latest is None or latest.is_deleted:
                    raise KeyError(entry_id)
                raise ConflictError(latest)
        return self.get(entry_id)  # type: ignore[return-value]

    def set_status(self, entry_id: str, status: str) -> LogEntry:
        """Promote a draft to kept, or demote it.

        Separate from :meth:`update` because promoting an agent's draft is
        a human act on unchanged text, and conflating it with an edit would
        let an agent promote its own output by rewriting it.
        """
        with self._connect() as conn, _transaction(conn):
            self._snapshot(conn, entry_id, "status")
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
        with self._connect() as conn, _transaction(conn):
            self._snapshot(conn, entry_id, "attach")
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
        try:
            with self._connect() as conn, _transaction(conn):
                self._snapshot(conn, entry_id, "delete")
                now = _now().isoformat()
                cursor = conn.execute(
                    "UPDATE entries SET deleted_at = ?, updated_at = ?,"
                    " version = version + 1, mirrored_at = NULL"
                    " WHERE entry_id = ? AND deleted_at IS NULL",
                    (now, now, entry_id),
                )
                if cursor.rowcount == 0:
                    # Missing or already a tombstone: nothing was replaced,
                    # so the snapshot must not be kept either.
                    raise _NothingToDelete
        except _NothingToDelete:
            return False
        return True


class _NothingToDelete(Exception):
    """Private: unwinds a delete that matched no live row, rolling back."""


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """``BEGIN IMMEDIATE`` … ``COMMIT``, rolling back on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


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
        entry.edited_by,
        entry.updated_at.isoformat(),
        entry.deleted_at.isoformat() if entry.deleted_at else None,
        entry.version,
        entry.schema_version,
        entry.book,
        json.dumps(entry.tags),
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
            "edited_by": row["edited_by"],
            "updated_at": row["updated_at"],
            "deleted_at": row["deleted_at"],
            "version": row["version"],
            "book": row["book"],
            "tags": json.loads(row["tags"] or "[]"),
        }
    )
