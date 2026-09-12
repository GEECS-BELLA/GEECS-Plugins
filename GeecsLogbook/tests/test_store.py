"""The notes store: the one thing in the package that cannot be rebuilt."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from geecs_schemas.log_entry import Attachment

from geecs_logbook.store import ConflictError, NotesStore

DAY = "2026-09-11"


@pytest.fixture
def store(tmp_path: Path) -> NotesStore:
    """A fresh store in a temp directory."""
    return NotesStore(tmp_path / "notes.db")


class TestCreate:
    """Storing a new entry."""

    def test_returns_a_versioned_entry(self, store: NotesStore) -> None:
        """A new entry has an id, version 1, and the note defaults."""
        e = store.create(day=DAY, scan=5, author="sbarber", body_md="hello")
        assert e.entry_id and e.version == 1
        assert e.kind == "note" and e.status == "kept"
        assert e.created_at.tzinfo is not None  # aware, never naive

    def test_at_most_one_anchor(self, store: NotesStore) -> None:
        """An entry is on a scan, after one, or on the day — never both."""
        with pytest.raises(ValueError):
            store.create(day=DAY, scan=1, after=1, author="a", body_md="x")

    def test_three_anchors(self, store: NotesStore) -> None:
        """`after=`, `scan=` and neither are the three places an entry lives."""
        between = store.create(day=DAY, after=3, author="a", body_md="x")
        on = store.create(day=DAY, scan=3, author="a", body_md="x")
        day = store.create(day=DAY, author="a", body_md="x")
        assert between.is_interscan and between.anchor == "after-0003"
        assert on.anchor == "scan-0003"
        assert day.is_day_level and day.anchor == "day"

    def test_an_agent_entry_is_born_a_draft(self, store: NotesStore) -> None:
        """Only a person keeps an agent's output; the store refuses otherwise."""
        for kind in ("agent_analysis", "agent_draft"):
            with pytest.raises(ValueError, match="draft"):
                store.create(day=DAY, scan=1, author="osprey", body_md="x", kind=kind)
            e = store.create(
                day=DAY, scan=1, author="osprey", body_md="x", kind=kind, status="draft"
            )
            assert e.status == "draft"
        # A human's note is kept on arrival, as before.
        assert store.create(day=DAY, scan=1, author="a", body_md="x").status == "kept"

    def test_updated_at_starts_at_creation(self, store: NotesStore) -> None:
        """A new entry's last-change time is its creation time."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        assert e.updated_at == e.created_at and e.edited_at is None

    def test_refuses_a_missing_directory(self, tmp_path: Path) -> None:
        """A mistyped path fails loudly rather than writing somewhere unseen."""
        with pytest.raises(NotADirectoryError):
            NotesStore(tmp_path / "nope" / "notes.db")


class TestRead:
    """Getting entries back."""

    def test_get_round_trips(self, store: NotesStore) -> None:
        """What went in comes back, payload included."""
        e = store.create(
            status="draft",
            day=DAY,
            scan=5,
            author="osprey",
            body_md="peak at 3.5 mm",
            kind="agent_analysis",
            template="figures",
            payload={
                "kind": "analysis",
                "analyzer": "Array1D",
                "metrics": {"onset_mm": 4.1},
            },
        )
        got = store.get(e.entry_id)
        assert got is not None
        assert got.body_md == "peak at 3.5 mm"
        assert got.payload is not None and got.payload.metrics == {"onset_mm": 4.1}
        assert got.template == "figures"

    def test_missing_is_none(self, store: NotesStore) -> None:
        """An unknown id is None, not an exception."""
        assert store.get("nope") is None

    def test_for_day_is_ordered_and_scoped(self, store: NotesStore) -> None:
        """A day's entries come back oldest first, other days excluded."""
        a = store.create(day=DAY, scan=1, author="a", body_md="first")
        b = store.create(day=DAY, scan=1, author="a", body_md="second")
        store.create(day="2026-09-10", scan=1, author="a", body_md="other day")
        got = store.for_day(DAY)
        assert [e.entry_id for e in got] == [a.entry_id, b.entry_id]


class TestUpdate:
    """Editing, and the conflict rule that protects prose."""

    def test_bumps_version_and_marks_unmirrored(self, store: NotesStore) -> None:
        """An edit advances the version and owes a fresh mirror write."""
        e = store.create(day=DAY, scan=5, author="a", body_md="v1")
        store.mark_mirrored(e.entry_id)
        assert store.unmirrored() == []
        got = store.update(e.entry_id, body_md="v2", author="b", expected_version=1)
        assert got.body_md == "v2" and got.version == 2
        assert got.edited_at is not None
        assert [x.entry_id for x in store.unmirrored()] == [e.entry_id]

    def test_stale_version_conflicts_with_the_current_entry(
        self, store: NotesStore
    ) -> None:
        """Last-writer-wins would silently eat a paragraph; this refuses.

        The exception carries what is there now, so the UI can show what
        the loser lost to rather than only that it lost.
        """
        e = store.create(day=DAY, scan=5, author="a", body_md="v1")
        store.update(e.entry_id, body_md="v2", author="b", expected_version=1)
        with pytest.raises(ConflictError) as exc:
            store.update(e.entry_id, body_md="v3", author="c", expected_version=1)
        assert exc.value.current.body_md == "v2"
        assert exc.value.current.version == 2
        assert store.get(e.entry_id).body_md == "v2"  # untouched

    def test_missing_entry_is_key_error(self, store: NotesStore) -> None:
        """Editing nothing is the caller's error."""
        with pytest.raises(KeyError):
            store.update("nope", body_md="x", author="a", expected_version=1)


class TestStatusAndAttachments:
    """Promotion and the attachment manifest."""

    def test_promote_draft_is_separate_from_edit(self, store: NotesStore) -> None:
        """Keeping an agent draft changes status, not text."""
        e = store.create(
            day=DAY,
            scan=5,
            author="osprey",
            body_md="guess",
            kind="agent_draft",
            status="draft",
        )
        got = store.set_status(e.entry_id, "kept")
        assert got.status == "kept" and got.version == 2
        assert got.body_md == "guess"

    def test_attachments_accumulate(self, store: NotesStore) -> None:
        """Each upload appends to the manifest."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        now = datetime.now(timezone.utc)
        store.add_attachment(
            e.entry_id,
            Attachment(
                id=e.entry_id,
                filename="a.png",
                content_type="image/png",
                size_bytes=10,
                uploaded_at=now,
            ),
        )
        got = store.add_attachment(
            e.entry_id,
            Attachment(
                id=e.entry_id,
                filename="b.pdf",
                content_type="application/pdf",
                size_bytes=20,
                uploaded_at=now,
            ),
        )
        assert [a.filename for a in got.attachments] == ["a.png", "b.pdf"]


class TestUpdatedAt:
    """`updated_at` moves on every change, not only on edits to the text."""

    def test_status_and_attachments_move_it(self, store: NotesStore) -> None:
        """A promotion or an upload leaves edited_at alone but not updated_at."""
        e = store.create(
            day=DAY, scan=5, author="a", body_md="x", kind="agent_draft", status="draft"
        )
        kept = store.set_status(e.entry_id, "kept")
        assert kept.updated_at > e.updated_at and kept.edited_at is None
        with_file = store.add_attachment(
            e.entry_id,
            Attachment(
                id="f1",
                filename="a.png",
                content_type="image/png",
                size_bytes=1,
                uploaded_at=kept.updated_at,
            ),
        )
        assert with_file.updated_at > kept.updated_at and with_file.edited_at is None

    def test_edit_moves_both(self, store: NotesStore) -> None:
        """An edit is the one change a reader is told about."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        e2 = store.update(e.entry_id, body_md="y", author="a", expected_version=1)
        assert e2.edited_at is not None and e2.updated_at == e2.edited_at


class TestDelete:
    """Removal is a tombstone, not a hole."""

    def test_delete_reports_whether_anything_went(self, store: NotesStore) -> None:
        """True once, False after; the entry then reads as missing."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        assert store.delete(e.entry_id) is True
        assert store.delete(e.entry_id) is False
        assert store.get(e.entry_id) is None
        assert store.for_day(DAY) == []

    def test_the_row_survives_as_a_tombstone(self, store: NotesStore) -> None:
        """A deleted entry is still there for whoever needs to learn it went."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        store.delete(e.entry_id)
        gone = store.get(e.entry_id, include_deleted=True)
        assert gone is not None and gone.is_deleted
        assert gone.updated_at == gone.deleted_at
        assert gone.version == e.version + 1
        # Owed to the mirror: the file has to come off the share.
        assert [x.entry_id for x in store.unmirrored()] == [e.entry_id]

    def test_a_deleted_entry_refuses_writes(self, store: NotesStore) -> None:
        """Editing, promoting or attaching to a tombstone is a missing-entry error."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        store.delete(e.entry_id)
        with pytest.raises(KeyError):
            store.update(e.entry_id, body_md="y", author="a", expected_version=2)
        with pytest.raises(KeyError):
            store.set_status(e.entry_id, "draft")
        with pytest.raises(KeyError):
            store.add_attachment(
                e.entry_id,
                Attachment(
                    id="f",
                    filename="a.png",
                    content_type="image/png",
                    size_bytes=1,
                    uploaded_at=e.created_at,
                ),
            )


class TestMigration:
    """A database from before a column existed opens and reads."""

    def test_adds_missing_columns_and_fills_them(self, tmp_path: Path) -> None:
        """An old file gains updated_at (from edited/created) and deleted_at."""
        import sqlite3

        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE entries (
                entry_id TEXT PRIMARY KEY, day TEXT NOT NULL, scan INTEGER,
                after_scan INTEGER, author TEXT NOT NULL, kind TEXT NOT NULL,
                status TEXT NOT NULL, template TEXT NOT NULL, body_md TEXT NOT NULL,
                payload TEXT, attachments TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL, edited_at TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                schema_version INTEGER NOT NULL DEFAULT 1, mirrored_at TEXT
            );
            INSERT INTO entries VALUES ('old1', '2026-09-11', 5, NULL, 'a', 'note',
                'kept', 'blank', 'x', NULL, '[]', '2026-09-11T10:00:00+00:00',
                '2026-09-11T11:00:00+00:00', 2, 1, NULL);
            INSERT INTO entries VALUES ('old2', '2026-09-11', 6, NULL, 'a', 'note',
                'kept', 'blank', 'x', NULL, '[]', '2026-09-11T10:00:00+00:00',
                NULL, 1, 1, NULL);
            """
        )
        conn.close()

        store = NotesStore(path)
        edited, fresh = store.get("old1"), store.get("old2")
        assert edited is not None and edited.updated_at == edited.edited_at
        assert fresh is not None and fresh.updated_at == fresh.created_at
        assert not edited.is_deleted and store.for_day("2026-09-11") == [edited, fresh]
        # Opening again is a no-op, not a second ALTER.
        NotesStore(path)
