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
        got = store.update(e.entry_id, body_md="v2", editor="b", expected_version=1)
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
        store.update(e.entry_id, body_md="v2", editor="b", expected_version=1)
        with pytest.raises(ConflictError) as exc:
            store.update(e.entry_id, body_md="v3", editor="c", expected_version=1)
        assert exc.value.current.body_md == "v2"
        assert exc.value.current.version == 2
        assert store.get(e.entry_id).body_md == "v2"  # untouched

    def test_missing_entry_is_key_error(self, store: NotesStore) -> None:
        """Editing nothing is the caller's error."""
        with pytest.raises(KeyError):
            store.update("nope", body_md="x", editor="a", expected_version=1)


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
        e2 = store.update(e.entry_id, body_md="y", editor="a", expected_version=1)
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
            store.update(e.entry_id, body_md="y", editor="a", expected_version=2)
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


class TestMirrorBookkeeping:
    """The queue rotates, and a mark cannot cover a newer version."""

    def test_deferred_entries_rotate_to_the_back(self, store: NotesStore) -> None:
        """A never-tried entry is served before one that failed already."""
        stuck = store.create(day="2026-09-12", author="a", body_md="no day yet")
        store.mark_deferred(stuck.entry_id)
        fresh = store.create(day=DAY, scan=1, author="a", body_md="now")
        assert [e.entry_id for e in store.unmirrored()] == [
            fresh.entry_id,
            stuck.entry_id,
        ]
        # A limit of one no longer starves the fresh one.
        assert [e.entry_id for e in store.unmirrored(limit=1)] == [fresh.entry_id]

    def test_mark_is_pinned_to_the_version_written(self, store: NotesStore) -> None:
        """An edit between the read and the mark keeps the entry owed."""
        e = store.create(day=DAY, scan=1, author="a", body_md="v1")
        store.update(e.entry_id, body_md="v2", editor="a", expected_version=1)
        assert store.mark_mirrored(e.entry_id, version=e.version) is False
        assert [x.entry_id for x in store.unmirrored()] == [e.entry_id]
        assert store.mark_mirrored(e.entry_id, version=2) is True
        assert store.unmirrored() == []

    def test_concurrent_attachment_appends_both_land(self, store: NotesStore) -> None:
        """Two appends from the same read do not lose one another."""
        e = store.create(day=DAY, scan=1, author="a", body_md="x")

        def mk(i: int) -> Attachment:
            return Attachment(
                id=f"f{i}",
                filename=f"{i}.png",
                content_type="image/png",
                size_bytes=1,
                uploaded_at=e.created_at,
            )

        store.add_attachment(e.entry_id, mk(1))
        store.add_attachment(e.entry_id, mk(2))
        got = store.get(e.entry_id)
        assert got is not None and [a.id for a in got.attachments] == ["f1", "f2"]
        assert got.version == 3


class TestBooks:
    """Two books, one store."""

    def test_default_book_is_scans(self, store: NotesStore) -> None:
        """Existing callers get the campaign record."""
        assert store.create(day=DAY, scan=1, author="a", body_md="x").book == "scans"

    def test_ops_is_day_level_only(self, store: NotesStore) -> None:
        """An ops entry cannot be anchored to a scan or a gap."""
        for anchor in ({"scan": 1}, {"after": 0}):
            with pytest.raises(ValueError, match="ops"):
                store.create(day=DAY, author="a", body_md="x", book="ops", **anchor)
        e = store.create(day=DAY, author="a", body_md="x", book="ops")
        assert e.book == "ops" and e.is_day_level

    def test_for_day_filters_by_book(self, store: NotesStore) -> None:
        """Each page asks for its own book; the API can ask for both."""
        store.create(day=DAY, scan=1, author="a", body_md="s")
        store.create(day=DAY, author="a", body_md="o", book="ops")
        assert [e.body_md for e in store.for_day(DAY, book="scans")] == ["s"]
        assert [e.body_md for e in store.for_day(DAY, book="ops")] == ["o"]
        assert len(store.for_day(DAY)) == 2


class TestTags:
    """Tags are read out of the body, never set."""

    def test_parsed_at_create_and_reparsed_on_edit(self, store: NotesStore) -> None:
        """The body is the truth; the column follows it."""
        e = store.create(day=DAY, scan=1, author="a", body_md="#Laser drift, #jet ok")
        assert e.tags == ["laser", "jet"]
        e2 = store.update(e.entry_id, body_md="no tags", editor="a", expected_version=1)
        assert e2.tags == []


class TestQuery:
    """One question, many parameters."""

    def _seed(self, store: NotesStore) -> None:
        store.create(day="2026-09-01", scan=1, author="a", body_md="scan #laser")
        store.create(day="2026-09-02", author="b", body_md="ops #laser", book="ops")
        store.create(day="2026-09-03", author="a", body_md="ops #jet", book="ops")
        store.create(day="2026-10-01", author="a", body_md="next month", book="ops")
        gone = store.create(day="2026-09-04", author="a", body_md="deleted", book="ops")
        store.delete(gone.entry_id)

    def test_range_book_tag_and_author(self, store: NotesStore) -> None:
        """Filters compose; deleted rows never appear; order is by day."""
        self._seed(store)
        sep = store.query(day_from="2026-09-01", day_to="2026-09-30")
        assert [e.day for e in sep] == ["2026-09-01", "2026-09-02", "2026-09-03"]
        ops = store.query(day_from="2026-09-01", day_to="2026-09-30", book="ops")
        assert [e.body_md for e in ops] == ["ops #laser", "ops #jet"]
        laser = store.query(day_from="2026-09-01", day_to="2026-09-30", tag="LASER")
        assert [e.body_md for e in laser] == ["scan #laser", "ops #laser"]
        by_b = store.query(day_from="2026-09-01", day_to="2026-09-30", author="b")
        assert [e.body_md for e in by_b] == ["ops #laser"]

    def test_scan_anchored_can_be_hidden(self, store: NotesStore) -> None:
        """The month page's default: the campaign record stays out of the way."""
        self._seed(store)
        day_level = store.query(
            day_from="2026-09-01", day_to="2026-09-30", include_scan_anchored=False
        )
        assert all(e.is_day_level for e in day_level) and len(day_level) == 2


class TestHistory:
    """Every change keeps the state it replaced."""

    def test_snapshots_before_each_change(self, store: NotesStore) -> None:
        """edit, status, attach, delete — in order, with the pre-change entry."""
        e = store.create(
            day=DAY,
            scan=1,
            author="a",
            body_md="v1",
            kind="agent_draft",
            status="draft",
        )
        assert store.history(e.entry_id) == []
        store.update(e.entry_id, body_md="v2", editor="a", expected_version=1)
        store.set_status(e.entry_id, "kept")
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
        store.delete(e.entry_id)
        hist = store.history(e.entry_id)
        assert [(h.reason, h.version) for h in hist] == [
            ("edit", 1),
            ("status", 2),
            ("attach", 3),
            ("delete", 4),
        ]
        assert hist[0].entry.body_md == "v1" and hist[1].entry.status == "draft"
        assert hist[3].entry.attachments[0].filename == "a.png"
        assert all(h.recorded_at.tzinfo is not None for h in hist)

    def test_a_refused_edit_leaves_no_history(self, store: NotesStore) -> None:
        """A conflict rolls the snapshot back with the update."""
        e = store.create(day=DAY, scan=1, author="a", body_md="v1")
        with pytest.raises(ConflictError):
            store.update(e.entry_id, body_md="x", editor="a", expected_version=9)
        assert store.history(e.entry_id) == []
