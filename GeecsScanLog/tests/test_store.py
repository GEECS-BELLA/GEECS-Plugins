"""The notes store: the one thing in the package that cannot be rebuilt."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from geecs_schemas.log_entry import Attachment

from geecs_scan_log.store import ConflictError, NotesStore

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

    def test_needs_exactly_one_anchor(self, store: NotesStore) -> None:
        """An entry is on a scan or after one — never both, never neither."""
        with pytest.raises(ValueError):
            store.create(day=DAY, author="a", body_md="x")
        with pytest.raises(ValueError):
            store.create(day=DAY, scan=1, after=1, author="a", body_md="x")

    def test_interscan_and_intro_anchors(self, store: NotesStore) -> None:
        """`after=` and `scan=0` are both valid anchors."""
        between = store.create(day=DAY, after=3, author="a", body_md="x")
        intro = store.create(day=DAY, scan=0, author="a", body_md="x")
        assert between.is_interscan and between.anchor == "after-0003"
        assert not intro.is_interscan and intro.anchor == "intro"

    def test_refuses_a_missing_directory(self, tmp_path: Path) -> None:
        """A mistyped path fails loudly rather than writing somewhere unseen."""
        with pytest.raises(NotADirectoryError):
            NotesStore(tmp_path / "nope" / "notes.db")


class TestRead:
    """Getting entries back."""

    def test_get_round_trips(self, store: NotesStore) -> None:
        """What went in comes back, payload included."""
        e = store.create(
            day=DAY, scan=5, author="osprey", body_md="peak at 3.5 mm",
            kind="agent_analysis", template="figures",
            payload={"kind": "analysis", "analyzer": "Array1D",
                     "metrics": {"onset_mm": 4.1}},
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
        e = store.create(day=DAY, scan=5, author="osprey", body_md="guess",
                         kind="agent_draft", status="draft")
        got = store.set_status(e.entry_id, "kept")
        assert got.status == "kept" and got.version == 2
        assert got.body_md == "guess"

    def test_attachments_accumulate(self, store: NotesStore) -> None:
        """Each upload appends to the manifest."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        now = datetime.now(timezone.utc)
        store.add_attachment(e.entry_id, Attachment(
            id=e.entry_id, filename="a.png", content_type="image/png",
            size_bytes=10, uploaded_at=now))
        got = store.add_attachment(e.entry_id, Attachment(
            id=e.entry_id, filename="b.pdf", content_type="application/pdf",
            size_bytes=20, uploaded_at=now))
        assert [a.filename for a in got.attachments] == ["a.png", "b.pdf"]


class TestDelete:
    """Removal."""

    def test_delete_reports_whether_anything_went(self, store: NotesStore) -> None:
        """True once, False after."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        assert store.delete(e.entry_id) is True
        assert store.delete(e.entry_id) is False
        assert store.get(e.entry_id) is None
