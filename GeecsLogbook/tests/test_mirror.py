"""The markdown mirror: durable, relative-linked, and never a scans/ writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from geecs_logbook import mirror
from geecs_logbook.mirror import MirrorUnavailable
from geecs_logbook.store import NotesStore

DAY = "2026-09-11"
EXP = "Undulator"


@pytest.fixture
def share(tmp_path: Path) -> Path:
    """A share with the day folder and its scans/ present, as the scanner leaves it."""
    scans = tmp_path / EXP / "Y2026" / "09-Sep" / "26_0911" / "scans"
    (scans / "Scan005").mkdir(parents=True)
    (scans / "Scan005" / "scan.log").write_text("")
    return tmp_path


@pytest.fixture
def store(tmp_path: Path) -> NotesStore:
    """A store beside the share."""
    return NotesStore(tmp_path / "notes.db")


def _snapshot(root: Path) -> set[Path]:
    return set(root.rglob("*"))


class TestPaths:
    """Where things go. Nothing here touches the disk."""

    def test_logbook_is_a_sibling_of_scans(self, share: Path) -> None:
        """logbook/ sits beside scans/ and analysis/, never inside."""
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        assert root.name == "logbook"
        assert (root.parent / "scans").is_dir()
        assert "scans" not in root.parts

    def test_three_anchors_three_places(self, store: NotesStore, share: Path) -> None:
        """Day-level at the root, scan entries under ScanNNN/, interscan under after-ScanNNN/."""
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        day = store.create(day=DAY, author="T. Operator", body_md="x")
        on = store.create(day=DAY, scan=5, author="S. Barber", body_md="x")
        between = store.create(day=DAY, after=3, author="osprey", body_md="x")
        assert mirror.entry_path(day, root).parent == root
        assert mirror.entry_path(day, root).name.endswith(
            f"-toperator-{day.entry_id[:6]}.md"
        )
        assert mirror.entry_path(on, root).parent == root / "Scan005"
        assert mirror.entry_path(between, root).parent == root / "after-Scan003"

    def test_name_is_stable_across_edits(self, store: NotesStore, share: Path) -> None:
        """An edit overwrites the same file rather than leaving a trail."""
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        e = store.create(day=DAY, scan=5, author="S. Barber", body_md="v1")
        before = mirror.entry_path(e, root)
        e2 = store.update(
            e.entry_id, body_md="v2", editor="T. Operator", expected_version=1
        )
        # A colleague's edit neither takes the entry over nor renames its file.
        assert e2.author == "S. Barber" and e2.edited_by == "T. Operator"
        assert mirror.entry_path(e2, root) == before
        assert before.name.endswith(f"-sbarber-{e.entry_id[:6]}.md")
        assert "edited_by: T. Operator" in mirror.render(e2)

    def test_attachment_link_is_relative(self, store: NotesStore) -> None:
        """A body's image link resolves from the file on disk, no server needed."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        link = mirror.attachment_link(e, "jet-trace.png")
        assert link == f"attachments/{e.entry_id}/jet-trace.png"
        assert not link.startswith("/") and "://" not in link


class TestRender:
    """The file's shape."""

    def test_front_matter_then_verbatim_body(self, store: NotesStore) -> None:
        """Envelope up top as key: value; the author's text untouched below."""
        body = "### What were we trying to do?\n\nRolloff is **real**.\n"
        e = store.create(
            day=DAY, scan=5, author="S. Barber", body_md=body, template="scan_note"
        )
        text = mirror.render(e)
        head, _, rest = text.partition("\n---\n")
        assert head.startswith("---\n")
        for key in (
            "entry_id:",
            "day: 2026-09-11",
            "scan: 5",
            "author: S. Barber",
            "kind: note",
            "status: kept",
            "template: scan_note",
            "version: 1",
            "schema_version: 1",
        ):
            assert key in head, key
        assert rest.strip() == body.strip()

    def test_interscan_writes_after_not_scan(self, store: NotesStore) -> None:
        """The anchor is recorded the way it was made."""
        e = store.create(day=DAY, after=3, author="a", body_md="x")
        head = mirror.render(e).split("\n---\n")[0]
        assert "after: 3" in head and "scan:" not in head

    def test_payload_and_attachments_are_listed(self, store: NotesStore) -> None:
        """Machine structure and the manifest ride in the front matter."""
        from datetime import datetime, timezone
        from geecs_schemas.log_entry import Attachment

        e = store.create(
            day=DAY,
            scan=5,
            author="osprey",
            body_md="x",
            payload={
                "kind": "analysis",
                "analyzer": "A1D",
                "metrics": {"onset_mm": 4.1},
            },
        )
        e = store.add_attachment(
            e.entry_id,
            Attachment(
                id=e.entry_id,
                filename="a.png",
                content_type="image/png",
                size_bytes=1,
                uploaded_at=datetime.now(timezone.utc),
            ),
        )
        head = mirror.render(e).split("\n---\n")[0]
        assert '"onset_mm": 4.1' in head
        assert f"- attachments/{e.entry_id}/a.png" in head


class TestWrite:
    """Landing on the share, and what it must never do."""

    def test_writes_under_an_existing_day(self, store: NotesStore, share: Path) -> None:
        """With the day present, the file lands and scans/ is untouched."""
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        scans_before = _snapshot(root.parent / "scans")
        e = store.create(day=DAY, scan=5, author="a", body_md="hello")
        path = mirror.write_entry(e, root)
        assert path.is_file() and path.read_text().endswith("hello\n")
        assert not path.with_name(path.name + ".tmp").exists()  # atomic, no debris
        assert _snapshot(root.parent / "scans") == scans_before

    def test_refuses_to_create_the_day(self, store: NotesStore, tmp_path: Path) -> None:
        """The scanner makes days. With none present, nothing is created at all.

        This is the one creation the scan-folder invariant forbids that a
        `mkdir(parents=True)` on logbook/ScanNNN/ could otherwise commit.
        """
        root = mirror.logbook_root("2026-09-12", EXP, base_directory=tmp_path)
        e = store.create(day="2026-09-12", scan=1, author="a", body_md="early")
        before = _snapshot(tmp_path)
        with pytest.raises(MirrorUnavailable):
            mirror.write_entry(e, root)
        with pytest.raises(MirrorUnavailable):
            mirror.write_attachment(e, "x.png", b"\x89PNG", root)
        assert _snapshot(tmp_path) == before

    def test_never_calls_mkdir_when_day_absent(
        self, store: NotesStore, tmp_path: Path, monkeypatch
    ) -> None:
        """Belt and braces: the guard fires before any mkdir is reachable."""
        root = mirror.logbook_root("2026-09-12", EXP, base_directory=tmp_path)
        e = store.create(day="2026-09-12", scan=1, author="a", body_md="x")

        def explode(*_a, **_k):
            raise AssertionError("mkdir reached with no day folder present")

        monkeypatch.setattr(Path, "mkdir", explode)
        with pytest.raises(MirrorUnavailable):
            mirror.write_entry(e, root)

    def test_attachment_lands_beside_the_entry(
        self, store: NotesStore, share: Path
    ) -> None:
        """Bytes go to attachments/<id>/; the returned link points at them."""
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        link = mirror.write_attachment(e, "trace.png", b"\x89PNGdata", root)
        target = mirror.entry_dir(e, root) / link
        assert target.read_bytes() == b"\x89PNGdata"

    def test_remove_reports_and_tolerates_absence(
        self, store: NotesStore, share: Path
    ) -> None:
        """Removing twice is fine; the second is a no-op."""
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        mirror.write_entry(e, root)
        assert mirror.remove_entry(e, root) is True
        assert mirror.remove_entry(e, root) is False


class TestSync:
    """Paying what the store owes."""

    def test_writes_owed_entries_and_marks_them(
        self, store: NotesStore, share: Path
    ) -> None:
        """Unmirrored entries land and stop being owed."""
        a = store.create(day=DAY, scan=5, author="a", body_md="one")
        b = store.create(day=DAY, after=5, author="a", body_md="two")
        written, deferred = mirror.sync(store, EXP, base_directory=share)
        assert (written, deferred) == (2, 0)
        assert store.unmirrored() == []
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        assert mirror.entry_path(a, root).is_file()
        assert mirror.entry_path(b, root).is_file()

    def test_defers_a_day_that_does_not_exist_yet(
        self, store: NotesStore, share: Path
    ) -> None:
        """A note written before the day's first scan waits; the others land."""
        today = store.create(day=DAY, scan=5, author="a", body_md="now")
        early = store.create(day="2026-09-12", author="a", body_md="tomorrow")
        written, deferred = mirror.sync(store, EXP, base_directory=share)
        assert (written, deferred) == (1, 1)
        owed = [e.entry_id for e in store.unmirrored()]
        assert owed == [early.entry_id]
        assert store.get(today.entry_id) is not None  # the words are safe either way

    def test_removes_the_file_of_a_deleted_entry(
        self, store: NotesStore, share: Path
    ) -> None:
        """A tombstone's owed operation is the removal; it is paid and marked."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        assert mirror.sync(store, EXP, base_directory=share) == (1, 0)
        path = mirror.entry_path(e, mirror.logbook_root(DAY, EXP, base_directory=share))
        assert path.is_file()
        store.delete(e.entry_id)
        assert mirror.sync(store, EXP, base_directory=share) == (1, 0)
        assert not path.exists()
        assert store.unmirrored() == []

    def test_a_deleted_entry_never_mirrored_is_settled_without_a_file(
        self, store: NotesStore, share: Path
    ) -> None:
        """Delete before the mirror ever ran: nothing to remove, nothing owed."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        store.delete(e.entry_id)
        assert mirror.sync(store, EXP, base_directory=share) == (1, 0)
        assert store.unmirrored() == []

    def test_sync_never_covers_a_newer_file_with_an_older_one(
        self, store: NotesStore, share: Path
    ) -> None:
        """What is written is what was read under the lock, never a stale copy."""
        e = store.create(day=DAY, scan=5, author="a", body_md="v1")
        stale = store.unmirrored()  # a sync that read v1 …
        e2 = store.update(e.entry_id, body_md="v2", editor="a", expected_version=1)
        assert stale[0].version == 1 and e2.version == 2
        # … then gets around to mirroring: it re-reads, and v2 is what lands.
        mirror.mirror_one(store, e.entry_id, EXP, base_directory=share)
        path = mirror.entry_path(
            e2, mirror.logbook_root(DAY, EXP, base_directory=share)
        )
        assert path.read_text().rstrip().endswith("v2")
        assert store.unmirrored() == []

    def test_mirror_files_are_readable_by_others(
        self, store: NotesStore, share: Path
    ) -> None:
        """A mirror is for people: it gets the mode a plain write would."""
        import stat

        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        md = mirror.write_entry(e, root)
        link = mirror.write_attachment(e, "t.png", b"\x89PNG", root)
        expected = 0o666 & ~mirror._UMASK
        for p in (md, mirror.entry_dir(e, root) / link):
            assert stat.S_IMODE(p.stat().st_mode) == expected, p

    def test_parallel_same_name_uploads_all_get_their_own_file(
        self, store: NotesStore, share: Path
    ) -> None:
        """Four pastes of image.png in flight at once: four files, four names."""
        from concurrent.futures import ThreadPoolExecutor

        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        with ThreadPoolExecutor(4) as pool:
            links = list(
                pool.map(
                    lambda i: mirror.write_attachment(
                        e, "image.png", bytes([i]) * 8, root
                    ),
                    range(4),
                )
            )
        names = sorted(link.rsplit("/", 1)[-1] for link in links)
        assert names == ["image-2.png", "image-3.png", "image-4.png", "image.png"]
        folder = mirror.entry_dir(e, root) / mirror.ATTACHMENTS_DIR / e.entry_id
        assert sorted(p.name for p in folder.iterdir()) == names

    def test_failed_attachment_write_leaves_no_placeholder(
        self, store: NotesStore, share: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A claimed name whose bytes never land is released, not served empty."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        root = mirror.logbook_root(DAY, EXP, base_directory=share)

        def hiccup(path: Path, data: bytes) -> None:
            raise OSError("share hiccup")

        monkeypatch.setattr(mirror, "_replace_with", hiccup)
        with pytest.raises(mirror.MirrorUnavailable):
            mirror.write_attachment(e, "hic.png", b"\x89PNG", root)
        folder = mirror.entry_dir(e, root) / mirror.ATTACHMENTS_DIR / e.entry_id
        assert not (folder / "hic.png").exists()
        monkeypatch.undo()
        # The retry gets the original name, not hic-2.png.
        assert mirror.write_attachment(e, "hic.png", b"\x89PNG", root).endswith(
            "/hic.png"
        )
