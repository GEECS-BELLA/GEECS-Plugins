"""The markdown mirror: durable, relative-linked, in a tree of its own, never a scans/ writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from geecs_logbook import mirror
from geecs_logbook._fs import UMASK
from geecs_logbook.attachments import AttachmentStore
from geecs_logbook.store import NotesStore

DAY = "2026-09-11"
EXP = "Undulator"


@pytest.fixture
def share(tmp_path: Path) -> Path:
    """A share with the day folder and its scans/ present, as the scanner leaves it.

    The experiment directory is what the mirror checks for; the day and
    its scans are there so the reader half of the tests has something.
    """
    scans = tmp_path / EXP / "Y2026" / "09-Sep" / "26_0911" / "scans"
    (scans / "Scan005").mkdir(parents=True)
    (scans / "Scan005" / "scan.log").write_text("")
    return tmp_path


@pytest.fixture
def store(tmp_path: Path) -> NotesStore:
    """A store beside the share."""
    return NotesStore(tmp_path / "notes.db")


@pytest.fixture
def blobs(tmp_path: Path) -> AttachmentStore:
    """The host-side attachment store, beside the database."""
    return AttachmentStore(tmp_path / "attachments")


def _snapshot(root: Path) -> set[Path]:
    return set(root.rglob("*"))


class TestPaths:
    """Where things go. Nothing here touches the disk."""

    def test_logbook_owns_its_own_tree(self, share: Path) -> None:
        """{experiment}/logbook/Y/M/D — the data tree's date shape, outside it."""
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        assert root == share / EXP / "logbook" / "Y2026" / "09-Sep" / "26_0911"
        assert "scans" not in root.parts

    def test_a_share_root_containing_scans_is_fine(self, tmp_path: Path) -> None:
        """Only the mirror's own segments are inspected; /mnt/scans/data is a valid site."""
        base = tmp_path / "scans" / "data"
        (base / EXP).mkdir(parents=True)
        root = mirror.logbook_root(DAY, EXP, base_directory=base)
        assert root == base / EXP / "logbook" / "Y2026" / "09-Sep" / "26_0911"

    def test_missing_experiment_directory_means_unmounted(self, tmp_path: Path) -> None:
        """With no experiment directory the share is not there; nothing is built."""
        with pytest.raises(mirror.MirrorUnavailable, match="not mounted"):
            mirror.logbook_root(DAY, EXP, base_directory=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_a_path_into_the_data_tree_is_refused(self, share: Path) -> None:
        """The invariant is pinned in code, not only by construction."""
        with pytest.raises(mirror.MirrorUnavailable):
            mirror._assert_own_tree(
                share / EXP / "Y2026" / "09-Sep" / "26_0911" / "scans"
            )
        with pytest.raises(mirror.MirrorUnavailable):
            mirror._assert_own_tree(share / EXP / "logbook" / "x" / "scans")

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

    def test_writes_into_the_logbook_tree(self, store: NotesStore, share: Path) -> None:
        """The file lands under logbook/ and the data tree is untouched."""
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        data_tree = share / EXP / "Y2026"
        before = _snapshot(data_tree)
        e = store.create(day=DAY, scan=5, author="a", body_md="hello")
        path = mirror.write_entry(e, root)
        assert path.is_file() and path.read_text().endswith("hello\n")
        assert not path.with_name(path.name + ".tmp").exists()  # atomic, no debris
        assert _snapshot(data_tree) == before

    def test_writes_a_day_the_scanner_never_made(
        self, store: NotesStore, tmp_path: Path
    ) -> None:
        """A note on a day with no scans has a home: the tree is the logbook's."""
        (tmp_path / EXP).mkdir()  # the share is mounted; the year has no days yet
        root = mirror.logbook_root("2026-09-12", EXP, base_directory=tmp_path)
        e = store.create(day="2026-09-12", author="a", body_md="quiet day", book="ops")
        path = mirror.write_entry(e, root)
        assert path.is_file()
        assert not (tmp_path / EXP / "Y2026").exists()  # no day folder was made

    def test_never_creates_anything_in_the_data_tree(
        self,
        store: NotesStore,
        share: Path,
        blobs: AttachmentStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every mkdir the mirror issues — markdown and attachments — is under logbook/."""
        real_mkdir = Path.mkdir
        made: list[Path] = []

        def guarded(self: Path, *a: object, **k: object) -> None:
            assert "logbook" in self.parts and "scans" not in self.parts, self
            made.append(self)
            return real_mkdir(self, *a, **k)

        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        blobs.save(e.entry_id, "a.png", b"x")
        monkeypatch.setattr(Path, "mkdir", guarded)
        mirror.write_entry(e, root)
        mirror.mirror_attachments(e, root, blobs)
        assert len(made) >= 2

    def test_attachments_are_copied_beside_the_entry(
        self, store: NotesStore, share: Path, blobs: AttachmentStore
    ) -> None:
        """Bytes stored on the host land in attachments/<id>/; a second pass copies nothing."""
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        blobs.save(e.entry_id, "trace.png", b"\x89PNGdata")
        assert mirror.mirror_attachments(e, root, blobs) == 1
        target = mirror.entry_dir(e, root) / mirror.attachment_link(e, "trace.png")
        assert target.read_bytes() == b"\x89PNGdata"
        assert mirror.mirror_attachments(e, root, blobs) == 0

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

    def test_a_dayless_day_lands_too(self, store: NotesStore, share: Path) -> None:
        """A note on a day the scanner never made is mirrored like any other."""
        store.create(day=DAY, scan=5, author="a", body_md="now")
        store.create(day="2026-09-12", author="a", body_md="tomorrow", book="ops")
        assert mirror.sync(store, EXP, base_directory=share) == (2, 0)
        assert store.unmirrored() == []

    def test_sync_carries_attachments(
        self, store: NotesStore, share: Path, blobs: AttachmentStore
    ) -> None:
        """The sync copies stored bytes as well as the markdown."""
        e = store.create(day=DAY, scan=5, author="a", body_md="x")
        blobs.save(e.entry_id, "a.png", b"\x89PNG")
        assert mirror.sync(store, EXP, base_directory=share, attachments=blobs) == (
            1,
            0,
        )
        root = mirror.logbook_root(DAY, EXP, base_directory=share)
        assert (
            mirror.entry_dir(e, root) / "attachments" / e.entry_id / "a.png"
        ).is_file()

    def test_share_down_defers_and_keeps_the_words(
        self, store: NotesStore, share: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unresolvable share defers; nothing is lost; the queue rotates."""
        e = store.create(day=DAY, scan=5, author="a", body_md="safe")

        def down(*a: object, **k: object) -> Path:
            raise mirror.MirrorUnavailable("unmounted")

        monkeypatch.setattr(mirror, "logbook_root", down)
        assert mirror.sync(store, EXP, base_directory=share) == (0, 1)
        owed = store.unmirrored()
        assert [x.entry_id for x in owed] == [e.entry_id] and store.get(
            e.entry_id
        ) is not None

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
        blobs = AttachmentStore(share / "blobs")
        blobs.save(e.entry_id, "t.png", b"\x89PNG")
        mirror.mirror_attachments(e, root, blobs)
        expected = 0o666 & ~UMASK
        for p in (md, mirror.entry_dir(e, root) / mirror.attachment_link(e, "t.png")):
            assert stat.S_IMODE(p.stat().st_mode) == expected, p
