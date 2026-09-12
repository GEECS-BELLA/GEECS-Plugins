"""The scan logbook mounted in the portal: opt-in, at /log, experiment-gated.

Writes are a second opt-in (``notes_db``): without it the mount is the
read-only day view and the entry routes do not exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_portal.app import create_app
from test_app import FakeCatalog

pytest.importorskip("geecs_logbook")


class TestScanLogMount:
    """The logbook is off unless asked for, and needs an experiment."""

    def test_absent_by_default(self) -> None:
        """Without --scan-log the portal does not serve /log."""
        client = TestClient(create_app(FakeCatalog()))
        assert client.get("/log/api/day/2026-09-11").status_code == 404

    def test_mounted_when_requested(self) -> None:
        """With an experiment and the flag, /log answers."""
        client = TestClient(
            create_app(FakeCatalog(), default_experiment="Undulator", scan_log=True)
        )
        res = client.get("/log/api/day/2019-01-01")
        # No share in the test environment: an empty day or an honest 503,
        # never a 404 — the route exists.
        assert res.status_code in (200, 503)

    def test_not_mounted_without_an_experiment(self) -> None:
        """The logbook reads one experiment's share; it carries no default."""
        client = TestClient(create_app(FakeCatalog(), scan_log=True))
        assert client.get("/log/api/day/2026-09-11").status_code == 404


class TestNotesDb:
    """Writing is its own opt-in, and writes only where the charter allows."""

    _ENTRY = {"day": "2026-09-11", "author": "S. Barber", "body_md": "hello"}

    def test_read_only_without_a_notes_db(self) -> None:
        """--scan-log alone serves no entry routes."""
        client = TestClient(
            create_app(FakeCatalog(), default_experiment="Undulator", scan_log=True)
        )
        assert client.post("/log/api/entries", json=self._ENTRY).status_code in (
            404,
            405,
        )

    def test_writable_with_a_notes_db(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With --notes-db an entry is accepted and the words are in the file.

        The share is made unresolvable here on purpose — a developer's box
        may have the real one mounted, and this test must never write to
        it. The row lands, the markdown mirror is deferred, and the client
        sees 201 either way: a save never fails because the share is
        unreachable.
        """
        from geecs_logbook import mirror

        def no_share(*args: object, **kwargs: object) -> Path:
            raise mirror.MirrorUnavailable("no share in tests")

        monkeypatch.setattr(mirror, "logbook_root", no_share)
        db = tmp_path / "logbook.db"
        client = TestClient(
            create_app(
                FakeCatalog(),
                default_experiment="Undulator",
                scan_log=True,
                notes_db=db,
            )
        )
        res = client.post("/log/api/entries", json=self._ENTRY)
        assert res.status_code == 201, res.text
        assert db.is_file()
        listed = client.get("/log/api/day/2026-09-11/entries").json()
        assert [e["body_md"] for e in listed] == ["hello"]
        # Nothing was written anywhere but the database (and its attachment
        # directory beside it), and the mirror is owed.
        assert sorted(
            p.name for p in tmp_path.iterdir() if not p.name.startswith("logbook.db")
        ) == ["attachments"]
        from geecs_logbook.store import NotesStore

        assert [e.body_md for e in NotesStore(db).unmirrored()] == ["hello"]


class TestSeedTemplates:
    """The logbook's type buttons come from the configs checkout the portal reads."""

    def test_templates_dir_is_beside_the_analysis_tree(self, tmp_path: Path) -> None:
        """``<configs>/logbook_templates/`` — a sibling of --processing-configs."""
        configs = tmp_path / "configs"
        analysis = configs / "scan_analysis_configs"
        analysis.mkdir(parents=True)
        (configs / "logbook_templates").mkdir()
        (configs / "logbook_templates" / "laser.md").write_text(
            "---\nlabel: Laser\ncolour: ok\nbook: ops\n---\n#laser\n"
        )
        client = TestClient(
            create_app(
                FakeCatalog(),
                default_experiment="Undulator",
                processing_config_dir=analysis,
                scan_log=True,
                notes_db=tmp_path / "logbook.db",
            )
        )
        html = client.get("/log/month/2026-09").text
        assert 'data-type="laser"' in html and "tone-ok" in html

    def test_no_configs_tree_means_plain_composers(self, tmp_path: Path) -> None:
        """Without --processing-configs there is no checkout to read templates from."""
        client = TestClient(
            create_app(
                FakeCatalog(),
                default_experiment="Undulator",
                scan_log=True,
                notes_db=tmp_path / "logbook.db",
            )
        )
        html = client.get("/log/month/2026-09").text
        assert client.get("/log/month/2026-09").status_code == 200
        assert "typebtn" not in html
