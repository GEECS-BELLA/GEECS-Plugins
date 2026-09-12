"""The mounted logbook router: routes, status codes and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from geecs_logbook.router import create_log_router


@pytest.fixture
def client(share: Path) -> TestClient:
    """Mount the router the way the Data Portal does."""
    app = FastAPI()
    app.include_router(
        create_log_router("Undulator", base_directory=share), prefix="/log"
    )
    return TestClient(app)


class TestDayJson:
    """The JSON peer of the day page."""

    def test_returns_the_days_scans(self, client: TestClient) -> None:
        """Every scan folder present appears, in number order."""
        body = client.get("/log/api/day/2026-09-11").json()
        assert [s["number"] for s in body["scans"]] == [1, 6, 31, 40]
        assert body["exists"] is True

    def test_carries_status_and_failure_reason(self, client: TestClient) -> None:
        """A failed scan exposes why it failed."""
        scans = client.get("/log/api/day/2026-09-11").json()["scans"]
        failed = next(s for s in scans if s["number"] == 6)
        assert failed["status"] == "failed"
        assert "uc_amp4_ir_input-hdf-capture" in failed["failure_reason"]

    def test_empty_day_is_not_an_error(self, client: TestClient) -> None:
        """A date with no scans returns 200 and an empty day."""
        res = client.get("/log/api/day/2019-01-01")
        assert res.status_code == 200
        assert res.json()["exists"] is False

    def test_malformed_date_is_a_400(self, client: TestClient) -> None:
        """A path segment that is not YYYY-MM-DD is the caller's error."""
        assert client.get("/log/api/day/yesterday").status_code == 400


class TestDayPage:
    """The rendered day document."""

    def test_renders_each_scan(self, client: TestClient) -> None:
        """Every scan gets a collapsible block anchored by its label."""
        html = client.get("/log/day/2026-09-11").text
        assert 'id="Scan001"' in html
        assert 'id="Scan006"' in html
        assert 'id="Scan031"' in html

    def test_shows_the_purpose_from_scan_start_info(self, client: TestClient) -> None:
        """ScanStartInfo is displayed rather than asked for again."""
        html = client.get("/log/day/2026-09-11").text
        assert "807 phase 1 acceptance" in html

    def test_shows_the_failure_reason(self, client: TestClient) -> None:
        """The failure strip carries the ScanEndInfo text."""
        html = client.get("/log/day/2026-09-11").text
        assert "Scan ended with an error" in html
        assert "uc_amp4_ir_input-hdf-capture" in html

    def test_serves_its_stylesheet(self, client: TestClient) -> None:
        """The mounted static files resolve under the router prefix."""
        res = client.get("/log/static/scanlog.css")
        assert res.status_code == 200
        assert "--accent" in res.text

    def test_small_day_opens_expanded(self, client: TestClient) -> None:
        """Under the threshold every scan block starts open."""
        html = client.get("/log/day/2026-09-11").text
        assert html.count('<details class="card scan"') == 4
        assert "Collapse all" in html


class TestBusyDay:
    """A day over the grouping threshold renders campaigns, not a flat list."""

    @pytest.fixture
    def busy(self, make_run) -> TestClient:
        """Build a day of 25 identical scans — one campaign."""
        root = make_run(25)
        app = FastAPI()
        app.include_router(
            create_log_router("Undulator", base_directory=root), prefix="/log"
        )
        return TestClient(app)

    def test_groups_into_campaigns(self, busy: TestClient) -> None:
        """Twenty-five scans render inside one campaign block."""
        html = busy.get("/log/day/2026-09-11").text
        assert html.count('<details class="campaign"') == 1
        assert html.count('<details class="card scan"') == 25

    def test_rail_lists_campaigns_not_scans(self, busy: TestClient) -> None:
        """The rail shows one row per campaign so it stays scannable."""
        html = busy.get("/log/day/2026-09-11").text
        assert html.count('class="scanrow"') == 1
        assert "Campaigns" in html

    def test_busy_day_starts_collapsed(self, busy: TestClient) -> None:
        """Nothing is expanded on arrival; the button offers Expand all."""
        html = busy.get("/log/day/2026-09-11").text
        assert " open>" not in html
        assert "Expand all" in html


class TestHonestChips:
    """The status chip must not contradict the card under it."""

    def test_unfinalised_scan_does_not_claim_to_lack_scan_info(
        self, client: TestClient
    ) -> None:
        """Scan040 has a full ScanInfo, just no ScanEndInfo yet.

        Both it and a folder with no ScanInfo at all are `incomplete`, but
        rendering "no scan info" over a card listing the scan variable and
        shot count parsed *from* ScanInfo is a plain contradiction — and it
        hit the most common state on the real share.
        """
        html = client.get("/log/day/2026-09-11").text
        assert "not finalised" in html
        assert "no scan info" in html  # Scan031, which really has none
        assert html.count("not finalised") == 1
        assert html.count("no scan info") == 1


@pytest.fixture
def writable(share: Path, tmp_path: Path) -> TestClient:
    """The router with a notes store, as the portal mounts it with --notes-db."""
    app = FastAPI()
    app.include_router(
        create_log_router(
            "Undulator", base_directory=share, notes_db=tmp_path / "notes.db"
        ),
        prefix="/log",
    )
    return TestClient(app)


def _post(client: TestClient, **body: object) -> dict:
    payload = {"day": "2026-09-11", "author": "S. Barber", "body_md": "hello"}
    payload.update(body)
    r = client.post("/log/api/entries", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


class TestWriteApi:
    """The entry endpoints, as the composer and an agent use them."""

    def test_read_only_mount_has_no_write_routes(self, client: TestClient) -> None:
        """Without a store the write routes do not exist at all."""
        r = client.post(
            "/log/api/entries",
            json={"day": "2026-09-11", "author": "a", "body_md": "x", "scan": 1},
        )
        assert r.status_code in (404, 405)

    def test_day_level_entry_needs_no_scan(self, writable: TestClient) -> None:
        """Neither scan nor after: the entry is about the day, and lands at the root."""
        e = _post(writable)
        assert e["scan"] is None and e["after"] is None
        assert e["updated_at"] == e["created_at"]
        listed = writable.get("/log/api/day/2026-09-11/entries").json()
        assert [x["entry_id"] for x in listed] == [e["entry_id"]]
        page = writable.get("/log/day/2026-09-11").text
        assert "hello" in page

    def test_scan_zero_is_not_an_anchor(self, writable: TestClient) -> None:
        """Scans start at 1; 0 is refused rather than silently meaning the day."""
        r = writable.post(
            "/log/api/entries",
            json={"day": "2026-09-11", "author": "a", "body_md": "x", "scan": 0},
        )
        assert r.status_code == 422

    def test_both_anchors_is_a_422(self, writable: TestClient) -> None:
        """On a scan and after one at once is refused."""
        r = writable.post(
            "/log/api/entries",
            json={
                "day": "2026-09-11",
                "author": "a",
                "body_md": "x",
                "scan": 1,
                "after": 1,
            },
        )
        assert r.status_code == 422

    def test_an_agent_cannot_be_born_kept(self, writable: TestClient) -> None:
        """The draft rule holds at the API, not only in the docs."""
        r = writable.post(
            "/log/api/entries",
            json={
                "day": "2026-09-11",
                "author": "osprey",
                "body_md": "x",
                "scan": 1,
                "kind": "agent_analysis",
                "status": "kept",
            },
        )
        assert r.status_code == 422 and "draft" in r.text
        e = _post(
            writable, author="osprey", scan=1, kind="agent_analysis", status="draft"
        )
        assert e["status"] == "draft"
        kept = writable.post(
            f"/log/api/entries/{e['entry_id']}/status", json={"status": "kept"}
        ).json()
        assert kept["status"] == "kept" and kept["updated_at"] > e["updated_at"]

    def test_edit_conflict_carries_the_current_entry(
        self, writable: TestClient
    ) -> None:
        """A stale version is a 409 with what is there now."""
        e = _post(writable, scan=1)
        first = writable.patch(
            f"/log/api/entries/{e['entry_id']}",
            json={"body_md": "one", "editor": "a", "expected_version": 1},
        )
        assert first.status_code == 200 and first.json()["version"] == 2
        stale = writable.patch(
            f"/log/api/entries/{e['entry_id']}",
            json={"body_md": "two", "editor": "b", "expected_version": 1},
        )
        assert stale.status_code == 409
        assert stale.json()["detail"]["current"]["body_md"] == "one"

    def test_delete_hides_the_entry_everywhere(self, writable: TestClient) -> None:
        """After a delete: 204, then gone from the listing and from every route."""
        e = _post(writable, scan=1)
        assert writable.delete(f"/log/api/entries/{e['entry_id']}").status_code == 204
        assert writable.get("/log/api/day/2026-09-11/entries").json() == []
        assert writable.delete(f"/log/api/entries/{e['entry_id']}").status_code == 404
        again = writable.patch(
            f"/log/api/entries/{e['entry_id']}",
            json={"body_md": "x", "editor": "a", "expected_version": 2},
        )
        assert again.status_code == 404
        assert "hello" not in writable.get("/log/day/2026-09-11").text


_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class TestAttachments:
    """Upload lands on the share beside the entry; serving is contained."""

    def _upload(
        self,
        client: TestClient,
        entry_id: str,
        name: str = "shot.png",
        data: bytes = _PNG,
        ctype: str = "image/png",
    ):
        return client.post(
            f"/log/api/entries/{entry_id}/attachments",
            files={"file": (name, data, ctype)},
        )

    def test_upload_then_serve(self, writable: TestClient, share: Path) -> None:
        """201 with a relative link; the served route returns the bytes."""
        e = _post(writable, scan=1)
        r = self._upload(writable, e["entry_id"])
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["link"] == f"attachments/{e['entry_id']}/shot.png"
        assert body["attachment"]["id"] != e["entry_id"]  # its own id
        got = writable.get(
            f"/log/attachments/2026-09-11/Scan001/{e['entry_id']}/shot.png"
        )
        assert got.status_code == 200 and got.content == _PNG
        listed = writable.get("/log/api/day/2026-09-11/entries").json()
        assert [a["filename"] for a in listed[0]["attachments"]] == ["shot.png"]
        assert listed[0]["version"] == 2

    def test_same_name_twice_is_numbered_not_overwritten(
        self, writable: TestClient
    ) -> None:
        """Every clipboard paste is image.png; the second must not eat the first."""
        e = _post(writable, scan=1)
        first = self._upload(writable, e["entry_id"], "image.png", _PNG)
        second = self._upload(writable, e["entry_id"], "image.png", _PNG + b"2")
        assert first.json()["link"].endswith("/image.png")
        assert second.json()["link"].endswith("/image-2.png")
        base = f"/log/attachments/2026-09-11/Scan001/{e['entry_id']}"
        assert writable.get(f"{base}/image.png").content == _PNG
        assert writable.get(f"{base}/image-2.png").content == _PNG + b"2"
        ids = {
            a["id"]
            for a in writable.get("/log/api/day/2026-09-11/entries").json()[0][
                "attachments"
            ]
        }
        assert len(ids) == 2

    def test_type_and_size_limits(self, writable: TestClient) -> None:
        """415 for a type we do not take; 413 over the cap; 422 for nothing."""
        e = _post(writable, scan=1)
        assert (
            self._upload(
                writable, e["entry_id"], "x.exe", b"MZ", "application/octet-stream"
            ).status_code
            == 415
        )
        big = b"\x00" * (20 * 1024 * 1024 + 1)
        assert self._upload(writable, e["entry_id"], "big.png", big).status_code == 413
        assert self._upload(writable, e["entry_id"], "e.png", b"").status_code == 422
        assert self._upload(writable, "nope").status_code == 404

    def test_serving_is_contained(self, writable: TestClient) -> None:
        """No path escapes the entry's attachment directory."""
        e = _post(writable, scan=1)
        self._upload(writable, e["entry_id"])
        for bad in (
            "../../Scan001/shot.png",
            "..%2F..%2Fshot.png",
            "../../../scans/Scan001/ScanInfoScan001.ini",
        ):
            r = writable.get(
                f"/log/attachments/2026-09-11/Scan001/{e['entry_id']}/{bad}"
            )
            assert r.status_code == 404, bad

    def test_unresolvable_share_is_a_503(
        self, writable: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Upload and serving say the share is down, rather than 500."""
        from geecs_logbook import mirror

        e = _post(writable, scan=1)

        def no_share(*a: object, **k: object) -> Path:
            raise mirror.MirrorUnavailable("gone")

        monkeypatch.setattr(mirror, "logbook_root", no_share)
        assert self._upload(writable, e["entry_id"]).status_code == 503
        assert (
            writable.get(
                f"/log/attachments/2026-09-11/Scan001/{e['entry_id']}/x.png"
            ).status_code
            == 503
        )
