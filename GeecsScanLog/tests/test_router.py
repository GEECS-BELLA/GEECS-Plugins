"""The mounted logbook router: routes, status codes and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from geecs_scan_log.router import create_log_router


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
