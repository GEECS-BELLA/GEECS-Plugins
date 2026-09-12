"""The scan logbook mounted in the portal: opt-in, at /log, experiment-gated."""

from __future__ import annotations

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
