"""The run page's link to the scan's card in the logbook — a peer service.

The portal no longer mounts the logbook (GeecsLogbook 0.10.0 is its own
process on its own port); it only knows the logbook's base URL and builds
``<base>/day/YYYY-MM-DD#ScanNNN`` from it. No flag, no link.
"""

from __future__ import annotations

import dataclasses

from fastapi.testclient import TestClient

from geecs_portal.app import create_app
from test_app import FakeCatalog

_UID = "uid-002"  # TEST_DAY, Scan 002 in the fake catalog


class TestRunPageLink:
    """The link exists exactly when a logbook URL is configured."""

    def test_absolute_url_is_used_verbatim(self) -> None:
        """Until the front door exists the logbook is another host:port."""
        client = TestClient(
            create_app(
                FakeCatalog(),
                default_experiment="Undulator",
                logbook_url="http://logbook.example:8400/",
            )
        )
        html = client.get(f"/run/{_UID}").text
        assert 'href="http://logbook.example:8400/day/2026-07-12#Scan002"' in html
        payload = client.get(f"/api/run/{_UID}").json()
        assert (
            payload["logbook"] == "http://logbook.example:8400/day/2026-07-12#Scan002"
        )

    def test_path_is_same_origin_and_carries_the_proxy_prefix(self) -> None:
        """Behind the front door the logbook is ``/log`` on the portal's origin."""
        client = TestClient(
            create_app(
                FakeCatalog(), default_experiment="Undulator", logbook_url="/log"
            )
        )
        assert 'href="/log/day/2026-07-12#Scan002"' in client.get(f"/run/{_UID}").text
        html = client.get(
            f"/run/{_UID}", headers={"X-Forwarded-Prefix": "/portal"}
        ).text
        assert 'href="/portal/log/day/2026-07-12#Scan002"' in html

    def test_absent_without_a_logbook_url(self) -> None:
        client = TestClient(create_app(FakeCatalog(), default_experiment="Undulator"))
        html = client.get(f"/run/{_UID}").text
        assert "/day/2026-07-12#Scan002" not in html
        assert client.get(f"/api/run/{_UID}").json()["logbook"] is None

    def test_the_portal_serves_no_log_routes(self) -> None:
        """The mount is gone, not hidden: /log is nobody's route here."""
        client = TestClient(
            create_app(
                FakeCatalog(), default_experiment="Undulator", logbook_url="/log"
            )
        )
        assert client.get("/log/api/day/2026-07-12").status_code == 404
        assert client.get("/log/day/2026-07-12").status_code == 404

    def test_absent_for_a_run_from_another_experiment(self) -> None:
        """The logbook is one experiment's; scan numbers restart per experiment."""
        catalog = FakeCatalog()
        detail = catalog.details[_UID]
        catalog.details[_UID] = dataclasses.replace(
            detail, summary=dataclasses.replace(detail.summary, experiment="Thomson")
        )
        client = TestClient(
            create_app(catalog, default_experiment="Undulator", logbook_url="/log")
        )
        assert "/day/2026-07-12#Scan002" not in client.get(f"/run/{_UID}").text
        assert client.get(f"/api/run/{_UID}").json()["logbook"] is None

    def test_absent_without_a_default_experiment(self) -> None:
        """No experiment named means no way to know whose logbook it is."""
        client = TestClient(create_app(FakeCatalog(), logbook_url="/log"))
        assert client.get(f"/api/run/{_UID}").json()["logbook"] is None
