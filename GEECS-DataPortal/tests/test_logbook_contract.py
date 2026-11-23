"""The Plot tab's logbook send against the logbook's own app, in process.

The logbook is built the way its own suite builds it (a temp share, a temp
SQLite store) and reached through the portal's ``logbook_client`` (a
``TestClient``), so a change to the routes, bodies or reply fields of the four
calls (:mod:`geecs_portal.logbook_send`: create, upload, read, patch) fails
this suite in the same PR. geecs-logbook is in the optional ``contract``
group (skipped without it): ``geecs_portal`` never imports it (the
``portal-edges`` contract).
"""

from __future__ import annotations

import importlib.util
import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_portal import logbook_send
from geecs_portal.app import create_app
from test_app import TEST_DAY, FakeCatalog

# Skip only when the peer is not installed at all: a peer that is installed
# but fails to import (a missing dependency of its own) must error, naming
# the module, not read as an absent group.
if importlib.util.find_spec("geecs_logbook") is None:
    pytest.skip(
        "the contract group is not installed: poetry install --with dev,contract",
        allow_module_level=True,
    )

URL = "http://logbook.test"
UID = "uid-002"  # TEST_DAY, Scan 002 in the fake catalog
SCAN = 2
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
DATA_URL = "data:image/png;base64," + base64.b64encode(_PNG).decode()
SETTINGS = {"x": "mono", "y": ["cam-MaxCounts"]}


@pytest.fixture
def logbook(tmp_path: Path) -> TestClient:
    from geecs_logbook.app import create_app as create_logbook

    app = create_logbook(
        "Undulator", base_directory=tmp_path, notes_db=tmp_path / "notes.sqlite"
    )
    return TestClient(app, base_url=URL)


@pytest.fixture
def portal(logbook: TestClient) -> TestClient:
    app = create_app(
        FakeCatalog(),
        default_experiment="Undulator",
        logbook_url=URL,
        logbook_client=logbook,
    )
    return TestClient(app)


def _send(portal: TestClient, **extra):
    body = {"author": "Ada", "image": DATA_URL, "caption": "charge"} | extra
    return portal.post(f"/api/run/{UID}/logbook", json=body)


def _entry(logbook: TestClient, entry_id: str) -> dict:
    reply = logbook.get(f"/api/entries/{entry_id}")
    assert reply.status_code == 200, reply.text
    return reply.json()


def test_a_plot_makes_an_entry_on_the_scan_with_its_settings(portal, logbook) -> None:
    sent = _send(
        portal,
        source_url="http://portal.test/plot?uid=uid-002",
        settings=SETTINGS,
        settings_name="default",
    )
    assert sent.status_code == 201, sent.text
    out = sent.json()
    assert out["appended"] is False
    assert out["link"].startswith(f"attachments/{out['entry_id']}/")
    assert out["settings_link"].endswith("-settings.json")
    stored = _entry(logbook, out["entry_id"])
    assert stored["day"] == TEST_DAY.isoformat()
    assert stored["book"] == "scans" and stored["scan"] == SCAN
    assert stored["author"] == "Ada" and stored["template"] == "blank"
    assert stored["kind"] == "note" and stored["status"] == "kept"
    body = stored["body_md"]
    assert body.startswith("[Plotted in the data portal](http://portal.test/plot")
    assert f"![charge]({out['link']})" in body
    assert out["settings_link"] in body
    assert len(stored["attachments"]) == 2
    assert logbook.get(f"/{out['link']}").content == _PNG


def test_a_second_plot_joins_the_same_entry(portal, logbook) -> None:
    first = _send(portal).json()
    second = _send(portal, entry=first["entry_id"], caption="again")
    assert second.status_code == 201, second.text
    out = second.json()
    assert out["entry_id"] == first["entry_id"] and out["appended"] is True
    stored = _entry(logbook, out["entry_id"])
    assert f"![charge]({first['link']})" in stored["body_md"]
    assert f"![again]({out['link']})" in stored["body_md"]
    assert len(stored["attachments"]) == 2


def test_a_deleted_entry_gets_a_fresh_one(portal, logbook) -> None:
    """The upload's 404 on a gone entry is what the stale-pointer fallback reads."""
    first = _send(portal).json()
    assert logbook.delete(f"/api/entries/{first['entry_id']}").status_code == 204
    out = _send(portal, entry=first["entry_id"]).json()
    assert out["entry_id"] != first["entry_id"] and out["appended"] is False


def test_a_required_entry_that_is_gone_is_a_404_and_nothing_new(
    portal, logbook
) -> None:
    first = _send(portal).json()
    logbook.delete(f"/api/entries/{first['entry_id']}")
    refused = _send(portal, entry=first["entry_id"], entry_required=True)
    assert refused.status_code == 404
    feed = logbook.get("/api/entries", params={"since": "1970-01-01T00:00:00+00:00"})
    assert [e["entry_id"] for e in feed.json()["entries"]] == [first["entry_id"]]


class _EditedUnderUs(TestClient):
    """The logbook's client, letting someone else save the entry before our first PATCH."""

    edits = 0

    def patch(self, url, **kwargs):
        if self.edits == 0:
            self.edits += 1
            current = super().get(url).json()
            saved = super().patch(
                url,
                json={
                    "body_md": current["body_md"] + "\nA colleague's line.\n",
                    "editor": "Grace",
                    "expected_version": current["version"],
                },
            )
            assert saved.status_code == 200, saved.text
        reply = super().patch(url, **kwargs)
        self.statuses.append(reply.status_code)
        return reply


def test_a_conflicting_save_is_the_409_the_retry_reads(logbook) -> None:
    """A PATCH with the version read before someone else's save is refused, then re-read."""
    client = _EditedUnderUs(logbook.app, base_url=URL)
    client.statuses = []
    result = logbook_send.send_plot(
        base_url=URL,
        day=TEST_DAY.isoformat(),
        scan=SCAN,
        author="Ada",
        png=_PNG,
        caption="charge",
        client=client,
    )
    assert client.statuses == [409, 200]
    body = _entry(logbook, result.entry_id)["body_md"]
    assert "A colleague's line." in body
    assert f"![charge]({result.link})" in body
