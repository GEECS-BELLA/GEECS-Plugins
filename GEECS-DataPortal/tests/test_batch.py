"""The day page's batch analysis: the portal forwards to the analysis worker over HTTP."""

from __future__ import annotations

import json

import httpx
import pytest

from geecs_portal.batch import AutoStart, WorkerLink, WorkerUnavailable
from test_app import TEST_DAY, FakeCatalog, _client

URL = "http://127.0.0.1:8600"


def fake_worker(seen: list):
    """An httpx transport answering like scan-analysis-worker."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        seen.append((request.method, request.url.path, body))
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"ok": True})
        if path == "/api/groups":
            return httpx.Response(
                200, json={"groups": [{"id": "PW/g", "members": ["A"]}]}
            )
        if path == "/api/plan":
            return httpx.Response(
                200, json={"total": 2, "tasks": [], "open_scans": [], "already_done": 0}
            )
        if path == "/api/jobs" and request.method == "POST":
            if body["group"] == "nope":
                return httpx.Response(404, json={"detail": "group 'nope': unknown"})
            return httpx.Response(202, json={"id": "j1", "state": "queued"})
        if path == "/api/jobs":
            return httpx.Response(200, json={"jobs": [{"id": "j1"}]})
        if path == "/api/jobs/j1/cancel":
            return httpx.Response(200, json={"id": "j1", "cancel_requested": True})
        return httpx.Response(404, json={"detail": "no such route"})

    return httpx.Client(transport=httpx.MockTransport(handle))


def down_client():
    def handle(request):
        raise httpx.ConnectError("refused", request=request)

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_without_a_worker_there_is_no_panel_and_no_routes():
    client = _client(FakeCatalog(), default_experiment="Undulator")
    page = client.get(f"/day/{TEST_DAY.isoformat()}")
    assert "Analyze this day" not in page.text
    assert client.get("/api/batch/groups").status_code == 404
    assert client.get("/api/batch/status").json()["configured"] is False


def test_the_routes_forward_and_name_the_portals_experiment():
    seen: list = []
    worker = WorkerLink(URL, client=fake_worker(seen))
    client = _client(FakeCatalog(), default_experiment="Undulator", worker=worker)
    assert "Analyze this day" in client.get(f"/day/{TEST_DAY.isoformat()}").text
    # another experiment's day: the worker runs the portal's own only
    other = client.get(f"/day/{TEST_DAY.isoformat()}?experiment=Thomson")
    assert "Analyze this day" not in other.text
    assert client.get("/api/batch/groups").json()["groups"][0]["id"] == "PW/g"
    job = client.post("/api/batch/jobs", json={"day": "2026-07-12", "group": "PW/g"})
    assert job.status_code == 202
    assert seen[-1] == (
        "POST",
        "/api/jobs",
        {"experiment": "Undulator", "day": "2026-07-12", "group": "PW/g"},
    )
    assert (
        client.post(
            "/api/batch/plan", json={"day": "2026-07-12", "group": "PW/g"}
        ).json()["total"]
        == 2
    )
    # the worker's own errors come back as they were
    bad = client.post("/api/batch/jobs", json={"day": "2026-07-12", "group": "nope"})
    assert bad.status_code == 404 and "unknown" in bad.json()["detail"]
    assert client.post("/api/batch/jobs/j1/cancel").json()["cancel_requested"] is True
    assert client.get("/api/batch/jobs").headers["cache-control"] == "no-store"
    assert client.get("/api/batch/status").json() == {
        "configured": True,
        "reachable": True,
        "autostart": False,
    }


def test_a_silent_worker_is_503_and_never_started_without_autostart():
    worker = WorkerLink(URL, client=down_client())
    client = _client(FakeCatalog(), default_experiment="Undulator", worker=worker)
    r = client.get("/api/batch/groups")
    assert r.status_code == 503
    assert "does not answer" in r.json()["detail"]
    assert worker._started is None


def test_autostart_only_for_this_hosts_address(tmp_path):
    auto = AutoStart(tmp_path, "Undulator")
    assert WorkerLink("http://127.0.0.1:8600", autostart=auto).can_autostart
    assert WorkerLink("http://localhost:8600/", autostart=auto).can_autostart
    assert not WorkerLink("http://10.0.0.5:8600", autostart=auto).can_autostart
    assert not WorkerLink("http://127.0.0.1:8600").can_autostart
    remote = WorkerLink("http://10.0.0.5:8600", autostart=auto, client=down_client())
    with pytest.raises(WorkerUnavailable):
        remote.call("GET", "/api/groups")
    assert remote._started is None
