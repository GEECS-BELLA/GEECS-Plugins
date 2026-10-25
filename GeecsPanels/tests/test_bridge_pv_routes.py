"""The raw-stream, inspector and composed-panel routes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from geecs_panels.app import create_app
from geecs_panels.bridge.pvstream import PvStreams
from geecs_panels.bridge.routes import Bridge
from tests._bridge import FakeCA, devices

A = "expt:x:field"


async def _info(pv: str) -> dict:
    if pv == "nope:pv":
        raise TimeoutError("no answer")
    nan = float("nan")
    return {
        "value": 1.5,
        "severity": "NO_ALARM",
        "units": "A",
        "precision": 3,
        # A record without alarm limits reports NaN.
        "alarm_limits": {"lolo": nan, "low": nan, "high": 4.0, "hihi": nan},
        "display_limits": (nan, nan),
    }


def _client(ca: FakeCA, panels: Path | None = None) -> TestClient:
    served = devices()
    bridge = Bridge(
        "expt",
        list(served.values()),
        [],
        pvs=PvStreams(ca.subscribe, push_s=0.01),
        read_info=_info,
        panels_dir=panels,
    )
    return TestClient(create_app("expt", [], object(), bridge=bridge))


def test_a_subscription_is_a_handle_and_its_stream_a_full_frame_first() -> None:
    ca = FakeCA(needs_loop=True)
    with _client(ca) as client:
        response = client.post("/api/subscriptions", json={"pvs": [A]})
        assert response.status_code == 200  # subscribed on the loop, as aioca needs
        ack = response.json()
        ca.post(A, 0.5)
        with client.stream("GET", f"/api/stream/{ack['handle']}?limit=1") as r:
            lines = [line for line in r.iter_lines() if line]
    assert ack["pvs"] == [A] and ack["grace_seconds"] == 30.0
    assert lines[1] == "event: pv"
    frame = json.loads(lines[2].removeprefix("data: "))
    assert frame["full"] and frame["pvs"][A]["value"] == 0.5


def test_a_bad_subscription_or_an_unknown_handle_is_refused() -> None:
    with _client(FakeCA()) as client:
        twice = client.post("/api/subscriptions", json={"pvs": [A, A]})
        none = client.post("/api/subscriptions", json={"pvs": []})
        unknown = client.get("/api/stream/h-nope")
    assert twice.status_code == 422 and none.status_code == 422
    assert unknown.status_code == 404


def test_the_inspector_reads_one_pv() -> None:
    with _client(FakeCA()) as client:
        up = client.get(f"/api/pv/{A}").json()
        down = client.get("/api/pv/nope:pv").json()
    assert up["connected"] and up["units"] == "A" and up["value"] == 1.5
    assert up["alarm_limits"] == {"lolo": None, "low": None, "high": 4.0, "hihi": None}
    assert up["display_limits"] is None
    assert not down["connected"] and "no answer" in down["error"]


PANEL = """
title: Chicane
sections:
  - title: Supply
    columns: 2
    widgets:
      - kind: device
        device: PS1
        roles: [current, current_sp]
"""

BROKEN = """
title: Broken
sections:
  - title: S
    columns: 1
    widgets:
      - kind: device
        device: Nope
"""

OVERRIDE = """
family: fastps
hide: [loop_mode]
"""


def test_the_panels_that_fit_are_served_and_the_others_named(tmp_path: Path) -> None:
    (tmp_path / "overrides").mkdir()
    (tmp_path / "chicane.yaml").write_text(PANEL)
    (tmp_path / "broken.yaml").write_text(BROKEN)
    (tmp_path / "overrides" / "fastps.yaml").write_text(OVERRIDE)
    (tmp_path / "overrides" / "easydriver.yaml").write_text(OVERRIDE)
    with _client(FakeCA(), tmp_path) as client:
        body = client.get("/api/panels").json()
    assert set(body["panels"]) == {"chicane"}
    assert body["panels"]["chicane"]["title"] == "Chicane"
    assert set(body["overrides"]) == {"fastps"}
    assert any("broken.yaml" in e and "Nope" in e for e in body["errors"])
    assert any("easydriver.yaml" in e for e in body["errors"])


def test_no_panels_folder_is_no_panels() -> None:
    with _client(FakeCA()) as client:
        body = client.get("/api/panels").json()
    assert body == {"panels": {}, "overrides": {}, "errors": []}


def test_a_panel_file_that_is_not_utf8_is_named_not_a_500(tmp_path: Path) -> None:
    (tmp_path / "chicane.yaml").write_text(PANEL)
    (tmp_path / "windows.yaml").write_bytes("title: 5 \xb5m\n".encode("cp1252"))
    with _client(FakeCA(), tmp_path) as client:
        response = client.get("/api/panels")
    assert response.status_code == 200
    body = response.json()
    assert set(body["panels"]) == {"chicane"}
    assert any("windows.yaml" in e for e in body["errors"])
