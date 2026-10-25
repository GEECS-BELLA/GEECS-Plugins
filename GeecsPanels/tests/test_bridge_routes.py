"""The bridge's /api routes, over the fake transport: listings, the device stream, arrays, the frame."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from geecs_panels.app import create_app
from geecs_panels.bridge.routes import Bridge
from geecs_panels.bridge.stream import DeviceStreams
from geecs_panels.contracts import DeviceFrame
from geecs_panels.feed import FrameSource
from geecs_panels.registry import Device
from tests._bridge import CAM, SUPPLY, FakeCA, devices, healthy_supply


class _Reads:
    """A fake on-request read: counts reads, answers per PV."""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers, self.calls = answers, 0

    async def __call__(self, pvs: list[str]) -> list[object]:
        self.calls += 1
        await asyncio.sleep(0.01)
        return [self.answers[pv] for pv in pvs]


def _client(
    ca: FakeCA, reads: _Reads | None = None, cameras: object | None = None
) -> TestClient:
    served = devices()
    streams = DeviceStreams(served, ca.subscribe, read=reads, push_s=0.01)
    bridge = Bridge("expt", list(served.values()), ["a broken entry"], streams)
    camera = Device("Cam", "aravis_camera", CAM)
    return TestClient(create_app("expt", [camera], cameras or object(), bridge=bridge))


def _frames(
    client: TestClient, url: str, headers: dict | None = None
) -> Iterator[dict]:
    with client.stream("GET", url, headers=headers or {}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        event: dict = {}
        for line in response.iter_lines():
            if line.startswith("id: "):
                event["id"] = int(line[4:])
            elif line.startswith("event: "):
                event["event"] = line[7:]
            elif line.startswith("data: "):
                event["data"] = json.loads(line[6:])
            elif not line and event:
                yield event
                event = {}


def test_the_devices_are_listed_with_their_resolved_roles_and_the_errors() -> None:
    with _client(FakeCA()) as client:
        body = client.get("/api/devices").json()
    assert body["experiment"] == "expt"
    assert body["errors"] == ["a broken entry"]
    rows = {d["name"]: d for d in body["devices"]}
    assert rows["Cam"]["family"] == "ad_camera_chain"
    assert rows["Stage"]["axes"] == ["m1", "m2"]
    assert rows["PS1"]["source"] == "panel_devices"
    assert rows["PS1"]["prefix"] == SUPPLY
    assert rows["PS1"]["actions"] == ["current", "output", "reset"]
    assert any(
        r["name"] == "output_sp" and r["confirm"] == "hazard"
        for r in rows["PS1"]["roles"]
    )


def test_the_families_are_served() -> None:
    with _client(FakeCA()) as client:
        body = client.get("/api/families").json()
    assert {"motor_record", "ad_camera_chain", "fastps"} <= set(body)
    assert body["motor_record"]["per_axis"] is True


def test_an_unknown_device_is_a_404_before_any_stream() -> None:
    with _client(FakeCA()) as client:
        assert client.get("/api/devices/Nope/events").status_code == 404
        assert client.get("/api/devices/Nope/arrays").status_code == 404


def test_the_stream_is_a_full_frame_then_deltas_with_seq_as_the_id() -> None:
    ca = FakeCA()
    with _client(ca) as client:
        frames = _frames(client, "/api/devices/PS1/events?limit=2")
        first = next(frames)
        healthy_supply(ca)
        second = next(frames)
    assert first["event"] == "device" and first["data"]["full"] is True
    assert first["id"] == first["data"]["seq"]
    assert first["data"]["roles"]["current"]["freshness"] == "unknown"
    assert second["data"]["full"] is False
    assert second["id"] == first["id"] + 1
    assert second["data"]["roles"]["current"]["value"] == 0.0
    for event in (first, second):
        DeviceFrame.model_validate(event["data"])


def test_a_nan_reading_travels_as_null_and_the_stream_lives() -> None:
    """Rule 10 on the wire: the next frame still arrives."""
    ca = FakeCA()
    with _client(ca) as client:
        frames = _frames(client, "/api/devices/PS1/events?limit=3")
        next(frames)
        ca.post(SUPPLY + "current", math.nan)
        nan = next(frames)
        ca.post(SUPPLY + "current", 0.5)
        after = next(frames)
    assert nan["data"]["roles"]["current"]["value"] is None
    assert after["data"]["roles"]["current"]["value"] == 0.5


def test_a_last_event_id_the_bridge_holds_resumes_without_a_full_frame() -> None:
    ca = FakeCA()
    with _client(ca) as client:
        frames = _frames(client, "/api/devices/PS1/events?limit=1")
        full = next(frames)
        list(frames)
        ca.post(SUPPLY + "current", 0.25)
        resumed = next(
            _frames(
                client,
                "/api/devices/PS1/events?limit=1",
                {"Last-Event-ID": str(full["id"])},
            )
        )
        restarted = next(
            _frames(client, "/api/devices/PS1/events?limit=1", {"Last-Event-ID": "7"})
        )
    assert resumed["data"]["full"] is False
    assert resumed["id"] == full["id"] + 1
    assert restarted["data"]["full"] is True


def test_arrays_are_read_on_request_once_per_interval_and_decimated() -> None:
    """Rule 3: shared, at most one read per interval, min/max to the columns asked."""
    profile = [float(i % 7) for i in range(1000)]
    answers = {
        CAM + "STAT:ProfileAverageX_RBV": profile,
        CAM + "STAT:ProfileAverageY_RBV": profile,
        CAM + "STAT:ProfileCursorX_RBV": profile,
        CAM + "STAT:ProfileCursorY_RBV": profile,
        CAM + "STAT:Histogram_RBV": [0.0, 5.0, math.nan],
    }
    reads, ca = _Reads(answers), FakeCA()
    with _client(ca, reads) as client:
        first = client.get("/api/devices/Cam/arrays?columns=100").json()
        ca.post(CAM + "STAT:ProfileSizeX_RBV", 600)  # the feed exists now
        second = client.get("/api/devices/Cam/arrays?columns=100").json()
    assert reads.calls == 1  # the second ask shares the first read
    assert first["traces"]["profile_x"]["x"][-1] > 900  # the whole array, unsized
    traces = second["traces"]
    assert len(traces["profile_x"]["y"]) <= 200
    assert traces["profile_x"]["x"][-1] < 600  # cut to the valid length
    assert traces["histogram"]["y"] == [0.0, 5.0, None]


def test_a_failed_array_read_says_why() -> None:
    class Failing:
        async def __call__(self, pvs: list[str]) -> list[object]:
            raise TimeoutError("no answer")

    with _client(FakeCA(), Failing()) as client:  # type: ignore[arg-type]
        body = client.get("/api/devices/Cam/arrays").json()
    assert body["traces"] == {}
    assert "no answer" in body["error"]


def test_the_api_frame_and_the_camera_frame_share_one_upstream() -> None:
    """Rule 2: however it is asked for, one fetch in flight per camera."""
    fetches: list[str] = []

    class Cameras:
        def __init__(self) -> None:
            self.source: FrameSource | None = None

        def frames(self, device: Device) -> FrameSource:
            if self.source is None:

                async def resolve() -> str:
                    return "http://ioc/x.jpg"

                async def fetch(url: str) -> bytes:
                    fetches.append(url)
                    await asyncio.sleep(0.05)
                    return b"\xff\xd8jpeg"

                self.source = FrameSource(resolve, fetch, 10.0)
            return self.source

    with _client(FakeCA(), cameras=Cameras()) as client:
        a = client.get("/api/frame/Cam")
        b = client.get("/camera/Cam/frame.jpg")
    assert a.status_code == b.status_code == 200
    assert a.content == b.content == b"\xff\xd8jpeg"
    assert len(fetches) == 1


@pytest.mark.parametrize(
    "path",
    [
        "/api/devices",
        "/api/families",
        "/api/devices/{name}/events",
        "/api/frame/{name}",
    ],
)
def test_the_routes_have_stable_operation_ids(path: str) -> None:
    with _client(FakeCA()) as client:
        doc = client.get("/openapi.json").json()
    op = doc["paths"][path]["get"]["operationId"]
    assert op == "get" + path.replace("/", "_").replace("{", "").replace("}", "")
