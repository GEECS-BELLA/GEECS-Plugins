"""The routes, over fake cameras: no CA, no IOC."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from geecs_panels.app import create_app
from geecs_panels.camera import counter_suffix
from geecs_panels.feed import CameraFeed, FrameSource, FrameUnavailable
from geecs_panels.registry import Device

CAM = Device("ADTEST_CAM1", "aravis_camera", "undulator:adtest_cam1:")
OTHER = Device("SOME_STAGE", "motor_stage", "undulator:some_stage:")


class _Sub:
    def close(self) -> None:
        pass


class FakeCameras:
    def __init__(self, frame: bytes | Exception = b"\xff\xd8jpeg") -> None:
        self.frame = frame
        self.feeds: dict[str, CameraFeed] = {}

    def feed(self, device: Device) -> CameraFeed:
        if device.name not in self.feeds:
            feed = CameraFeed(device.prefix, lambda pv, s, cb: _Sub())
            feed.start()
            feed.update("cam1:DetectorState_RBV", "Acquire")
            feed.update(counter_suffix("MJPG1"), 1)
            self.feeds[device.name] = feed
        return self.feeds[device.name]

    def frames(self, device: Device) -> FrameSource:
        frame = self.frame

        async def resolve() -> str:
            return "http://ioc/x.jpg"

        async def fetch(url: str) -> bytes:
            if isinstance(frame, Exception):
                raise frame
            return frame

        return FrameSource(resolve, fetch, 0.1)


def _client(devices=(CAM, OTHER), cameras=None, **kwargs) -> TestClient:
    return TestClient(
        create_app("Undulator", list(devices), cameras or FakeCameras(), **kwargs)
    )


def test_the_landing_page_links_cameras_and_lists_kinds_without_a_panel() -> None:
    html = _client().get("/").text
    assert 'href="/camera/ADTEST_CAM1"' in html
    assert "SOME_STAGE" in html and "no panel for this kind yet" in html


def test_an_empty_registry_says_where_it_looked(tmp_path: Path) -> None:
    html = (
        _client(devices=(), registry_path=tmp_path / "native_devices.yaml")
        .get("/")
        .text
    )
    assert 'data-state="empty"' in html and "native_devices.yaml" in html


def test_a_broken_registry_is_an_error_not_an_empty_page() -> None:
    html = _client(devices=(), registry_error="bad yaml").get("/").text
    assert 'data-state="error"' in html and "bad yaml" in html


def test_the_camera_page_renders_every_reading() -> None:
    response = _client().get("/camera/ADTEST_CAM1")
    assert response.status_code == 200
    assert 'data-key="cx"' in response.text and 'data-key="raw_max"' in response.text
    assert 'data-frame-url="/camera/ADTEST_CAM1/frame.jpg"' in response.text


@pytest.mark.parametrize("name", ["NOPE", "SOME_STAGE"])
def test_only_registered_cameras_have_a_camera_page(name: str) -> None:
    assert _client().get(f"/camera/{name}").status_code == 404


def test_the_frame_route_serves_the_jpeg_uncached() -> None:
    response = _client().get("/camera/ADTEST_CAM1/frame.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == b"\xff\xd8jpeg"


def test_a_frame_the_ioc_cannot_give_is_a_503_saying_why() -> None:
    cameras = FakeCameras(frame=FrameUnavailable("frame fetch failed: refused"))
    response = _client(cameras=cameras).get("/camera/ADTEST_CAM1/frame.jpg")
    assert response.status_code == 503
    assert "refused" in response.text


def test_events_once_sends_one_state_frame() -> None:
    with _client().stream("GET", "/camera/ADTEST_CAM1/events?once=1") as response:
        body = "".join(response.iter_text())
    event, data = body.strip().split("\n", 1)
    assert event == "event: state"
    snapshot = json.loads(data.removeprefix("data: "))
    state = next(r for r in snapshot["readings"] if r["key"] == "state")
    assert state == {"key": "state", "value": "Acquire", "state": "fresh", "age": None}
    assert snapshot["image"]["state"] == "stale"


def test_health() -> None:
    assert _client().get("/health").json() == {
        "ok": True,
        "experiment": "Undulator",
        "devices": 2,
    }
