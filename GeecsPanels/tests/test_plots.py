"""The plots: the shared arrays read, the history, the payload, the cursor route."""

from __future__ import annotations

import asyncio
import json
import time

import pytest


from geecs_panels.camera import counter_suffix
from geecs_panels.feed import (
    HISTORY_EVERY_S,
    ArraysSource,
    ArraysUnavailable,
    CameraFeed,
    plot_payload,
)
from tests.test_app import CAM, _client
from tests.test_feed import Clock
from tests.test_writes import RecordingCameras


class _Sub:
    def close(self) -> None:
        pass


def _feed(clock: Clock) -> CameraFeed:
    feed = CameraFeed(CAM.prefix, lambda pv, s, cb: _Sub(), clock=clock, wall=clock)
    feed.start()
    return feed


def _roi(feed: CameraFeed, minx=100, miny=200, sizex=400, sizey=300, binx=2, revx=0):
    for suffix, value in {
        "ROI1:MinX_RBV": minx,
        "ROI1:MinY_RBV": miny,
        "ROI1:SizeX_RBV": sizex,
        "ROI1:SizeY_RBV": sizey,
        "ROI1:BinX_RBV": binx,
        "ROI1:BinY_RBV": 1,
        "ROI1:ReverseX_RBV": revx,
        "ROI1:ReverseY_RBV": 0,
    }.items():
        feed.update(suffix, value)


def test_concurrent_viewers_share_one_arrays_read() -> None:
    reads: list[int] = []

    async def fetch() -> dict[str, list[float]]:
        reads.append(1)
        await asyncio.sleep(0.01)
        return {"histogram": [1.0]}

    async def run() -> dict[str, list[float]]:
        source = ArraysSource(fetch, 0.5, clock=Clock())
        for _ in range(4):
            source.kick()  # four viewers' pushes while the read runs
        await asyncio.sleep(0.05)
        source.kick()  # inside the interval: no new read
        return source.current()

    assert asyncio.run(run()) == {"histogram": [1.0]}
    assert len(reads) == 1


def test_a_failed_read_is_held_for_the_interval_not_retried_every_push() -> None:
    clock = Clock()
    reads: list[int] = []

    async def fetch() -> dict[str, list[float]]:
        reads.append(1)
        raise TimeoutError("caget timed out")

    async def run() -> None:
        source = ArraysSource(fetch, 0.5, clock=clock)
        source.kick()
        await asyncio.sleep(0.01)
        with pytest.raises(ArraysUnavailable, match="timed out"):
            source.current()
        source.kick()  # within the interval: the failure stands
        await asyncio.sleep(0.01)
        clock.now += 1
        source.kick()  # past it: one retry
        await asyncio.sleep(0.01)

    asyncio.run(run())
    assert len(reads) == 2


def test_history_is_sampled_on_stat_frames_decimated() -> None:
    """A steady centroid still makes points (the tick, not the value's monitor, samples)."""
    clock = Clock()
    feed = _feed(clock)
    feed.update("STAT:CentroidX_RBV", 640.0)
    feed.update(counter_suffix("STAT"), 0)
    for n in range(1, 41):  # 40 frames at 20 Hz = 2 s
        clock.now += 0.05
        feed.update(counter_suffix("STAT"), n)
    points = feed.history(60)
    # At most one point per HISTORY_EVERY_S (on 20 Hz frames: every 4th or 5th).
    assert 2 / (HISTORY_EVERY_S + 0.05) <= len(points) <= 2 / HISTORY_EVERY_S + 1
    assert all(p[1] == 640.0 for p in points)


def test_the_payload_maps_profiles_to_image_pixels() -> None:
    clock = Clock()
    feed = _feed(clock)
    _roi(feed)  # MinX 100, bin x2
    feed.update("STAT:ProfileSizeX_RBV", 3)
    feed.update("STAT:ProfileSizeY_RBV", 2)
    feed.update("STAT:CursorX_RBV", 1)
    feed.update("STAT:CursorY_RBV", 0)
    arrays = {
        "profile_x": [1.0, 2.0, 3.0, 99.0],  # longer than the valid size
        "profile_y": [5.0, float("nan"), 7.0],
        "cursor_x": [0.0, 0.0, 0.0],
        "cursor_y": [0.0, 0.0],
        "histogram": [10.0, 0.0],
    }
    p = plot_payload(feed, arrays, 60)
    assert p["profile_x"] == [1.0, 2.0, 3.0]  # trimmed to ProfileSizeX
    assert p["profile_y"] == [5.0, None]  # NaN as null, trimmed
    assert p["x_axis"] == {"start": 100.5, "step": 2.0}  # bin 2: centres 100.5, 102.5…
    assert p["cursor"] == [102.5, 200.0]
    json.dumps(p, allow_nan=False)  # it must travel as JSON a browser parses


def test_infinities_never_reach_the_page() -> None:
    feed = _feed(Clock())
    p = plot_payload(feed, {"histogram": [1.0, float("inf"), float("-inf")]}, 60)
    assert p["histogram"] == [1.0, None, None]
    json.dumps(p, allow_nan=False)


def test_events_carry_plots_only_when_asked() -> None:
    cameras = RecordingCameras()

    async def fetch() -> dict[str, list[float]]:  # pragma: no cover - pre-filled below
        return {}

    held = ArraysSource(fetch, 0.5, clock=lambda: 0.0)
    held._arrays, held._at = {"histogram": [1.0, 2.0]}, 0.0  # a read already done
    cameras.arrays = lambda device: held  # type: ignore[method-assign]
    client = _client(cameras=cameras)

    def once(query: str) -> dict:
        with client.stream("GET", f"/camera/ADTEST_CAM1/events?once=1{query}") as r:
            body = "".join(r.iter_text())
        return json.loads(body.strip().split("\n", 1)[1].removeprefix("data: "))

    assert "plots" not in once("")
    assert once("&plots=1")["plots"]["histogram"] == [1.0, 2.0]


def test_a_failed_arrays_read_is_said_not_fatal() -> None:
    cameras = RecordingCameras()

    async def fetch() -> dict[str, list[float]]:  # pragma: no cover - pre-failed below
        return {}

    failed = ArraysSource(fetch, 0.5, clock=lambda: 0.0)
    failed._error, failed._at = "plots unavailable: caget timed out", 0.0
    cameras.arrays = lambda device: failed  # type: ignore[method-assign]
    with _client(cameras=cameras).stream(
        "GET", "/camera/ADTEST_CAM1/events?once=1&plots=1"
    ) as r:
        body = "".join(r.iter_text())
    assert "caget timed out" in body


def test_a_read_that_never_returns_does_not_hold_back_the_readings() -> None:
    """A PV not served = a 3 s CA timeout per read: the push must not wait for it."""
    cameras = RecordingCameras()

    async def fetch() -> dict[str, list[float]]:
        await asyncio.sleep(3600)
        return {}

    cameras.arrays = lambda device: ArraysSource(fetch, 0.5)  # type: ignore[method-assign]
    started = time.monotonic()
    with _client(cameras=cameras).stream(
        "GET", "/camera/ADTEST_CAM1/events?once=1&plots=1"
    ) as r:
        body = "".join(r.iter_text())
    assert time.monotonic() - started < 2
    assert '"readings"' in body


@pytest.mark.parametrize(
    ("revx", "x", "expected"),
    [
        (0, 10.7, 10),  # pixel 10 covers [10, 11): a click at .7 is pixel 10
        (0, 10.2, 10),
        (1, 10.7, 89),  # reversed, SizeX 100: image 10 is output 89
    ],
)
def test_a_click_is_the_pixel_under_it(revx, x, expected) -> None:
    cameras = RecordingCameras()
    _roi(cameras.feed(CAM), minx=0, miny=0, sizex=100, sizey=100, binx=1, revx=revx)
    _client(cameras=cameras).post("/camera/ADTEST_CAM1/cursor", json={"x": x, "y": 0})
    assert cameras.puts[0] == ("STAT:CursorX", expected)


def test_a_click_puts_the_cursor_in_roi_pixels_clamped() -> None:
    cameras = RecordingCameras()
    _roi(cameras.feed(CAM))  # MinX 100, bin x2, SizeX 400 → 200 output pixels
    client = _client(cameras=cameras)
    assert (
        client.post(
            "/camera/ADTEST_CAM1/cursor", json={"x": 140.7, "y": 250}
        ).status_code
        == 200
    )
    assert cameras.puts[-2:] == [("STAT:CursorX", 20), ("STAT:CursorY", 50)]
    client.post("/camera/ADTEST_CAM1/cursor", json={"x": 9999, "y": -50})
    assert cameras.puts[-2:] == [("STAT:CursorX", 199), ("STAT:CursorY", 0)]


def test_a_read_only_panel_does_not_move_the_cursor() -> None:
    cameras = RecordingCameras()
    _roi(cameras.feed(CAM))
    r = _client(cameras=cameras, writable=False).post(
        "/camera/ADTEST_CAM1/cursor", json={"x": 1, "y": 1}
    )
    assert r.status_code == 403 and cameras.puts == []
