"""The plots: the shared arrays read, the history, the payload, the cursor route."""

from __future__ import annotations

import asyncio
import json


from geecs_panels.camera import counter_suffix
from geecs_panels.feed import (
    HISTORY_EVERY_S,
    ArraysSource,
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

    async def run() -> None:
        source = ArraysSource(fetch, 0.5, clock=Clock())
        await asyncio.gather(*(source.latest() for _ in range(4)))
        await source.latest()  # inside the interval: the same read

    asyncio.run(run())
    assert len(reads) == 1


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
    json.dumps(p)  # it must travel as JSON


def test_events_carry_plots_only_when_asked() -> None:
    cameras = RecordingCameras()

    async def fetch() -> dict[str, list[float]]:
        return {"histogram": [1.0, 2.0]}

    cameras.arrays = lambda device: ArraysSource(fetch, 0.5)  # type: ignore[method-assign]
    client = _client(cameras=cameras)

    def once(query: str) -> dict:
        with client.stream("GET", f"/camera/ADTEST_CAM1/events?once=1{query}") as r:
            body = "".join(r.iter_text())
        return json.loads(body.strip().split("\n", 1)[1].removeprefix("data: "))

    assert "plots" not in once("")
    assert once("&plots=1")["plots"]["histogram"] == [1.0, 2.0]


def test_a_failed_arrays_read_is_said_not_fatal() -> None:
    cameras = RecordingCameras()

    async def fetch() -> dict[str, list[float]]:
        raise TimeoutError("caget timed out")

    cameras.arrays = lambda device: ArraysSource(fetch, 0.5)  # type: ignore[method-assign]
    with _client(cameras=cameras).stream(
        "GET", "/camera/ADTEST_CAM1/events?once=1&plots=1"
    ) as r:
        body = "".join(r.iter_text())
    assert "caget timed out" in body


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
