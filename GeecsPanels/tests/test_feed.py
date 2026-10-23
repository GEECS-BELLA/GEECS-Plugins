"""The camera feed (monitored PVs → the page's snapshot) and the shared frame source."""

from __future__ import annotations

import asyncio

import pytest

from geecs_panels.camera import counter_suffix
from geecs_panels.feed import CONNECT_GRACE_S, CameraFeed, FrameSource, FrameUnavailable

PREFIX = "expt:cam:"


class Clock:
    """A settable monotonic clock."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Sub:
    def __init__(self, log: list[str], pv: str) -> None:
        self.log, self.pv = log, pv

    def close(self) -> None:
        self.log.append(self.pv)


def _feed(clock: Clock) -> tuple[CameraFeed, list[tuple[str, bool]], list[str]]:
    subscribed: list[tuple[str, bool]] = []
    closed: list[str] = []

    def subscribe(pv, as_string, callback):
        subscribed.append((pv, as_string))
        return Sub(closed, pv)

    feed = CameraFeed(PREFIX, subscribe, clock=clock)
    feed.start()
    return feed, subscribed, closed


def _reading(snapshot: dict, key: str) -> dict:
    return next(r for r in snapshot["readings"] if r["key"] == key)


def test_start_subscribes_once_under_the_prefix() -> None:
    clock = Clock()
    feed, subscribed, _ = _feed(clock)
    feed.start()
    names = [pv for pv, _ in subscribed]
    assert len(names) == len(set(names))
    assert all(pv.startswith(PREFIX) for pv in names)
    assert (PREFIX + "cam1:DetectorState_RBV", True) in subscribed


def test_the_first_counter_value_is_a_position_not_a_frame() -> None:
    """Connecting to a stopped camera must not look like one fresh frame."""
    clock = Clock()
    feed, _, _ = _feed(clock)
    feed.update(counter_suffix("STAT"), 41)
    assert feed.counter_ages()["STAT"] is None
    clock.now += 1
    feed.update(counter_suffix("STAT"), 42)
    clock.now += 0.5
    assert feed.counter_ages()["STAT"] == pytest.approx(0.5)


def test_a_steady_centroid_stays_fresh_while_frames_arrive() -> None:
    clock = Clock()
    feed, _, _ = _feed(clock)
    feed.update("STAT:ComputeCentroid", 1)
    feed.update("STAT:CentroidX_RBV", 640.0)
    feed.update(counter_suffix("STAT"), 1)
    for n in range(2, 6):
        clock.now += 1
        feed.update(counter_suffix("STAT"), n)
    cx = _reading(feed.snapshot(stale_after=3.0), "cx")
    assert (cx["value"], cx["state"]) == (640.0, "fresh")
    clock.now += 4
    assert _reading(feed.snapshot(stale_after=3.0), "cx")["state"] == "stale"


def test_a_disconnect_marks_the_value_disconnected() -> None:
    clock = Clock()
    feed, _, _ = _feed(clock)
    feed.update("cam1:AcquireTime_RBV", 0.01)
    assert _reading(feed.snapshot(3.0), "exposure")["state"] == "fresh"
    feed.update("cam1:AcquireTime_RBV", None)
    assert _reading(feed.snapshot(3.0), "exposure")["state"] == "disconnected"


def test_the_image_age_is_mjpg1s_counter() -> None:
    clock = Clock()
    feed, _, _ = _feed(clock)
    assert feed.snapshot(3.0)["image"]["state"] == "disconnected"
    feed.update(counter_suffix("MJPG1"), 7)
    assert feed.snapshot(3.0)["image"]["state"] == "stale"
    clock.now += 0.1
    feed.update(counter_suffix("MJPG1"), 8)
    assert feed.snapshot(3.0)["image"]["state"] == "fresh"


def test_never_connected_pvs_are_named_after_the_grace_period() -> None:
    clock = Clock()
    feed, _, _ = _feed(clock)
    for suffix in feed.connected:
        if not suffix.startswith("STAT0:"):
            feed.update(suffix, 1)
    assert feed.missing() == []
    clock.now += CONNECT_GRACE_S + 0.1
    assert feed.missing() and all(s.startswith("STAT0:") for s in feed.missing())


def test_close_drops_every_subscription() -> None:
    feed, subscribed, closed = _feed(Clock())
    feed.close()
    assert sorted(closed) == sorted(pv for pv, _ in subscribed)


# ---- FrameSource -------------------------------------------------------------


def _source(fetch, resolve=None, min_interval=0.1, clock=None) -> FrameSource:
    async def default_resolve() -> str:
        return "http://ioc/frame.jpg"

    return FrameSource(
        resolve or default_resolve, fetch, min_interval, clock=clock or Clock()
    )


def test_concurrent_viewers_share_one_upstream_request() -> None:
    calls: list[str] = []

    async def fetch(url: str) -> bytes:
        calls.append(url)
        await asyncio.sleep(0.01)
        return b"jpeg"

    async def run() -> list[bytes]:
        source = _source(fetch)
        return await asyncio.gather(*(source.latest() for _ in range(5)))

    assert asyncio.run(run()) == [b"jpeg"] * 5
    assert calls == ["http://ioc/frame.jpg"]


def test_a_frame_is_reused_within_the_interval_only() -> None:
    clock = Clock()
    frames = iter([b"one", b"two"])

    async def fetch(url: str) -> bytes:
        return next(frames)

    async def run() -> list[bytes]:
        source = _source(fetch, clock=clock)
        out = [await source.latest(), await source.latest()]
        clock.now += 0.2
        out.append(await source.latest())
        return out

    assert asyncio.run(run()) == [b"one", b"one", b"two"]


def test_a_failed_fetch_re_resolves_the_url_next_time() -> None:
    resolved: list[int] = []
    attempts = iter([RuntimeError("refused"), b"ok"])

    async def resolve() -> str:
        resolved.append(1)
        return "http://ioc/frame.jpg"

    async def fetch(url: str) -> bytes:
        item = next(attempts)
        if isinstance(item, Exception):
            raise item
        return item

    async def run() -> bytes:
        source = _source(fetch, resolve=resolve)
        with pytest.raises(FrameUnavailable, match="refused"):
            await source.latest()
        return await source.latest()

    assert asyncio.run(run()) == b"ok"
    assert len(resolved) == 2


def test_an_unresolvable_url_is_a_frame_unavailable() -> None:
    async def resolve() -> str:
        raise TimeoutError("no channel")

    async def fetch(url: str) -> bytes:  # pragma: no cover - never reached
        raise AssertionError

    async def run() -> None:
        await _source(fetch, resolve=resolve).latest()

    with pytest.raises(FrameUnavailable, match="stream URL unavailable"):
        asyncio.run(run())


def test_a_reconnect_is_a_position_not_a_frame() -> None:
    """An IOC restart or a VPN blip on an idle camera must not look like a fresh frame."""
    clock = Clock()
    feed, _, _ = _feed(clock)
    feed.update(counter_suffix("STAT"), 500)
    feed.update(counter_suffix("STAT"), 501)
    clock.now += 10
    assert feed.counter_ages()["STAT"] == pytest.approx(10)
    feed.update(counter_suffix("STAT"), None)  # the IOC goes away
    feed.update(counter_suffix("STAT"), 0)  # and comes back, restarted, idle
    assert feed.counter_ages()["STAT"] == pytest.approx(10)
    clock.now += 1
    feed.update(counter_suffix("STAT"), 1)  # a real frame
    assert feed.counter_ages()["STAT"] == pytest.approx(0)


def test_a_pv_that_dropped_after_connecting_is_not_missing() -> None:
    """Missing means never served; a disconnect is shown on the reading instead."""
    clock = Clock()
    feed, _, _ = _feed(clock)
    feed.update(counter_suffix("STAT"), 1)
    feed.update(counter_suffix("STAT"), None)
    clock.now += CONNECT_GRACE_S + 0.1
    assert counter_suffix("STAT") not in feed.missing()
