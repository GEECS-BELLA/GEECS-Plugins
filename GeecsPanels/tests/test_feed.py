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
        return [
            f.jpeg for f in await asyncio.gather(*(source.latest() for _ in range(5)))
        ]

    assert asyncio.run(run()) == [b"jpeg"] * 5
    assert calls == ["http://ioc/frame.jpg"]


def test_a_frame_is_reused_within_the_interval_only() -> None:
    clock = Clock()
    frames = iter([b"one", b"two"])

    async def fetch(url: str) -> bytes:
        return next(frames)

    async def run() -> list[bytes]:
        source = _source(fetch, clock=clock)
        out = [(await source.latest()).jpeg, (await source.latest()).jpeg]
        clock.now += 0.2
        out.append((await source.latest()).jpeg)
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
        return (await source.latest()).jpeg

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


def test_an_unchanged_frame_counter_means_no_new_fetch() -> None:
    """A 1 Hz camera polled at 10 Hz is fetched once per frame, not ten times."""
    clock = Clock()
    counter = {"n": 7}
    fetched: list[int] = []

    async def fetch(url: str) -> bytes:
        fetched.append(counter["n"])
        return b"frame%d" % counter["n"]

    async def run() -> list[tuple[bytes, object]]:
        source = FrameSource(
            lambda: _resolved(), fetch, 0.1, counter=lambda: counter["n"], clock=clock
        )
        out = []
        for _ in range(5):  # half a second of polling, the counter still
            clock.now += 0.1
            f = await source.latest()
            out.append((f.jpeg, f.counter))
        counter["n"] = 8  # the camera produced a frame
        clock.now += 0.1
        f = await source.latest()
        out.append((f.jpeg, f.counter))
        return out

    out = asyncio.run(run())
    assert fetched == [7, 8]
    assert out[0] == (b"frame7", 7) and out[-1] == (b"frame8", 8)


def test_without_a_counter_the_interval_alone_decides() -> None:
    clock = Clock()
    fetched: list[int] = []

    async def fetch(url: str) -> bytes:
        fetched.append(1)
        return b"x"

    async def run() -> None:
        source = FrameSource(lambda: _resolved(), fetch, 0.1, clock=clock)
        for _ in range(3):
            clock.now += 0.2
            assert (await source.latest()).counter is None

    asyncio.run(run())
    assert len(fetched) == 3


async def _resolved() -> str:
    return "http://ioc/frame.jpg"


def test_the_reuse_window_is_timed_from_the_request_not_the_reply() -> None:
    """A slow fetch must not push the next poll inside the window (it halved the rate)."""
    clock = Clock()
    fetched: list[float] = []

    async def fetch(url: str) -> bytes:
        clock.now += 0.06  # the upstream fetch takes 60 ms
        fetched.append(clock.now)
        return b"x"

    async def run() -> None:
        source = FrameSource(lambda: _resolved(), fetch, 0.05, clock=clock)
        start = clock.now
        for i in range(4):  # polls every 100 ms, measured from each request
            clock.now = start + 0.1 * i
            await source.latest()

    asyncio.run(run())
    assert len(fetched) == 4


def test_every_viewer_shares_one_fetch_per_floor_interval() -> None:
    """A fast camera and many viewers: the IOC sees one fetch per floor, not one per poll."""
    clock = Clock()
    counter = {"n": 0}
    fetched: list[int] = []

    async def fetch(url: str) -> bytes:
        fetched.append(counter["n"])
        return b"x"

    async def run() -> None:
        source = FrameSource(
            lambda: _resolved(), fetch, 0.05, counter=lambda: counter["n"], clock=clock
        )
        # 5 viewers out of step, a 100 Hz camera, one second.
        for step in range(500):
            clock.now = 100.0 + step * 0.002
            counter["n"] = step // 5
            await source.latest()

    asyncio.run(run())
    assert len(fetched) <= 21  # one per 50 ms floor, however many viewers


def test_an_unchanged_counter_still_refetches_after_max_reuse() -> None:
    """The safety net: a stale frame lasts max_reuse, never a whole camera period."""
    clock = Clock()
    fetched: list[float] = []

    async def fetch(url: str) -> bytes:
        fetched.append(clock.now)
        return b"x"

    async def run() -> list[str]:
        source = FrameSource(
            lambda: _resolved(),
            fetch,
            0.05,
            counter=lambda: 7,
            max_reuse=1.0,
            clock=clock,
        )
        tags = []
        for _ in range(25):  # 2.5 s of polling, the counter never moves
            tags.append((await source.latest()).tag)
            clock.now += 0.1
        return tags

    tags = asyncio.run(run())
    assert len(fetched) == 3  # at 0, ~1 s and ~2 s
    assert len(set(tags)) == 3  # each refetch is a new tag: the page redraws it
