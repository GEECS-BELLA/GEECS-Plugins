"""The raw PV stream: shared ref-counted monitors, handles with a grace period, one coalesced frame per tick."""

from __future__ import annotations

import asyncio
import math
from typing import Callable, Optional

import pytest

from geecs_panels.bridge.feed import ReadAs, Sample
from geecs_panels.bridge.pvstream import (
    PvStreams,
    TooManyHandles,
    UnknownHandle,
    pv_info,
    pv_sse,
)
from tests._bridge import Clock, FakeCA

A, B = "expt:x:field", "expt:y:state"


def _streams(ca: FakeCA, clock: Clock, **kw: object) -> PvStreams:
    return PvStreams(ca.subscribe, clock=clock.mono, wall=clock.wall, **kw)


def test_handles_share_one_monitor_per_pv_and_the_last_closes_it() -> None:
    ca, clock = FakeCA(), Clock()
    s = _streams(ca, clock, grace_s=30.0)
    s.open([A, B])
    s.open([A])
    assert [pv for pv, _ in ca.subscribed] == [A, B]
    assert ca.subscribed[0][1] == "auto"
    clock.advance(31.0)
    assert len(s.sweep()) == 2
    assert sorted(ca.closed) == [A, B]


def test_a_pv_another_handle_still_names_keeps_its_monitor() -> None:
    async def run() -> None:
        ca, clock = FakeCA(), Clock()
        s = _streams(ca, clock, grace_s=30.0, push_s=0.01)
        s.open([A, B])
        kept = s.open([A])
        frames = s.listen(kept.handle)
        await frames.__anext__()
        clock.advance(31.0)
        assert len(s.sweep()) == 1  # the unread handle goes
        assert ca.closed == [B]  # A is still read through the other
        await frames.aclose()
        await s.aclose()

    asyncio.run(run())


def test_a_handle_is_released_only_after_its_grace_period_alone() -> None:
    async def run() -> None:
        ca, clock = FakeCA(), Clock()
        s = _streams(ca, clock, grace_s=30.0, push_s=0.01)
        ack = s.open([A])
        frames = s.listen(ack.handle)
        await frames.__anext__()
        clock.advance(100.0)
        assert s.sweep() == []  # read: kept
        await frames.aclose()
        clock.advance(29.0)
        assert s.sweep() == []
        clock.advance(2.0)
        assert s.sweep() == [ack.handle]
        assert ca.closed == [A]
        with pytest.raises(UnknownHandle):
            await s.listen(ack.handle).__anext__()
        await s.aclose()

    asyncio.run(run())


def test_a_handle_never_read_is_released_after_its_grace_period() -> None:
    ca, clock = FakeCA(), Clock()
    s = _streams(ca, clock, grace_s=30.0)
    ack = s.open([A])
    clock.advance(31.0)
    assert s.sweep() == [ack.handle]


def test_too_many_handles_are_refused() -> None:
    s = _streams(FakeCA(), Clock(), max_handles=2)
    s.open([A])
    s.open([B])
    with pytest.raises(TooManyHandles):
        s.open([A])


def test_a_full_frame_then_only_the_pvs_that_changed() -> None:
    async def run() -> list:
        ca, clock = FakeCA(), Clock()
        s = _streams(ca, clock, push_s=0.01)
        ack = s.open([A, B])
        ca.post(A, 0.5)
        ca.post(B, "Open", severity=1)
        frames = s.listen(ack.handle)
        full = await frames.__anext__()
        ca.post(A, 0.75)
        delta = await frames.__anext__()
        await frames.aclose()
        return [full, delta]

    full, delta = asyncio.run(run())
    assert full.full and set(full.pvs) == {A, B}
    assert full.pvs[B].value == "Open" and full.pvs[B].severity == "MINOR"
    assert not delta.full and set(delta.pvs) == {A}
    assert delta.pvs[A].value == 0.75 and delta.seq == full.seq + 1


def test_a_fast_counter_makes_one_frame_per_tick_with_its_latest_value() -> None:
    """Coalescing: a PV posting every millisecond is sent once per tick, its latest value."""

    async def run() -> tuple[list, int]:
        ca, clock = FakeCA(), Clock()
        s = _streams(ca, clock, push_s=0.1)
        ack = s.open([A])
        posted = 0

        async def counter() -> None:
            nonlocal posted
            while True:
                posted += 1
                ca.post(A, posted)
                await asyncio.sleep(0.001)

        task = asyncio.get_running_loop().create_task(counter())
        frames = s.listen(ack.handle)
        got = [await frames.__anext__() for _ in range(4)]
        task.cancel()
        await frames.aclose()
        return got, posted

    frames, posted = asyncio.run(run())
    assert posted > 4 * len(frames)  # far more posts than frames
    values = [f.pvs[A].value for f in frames[1:]]
    assert values == sorted(values) and len(set(values)) == len(values)
    assert [f.seq - frames[0].seq for f in frames] == [0, 1, 2, 3]


def test_a_dropped_pv_is_unknown_with_no_value() -> None:
    ca, clock = FakeCA(), Clock()
    s = _streams(ca, clock)
    s.open([A])
    ca.post(A, 1.0)
    ca.drop(A)
    reading = s.monitors.reading(A)
    assert reading.value is None and reading.freshness == "unknown"


def test_a_nan_and_an_array_travel_safely() -> None:
    ca, clock = FakeCA(), Clock()
    s = _streams(ca, clock)
    ack = s.open([A, B])
    ca.post(A, math.nan)
    ca.post(B, [1.0, 2.0, 3.0])
    frame = s.frame(
        s.handles[ack.handle], 1, True, {pv: s.monitors.reading(pv) for pv in (A, B)}
    )
    text = pv_sse(frame)
    assert '"value":null' in text and "[3 values]" in text
    assert text.startswith("id: 1\nevent: pv\n")


def test_the_inspector_read_says_why_a_pv_did_not_answer() -> None:
    async def failing(pv: str) -> dict:
        raise TimeoutError("no answer")

    async def answering(pv: str) -> dict:
        return {"value": 3, "units": "mm", "choices": None, "precision": 3,
                "control_limits": (0.0, 10.0), "count": 1}  # fmt: skip

    down = asyncio.run(pv_info(A, failing))
    up = asyncio.run(pv_info(A, answering))
    assert not down.connected and "no answer" in (down.error or "")
    assert up.connected and up.units == "mm" and up.control_limits == (0.0, 10.0)


def test_a_failed_subscribe_leaves_no_dead_monitor_and_no_half_handle() -> None:
    """A subscribe that raises (aioca off the loop) must not leave the PV silently dead."""
    ca, clock = FakeCA(needs_loop=True), Clock()
    s = _streams(ca, clock)
    with pytest.raises(RuntimeError):
        s.open([A])  # no running loop: as a threadpool route
    assert s.monitors.monitors == {} and s.handles == {}

    async def on_the_loop() -> None:
        s.open([A])

    asyncio.run(on_the_loop())
    assert [pv for pv, _ in ca.subscribed] == [A]


def test_a_handle_that_fails_part_way_releases_what_it_took() -> None:
    ca, clock = FakeCA(), Clock()

    def subscribe(
        pv: str, how: ReadAs, callback: Callable[[Optional[Sample]], None]
    ) -> object:
        if pv == B:
            raise RuntimeError("no channel")
        return ca.subscribe(pv, how, callback)

    s = PvStreams(subscribe, clock=clock.mono, wall=clock.wall)
    with pytest.raises(RuntimeError):
        s.open([A, B])
    assert s.monitors.monitors == {} and ca.closed == [A]
