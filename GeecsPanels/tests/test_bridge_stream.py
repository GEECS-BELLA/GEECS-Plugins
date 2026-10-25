"""Device streams: full on connect, then deltas; Last-Event-ID; one shared feed per device."""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from geecs_panels.bridge.stream import QUEUE_FRAMES, frame_json, sse
from geecs_panels.contracts import DeviceFrame
from tests._bridge import (
    CAM,
    MOTOR,
    SUPPLY,
    Clock,
    FakeCA,
    healthy_axis,
    healthy_supply,
    hub,
    streams,
)


def _supply_hub() -> tuple[FakeCA, Clock, object]:
    ca, clock = FakeCA(), Clock()
    h = hub("PS1", ca, clock)
    healthy_supply(ca)
    h.tick()  # the first tick takes the device as it is: no frame
    return ca, clock, h


# ---- full, then deltas -----------------------------------------------------------------


def test_a_full_frame_carries_every_role_and_block() -> None:
    ca, clock, h = _supply_hub()
    frame = h.full_frame()
    assert frame.full and frame.connected
    assert set(frame.roles) == {r.name for r in h.device.roles if r.name != "reset"}
    assert frame.state is not None and frame.state.word == "ok"
    assert frame.roles["current_sp"].fields["DRVH"] == 1.0
    assert frame.history is not None and frame.history.keys == ["current", "voltage"]


def test_a_delta_carries_only_the_changed_role_whole() -> None:
    ca, clock, h = _supply_hub()
    clock.advance(0.25)
    ca.post(SUPPLY + "current", 0.31)
    frame = h.tick()
    assert frame is not None and not frame.full
    assert set(frame.roles) == {"current"}
    current = frame.roles["current"]
    assert current.value == 0.31
    assert current.fields == {
        "EGU": "A",
        "PREC": 3,
        "SEVR": "NO_ALARM",
        "STAT": "NO_ALARM",
    }
    assert frame.state is None  # unchanged: absent
    assert frame.seq == h.full_frame().seq


def test_a_steady_value_is_not_re_sent_as_it_ages() -> None:
    ca, clock, h = _supply_hub()
    for _ in range(20):
        clock.advance(0.25)
        frame = h.tick()
        # The history's own points are the only thing a quiet supply sends.
        assert frame is None or (not frame.roles and frame.state is None)


def test_going_stale_is_a_change_sent_once() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Cam", ca, clock)
    ca.post(CAM + "STAT:ComputeCentroid", 1)
    ca.post(CAM + "STAT:CentroidX_RBV", 640.0)
    ca.post(CAM + "STAT:ArrayCounter_RBV", 1)
    clock.advance(0.1)
    ca.post(CAM + "STAT:ArrayCounter_RBV", 2)
    h.tick()
    clock.advance(4.0)
    stale = h.tick()
    assert stale is not None and stale.roles["cx"].freshness == "stale"
    clock.advance(0.25)
    again = h.tick()
    assert again is None or "cx" not in again.roles


def test_a_state_change_is_in_the_delta() -> None:
    ca, clock, h = _supply_hub()
    ca.post(SUPPLY + "enable_output", 1)
    frame = h.tick()
    assert frame is not None and frame.state is not None
    assert frame.state.word == "running"


def test_a_per_axis_delta_names_only_the_axis_that_changed() -> None:
    ca, clock = FakeCA(), Clock()
    h = hub("Stage", ca, clock)
    healthy_axis(ca, "m1")
    healthy_axis(ca, "m2")
    h.tick()
    ca.post(MOTOR + "m2.RBV", 1.5)
    frame = h.tick()
    assert frame is not None
    assert set(frame.axes) == {"m2"}
    assert set(frame.axes["m2"].roles) == {"rbv"}
    assert frame.axes["m2"].state is None


def test_seq_grows_by_one_per_frame() -> None:
    ca, clock, h = _supply_hub()
    start = h.seq
    for value in (0.1, 0.2, 0.3):
        clock.advance(0.25)
        ca.post(SUPPLY + "current", value)
        assert h.tick().seq == start + 1
        start += 1


# ---- the wire ------------------------------------------------------------------------------


def test_the_sse_event_is_the_seq_and_a_device_frame() -> None:
    ca, clock, h = _supply_hub()
    text = sse(h.full_frame())
    head, data = text.split("data: ", 1)
    assert head == f"id: {h.seq}\nevent: device\n"
    assert text.endswith("\n\n")
    DeviceFrame.model_validate(json.loads(data))


def test_a_nan_reading_travels_as_null() -> None:
    """Rule 10: one NaN must not kill the stream (JSON.parse rejects it)."""
    ca, clock, h = _supply_hub()
    ca.post(SUPPLY + "current", math.nan)
    ca.post(SUPPLY + "current.PREC", math.inf)
    body = json.loads(frame_json(h.tick()))
    assert body["roles"]["current"]["value"] is None
    assert body["roles"]["current"]["fields"]["PREC"] is None


def test_an_unchanged_block_is_absent_from_a_delta_on_the_wire() -> None:
    ca, clock, h = _supply_hub()
    ca.post(SUPPLY + "current", 0.4)
    body = json.loads(frame_json(h.tick()))
    assert "state" not in body and "lock" not in body and "saturation" not in body


# ---- resuming ---------------------------------------------------------------------------------


def test_a_reconnect_resumes_from_the_ring() -> None:
    ca, clock, h = _supply_hub()
    seen = h.seq
    for value in (0.1, 0.2):
        ca.post(SUPPLY + "current", value)
        h.tick()
    replay = h.resume(seen)
    assert [f.seq for f in replay] == [seen + 1, seen + 2]
    assert h.resume(h.seq) == []


def test_an_id_the_hub_cannot_resume_from_is_a_full_frame() -> None:
    ca, clock, h = _supply_hub()
    assert h.resume(h.seq - 5) is None  # before anything this hub sent
    assert h.resume(h.seq + 1) is None  # from the future (another process)
    assert h.resume(3) is None  # an old hub's numbering


def test_a_new_hub_numbers_past_every_older_one() -> None:
    """An evicted hub's ids never fall inside a later hub's ring."""
    ca, clock = FakeCA(), Clock()
    first = hub("PS1", ca, clock)
    first.seq += 10_000  # a long life at the push period
    clock.advance(3600.0)
    assert hub("PS1", FakeCA(), clock).seq > first.seq


# ---- one feed per device, shared; eviction; lag ---------------------------------------------------


def test_listeners_share_one_feed_and_get_the_same_deltas() -> None:
    async def run() -> None:
        ca, clock = FakeCA(), Clock()
        s = streams(ca, clock, push_s=3600.0)
        a, b = s.listen("PS1"), s.listen("PS1")
        first_a, first_b = await a.__anext__(), await b.__anext__()
        assert first_a.full and first_b.full
        assert len([pv for pv, _ in ca.subscribed if pv == SUPPLY + "current"]) == 1
        ca.post(SUPPLY + "current", 0.5)
        frame = s.hubs["PS1"].tick()
        assert await a.__anext__() is frame
        assert await b.__anext__() is frame
        await a.aclose()
        await b.aclose()
        await s.aclose()

    asyncio.run(run())


def test_a_hub_alone_for_the_grace_period_is_evicted_and_unsubscribed() -> None:
    async def run() -> None:
        ca, clock = FakeCA(), Clock()
        s = streams(ca, clock, push_s=3600.0, grace_s=30.0)
        frames = s.listen("PS1")
        await frames.__anext__()
        hub_ = s.hubs["PS1"]
        assert not s.expired(hub_)
        await frames.aclose()
        clock.advance(29.0)
        assert not s.expired(hub_)
        clock.advance(2.0)
        assert s.expired(hub_)
        s.evict("PS1")
        assert "PS1" not in s.hubs
        assert SUPPLY + "current" in ca.closed
        await s.aclose()

    asyncio.run(run())


def test_a_listener_too_far_behind_gets_a_full_frame_not_the_backlog() -> None:
    async def run() -> None:
        ca, clock = FakeCA(), Clock()
        s = streams(ca, clock, push_s=3600.0)
        frames = s.listen("PS1")
        await frames.__anext__()
        h = s.hubs["PS1"]
        for n in range(QUEUE_FRAMES + 5):
            ca.post(SUPPLY + "current", float(n))
            h.tick()
        caught_up = await frames.__anext__()
        assert caught_up.full and caught_up.seq == h.seq
        # What was queued behind the lag is in the full frame already: the
        # next frame is the next tick's, never an older seq.
        ca.post(SUPPLY + "current", -1.0)
        h.tick()
        after = await frames.__anext__()
        assert not after.full and after.seq == caught_up.seq + 1
        await frames.aclose()
        await s.aclose()

    asyncio.run(run())


def test_listening_with_a_last_event_id_replays_instead_of_a_full_frame() -> None:
    async def run() -> None:
        ca, clock = FakeCA(), Clock()
        s = streams(ca, clock, push_s=3600.0)
        first = s.listen("PS1")
        full = await first.__anext__()
        h = s.hubs["PS1"]
        ca.post(SUPPLY + "current", 0.7)
        h.tick()
        await first.aclose()
        again = s.listen("PS1", full.seq)
        resumed = await again.__anext__()
        assert not resumed.full and resumed.seq == full.seq + 1
        await again.aclose()
        await s.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("name", ["Cam", "Stage", "PS1"])
def test_every_full_frame_validates_against_the_contract(name: str) -> None:
    ca, clock = FakeCA(), Clock()
    h = hub(name, ca, clock)
    DeviceFrame.model_validate(json.loads(frame_json(h.full_frame())))
