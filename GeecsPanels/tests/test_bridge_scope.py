"""The scope's capture through the bridge: read once per capture, and never drawn against a newer timebase."""

from __future__ import annotations

import asyncio

from geecs_schemas.epics.family import load_families
from geecs_schemas.epics.panel_devices import PanelDevices

from geecs_panels.bridge.catalog import build_catalog
from geecs_panels.bridge.stream import DeviceStreams
from geecs_panels.registry import Device
from tests._bridge import Clock, FakeCA

SCOPE = "bench:scope1:"


class Reads:
    """A capture: two channels interleaved, its identity read with it."""

    def __init__(self) -> None:
        self.calls = 0
        self.uid = 5

    async def __call__(self, pvs: list[str]) -> list[object]:
        self.calls += 1
        by_suffix = {
            "TimeAxis_RBV": [0.0, 1.0, 2.0],
            "ARR:ArrayData": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0],
            "ARR:UniqueId_RBV": self.uid,
            "ARR:ArraySize0_RBV": 2,
            "ARR:ArraySize1_RBV": 3,
        }
        return [by_suffix[pv.removeprefix(SCOPE)] for pv in pvs]


def _scope() -> tuple[FakeCA, Reads, object]:
    (device,), _ = build_catalog(
        "expt", [Device("Scope", "scope", SCOPE)], PanelDevices(), load_families()
    )
    ca, reads, clock = FakeCA(), Reads(), Clock()
    streams = DeviceStreams(
        {"Scope": device}, ca.subscribe, read=reads, clock=clock.mono, wall=clock.wall
    )
    hub = streams.hub("Scope", run=False)
    for ch in "AB":
        ca.post(f"{SCOPE}{ch}:Present_RBV", "Present")
    for ch in "CD":
        ca.post(f"{SCOPE}{ch}:Present_RBV", "Absent")
    ca.post(SCOPE + "SampleInterval_RBV", 8.0)
    ca.post(SCOPE + "ARR:UniqueId_RBV", 5)
    return ca, reads, hub


def test_each_present_channel_is_a_trace_against_time() -> None:
    ca, reads, hub = _scope()
    reply = asyncio.run(hub.arrays(100))
    assert reply.traces["A"].x == [0.0, 1.0, 2.0]
    assert reply.traces["A"].y == [1.0, 2.0, 3.0]
    assert reply.traces["B"].y == [10.0, 20.0, 30.0]
    assert set(reply.traces) == {"A", "B"}


def test_a_capture_is_read_once_however_often_it_is_asked_for() -> None:
    async def run() -> int:
        ca, reads, hub = _scope()
        await hub.arrays(100)
        await asyncio.sleep(0.6)  # past the read interval: the capture is the same
        await hub.arrays(100)
        return reads.calls

    assert asyncio.run(run()) == 1


def test_a_capture_older_than_a_timebase_change_is_waiting_not_drawn() -> None:
    async def run() -> tuple:
        ca, reads, hub = _scope()
        ca.post(SCOPE + "SampleInterval_RBV", 16.0)  # the axis changes now
        stale = await hub.arrays(100)
        reads.uid = 6
        ca.post(SCOPE + "ARR:UniqueId_RBV", 6)  # a capture at the new timebase
        await asyncio.sleep(0.6)
        fresh = await hub.arrays(100)
        return stale, fresh

    stale, fresh = asyncio.run(run())
    assert stale.traces == {} and "timebase changed" in (stale.waiting or "")
    assert fresh.waiting is None and set(fresh.traces) == {"A", "B"}


def test_an_ioc_restart_forgets_the_timebase_mark() -> None:
    ca, reads, hub = _scope()
    ca.post(SCOPE + "SampleInterval_RBV", 16.0)
    ca.drop(SCOPE + "ARR:UniqueId_RBV")
    assert "timebase_uid" not in hub.feed.state
