"""connect_on_demand + stock plans over namespace devices (#807 phase 1).

The phase-1 acceptance on a mock RunEngine: ``bp.count`` and ``bp.scan``
from ``bluesky.plans`` run against :class:`GeecsNamespace` devices with
**no GEECS preamble**, connected lazily by the preprocessor, shots paced by
``set_mock_value`` on ``acq_timestamp`` (the fake trigger).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import pytest
from bluesky.utils import Msg
from ophyd_async.core import set_mock_value

from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace
from geecs_bluesky.preprocessors import (
    connect_on_demand,
    install_connect_on_demand,
    is_namespace_object,
)
from geecs_bluesky.session import GeecsSession

ROSTER = DeviceRoster(
    experiment="TestExp",
    variables={
        "UC_TestCam": [
            {"name": "trigger", "settable": True, "variabletype": "choice"},
            {"name": "MeanCounts", "variabletype": "numeric"},
            {"name": "MaxCounts", "variabletype": "numeric"},
        ],
        "U_S1H": [
            {
                "name": "current",
                "settable": True,
                "variabletype": "numeric",
                "tolerance": 0.01,
            },
        ],
    },
    subscribed={"UC_TestCam": ["MeanCounts", "MaxCounts"]},
)


def _is_connected(device) -> bool:
    """Connected in mock mode (a DeviceMock is installed) or for real (task done)."""
    if getattr(device, "_mock", None) is not None:
        return True
    task = getattr(device, "_connect_task", None)
    return bool(task is not None and task.done() and task.exception() is None)


@pytest.fixture
def session() -> GeecsSession:
    return GeecsSession("TestExp", tiled=False, mock=True)


@pytest.fixture
def namespace() -> GeecsNamespace:
    return GeecsNamespace(ROSTER)


class _Docs:
    def __init__(self) -> None:
        self.docs: dict[str, list[dict]] = defaultdict(list)

    def __call__(self, name: str, doc: dict) -> None:
        self.docs[name].append(doc)

    def primary_descriptor(self) -> dict:
        """The 'primary' stream descriptor (skip the interruptions baseline)."""
        return next(d for d in self.docs["descriptor"] if d["name"] == "primary")

    def primary_events(self) -> list[dict]:
        pd = self.primary_descriptor()["uid"]
        return [e for e in self.docs["event"] if e["descriptor"] == pd]


# ------------------------------------------------------------ message level
def _drive(plan) -> list[Msg]:
    """Run a plan generator to exhaustion sending None, collecting its messages."""
    out: list[Msg] = []
    try:
        msg = next(plan)
        while True:
            out.append(msg)
            msg = plan.send(None)
    except StopIteration:
        return out


def test_connect_is_inserted_once_per_namespace_object_only(
    namespace, monkeypatch
) -> None:
    cam, magnet = namespace["UC_TestCam"], namespace["U_S1H"]

    def fake_ensure_connected(*devices, mock=False, timeout=0.0, force_reconnect=False):
        # stand-in for ophyd-async's stub (whose wait_for needs a live loop)
        for device in devices:
            yield Msg("connect_stub", device)

    monkeypatch.setattr(
        "geecs_bluesky.preprocessors.ensure_connected", fake_ensure_connected
    )

    class Foreign:  # not a namespace member
        name = "foreign"
        parent = None

    foreign = Foreign()

    def plan():
        yield Msg("stage", cam)
        yield Msg("read", cam)  # already seen: no second connect
        yield Msg("stage", magnet.current)  # a child: connects the child subtree
        yield Msg("stage", foreign)  # not ours: untouched
        yield Msg("checkpoint")  # no object: untouched

    msgs = _drive(connect_on_demand(plan(), mock=True))
    commands = [(m.command, getattr(m.obj, "name", None)) for m in msgs]
    # one connect before the first touch of each namespace object, nothing else
    assert commands == [
        ("connect_stub", "UC_TestCam"),
        ("stage", "UC_TestCam"),
        ("read", "UC_TestCam"),
        ("connect_stub", "U_S1H-current"),
        ("stage", "U_S1H-current"),
        ("stage", "foreign"),
        ("checkpoint", None),
    ]
    assert is_namespace_object(cam.MeanCounts) and is_namespace_object(
        magnet.current.position
    )
    assert not is_namespace_object(foreign)


def test_install_is_idempotent(session) -> None:
    install_connect_on_demand(session.RE, mock=True)
    install_connect_on_demand(session.RE, mock=True)
    assert [p.func for p in session.RE.preprocessors] == [connect_on_demand]


# -------------------------------------------------------------- stock plans
def test_stock_count_runs_over_a_namespace_camera(session, namespace) -> None:
    RE = session.RE
    install_connect_on_demand(RE, mock=True)
    cam = namespace["UC_TestCam"]
    cam._trigger_timeout = 2.0
    assert not _is_connected(cam)

    async def pace() -> None:
        # wait for the preprocessor to connect the camera, then act as the trigger
        while not cam._monitoring:
            await asyncio.sleep(0.01)
        ticks = 0
        while True:
            ticks += 1
            set_mock_value(cam.acq_timestamp, 1000.0 + ticks)
            set_mock_value(cam.MeanCounts, 10.0 * ticks)
            await asyncio.sleep(0.05)

    pacer = asyncio.run_coroutine_threadsafe(pace(), RE._loop)
    docs = _Docs()
    try:
        RE(bp.count([cam], num=3), docs)
    finally:
        pacer.cancel()

    assert _is_connected(cam)
    assert (
        len(docs.docs["start"]) == 1 and docs.docs["start"][0]["plan_name"] == "count"
    )
    events = docs.primary_events()
    assert len(events) == 3
    for ev in events:
        assert set(ev["data"]) == {
            "UC_TestCam-acq_timestamp",
            "UC_TestCam-MeanCounts",
            "UC_TestCam-MaxCounts",
        }
    stamps = [ev["data"]["UC_TestCam-acq_timestamp"] for ev in events]
    assert len(set(stamps)) == 3  # three distinct shots
    config = docs.primary_descriptor()["configuration"]
    assert list(config) == ["UC_TestCam"], config
    assert (
        config["UC_TestCam"]["data"]["UC_TestCam-variables"] == "MeanCounts,MaxCounts"
    )
    assert docs.docs["stop"][0]["exit_status"] == "success"


def test_stock_scan_moves_a_namespace_motor_child(session, namespace) -> None:
    RE = session.RE
    install_connect_on_demand(RE, mock=True)
    cam, magnet = namespace["UC_TestCam"], namespace["U_S1H"]
    cam._trigger_timeout = 2.0

    # First touch connects the Movable child alone (not every U_S1H variable)
    assert not _is_connected(magnet.current)
    RE(bps.stage(magnet.current))
    RE(bps.unstage(magnet.current))
    assert _is_connected(magnet.current) and not _is_connected(magnet)

    async def pace() -> None:
        # the fake trigger, plus a GEECS convergence stand-in: mirror the
        # motor's :SP onto its streamed readback so a staged move converges.
        while not cam._monitoring:
            await asyncio.sleep(0.01)
        ticks = 0
        while True:
            ticks += 1
            set_mock_value(cam.acq_timestamp, 2000.0 + ticks)
            set_mock_value(
                magnet.current.position, await magnet.current._setpoint.get_value()
            )
            await asyncio.sleep(0.02)

    pacer = asyncio.run_coroutine_threadsafe(pace(), RE._loop)
    docs = _Docs()
    try:
        RE(bp.scan([cam], magnet.current, -1.0, 1.0, 5), docs)
    finally:
        pacer.cancel()

    events = docs.primary_events()
    assert len(events) == 5
    positions = [ev["data"]["U_S1H-current-position"] for ev in events]
    assert positions == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])
    start = docs.docs["start"][0]
    assert start["plan_name"] == "scan" and start["motors"] == ("U_S1H-current",)
    assert docs.docs["stop"][0]["exit_status"] == "success"


def test_configure_in_a_plan_changes_the_logged_columns(session, namespace) -> None:
    RE = session.RE
    install_connect_on_demand(RE, mock=True)
    cam = namespace["UC_TestCam"]
    cam._trigger_timeout = 2.0

    async def pace() -> None:
        while not cam._monitoring:
            await asyncio.sleep(0.01)
        ticks = 0
        while True:
            ticks += 1
            set_mock_value(cam.acq_timestamp, 3000.0 + ticks)
            await asyncio.sleep(0.05)

    def plan():
        yield from bps.configure(cam, variables=["MaxCounts"])
        yield from bp.count([cam], num=1)

    pacer = asyncio.run_coroutine_threadsafe(pace(), RE._loop)
    docs = _Docs()
    try:
        RE(plan(), docs)
    finally:
        pacer.cancel()
    (event,) = docs.primary_events()
    assert set(event["data"]) == {"UC_TestCam-acq_timestamp", "UC_TestCam-MaxCounts"}
