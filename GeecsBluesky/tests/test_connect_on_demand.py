"""connect_on_demand + stock plans over namespace devices (#807 phase 1).

The phase-1 acceptance on a mock RunEngine: ``bp.count`` and ``bp.scan``
from ``bluesky.plans`` run against :class:`GeecsNamespace` devices — a
:class:`GeecsDetector` camera and a :class:`CaMotor` child — with **no
GEECS preamble**, connected lazily by the preprocessor, shots paced by
``set_mock_value`` on ``acq_timestamp`` (the fake trigger).  The camera is
prepared for external edges by the plan (``STRICT_TRIGGER_INFO``) and its
``trigger()`` waits for the pacer's next stamp — the strict fire itself is
``tests/test_strict_plans.py``'s subject.
"""

from __future__ import annotations

import asyncio

import bluesky.plans as bp
import pytest

pytest.importorskip(
    "aioca"
)  # CA backend needs the `ca` extra (CI's pure-unit job lacks it)
from functools import partial

import bluesky.plan_stubs as bps
from bluesky import RunEngine
from bluesky.preprocessors import SupplementalData
from bluesky.utils import Msg
from ophyd_async.core import StandardDetector, set_mock_value

from geecs_bluesky.devices.detector import STRICT_TRIGGER_INFO
from geecs_bluesky.namespace import DeviceRoster, GeecsNamespace
from geecs_bluesky.preprocessors import (
    connect_on_demand,
    install_connect_on_demand,
    is_connected,
    is_namespace_object,
)
from geecs_bluesky.run_engine import make_run_engine
from tests.ca_mock_helpers import DocCollector


def row(name, *, settable=False, choices="numeric", tolerance=None):
    return {
        "name": name,
        "settable": settable,
        "variabletype": None,
        "choices": choices,
        "tolerance": tolerance,
        "units": "",
        "min": None,
        "max": None,
    }


ROSTER = DeviceRoster(
    experiment="TestExp",
    variables={
        "UC_TestCam": [
            row("trigger", settable=True, choices="on,off"),
            row("MeanCounts"),
            row("MaxCounts"),
        ],
        "U_S1H": [row("Current", settable=True, tolerance=0.01), row("Voltage")],
    },
    types={"UC_TestCam": "Point Grey Camera", "U_S1H": "Magnet PS"},
    subscribed={"UC_TestCam": ["MeanCounts", "MaxCounts"], "U_S1H": ["Voltage"]},
)


@pytest.fixture
def RE() -> RunEngine:
    return make_run_engine(mock=True)


@pytest.fixture
def namespace() -> GeecsNamespace:
    return GeecsNamespace(ROSTER)


def _pacer(RE, cam, magnet=None, *, t0: float):
    """The fake trigger (+ a GEECS convergence stand-in for a staged motor)."""

    async def pace() -> None:
        while not cam._acquire.monitoring:  # the connect the preprocessor inserts
            await asyncio.sleep(0.01)
        ticks = 0
        while True:
            ticks += 1
            set_mock_value(cam.acq_timestamp, t0 + ticks)
            if magnet is not None and is_connected(magnet.current):
                set_mock_value(
                    magnet.current.position, await magnet.current._setpoint.get_value()
                )
            await asyncio.sleep(0.02)

    return asyncio.run_coroutine_threadsafe(pace(), RE._loop)


# ------------------------------------------------------------ message level
def _drive(plan) -> list[Msg]:
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
        for device in devices:
            yield Msg("connect_stub", device)

    monkeypatch.setattr(
        "geecs_bluesky.preprocessors.ensure_connected", fake_ensure_connected
    )

    class Foreign:
        name = "foreign"
        parent = None

    foreign = Foreign()

    def plan():
        yield Msg("stage", cam)
        yield Msg("read", cam)  # already seen: no second connect
        yield Msg("stage", magnet.current)  # a child: connects the child subtree
        yield Msg("stage", foreign)  # not ours: untouched
        yield Msg("checkpoint")

    commands = [
        (m.command, getattr(m.obj, "name", None))
        for m in _drive(connect_on_demand(plan(), mock=True))
    ]
    assert commands == [
        ("connect_stub", "uc_testcam"),
        ("stage", "uc_testcam"),
        ("read", "uc_testcam"),
        ("connect_stub", "u_s1h-current"),
        ("stage", "u_s1h-current"),
        ("stage", "foreign"),
        ("checkpoint", None),
    ]
    assert is_namespace_object(cam.meancounts) and is_namespace_object(
        magnet.current.position
    )
    assert not is_namespace_object(foreign)


def test_install_puts_connect_on_demand_outermost(RE, namespace) -> None:
    RE.preprocessors.append(SupplementalData(baseline=[namespace["U_S1H"]]))
    install_connect_on_demand(RE, mock=True)  # re-install after other preprocessors
    funcs = [getattr(p, "func", p) for p in RE.preprocessors]
    assert funcs[-1] is connect_on_demand and funcs.count(connect_on_demand) == 1


# -------------------------------------------------------------- stock plans
def _edge_take_reading(devices):
    """``trigger_and_read`` over detectors prepared for external edges.

    A GEECS camera cannot self-trigger, so the implicit INTERNAL prepare
    inside ``trigger()`` is refused; the strict ``take_reading`` prepares
    per shot (``stage`` resets the prepare context, so it cannot be done
    before the plan), and here the pacer plays the trigger box.
    """
    for det in devices:
        if isinstance(det, StandardDetector):
            yield from bps.prepare(det, STRICT_TRIGGER_INFO, wait=True)
    return (yield from bps.trigger_and_read(devices))


_PER_SHOT = partial(bps.one_shot, take_reading=_edge_take_reading)
_PER_STEP = partial(bps.one_nd_step, take_reading=_edge_take_reading)


def test_stock_count_runs_over_a_namespace_camera(RE, namespace) -> None:
    cam = namespace["UC_TestCam"]
    cam._acquire.shot_timeout = 2.0
    assert not is_connected(cam)
    pacer = _pacer(RE, cam, t0=1000.0)
    docs = DocCollector()
    try:
        RE(bp.count([cam], num=3, per_shot=_PER_SHOT), docs)
    finally:
        pacer.cancel()
    assert is_connected(cam)
    assert docs.docs["start"][0]["plan_name"] == "count"
    events = docs.primary_events()
    assert len(events) == 3
    for ev in events:
        assert set(ev["data"]) == {
            "uc_testcam-acq_timestamp",
            "uc_testcam-meancounts",
            "uc_testcam-maxcounts",
        }
    assert len({ev["data"]["uc_testcam-acq_timestamp"] for ev in events}) == 3
    assert docs.docs["stop"][0]["exit_status"] == "success"


def test_stock_scan_moves_a_namespace_motor_child(RE, namespace) -> None:
    cam, magnet = namespace["UC_TestCam"], namespace["U_S1H"]
    cam._acquire.shot_timeout = 2.0
    assert not is_connected(magnet.current)
    pacer = _pacer(RE, cam, magnet, t0=2000.0)
    docs = DocCollector()
    try:
        RE(bp.scan([cam], magnet.current, -1.0, 1.0, 5, per_step=_PER_STEP), docs)
    finally:
        pacer.cancel()
    # bluesky's stage_wrapper stages the ROOT ancestor of a motor, so the whole
    # U_S1H device (all served children) is connected, not just Current.
    assert is_connected(magnet.current) and is_connected(magnet)
    events = docs.primary_events()
    assert [ev["data"]["u_s1h-current-position"] for ev in events] == pytest.approx(
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    )
    start = docs.docs["start"][0]
    assert start["plan_name"] == "scan" and start["motors"] == ("u_s1h-current",)
    assert docs.docs["stop"][0]["exit_status"] == "success"


def test_baseline_of_an_unconnected_device_connects_when_installed_last(
    RE, namespace
) -> None:
    """The P1 from review: SupplementalData appended after us must still see connects."""
    cam, magnet = namespace["UC_TestCam"], namespace["U_S1H"]
    cam._acquire.shot_timeout = 2.0
    RE.preprocessors.append(SupplementalData(baseline=[magnet]))
    install_connect_on_demand(RE, mock=True)  # outermost → sees the baseline reads
    pacer = _pacer(RE, cam, t0=3000.0)
    docs = DocCollector()
    try:
        RE(bp.count([cam], num=1, per_shot=_PER_SHOT), docs)
    finally:
        pacer.cancel()
    baseline = [d for d in docs.docs["descriptor"] if d["name"] == "baseline"]
    assert baseline and "u_s1h-voltage" in baseline[0]["data_keys"]
    assert is_connected(magnet)
