"""Interlock suspenders pause strict scans, run their action plans and resume.

On the strict-plan harness (``tests/test_strict_plans.py``): a fake trigger
box whose SINGLESHOT put is the shot, and a mock gauge whose ``Pressure``
the test drives.  The suspender is the real
:class:`~geecs_bluesky.interlocks.suspender.GeecsInterlockSuspender` on a
stock RunEngine.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky import RunEngine
from ophyd_async.core import set_mock_value
from geecs_schemas import Interlock

from geecs_bluesky.devices.ca import CaSnapshotReadable
from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.interlocks.install import install_on_loop
from geecs_bluesky.interlocks.suspender import GeecsInterlockSuspender, Watched
from geecs_bluesky.plans.strict import geecs_per_shot
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_strict_plans import WRITES, FakeBox, _camera, _plugin_camera

SAFE, UNSAFE = 1e-6, 1e-3


def _gauge(RE: RunEngine) -> CaSnapshotReadable:
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, gauge)
    set_mock_value(gauge.pressure, SAFE)
    set_mock_value(gauge.connected_status, "Connected")
    return gauge


def _interlock(**kw) -> Interlock:
    return Interlock.model_validate(
        {
            "name": "vacuum",
            "condition": {
                "kind": "threshold",
                "device": "U_Gauge",
                "variable": "Pressure",
                "operator": ">",
                "limit": 1e-5,
            },
            **kw,
        }
    )


def _install(RE: RunEngine, gauge, interlock: Interlock, **kw):
    suspender = GeecsInterlockSuspender(
        interlock,
        Watched(value=gauge.pressure, connected={"U_Gauge": gauge.connected_status}),
        **kw,
    )
    install_on_loop(RE, suspender)
    return suspender


def _recording(ran: list[str], label: str):
    """A pre/post plan factory that records that it ran."""

    def factory():
        ran.append(label)
        yield from ()

    return factory


class TrippingBox(FakeBox):
    """FakeBox that calls ``on_fire(n)`` right after the n-th shot fires.

    Right after the fire is the worst place a suspension can land: the shot
    has fired and its frame is on the way, but its row is not saved.  The
    hook is installed before the trigger box caches its setters.
    """

    def __init__(self) -> None:
        super().__init__()
        self.on_fire = lambda fire: None
        self.fire_times: list[float] = []

    def __call__(self, device: str, variable: str):
        setter = super().__call__(device, variable)
        put = setter.put
        box = self

        async def put_then_hook(value: str) -> None:
            await put(value)
            if variable == "Trigger.ExecuteSingleShot":
                box.fire_times.append(time.monotonic())
                box.on_fire(box.fires)

        setter.put = put_then_hook
        return setter


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def box() -> TrippingBox:
    return TrippingBox()


@pytest.fixture
def shot_control(RE: RunEngine, box: TrippingBox) -> ShotControl:
    sc = ShotControl(
        WRITES, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    RE(bps.mv(sc, "ARMED"))
    return sc


def _trip_at(
    box: TrippingBox, gauge, *, fire: int, clear_after: float
) -> dict[str, float]:
    """Drive the gauge unsafe right after the *fire*-th shot, safe again later.

    Returns the trip and clear times (monotonic), filled in as they happen.
    """
    window: dict[str, float] = {}

    def on_fire(n: int) -> None:
        if n != fire:
            return
        window["tripped"] = time.monotonic()
        set_mock_value(gauge.pressure, UNSAFE)

        def clear() -> None:
            time.sleep(clear_after)
            window["cleared"] = time.monotonic()
            set_mock_value(gauge.pressure, SAFE)

        threading.Thread(target=clear, daemon=True).start()

    box.on_fire = on_fire
    return window


def _datum_indices(col: DocCollector, key: str) -> list[dict]:
    resources = {r["uid"]: r["data_key"] for r in col.docs["stream_resource"]}
    return [
        d["indices"]
        for d in col.docs["stream_datum"]
        if resources[d["stream_resource"]] == key
    ]


def test_a_safe_interlock_leaves_the_scan_alone(
    RE: RunEngine, box: TrippingBox, shot_control: ShotControl
) -> None:
    gauge = _gauge(RE)
    suspender = _install(RE, gauge, _interlock())
    cam = _camera(RE, box, "UC_Cam")
    RE(bp.count([cam], num=3, per_shot=geecs_per_shot(shot_control)))
    assert box.fires == 3
    assert not suspender.tripped
    assert suspender.verdict is not None and not suspender.verdict.unsafe


def test_a_trip_mid_shot_suspends_runs_the_plans_and_resumes(
    RE: RunEngine, box: TrippingBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """Tripped inside shot 2's fire: on_trip, then before_resume, every row once, frames aligned."""
    gauge = _gauge(RE)
    ran: list[str] = []
    suspender = _install(
        RE,
        gauge,
        _interlock(settle_s=0.1),
        pre_plan=_recording(ran, "on_trip"),
        post_plan=_recording(ran, "before_resume"),
    )
    cam, _ = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=2.0)
    window = _trip_at(box, gauge, fire=2, clear_after=0.3)
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([cam], num=4, per_shot=geecs_per_shot(shot_control)))

    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert ran == ["on_trip", "before_resume"]
    assert not suspender.tripped
    events = col.primary_events()
    assert len(events) == 4
    stamps = [e["data"]["uc_a-acq_timestamp"] for e in events]
    assert stamps == sorted(set(stamps)), stamps  # no row recorded twice
    # One frame per row, each row's datum its own frame.
    indices = _datum_indices(col, "uc_a")
    assert indices == [{"start": i, "stop": i + 1} for i in range(4)], (
        box.fires,
        indices,
    )
    # The interrupted shot is fired again (one extra physical shot), and
    # nothing fires while the interlock is tripped.
    assert box.fires == 5
    assert not [t for t in box.fire_times if window["tripped"] < t < window["cleared"]]


def test_an_immediate_pause_mid_shot_replays_the_shot_cleanly(
    RE: RunEngine, box: TrippingBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """The operator's immediate pause cuts a shot the same way: the replayed row keeps one frame."""
    from bluesky.utils import RunEngineInterrupted

    cam, _ = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=2.0)

    def pause_after_second(n: int) -> None:
        if n == 2:  # off the RunEngine's loop, as the manager requests it
            threading.Thread(
                target=RE.request_pause, kwargs={"defer": False}, daemon=True
            ).start()

    box.on_fire = pause_after_second
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(RunEngineInterrupted):
        RE(bp.count([cam], num=4, per_shot=geecs_per_shot(shot_control)))
    RE.resume()
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert _datum_indices(col, "uc_a") == [
        {"start": i, "stop": i + 1} for i in range(4)
    ]


def test_tripped_before_the_run_holds_it_at_the_start(
    RE: RunEngine, box: TrippingBox, shot_control: ShotControl
) -> None:
    """Already unsafe when the plan starts: nothing fires until it clears; no on_trip runs."""
    gauge = _gauge(RE)
    set_mock_value(gauge.pressure, UNSAFE)
    ran: list[str] = []
    suspender = _install(RE, gauge, _interlock(), pre_plan=_recording(ran, "on_trip"))
    assert suspender.tripped
    assert suspender.verdict is not None
    assert suspender.verdict.diagnostic == "Pressure=0.001 (limit: >1e-05)"
    cam = _camera(RE, box, "UC_Cam")
    cleared: dict[str, float] = {}

    def clear() -> None:
        time.sleep(0.3)
        cleared["at"] = time.monotonic()
        set_mock_value(gauge.pressure, SAFE)

    threading.Thread(target=clear, daemon=True).start()
    RE(bp.count([cam], num=2, per_shot=geecs_per_shot(shot_control)))
    assert box.fires == 2
    assert min(box.fire_times) > cleared["at"]
    assert ran == []


def test_a_disconnected_device_trips_whatever_its_value(
    RE: RunEngine,
) -> None:
    gauge = _gauge(RE)
    suspender = _install(RE, gauge, _interlock())
    assert not suspender.tripped
    set_mock_value(gauge.connected_status, "Disconnected")
    assert suspender.tripped
    assert suspender.verdict.diagnostic == "U_Gauge disconnected"
    set_mock_value(gauge.connected_status, "Connected")
    assert not suspender.tripped


def test_a_frozen_stamp_trips_a_stale_interlock(RE: RunEngine) -> None:
    """``max_age_s`` without a new acq_timestamp trips; the next stamp clears it."""
    from geecs_bluesky.devices.ca import CaSnapshotReadable

    gauge = CaSnapshotReadable(
        "U_Gauge", ["acq_timestamp"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, gauge)
    set_mock_value(gauge.acq_timestamp, 1.0)
    interlock = Interlock.model_validate(
        {
            "name": "gauge alive",
            "condition": {"kind": "stale", "device": "U_Gauge", "max_age_s": 0.2},
        }
    )
    suspender = GeecsInterlockSuspender(interlock, Watched(stamp=gauge.acq_timestamp))
    install_on_loop(RE, suspender)
    for stamp in (2.0, 3.0):  # publishing faster than max_age_s
        time.sleep(0.1)
        set_mock_value(gauge.acq_timestamp, stamp)
    assert not suspender.tripped
    time.sleep(0.4)
    assert suspender.tripped
    assert "no new data" in suspender.verdict.diagnostic
    set_mock_value(gauge.acq_timestamp, 4.0)
    assert not suspender.tripped


class _SilentSignal:
    """A subscribable signal that never reports (a PV whose monitor never fires)."""

    name = "silent"

    def subscribe(self, callback) -> None:
        pass

    def clear_sub(self, callback) -> None:
        pass


def test_no_verdict_until_the_first_readings_then_fail_safe(
    RE: RunEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Judged only once every signal reported; a silent one trips after the grace."""
    from geecs_bluesky.interlocks import suspender as module

    monkeypatch.setattr(module, "FIRST_READING_GRACE_S", 0.2)
    gauge = _gauge(RE)
    changes: list = []
    suspender = GeecsInterlockSuspender(
        _interlock(),
        Watched(value=_SilentSignal(), connected={"U_Gauge": gauge.connected_status}),
        on_change=lambda s: changes.append(s.verdict),
    )
    install_on_loop(RE, suspender)
    assert suspender.verdict is None and not suspender.tripped
    time.sleep(0.4)
    assert suspender.tripped
    assert suspender.verdict.diagnostic == "Pressure: no data"
    assert len(changes) == 1  # no transient before it


def test_readings_present_at_install_give_no_transient_trip(RE: RunEngine) -> None:
    gauge = _gauge(RE)
    changes: list = []
    suspender = GeecsInterlockSuspender(
        _interlock(),
        Watched(value=gauge.pressure, connected={"U_Gauge": gauge.connected_status}),
        on_change=lambda s: changes.append(s.verdict.unsafe),
    )
    install_on_loop(RE, suspender)
    time.sleep(0.1)
    assert suspender.verdict is not None and not suspender.verdict.unsafe
    assert changes == []  # a first "safe" is not reported as a clear
