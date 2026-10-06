"""BackgroundSnapshot — the run's soft background telemetry (#1016, #929).

The object alone first (probe, exclusion, the soft read, the lifecycle),
then through the bound plans: the strict row and the gated sampler both
carry the background columns, the switch, the start-document record.
"""

from __future__ import annotations

import asyncio
import logging
import math
from functools import partial

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from ophyd_async.core import (  # noqa: E402
    Device,
    DeviceConnector,
    NotConnectedError,
    set_mock_value,
)

from geecs_bluesky.devices.background import (  # noqa: E402
    BackgroundSnapshot,
    _blank,
    _invalid,
)
from geecs_bluesky.devices.ca import CaMotor, CaSnapshotReadable  # noqa: E402
from geecs_bluesky.devices.detector import GeecsDetector  # noqa: E402
from geecs_bluesky.devices.shot_control import ShotControl  # noqa: E402
from geecs_bluesky.plans.registry import (  # noqa: E402
    TriggerProfiles,
    bind_plans,
    resolve_background_telemetry,
)
from geecs_bluesky.preprocessors import scalar_headers  # noqa: E402
from geecs_bluesky.utils import is_connected  # noqa: E402
from tests.ca_mock_helpers import DocCollector, connect_mock, follow_setpoint  # noqa: E402
from tests.test_plan_registry import payload  # noqa: E402
from tests.test_strict_plans import WRITES, FakeBox, _camera  # noqa: E402


class Fake:
    """A member with scripted answers: it connects, stages, reads as told."""

    parent = None

    def __init__(
        self,
        name: str,
        *,
        geecs: str = "U_Fake",
        connectable: bool = True,
        hang: bool = False,
        value: float = 1.0,
        severity: int = 0,
    ) -> None:
        self.name = name
        self._geecs_device_name = geecs
        self.connectable = connectable
        self.hang = hang
        self.value = value
        self.severity = severity
        self.fail_read = False
        self.staged = 0
        self.unstaged = 0
        self._mock = None  # what ``is_connected`` reads
        self._column_headers = {f"{name}-x": f"{geecs} X"}

    async def connect(self, mock=False, timeout=10.0, force_reconnect=False):
        if self.hang:
            await asyncio.sleep(timeout + 5.0)
        if not self.connectable:
            raise NotConnectedError(f"ca://{self.name}")
        self._mock = object()

    def stage(self):
        self.staged += 1

    def unstage(self):
        self.unstaged += 1

    async def describe(self):
        return {f"{self.name}-x": {"source": "fake", "dtype": "number", "shape": []}}

    async def read(self):
        if self.fail_read:
            raise RuntimeError("gone")
        return {
            f"{self.name}-x": {
                "value": self.value,
                "timestamp": 1.0,
                "alarm_severity": self.severity,
            }
        }


def _magnet(RE: RunEngine) -> CaSnapshotReadable:
    """A scalar device with a scanned settable child, as the namespace builds one."""
    magnet = CaSnapshotReadable(
        "U_S1H", ["Voltage"], experiment="TestExp", name="u_s1h"
    )
    magnet.current = CaMotor("U_S1H", "Current", experiment="TestExp", tolerance=0.01)
    magnet.add_readables([magnet.current])  # subscribed settable: its readback logs
    connect_mock(RE, magnet)
    return magnet


def _gauge(RE: RunEngine, name: str = "U_Gauge", value: float = 2.5):
    gauge = CaSnapshotReadable(
        name, ["Pressure"], experiment="TestExp", name=name.lower()
    )
    connect_mock(RE, gauge)
    set_mock_value(gauge.pressure, value)
    return gauge


def _cam(RE: RunEngine, name: str = "UC_Cam"):
    cam = GeecsDetector(name, ["MeanCounts"], experiment="TestExp", name=name.lower())
    connect_mock(RE, cam)
    set_mock_value(cam.meancounts, 7.0)
    set_mock_value(cam.acq_timestamp, 1000.0)
    return cam


def run(RE: RunEngine, coro_factory):
    """Run *coro_factory()* on the RunEngine's loop (the probe is a coroutine)."""
    return asyncio.run_coroutine_threadsafe(coro_factory(), RE._loop).result(10.0)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


# ------------------------------------------------------------- the object
def test_probe_leaves_the_runs_own_devices_to_the_run(RE) -> None:
    """Exclusion is by root device: a detector's signals, a scalar device, a motor's owner."""
    gauge = _gauge(RE)
    cam = _cam(RE)
    magnet = _magnet(RE)
    other = _gauge(RE, "U_Other", 1.0)
    candidates = [gauge, *cam._scalar_signals(), magnet, other]
    snapshot = BackgroundSnapshot(candidates, probe_timeout=0.5)
    # The plan stages roots: the camera and the view's owner are the run's
    # own readers (their signals go, whatever else is scanned on them); the
    # magnet, staged only as the scanned child's root, is parked whole until
    # the step admits it.
    run(
        RE,
        lambda: snapshot.probe(staged=[cam, magnet, other], own=[cam, other.scalars]),
    )
    assert snapshot.members == [gauge]
    assert snapshot._parked == [magnet]
    assert snapshot.dropped == []
    keys = run(RE, snapshot.describe)
    assert set(keys) == {"u_gauge-pressure"}
    assert snapshot._column_headers == {"u_gauge-pressure": "U_Gauge Pressure"}
    # Nothing staged: everything is background, the detector's signals
    # included (read from its monitor cache: the last frame's scalars).
    run(RE, lambda: snapshot.probe(staged=[]))
    assert set(run(RE, snapshot.describe)) == {
        "u_gauge-pressure",
        "uc_cam-meancounts",
        "uc_cam-acq_timestamp",
        "u_s1h-voltage",
        "u_s1h-current-position",
        "u_other-pressure",
    }
    assert snapshot._column_headers["uc_cam-meancounts"] == "UC_Cam MeanCounts"


def test_the_step_admits_a_scanned_devices_other_variables(RE, monkeypatch) -> None:
    """A scanned child's device comes back minus the child's own column; once."""
    gauge = _gauge(RE)
    magnet = _magnet(RE)
    other = _gauge(RE, "U_Other", 1.0)
    snapshot = BackgroundSnapshot([gauge, magnet, other], probe_timeout=0.5)
    run(RE, lambda: snapshot.probe(staged=[magnet, other], own=[other]))
    assert snapshot.members == [gauge]
    probes: list[str] = []
    original = snapshot._probe_one

    async def counted(member, **kw):
        probes.append(member.name)
        return await original(member, **kw)

    monkeypatch.setattr(snapshot, "_probe_one", counted)
    # The step's mover is the magnet's child: the magnet returns without
    # the readback the row carries as the motor's own column.
    run(RE, lambda: snapshot.admit([magnet.current]))
    assert snapshot.members == [gauge, magnet]
    assert set(run(RE, snapshot.describe)) == {"u_gauge-pressure", "u_s1h-voltage"}
    assert snapshot._column_headers["u_s1h-voltage"] == "U_S1H Voltage"
    assert "u_s1h-current-position" not in snapshot._column_headers
    assert set(run(RE, snapshot.read)) == {"u_gauge-pressure", "u_s1h-voltage"}
    assert probes == ["u_s1h"]
    # Later steps: nothing left to decide, nothing probed again.
    run(RE, lambda: snapshot.admit([magnet.current]))
    assert probes == ["u_s1h"] and snapshot.members == [gauge, magnet]
    # A whole device moved stays the run's: nothing comes back for it.
    run(RE, lambda: snapshot.admit([other]))
    assert snapshot.members == [gauge, magnet] and probes == ["u_s1h"]
    # A mover whose device the run reads itself — listed whole or through
    # its view — was never parked: the row carries that device already.
    for own in ([magnet], [magnet.scalars]):
        snapshot = BackgroundSnapshot([gauge, magnet], probe_timeout=0.5)
        run(RE, lambda: snapshot.probe(staged=[magnet], own=own))
        assert snapshot._parked == []
        run(RE, lambda: snapshot.admit([magnet.current]))
        assert snapshot.members == [gauge]


def test_an_admitted_device_is_the_run_engines_to_stage_and_unstage(RE) -> None:
    """A parked device was staged by the RunEngine: the snapshot never stages or unstages it."""

    class Wide(Fake):  # two logged variables, one of them the axis
        async def describe(self):
            return {
                f"{self.name}-x": {"source": "fake", "dtype": "number", "shape": []},
                f"{self.name}-y": {"source": "fake", "dtype": "number", "shape": []},
            }

        async def read(self):
            return {
                f"{self.name}-x": {"value": 1.0, "timestamp": 1.0, "alarm_severity": 0},
                f"{self.name}-y": {"value": 2.0, "timestamp": 1.0, "alarm_severity": 0},
            }

    class Axis:  # the scanned child, as the step names it
        def __init__(self, parent, key):
            self.parent, self.name, self._key = parent, f"{parent.name}-axis", key

        async def describe(self):
            return {self._key: {"source": "fake", "dtype": "number", "shape": []}}

    wide = Wide("u_wide", geecs="U_Wide")
    narrow = Fake("u_narrow", geecs="U_Narrow")  # its only variable is the axis
    free = Fake("u_free", geecs="U_Free")
    snapshot = BackgroundSnapshot([wide, narrow, free])
    run(RE, lambda: snapshot.probe(staged=[wide, narrow]))
    assert snapshot.members == [free] and free.staged == 1
    run(
        RE, lambda: snapshot.admit([Axis(wide, "u_wide-x"), Axis(narrow, "u_narrow-x")])
    )
    assert snapshot.members == [free, wide]
    assert set(run(RE, snapshot.describe)) == {"u_free-x", "u_wide-y"}
    assert (wide.staged, wide.unstaged, narrow.staged, narrow.unstaged) == (0, 0, 0, 0)
    RE(bps.unstage(snapshot, wait=True))
    assert free.unstaged == 1 and (wide.unstaged, narrow.unstaged) == (0, 0)


def test_a_mover_that_does_not_describe_keeps_its_whole_device_out(RE, caplog) -> None:
    """Never a key twice: a mover the step cannot describe leaves its device parked."""
    gauge = _gauge(RE)
    magnet = _magnet(RE)

    class Mute:  # a mover whose describe never answers
        parent = magnet
        name = "u_s1h-current"

        async def describe(self):
            await asyncio.sleep(5.0)

    class Broken:  # one whose describe raises
        parent = magnet
        name = "u_s1h-current"

        async def describe(self):
            raise RuntimeError("no metadata")

    for mover in (Mute(), Broken()):
        snapshot = BackgroundSnapshot([gauge, magnet], probe_timeout=0.3)
        run(RE, lambda: snapshot.probe(staged=[magnet]))
        caplog.clear()
        with caplog.at_level(
            logging.WARNING, logger="geecs_bluesky.devices.background"
        ):
            run(RE, lambda: snapshot.admit([mover]))
        assert snapshot.members == [gauge]
        assert set(run(RE, snapshot.describe)) == {"u_gauge-pressure"}
        assert "U_S1H (u_s1h-current) did not describe" in caplog.text
        assert "whole device stays the run's this time" in caplog.text
        # Decided for the run: a later step does not ask again.
        caplog.clear()
        run(RE, lambda: snapshot.admit([mover]))
        assert snapshot.members == [gauge] and "did not describe" not in caplog.text


def test_probe_drops_what_does_not_answer_and_keeps_the_rest(RE, caplog) -> None:
    """An unserved PV and a hanging connect cost one bounded probe, never the run."""
    gauge = _gauge(RE)
    dead = Fake("u_dead", geecs="U_Dead", connectable=False)
    slow = Fake("u_slow", geecs="U_Slow", hang=True)
    snapshot = BackgroundSnapshot([dead, gauge, slow], probe_timeout=0.2)
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.background"):
        run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [gauge]
    assert snapshot.dropped == ["U_Dead", "U_Slow"]
    assert "left out of this run" in caplog.text
    assert "U_Dead (NotConnectedError: ca://u_dead)" in caplog.text
    assert "U_Slow (no answer within 0.2 s)" in caplog.text
    # A dropped member's half-staged cache is released.
    assert slow.unstaged == 1 and dead.unstaged == 1
    # Probed again at the next run: the device is back.
    dead.connectable = False
    slow.hang = False
    run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [gauge, slow] and snapshot.dropped == ["U_Dead"]


def test_probe_connects_a_member_only_once(RE) -> None:
    """A member connected already is not reconnected (a mock's callbacks would be lost)."""
    member = Fake("u_fake")
    snapshot = BackgroundSnapshot([member], probe_timeout=0.5)
    run(RE, lambda: snapshot.probe(staged=[]))
    token = member._mock
    run(RE, lambda: snapshot.probe(staged=[]))
    assert member._mock is token and member.staged == 2


def test_read_fills_every_declared_key_and_nans_the_dead(RE, caplog) -> None:
    """INVALID → NaN, MAJOR keeps its value, a failing member → NaN and one warning."""
    invalid = Fake("u_invalid", geecs="U_Invalid", value=3.0, severity=-1)
    major = Fake("u_major", geecs="U_Major", value=4.0, severity=2)
    live = Fake("u_live", geecs="U_Live", value=5.0)
    snapshot = BackgroundSnapshot([invalid, major, live], read_timeout=0.2)
    run(RE, lambda: snapshot.probe(staged=[]))
    row = run(RE, snapshot.read)
    assert math.isnan(row["u_invalid-x"]["value"])
    assert row["u_major-x"]["value"] == 4.0 and row["u_live-x"]["value"] == 5.0
    live.fail_read = True
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.background"):
        first = run(RE, snapshot.read)
        second = run(RE, snapshot.read)
    assert math.isnan(first["u_live-x"]["value"])
    assert math.isnan(second["u_live-x"]["value"])
    assert caplog.text.count("U_Live stopped answering") == 1
    assert set(second) == {"u_invalid-x", "u_major-x", "u_live-x"}


def test_read_outlasting_its_budget_is_nan_not_a_wait(RE) -> None:
    class Hanging(Fake):
        async def read(self):
            await asyncio.sleep(5.0)
            return await super().read()

    hanging = Hanging("u_hang", geecs="U_Hang")
    snapshot = BackgroundSnapshot([hanging], probe_timeout=0.2, read_timeout=0.2)
    run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == []  # the probe's first read did not land either
    snapshot._active = [hanging]  # as if it had: the per-shot backstop
    snapshot._datakeys = {id(hanging): {"u_hang-x": {"dtype": "number"}}}
    row = run(RE, snapshot.read)
    assert math.isnan(row["u_hang-x"]["value"])


def test_blank_and_invalid_rules() -> None:
    assert math.isnan(_blank({"dtype": "number"})["value"])
    assert math.isnan(_blank({"dtype": "integer"})["value"])
    assert _blank({"dtype": "string"})["value"] == ""
    assert _blank({"dtype": "boolean"})["value"] is False
    assert not _invalid({"value": 1.0, "timestamp": 0.0})  # no severity: live
    assert not _invalid({"value": 1.0, "timestamp": 0.0, "alarm_severity": 2})
    assert _invalid({"value": 1.0, "timestamp": 0.0, "alarm_severity": -1})
    assert _invalid({"value": 1.0, "timestamp": 0.0, "alarm_severity": 3})


def test_stage_resets_and_unstage_releases_the_members(RE) -> None:
    member = Fake("u_fake")
    snapshot = BackgroundSnapshot([member])
    run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [member] and member.staged == 1
    RE(bps.unstage(snapshot, wait=True))
    assert member.unstaged == 1 and snapshot.members == []
    RE(bps.stage(snapshot, wait=True))
    assert snapshot.members == [] and run(RE, snapshot.read) == {}


def test_a_run_reads_the_snapshot_per_row(RE) -> None:
    """Through the RunEngine: stage, the probe before open_run, one read per row."""
    gauge = _gauge(RE, value=2.5)
    dead = Fake("u_dead", geecs="U_Dead", connectable=False)
    snapshot = BackgroundSnapshot([gauge, dead], probe_timeout=0.2)
    col = DocCollector()
    RE.subscribe(col)

    def plan():
        yield from bps.stage(snapshot, wait=True)
        yield from bps.wait_for([partial(snapshot.probe, [])])
        yield from bps.open_run(md={"background_dropped": snapshot.dropped})
        for value in (2.5, 3.5):
            set_mock_value(gauge.pressure, value)
            yield from bps.create("primary")
            yield from bps.read(snapshot)
            yield from bps.save()
        yield from bps.close_run()
        yield from bps.unstage(snapshot, wait=True)

    RE(plan())
    events = col.primary_events()
    assert [e["data"]["u_gauge-pressure"] for e in events] == [2.5, 3.5]
    assert col.docs["start"][0]["background_dropped"] == ["U_Dead"]
    descriptor = col.docs["descriptor"][0]
    assert set(descriptor["data_keys"]) == {"u_gauge-pressure"}
    assert descriptor["object_keys"] == {"background": ["u_gauge-pressure"]}


# --------------------------------------------------------- the bound plans
class Namespace(dict):
    """A hermetic stand-in for ``GeecsNamespace``: the bindings plus a telemetry set."""

    def __init__(self, bindings: dict, telemetry: list) -> None:
        super().__init__(bindings)
        self._telemetry = list(telemetry)

    def telemetry(self) -> list:
        return list(self._telemetry)


class _Defaults:
    """A resolver whose experiment defaults carry one background_telemetry value."""

    def __init__(self, on: bool) -> None:
        self.value = on

    def resolve_experiment_defaults(self):
        from geecs_schemas import ExperimentDefaults

        return ExperimentDefaults(background_telemetry=self.value)


@pytest.fixture
def box() -> FakeBox:
    return FakeBox()


@pytest.fixture
def profiles(RE: RunEngine, box: FakeBox) -> TriggerProfiles:
    sc = ShotControl(WRITES, experiment="TestExp", name="htu_test", setter_factory=box)
    connect_mock(RE, sc)
    return TriggerProfiles({"HTU-Test": sc}, default="HTU-Test")


def test_bound_count_reads_the_background_into_every_row(
    RE, box, profiles, caplog
) -> None:
    """Strict: one more device of the row; the start document records the run's set."""
    cam = _camera(RE, box, "UC_Cam")
    set_mock_value(cam.meancounts, 7.0)
    gauge = _gauge(RE, value=2.5)
    dead = Fake("u_dead", geecs="U_Dead", connectable=False)
    ns = Namespace({"UC_Cam": cam}, [gauge, *cam._scalar_signals(), dead])
    RE.preprocessors.append(scalar_headers)
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles, settables=ns)["count"]
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.background"):
        RE(count([cam], 2))
    events = col.primary_events()
    assert box.fires == 2 and len(events) == 2
    assert [e["data"]["u_gauge-pressure"] for e in events] == [2.5, 2.5]
    assert [e["data"]["uc_cam-meancounts"] for e in events] == [7.0, 7.0]
    start = col.docs["start"][0]
    assert start["background_telemetry"] is True
    assert start["background_dropped"] == ["U_Dead"]
    assert start["detectors"] == ["uc_cam", "background"]
    assert start["geecs_scalar_headers"]["u_gauge-pressure"] == "U_Gauge Pressure"
    assert start["geecs_scalar_headers"]["uc_cam-meancounts"] == "UC_Cam MeanCounts"
    assert "U_Dead (NotConnectedError" in caplog.text
    # Probed again at the next run: the device is back, no reopen needed.
    dead.connectable = True
    RE(count([cam], 1))
    assert col.docs["start"][-1]["background_dropped"] == []
    assert col.primary_events()[-1]["data"]["u_dead-x"] == 1.0


def test_bound_sweep_keeps_the_scanned_devices_other_variables(
    RE, box, profiles
) -> None:
    """The axis is resolved inside the sweep: its readback is the row's, its device's other variables are background."""
    cam = _camera(RE, box, "UC_Cam")
    magnet = _magnet(RE)
    follow_setpoint(magnet.current)
    gauge = _gauge(RE)
    ns = Namespace(
        {"U_S1H": magnet, "UC_Cam": cam}, [magnet, gauge, *cam._scalar_signals()]
    )
    col = DocCollector()
    RE.subscribe(col)
    scan = bind_plans(profiles, settables=ns)["sweep"]
    RE(scan([cam.scalars], sweep=payload()))
    events = col.primary_events()
    assert len(events) == 3
    data = events[0]["data"]
    assert "u_gauge-pressure" in data and "uc_cam-meancounts" in data
    assert "u_s1h-voltage" in data  # the scanned device's other variable: background
    assert [e["data"]["u_s1h-current-position"] for e in events] == pytest.approx(
        [-1.0, 0.0, 1.0]
    )  # the readback: the motor's own column, once
    descriptor = col.docs["descriptor"][0]["object_keys"]
    assert "u_s1h-voltage" in descriptor["background"]
    assert "u_s1h-current-position" not in descriptor["background"]
    assert col.docs["start"][0]["background_dropped"] == []


def test_bound_sweep_leaves_a_device_the_run_reads_alone(RE, box, profiles) -> None:
    """``sweep([X], X.current)``, ``[X.scalars]``, a camera's child: the row has the device whole, the background nothing of it."""
    cam = _camera(RE, box, "UC_Cam")
    cam.exposure = CaMotor("UC_Cam", "Exposure", experiment="TestExp", tolerance=0.01)
    cam.add_readables([cam.exposure])
    magnet = _magnet(RE)
    connect_mock(RE, cam.exposure)
    follow_setpoint(magnet.current)
    follow_setpoint(cam.exposure)
    gauge = _gauge(RE)
    ns = Namespace(
        {"U_S1H": magnet, "UC_Cam": cam}, [magnet, gauge, *cam._scalar_signals()]
    )
    sweep = bind_plans(profiles, settables=ns)["sweep"]
    cases = [
        (
            [cam.scalars, magnet],
            payload(),
            "u_s1h-current-position",
            ("u_s1h", "uc_cam"),
        ),
        (
            [cam.scalars, magnet.scalars],
            payload(),
            "u_s1h-current-position",
            ("u_s1h", "uc_cam"),
        ),
        (
            [cam.scalars],
            payload("UC_Cam.exposure", 1, 3, 3),
            "uc_cam-exposure-position",
            ("uc_cam",),
        ),
    ]
    for detectors, trajectory, axis_key, own in cases:
        col = DocCollector()
        token = RE.subscribe(col)
        RE(sweep(detectors, sweep=trajectory))
        RE.unsubscribe(token)
        assert col.docs["stop"][-1]["exit_status"] == "success"
        events = col.primary_events()
        assert len(events) == 3
        data = events[0]["data"]
        assert (
            axis_key in data and "u_s1h-voltage" in data and "uc_cam-meancounts" in data
        )
        background = col.docs["descriptor"][0]["object_keys"]["background"]
        assert "u_gauge-pressure" in background
        assert not [k for k in background if k.split("-")[0] in own]
    # The third case: the magnet is nobody's, so its voltage came from the background.
    assert "u_s1h-voltage" in background


def test_gated_rows_carry_the_background(RE, monkeypatch, tmp_path) -> None:
    """Gated: a sampler member, read at the tick."""
    from geecs_bluesky.plans import gated
    from tests.test_gated_plans import GATED_WRITES, GatedBox, _events_from_pages
    from tests.test_strict_plans import _plugin_camera

    monkeypatch.setattr(gated, "TRIGGER_PERIOD_S", 0.08)
    monkeypatch.setattr(gated, "DRAIN_MARGIN_S", 0.04)
    box = GatedBox()
    sc = ShotControl(
        GATED_WRITES, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    profiles = TriggerProfiles({"HTU-Test": sc}, default="HTU-Test")
    a, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    gauge = _gauge(RE, value=2.5)
    ns = Namespace({"UC_A": a}, [gauge, *a._scalar_signals()])
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles, settables=ns)["count"]
    RE(count([a], 3, acquisition="gated"))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    rows = _events_from_pages(col, "shots")
    assert len(rows) == 3
    assert [r["data"]["u_gauge-pressure"] for r in rows] == [2.5, 2.5, 2.5]
    assert "uc_a-acq_timestamp" in rows[0]["data"]  # the clock, once
    assert col.docs["start"][0]["background_telemetry"] is True


def test_gated_sweep_keeps_the_scanned_devices_other_variables(
    RE, monkeypatch, tmp_path
) -> None:
    """Gated: the step admits the axis's device before the ``shots`` stream is declared."""
    from geecs_bluesky.plans import gated
    from tests.test_gated_plans import GATED_WRITES, GatedBox, _events_from_pages
    from tests.test_strict_plans import _plugin_camera

    monkeypatch.setattr(gated, "TRIGGER_PERIOD_S", 0.08)
    monkeypatch.setattr(gated, "DRAIN_MARGIN_S", 0.04)
    box = GatedBox()
    sc = ShotControl(
        GATED_WRITES, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    profiles = TriggerProfiles({"HTU-Test": sc}, default="HTU-Test")
    a, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    magnet = _magnet(RE)
    follow_setpoint(magnet.current)
    gauge = _gauge(RE, value=2.5)
    ns = Namespace({"U_S1H": magnet, "UC_A": a}, [magnet, gauge, *a._scalar_signals()])
    col = DocCollector()
    RE.subscribe(col)
    sweep = bind_plans(profiles, settables=ns)["sweep"]
    RE(sweep([a], sweep=payload(), acquisition="gated"))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    rows = _events_from_pages(col, "shots")
    assert len(rows) == 3
    assert [r["data"]["u_gauge-pressure"] for r in rows] == [2.5, 2.5, 2.5]
    assert "u_s1h-voltage" in rows[0]["data"]  # the scanned device's other variable
    assert [r["data"]["u_s1h-current-position"] for r in rows] == pytest.approx(
        [-1.0, 0.0, 1.0]
    )  # the readback once, the motor's own
    assert col.docs["start"][0]["background_dropped"] == []
    # The magnet listed as a detector and scanned: the row carries it whole,
    # the step brings nothing back, the batch still runs.
    col = DocCollector()
    RE.subscribe(col)
    RE(sweep([a, magnet], sweep=payload(), acquisition="gated"))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    rows = _events_from_pages(col, "shots")
    assert len(rows) == 3 and "u_s1h-voltage" in rows[0]["data"]
    assert [r["data"]["u_s1h-current-position"] for r in rows] == pytest.approx(
        [-1.0, 0.0, 1.0]
    )


def test_the_switch_reads_the_defaults_per_run_and_the_item_wins(
    RE, box, profiles
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    gauge = _gauge(RE)
    ns = Namespace({"UC_Cam": cam}, [gauge])
    col = DocCollector()
    RE.subscribe(col)
    defaults = _Defaults(False)
    count = bind_plans(profiles, resolver=defaults, settables=ns)["count"]
    RE(count([cam], 1))
    start = col.docs["start"][-1]
    assert start["background_telemetry"] is False
    assert "background_dropped" not in start and start["detectors"] == ["uc_cam"]
    assert "u_gauge-pressure" not in col.primary_events()[-1]["data"]
    defaults.value = True  # the file was edited: no rebind, no reopen
    RE(count([cam], 1))
    assert col.docs["start"][-1]["background_telemetry"] is True
    assert col.primary_events()[-1]["data"]["u_gauge-pressure"] == 2.5
    RE(count([cam], 1, background_telemetry=False))
    assert col.docs["start"][-1]["background_telemetry"] is False
    assert "u_gauge-pressure" not in col.primary_events()[-1]["data"]


def test_the_switch_is_on_when_the_defaults_cannot_be_read(caplog) -> None:
    class Broken:
        def resolve_experiment_defaults(self):
            raise OSError("configs root unreadable")

    assert resolve_background_telemetry(None, None) is True
    assert resolve_background_telemetry(False, Broken()) is False
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.plans.registry"):
        assert resolve_background_telemetry(None, Broken()) is True
    assert "background telemetry stays on" in caplog.text


def test_no_telemetry_set_or_nothing_outside_the_run_is_fine(RE, box, profiles):
    cam = _camera(RE, box, "UC_Cam")
    col = DocCollector()
    RE.subscribe(col)
    # A mapping without a telemetry set (hermetic tests): no background at all.
    RE(bind_plans(profiles, settables={"UC_Cam": cam})["count"]([cam], 1))
    start = col.docs["start"][-1]
    assert start["background_telemetry"] is False
    assert "background_dropped" not in start and start["detectors"] == ["uc_cam"]
    # Every candidate is the run's own: the snapshot rides along, empty.
    ns = Namespace({"UC_Cam": cam}, list(cam._scalar_signals()))
    RE(bind_plans(profiles, settables=ns)["count"]([cam], 1))
    start = col.docs["start"][-1]
    assert start["background_telemetry"] is True and start["background_dropped"] == []
    assert set(col.primary_events()[-1]["data"]) == {
        "uc_cam-meancounts",
        "uc_cam-acq_timestamp",
        "bin_number",
    }


# ------------------------------------------------------- review of #1018
class _ScriptedConnector(DeviceConnector):
    """A real ophyd-async connect: *delay* seconds, then served or not."""

    def __init__(self, delay: float, served: bool) -> None:
        self.delay = delay
        self.served = served

    async def connect_real(self, device, timeout, force_reconnect):
        await asyncio.sleep(self.delay)
        if not self.served:
            raise NotConnectedError(f"ca://{device.name}")

    async def connect_mock(self, device, mock):
        return None


class UnservedDevice(Device):
    """A real ophyd-async device (a cached connect task) whose PV may not be served."""

    def __init__(self, name: str, *, delay: float, served: bool) -> None:
        self._geecs_device_name = name.upper()
        self.scripted = _ScriptedConnector(delay, served)
        super().__init__(name=name, connector=self.scripted)

    async def describe(self):
        return {f"{self.name}-x": {"source": "fake", "dtype": "number", "shape": []}}

    async def read(self):
        return {f"{self.name}-x": {"value": 1.0, "timestamp": 1.0, "alarm_severity": 0}}


def test_a_connect_outlasting_the_budget_never_poisons_the_next_probe(RE) -> None:
    """P1 of the review: the probe's timeout must not cancel ophyd-async's cached connect.

    A cancelled connect task made ``is_connected`` raise and ophyd-async's
    ``connect`` refuse ever after, so from the second run on the probe died
    silently and every background column vanished.
    """
    gauge = _gauge(RE)
    slow = UnservedDevice("u_slow", delay=0.5, served=False)
    snapshot = BackgroundSnapshot([slow, gauge], probe_timeout=0.2)
    run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [gauge] and snapshot.dropped == ["U_SLOW"]
    assert snapshot.probe_error == ""
    # At once, while the shielded connect is still pending: the same verdict.
    run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [gauge] and snapshot.dropped == ["U_SLOW"]
    # The connect ended on its own (its own timeout): a verdict, not a cancel.
    run(RE, lambda: asyncio.sleep(0.6))
    assert slow._connect_task.done() and not slow._connect_task.cancelled()
    assert run(RE, lambda: asyncio.sleep(0, result=is_connected(slow))) is False
    # The PV is served now (a gateway restart): the next probe has it.
    slow.scripted.served, slow.scripted.delay = True, 0.0
    run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [slow, gauge] and snapshot.dropped == []


def test_the_probe_connects_what_nobody_connected_yet(RE, box, profiles) -> None:
    """A member never touched before the run is connected by the probe (mock here)."""
    cam = _camera(RE, box, "UC_Cam")
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    assert not is_connected(gauge)
    ns = Namespace({"UC_Cam": cam}, [gauge])
    col = DocCollector()
    RE.subscribe(col)
    RE(bind_plans(profiles, settables=ns, mock=True)["count"]([cam], 1))
    assert is_connected(gauge)
    assert "u_gauge-pressure" in col.primary_events()[-1]["data"]
    assert col.docs["start"][-1]["background_dropped"] == []


def test_a_failed_probe_is_recorded_and_the_run_still_opens(
    RE, box, profiles, monkeypatch, caplog
) -> None:
    """P2 of the review: the probe's own failure is loud and in the start document."""
    cam = _camera(RE, box, "UC_Cam")
    gauge = _gauge(RE)
    ns = Namespace({"UC_Cam": cam}, [gauge])

    async def broken(self, staged, own=()):
        raise RuntimeError("an unexpected staged object")

    monkeypatch.setattr(BackgroundSnapshot, "_probe", broken)
    col = DocCollector()
    RE.subscribe(col)
    with caplog.at_level(logging.ERROR, logger="geecs_bluesky.devices.background"):
        RE(bind_plans(profiles, settables=ns)["count"]([cam], 1))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    start = col.docs["start"][-1]
    assert start["background_telemetry"] is True and start["background_dropped"] == []
    assert (
        start["background_probe_error"] == "RuntimeError: an unexpected staged object"
    )
    assert "u_gauge-pressure" not in col.primary_events()[-1]["data"]
    assert "the probe failed" in caplog.text


def test_a_member_invalid_at_the_start_is_kept_and_named(RE, caplog) -> None:
    """Served but INVALID (the gateway marks the device down): NaN, and said once."""
    stale = Fake("u_stale", geecs="U_Stale", value=3.0, severity=-1)
    live = Fake("u_live", geecs="U_Live", value=5.0)
    snapshot = BackgroundSnapshot([stale, live])
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.background"):
        run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [stale, live] and snapshot.dropped == []
    assert "U_Stale (served but INVALID at the start)" in caplog.text
    row = run(RE, snapshot.read)
    assert math.isnan(row["u_stale-x"]["value"]) and row["u_live-x"]["value"] == 5.0


# --------------------------------------------------- the environment-open warm-up
def test_warm_up_connects_the_unconnected_and_names_the_rest(RE, caplog) -> None:
    """26_0929 Scan001: the first run must not pay every first connect in its probe."""
    from geecs_bluesky.devices.background import warm_up_on

    fresh = Fake("u_fresh", geecs="U_Fresh")
    done = Fake("u_done", geecs="U_Done")
    run(RE, lambda: done.connect())
    token = done._mock
    dead = Fake("u_dead", geecs="U_Dead", connectable=False)
    slow = UnservedDevice("u_slow", delay=0.5, served=False)
    with caplog.at_level(logging.INFO, logger="geecs_bluesky.devices.background"):
        failed = warm_up_on(RE, [fresh, done, dead, slow], timeout=0.2)
    assert failed == ["U_Dead", "U_SLOW"]
    assert is_connected(fresh) and done._mock is token  # connected; left alone
    assert "3 of 4 candidate device(s)" not in caplog.text  # two failed, two connected
    assert "2 of 4 candidate device(s) connected at environment open" in caplog.text
    assert (
        "not connected at environment open (probed again at every run): U_Dead, U_SLOW"
        in caplog.text
    )
    # The warm-up never poisons a later probe either: the slow one is a
    # verdict in the cache, the fresh one is a member at once.
    snapshot = BackgroundSnapshot([fresh, dead, slow], probe_timeout=0.2)
    run(RE, lambda: snapshot.probe(staged=[]))
    assert snapshot.members == [fresh] and snapshot.dropped == ["U_Dead", "U_SLOW"]
