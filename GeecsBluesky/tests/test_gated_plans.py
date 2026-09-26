"""The gated batch and the non-essential stream on mocks (phase 2b, #807).

Driven through the stock plans:
the fake trigger box free-runs a pacer while it stands in SCAN (every edge
advances every camera's stamp and every plugin-backed camera's frame
count), stops it on OFF and lets **one more edge** land shortly after —
the in-flight frame the plan's drain wait and ``truncate_to_quota`` exist
for.  Frames, stamps and rows are asserted from the documents.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("aioca")

import bluesky.plans as bp  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from bluesky.utils import RunEngineInterrupted  # noqa: E402
from ophyd_async.core import set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaMotor  # noqa: E402
from geecs_bluesky.devices.ca.snapshot import CaSnapshotReadable  # noqa: E402
from geecs_bluesky.devices.detector import GeecsDetector  # noqa: E402
from geecs_bluesky.devices.shot_control import ShotControl  # noqa: E402
from geecs_bluesky.exceptions import (  # noqa: E402
    GeecsConfigurationError,
    GeecsTriggerTimeoutError,
)
from geecs_bluesky.models.shot_control import ShotControlWrites  # noqa: E402
from geecs_bluesky.plans import gated  # noqa: E402
from geecs_bluesky.plans.gated import gated_per_shot, gated_per_step  # noqa: E402
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans  # noqa: E402
from tests.ca_mock_helpers import DocCollector, connect_mock, follow_setpoint  # noqa: E402
from tests.test_strict_plans import FakeBox, _camera, _plugin_camera  # noqa: E402

GATED_WRITES = ShotControlWrites(
    name="test",
    states={
        "OFF": [("DG", "Trigger.Source", "off")],
        "STANDBY": [("DG", "Trigger.Source", "edges")],
        "SCAN": [("DG", "Trigger.Source", "scan")],
        "ARMED": [("DG", "Trigger.Source", "single")],
        "SINGLESHOT": [("DG", "Trigger.ExecuteSingleShot", "on")],
    },
)


class GatedBox(FakeBox):
    """The fake box free-runs in SCAN: a pacer lands an edge every *interval*.

    OFF stops the pacer and lands **one more edge** half an interval later
    — the frame that was in flight.  ``stall`` names cameras that never
    receive a frame (a camera that stopped acquiring).
    """

    def __init__(
        self,
        interval: float = 0.05,
        *,
        late_edge: bool = True,
        stamp_step: float = 1.0,
    ) -> None:
        super().__init__()
        self.interval = interval
        #: Seconds between consecutive shots' ``acq_timestamp`` values — the
        #: rep rate the stamps claim, independent of the pacer's wall clock.
        self.stamp_step = stamp_step
        self.late_edge = (
            late_edge  # an edge in flight when OFF lands (the ~10% case at 1 Hz)
        )
        self.states: list[str] = []
        self.edges = 0
        self.stall: set[str] = set()
        self._pacer: asyncio.Task | None = None
        self._late: asyncio.TimerHandle | None = None

    def __call__(self, device: str, variable: str):
        box = self
        inner = super().__call__(device, variable)

        class Setter:
            async def put(self, value: str) -> None:
                if variable == "Trigger.ExecuteSingleShot":
                    # the strict fire: one edge, now (the stalled cameras excepted)
                    box.puts.append((device, variable, value))
                    await asyncio.sleep(0.02)
                    box.fires += 1
                    box.edge()
                    return
                if variable != "Trigger.Source":
                    await inner.put(value)
                    return
                box.puts.append((device, variable, value))
                box.states.append(value)
                if value == "scan":
                    box._start()
                else:
                    box._stop()

        return Setter()

    def edge(self) -> None:
        self.edges += 1
        self.stamp += self.stamp_step
        for cam in self.cameras:
            if cam.name in self.stall or (cam.name, self.edges) in self.drop:
                continue
            hdf = getattr(cam, "hdf", None)
            if hdf is not None:
                self.counts[cam.name] = self.counts.get(cam.name, 0) + 1
                set_mock_value(hdf.num_captured, self.counts[cam.name])
            set_mock_value(cam.acq_timestamp, self.stamp)

    def _start(self) -> None:
        if self._late is not None:
            self._late.cancel()
            self._late = None
        if self._pacer is None or self._pacer.done():
            self._pacer = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            self.edge()

    def _stop(self) -> None:
        if self._pacer is not None and not self._pacer.done():
            self._pacer.cancel()
            if self.late_edge:
                loop = asyncio.get_running_loop()
                self._late = loop.call_later(self.interval / 2, self.edge)
        self._pacer = None

    @property
    def scan_runs(self) -> int:
        return self.states.count("scan")


@pytest.fixture(autouse=True)
def _short_drain(monkeypatch):
    monkeypatch.setattr(gated, "TRIGGER_PERIOD_S", 0.08)
    monkeypatch.setattr(gated, "DRAIN_MARGIN_S", 0.04)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def box() -> GatedBox:
    return GatedBox()


@pytest.fixture
def shot_control(RE: RunEngine, box: GatedBox) -> ShotControl:
    sc = ShotControl(
        GATED_WRITES, experiment="TestExp", name="shot_control", setter_factory=box
    )
    connect_mock(RE, sc)
    return sc


@pytest.fixture
def profiles(shot_control: ShotControl) -> TriggerProfiles:
    return TriggerProfiles({"HTU-Test": shot_control}, default="HTU-Test")


def _datums_by_key(col: DocCollector) -> dict[str, list[dict]]:
    resources = {r["uid"]: r["data_key"] for r in col.docs["stream_resource"]}
    out: dict[str, list[dict]] = {}
    for d in col.docs["stream_datum"]:
        out.setdefault(resources[d["stream_resource"]], []).append(d["indices"])
    return out


def _stream_events(col: DocCollector, name: str) -> list[dict]:
    uids = {d["uid"] for d in col.docs["descriptor"] if d["name"] == name}
    return [e for e in col.docs["event"] if e["descriptor"] in uids]


def _events_from_pages(col: DocCollector, name: str) -> list[dict]:
    """The ``shots`` rows arrive as event pages; unpack them."""
    uids = {d["uid"] for d in col.docs["descriptor"] if d["name"] == name}
    rows = []
    for page in col.docs["event_page"]:
        if page["descriptor"] not in uids:
            continue
        n = len(page["seq_num"])
        for i in range(n):
            rows.append(
                {
                    "seq_num": page["seq_num"][i],
                    "data": {k: v[i] for k, v in page["data"].items()},
                }
            )
    return rows + _stream_events(col, name)


def _magnet(RE: RunEngine) -> CaMotor:
    magnet = CaMotor(
        "U_S1H", "Current", experiment="TestExp", tolerance=0.01, name="u_s1h-current"
    )
    connect_mock(RE, magnet)
    follow_setpoint(magnet)
    return magnet


# ------------------------------------------------------------ gated count
def test_gated_count_two_plugin_cameras(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """One batch: SCAN until both counted 3, OFF, the in-flight frame trimmed.

    ``primary`` is a datum stream (one datum per camera, width 3, no
    events); ``shots`` carries one event per shot with the clock stamp and
    ``bin_number = 1``; the run closes in STANDBY.
    """
    a, a_rewinds = _plugin_camera(RE, box, "UC_A", tmp_path)
    b, b_rewinds = _plugin_camera(RE, box, "UC_B", tmp_path)
    # A's plugin still reports the previous session's count (found on
    # hardware, A2): the run's first arm zeroes it before the batch baselines
    box.counts["uc_a"] = 5
    set_mock_value(a.hdf.num_captured, 5)
    set_mock_value(a.meancounts, 5.0)
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([a, b], 3, per_shot=gated_per_shot(shot_control, quota=3)))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert {d["name"] for d in col.docs["descriptor"]} == {"primary", "shots"}
    assert _stream_events(col, "primary") == []  # datums only
    datums = _datums_by_key(col)
    assert datums["uc_a"] == [{"start": 0, "stop": 3}]
    assert datums["uc_b"] == [{"start": 0, "stop": 3}]
    # the stale count zeroed at the first arm, then the in-flight edge after
    # OFF landed a 4th frame of the batch and the trim took it out
    assert a_rewinds == [0, 3] and b_rewinds == [0, 3]
    rows = _events_from_pages(col, "shots")
    assert len(rows) == 3
    assert [r["data"]["bin_number"] for r in rows] == [1, 1, 1]
    stamps = [r["data"]["uc_a-acq_timestamp"] for r in rows]  # the clock column
    assert stamps == sorted(stamps) and len(set(stamps)) == 3
    assert (
        "uc_a-meancounts" not in rows[0]["data"]
    )  # a plugin camera's scalars ride in its stack
    # the hook alone: OFF opens the step and OFF closes it (the bound plan
    # adds the run bracket, STANDBY at the end — see the bound-plan tests)
    assert box.states[0] == "off" and box.states[-1] == "off"
    assert box.scan_runs == 1
    assert shot_control.standing_state == "OFF"


def test_gated_scan_batches_per_position_with_motor_and_bin(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """Three positions, two shots each: one datum per camera per step, six rows."""
    cam, rewinds = _plugin_camera(RE, box, "UC_A", tmp_path)
    magnet = _magnet(RE)
    col = DocCollector()
    RE.subscribe(col)
    RE(
        bp.scan(
            [cam],
            magnet,
            -1.0,
            1.0,
            3,
            per_step=gated_per_step(shot_control, shots_per_step=2),
        )
    )
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert _datums_by_key(col)["uc_a"] == [
        {"start": 0, "stop": 2},
        {"start": 2, "stop": 4},
        {"start": 4, "stop": 6},
    ]
    assert rewinds == [
        0,
        2,
        4,
        6,
    ]  # the first arm's zero, then baseline + quota per step
    rows = _events_from_pages(col, "shots")
    assert [r["data"]["bin_number"] for r in rows] == [1, 1, 2, 2, 3, 3]
    assert [r["data"]["u_s1h-current-position"] for r in rows] == pytest.approx(
        [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]
    )
    assert box.scan_runs == 3
    assert list(col.docs["start"][0]["motors"]) == ["u_s1h-current"]


def test_gated_with_a_triggered_scalar_clock_and_no_camera(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl
) -> None:
    """An ICT-like device (a stamp, no plugin) clocks the batch: shots only, no datums."""
    ict = _camera(RE, box, "U_ICT")  # a GeecsDetector without a plugin
    set_mock_value(ict.meancounts, 0.42)
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, gauge)
    set_mock_value(gauge.pressure, 1e-6)
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([ict, gauge], 4, per_shot=gated_per_shot(shot_control, quota=4)))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert {d["name"] for d in col.docs["descriptor"]} == {"shots"}
    assert col.docs["stream_datum"] == []
    rows = _events_from_pages(col, "shots")
    assert len(rows) == 4
    assert [r["data"]["u_ict-meancounts"] for r in rows] == [0.42] * 4
    assert [r["data"]["u_gauge-pressure"] for r in rows] == [1e-6] * 4
    stamps = [r["data"]["u_ict-acq_timestamp"] for r in rows]
    assert stamps == sorted(stamps) and len(set(stamps)) == 4


def test_gated_refuses_a_step_with_no_triggered_device(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl
) -> None:
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, gauge)
    with pytest.raises(GeecsConfigurationError, match="nothing here counts shots"):
        RE(bp.count([gauge], 2, per_shot=gated_per_shot(shot_control, quota=2)))
    assert box.scan_runs == 0


def test_gated_stalled_camera_fails_loudly_with_the_box_off(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """A camera that counts nothing while the box runs: the step fails, naming it."""
    a, _ = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=0.3)
    b, _ = _plugin_camera(RE, box, "UC_B", tmp_path, shot_timeout=0.3)
    box.stall = {"uc_b"}
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(GeecsTriggerTimeoutError, match="UC_B"):
        RE(
            bp.count(
                [a, b],
                3,
                per_shot=gated_per_shot(shot_control, quota=3, shot_timeout=0.3),
            )
        )
    assert box.states[-1] == "off"  # OFF on failure
    assert shot_control.standing_state == "OFF"


def test_immediate_pause_mid_batch_retakes_the_step(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """Sam 2026-09-12: an immediate pause drives OFF; resume retakes the step.

    The pause lands after a couple of shots; ``ShotControl.pause`` drives
    OFF, ``resume`` restores SCAN before the plan runs again (so edges
    slip in), the plan sees the pause counter advanced, abandons the
    batch, rewinds the stack to the step's baseline and takes the whole
    step again: the datum still covers exactly ``quota`` frames.
    """
    cam, rewinds = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=0.4)
    col = DocCollector()
    RE.subscribe(col)

    def pause_soon() -> None:
        while box.edges < 2:
            time.sleep(0.01)
        RE.request_pause()

    threading.Thread(target=pause_soon, daemon=True).start()
    with pytest.raises(RunEngineInterrupted):
        RE(
            bp.count(
                [cam],
                6,
                per_shot=gated_per_shot(shot_control, quota=6, shot_timeout=0.4),
            )
        )
    assert RE.state == "paused"
    assert box.states[-1] == "off"  # pause() drove OFF
    time.sleep(0.2)
    RE.resume()
    assert col.docs["stop"][-1]["exit_status"] == "success"
    # the interrupted batch, the resume's restore (SCAN before the plan
    # runs, undone by the step's OFF), then the retake
    assert box.scan_runs == 3, box.states
    datums = _datums_by_key(col)["uc_a"]
    assert datums == [{"start": 0, "stop": 6}]
    # rewound to the step's baseline (0) before the retake, then trimmed to 6
    assert rewinds[0] == 0 and rewinds[-1] == 6
    rows = _events_from_pages(col, "shots")
    assert len(rows) == 6
    assert shot_control.standing_state == "OFF"


def test_immediate_pause_with_a_native_essential_toggles_saving_once(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """The retake path (review finding 2): the native saver is prepared once, before the pause.

    The retaken step keeps writing under the same run-long ``save=on`` —
    no second prepare, no toggle; the rows are the retake's quota and the
    save path is constant across the abandoned attempt and the retake.
    """
    (tmp_path / "Scan001").mkdir()
    cam, _ = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=0.4)
    native = _camera(RE, box, "UC_Native", tmp_path=tmp_path, shot_timeout=0.4)
    saves = _saves(native)
    col = DocCollector()
    RE.subscribe(col)

    def pause_soon() -> None:
        while box.edges < 2:
            time.sleep(0.01)
        RE.request_pause()

    threading.Thread(target=pause_soon, daemon=True).start()
    with pytest.raises(RunEngineInterrupted):
        RE(
            bp.count(
                [cam, native],
                6,
                per_shot=gated_per_shot(shot_control, quota=6, shot_timeout=0.4),
            )
        )
    assert RE.state == "paused"
    assert saves == ["off", "on"]  # prepared before the pause, still saving
    time.sleep(0.2)
    RE.resume()
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert box.scan_runs == 3, box.states
    assert saves == ["off", "on", "off"]  # the retake toggled nothing
    rows = _events_from_pages(col, "shots")
    assert len(rows) == 6
    directory = str(tmp_path / "Scan001" / "UC_Native")
    assert [r["data"]["uc_native-nonscalar_save_path"] for r in rows] == [directory] * 6
    assert _datums_by_key(col)["uc_a"] == [{"start": 0, "stop": 6}]


def test_pause_resumed_inside_the_count_timeout_window_still_retakes(
    RE: RunEngine, tmp_path: Path
) -> None:
    """Review of #850 finding 1: no edge in flight at OFF, resume 0.3 s later.

    The camera's ``complete`` (0.4 s per-frame budget) times out *after*
    the resume but *before* the plan's settle: the abandonment is marked
    synchronously the moment the interrupted wait returns, so the late
    failure is pardoned instead of thrown into the plan at ``mv(OFF)`` or
    the drain sleep — and the step is retaken.
    """
    box = GatedBox(late_edge=False)
    sc = ShotControl(GATED_WRITES, experiment="TestExp", name="sc", setter_factory=box)
    connect_mock(RE, sc)
    cam, rewinds = _plugin_camera(RE, box, "UC_A", tmp_path, shot_timeout=0.4)
    col = DocCollector()
    RE.subscribe(col)

    def pause_soon() -> None:
        while box.edges < 2:
            time.sleep(0.01)
        RE.request_pause()

    threading.Thread(target=pause_soon, daemon=True).start()
    with pytest.raises(RunEngineInterrupted):
        RE(bp.count([cam], 6, per_shot=gated_per_shot(sc, quota=6, shot_timeout=0.4)))
    time.sleep(0.3)  # inside the 0.4 s window measured from the last frame
    RE.resume()
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert _datums_by_key(col)["uc_a"] == [{"start": 0, "stop": 6}]
    assert len(_events_from_pages(col, "shots")) == 6
    assert rewinds[0] == 0  # the partial frames left the stack before the retake


def _saves(device: GeecsDetector) -> list[str]:
    """Every put to the device's ``save`` control, in order."""
    from ophyd_async.core import callback_on_mock_put

    puts: list[str] = []
    callback_on_mock_put(device.save, lambda value, **_: puts.append(value))
    return puts


def _shots_descriptor(col: DocCollector) -> dict:
    return next(d for d in col.docs["descriptor"] if d["name"] == "shots")


def test_a_native_saving_device_is_a_gated_essential_saving_run_long(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """The 2026-09-25 ruling: no plugin, LabVIEW files as the record — admitted.

    Beside the plugin camera (the clock), the native saver is a sampler
    member: saving switches on exactly once (the run's first prepare) and
    off exactly once (unstage); every ``shots`` row carries its scalars,
    its stamp and its save path as a run-long constant.
    """
    (tmp_path / "Scan001").mkdir()
    plugin, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    native = _camera(RE, box, "UC_Native", tmp_path=tmp_path)
    set_mock_value(native.meancounts, 7.0)
    saves = _saves(native)
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([plugin, native], 3, per_shot=gated_per_shot(shot_control, quota=3)))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    # stage clears a stale flag, the first prepare switches on, unstage off
    assert saves == ["off", "on", "off"]
    directory = tmp_path / "Scan001" / "UC_Native"
    assert directory.is_dir()
    rows = _events_from_pages(col, "shots")
    assert len(rows) == 3
    assert [r["data"]["uc_native-nonscalar_save_path"] for r in rows] == [
        str(directory)
    ] * 3
    assert [r["data"]["uc_native-meancounts"] for r in rows] == [7.0] * 3
    stamps = [r["data"]["uc_native-acq_timestamp"] for r in rows]
    assert stamps == sorted(stamps) and len(set(stamps)) == 3
    keys = _shots_descriptor(col)["data_keys"]
    assert keys["uc_native-nonscalar_save_path"]["dtype"] == "string"
    assert "uc_a-nonscalar_save_path" not in keys  # the plugin camera: its stack
    assert _datums_by_key(col)["uc_a"] == [{"start": 0, "stop": 3}]
    assert box.scan_runs == 1 and shot_control.standing_state == "OFF"


def test_a_native_essential_keeps_saving_across_the_steps_of_a_gated_scan(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """Three positions, two shots each: one ``save=on``, one ``save=off`` — never per step.

    A toggle costs the device a LabVIEW loop period and the box is OFF
    between steps, so the saving is run-long by design; with no plugin
    camera the native saver is the clock and ``primary`` is never declared.
    """
    (tmp_path / "Scan001").mkdir()
    native = _camera(RE, box, "UC_Native", tmp_path=tmp_path)
    magnet = _magnet(RE)
    saves = _saves(native)
    col = DocCollector()
    RE.subscribe(col)
    RE(
        bp.scan(
            [native],
            magnet,
            -1.0,
            1.0,
            3,
            per_step=gated_per_step(shot_control, shots_per_step=2),
        )
    )
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert saves == ["off", "on", "off"]
    assert box.scan_runs == 3
    assert {d["name"] for d in col.docs["descriptor"]} == {"shots"}
    rows = _events_from_pages(col, "shots")
    assert [r["data"]["bin_number"] for r in rows] == [1, 1, 2, 2, 3, 3]
    assert [r["data"]["u_s1h-current-position"] for r in rows] == pytest.approx(
        [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]
    )
    directory = str(tmp_path / "Scan001" / "UC_Native")
    assert [r["data"]["uc_native-nonscalar_save_path"] for r in rows] == [directory] * 6
    stamps = [r["data"]["uc_native-acq_timestamp"] for r in rows]
    assert stamps == sorted(stamps) and len(set(stamps)) == 6


def test_a_native_devices_scalars_view_saves_nothing_in_a_gated_run(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """``save_images: false``: the view rides in the sampler, the owner writes nothing."""
    (tmp_path / "Scan001").mkdir()
    plugin, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    native = _camera(RE, box, "UC_Native", tmp_path=tmp_path)
    saves = _saves(native)
    col = DocCollector()
    RE.subscribe(col)
    RE(
        bp.count(
            [plugin, native.scalars], 2, per_shot=gated_per_shot(shot_control, quota=2)
        )
    )
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert "on" not in saves
    assert not (tmp_path / "Scan001" / "UC_Native").exists()
    rows = _events_from_pages(col, "shots")
    assert "uc_native-meancounts" in rows[0]["data"]
    assert "uc_native-nonscalar_save_path" not in rows[0]["data"]


def test_gated_quota_one_is_a_fly_prepare_without_native_saving(
    RE: RunEngine, box: GatedBox, shot_control: ShotControl, tmp_path: Path
) -> None:
    """Review of #850 finding 2: shots_per_step=1 (the default) must not switch save=on.

    A plugin-backed camera in a gated run never saves natively (the stack
    is its record; the #738 dual-write is strict-only) — whatever the
    2026-09-25 ruling admits for a device *without* a plugin.
    """
    from ophyd_async.core import StaticFilenameProvider, StaticPathProvider

    (tmp_path / "Scan001").mkdir()
    provider = StaticPathProvider(
        StaticFilenameProvider("UC_Both"), tmp_path / "Scan001" / "UC_Both"
    )
    cam = GeecsDetector(
        "UC_Both",
        ["MeanCounts"],
        experiment="TestExp",
        name="uc_both",
        path_provider=provider,
        hdf_plugins=[("image", provider)],
    )
    connect_mock(RE, cam)
    set_mock_value(cam.acq_timestamp, box.stamp)
    set_mock_value(cam.hdf.file_path_exists, True)
    set_mock_value(cam.hdf.data_type, "UInt16")
    set_mock_value(cam.hdf.color_mode, "Mono")
    from ophyd_async.core import callback_on_mock_put

    callback_on_mock_put(
        cam.hdf.rewind, lambda value, **_: set_mock_value(cam.hdf.num_captured, value)
    )
    box.cameras.append(cam)
    saves: list[str] = []
    callback_on_mock_put(cam.save, lambda value, **_: saves.append(value))
    magnet = _magnet(RE)
    col = DocCollector()
    RE.subscribe(col)
    RE(
        bp.scan(
            [cam],
            magnet,
            -1.0,
            1.0,
            2,
            per_step=gated_per_step(shot_control, shots_per_step=1),
        )
    )
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert "on" not in saves  # stage/unstage clear a stale flag; nothing switches it on
    assert _datums_by_key(col)["uc_both"] == [
        {"start": 0, "stop": 1},
        {"start": 1, "stop": 2},
    ]
    assert not (tmp_path / "Scan001" / "UC_Both").exists()
    assert not any(
        k.endswith("-nonscalar_save_path") for k in _shots_descriptor(col)["data_keys"]
    )


# --------------------------------------------------------- bound plans / md
def test_bound_gated_count_records_its_description(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles, tmp_path: Path
) -> None:
    a, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles)["count"]
    RE(count([a], 2, acquisition="gated"))
    start = col.docs["start"][0]
    assert start["acquisition"] == "gated"
    assert start["shot_clock"] == "UC_A"
    assert start["non_essential"] == []
    assert start["trigger_profile"] == "HTU-Test"
    assert _datums_by_key(col)["uc_a"] == [{"start": 0, "stop": 2}]
    assert box.states[0] == "off"


def test_bound_plan_refuses_bad_mode_and_a_throttled_gated_run(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles, tmp_path: Path
) -> None:
    a, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    count = bind_plans(profiles)["count"]
    with pytest.raises(GeecsConfigurationError, match="both as a detector"):
        RE(count([a], 1, non_essential=[a]))
    with pytest.raises(GeecsConfigurationError, match="acquisition='sloppy'"):
        RE(count([a], 1, acquisition="sloppy"))
    with pytest.raises(GeecsConfigurationError, match="strict-mode throttle"):
        RE(count([a], 1, acquisition="gated", shot_period=2.0))
    with pytest.raises(GeecsConfigurationError, match="positive"):
        RE(count([a], 1, shot_period=0))
    # Codex review of #850: the stock repeat loop would sleep `delay` after
    # every no-op iteration of the batch hook — refused, like shot_period
    with pytest.raises(GeecsConfigurationError, match="delay=4.0 is a strict-mode"):
        RE(count([a], 5, 4.0, acquisition="gated"))
    with pytest.raises(GeecsConfigurationError, match="strict-mode spacing"):
        RE(count([a], 3, delay=[0.0, 2.0], acquisition="gated"))
    assert box.scan_runs == 0 and box.fires == 0
    # zero (the default, or an explicit zero list) is fine
    RE(count([a], 2, 0.0, acquisition="gated"))
    assert box.scan_runs == 1


def test_shot_period_throttles_strict_fires(
    RE: RunEngine,
    box: GatedBox,
    profiles: TriggerProfiles,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#840: one shot per period — the second and third fires wait.

    The period exceeds the camera's shot budget on purpose: the wait must
    happen before the detectors are triggered, or the triggered count/stamp
    wait times the shot out during the pause (found on hardware, A8).
    Pinned twice: a native camera's stamp wait uses its own 0.2 s budget,
    and a plugin camera's count wait uses ``STRICT_TRIGGER_INFO``'s
    ``exposure_timeout`` (patched down to 0.2 s here) — the reviewer found
    the plugin-only form passing on the pre-A8 code.
    """
    from geecs_bluesky.plans import strict as strict_module

    monkeypatch.setattr(
        strict_module,
        "STRICT_TRIGGER_INFO",
        strict_module.STRICT_TRIGGER_INFO.model_copy(update={"exposure_timeout": 0.2}),
    )
    native = _camera(RE, box, "UC_Native", shot_timeout=0.2)
    plugin, _ = _plugin_camera(RE, box, "UC_Cam", tmp_path, shot_timeout=0.2)
    count = bind_plans(profiles)["count"]
    for cam in (native, plugin):
        box.fires = 0
        t0 = time.monotonic()
        RE(count([cam], 3, shot_period=0.5))
        elapsed = time.monotonic() - t0
        assert box.fires == 3, cam.name
        assert elapsed >= 1.0, cam.name
    col = DocCollector()
    RE.subscribe(col)
    RE(count([plugin], 1, shot_period=0.25))
    assert col.docs["start"][0]["shot_period"] == 0.25


# --------------------------------------------------- non-essential stream
def test_non_essential_camera_streams_for_the_run(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles, tmp_path: Path, caplog
) -> None:
    """A strict count with B non-essential: B flies in ``uc_b_stream``, never waited on.

    B's plugin still reports a previous session's count at the arm
    (GEECS-Plugins#853, found on hardware in A4): zeroed before the kickoff
    baselines, so the close's count wait returns and the datum covers the
    run's frames from 0.
    """
    import logging

    a = _camera(RE, box, "UC_A")
    b, b_rewinds = _plugin_camera(RE, box, "UC_B", tmp_path)
    box.counts["uc_b"] = 6
    set_mock_value(b.hdf.num_captured, 6)
    col = DocCollector()
    RE.subscribe(col)
    caplog.set_level(logging.WARNING, logger="geecs_bluesky.plans.gated")
    count = bind_plans(profiles)["count"]
    t0 = time.monotonic()
    RE(count([a], 3, non_essential=[b]))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    # zeroed BEFORE the kickoff baselined: the close's complete returned at
    # once (a baseline of 6 would have timed it out, skipped, and still
    # collected — the same datum, a shot_timeout later, with a warning)
    assert time.monotonic() - t0 < 2.0
    assert not [
        r for r in caplog.records if "failed at the run's close" in r.getMessage()
    ]
    assert col.docs["start"][0]["non_essential"] == ["uc_b"]
    names = {d["name"] for d in col.docs["descriptor"]}
    assert names == {"primary", "uc_b_stream"}
    primary = next(d for d in col.docs["descriptor"] if d["name"] == "primary")
    assert not any(k.startswith("uc_b") for k in primary["data_keys"])
    assert len(_stream_events(col, "primary")) == 3
    # every fire advanced B's count too: one datum covering the run's frames
    assert _datums_by_key(col)["uc_b"] == [{"start": 0, "stop": 3}]
    assert b_rewinds[0] == 0  # the stale count zeroed at the arm
    assert box.states[0] == "single" and box.states[-1] == "edges"


def test_non_essential_camera_that_never_delivers_does_not_hold_the_run(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles, tmp_path: Path
) -> None:
    a = _camera(RE, box, "UC_A")
    b, _ = _plugin_camera(RE, box, "UC_B", tmp_path, shot_timeout=0.3)
    box.stall = {"uc_b"}
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles)["count"]
    t0 = time.monotonic()
    RE(count([a], 2, non_essential=[b]))
    assert time.monotonic() - t0 < 2.0
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert len(_stream_events(col, "primary")) == 2
    assert "uc_b" not in _datums_by_key(col)


def test_non_essential_with_a_gated_run(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles, tmp_path: Path
) -> None:
    """Both phase-2 mechanisms in one run: A gated, B streamed."""
    a, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    b, _ = _plugin_camera(RE, box, "UC_B", tmp_path)
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles)["count"]
    RE(count([a], 3, acquisition="gated", non_essential=[b]))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    datums = _datums_by_key(col)
    assert datums["uc_a"] == [{"start": 0, "stop": 3}]
    (b_datum,) = datums["uc_b"]
    assert b_datum["start"] == 0 and b_datum["stop"] >= 3  # every edge, the trim's too
    assert {d["name"] for d in col.docs["descriptor"]} == {
        "primary",
        "shots",
        "uc_b_stream",
    }


def test_a_non_essential_scope_with_no_streams_streams_by_stamp(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles, tmp_path: Path, caplog
) -> None:
    """A non-essential scope with every channel disabled gets its stamp stream.

    It used to sit the run out with a WARNING (Codex review of #948: the
    stock path aborted at kickoff, "not streamable").  Since the 2026-09-26
    ruling a triggered device without a plugin is recorded by stamp in its
    own ``u_ict_stream`` — here every edge stamps it, so the stream carries
    events and the run succeeds; nothing is prepared (no saving controls).
    """
    a = _camera(RE, box, "UC_A")
    ict = _all_off_scope(RE, tmp_path, native_save=False)
    box.cameras.append(ict)
    assert not ict.plugin_backed
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles)["count"]
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.plans.gated"):
        RE(count([a], 2, acquisition="gated", non_essential=[ict]))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert not [r for r in caplog.records if "u_ict" in r.getMessage().lower()]
    events = _events_from_pages(col, "u_ict_stream")
    assert len(events) >= 2
    assert all("u_ict-acq_timestamp" in e["data"] for e in events)
    assert not any(k.startswith("u_ict") for k in _shots_descriptor(col)["data_keys"])


def _all_off_scope(
    RE: RunEngine, tmp_path: Path, *, native_save: bool
) -> GeecsDetector:
    """A scope with every channel disabled.

    Since gating moved to the DB (``geecs_core.db.device_streams``), the
    namespace filters the disabled channels out before construction — so an
    all-off scope reaches the plan layer with **no** file plugins, exactly
    like a camera that never had one. There is nothing left to latch.  With
    *native_save* it carries the namespace's path provider, as a
    native-saving device does.
    """
    from ophyd_async.core import StaticFilenameProvider, StaticPathProvider

    provider = (
        StaticPathProvider(StaticFilenameProvider("f"), tmp_path / "Scan001" / "U_ICT")
        if native_save
        else None
    )
    ict = GeecsDetector(
        "U_ICT",
        ["MeanCounts"],
        experiment="TestExp",
        name="u_ict",
        native_save=native_save,
        path_provider=provider,
        hdf_plugins=[],
    )
    connect_mock(RE, ict)
    set_mock_value(ict.acq_timestamp, 1000.0)
    return ict


def test_a_native_saving_scope_with_every_channel_off_is_a_native_essential(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles, tmp_path: Path
) -> None:
    """Every channel disabled leaves no plugin: the scope saves natively, run-long.

    The 2026-09-25 ruling supersedes the refusal this test used to pin: a
    device without a file plugin is a gated essential exactly as strict
    treats it — the plugin count is a convenience, not what makes a batch —
    and an all-off scope with LabVIEW saving controls is one of them.  The
    plugin camera clocks the batch; the scope's save path rides in every
    ``shots`` row.
    """
    (tmp_path / "Scan001").mkdir()
    a, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    ict = _all_off_scope(RE, tmp_path, native_save=True)
    box.cameras.append(ict)
    assert ict.native_save and not ict.plugin_backed
    saves = _saves(ict)
    col = DocCollector()
    RE.subscribe(col)
    count = bind_plans(profiles)["count"]
    RE(count([a, ict], 2, acquisition="gated"))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert col.docs["start"][0]["shot_clock"] == "UC_A"
    assert saves == ["off", "on", "off"]
    rows = _events_from_pages(col, "shots")
    assert [r["data"]["u_ict-nonscalar_save_path"] for r in rows] == [
        str(tmp_path / "Scan001" / "U_ICT")
    ] * 2


def test_non_essential_wrapper_without_flyers_is_the_plan(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles
) -> None:
    cam = _camera(RE, box, "UC_Cam")
    count = bind_plans(profiles)["count"]
    col = DocCollector()
    RE.subscribe(col)
    RE(count([cam], 1, non_essential=None))
    assert col.docs["start"][0]["non_essential"] == []
    assert {d["name"] for d in col.docs["descriptor"]} == {"primary"}


def _all_docs(col: DocCollector) -> list[tuple[str, dict[str, Any]]]:
    return col.ordered


def test_scalars_view_essential_with_its_owner_non_essential_is_refused(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles, tmp_path: Path
) -> None:
    """Review of #850 R2: the same camera twice, by owner — its one fly flag would lie."""
    a, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    count = bind_plans(profiles)["count"]
    with pytest.raises(GeecsConfigurationError, match="scalars view"):
        RE(count([a.scalars], 2, non_essential=[a]))
    assert box.fires == 0


def test_the_bound_gated_plan_admits_a_native_essential_and_lets_it_clock(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles, tmp_path: Path
) -> None:
    """The refusal of #850 R1 is gone (2026-09-25 ruling): bind time admits it.

    Alone with a gauge the native saver is the shot clock (it has a stamp),
    named in the start document as any triggered device would be; beside a
    plugin camera that camera clocks and the native saver still saves.
    """
    (tmp_path / "Scan001").mkdir()
    native = _camera(RE, box, "UC_Native", tmp_path=tmp_path)
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, gauge)
    set_mock_value(gauge.pressure, 1e-6)
    count = bind_plans(profiles)["count"]
    col = DocCollector()
    token = RE.subscribe(col)
    RE(count([native, gauge], 4, acquisition="gated"))
    start = col.docs["start"][0]
    assert start["acquisition"] == "gated" and start["shot_clock"] == "UC_Native"
    assert start["shot_clock_column"] == "uc_native-acq_timestamp"
    assert col.docs["stop"][-1]["exit_status"] == "success"
    rows = _events_from_pages(col, "shots")
    assert len(rows) == 4
    assert [r["data"]["u_gauge-pressure"] for r in rows] == [1e-6] * 4
    directory = str(tmp_path / "Scan001" / "UC_Native")
    assert [r["data"]["uc_native-nonscalar_save_path"] for r in rows] == [directory] * 4
    assert box.states[0] == "off" and profiles.resolve(None).standing_state == "STANDBY"
    RE.unsubscribe(token)

    plugin, _ = _plugin_camera(RE, box, "UC_A", tmp_path)
    col = DocCollector()
    RE.subscribe(col)
    RE(count([native, plugin], 2, acquisition="gated"))
    assert col.docs["start"][0]["shot_clock"] == "UC_A"
    rows = _events_from_pages(col, "shots")
    assert [r["data"]["uc_native-nonscalar_save_path"] for r in rows] == [directory] * 2
    assert _datums_by_key(col)["uc_a"] == [{"start": 0, "stop": 2}]


def test_non_essential_that_fails_at_the_close_does_not_fail_the_run(
    RE: RunEngine, box: GatedBox, profiles: TriggerProfiles, tmp_path: Path, caplog
) -> None:
    """Review of #850 R4: a plugin whose complete/collect raises at close is logged and skipped."""
    import logging

    a = _camera(RE, box, "UC_A")
    b, _ = _plugin_camera(RE, box, "UC_B", tmp_path)

    original = b.complete

    def failing_complete():
        from ophyd_async.core import AsyncStatus

        async def boom():
            raise ConnectionError("hdf1 PVs unreachable")

        return AsyncStatus(boom())

    b.complete = failing_complete
    original_unstage = b.unstage

    def failing_unstage():
        from ophyd_async.core import AsyncStatus

        async def boom():
            raise ConnectionError("hdf1 PVs unreachable (Capture put)")

        return AsyncStatus(boom())

    b.unstage = failing_unstage  # review of #850 R5: the dead flyer's unstage too
    col = DocCollector()
    RE.subscribe(col)
    caplog.set_level(logging.WARNING, logger="geecs_bluesky.plans.gated")
    count = bind_plans(profiles)["count"]
    RE(count([a], 2, non_essential=[b]))  # returns: the item does not fail
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert len(_stream_events(col, "primary")) == 2
    messages = [r.getMessage() for r in caplog.records]
    assert any("non-essential uc_b: complete failed" in m for m in messages)
    assert not any("collect failed" in m for m in messages)
    # the failed complete never costs the datums: collect ran on its own
    assert _datums_by_key(col)["uc_b"] == [{"start": 0, "stop": 2}]
    assert any("non-essential uc_b: unstage failed" in m for m in messages)
    assert box.states[-1] == "edges"  # STANDBY still driven on the way out
    b.complete = original
    b.unstage = original_unstage
