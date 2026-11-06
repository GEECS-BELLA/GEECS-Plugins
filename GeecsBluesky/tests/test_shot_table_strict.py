"""A strict count writes the shot table: ``primary`` per fire, every device in its own stream.

Driven through the bound ``count`` on mocks.  The box stamps every device
with the worker's clock (LabVIEW epoch) at the fire, so the readers —
the s-file and the stack check, fed the run's own documents — join the
streams onto the rows by fire interval, as they do a real run's.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time

import pandas as pd
import pytest

pytest.importorskip("aioca")

from bluesky import RunEngine
from bluesky.utils import RunEngineInterrupted
from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET as LV
from ophyd_async.core import set_mock_value

from geecs_bluesky.devices.ca.snapshot import CaSnapshotReadable
from geecs_bluesky.devices.shot_control import ShotControl
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_callbacks import _stack_uri, _write_plugin_stack
from tests.test_strict_plans import WRITES, FakeBox, _camera, _plugin_camera

SHOT_KEYS = {
    "shot_index",
    "attempt",
    "fire_stamp",
    "complete",
    "missing",
    "unwitnessed",
    "witness_late",
    "interrupted",
    "bin_number",
}


class ClockBox(FakeBox):
    """The fire stamps every camera with the worker's clock, 10 ms after the put.

    ``quiet`` names cameras that never deliver; ``on_fire(n)`` runs at fire
    *n* and returns ``False`` to hold that fire's frames back.
    """

    def __init__(self) -> None:
        super().__init__()
        self.quiet: set[str] = set()
        self.on_fire = lambda n: True
        self.stamps: list[float] = []

    def __call__(self, device: str, variable: str):
        box = self

        class Setter:
            async def put(self, value: str) -> None:
                box.puts.append((device, variable, value))
                if variable != "Trigger.ExecuteSingleShot":
                    return
                await asyncio.sleep(0.01)
                box.fires += 1
                if box.on_fire(box.fires):
                    box.deliver()

        return Setter()

    def deliver(self) -> None:
        self.stamp = time.time() + LV
        self.stamps.append(self.stamp)
        for cam in self.cameras:
            if cam.name in self.quiet:
                continue
            hdf = getattr(cam, "hdf", None)
            if hdf is not None:
                self.counts[cam.name] = self.counts.get(cam.name, 0) + 1
                set_mock_value(hdf.num_captured, self.counts[cam.name])
            set_mock_value(cam.acq_timestamp, self.stamp)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def box() -> ClockBox:
    return ClockBox()


@pytest.fixture
def count(RE: RunEngine, box: ClockBox):
    sc = ShotControl(WRITES, experiment="TestExp", name="box", setter_factory=box)
    connect_mock(RE, sc)
    return bind_plans(TriggerProfiles({"P": sc}, default="P"))["count"]


def _gauge(RE: RunEngine) -> CaSnapshotReadable:
    gauge = CaSnapshotReadable(
        "U_Gauge", ["Pressure"], experiment="TestExp", name="u_gauge"
    )
    connect_mock(RE, gauge)
    set_mock_value(gauge.pressure, 2.5)
    return gauge


def test_the_shot_table_and_its_streams_read_through_the_w3b_readers(
    RE, box, count, tmp_path, caplog
) -> None:
    """Start keys, a device-free ``primary``, one stream per device; the s-file and stack check join them."""
    from geecs_bluesky.callbacks import SFileCallback, StackCheckCallback

    caplog.set_level(logging.INFO, logger="geecs_bluesky.callbacks")
    folder = tmp_path / "scans" / "Scan001"
    folder.mkdir(parents=True)
    cam, _ = _plugin_camera(RE, box, "UC_Cam", tmp_path)
    ps = _camera(RE, box, "U_PS")
    set_mock_value(ps.meancounts, 7.0)
    col = DocCollector()
    RE.subscribe(col)
    headers = {
        "u_ps-meancounts": "U_PS MeanCounts",
        "uc_cam-meancounts": "UC_Cam MeanCounts",
    }
    md = {"scan_folder": str(folder), "scan_number": 1, "geecs_scalar_headers": headers}
    RE(count([cam, ps, _gauge(RE)], 3, md=md, shot_period=0.15))
    start = col.docs["start"][0]
    assert start["layout"] == "shot-table/1" and "non_essential" not in start
    assert start["per_shot_streams"] == ["uc_cam_stream", "u_ps_stream"]
    assert start["required"] == ["uc_cam", "u_ps"]
    assert start["shot_clock_column"] == "fire_stamp"
    assert start["stream_meta"]["u_ps_stream"]["sync_class"] == "synchronous"
    rows = [e["data"] for e in col.primary_events()]
    assert [r["shot_index"] for r in rows] == [1, 2, 3]
    assert all(r["complete"] and r["unwitnessed"] for r in rows)
    assert set(rows[0]) == SHOT_KEYS | {
        "uc_cam-baseline",
        "u_ps-baseline",
        "u_gauge-pressure",
    }
    assert [r["uc_cam-baseline"] for r in rows] == [0, 1, 2]
    assert len(col.stream_events("u_ps_stream")) == 3
    assert sum(d["stop"] - d["start"] for d in col.datums("uc_cam")) == 3
    # The run's stack, as the plugin would have finalized it, then the readers.
    _write_plugin_stack(
        _stack_uri(col, "uc_cam"),
        box.stamps,
        scalars={"meancounts": [1.0, 2.0, 3.0]},
        device="uc_cam",
    )
    for cb in (
        SFileCallback(finalize_timeout=2.0),
        StackCheckCallback(finalize_timeout=2.0),
    ):
        for name, doc in col.ordered:
            cb(name, doc)
        cb.join(10.0)
    sfile = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    assert list(sfile["Shotnumber"]) == [1, 2, 3]
    assert list(sfile["U_PS MeanCounts"]) == [7.0] * 3
    assert list(sfile["UC_Cam MeanCounts"]) == [1.0, 2.0, 3.0]
    checks = [
        m
        for m in map(logging.LogRecord.getMessage, caplog.records)
        if "assigned to a shot-table row" in m
    ]
    assert checks == [
        "scan 1: uc_cam: 3 of 3 frame(s) in UC_Cam.h5 assigned to a shot-table row — "
        "0 extra(s), 0 row(s) flagged, 0 row(s) with no frame"
    ]


def _pause_from_a_thread(RE: RunEngine) -> None:
    threading.Thread(target=RE.request_pause, kwargs={"defer": False}).start()


def test_a_pause_before_the_verdict_counts_the_frame_that_landed(
    RE, box, count
) -> None:
    """The pause cuts the wait; the frame lands while paused; the re-read counts it: one complete row."""
    cam = _camera(RE, box, "UC_A")

    def hold_first(n: int) -> bool:
        if n == 1:
            _pause_from_a_thread(RE)
            return False
        return True

    box.on_fire = hold_first
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(RunEngineInterrupted):
        RE(count([cam], 2))
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), RE._loop).result()
    RE._loop.call_soon_threadsafe(box.deliver)  # the held frame lands during the pause
    time.sleep(0.05)
    RE.resume()
    rows = [e["data"] for e in col.primary_events()]
    assert box.fires == 2 and [r["complete"] for r in rows] == [True, True]
    assert [r["interrupted"] for r in rows] == [True, False]
    assert col.docs["stop"][-1]["exit_status"] == "success"


class _PausingReader:
    """A condition whose first read pauses the run and outlasts the pause."""

    name, parent = "u_cond", None

    def __init__(self, RE: RunEngine) -> None:
        self.RE, self.reads = RE, 0

    async def read(self):
        self.reads += 1
        if self.reads == 1:
            _pause_from_a_thread(self.RE)
            await asyncio.sleep(1.0)
        return {"u_cond-x": {"value": 1.0, "timestamp": time.time()}}

    async def describe(self):
        return {"u_cond-x": {"source": "test", "dtype": "number", "shape": []}}


def test_a_pause_inside_the_bundle_saves_no_short_row(RE, box, count) -> None:
    """The interrupted read leaves a short bundle: dropped and read again, never saved, no run failure."""
    cam = _camera(RE, box, "UC_A")
    cond = _PausingReader(RE)
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(RunEngineInterrupted):
        RE(count([cam, cond], 2))
    RE.resume()
    rows = [e["data"] for e in col.primary_events()]
    assert box.fires == 2 and len(rows) == 2
    assert all(r["u_cond-x"] == 1.0 for r in rows) and rows[0]["interrupted"]
    assert col.docs["stop"][-1]["exit_status"] == "success"


def test_a_quiet_required_device_refires_then_flags_and_the_scan_continues(
    RE, box, count, caplog
) -> None:
    a = _camera(RE, box, "UC_A", shot_timeout=0.15)
    b = _camera(RE, box, "UC_B", shot_timeout=0.15)
    box.quiet = {"uc_a"}
    col = DocCollector()
    RE.subscribe(col)
    RE(count([a, b], 1))
    rows = [e["data"] for e in col.primary_events()]
    assert box.fires == 3 and [r["attempt"] for r in rows] == [1, 2, 3]
    assert not any(r["complete"] for r in rows) and {r["missing"] for r in rows} == {
        "uc_a"
    }
    assert (
        len(col.stream_events("uc_b_stream")) == 3
    )  # every fire's record of the other device
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert any(
        "last row is flagged incomplete" in r.getMessage() for r in caplog.records
    )


def test_two_exhausted_shots_in_a_row_pause_the_run(RE, box, count) -> None:
    a = _camera(RE, box, "UC_A", shot_timeout=0.1)
    box.quiet = {"uc_a"}
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(RunEngineInterrupted):
        RE(count([a], 3))
    assert box.fires == 6
    box.quiet = set()
    RE.resume()
    assert box.fires == 7 and col.primary_events()[-1]["data"]["complete"]


def test_a_device_down_on_the_last_attempt_pauses_and_the_resume_still_fires(
    RE, box, count
) -> None:
    a = _camera(RE, box, "UC_A", shot_timeout=0.1)
    box.quiet = {"uc_a"}

    def down_at_the_third(n: int) -> bool:
        if n == 3:
            set_mock_value(a.connected_status, "Disconnected")
        return True

    box.on_fire = down_at_the_third
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(RunEngineInterrupted):
        RE(count([a], 1))
    assert box.fires == 3
    set_mock_value(a.connected_status, "")
    box.quiet = set()
    RE.resume()
    rows = [e["data"] for e in col.primary_events()]
    assert box.fires == 4 and [r["complete"] for r in rows] == [False] * 3 + [True]


def test_a_refused_press_is_logged_once_and_never_fails_the_run(
    RE, box, count, tmp_path, caplog
) -> None:
    from ophyd_async.core import get_mock_execute

    from tests.test_native_camera import _camera as _native

    cam = _camera(RE, box, "UC_A")
    native, _ = _native(RE, tmp_path)

    def refuse(*_, **__):
        raise RuntimeError("TriggerSoftware refused")

    get_mock_execute(native.driver.trigger_software).side_effect = refuse
    col = DocCollector()
    RE.subscribe(col)
    RE(count([cam, native], 3, required=[cam]))
    assert col.docs["stop"][-1]["exit_status"] == "success"
    said = [
        r.getMessage() for r in caplog.records if "the press failed" in r.getMessage()
    ]
    assert len(said) == 1 and "ADTEST_CAM1" in said[0] and "refused" in said[0]


@pytest.mark.parametrize("required", [lambda a, b: [a], lambda a, b: []])
def test_a_non_required_device_going_quiet_never_refires(
    RE, box, count, required
) -> None:
    a = _camera(RE, box, "UC_A", shot_timeout=0.15)
    b = _camera(RE, box, "UC_B", shot_timeout=0.15)
    box.quiet = {"uc_b"}
    col = DocCollector()
    RE.subscribe(col)
    RE(count([a, b], 3, required=required(a, b)))
    rows = [e["data"] for e in col.primary_events()]
    assert box.fires == 3 and all(r["complete"] for r in rows)
    assert col.docs["start"][0]["required"] == [d.name for d in required(a, b)]
    assert col.stream_events("uc_b_stream") == []


def test_required_none_is_every_per_shot_device_and_a_condition_cannot_be_required(
    RE, box, count
) -> None:
    from geecs_bluesky.exceptions import GeecsConfigurationError

    a = _camera(RE, box, "UC_A")
    gauge = _gauge(RE)
    col = DocCollector()
    RE.subscribe(col)
    RE(count([a, gauge], 1))
    assert col.docs["start"][0]["required"] == ["uc_a"]
    with pytest.raises(GeecsConfigurationError, match="no shot stamp"):
        RE(count([a, gauge], 1, required=[gauge]))


def test_a_native_camera_skipping_a_frame_leaves_a_gap_in_its_stream(
    RE, box, count, tmp_path
) -> None:
    """A Software-source native, not required, beside a required gateway device: pressed per fire, no refire."""
    from ophyd_async.core import get_mock_execute

    from tests.test_native_camera import _camera as _native

    cam = _camera(RE, box, "UC_A")
    native, _ = _native(RE, tmp_path)
    presses: list[int] = []

    def press(*_, **__):
        presses.append(1)
        set_mock_value(native.hdf.num_captured, len(presses) - (len(presses) >= 2))

    get_mock_execute(native.driver.trigger_software).side_effect = press
    col = DocCollector()
    RE.subscribe(col)
    RE(count([cam, native], 3, required=[cam]))
    assert box.fires == 3 and len(presses) == 3
    assert "adtest_cam1_stream" in col.docs["start"][0]["per_shot_streams"]
    assert sum(d["stop"] - d["start"] for d in col.datums("adtest_cam1")) == 2
    assert all(e["data"]["complete"] for e in col.primary_events())
    assert not math.isnan(col.primary_events()[0]["data"]["fire_stamp"])


def test_a_down_pause_costs_the_shot_no_refire(RE, box) -> None:
    """The fire that paused for a down device is not one of the shot's paid attempts."""
    from tests.test_strict_plans import strict_count

    sc = ShotControl(WRITES, experiment="TestExp", name="box", setter_factory=box)
    connect_mock(RE, sc)
    a = _camera(RE, box, "UC_A", shot_timeout=0.1)
    box.quiet = {"uc_a"}

    def down_at_the_first(n: int) -> bool:
        if n == 1:
            set_mock_value(a.connected_status, "Disconnected")
        return True

    box.on_fire = down_at_the_first
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(RunEngineInterrupted):
        RE(strict_count(sc, [a], 1, max_refires=1))
    assert box.fires == 1
    set_mock_value(a.connected_status, "")
    RE.resume()  # still quiet: both paid attempts (fires 2 and 3) miss
    rows = [e["data"] for e in col.primary_events()]
    assert box.fires == 3 and [r["attempt"] for r in rows] == [1, 2, 3]
    assert col.docs["stop"][-1]["exit_status"] == "success"


@pytest.mark.parametrize("which", [1, 2], ids=["pre-fire", "post-wait"])
def test_an_immediate_pause_inside_a_counter_snapshot_never_ends_the_run(
    RE, box, count, monkeypatch, which
) -> None:
    """The snapshot cut by the pause is taken again before use: the run completes.

    ``uc_b`` is not required, so its counter is read only in the two
    snapshots: its first read is the pre-fire one, its second the post-wait.
    """
    from geecs_bluesky.plans import on_demand

    a = _camera(RE, box, "UC_A")
    b = _camera(RE, box, "UC_B")
    reads = {"n": 0}
    plain = on_demand._Progress.read

    async def read(self):
        if self.name == "uc_b":
            reads["n"] += 1
            if reads["n"] == which:
                _pause_from_a_thread(RE)
                await asyncio.sleep(0.5)
        return await plain(self)

    monkeypatch.setattr(on_demand._Progress, "read", read)
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(RunEngineInterrupted):
        RE(count([a, b], 2, required=[a]))
    RE.resume()
    rows = [e["data"] for e in col.primary_events()]
    assert col.docs["stop"][-1]["exit_status"] == "success"
    assert box.fires == 2 and all(r["complete"] for r in rows)
    assert rows[0]["interrupted"]
