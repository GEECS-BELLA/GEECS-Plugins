"""Fly mode: the soft tick is the shot clock, and the box is never touched.

Driven through the bound ``count`` on mocks.  A free-running gateway device
gets a fresh stamp every 20 ms from a thread (its own clock, no box); a
Software-source native camera writes a frame per press.
"""

from __future__ import annotations

import logging
import threading
import time

import bluesky.plans as bp
import pandas as pd
import pytest

pytest.importorskip("aioca")

from bluesky import RunEngine
from bluesky.utils import RunEngineInterrupted
from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET as LV
from ophyd_async.core import get_mock_execute, set_mock_value, soft_signal_rw

from geecs_bluesky.devices.shot_control import ShotControl, ShotControlWrites
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.plans.fly import BOX_NOT_DRIVEN, bind_fly
from geecs_bluesky.plans.registry import TriggerProfiles, bind_plans
from geecs_bluesky.plans.streams import DeviceStreams
from geecs_bluesky.qs_client.submit_preflight import scan_mode_refusal
from tests.ca_mock_helpers import DocCollector, connect_mock
from tests.test_native_camera import _camera as _native
from tests.test_shot_table_strict import _gauge
from tests.test_strict_plans import WRITES, FakeBox, _camera

PERIOD = 0.15


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


@pytest.fixture
def box() -> FakeBox:
    return FakeBox()


def _count(RE: RunEngine, box: FakeBox, **writes):
    sc = ShotControl(
        ShotControlWrites(name="P", states=WRITES.states, **writes),
        experiment="TestExp",
        name="box",
        setter_factory=box,
    )
    connect_mock(RE, sc)
    return bind_plans(TriggerProfiles({"P": sc}, default="P"))["count"]


class FreeRunner:
    """Stamps *device* with the worker's clock (LabVIEW epoch) every *every* s while running."""

    def __init__(self, RE: RunEngine, device, every: float = 0.02) -> None:
        self.RE, self.device, self.every = RE, device, every
        self.stamps: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            stamp = time.time() + LV
            self.stamps.append(stamp)
            self.RE.loop.call_soon_threadsafe(
                set_mock_value, self.device.acq_timestamp, stamp
            )

    def __enter__(self) -> FreeRunner:
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()


def _pressed(RE: RunEngine, tmp_path, *, skip: int = 0, on_press=None):
    """A Software-source native; press *skip* writes no frame; *on_press(n)* runs at each press."""
    cam, _ = _native(RE, tmp_path)
    presses: list[int] = []

    def press(*_, **__):
        presses.append(1)
        n = len(presses)
        if on_press is not None:
            on_press(n)
        set_mock_value(cam.hdf.num_captured, n - (skip and n >= skip))

    get_mock_execute(cam.driver.trigger_software).side_effect = press
    return cam, presses


def test_a_fly_count_writes_soft_tick_rows_and_never_touches_the_box(
    RE, box, tmp_path, caplog
) -> None:
    """Start keys, one row per tick on an absolute schedule, the run uid as epoch, no put on the box."""
    caplog.set_level(logging.INFO)
    count = _count(RE, box)
    free = _camera(RE, box, "U_PS")
    native, presses = _pressed(RE, tmp_path)
    col = DocCollector()
    RE.subscribe(col)
    with FreeRunner(RE, free):
        RE(count([free, native, _gauge(RE)], 4, acquisition="fly", shot_period=PERIOD))
    start = col.docs["start"][0]
    assert start["clock"] == "soft-tick" and start["acquisition"] == "fly"
    assert start["shot_clock_column"] == "witness_stamp"
    assert start["shot_period"] == PERIOD and start["trigger_profile"] is None
    meta = start["stream_meta"]
    assert meta["adtest_cam1_stream"]["sync_class"] == "synchronous"
    assert meta["u_ps_stream"]["sync_class"] == "asynchronous"
    assert start["required"] == ["u_ps", "adtest_cam1"]
    rows = [e["data"] for e in col.primary_events()]
    assert [r["witness_count"] for r in rows] == [1, 2, 3, 4]
    assert {r["witness_epoch"] for r in rows} == {start["uid"]}
    stamps = [r["witness_stamp"] for r in rows]
    assert all(abs(s - stamps[0] - k * PERIOD) < 0.03 for k, s in enumerate(stamps)), (
        stamps
    )
    assert all(r["complete"] for r in rows), rows
    assert not {"fire_stamp", "unwitnessed", "witness_late"} & set(rows[0])
    assert box.puts == [] and box.fires == 0 and len(presses) == 4
    assert any(BOX_NOT_DRIVEN in m for m in caplog.messages)
    assert col.docs["stop"][-1]["exit_status"] == "success"


def test_a_fly_run_reads_through_the_s_file(RE, box, tmp_path) -> None:
    """The s-file: one row per tick; the free-running device's last record before the tick."""
    from geecs_bluesky.callbacks import SFileCallback

    count = _count(RE, box)
    folder = tmp_path / "scans" / "Scan001"
    folder.mkdir(parents=True)
    free = _camera(RE, box, "U_PS")
    set_mock_value(free.meancounts, 7.0)
    col = DocCollector()
    RE.subscribe(col)
    md = {
        "scan_folder": str(folder),
        "scan_number": 1,
        "geecs_scalar_headers": {
            "u_ps-meancounts": "U_PS MeanCounts",
            "u_ps-acq_timestamp": "U_PS acq_timestamp",
        },
    }
    with FreeRunner(RE, free):
        RE(count([free, _gauge(RE)], 3, acquisition="fly", shot_period=PERIOD, md=md))
    cb = SFileCallback(finalize_timeout=2.0)
    for name, doc in col.ordered:
        cb(name, doc)
    cb.join(10.0)
    sfile = pd.read_csv(tmp_path / "analysis" / "s1.txt", sep="\t")
    assert list(sfile["Shotnumber"]) == [1, 2, 3]
    assert list(sfile["U_PS MeanCounts"]) == [7.0] * 3
    ticks = [e["data"]["witness_stamp"] + LV for e in col.primary_events()]
    joined = list(sfile["U_PS acq_timestamp"])
    assert all(0 <= t - s < 0.05 for t, s in zip(ticks, joined, strict=True))


def test_a_required_device_on_a_hardware_line_holds_flags_then_pauses_naming_it(
    RE, box, caplog
) -> None:
    """The box idle, a line-triggered device never records: held ticks, a flag, then the pause says why."""
    count = _count(RE, box)
    line = _camera(RE, box, "UC_Line")
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(RunEngineInterrupted):
        RE(count([line], 2, acquisition="fly", shot_period=0.05))
    rows = [e["data"] for e in col.primary_events()]
    assert [r["attempt"] for r in rows] == [1, 2, 3, 1, 2, 3]
    assert not any(r["complete"] for r in rows)
    assert {r["missing"] for r in rows} == {"uc_line"}
    reason = next(m for m in caplog.messages if m.startswith("pausing the run"))
    assert "UC_Line" in reason and "the box is not driven in fly mode" in reason
    assert box.puts == []
    RE.stop()


def test_a_skipped_press_holds_one_more_tick(RE, box, tmp_path) -> None:
    """A pressed native missing one frame: that row incomplete, the next tick completes the step."""
    count = _count(RE, box)
    native, presses = _pressed(RE, tmp_path, skip=2)
    col = DocCollector()
    RE.subscribe(col)
    RE(count([native], 3, acquisition="fly", shot_period=0.05))
    rows = [e["data"] for e in col.primary_events()]
    assert [r["attempt"] for r in rows] == [1, 1, 2, 1]
    assert [r["complete"] for r in rows] == [True, False, True, True]
    assert len(presses) == 4


def test_a_pause_inside_a_tick_flags_the_row_and_the_ticks_resume_without_a_burst(
    RE, box, tmp_path
) -> None:
    """An immediate pause at tick 2: its row ``interrupted``; after the resume ticks keep the period."""
    count = _count(RE, box)

    def pause_at_two(n: int) -> None:
        if n == 2:
            threading.Thread(target=RE.request_pause, kwargs={"defer": False}).start()

    native, _ = _pressed(RE, tmp_path, on_press=pause_at_two)
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(RunEngineInterrupted):
        RE(count([native], 6, acquisition="fly", shot_period=PERIOD))
    time.sleep(4 * PERIOD)
    RE.resume()
    rows = [e["data"] for e in col.primary_events()]
    assert [r["witness_count"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert [r["interrupted"] for r in rows] == [False, True] + [False] * 4
    gaps = [b["witness_stamp"] - a["witness_stamp"] for a, b in zip(rows, rows[1:])]
    assert min(gaps) > 0.5 * PERIOD, gaps
    assert col.docs["stop"][-1]["exit_status"] == "success"


def test_a_fly_step_holds_shots_per_step_ticks_with_the_motor_in_each_row(
    RE, box, tmp_path
) -> None:
    """The per-step hook: two ticks per position, the motor read in every row."""
    native, _ = _pressed(RE, tmp_path)
    motor = soft_signal_rw(float, name="m")
    streams = DeviceStreams([native], dual_write=True)
    clock, keys, hook = bind_fly(
        streams, [native], hook="per_step", shots_per_step=2, period=0.05
    )
    col = DocCollector()
    RE.subscribe(col)
    RE(
        clock.wrap(
            streams.wrap(bp.scan([clock], motor, 0, 1, 2, per_step=hook, md=keys))
        )
    )
    rows = [e["data"] for e in col.primary_events()]
    assert [r["m"] for r in rows] == [0.0, 0.0, 1.0, 1.0]
    assert [r["bin_number"] for r in rows] == [1, 1, 2, 2]
    assert [r["witness_count"] for r in rows] == [1, 2, 3, 4]


def test_a_scan_mode_profile_is_refused_before_anything_moves(RE, box) -> None:
    """Worker and client: scan mode needs a box and an announce, which fly has neither of."""
    count = _count(RE, box, use_scan_mode=True)
    cam = _camera(RE, box, "UC_A")
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(GeecsConfigurationError, match="scan mode"):
        RE(count([cam], 1, acquisition="fly"))
    assert box.puts == [] and col.docs["start"] == []

    class Item:
        kwargs = {"acquisition": "fly"}

    class Profile:
        name, use_scan_mode = "P", True

    assert "a fly run" in (scan_mode_refusal(Item(), Profile()) or "")


def test_a_fly_count_refuses_a_delay(RE, box) -> None:
    count = _count(RE, box)
    cam = _camera(RE, box, "UC_A")
    with pytest.raises(GeecsConfigurationError, match="tick"):
        RE(count([cam], 2, delay=0.5, acquisition="fly"))
    assert box.puts == []
