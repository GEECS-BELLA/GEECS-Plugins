"""The shot table: records assigned to shots over both run layouts, and the audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
from geecs_data_utils.shot_join import (
    FrameColumns,
    join_frames_to_shots,
    row_windows,
    clock_stamps,
    shot_clock_column,
    to_labview_epoch,
)
from geecs_data_utils.shot_table import (
    ASYNCHRONOUS,
    DeviceRecords,
    frame_for,
    shot_table,
)
from geecs_data_utils.tiled_export import join_frame_columns

T0 = 1_780_000_000.0  # Unix seconds
LV = float(LABVIEW_EPOCH_OFFSET)
START = {
    "layout": "shot-table/1",
    "per_shot_streams": ["cam_stream"],
    "required": ["cam"],
    "shot_clock_column": "witness_stamp",
    "shot_period": 1.0,
}


def _rows(n: int = 8, *, per_shot: int = 1, lead: float = 0.3) -> pd.DataFrame:
    """A witnessed shot table: one count per fire (*per_shot* for a doubled edge)."""
    stamps = T0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "shot_index": np.arange(1, n + 1),
            "witness_epoch": ["e1"] * n,
            "witness_count": 100.0 + per_shot * np.arange(1, n + 1),
            "witness_stamp": stamps,
            "fire_stamp": stamps - lead,
            "complete": [True] * n,
        }
    )


def _cam(
    rows: pd.DataFrame,
    *,
    drop: tuple[int, ...] = (),
    offset: float = 0.05,
    counter_on_drop: bool = False,
    attribute: bool = True,
) -> DeviceRecords:
    """A camera with one frame per row, LabVIEW stamps; *drop* removes rows' frames."""
    stamps, counter, attr, value, number = [], [], [], [], 0
    for k, stamp in enumerate(rows["witness_stamp"]):
        if k in drop:
            number += counter_on_drop
            continue
        number += 1
        stamps.append(stamp + LV + offset)
        counter.append(number)
        attr.append(rows["witness_count"][k])
        value.append(10.0 * k)
    return _device("cam", stamps, counter, attr if attribute else None, value)


def _device(name, stamps, counter=None, attr=None, value=None, **kwargs):
    stamps = np.asarray(stamps, dtype=float)
    columns = {f"{name}-acq_timestamp": stamps}
    if value is not None:
        columns[f"{name}-meancounts"] = np.asarray(value, dtype=float)
    return DeviceRecords(
        stream=f"{name}_stream",
        records=FrameColumns(name, stamps, columns),
        counter=None if counter is None else np.asarray(counter, dtype=float),
        witness_count=None if attr is None else np.asarray(attr, dtype=float),
        **kwargs,
    )


def _flagged(table, flag: str) -> list[int]:
    """The 1-based shots carrying *flag*."""
    return [k + 1 for k, flags in enumerate(table.flags) if flag in flags]


# ------------------------------------------------------------ shot_join growth
def test_a_start_document_may_name_a_non_device_clock() -> None:
    columns = ["witness_stamp", "fire_stamp", "uc_cam-acq_timestamp"]
    assert (
        shot_clock_column({"shot_clock_column": "witness_stamp"}, columns)
        == "witness_stamp"
    )
    assert (
        shot_clock_column({"shot_clock_column": "fire_stamp"}, columns[1:2])
        == "fire_stamp"
    )
    # an unnamed non-device column is never a clock; a device stamp still wins
    assert shot_clock_column({}, columns) == "uc_cam-acq_timestamp"
    assert shot_clock_column({"shot_clock_column": "fire_stamp"}, ["x"]) is None


def test_the_one_epoch_conversion_shifts_only_unix_columns() -> None:
    assert to_labview_epoch("witness_stamp", [1.0])[0] == 1.0 + LV
    assert to_labview_epoch("fire_stamp", [1.0])[0] == 1.0 + LV
    assert to_labview_epoch("uc_cam-acq_timestamp", [1.0])[0] == 1.0
    assert to_labview_epoch(None, [1.0])[0] == 1.0


# ------------------------------------------------------------ assignment
def test_a_clean_run_assigns_frame_k_to_shot_k_unflagged() -> None:
    rows = _rows()
    table = shot_table(START, rows, [_cam(rows)])
    assert table.record_for_shot["cam"] == tuple(range(8))
    assert all(flags == () for flags in table.flags)
    assert frame_for(table, "cam", 3) == ("cam_stream", 2)
    assert frame_for(table, "cam", 0) is None and frame_for(table, "cam", 9) is None
    assert list(table.rows["cam-meancounts"]) == [10.0 * k for k in range(8)]


@pytest.mark.parametrize(
    "counter_on_drop", [False, True], ids=["missed-edge", "lost-record"]
)
def test_a_miss_is_a_gap_and_shifts_nothing(counter_on_drop: bool) -> None:
    """A missed edge or a dropped buffer: a gap in time, later frames stay put."""
    rows = _rows()
    table = shot_table(
        START, rows, [_cam(rows, drop=(2,), counter_on_drop=counter_on_drop)]
    )
    assert table.record_for_shot["cam"] == (0, 1, None, 2, 3, 4, 5, 6)
    assert _flagged(table, "missing:cam") == [3]
    assert np.isnan(table.rows["cam-meancounts"][2])
    assert _flagged(table, "audit-mismatch:cam") == []
    assert _flagged(table, "complete-mismatch") == [3]  # the plan said complete


def test_an_extra_record_is_kept_out_and_its_counter_flags_both_ends() -> None:
    rows = _rows()
    cam = _cam(rows)
    stamps = np.insert(cam.records.stamps, 2, rows["witness_stamp"][1] + LV + 0.4)
    counter = np.arange(1.0, 10.0)
    extra = _device("cam", stamps, counter, np.insert(cam.witness_count, 2, 102.0))
    table = shot_table(START, rows, [extra])
    assert table.record_for_shot["cam"] == (0, 1, 3, 4, 5, 6, 7, 8)
    assert table.extras["cam"] == (2,)
    assert _flagged(table, "audit-mismatch:cam") == [2, 3]


def test_a_doubled_edge_flags_every_acquisition() -> None:
    rows = _rows(per_shot=2)
    cam = _cam(rows)
    table = shot_table(
        START, rows, [DeviceRecords("cam_stream", cam.records, cam.counter)]
    )
    assert table.record_for_shot["cam"] == tuple(range(8))
    assert _flagged(table, "audit-mismatch:cam") == list(range(1, 9))


def test_a_counter_reset_re_anchors_the_audit() -> None:
    rows = _rows()
    cam = _cam(rows)
    counter = np.array([5, 6, 7, 0, 1, 2, 3, 4], dtype=float)
    table = shot_table(START, rows, [DeviceRecords("cam_stream", cam.records, counter)])
    assert _flagged(table, "counter-reset:cam") == [4]
    assert _flagged(table, "audit-mismatch:cam") == []


def test_a_clock_step_inside_the_window_moves_nothing() -> None:
    rows = _rows()
    cam = _cam(rows)
    stamps = cam.records.stamps + np.where(np.arange(8) >= 4, 0.3, 0.0)
    table = shot_table(
        START, rows, [_device("cam", stamps, cam.counter, cam.witness_count)]
    )
    assert table.record_for_shot["cam"] == tuple(range(8))


def test_a_clock_step_is_flagged_where_it_appears() -> None:
    """A jump in residual is *clock-step*; the attribute verifies the rows after it."""
    rows = _rows()
    cam = _cam(rows)
    stamps = cam.records.stamps + np.where(np.arange(8) >= 4, 0.3, 0.0)
    table = shot_table(
        START, rows, [_device("cam", stamps, cam.counter, cam.witness_count)]
    )
    assert _flagged(table, "clock-step:cam") == [5]
    assert [f for row in table.flags for f in row] == ["clock-step:cam"]


def _stepped(
    attribute: bool = True, step: float = 0.8
) -> tuple[pd.DataFrame, DeviceRecords]:
    """100 shots at 1 Hz; the camera's stamps jump *step* s from frame 91 on."""
    rows = _rows(100)
    cam = _cam(rows)
    stamps = cam.records.stamps + np.where(np.arange(100) >= 90, step, 0.0)
    attr = cam.witness_count if attribute else None
    return rows, _device("cam", stamps, cam.counter, attr, np.arange(100.0))


@pytest.mark.parametrize("attribute", [True, False], ids=["attribute", "counter-only"])
def test_a_clock_step_that_moves_records_flags_every_moved_row(attribute) -> None:
    """The step lands frames 91.. on shots 92..: one gap, then nine wrong rows.

    The counters advance once over the gap, as for a missed edge, so only
    the witness attribute (one behind on every moved row) or, without it,
    the displaced residual can say the rows after the step are suspect.
    """
    rows, cam = _stepped(attribute)
    table = shot_table(START, rows, [cam])
    assert table.record_for_shot["cam"][89:] == (89, None) + tuple(range(90, 99))
    assert _flagged(table, "missing:cam") == [91]
    # With the attribute the stretch's edge row before the gap is flagged
    # too; without it the step's own row opens the stretch.
    edge = [90] if attribute else []
    assert _flagged(table, "clock-step:cam") == [*edge, *range(92, 101)]
    assert table.device_flags == {}


@pytest.mark.parametrize("step", [1.0, 0.95, 1.05])
def test_a_whole_period_step_is_flagged_by_the_attribute(step) -> None:
    """The residual does not move; the attribute lags one more on every moved row."""
    rows, cam = _stepped(step=step)
    table = shot_table(START, rows, [cam])
    assert table.record_for_shot["cam"][90:] == (None,) + tuple(range(90, 99))
    assert _flagged(table, "clock-step:cam") == [90, *range(92, 101)]


def test_a_whole_period_step_without_the_attribute_is_not_detected() -> None:
    """The operating limit: −1 s moves records as a missed edge would; no flag says so."""
    rows, cam = _stepped(attribute=False, step=-1.0)
    table = shot_table(START, rows, [cam])
    assert table.record_for_shot["cam"][90:99] == tuple(range(91, 100))
    assert _flagged(table, "clock-step:cam") == []


def test_a_lone_late_record_is_no_clock_step() -> None:
    rows = _rows(20)
    cam = _cam(rows)
    stamps = cam.records.stamps.copy()
    stamps[7] += 0.3
    table = shot_table(
        START, rows, [_device("cam", stamps, cam.counter, cam.witness_count)]
    )
    assert table.record_for_shot["cam"] == tuple(range(20))
    assert all(flags == () for flags in table.flags)


@pytest.mark.parametrize(
    ("lag_rows", "drop", "flagged"),
    [
        ((5, 6), (), []),  # two adjacent poll lags
        ((0, 1), (), []),  # at the run's start
        ((18, 19), (), []),  # at its end
        ((6,), (5,), []),  # a missed frame, then one poll lag
        ((7,), (5, 6), []),  # two missed, then one poll lag
        ((5, 6, 7), (), [5, 6, 7, 8, 9]),  # a sustained run, edges and all
        ((6, 7), (5,), [5, 7, 8, 9]),  # two after a gap: a forward step
    ],
)
def test_sporadic_poll_lag_is_no_clock_step_a_sustained_run_is(
    lag_rows, drop, flagged
) -> None:
    rows = _rows(20)
    cam = _cam(rows, drop=drop)
    attr = cam.witness_count.copy()
    kept = [k for k in range(20) if k not in drop]
    for k in lag_rows:
        attr[kept.index(k)] -= 1
    table = shot_table(
        START,
        rows,
        [
            _device(
                "cam",
                cam.records.stamps,
                cam.counter,
                attr,
                np.arange(float(len(kept))),
            )
        ],
    )
    assert _flagged(table, "clock-step:cam") == flagged


def test_residual_dither_within_the_stamp_budget_is_no_clock_step() -> None:
    rows = _rows(60)
    cam = _cam(rows)
    dither = np.random.default_rng(3).uniform(-0.03, 0.03, 60)
    attr = cam.witness_count - np.isin(np.arange(60), [9, 33]).astype(float)
    table = shot_table(
        START,
        rows,
        [_device("cam", cam.records.stamps + dither, cam.counter, attr)],
    )
    assert all(flags == () for flags in table.flags)


def test_shots_sharing_a_witness_stamp_take_records_in_counter_order() -> None:
    """A +2 update: shots 4 and 5 share one stamp; time cannot order them."""
    rows = _rows()
    rows.loc[4, "witness_stamp"] = rows["witness_stamp"][3]
    cam = _cam(rows)
    stamps = cam.records.stamps.copy()
    stamps[3], stamps[4] = stamps[3] + 0.04, stamps[3] - 0.02  # frame 5 nearer
    table = shot_table(
        START, rows, [_device("cam", stamps, cam.counter, cam.witness_count)]
    )
    assert table.record_for_shot["cam"] == tuple(range(8))
    assert _flagged(table, "witness-stamp-shared") == [4, 5]


def test_a_witness_outage_falls_back_to_fire_intervals_unaudited() -> None:
    rows = _rows()
    rows.loc[3:5, ["witness_count", "witness_stamp"]] = np.nan
    rows["unwitnessed"] = np.arange(8) >= 3
    rows.loc[6:, "unwitnessed"] = False
    cam = _cam(_rows())
    attr = cam.witness_count.copy()
    attr[3:6] = attr[2]  # the camera keeps the last count while the IOC is down
    table = shot_table(
        START, rows, [_device("cam", cam.records.stamps, cam.counter, attr)]
    )
    assert table.record_for_shot["cam"] == tuple(range(8))
    assert _flagged(table, "unwitnessed") == [4, 5, 6]
    assert [f for flags in table.flags for f in flags if f != "unwitnessed"] == []


def test_a_fire_nobody_saw_keeps_the_previous_row_its_edge_and_frame() -> None:
    """Sim-bench scan 4: row 5 fired, the witness did not count, the camera wrote nothing.

    Row 5 repeats row 4's count and stamp.  It has no edge of its own, so it
    is timed by its fire and flagged unwitnessed; row 4 keeps its frame.
    """
    clean = _rows()
    rows = clean.copy()
    rows.loc[4, ["witness_count", "witness_stamp"]] = rows.loc[
        3, ["witness_count", "witness_stamp"]
    ].to_numpy()
    rows.loc[4, "fire_stamp"] = rows["witness_stamp"][3] + 0.07  # after row 4's edge
    rows.loc[4, "complete"] = False
    rows["witness_late"] = False
    cam = _cam(clean, drop=(4,))
    table = shot_table(START, rows, [cam])
    assert table.record_for_shot["cam"] == (0, 1, 2, 3, None, 4, 5, 6)
    assert _flagged(table, "unwitnessed") == [5]
    assert not any("witness-stamp-shared" in f for row in table.flags for f in row)
    assert table.held.get("cam", ()) == () and table.extras["cam"] == ()
    assert [f for row in table.flags for f in row] == ["unwitnessed", "missing:cam"]
    assert list(table.rows["cam-meancounts"][:4]) == [0.0, 10.0, 20.0, 30.0]
    # the old layout never reads the witness columns
    old = {key: v for key, v in START.items() if key != "layout"}
    assert not any("unwitnessed" in row for row in shot_table(old, rows, [cam]).flags)


def test_a_failed_witness_read_does_not_hide_a_repeated_count() -> None:
    """Row 4 counted, row 5's witness read failed (NaN, ""), row 6 repeats row 4."""
    clean = _rows()
    rows = clean.copy()
    rows.loc[4, ["witness_count", "witness_stamp"]] = np.nan
    rows.loc[4, "witness_epoch"] = ""
    rows.loc[5, ["witness_count", "witness_stamp"]] = rows.loc[
        3, ["witness_count", "witness_stamp"]
    ].to_numpy()
    rows.loc[6:, "witness_count"] -= 1
    rows.loc[5, "fire_stamp"] = rows["witness_stamp"][3] + 1.7
    rows.loc[5, "complete"] = False
    cam = _cam(clean, drop=(5,), attribute=False)
    table = shot_table(START, rows, [cam])
    assert table.record_for_shot["cam"][3] == 3  # row 4 keeps its frame
    assert _flagged(table, "unwitnessed") == [5, 6]
    assert not any("witness-stamp-shared" in f for row in table.flags for f in row)
    assert table.held.get("cam", ()) == ()


def test_an_equal_count_across_an_epoch_change_is_a_restart_not_a_repeat() -> None:
    """The witness restarted: row 5's count equals row 4's in a new epoch."""
    rows = _rows()
    rows.loc[4:, "witness_epoch"] = "e2"
    rows.loc[4:, "witness_count"] = rows["witness_count"][3] + np.arange(4.0)
    table = shot_table(START, rows, [_cam(_rows(), attribute=False)])
    assert table.record_for_shot["cam"] == tuple(range(8))
    assert _flagged(table, "unwitnessed") == []


@pytest.mark.parametrize("late", [True, False])
def test_a_dropped_witness_count_times_its_row_by_the_fire(late: bool) -> None:
    """Sim-bench scan 3: the witness missed row 5's edge; the camera took it.

    Flagged ``witness_late`` by the plan or not, the repeated count is no
    edge: row 5 takes its own frame by its fire interval, row 4 keeps its.
    """
    clean = _rows()
    rows = clean.copy()
    rows.loc[4:, "witness_count"] -= 1  # one count lost from row 5 on
    rows.loc[4, "witness_stamp"] = rows["witness_stamp"][3]
    rows["witness_late"] = [k == 4 and late for k in range(8)]
    cam = _cam(clean, attribute=False)
    table = shot_table(START, rows, [cam])
    assert table.record_for_shot["cam"] == tuple(range(8))
    assert _flagged(table, "unwitnessed") == [5]
    assert not any("witness-stamp-shared" in f for row in table.flags for f in row)


def test_unwitnessed_assigns_by_interval_not_nearest() -> None:
    """The fire leads its edge by up to a period: here 0.7 s, nearer the next fire."""
    rows = _rows(lead=0.7).drop(columns=["witness_count", "witness_stamp"])
    start = {**START, "shot_clock_column": "fire_stamp"}
    cam = _cam(_rows(lead=0.7), offset=0.0)
    table = shot_table(start, rows, [cam])
    assert table.record_for_shot["cam"] == tuple(range(8))
    assert _flagged(table, "unwitnessed") == list(range(1, 9))
    late = _device("cam", cam.records.stamps[1:], value=np.arange(7.0))
    assert shot_table(start, rows, [late]).record_for_shot["cam"][0] is None


# ------------------------------------------------------- the witness attribute
def test_an_attribute_one_behind_is_the_measured_lag_not_a_mismatch() -> None:
    rows = _rows(60)
    cam = _cam(rows)
    attr = cam.witness_count.copy()
    attr[[7, 23, 41]] -= 1  # ~1.7 % of frames carry the previous count
    table = shot_table(
        START, rows, [_device("cam", cam.records.stamps, cam.counter, attr)]
    )
    assert all(flags == () for flags in table.flags)


def test_a_frozen_attribute_while_the_witness_advances_is_witness_stale() -> None:
    rows = _rows()
    cam = _cam(rows)
    attr = cam.witness_count.copy()
    attr[4:] = attr[4]  # the camera stopped seeing the witness after shot 5
    table = shot_table(
        START, rows, [_device("cam", cam.records.stamps, cam.counter, attr)]
    )
    assert _flagged(table, "witness-stale:cam") == [7, 8]  # shot 6 is a lag of one
    assert _flagged(table, "audit-mismatch:cam") == []


def test_an_attribute_off_by_more_than_one_is_an_audit_mismatch() -> None:
    rows = _rows()
    cam = _cam(rows)
    attr = cam.witness_count + np.where(np.arange(8) == 5, 3.0, 0.0)
    table = shot_table(
        START, rows, [_device("cam", cam.records.stamps, cam.counter, attr)]
    )
    assert _flagged(table, "audit-mismatch:cam") == [6]


def test_a_camera_booted_without_the_witness_has_no_attribute_audit() -> None:
    rows = _rows()
    cam = _cam(rows)
    nan = np.full(8, np.nan)
    table = shot_table(
        START, rows, [_device("cam", cam.records.stamps, cam.counter, nan)]
    )
    assert table.device_flags == {"cam": ("no-attribute-audit",)}
    assert table.record_for_shot["cam"] == tuple(range(8))
    assert all(flags == () for flags in table.flags)


def test_complete_disagreeing_with_the_join_is_flagged_both_ways() -> None:
    rows = _rows()
    rows.loc[1, "complete"] = False  # the plan refired, but a frame did land
    table = shot_table(START, rows, [_cam(rows, drop=(5,))])
    assert _flagged(table, "complete-mismatch") == [2, 6]


# ------------------------------------------------------------ fly mode
def test_soft_tick_pressed_by_stamp_free_running_last_before_and_stale() -> None:
    rows = _rows(6)
    start = {**START, "clock": "soft-tick", "required": ["cam", "gauge"]}
    gauge_stamps = T0 + LV + np.array([-0.5, 1.2, 1.9])  # then nothing
    gauge = _device(
        "gauge",
        gauge_stamps,
        value=[1.0, 2.0, 3.0],
        sync_class=ASYNCHRONOUS,
        stale_after=2.5,
    )
    cam = _cam(
        rows
    )  # its witness attribute is the idle hardware counter's, not the tick's
    cam = _device("cam", cam.records.stamps, cam.counter, np.full(6, 7.0))
    table = shot_table(start, rows, [cam, gauge])
    assert all(
        not f.startswith("audit-mismatch") for flags in table.flags for f in flags
    )
    assert table.record_for_shot["cam"] == tuple(range(6))
    assert table.record_for_shot["gauge"] == (0, 0, 2, 2, 2, 2)
    assert table.ages["gauge"] == pytest.approx([0.5, 1.5, 0.1, 1.1, 2.1, 3.1])
    assert _flagged(table, "stale:gauge") == [6]
    assert list(table.rows["gauge-meancounts"]) == [1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
    assert "gauge" not in table.extras


# ------------------------------------------------------------ the old layouts
def _old_rows(n: int = 5) -> pd.DataFrame:
    stamps = T0 + LV + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "uc_cam-acq_timestamp": stamps,
            "uc_cam-meancounts": np.arange(n, dtype=float),
            "bin_number": 1,
        }
    )


def test_old_strict_is_positional_in_primary_and_joined_elsewhere() -> None:
    rows = _old_rows()
    stack = DeviceRecords("primary", FrameColumns("uc_cam", np.full(4, np.nan)))
    side = _device(
        "uc_haso", rows["uc_cam-acq_timestamp"][[0, 2, 3]] + 0.05, value=[7.0, 8.0, 9.0]
    )
    start = {"detectors": ["uc_cam"]}
    table = shot_table(start, rows, [stack, side], drain_offsets={"uc_haso": 0.05})
    assert [frame_for(table, "uc_cam", s) for s in (1, 4, 5)] == [
        ("primary", 0),
        ("primary", 3),
        None,
    ]
    assert table.record_for_shot["uc_haso"] == (0, None, 1, 2, None)
    expected = join_frame_columns(
        start, rows, [side.records], drain_offsets={"uc_haso": 0.05}
    )
    pd.testing.assert_frame_equal(table.rows, expected)


def test_old_gated_reproduces_the_s_file_join() -> None:
    rows = _old_rows(6).drop(columns=["uc_cam-meancounts"])
    stamps = rows["uc_cam-acq_timestamp"].to_numpy()
    frames = _device("uc_probe", np.delete(stamps, 3) + 0.02, value=np.arange(5.0))
    start = {"shot_clock_column": "uc_cam-acq_timestamp", "shot_period": 1.0}
    table = shot_table(start, rows, [frames], row_stream="shots")
    join = join_frames_to_shots(
        stamps, frames.records.stamps, windows=row_windows(stamps, 1.0)
    )
    assert (
        table.record_for_shot["uc_probe"]
        == join.frame_for_shot
        == (0, 1, 2, None, 3, 4)
    )
    assert frame_for(table, "uc_probe", 5) == ("uc_probe_stream", 3)
    pd.testing.assert_frame_equal(
        table.rows, join_frame_columns(start, rows, [frames.records])
    )


# ------------------------------------------------------------ review round 1
def test_clock_stamps_reads_frames_and_document_rows_in_one_epoch() -> None:
    frame = pd.DataFrame({"witness_stamp": [1.0], "uc_a-acq_timestamp": [2.0]})
    assert clock_stamps(frame, "witness_stamp")[0] == 1.0 + LV
    assert (
        clock_stamps([{"uc_a-acq_timestamp": 2.0}, {}], "uc_a-acq_timestamp")[0] == 2.0
    )
    assert np.isnan(clock_stamps([{}], "fire_stamp")[0])


@pytest.mark.parametrize(
    "lead, skew", [(0.0005, -0.002), (0.003, -0.005), (0.0005, 0.002)]
)
def test_an_interval_keeps_a_record_stamped_just_before_its_fire(lead, skew) -> None:
    """A small lead and a few ms of skew either way: the record stays with its fire."""
    rows = _rows(6, lead=lead).drop(columns=["witness_count", "witness_stamp"])
    start = {**START, "shot_clock_column": "fire_stamp"}
    stamps = T0 + LV + np.arange(6.0) + skew
    table = shot_table(start, rows, [_device("cam", np.delete(stamps, 2))])
    assert table.record_for_shot["cam"] == (0, 1, None, 2, 3, 4)
    assert table.extras["cam"] == ()


def test_a_free_running_device_is_sampled_at_the_fire_on_loose_rows() -> None:
    rows = _rows()
    rows.loc[3:5, ["witness_count", "witness_stamp"]] = np.nan
    rows["unwitnessed"] = [k in (3, 4, 5) for k in range(8)]
    gauge = _device(
        "gauge",
        T0 + LV + np.arange(8) - 0.2,
        value=np.arange(8.0),
        sync_class=ASYNCHRONOUS,
    )
    table = shot_table(START, rows, [gauge])
    # fire = stamp - 0.3: the sample 0.2 s before the edge is not yet there
    assert table.record_for_shot["gauge"] == (0, 1, 2, 2, 3, 4, 6, 7)
    assert table.ages["gauge"][3] == pytest.approx(
        0.93, abs=1e-6
    )  # (fire + m) − the last sample


def test_soft_tick_pressed_frames_late_in_the_period_stay_on_their_own_tick() -> None:
    """A frame 0.6 P after its tick is that tick's, not the nearest (next) tick's."""
    rows = _rows(5)
    start = {**START, "clock": "soft-tick"}
    cam = _cam(rows, offset=0.6)
    cam = _device("cam", cam.records.stamps, cam.counter, None, np.arange(5.0))
    table = shot_table(start, rows, [cam])
    assert table.record_for_shot["cam"] == (0, 1, 2, 3, 4)
    assert not any(table.flags) and table.extras["cam"] == ()
    # a missed press is a gap on its own row, never a shift
    cam = _cam(rows, offset=0.6, drop=(2,))
    cam = _device("cam", cam.records.stamps, cam.counter, None, np.arange(4.0))
    table = shot_table(start, rows, [cam])
    assert table.record_for_shot["cam"] == (0, 1, None, 2, 3)
    assert _flagged(table, "missing:cam") == [3]


@pytest.mark.parametrize("n", [4, 3])
def test_soft_tick_a_held_ticks_late_frame_never_displaces_the_next_rows_own(n) -> None:
    """Row 2's press lands just before tick 3: row 3 keeps its own, later frame.

    ``n == 3`` is the case the counter audit cannot see (the last row).
    """
    rows = _rows(n)
    rows["complete"] = [k != 1 for k in range(n)]
    start = {**START, "clock": "soft-tick"}
    t = rows["witness_stamp"].to_numpy() + LV
    stamps = [t[0] + 0.05, t[2] - 0.01, t[2] + 0.05] + ([t[3] + 0.05] if n == 4 else [])
    cam = _device("cam", stamps, np.arange(1.0, n + 1), None, np.arange(float(n)))
    table = shot_table(start, rows, [cam])
    assert table.record_for_shot["cam"] == (0, None, 2, 3)[:n]
    assert table.extras["cam"] == (1,)


def test_a_stale_required_free_running_record_does_not_complete_a_row() -> None:
    rows = _rows(6)
    start = {**START, "clock": "soft-tick", "required": ["gauge"]}
    gauge = _device(
        "gauge",
        T0 + LV + np.array([-0.5, 0.5]),
        value=[1.0, 2.0],
        sync_class=ASYNCHRONOUS,
    )
    table = shot_table(start, rows, [gauge])
    assert _flagged(table, "stale:gauge") == [4, 5, 6]
    assert _flagged(table, "complete-mismatch") == [4, 5, 6]


def test_required_matches_ignoring_case_and_flags_an_unknown_name_once() -> None:
    rows = _rows()
    start = {**START, "required": ["UC-Cam", "u_gone"]}
    cam = _cam(rows)
    named = DeviceRecords("uc_cam_stream", FrameColumns("uc_cam", cam.records.stamps))
    table = shot_table(start, rows, [named])
    assert table.device_flags == {"u_gone": ("required-unresolved",)}
    assert all(flags == () for flags in table.flags)


def _stalled(
    drop: tuple[int, ...] = (), counter=None
) -> tuple[pd.DataFrame, DeviceRecords]:
    """Shots 4 and 5 coalesced into one +2 update stamped at shot 5's poll."""
    truth = _rows()
    rows = truth.copy()
    rows.loc[3, "witness_stamp"] = rows["witness_stamp"][4]
    cam = _cam(truth, drop=drop)
    counter = cam.counter if counter is None else np.asarray(counter, dtype=float)
    return rows, _device("cam", cam.records.stamps, counter, cam.witness_count)


def test_a_stalled_group_is_filled_when_its_counters_account_for_every_row() -> None:
    rows, cam = _stalled()
    table = shot_table(START, rows, [cam])
    assert table.record_for_shot["cam"] == tuple(range(8))
    assert _flagged(table, "witness-stamp-shared") == [4, 5]
    assert table.extras["cam"] == ()


@pytest.mark.parametrize(
    "drop, counter",
    [((3,), None), ((), [1, 2, 3, 4, 6, 7, 8, 9])],
    ids=["one-record-short", "counters-skip"],
)
def test_a_stalled_group_that_does_not_add_up_is_left_unresolved(drop, counter) -> None:
    rows, cam = _stalled(drop, counter)
    table = shot_table(START, rows, [cam])
    assert table.record_for_shot["cam"][3:5] == (None, None)
    assert _flagged(table, "witness-stamp-shared-unresolved:cam") == [4, 5]
    assert _flagged(table, "missing:cam") == []
    assert set(table.extras["cam"]) == ({3} if drop else {3, 4})


def test_a_run_filed_one_shot_late_is_attribute_lag_systematic() -> None:
    rows = _rows(30)
    cam = _cam(rows)
    late = _device("cam", cam.records.stamps + 1.0, cam.counter, cam.witness_count)
    assert shot_table(START, rows, [late]).device_flags == {
        "cam": ("attribute-lag-systematic",)
    }
    attr = cam.witness_count - np.isin(np.arange(30), [3, 17]).astype(float)
    ontime = _device("cam", cam.records.stamps, cam.counter, attr)
    assert shot_table(START, rows, [ontime]).device_flags == {}


def test_old_layout_rows_match_the_s_file_join_exactly() -> None:
    """Positional stacks add no columns; rows without a clock get nothing joined."""
    rows = _old_rows(3).drop(columns=["uc_cam-meancounts"])
    stack = DeviceRecords(
        "primary",
        FrameColumns("uc_cam", np.full(3, np.nan), {"uc_cam-frame_recv_x": np.ones(3)}),
    )
    table = shot_table({"detectors": ["uc_cam"]}, rows, [stack])
    pd.testing.assert_frame_equal(table.rows, join_frame_columns({}, rows, []))
    bare = pd.DataFrame({"bin_number": [1, 1, 1]}, index=[5, 6, 7])
    side = _device("uc_haso", T0 + LV + np.arange(3.0), value=[1.0, 2.0, 3.0])
    table = shot_table({}, bare, [side])
    pd.testing.assert_frame_equal(
        table.rows, join_frame_columns({}, bare, [side.records])
    )
    assert frame_for(table, "uc_haso", 1) is None


# ------------------------------------------------------------ review round 2
FIRE_CLOCK = {**START, "shot_clock_column": "fire_stamp"}


@pytest.mark.parametrize("drop", [(), (2,)], ids=["every-row", "a-miss"])
def test_a_phase_locked_loop_keeps_each_record_on_its_own_fire(drop) -> None:
    """Fire k+1 waits for the slowest device: the edge sits 25 ms (< m) before it."""
    rows = _rows(6, lead=0.975).drop(columns=["witness_count", "witness_stamp"])
    stamps = np.delete(T0 + LV + np.arange(6.0), drop)
    table = shot_table(FIRE_CLOCK, rows, [_device("cam", stamps)])
    expected = tuple(range(6)) if not drop else (0, 1, None, 2, 3, 4)
    assert table.record_for_shot["cam"] == expected
    assert table.extras["cam"] == ()


def test_an_overlap_record_after_a_filled_row_goes_to_the_next_row() -> None:
    rows = _rows(6, lead=0.0).drop(columns=["witness_count", "witness_stamp"])
    stamps = T0 + LV + np.arange(6.0) + 0.002
    stamps[3] -= 0.012  # 10 ms before its own fire, inside row 2's overlap
    table = shot_table(FIRE_CLOCK, rows, [_device("cam", stamps)])
    assert table.record_for_shot["cam"] == tuple(range(6))


def test_an_overlap_record_after_a_missed_row_with_no_evidence_is_ambiguous() -> None:
    """A witness outage that opens on a miss: whose is the record before fire 5?

    Nothing in the outage tells a late record from the next row's early one,
    so every overlap after the miss stays ambiguous (the rows' baselines would decide).
    """
    rows = _rows(8, lead=0.975)
    rows.loc[3:, ["witness_count", "witness_stamp"]] = np.nan
    rows["unwitnessed"] = np.arange(8) >= 3
    stamps = np.delete(T0 + LV + np.arange(8.0), 3)
    table = shot_table(START, rows, [_device("cam", stamps)])
    assert table.record_for_shot["cam"] == (0, 1, 2) + (None,) * 5
    assert _flagged(table, "interval-ambiguous:cam") == [5, 6, 7, 8]
    assert _flagged(table, "missing:cam") == [4]
    assert table.extras["cam"] == (3, 4, 5, 6)


@pytest.mark.parametrize("step", [1.0, 0.95, 1.05])
def test_an_early_whole_period_step_cannot_become_the_reference(step) -> None:
    """Shifted records outnumbering the true ones still read as moved."""
    rows = _rows(100)
    cam = _cam(rows)
    stamps = cam.records.stamps + np.where(np.arange(100) >= 10, step, 0.0)
    table = shot_table(
        START, rows, [_device("cam", stamps, cam.counter, cam.witness_count)]
    )
    wrong = [
        k
        for k, i in enumerate(table.record_for_shot["cam"])
        if i is not None and i != k
    ]
    assert len(wrong) == 89
    assert all("clock-step:cam" in table.flags[k] for k in wrong)


@pytest.mark.parametrize("step", [1.0, 0.95, 1.05])
def test_a_startup_poll_lag_does_not_hide_a_later_clock_step(step) -> None:
    """The reference is the contract lag, not the run's first agreeing rows."""
    rows = _rows(100)
    cam = _cam(rows)
    attribute = cam.witness_count.copy()
    attribute[2] -= 1
    stamps = cam.records.stamps + np.where(np.arange(100) >= 5, step, 0.0)
    table = shot_table(START, rows, [_device("cam", stamps, cam.counter, attribute)])
    wrong = [
        k
        for k, i in enumerate(table.record_for_shot["cam"])
        if i is not None and i != k
    ]
    assert len(wrong) == 94
    assert all("clock-step:cam" in table.flags[k] for k in wrong)
    assert "clock-step:cam" not in table.flags[2]  # the poll lag itself


def test_a_run_filed_one_shot_late_from_its_start_flags_every_moved_row() -> None:
    rows = _rows(30)
    cam = _cam(rows)
    late = _device("cam", cam.records.stamps + 1.0, cam.counter, cam.witness_count)
    table = shot_table(START, rows, [late])
    wrong = [
        k
        for k, i in enumerate(table.record_for_shot["cam"])
        if i is not None and i != k
    ]
    assert wrong and all("clock-step:cam" in table.flags[k] for k in wrong)


@pytest.mark.parametrize("step", [-1.0, -0.8, 1.0])
def test_a_poll_lag_inside_a_moved_stretch_does_not_verify_its_row(step) -> None:
    """A record moved early that also poll-lags reads as equal; its stretch decides."""
    rows = _rows(100)
    cam = _cam(rows)
    attribute = cam.witness_count.copy()
    attribute[[30, 60]] -= 1
    stamps = cam.records.stamps + np.where(np.arange(100) >= 10, step, 0.0)
    table = shot_table(START, rows, [_device("cam", stamps, cam.counter, attribute)])
    wrong = [
        k
        for k, i in enumerate(table.record_for_shot["cam"])
        if i is not None and i != k
    ]
    assert wrong and all("clock-step:cam" in table.flags[k] for k in wrong)


def test_a_poll_lag_on_a_backward_stretchs_last_row_is_flagged_with_it() -> None:
    """The stretch's edge row, which its window reads as on the contract, is flagged."""
    rows = _rows(100)
    cam = _cam(rows)
    attribute = cam.witness_count.copy()
    attribute[47] -= 1
    moved = (np.arange(100) >= 40) & (np.arange(100) <= 47)
    stamps = cam.records.stamps + np.where(moved, -1.0, 0.0)
    table = shot_table(START, rows, [_device("cam", stamps, cam.counter, attribute)])
    wrong = [
        k
        for k, i in enumerate(table.record_for_shot["cam"])
        if i is not None and i != k
    ]
    assert wrong and all("clock-step:cam" in table.flags[k] for k in wrong)


def test_frequent_poll_lag_qualifies_the_run_instead_of_flagging_rows() -> None:
    """A slow witness poll lags often; the attribute then audits no row."""
    rows = _rows(100)
    cam = _cam(rows)
    attribute = cam.witness_count.copy()
    attribute[np.random.default_rng(5).random(100) < 0.5] -= 1
    table = shot_table(
        START, rows, [_device("cam", cam.records.stamps, cam.counter, attribute)]
    )
    assert "attribute-lag-noisy" in table.device_flags["cam"]
    assert _flagged(table, "clock-step:cam") == []


def test_a_frozen_witness_attribute_is_no_poll_noise() -> None:
    """A stalled attribute climbs past one behind; a later step is still flagged."""
    rows = _rows(200)
    cam = _cam(rows)
    attribute = cam.witness_count.copy()
    attribute[20:45] = attribute[19]  # frozen for 25 rows while the count runs
    attribute[180:] -= 1  # every record from 180 one shot late
    table = shot_table(
        START, rows, [_device("cam", cam.records.stamps, cam.counter, attribute)]
    )
    assert "attribute-lag-noisy" not in table.device_flags.get("cam", ())
    assert all("clock-step:cam" in table.flags[k] for k in range(181, 200))


# ------------------------------------- unwitnessed rows filed by the baselines
# Two Bluesky runs of 2026-11-17, unwitnessed, every row complete in the plan;
# stamps relative to the first fire (s).  The fire interval misfiled both.
#: A camera triggered at 1 Hz: each frame lands ~25 ms before the next fire.
PHASE_LOCKED = {
    "fires": [0.0, 0.447, 1.404, 2.368, 3.37, 4.376, 5.372, 6.396, 7.38, 8.377],
    "baselines": [-6.65, 0.346, 1.346, 2.344, 3.344, 4.344, 5.343, 6.343, 7.342, 8.342],
    "records": [0.346, 1.346, 2.344, 3.344, 4.344, 5.343, 6.343, 7.342, 8.342, 9.341],
}
#: A free-running camera whose clock runs behind the worker's: every frame
#: is stamped before the fire it answered.
STAMPED_BEFORE_FIRE = {
    "fires": [0.0, 0.25, 0.412, 0.589, 0.831],
    "baselines": [-0.321, -0.081, 0.1, 0.287, 0.522],
    "records": [-0.081, 0.1, 0.287, 0.522, 0.704],
}


def _unwitnessed(fires, baselines, *, name: str = "cam") -> pd.DataFrame:
    """A strict run with no witness: fire stamps (Unix) and one stamp baseline."""
    n = len(fires)
    return pd.DataFrame(
        {
            "shot_index": np.arange(1, n + 1),
            "fire_stamp": T0 + np.asarray(fires, dtype=float),
            "unwitnessed": [True] * n,
            "complete": [True] * n,
            f"{name}-baseline": T0 + LV + np.asarray(baselines, dtype=float),
        }
    )


@pytest.mark.parametrize(
    "run", [PHASE_LOCKED, STAMPED_BEFORE_FIRE], ids=["phase-locked", "before-fire"]
)
def test_a_baseline_files_each_record_on_the_shot_that_produced_it(run) -> None:
    rows = _unwitnessed(run["fires"], run["baselines"])
    cam = _device("cam", T0 + LV + np.asarray(run["records"]))
    table = shot_table(FIRE_CLOCK, rows, [cam])
    n = len(run["fires"])
    assert table.record_for_shot["cam"] == tuple(range(n))
    assert table.extras["cam"] == ()
    assert all(flags == ("unwitnessed",) for flags in table.flags)


def test_without_the_baseline_the_same_run_is_misfiled() -> None:
    """The fallback this replaces: the record just before fire k+1 is ambiguous."""
    run = PHASE_LOCKED
    rows = _unwitnessed(run["fires"], run["baselines"]).drop(columns="cam-baseline")
    cam = _device("cam", T0 + LV + np.asarray(run["records"]))
    assert shot_table(FIRE_CLOCK, rows, [cam]).record_for_shot["cam"] != tuple(
        range(10)
    )


def test_a_miss_leaves_the_baseline_unmoved_and_only_its_row_empty() -> None:
    """Shot 3 recorded nothing: baselines 3 and 4 are the same stamp."""
    run = PHASE_LOCKED
    baselines = list(run["baselines"])
    baselines[3] = baselines[2]  # nothing new between fires 3 and 4
    records = np.delete(T0 + LV + np.asarray(run["records"]), 2)
    table = shot_table(
        FIRE_CLOCK, _unwitnessed(run["fires"], baselines), [_device("cam", records)]
    )
    assert table.record_for_shot["cam"] == (0, 1, None, 2, 3, 4, 5, 6, 7, 8)
    assert _flagged(table, "missing:cam") == [3]


def _late(table) -> list[int]:
    return _flagged(table, "late-stretch:cam")


def test_a_late_record_of_a_given_up_row_is_flagged_disputed() -> None:
    """Row 2 timed out; its frame (stamped at its fire) arrived after fire 3's read.

    Interval 2 is empty and interval 3 holds two records.  Arrival order
    reads it as a miss and a doubled frame; the fire interval reads a late
    record.  The baseline reading stays and both rows are flagged.
    """
    fires = [0.0, 1.0, 2.0, 3.0, 4.0]
    baselines = [-0.9, 0.1, 0.1, 2.1, 3.1]
    rows = _unwitnessed(fires, baselines)
    rows["complete"] = [True, False, True, True, True]
    records = T0 + LV + np.asarray([0.1, 1.1, 2.1, 3.1, 4.1])
    table = shot_table(FIRE_CLOCK, rows, [_device("cam", records)])
    assert table.record_for_shot["cam"] == (0, None, 1, 3, 4)
    assert _late(table) == [2, 3]
    assert _flagged(table, "missing:cam") == [2]


@pytest.mark.parametrize(
    "baselines, filed, disputed",
    [
        ([-0.9, 0.1, 0.1, 1.1, 2.1], (0, None, 1, 2, 3), [2, 3, 4, 5]),
        ([-0.9, -0.9, 0.1, 1.1, 2.1], (None, 0, 1, 2, 3), [1, 2, 3, 4, 5]),
    ],
    ids=["to-the-open-last-row", "over-the-whole-run"],
)
def test_a_long_late_stretch_flags_every_row_it_may_have_shifted(
    baselines, filed, disputed
) -> None:
    """The refire completed on the late frame, and so on, until the open last row."""
    fires = [0.0, 1.0, 2.0, 3.0, 4.0]
    records = T0 + LV + np.asarray([0.1, 1.1, 2.1, 3.1, 4.1])
    table = shot_table(
        FIRE_CLOCK, _unwitnessed(fires, baselines), [_device("cam", records)]
    )
    assert table.record_for_shot["cam"] == filed
    assert _late(table) == disputed


def test_a_miss_then_an_extra_frame_is_filed_as_such_and_flagged_late_stretch() -> None:
    """A genuine miss on row 3, a doubled frame after row 4's: one stretch, flagged."""
    fires = [0.0, 1.0, 2.0, 3.0, 4.0]
    baselines = [-0.9, 0.1, 1.1, 1.1, 3.15]  # the doubled frame landed before fire 5
    rows = _unwitnessed(fires, baselines)
    records = T0 + LV + np.asarray([0.1, 1.1, 3.1, 3.15, 4.1])
    table = shot_table(FIRE_CLOCK, rows, [_device("cam", records)])
    assert table.record_for_shot["cam"] == (0, 1, None, 2, 4)
    assert table.extras["cam"] == (3,)
    assert _flagged(table, "missing:cam") == [3]
    assert _late(table) == [3, 4]


def test_stamps_behind_the_worker_keep_a_miss_and_a_doubled_frame_on_their_rows() -> (
    None
):
    """Every frame stamped 0.1 s before its fire; a miss at shot 3, a doubled frame at shot 10.

    The fire interval would hand every row between them the next shot's
    frame, unflagged; the baselines file the truth, and the rows the fire
    interval reads differently say so.
    """
    fires = np.arange(12.0)
    frames: list[tuple[int, float]] = []  # (shot, stamp) in arrival order
    for k in range(12):
        if k == 2:
            continue
        frames.append((k, fires[k] - 0.1))
        if k == 9:
            frames.append((k, fires[k] - 0.05))
    stamps = np.array([t for _, t in frames])
    baselines = [
        max((t for shot, t in frames if shot < k), default=fires[0] - 1.0)
        for k in range(12)
    ]
    table = shot_table(
        FIRE_CLOCK, _unwitnessed(fires, baselines), [_device("cam", T0 + LV + stamps)]
    )
    assert table.record_for_shot["cam"] == (0, 1, None, 2, 3, 4, 5, 6, 7, 8, 10, 11)
    assert table.extras["cam"] == (9,)
    assert _late(table) and set(_late(table)) <= set(range(3, 11))


def test_two_late_frames_in_a_row_keep_one_stretch() -> None:
    """Two empty intervals are repaid by two surplus records, in one stretch."""
    fires = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    baselines = [-0.9, 0.1, 0.1, 0.1, 2.1, 4.1]
    records = T0 + LV + np.asarray([0.1, 1.1, 2.1, 3.1, 4.1, 5.1])
    table = shot_table(
        FIRE_CLOCK, _unwitnessed(fires, baselines), [_device("cam", records)]
    )
    assert table.record_for_shot["cam"] == (0, None, None, 1, 3, 5)
    assert _late(table) == [2, 3, 4, 5]


def _stepped_back(step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fires 0..5, the device clock stepping back by *step* at shot 3; the baselines its previous stamps."""
    fires = np.arange(6.0)
    records = fires + 0.1
    records[2:] -= step
    baselines = np.concatenate([[records[0] - 1.0], records[:-1]])
    return fires, baselines, records


def test_a_backward_clock_step_leaves_the_device_to_the_fire_interval() -> None:
    """Back 1.6 s, past a shot: the baselines run backwards and the intervals overlap."""
    fires, baselines, records = _stepped_back(1.6)
    cam = _device("cam", T0 + LV + records)
    table = shot_table(FIRE_CLOCK, _unwitnessed(fires, baselines), [cam])
    unbased = _unwitnessed(fires, baselines).drop(columns="cam-baseline")
    assert (
        table.record_for_shot == shot_table(FIRE_CLOCK, unbased, [cam]).record_for_shot
    )
    filed = [i for i in table.record_for_shot["cam"] if i is not None]
    assert len(filed) == len(set(filed))


def test_a_step_back_that_keeps_the_stamps_rising_is_still_filed_by_baseline() -> None:
    """Back 0.5 s, within a period: the arrival order still holds every row."""
    fires, baselines, records = _stepped_back(0.5)
    cam = _device("cam", T0 + LV + records)
    table = shot_table(FIRE_CLOCK, _unwitnessed(fires, baselines), [cam])
    assert table.record_for_shot["cam"] == tuple(range(6))


def test_the_drain_offset_moves_the_baseline_with_the_records() -> None:
    run = PHASE_LOCKED
    rows = _unwitnessed(run["fires"], np.asarray(run["baselines"]) + 0.2)
    cam = _device("cam", T0 + LV + np.asarray(run["records"]) + 0.2)
    table = shot_table(FIRE_CLOCK, rows, [cam], drain_offsets={"cam": 0.2})
    assert table.record_for_shot["cam"] == tuple(range(10))


def test_a_frame_count_baseline_is_not_a_stamp_and_the_interval_decides() -> None:
    """A plugin camera's baseline is its frame count: not on the records' scale."""
    run = PHASE_LOCKED
    rows = _unwitnessed(run["fires"], run["baselines"])
    rows["cam-baseline"] = 100.0 + np.arange(10)
    cam = _device("cam", T0 + LV + np.asarray(run["records"]))
    without = _unwitnessed(run["fires"], run["baselines"]).drop(columns="cam-baseline")
    assert (
        shot_table(FIRE_CLOCK, rows, [cam]).record_for_shot
        == shot_table(FIRE_CLOCK, without, [cam]).record_for_shot
    )


def test_a_row_whose_baseline_was_not_read_falls_back_to_its_fire_interval() -> None:
    """Records stamped 0.1 s before their fires: only a baseline keeps them.

    Row 3's baseline is NaN, so its fire interval decides and finds nothing
    (its record is before fire 3 − m).  Read as "no lower bound", the
    baseline rule would have filed it; with no baselines at all, every row
    would have taken the next shot's record.
    """
    fires = [0.0, 1.0, 2.0, 3.0]
    records = [-0.1, 0.9, 1.9, 2.9]
    rows = _unwitnessed(fires, [-1.1, -0.1, 0.9, 1.9])
    rows.loc[2, "cam-baseline"] = np.nan
    table = shot_table(
        FIRE_CLOCK, rows, [_device("cam", T0 + LV + np.asarray(records))]
    )
    assert table.record_for_shot["cam"] == (0, 1, None, 3)
    assert _flagged(table, "missing:cam") == [3]


def test_witnessed_rows_ignore_the_baseline() -> None:
    """The witness still times a witnessed row; a wrong baseline there moves nothing."""
    rows = _rows(6)
    rows["cam-baseline"] = T0 + LV + 100.0  # nonsense, after every record
    table = shot_table(START, rows, [_cam(rows)])
    assert table.record_for_shot["cam"] == tuple(range(6))


def test_the_baseline_resolves_an_outage_that_opens_on_a_miss() -> None:
    """The ambiguous case above, with the baselines the plan recorded."""
    rows = _rows(8, lead=0.975)
    rows.loc[3:, ["witness_count", "witness_stamp"]] = np.nan
    rows["unwitnessed"] = np.arange(8) >= 3
    edges = T0 + LV + np.arange(8.0)
    stamps = np.delete(edges, 3)
    last_before = [edges[k - 1] if k else edges[0] - 1.0 for k in range(8)]
    last_before[4] = edges[2]  # shot 4 recorded nothing
    rows["cam-baseline"] = last_before
    table = shot_table(START, rows, [_device("cam", stamps)])
    assert table.record_for_shot["cam"] == (0, 1, 2, None, 3, 4, 5, 6)
    assert _flagged(table, "interval-ambiguous:cam") == []
    assert _flagged(table, "missing:cam") == [4]


def test_an_earlier_plain_miss_does_not_hide_a_later_late_record() -> None:
    """A genuine miss at shot 2, then shot 5's frame arrives after fire 6's read."""
    fires = np.arange(8.0)
    rows = _unwitnessed(fires, [-0.9, 0.1, 0.1, 2.1, 3.1, 3.1, 5.1, 6.1])
    records = T0 + LV + np.asarray([0.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1])
    table = shot_table(FIRE_CLOCK, rows, [_device("cam", records)])
    assert table.record_for_shot["cam"] == (0, None, 1, 2, None, 3, 5, 6)
    assert _late(table) == [5, 6]
    assert 2 not in _late(table)  # the plain miss is no stretch


def test_a_stretch_runs_from_the_miss_its_surplus_repays_to_the_surplus() -> None:
    """Phase-locked (frames 25 ms before the next fire), a miss at shot 3, a doubled frame at 10.

    The filing is the miss-and-double reading, the truth here; every row
    between is flagged, since a late shot-3 frame would read the same.
    """
    fires = np.arange(12.0)
    frames: list[tuple[int, float]] = []
    for k in range(12):
        if k == 2:
            continue
        frames.append((k, fires[k] + 0.975))
        if k == 9:
            frames.append((k, fires[k] + 0.98))
    stamps = np.array([t for _, t in frames])
    baselines = [
        max((t for shot, t in frames if shot < k), default=fires[0] - 1.0)
        for k in range(12)
    ]
    table = shot_table(
        FIRE_CLOCK, _unwitnessed(fires, baselines), [_device("cam", T0 + LV + stamps)]
    )
    assert table.record_for_shot["cam"] == (0, 1, None, 2, 3, 4, 5, 6, 7, 8, 10, 11)
    assert _late(table) == list(range(3, 11))


def test_a_late_record_behind_an_unread_baseline_still_flags_its_stretch() -> None:
    """Shot 1's baseline was not read and its frame arrived after fire 2's read."""
    fires = [0.0, 1.0, 2.0, 3.0]
    rows = _unwitnessed(fires, [np.nan, -0.9, 1.1, 2.1])
    records = T0 + LV + np.asarray([0.1, 1.1, 2.1, 3.1])
    table = shot_table(FIRE_CLOCK, rows, [_device("cam", records)])
    assert _late(table) == [1, 2]


def test_an_unread_next_baseline_bounds_the_row_by_the_next_one_read() -> None:
    """Shots 2 and 3 missed and shot 3's baseline was not read: shot 2 must not reach shot 4's frame."""
    fires = np.arange(6.0)
    rows = _unwitnessed(fires, [-0.9, 0.1, np.nan, 0.1, 3.1, 4.1])
    records = T0 + LV + np.asarray([0.1, 3.1, 4.1, 5.1])
    table = shot_table(FIRE_CLOCK, rows, [_device("cam", records)])
    assert table.record_for_shot["cam"][:2] == (0, None)
    assert table.record_for_shot["cam"][3:] == (1, 2, 3)
    assert _flagged(table, "baseline-unread:cam") == [3]
