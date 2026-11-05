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


@pytest.mark.xfail(reason="residual tracker, later slice", strict=True)
def test_a_clock_step_is_flagged_where_it_appears() -> None:
    """§4.3: equal counters with a jump in residual is *clock-step*."""
    rows = _rows()
    cam = _cam(rows)
    stamps = cam.records.stamps + np.where(np.arange(8) >= 4, 0.3, 0.0)
    table = shot_table(
        START, rows, [_device("cam", stamps, cam.counter, cam.witness_count)]
    )
    assert _flagged(table, "clock-step:cam") == [5]


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


# ------------------------------------------------------------ the W5 attribute
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
    so every overlap after the miss stays ambiguous (W3c's baselines decide).
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
