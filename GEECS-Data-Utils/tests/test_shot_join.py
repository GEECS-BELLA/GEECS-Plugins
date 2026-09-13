"""The shot join: frames onto rows by offset-corrected stamp (08 §4.5)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
from geecs_data_utils.shot_join import (
    DEFAULT_SHOT_PERIOD_S,
    FrameColumns,
    clock_device,
    frame_columns_from_attributes,
    join_frames_to_shots,
    row_windows,
    shot_clock_column,
)

SHOTS = np.array([1001.0, 1002.0, 1003.0])


def _join(shots, frames, *, period=DEFAULT_SHOT_PERIOD_S, shot=0.0, frame=0.0):
    """Join *frames* to *shots* the way every caller does: windows from the rows."""
    return join_frames_to_shots(
        shots,
        frames,
        windows=row_windows(shots, period),
        shot_offset=shot,
        frame_offset=frame,
    )


# -------------------------------------------------------------- the windows
def test_each_row_gets_half_the_period_unless_a_neighbour_is_closer() -> None:
    assert list(row_windows(SHOTS)) == pytest.approx([0.5, 0.5, 0.5])
    # a 10 Hz run narrows every window by itself
    assert list(row_windows(np.array([1001.0, 1001.1, 1001.2]))) == pytest.approx(
        [0.05, 0.05, 0.05]
    )
    # a step gap never *widens* a window past half the period
    assert list(row_windows(np.array([1001.0, 1005.0, 1009.0]))) == pytest.approx(
        [0.5, 0.5, 0.5]
    )
    # an explicit period narrows them
    assert list(row_windows(SHOTS, period=0.2)) == pytest.approx([0.1, 0.1, 0.1])
    # one row, or none
    assert list(row_windows(np.array([1001.0]))) == pytest.approx([0.5])
    assert list(row_windows(np.array([]))) == []


def test_one_close_pair_narrows_only_those_two_rows() -> None:
    """The cap is per row: an anomalous pair must not shrink the whole run.

    Finding 4 of the #858 review — a global minimum let a single 0.3 s gap
    collapse every row's window below the real cross-device drain spread.
    """
    windows = row_windows(np.array([1001.0, 1001.3, 1002.3, 1003.3]))
    assert list(windows) == pytest.approx([0.15, 0.15, 0.5, 0.5])


def test_a_sub_millisecond_neighbour_is_one_rows_publish_race() -> None:
    """Skipped when measuring: each window comes from the next *real* row.

    Rows 0 and 1 are 0.2 ms apart, so neither narrows the other; row 0 looks
    past its twin to 1002.0 (a full period away, so half the period stands)
    and rows 1 and 2 measure the 0.9998 s between them.
    """
    windows = row_windows(np.array([1001.0, 1001.0002, 1002.0]))
    assert list(windows) == pytest.approx([0.5, 0.4999, 0.4999])


def test_a_non_finite_row_keeps_the_periods_window() -> None:
    windows = row_windows(np.array([1001.0, float("nan"), 1002.0]))
    assert list(windows) == pytest.approx([0.5, 0.5, 0.5])


# ----------------------------------------------------------------- the join
def test_same_device_frames_land_exactly_on_their_rows() -> None:
    join = _join(SHOTS, SHOTS)
    assert join.frame_for_shot == (0, 1, 2)
    assert join.orphans == () and join.contested == ()
    assert join.matched == 3


def test_a_second_camera_joins_after_its_drain_offset_is_backed_out() -> None:
    """The clock camera stamps 100 ms after the trigger, this one 180 ms."""
    frames = SHOTS + 0.08  # the raw cross-device difference
    assert _join(SHOTS, frames).frame_for_shot == (0, 1, 2)
    # corrected, the same frames land at delta 0 — and windows far too tight
    # for the raw difference still match
    join = _join(SHOTS, frames, period=0.02, shot=0.10, frame=0.18)
    assert join.frame_for_shot == (0, 1, 2)
    # uncorrected, those windows miss every frame: they all orphan
    raw = _join(SHOTS, frames, period=0.02)
    assert raw.frame_for_shot == (None, None, None) and len(raw.orphans) == 3


def test_an_orphan_frame_is_dropped_and_a_shot_without_one_is_none() -> None:
    """The extra edge at a step's end, and a camera that missed shot 2."""
    join = _join(SHOTS, np.array([1001.0, 1003.0, 1004.0]))
    assert join.frame_for_shot == (0, None, 1)
    assert join.orphans == (2,)  # 1004.0 has no shot row
    assert join.matched == 2


def test_two_frames_for_one_shot_keep_the_nearer_one() -> None:
    """Finding 9 of the #858 review: nearest wins, not first in the file.

    Frame 0 is 0.40 s from the row and frame 1 is 0.01 s from it; the s-file
    must carry frame 1's values, and frame 0 is reported as contested.
    """
    join = _join(np.array([1001.0, 1002.5]), np.array([1000.6, 1000.99]))
    assert join.frame_for_shot == (1, None)
    assert join.contested == (0,)
    assert join.orphans == ()
    # and the same when the nearer frame comes first
    join = _join(np.array([1001.0, 1002.5]), np.array([1000.99, 1000.6]))
    assert join.frame_for_shot == (0, None)
    assert join.contested == (1,)


def test_a_frame_on_the_boundary_two_rows_share_goes_to_one_of_them() -> None:
    """The windows are half-open, so no frame can ever be given to two rows.

    A frame exactly midway between two rows is inside both rows' closed
    windows; ``[stamp - w, stamp + w)`` gives it to the later row alone.
    """
    rows = np.array([1001.0, 1002.0])
    join = _join(rows, np.array([1001.5]))
    assert join.frame_for_shot == (None, 0)
    assert join.orphans == () and join.contested == ()
    # and with a frame of its own, each row keeps the nearer one
    join = _join(rows, np.array([1001.0, 1001.5, 1002.0]))
    assert join.frame_for_shot == (0, 2)
    assert join.contested == (1,)


def test_non_finite_stamps_match_nothing() -> None:
    join = _join(np.array([1001.0, float("nan")]), np.array([1001.0, float("nan")]))
    assert join.frame_for_shot == (0, None)
    assert join.orphans == (1,)
    # no rows at all: every frame is an orphan
    assert _join(np.array([]), SHOTS).orphans == (0, 1, 2)


def test_rows_out_of_time_order_still_join() -> None:
    """A retaken step can leave the rows unsorted; the join sorts internally."""
    join = _join(np.array([1003.0, 1001.0, 1002.0]), SHOTS)
    assert join.frame_for_shot == (2, 0, 1)


def test_windows_must_match_the_rows() -> None:
    with pytest.raises(ValueError, match="2 entry/entries for 3 row"):
        join_frames_to_shots(SHOTS, SHOTS, windows=np.array([0.5, 0.5]))


# ------------------------------------------------------------ the clock column
def test_the_clock_column_prefers_the_one_the_plan_recorded() -> None:
    columns = ["bin_number", "u_ict-acq_timestamp", "uc_amp4_ir_input-acq_timestamp"]
    start = {
        "shot_clock_column": "uc_amp4_ir_input-acq_timestamp",
        "shot_clock": "U_BCaveICT",  # deliberately disagreeing: the column wins
        "detectors": ["u_ict"],
    }
    assert shot_clock_column(start, columns) == "uc_amp4_ir_input-acq_timestamp"


def test_the_clock_column_matches_shot_clock_for_a_run_recorded_before_0_85(
    caplog,
) -> None:
    columns = ["bin_number", "u_ict-acq_timestamp", "uc_amp4_ir_input-acq_timestamp"]
    start = {
        "shot_clock": "UC_Amp4_IR_input",
        "detectors": ["u_ict", "uc_amp4_ir_input"],
    }
    assert shot_clock_column(start, columns) == "uc_amp4_ir_input-acq_timestamp"
    # a stale recorded column falls back to the device name, loudly
    with caplog.at_level(logging.WARNING):
        assert (
            shot_clock_column(
                {**start, "shot_clock_column": "gone-acq_timestamp"}, columns
            )
            == "uc_amp4_ir_input-acq_timestamp"
        )
    assert "is not a row column" in caplog.text


def test_the_clock_column_falls_back_to_detector_order_then_anything(caplog) -> None:
    columns = ["bin_number", "u_ict-acq_timestamp", "uc_cam-acq_timestamp"]
    assert (
        shot_clock_column({"detectors": ["u_gauge", "uc_cam"]}, columns)
        == "uc_cam-acq_timestamp"
    )
    assert shot_clock_column({}, columns) == "u_ict-acq_timestamp"
    assert shot_clock_column({}, ["bin_number"]) is None
    with caplog.at_level(logging.WARNING):
        assert (
            shot_clock_column({"shot_clock": "U_Gone"}, columns)
            == "u_ict-acq_timestamp"
        )
    assert "has no stamp column" in caplog.text


def test_clock_device_strips_the_stamp_suffix() -> None:
    assert clock_device("uc_a-acq_timestamp") == "uc_a"
    assert clock_device("bin_number") == "bin_number"


# ------------------------------------------------- columns from a stack's attrs
def test_attributes_become_strict_row_column_names() -> None:
    columns = frame_columns_from_attributes(
        "uc_a",
        {
            "uc_a-hdf-image-frame_acq_timestamp": np.array([1.0, 2.0]),
            "uc_a-hdf-image-frame_recv_timestamp": np.array([1.1, 2.1]),
            "uc_a-hdf-image-meancounts": np.array([10.0, 20.0]),
        },
        variables={"uc_a-hdf-image-meancounts": "MeanCounts"},
        labview_epoch_offset=LABVIEW_EPOCH_OFFSET,
    )
    assert columns is not None
    assert columns.object_name == "uc_a"
    assert len(columns) == 2
    # the frame stamp is spelled like the strict row's stamp column, in the
    # rows' LabVIEW epoch
    assert list(columns.columns["uc_a-acq_timestamp"]) == [
        1.0 + LABVIEW_EPOCH_OFFSET,
        2.0 + LABVIEW_EPOCH_OFFSET,
    ]
    assert list(columns.stamps) == list(columns.columns["uc_a-acq_timestamp"])
    # a subscribed scalar is spelled as a strict row spells it, and the
    # manifest's raw GEECS name is carried through for messages
    assert list(columns.columns["uc_a-meancounts"]) == [10.0, 20.0]
    assert columns.raw_names["uc_a-meancounts"] == "MeanCounts"
    # the plugin's receive stamp keeps its own name (no header ever claims it)
    assert "uc_a-frame_recv_timestamp" in columns.columns


def test_a_ragged_attribute_column_is_dropped_not_padded(caplog) -> None:
    """Finding 7 of the #858 review: a short dataset is a defect, not missing data."""
    with caplog.at_level(logging.WARNING):
        columns = frame_columns_from_attributes(
            "uc_a",
            {
                "uc_a-hdf-image-frame_acq_timestamp": np.array([1.0, 2.0, 3.0, 4.0]),
                "uc_a-hdf-image-meancounts": np.array([10.0, 20.0]),
            },
        )
    assert columns is not None
    assert len(columns) == 4
    assert "uc_a-meancounts" not in columns.columns
    assert "2 value(s) for 4 frame(s); dropped" in caplog.text


def test_a_non_numeric_attribute_is_skipped_not_raised(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        columns = frame_columns_from_attributes(
            "uc_a",
            {
                "uc_a-hdf-image-frame_acq_timestamp": np.array([1.0]),
                "uc_a-hdf-image-mode": np.array(["wide"], dtype=object),
            },
        )
    assert columns is not None
    assert "uc_a-mode" not in columns.columns
    assert "is not numeric" in caplog.text


def test_a_pre_0_8_stack_uses_the_stream_name_for_its_bare_stamp() -> None:
    columns = frame_columns_from_attributes("uc_a", {"acq_timestamp": np.array([5.0])})
    assert columns is not None
    assert list(columns.columns["uc_a-acq_timestamp"]) == [5.0]


def test_attributes_without_a_frame_stamp_are_not_joinable(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert (
            frame_columns_from_attributes("uc_a", {"uc_a-hdf-image-meancounts": [1.0]})
            is None
        )
    assert "no frame stamp" in caplog.text


def test_frame_columns_length_is_its_frame_count() -> None:
    assert len(FrameColumns("uc_a", np.zeros(7))) == 7


def test_truncated_keeps_only_the_referenced_frames() -> None:
    """Finding 6 of the #858 review: frames past the datums are not s-file data."""
    source = FrameColumns(
        "uc_a",
        np.arange(5.0),
        columns={"uc_a-acq_timestamp": np.arange(5.0), "uc_a-max": np.arange(5.0) * 2},
        raw_names={"uc_a-max": "MaxCounts"},
    )
    cut = source.truncated(3)
    assert len(cut) == 3
    assert list(cut.columns["uc_a-max"]) == [0.0, 2.0, 4.0]
    assert cut.raw_names == {"uc_a-max": "MaxCounts"}
    # a limit at or past the end, or a nonsense one, changes nothing
    assert len(source.truncated(5)) == 5
    assert len(source.truncated(9)) == 5
    assert len(source.truncated(-1)) == 5
