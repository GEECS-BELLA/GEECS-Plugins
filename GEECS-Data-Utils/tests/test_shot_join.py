"""The shot join: frames onto rows by offset-corrected stamp (08 §4.5)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from geecs_data_utils.io.scan_stack import LABVIEW_EPOCH_OFFSET
from geecs_data_utils.shot_join import (
    DEFAULT_SHOT_PERIOD_S,
    FrameColumns,
    frame_columns_from_attributes,
    join_frames_to_shots,
    join_window,
    shot_clock_column,
)

SHOTS = np.array([1001.0, 1002.0, 1003.0])


# --------------------------------------------------------------- the window
def test_window_is_half_the_period_and_never_reaches_a_neighbour() -> None:
    assert join_window(SHOTS) == pytest.approx(DEFAULT_SHOT_PERIOD_S / 2)
    # a 10 Hz run narrows the window by itself — no neighbouring shot can match
    fast = np.array([1001.0, 1001.1, 1001.2])
    assert join_window(fast) == pytest.approx(0.05)
    # a step gap never *widens* it past half the period
    sparse = np.array([1001.0, 1005.0, 1009.0])
    assert join_window(sparse) == pytest.approx(0.5)
    # an explicit period narrows it
    assert join_window(SHOTS, period=0.2) == pytest.approx(0.1)
    # one row, or none, leaves the period alone
    assert join_window(np.array([1001.0])) == pytest.approx(0.5)
    assert join_window(np.array([])) == pytest.approx(0.5)
    # a sub-millisecond repeat is one row's publish race, not a second shot:
    # ignored, so the window comes from the real gap to the next row
    assert join_window(np.array([1001.0, 1001.0002, 1002.0])) == pytest.approx(
        0.9998 / 2
    )


# ----------------------------------------------------------------- the join
def test_same_device_frames_land_exactly_on_their_rows() -> None:
    join = join_frames_to_shots(SHOTS, SHOTS, window=0.5)
    assert join.frame_for_shot == (0, 1, 2)
    assert join.orphans == () and join.contested == ()
    assert join.matched == 3


def test_a_second_camera_joins_after_its_drain_offset_is_backed_out() -> None:
    """The clock camera stamps 100 ms after the trigger, this one 180 ms."""
    frames = SHOTS + 0.08  # the raw cross-device difference
    assert join_frames_to_shots(SHOTS, frames, window=0.5).frame_for_shot == (0, 1, 2)
    # corrected, the same frames land at delta 0 — and a window far too tight
    # for the raw difference still matches
    join = join_frames_to_shots(
        SHOTS, frames, window=0.01, shot_offset=0.10, frame_offset=0.18
    )
    assert join.frame_for_shot == (0, 1, 2)
    # uncorrected, that window misses every frame: they all orphan
    raw = join_frames_to_shots(SHOTS, frames, window=0.01)
    assert raw.frame_for_shot == (None, None, None) and len(raw.orphans) == 3


def test_an_orphan_frame_is_dropped_and_a_shot_without_one_is_none() -> None:
    """The extra edge at a step's end, and a camera that missed shot 2."""
    join = join_frames_to_shots(SHOTS, np.array([1001.0, 1003.0, 1004.0]), window=0.5)
    assert join.frame_for_shot == (0, None, 1)
    assert join.orphans == (2,)  # 1004.0 has no shot row
    assert join.matched == 2


def test_two_frames_for_one_shot_keep_the_first() -> None:
    join = join_frames_to_shots(SHOTS, np.array([1001.0, 1001.1, 1002.0]), window=0.5)
    assert join.frame_for_shot == (0, 2, None)
    assert join.contested == (1,)


def test_non_finite_stamps_match_nothing() -> None:
    join = join_frames_to_shots(
        np.array([1001.0, float("nan")]), np.array([1001.0, float("nan")]), window=0.5
    )
    assert join.frame_for_shot == (0, None)
    assert join.orphans == (1,)
    # no rows at all: every frame is an orphan
    assert join_frames_to_shots(np.array([]), SHOTS, window=0.5).orphans == (0, 1, 2)


def test_rows_out_of_time_order_still_join() -> None:
    """A retaken step can leave the rows unsorted; the join sorts internally."""
    join = join_frames_to_shots(np.array([1003.0, 1001.0, 1002.0]), SHOTS, window=0.5)
    assert join.frame_for_shot == (2, 0, 1)


# ------------------------------------------------------------ the clock column
def test_the_clock_column_is_the_one_shot_clock_names() -> None:
    columns = ["bin_number", "u_ict-acq_timestamp", "uc_amp4_ir_input-acq_timestamp"]
    start = {
        "shot_clock": "UC_Amp4_IR_input",
        "detectors": ["u_ict", "uc_amp4_ir_input"],
    }
    assert shot_clock_column(start, columns) == "uc_amp4_ir_input-acq_timestamp"


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


# ------------------------------------------------- columns from a stack's attrs
def test_attributes_become_strict_row_column_names() -> None:
    columns = frame_columns_from_attributes(
        "uc_a",
        {
            "uc_a-hdf-image-frame_acq_timestamp": np.array([1.0, 2.0]),
            "uc_a-hdf-image-frame_recv_timestamp": np.array([1.1, 2.1]),
            "uc_a-hdf-image-meancounts": np.array([10.0, 20.0]),
        },
        drain_offset=0.07,
        labview_epoch_offset=LABVIEW_EPOCH_OFFSET,
    )
    assert columns is not None
    assert columns.object_name == "uc_a" and columns.drain_offset == 0.07
    assert len(columns) == 2
    # the frame stamp is spelled like the strict row's stamp column, in the
    # rows' LabVIEW epoch
    assert list(columns.columns["uc_a-acq_timestamp"]) == [
        1.0 + LABVIEW_EPOCH_OFFSET,
        2.0 + LABVIEW_EPOCH_OFFSET,
    ]
    assert list(columns.stamps) == list(columns.columns["uc_a-acq_timestamp"])
    # a subscribed scalar is spelled as a strict row spells it
    assert list(columns.columns["uc_a-meancounts"]) == [10.0, 20.0]
    # the plugin's receive stamp keeps its own name (no header ever claims it)
    assert "uc_a-frame_recv_timestamp" in columns.columns


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
