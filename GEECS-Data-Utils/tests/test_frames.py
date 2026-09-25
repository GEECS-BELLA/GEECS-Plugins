"""Coordinate and ownership invariants for trace/image processing frames."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from geecs_data_utils.frames import Axis, Frame, ShotMeta


def test_frame_owns_live_buffers_and_rejects_inplace_writes():
    source = np.arange(12, dtype=np.uint16).reshape(3, 4)
    frame = Frame.from_array(source)
    source[:] = 0
    np.testing.assert_array_equal(frame.data, np.arange(12).reshape(3, 4))
    assert frame.data.dtype == np.float64
    with pytest.raises(ValueError, match="read-only"):
        frame.data[0, 0] = 2
    with pytest.raises(FrozenInstanceError):
        frame.shot = ShotMeta("other")


def test_axis_owns_coordinate_buffer_and_preserves_nonuniform_descending_order():
    coordinates = np.array([12.0, 7.0, -3.0])
    axis = Axis(coordinates, "MeV", "energy")
    coordinates[:] = 5
    np.testing.assert_array_equal(axis.values, [12, 7, -3])
    with pytest.raises(ValueError, match="read-only"):
        axis.values[0] = 0


@pytest.mark.parametrize("values", [[], [[1, 2]], [1, np.nan], [np.inf]])
def test_invalid_coordinates_fail_at_boundary(values):
    with pytest.raises(ValueError):
        Axis(np.asarray(values))


@pytest.mark.parametrize("data", [np.zeros(()), np.zeros((2, 2, 2)), np.zeros((0, 2))])
def test_invalid_frame_dimensions(data):
    with pytest.raises(ValueError):
        Frame.from_array(data)


@pytest.mark.parametrize("data", [[1j], ["1"], [object()]])
def test_nonreal_samples_are_not_silently_coerced(data):
    with pytest.raises(TypeError, match="real numerical"):
        Frame.from_array(data)


def test_invalid_axis_count_and_length_fail():
    with pytest.raises(ValueError, match="one Axis"):
        Frame(np.zeros((2, 3)), (Axis(np.arange(2)),))
    with pytest.raises(ValueError, match="shape"):
        Frame(np.zeros((2, 3)), (Axis(np.arange(3)), Axis(np.arange(2))))


def test_default_image_axes_follow_numpy_order():
    frame = Frame.from_array(np.zeros((2, 3)))
    assert [axis.label for axis in frame.axes] == ["y", "x"]
    np.testing.assert_array_equal(frame.axes[0].values, [0, 1])
    np.testing.assert_array_equal(frame.axes[1].values, [0, 1, 2])


def test_crop_keeps_global_calibrated_coordinates_and_provenance():
    shot = ShotMeta("Camera", 7, 100.5)
    frame = Frame.from_array(
        np.arange(20).reshape(4, 5),
        axes=(Axis(np.arange(4) * 2, "mm", "y"), Axis(np.arange(5) + 10, "mm", "x")),
        shot=shot,
        unit="counts",
        label="intensity",
    )
    cropped = frame.crop((slice(1, 3), slice(2, 5)))
    np.testing.assert_array_equal(cropped.data, [[7, 8, 9], [12, 13, 14]])
    np.testing.assert_array_equal(cropped.axes[0].values, [2, 4])
    np.testing.assert_array_equal(cropped.axes[1].values, [12, 13, 14])
    assert cropped.axes[0].unit == "mm"
    assert cropped.shot is shot
    assert (cropped.unit, cropped.label) == ("counts", "intensity")


def test_nested_crop_and_reversal_keep_trace_coordinates():
    frame = Frame.from_trace([[1, 10], [3, 20], [8, 30], [9, 40]], x_unit="s")
    cropped = frame.crop((slice(1, None),)).crop((slice(None, None, -1),))
    np.testing.assert_array_equal(cropped.as_trace(), [[9, 40], [8, 30], [3, 20]])
    assert cropped.axes[0].unit == "s"


def test_empty_or_dimension_dropping_crop_is_rejected():
    frame = Frame.from_array(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        frame.crop((slice(1, 1), slice(None)))
    with pytest.raises(ValueError, match="one slice"):
        frame.crop((1, slice(None)))


def test_replacement_revalidates_geometry_and_retains_original():
    frame = Frame.from_array([1, 2, 3])
    result = frame.replace(data=frame.data - 2)
    np.testing.assert_array_equal(frame.data, [1, 2, 3])
    np.testing.assert_array_equal(result.data, [-1, 0, 1])
    with pytest.raises(ValueError, match="shape"):
        frame.replace(data=[1, 2])


def test_trace_adapter_preserves_axes_units_and_returns_independent_array():
    trace = np.array([[2.0, 4.0], [3.0, 8.0]])
    frame = Frame.from_trace(
        trace, x_unit="s", y_unit="V", x_label="time", y_label="signal"
    )
    copy = frame.as_trace()
    np.testing.assert_array_equal(copy, trace)
    copy[:] = 99
    trace[:] = -1
    np.testing.assert_array_equal(frame.as_trace(), [[2, 4], [3, 8]])
    assert (frame.axes[0].unit, frame.unit, frame.label) == ("s", "V", "signal")


def test_invalid_signal_values_survive_for_measure_to_handle():
    frame = Frame.from_array([0, np.nan, np.inf])
    assert np.isnan(frame.data[1])
    assert np.isinf(frame.data[2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": ""},
        {"device": "cam", "shot_number": 0},
        {"device": "cam", "shot_number": True},
        {"device": "cam", "acq_timestamp": np.nan},
    ],
)
def test_invalid_shot_identity(kwargs):
    with pytest.raises(ValueError):
        ShotMeta(**kwargs)
