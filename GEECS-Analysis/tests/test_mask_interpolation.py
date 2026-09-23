"""Geometry and pure-step behavior beyond the v2 differential oracle."""

import numpy as np
import pytest

from geecs_analysis.pipeline import apply_pipeline
from geecs_analysis.specs import Pipeline
from geecs_data_utils.frames import Axis, Frame, ShotMeta


@pytest.mark.parametrize("units, center", [("axis", [22, 12]), ("index", [2, 2])])
def test_circle_boundary_and_crop_coordinates(units, center):
    source = Frame(
        np.ones((7, 7)),
        (Axis(np.arange(20, 27), unit="mm"), Axis(np.arange(10, 17), unit="mm")),
        shot=ShotMeta("camera", 3),
    )
    cropped = source.crop((slice(0, 5), slice(0, 5)))
    result = apply_pipeline(
        cropped,
        Pipeline(
            steps=[
                {
                    "step": "circular_mask",
                    "center": center,
                    "radius": 2,
                    "units": units,
                }
            ]
        ),
    )
    assert np.count_nonzero(result.data) == 13
    assert result.data[0, 2] == 1  # the exact boundary is inside
    assert result.data[0, 0] == 0
    np.testing.assert_array_equal(result.axes[0].values, np.arange(20, 25))
    np.testing.assert_array_equal(result.axes[1].values, np.arange(10, 15))
    assert result.axes[0].unit == "mm"
    assert result.shot == source.shot
    np.testing.assert_array_equal(source.data, np.ones((7, 7)))


def test_interpolation_zero_padding_preserves_units_and_provenance():
    frame = Frame(
        [2, 6, 2],
        (Axis([1, 3, 5], unit="ns", label="time"),),
        shot=ShotMeta("scope", 5, 123),
        unit="V",
        label="signal",
    )
    result = apply_pipeline(
        frame,
        Pipeline(
            steps=[
                {
                    "step": "interpolate",
                    "count": 7,
                    "lower": 0,
                    "upper": 6,
                }
            ]
        ),
    )
    np.testing.assert_array_equal(result.data, [0, 2, 4, 6, 4, 2, 0])
    np.testing.assert_array_equal(result.axes[0].values, np.arange(7))
    assert result.axes[0].unit == "ns" and result.axes[0].label == "time"
    assert result.unit == "V" and result.label == "signal" and result.shot == frame.shot
    np.testing.assert_array_equal(frame.data, [2, 6, 2])


@pytest.mark.parametrize(
    "step, ndim",
    [
        ({"step": "circular_mask", "center": [1, 1], "radius": 1}, 1),
        ({"step": "interpolate", "count": 10}, 2),
    ],
)
def test_new_steps_reject_wrong_dimensions(step, ndim):
    with pytest.raises(ValueError, match="support"):
        apply_pipeline(Frame.from_array(np.ones((3,) * ndim)), Pipeline(steps=[step]))
