"""Camera masks and fixed-canvas rotation retain legacy rasterization exactly."""

import numpy as np
import pytest
from pydantic import ValidationError

from geecs_analysis.pipeline import apply_pipeline
from geecs_analysis.specs import Pipeline
from geecs_data_utils.frames import Axis, Frame, ShotMeta


@pytest.mark.parametrize("angle", [0, 1e-7, -3.5, 2.5, -8, 90])
@pytest.mark.parametrize("center, thickness", [((6, 8), 4), ((0, 0), 3), ((50, 70), 1)])
def test_crosshair_mask_matches_legacy_at_edges_and_rotated(angle, center, thickness):
    from geecs_schemas.analysis.processing_2d import CrosshairMaskingConfig
    from image_analysis.processing.array2d.masking import apply_crosshair_masking

    raw = np.random.default_rng(8).normal(size=(13, 17))
    frame = Frame.from_array(raw, shot=ShotMeta("camera", 3, 12.5))
    step = dict(
        step="crosshair_mask",
        center=center,
        width=9,
        height=7,
        thickness=thickness,
        angle=angle,
        value=2.5,
    )
    result = apply_pipeline(frame, Pipeline(steps=[step]))
    old = apply_crosshair_masking(
        raw,
        CrosshairMaskingConfig(
            crosshairs=[
                dict(
                    center=(center[1], center[0]),
                    width=9,
                    height=7,
                    thickness=thickness,
                    angle=angle,
                )
            ],
            mask_value=2.5,
        ),
    )
    np.testing.assert_array_equal(result.data, old)
    np.testing.assert_array_equal(frame.data, raw)
    assert result.axes is frame.axes
    assert result.shot is frame.shot
    assert not result.data.flags.writeable


def test_crosshair_after_crop_uses_local_samples_and_preserves_nonfinite_behavior():
    from geecs_schemas.analysis.processing_2d import CrosshairMaskingConfig
    from image_analysis.processing.array2d.masking import apply_crosshair_masking

    raw = np.arange(100, dtype=float).reshape(10, 10)
    raw[5, 5] = np.nan
    raw[6, 6] = np.inf
    frame = Frame.from_array(raw)
    cross = dict(
        step="crosshair_mask",
        center=(3, 3),
        width=7,
        height=5,
        thickness=4,
        angle=10,
        value=-1,
    )
    result = apply_pipeline(
        frame,
        Pipeline(
            steps=[
                dict(step="roi", bounds=[[2, 9], [2, 9]]),
                cross,
                cross,
            ]
        ),
    )
    config = CrosshairMaskingConfig(
        crosshairs=[
            dict(
                center=(3, 3),
                width=7,
                height=5,
                thickness=4,
                angle=10,
            )
        ],
        mask_value=-1,
    )
    old = apply_crosshair_masking(
        apply_crosshair_masking(raw[2:9, 2:9], config), config
    )
    # Classify undefined samples separately from finite numerical equality.
    np.testing.assert_array_equal(np.isfinite(result.data), np.isfinite(old))
    np.testing.assert_array_equal(result.data[np.isfinite(old)], old[np.isfinite(old)])
    assert np.isnan(result.data[3, 3])
    np.testing.assert_array_equal(result.axes[0].values, np.arange(2, 9))


@pytest.mark.parametrize("angle", [0, -3.5, 2.5, -8, 90])
def test_rotation_matches_legacy_on_fixed_sample_canvas(angle):
    from image_analysis.processing.array2d.transforms import apply_rotation

    raw = np.random.default_rng(9).normal(size=(13, 17))
    frame = Frame.from_array(
        raw,
        axes=(
            Axis(np.arange(13) ** 2, "mm", "y"),
            Axis(np.arange(17) * 0.2, "mm", "x"),
        ),
        unit="counts",
        shot=ShotMeta("camera", 4, 5),
    )
    result = apply_pipeline(frame, Pipeline(steps=[dict(step="rotate", angle=angle)]))
    np.testing.assert_array_equal(result.data, apply_rotation(raw, angle))
    np.testing.assert_array_equal(frame.data, raw)
    assert result.axes is frame.axes
    assert result.unit == frame.unit and result.shot is frame.shot
    assert result.data.shape == raw.shape


@pytest.mark.parametrize(
    "step",
    [
        dict(step="rotate", angle=float("nan")),
        dict(step="rotate", angle=3, fill_value=float("inf")),
        dict(step="crosshair_mask", center=(1, 2), width=0, height=3, thickness=2),
        dict(step="crosshair_mask", center=(1.5, 2), width=3, height=3, thickness=2),
        dict(step="crosshair_mask", center=(-1, 2), width=3, height=3, thickness=2),
        dict(step="crosshair_mask", center=(1, 2), width=3, height=3, thickness=True),
    ],
)
def test_invalid_geometry_is_rejected(step):
    with pytest.raises(ValidationError):
        Pipeline(steps=[step])


@pytest.mark.parametrize(
    "step",
    [
        dict(step="rotate", angle=10),
        dict(
            step="crosshair_mask",
            center=(1, 2),
            width=3,
            height=3,
            thickness=2,
        ),
    ],
)
def test_image_geometry_refuses_traces(step):
    with pytest.raises(ValueError, match="does not support 1D"):
        apply_pipeline(Frame.from_array([1, 2, 3]), Pipeline(steps=[step]))
