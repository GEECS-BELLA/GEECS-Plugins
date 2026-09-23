"""Rotate image samples on the existing fixed canvas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class RotateSpec(StepSpec):
    """Rotate in sample-index space, retaining shape and the output axis grid.

    This moves the signal on a fixed canvas; it does not rotate world-coordinate
    axes or expand the image. On nonuniform axes the angle is still in index
    space. Cubic interpolation with prefilter=False preserves the v2 algorithm.
    """

    step: Literal["rotate"] = "rotate"
    angle: float
    fill_value: float = 0.0


@step(RotateSpec, ndim={2})
def rotate(frame: Frame, spec: RotateSpec) -> Frame:
    """Resample on the fixed canvas without changing input ownership or metadata."""
    from scipy.ndimage import rotate as scipy_rotate

    if spec.angle == 0:
        return frame
    return frame.replace(
        data=scipy_rotate(
            frame.data,
            spec.angle,
            reshape=False,
            cval=spec.fill_value,
            order=3,
            prefilter=False,
        )
    )
