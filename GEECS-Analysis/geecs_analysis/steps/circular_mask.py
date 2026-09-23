"""Circular image masks in sample-index or physical-axis coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class CircularMaskSpec(StepSpec):
    """Replace pixels inside or outside a circle, retaining image geometry.

    Center follows numpy dimension order (y, x). Index coordinates are local
    to the current samples; axis coordinates retain crop/calibration offsets.
    """

    step: Literal["circular_mask"] = "circular_mask"
    center: tuple[float, float]
    radius: float = Field(gt=0)
    units: Literal["index", "axis"] = "index"
    mask_outside: bool = True
    value: float = 0.0


@step(CircularMaskSpec, ndim={2})
def circular_mask(frame: Frame, spec: CircularMaskSpec) -> Frame:
    """Mask samples without changing their axes, metadata or input buffer."""
    import numpy as np

    y, x = (
        (np.arange(frame.data.shape[0]), np.arange(frame.data.shape[1]))
        if spec.units == "index"
        else (frame.axes[0].values, frame.axes[1].values)
    )
    # Keep the legacy sqrt comparison, including exact circle-boundary pixels.
    distance = np.sqrt(
        (y[:, None] - spec.center[0]) ** 2 + (x[None, :] - spec.center[1]) ** 2
    )
    mask = distance > spec.radius if spec.mask_outside else distance <= spec.radius
    data = frame.data.copy()
    data[mask] = spec.value
    return frame.replace(data=data)
