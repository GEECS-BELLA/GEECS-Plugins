"""Undo lens vignetting with a radial polynomial in full-sensor coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class VignetteSpec(StepSpec):
    """Divide by ``max(vgnt4·r⁴ + vgnt2·r² + vgnt0, min_model_value)``.

    ``r`` is the distance from the sensor centre, in the MATLAB-era
    convention ``pixel_1_based - full/2 + 0.5``. Pixels are the frame's own
    ``px`` axis values, so the map is the same before or after ``roi``.
    """

    step: Literal["vignette"] = "vignette"
    full_width: int = Field(..., gt=0, description="Full sensor width in pixels.")
    full_height: int = Field(..., gt=0, description="Full sensor height in pixels.")
    x_offset: int = Field(
        0,
        ge=0,
        description="X offset of the saved frame within the sensor (the camera's "
        "acquisition window, not an analysis ROI).",
    )
    y_offset: int = Field(
        0, ge=0, description="Y offset of the saved frame within the sensor."
    )
    vgnt4: float = Field(0.0, description="4th-order radial coefficient.")
    vgnt2: float = Field(0.0, description="2nd-order radial coefficient.")
    vgnt0: float = Field(1.0, description="0th-order (centre) coefficient.")
    min_model_value: float = Field(
        1e-9, gt=0.0, description="Floor on the model before it is inverted."
    )


@step(VignetteSpec, ndim={2})
def vignette(frame: Frame, spec: VignetteSpec) -> Frame:
    """Multiply by the reciprocal radial model, as the legacy map did."""
    import numpy as np

    if any(axis.unit != "px" for axis in frame.axes):
        raise ValueError("vignette needs pixel (px) axes")
    rows, cols = (np.asarray(axis.values, dtype=float) for axis in frame.axes)
    x_centered = spec.x_offset + cols + 1.0 - (spec.full_width / 2.0) + 0.5
    y_centered = spec.y_offset + rows + 1.0 - (spec.full_height / 2.0) + 0.5
    xx, yy = np.meshgrid(x_centered, y_centered)
    radius = np.sqrt(xx**2 + yy**2)
    attenuation = spec.vgnt4 * radius**4 + spec.vgnt2 * radius**2 + spec.vgnt0
    attenuation = np.maximum(attenuation, spec.min_model_value)
    return frame.replace(data=frame.data * (1.0 / attenuation))
