"""Mask one camera fiducial using the legacy pixel-grid rasterization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame

SampleIndex = Annotated[int, Field(ge=0, strict=True)]
PixelCount = Annotated[int, Field(gt=0, strict=True)]


class CrosshairMaskSpec(StepSpec):
    """Mask one cross on local sample indices, with center in (y, x) order.

    Integer half-widths and OpenCV rotation retain the v2 rasterization,
    including its empty bars for thickness one. Coordinates are local to the
    current cropped array; this operation never changes its axis grid.
    """

    step: Literal["crosshair_mask"] = "crosshair_mask"
    center: tuple[SampleIndex, SampleIndex]
    width: PixelCount
    height: PixelCount
    thickness: PixelCount
    angle: float = 0.0
    value: float = 0.0


@step(CrosshairMaskSpec, ndim={2})
def crosshair_mask(frame: Frame, spec: CrosshairMaskSpec) -> Frame:
    """Rasterize and apply one cross without mutating samples or coordinates."""
    import numpy as np

    rows, columns = frame.data.shape
    cy, cx = spec.center
    half_width, half_height = spec.width // 2, spec.height // 2
    half_thickness = spec.thickness // 2
    mask = np.zeros(frame.data.shape, dtype=np.float64)
    rectangles = (
        (
            max(0, cy - half_thickness),
            min(rows, cy + half_thickness),
            max(0, cx - half_width),
            min(columns, cx + half_width),
        ),
        (
            max(0, cy - half_height),
            min(rows, cy + half_height),
            max(0, cx - half_thickness),
            min(columns, cx + half_thickness),
        ),
    )
    for y0, y1, x0, x1 in rectangles:
        if y1 > y0 and x1 > x0:
            mask[y0:y1, x0:x1] = 1.0
    if abs(spec.angle) > 1e-6:
        import cv2

        matrix = cv2.getRotationMatrix2D((cx, cy), spec.angle, 1.0)
        mask = cv2.warpAffine(
            mask,
            matrix,
            (columns, rows),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask = (mask > 0.5).astype(np.float64)
    # Preserve v2 arithmetic (including NaN/Inf propagation at masked pixels).
    data = frame.data * (1 - mask) + spec.value * mask
    return frame.replace(data=data)
