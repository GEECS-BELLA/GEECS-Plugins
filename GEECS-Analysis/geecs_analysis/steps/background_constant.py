"""Subtract a constant level from all samples."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class BackgroundConstantSpec(StepSpec):
    """Subtract a constant level from all samples."""

    step: Literal["background_constant"] = "background_constant"
    level: float = Field(..., description="Level subtracted from every sample.")


@step(BackgroundConstantSpec, ndim={1, 2})
def background_constant(frame: Frame, spec: BackgroundConstantSpec) -> Frame:
    """Subtract a constant level from all samples."""
    return frame.replace(data=frame.data - spec.level)
