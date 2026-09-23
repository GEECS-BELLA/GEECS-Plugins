"""Subtract a resolved in-memory background, without discovering or loading it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class BackgroundFrameSpec(StepSpec):
    """Name a loaded background and its coordinate-alignment convention.

    Axis alignment requires the same coordinates and units. Sample alignment
    subtracts by array index, for legacy files with no coordinate metadata;
    it still requires identical shapes and never broadcasts or resamples.
    """

    step: Literal["background_frame"] = "background_frame"
    source: str = Field(min_length=1)
    alignment: Literal["axes", "samples"] = "axes"


@step(BackgroundFrameSpec, ndim={1, 2}, input_field="source")
def background_frame(
    frame: Frame, spec: BackgroundFrameSpec, background: Frame
) -> Frame:
    """Subtract owned samples, retaining negative values, axes and provenance."""
    import numpy as np

    if frame.data.shape != background.data.shape:
        raise ValueError("Background shape must match input shape")
    if spec.alignment == "axes":
        if frame.unit != background.unit:
            raise ValueError("Background signal units must match input units")
        for axis, other in zip(frame.axes, background.axes, strict=True):
            if axis.unit != other.unit or not np.array_equal(axis.values, other.values):
                raise ValueError(
                    "Background axes must match input coordinates and units"
                )
    return frame.replace(data=frame.data - background.data)
