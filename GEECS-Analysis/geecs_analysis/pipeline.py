"""Pure pipeline execution over caller-supplied frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

from geecs_analysis.registry import definition
from geecs_analysis.specs import Pipeline

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


def apply_pipeline(frame: Frame, pipeline: Pipeline) -> Frame:
    """Validate dimensions and fold steps without modifying the input frame."""
    for spec in pipeline.steps:
        if frame.data.ndim not in definition(spec).ndim:
            raise ValueError(f"{spec.step} does not support {frame.data.ndim}D frames")
    for spec in pipeline.steps:
        frame = definition(spec).function(frame, spec)
    return frame
