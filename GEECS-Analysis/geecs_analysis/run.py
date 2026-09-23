"""Write-free execution shared by interactive and live consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from geecs_analysis.pipeline import apply_pipeline
from geecs_analysis.registry import measure_definition
from geecs_analysis.specs import Analysis

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement


def analyze(frame: Frame, recipe: Analysis) -> Measurement:
    """Process and measure one frame without scan state, file access or rendering."""
    definition = measure_definition(recipe.measure)
    if frame.data.ndim not in definition.ndim:
        raise ValueError(
            f"{recipe.measure.kind} does not support {frame.data.ndim}D frames"
        )
    return definition.function(apply_pipeline(frame, recipe), recipe.measure)
