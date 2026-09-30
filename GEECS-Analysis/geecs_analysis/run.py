"""Write-free execution shared by interactive and live consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from geecs_analysis.pipeline import (
    apply_measure,
    apply_pipeline,
    bind_inputs,
    process_measure_input,
)
from geecs_analysis.registry import measure_definition
from geecs_analysis.specs import Analysis

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement


def analyze(
    frame: Frame, recipe: Analysis, *, inputs: Mapping[str, object] | None = None
) -> Measurement:
    """Process and measure one frame without scan state, file access or rendering.

    ``inputs`` holds the frames the steps bind and, for a measure that names
    a service (``frog``), the host's collaborator under that name. A
    measure's frame input (the ``haso`` reference) is processed through the
    same steps as ``frame`` before the measure sees it.
    """
    definition = measure_definition(recipe.measure)
    if frame.data.ndim not in definition.ndim:
        raise ValueError(
            f"{recipe.measure.kind} does not support {frame.data.ndim}D frames"
        )
    bound = bind_inputs(recipe.steps, inputs, measure=recipe.measure)
    measured = process_measure_input(
        recipe.measure, bound, lambda other: apply_pipeline(other, recipe, inputs=bound)
    )
    return apply_measure(
        apply_pipeline(frame, recipe, inputs=bound), recipe.measure, inputs=measured
    )
