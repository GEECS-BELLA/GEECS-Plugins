"""The scan's averaged frame, drawn exactly as a single frame is."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from geecs_schemas.analysis.recipe import AverageSummary

from geecs_analysis.registry import summary
from geecs_analysis.render import single

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from geecs_analysis.measurement import Measurement
    from geecs_analysis.render.specs import FigureSpec


@summary(AverageSummary, consumes="average", filename="average_processed_visual")
def average(
    results: Sequence[Measurement],
    positions: Sequence[float | None],
    label: str,
    options: AverageSummary,
    figure: FigureSpec,
) -> Figure:
    """Draw the one averaged measurement with the recipe's per-frame draw."""
    from geecs_analysis.render import RenderError

    if len(results) != 1:
        raise RenderError("The average summary draws exactly one measurement")
    return single(results[0], figure)
