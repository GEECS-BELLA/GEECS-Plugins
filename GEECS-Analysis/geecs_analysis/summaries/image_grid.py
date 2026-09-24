"""One panel per bin with one shared colour scale."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from geecs_schemas.analysis.recipe import ImageGridSummary

from geecs_analysis.registry import summary
from geecs_analysis.render import RenderError, image_grid as draw_grid

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from geecs_analysis.measurement import Measurement
    from geecs_analysis.render.specs import FigureSpec

#: Panel (width, height) in inches when the recipe does not say.
DEFAULT_PANEL_SIZE = (4.0, 3.5)


@summary(ImageGridSummary, consumes="panels", filename="averaged_image_grid")
def image_grid(
    results: Sequence[Measurement],
    positions: Sequence[float | None],
    label: str,
    options: ImageGridSummary,
    figure: FigureSpec,
) -> Figure:
    """Draw the bin images through the per-frame draw, titled by position.

    The palette comes from ``figure`` (``imshow`` / ``pcolormesh``); limits
    it leaves unset autoscale across every panel. The canvas is the panel
    size times the grid; ``figure.fig`` supplies everything else (dpi).
    ``label`` names the scanned parameter above the grid; empty draws none.
    """
    import math

    if not results or len(results) != len(positions):
        raise RenderError("Image grid requires one position per measurement")
    columns = min(options.columns or math.ceil(math.sqrt(len(results))), len(results))
    rows = math.ceil(len(results) / columns)
    width, height = options.panel_size or DEFAULT_PANEL_SIZE
    style = figure.model_copy(
        update={"fig": {**figure.fig, "figsize": (columns * width, rows * height)}}
    )
    fig = draw_grid(
        results,
        titles=[
            f"{p:.2f}" if p is not None else str(i + 1) for i, p in enumerate(positions)
        ],
        columns=columns,
        style=style,
    )
    if label:
        fig.suptitle(f"Scan parameter: {label}", fontsize=12)
    return fig
