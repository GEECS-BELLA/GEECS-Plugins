"""Translate v2 ``scan.renderer`` options into the per-frame draw and summary kinds.

The v2 document has no ``figure:`` or ``summaries:``; its ``RendererOptions``
fold both into one option set with data-dependent palette rules. This module
expresses those rules as a ``FigureSpec`` (static keywords, a centred norm
for the diverging image mode) and the fixed v2 summary pair (grid or
waterfall, plus the average), so a v2 document draws through the same
kinds a v3 recipe does. The ``*_v2`` entry points draw one product with
those translations; nothing here writes files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from geecs_schemas.analysis.recipe import (
    AverageSummary,
    ImageGridSummary,
    WaterfallSummary,
)

from geecs_analysis.render import single
from geecs_analysis.render.specs import FigureSpec
from geecs_analysis.summaries.image_grid import image_grid
from geecs_analysis.summaries.waterfall import waterfall

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from geecs_schemas.analysis.renderer import RendererOptions
    from geecs_analysis.measurement import Measurement

#: The v2 grid's panel size when ``figsize`` is unset.
V2_PANEL_SIZE = (6.0, 6.0)


def _axes(options: RendererOptions) -> dict[str, str]:
    return {
        key: getattr(options, key)
        for key in ("xlabel", "ylabel")
        if getattr(options, key) is not None
    }


def image_palette_v2(options: RendererOptions) -> dict:
    """The v2 image palette rule as static keywords.

    ``sequential`` (the default) runs from zero to the data maximum;
    ``diverging`` is symmetric about zero (a centred norm autoscaled over
    the drawn samples, the grid's included); ``auto`` and ``custom`` use
    the given limits and autoscale the rest.
    """
    mode = options.colormap_mode or "sequential"
    cmap = options.cmap or "plasma"
    if mode == "diverging":
        from matplotlib.colors import CenteredNorm

        return {"norm": CenteredNorm(vcenter=0), "cmap": cmap}
    palette = {"cmap": cmap}
    vmin = 0 if mode == "sequential" else options.vmin
    if vmin is not None:
        palette["vmin"] = vmin
    if options.vmax is not None:
        palette["vmax"] = options.vmax
    return palette


def figure_v2(
    options: RendererOptions, *, line: bool, title: str | None = None
) -> FigureSpec:
    """The v2 renderer's per-frame draw: palette, labels, canvas and dpi."""
    side = options.figsize_inches or 4
    palette = {} if line else image_palette_v2(options)
    # A trace's colorbar (the waterfall's) is labelled by its signal unless
    # the document says otherwise; an image's defaulted to "Intensity".
    colorbar = {"label": options.colorbar_label} if options.colorbar_label else {}
    if not line:
        colorbar = {"label": options.colorbar_label or "Intensity"}
    return FigureSpec(
        imshow=palette,
        pcolormesh=palette,
        axes={**_axes(options), **({"title": title} if title else {})},
        colorbar=colorbar,
        fig={"figsize": (8, 6) if line else (side, side), "dpi": options.dpi or 150},
    )


def summaries_v2(
    options: RendererOptions, *, line: bool
) -> tuple[WaterfallSummary | ImageGridSummary, AverageSummary]:
    """The fixed v2 summary pair: waterfall or grid, then the average."""
    if line:
        return (
            WaterfallSummary(
                sort_key=options.waterfall_sort_key,
                sort_sigma=(
                    options.waterfall_sort_sigma
                    if options.waterfall_sort_sigma is not None
                    else 3.0
                ),
                sort_bounds=options.waterfall_sort_bounds,
                even_spacing=options.waterfall_even_y_spacing,
                scale=options.colormap_mode or "auto",
                cmap=options.cmap,
                vmin=options.vmin,
                vmax=options.vmax,
            ),
            AverageSummary(),
        )
    return (
        ImageGridSummary(panel_size=options.figsize or V2_PANEL_SIZE),
        AverageSummary(),
    )


def single_v2(
    result: Measurement, options: RendererOptions, *, title: str | None = None
) -> Figure:
    """Render a v2 single product with configured labels and color limits."""
    line = result.frame.data.ndim == 1
    return single(result, figure_v2(options, line=line, title=title))


def image_grid_v2(
    results: Sequence[Measurement],
    positions: Sequence[float | None],
    options: RendererOptions,
    *,
    label: str = "",
) -> Figure:
    """Draw v2 bin images with one palette computed across all panels.

    ``label`` names the scanned parameter above the grid, as the legacy
    renderer's ``Scan parameter: …`` suptitle did; empty draws no title.
    """
    grid, _ = summaries_v2(options, line=False)
    return image_grid(results, positions, label, grid, figure_v2(options, line=False))


def waterfall_v2(
    results: Sequence[Measurement],
    positions: Sequence[float],
    label: str,
    options: RendererOptions,
) -> Figure:
    """Draw the v2 waterfall's index-wise stack, using the first trace's x axis."""
    stack, _ = summaries_v2(options, line=True)
    return waterfall(results, positions, label, stack, figure_v2(options, line=True))
