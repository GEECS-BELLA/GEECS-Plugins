"""Image rendering utilities.

Provides composable functions to display 2‑D image data with Matplotlib,
including base rendering and optional overlay components. All functions work
with ImageAnalyzerResult objects. The implementation follows NumPy‑style
docstring conventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

from image_analysis.types import ImageAnalyzerResult


def base_render_image(
    result: ImageAnalyzerResult,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "plasma",
    figsize: Tuple[float, float] = (4, 4),
    dpi: int = 150,
    ax: Optional[plt.Axes] = None,
) -> Tuple[Figure, Axes]:
    """Render a 2D image from ImageAnalyzerResult.

    This is the base rendering function that displays only the processed image.
     Use overlay functions like `add_xy_projections` or
    `add_centroid_marker` to add additional visual elements.

    Raises
    ------
    ValueError
        If result.data_type is not "2d" or processed_image is None.
    """
    # Validate input
    if result.data_type != "2d":
        raise ValueError(
            f"base_render_image requires data_type='2d', got '{result.data_type}'"
        )

    if result.processed_image is None:
        raise ValueError("processed_image cannot be None for 2d data_type")

    # Create figure/axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
        created_ax = True
    else:
        fig = ax.figure
        created_ax = False

    # Render the image
    im = ax.imshow(result.processed_image, cmap=cmap, vmin=vmin, vmax=vmax)

    # Styling
    axis_fontsize = 10
    tick_fontsize = 9
    colorbar_fontsize = 9

    ax.set_xlabel("X Pixels", fontsize=axis_fontsize)
    ax.set_ylabel("Y Pixels", fontsize=axis_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    # Add colorbar if we created the axes
    if created_ax:
        cbar = fig.colorbar(im, ax=ax, shrink=0.65)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

    return fig, ax


def add_xy_projections(
    ax: Axes,
    result: ImageAnalyzerResult,
    scale: float = 0.2,
    color_h: str = "cyan",
    color_v: str = "magenta",
    linewidth: float = 1.0,
    alpha: float = 0.8,
) -> None:
    """Add horizontal and vertical projection overlays to an image plot.

    Overlays normalized projections along the edges of the image. Projections
    are extracted from result.render_data['xy_projections'] using the
    get_xy_projections() method.

    Notes
    -----
    Projections are clipped to non-negative values and normalized before plotting.
    If no projections exist in result or they are empty, this function does nothing.

    """
    projections = result.get_xy_projections()
    if projections is None:
        return

    horiz, vert = projections
    img_h, img_w = result.processed_image.shape

    # Clip to positive values
    horiz = np.clip(horiz, 0, None)
    vert = np.clip(vert, 0, None)

    if len(horiz) == 0 or len(vert) == 0:
        return

    # Normalize projections for overlay
    max_h = np.max(horiz)
    max_v = np.max(vert)

    if max_h > 0:
        horiz_norm = horiz / max_h * img_h * scale
    else:
        horiz_norm = horiz

    if max_v > 0:
        vert_norm = vert / max_v * img_w * scale
    else:
        vert_norm = vert

    # Plot projections
    ax.plot(
        np.arange(len(horiz)),
        img_h - horiz_norm,
        color=color_h,
        lw=linewidth,
        label="Horizontal Projection",
        alpha=alpha,
    )
    ax.plot(
        vert_norm,
        np.arange(len(vert)),
        color=color_v,
        lw=linewidth,
        label="Vertical Projection",
        alpha=alpha,
    )


def add_marker(
    ax: Axes,
    position: Tuple[float, float],
    marker: str = "o",
    color: str = "red",
    size: float = 5,
    edge_color: Optional[str] = None,
    edge_width: float = 0,
    alpha: float = 1.0,
    label: Optional[str] = None,
) -> None:
    """Add a marker/dot at the specified position."""
    ax.plot(
        position[0],
        position[1],
        marker=marker,
        color=color,
        markersize=size,
        markeredgecolor=edge_color if edge_color else color,
        markeredgewidth=edge_width,
        alpha=alpha,
        label=label,
    )
