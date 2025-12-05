"""Image rendering utilities.

Provides composable functions to display 2‑D image data with Matplotlib,
including base rendering and optional overlay components. All functions work
with ImageAnalyzerResult objects. The implementation follows NumPy‑style
docstring conventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Literal

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


def add_line_overlay(
    ax: Axes,
    lineout: np.ndarray,
    direction: Literal["horizontal", "vertical", "h", "v"],
    scale: float = 0.3,
    offset: float = 0.0,
    color: str = "cyan",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    normalize: bool = True,
    clip_positive: bool = False,
    label: Optional[str] = None,
) -> None:
    """Add a line overlay in horizontal or vertical direction.

    This is the generic function for overlaying 1D data on a 2D image plot.
    It handles both horizontal and vertical orientations and is used internally
    by convenience functions like `add_xy_projections`.

    Common use cases:
    - Displaying projections/sums along image axes
    - Showing fitted profiles or analysis results
    - Overlaying derived quantities (e.g., bowtie fit weights)
    - Adding multiple reference lines at different positions

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    lineout : numpy.ndarray
        1D array to plot as a line overlay
    direction : {'horizontal', 'vertical', 'h', 'v'}
        Direction of the line overlay:
        - 'horizontal' or 'h': Plot along x-axis (varies with x, constant y base)
        - 'vertical' or 'v': Plot along y-axis (varies with y, constant x base)
    scale : float, default=0.3
        Fraction of image dimension (height for horizontal, width for vertical)
        to use for lineout amplitude scaling after normalization.
    offset : float, default=0.0
        Position offset in the perpendicular direction (pixels):
        - For horizontal: Y-coordinate offset (positive = down)
        - For vertical: X-coordinate offset (positive = right)
        Default of 0 places baseline at top (horizontal) or left (vertical).
    color : str, default="cyan"
        Line color
    linewidth : float, default=1.0
        Line width
    alpha : float, default=1.0
        Line transparency (0=transparent, 1=opaque)
    normalize : bool, default=True
        If True, normalize lineout to its maximum before scaling.
        If False, use lineout values as-is (useful for pre-normalized data).
    clip_positive : bool, default=False
        If True, clip lineout to non-negative values before processing.
        Useful for projection data that should not have negative values.
    label : str, optional
        Legend label for the line

    Notes
    -----
    - Lineout length should match the corresponding image dimension for alignment
    - For projections along image edges, use `add_xy_projections` convenience wrapper
    - Multiple lines can be added by calling this function repeatedly with different offsets

    Examples
    --------
    Add horizontal lineout at top of image:

    >>> add_line_overlay(ax, lineout, direction='horizontal', offset=0)

    Add vertical lineout on right side:

    >>> img_width = 512
    >>> add_line_overlay(ax, lineout, direction='vertical', offset=img_width)

    Add multiple horizontal lines at different heights:

    >>> for i, line in enumerate(lineouts):
    ...     add_line_overlay(ax, line, 'h', offset=i*50, color=colors[i])
    """
    if lineout is None or len(lineout) == 0:
        return

    # Normalize direction input
    dir_map = {"h": "horizontal", "v": "vertical"}
    direction = dir_map.get(direction, direction)

    if direction not in ["horizontal", "vertical"]:
        raise ValueError(
            f"direction must be 'horizontal', 'vertical', 'h', or 'v', got '{direction}'"
        )

    # Get image dimensions for scaling
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    img_height = ylim[0] - ylim[1]  # ylim is inverted for images
    img_width = xlim[1] - xlim[0]

    # Process lineout
    line_data = lineout.copy()

    # Clip to positive if requested
    if clip_positive:
        line_data = np.clip(line_data, 0, None)

    # Normalize if requested
    if normalize and np.max(line_data) > 0:
        line_data = line_data / np.max(line_data)

    # Plot based on direction
    if direction == "horizontal":
        # Horizontal line: x varies, y is offset - scaled amplitude
        # (subtract because y increases downward in image coordinates)
        x_vals = np.arange(len(line_data))
        scaled_amplitude = line_data * abs(img_height) * scale
        y_vals = offset - scaled_amplitude

        ax.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=10,
        )
    else:  # vertical
        # Vertical line: y varies, x is offset + scaled amplitude
        y_vals = np.arange(len(line_data))
        scaled_amplitude = line_data * abs(img_width) * scale
        x_vals = offset + scaled_amplitude

        ax.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=10,
        )


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

    This is a convenience wrapper around `add_line_overlay` that adds both
    horizontal and vertical projections along the edges of the image.
    Projections are extracted from result.render_data['xy_projections']
    using the get_xy_projections() method.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    result : ImageAnalyzerResult
        Result object containing projections in render_data
    scale : float, default=0.2
        Fraction of image dimension to use for projection amplitude
    color_h : str, default="cyan"
        Color for horizontal projection
    color_v : str, default="magenta"
        Color for vertical projection
    linewidth : float, default=1.0
        Line width for both projections
    alpha : float, default=0.8
        Transparency for both projections

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
    img_h = img_h - 1

    if len(horiz) == 0 or len(vert) == 0:
        return

    # Add horizontal projection along bottom edge
    add_line_overlay(
        ax=ax,
        lineout=horiz,
        direction="horizontal",
        scale=scale,
        offset=img_h,  # Place at bottom (remember y increases downward in image coords)
        color=color_h,
        linewidth=linewidth,
        alpha=alpha,
        normalize=True,
        clip_positive=True,
        label="Horizontal Projection",
    )

    # Add vertical projection along left edge
    add_line_overlay(
        ax=ax,
        lineout=vert,
        direction="vertical",
        scale=scale,
        offset=0,  # Place at left edge
        color=color_v,
        linewidth=linewidth,
        alpha=alpha,
        normalize=True,
        clip_positive=True,
        label="Vertical Projection",
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
