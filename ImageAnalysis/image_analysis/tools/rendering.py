"""Image rendering utilities.

Provides a helper function to display 2‑D image data with Matplotlib,
including optional overlay of analysis results, line‑out extraction, and
customizable styling. The implementation follows NumPy‑style docstring
conventions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def base_render_image(
    image: np.ndarray,
    analysis_results_dict: Optional[dict[str, Union[float, int]]] = None,
    input_params_dict: Optional[dict[str, Union[float, int, str]]] = None,
    lineouts: Optional[List[np.ndarray]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "plasma",
    figsize: Tuple[float, float] = (4, 4),
    dpi: int = 150,
    ax: Optional[plt.Axes] = None,
) -> Tuple[Figure, Axes]:
    """Render an image with optional analysis overlays.

    Parameters
    ----------
    image : np.ndarray
        2‑D array representing the image to display.
    analysis_results_dict : dict, optional
        Mapping of result names to numeric values that can be displayed in a
        caption or used for further annotation. Keys are ignored by the
        rendering routine; the dictionary is retained for downstream use.
    input_params_dict : dict, optional
        Mapping of input parameter names to values. Like
        ``analysis_results_dict`` this information is not plotted directly but
        can be attached to the figure for reference.
    lineouts : list of np.ndarray, optional
        List of 1‑D intensity profiles extracted from the image. Not plotted
        by this function but kept for possible further processing.
    vmin, vmax : float, optional
        Minimum and maximum data values for colormap scaling. If ``None`` the
        defaults of :func:`matplotlib.pyplot.imshow` are used.
    cmap : str, default ``"plasma"``
        Name of the Matplotlib colormap to apply.
    figsize : tuple of float, default ``(4, 4)``
        Size of the created figure in inches (width, height). Ignored when an
        existing ``ax`` is supplied.
    dpi : int, default ``150``
        Resolution of the figure in dots per inch.
    ax : matplotlib.axes.Axes, optional
        Existing Axes object to draw on. If omitted a new Figure and Axes are
        created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the rendered image.
    ax : matplotlib.axes.Axes
        The Axes object on which the image was plotted.

    Notes
    -----
    The function creates a colorbar that is automatically sized to match the
    image. When an external ``ax`` is provided the caller is responsible for
    managing the Figure and any associated colorbars.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
        created_ax = True
    else:
        fig = ax.figure
        created_ax = False

    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)

    # Styling
    axis_fontsize = 10
    tick_fontsize = 9
    colorbar_fontsize = 9

    ax.set_xlabel("X Pixels", fontsize=axis_fontsize)
    ax.set_ylabel("Y Pixels", fontsize=axis_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    if created_ax:
        cbar = fig.colorbar(im, ax=ax, shrink=0.65)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

    return fig, ax
