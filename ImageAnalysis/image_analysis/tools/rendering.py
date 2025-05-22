# image_analysis/tools/rendering.py

from typing import Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt

def base_render_image(
    image: np.ndarray,
    analysis_results_dict: Optional[dict[str, Union[float, int]]] = None,
    input_params_dict: Optional[dict[str, Union[float, int, str]]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'plasma',
    figsize: Tuple[float, float] = (4, 4),
    dpi: int = 150,
    ax: Optional[plt.Axes] = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Base image rendering with optional styling, axis, and colorbar.

    Returns:
        (fig, ax): Matplotlib Figure and Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
        created_ax = True
    else:
        fig = ax.figure
        created_ax = False

    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)

    # --- Styling ---
    axis_fontsize = 10
    tick_fontsize = 9
    colorbar_fontsize = 9

    ax.set_xlabel("X Pixels", fontsize=axis_fontsize)
    ax.set_ylabel("Y Pixels", fontsize=axis_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    if created_ax:
        cbar = fig.colorbar(im, ax=ax, shrink=0.65)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

    return fig, ax