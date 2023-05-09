import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Any, Union


def show_one(image: np.ndarray,
             size_factor: float = 1.,
             x_lim: Optional[tuple[float, float]] = None,
             y_lim: Optional[tuple[float, float]] = None,
             colormap: Any = plt.cm.hot,
             markers_ij: Optional[Union[list[tuple[int, int]], np.ndarray]] = None,
             markers_color: str = 'w.',
             hide_ticks: bool = True,
             show_colorbar: bool = True,
             show_contours: bool = True,
             contours: int = 5,
             contours_levels: Optional[Union[np.ndarray, list[float]]] = None,
             contours_colors: Any = 'black',
             contours_labels: bool = True,
             contours_fontsize: int = 8,
             show: bool = True,
             block_execution: bool = True):
    plt.figure(figsize=(6.4 * size_factor, 4.8 * size_factor))
    ax = plt.subplot(111)
    im = ax.imshow(image, cmap=colormap, aspect='equal', origin='upper')

    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])

    if show_contours:
        if contours_levels:
            contours = plt.contour(image, levels=contours_levels, colors=contours_colors)
        else:
            contours = plt.contour(image, levels=contours, colors=contours_colors)

        if contours_labels:
            plt.clabel(contours, inline=True, fontsize=contours_fontsize)

    if hide_ticks:
        plt.xticks([])
        plt.yticks([])

    if markers_ij.any():
        for marker in markers_ij:
            plt.plot(marker[1], marker[0], markers_color)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.2, pad=0.1)
        plt.colorbar(im, cax=cax)

    if show:
        plt.show(block=block_execution)

    return ax, im
