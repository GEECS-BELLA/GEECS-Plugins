import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Any, Union


def polyfit_label(fit_pars: Union[list, np.ndarray], res: int = 2, var_str: str = 'x',
                  sep_str: str = ' ', latex: bool = True) -> str:
    label: str = ''
    if isinstance(fit_pars, np.ndarray):
        order = fit_pars.size - 1
        fit_pars = list(fit_pars)
    else:
        order = len(fit_pars) - 1

    res_str = f'.{res}f'
    for o, par in enumerate(fit_pars):
        if par < 0:
            sign_str = '- '
        elif (par >= 0) and (o > 0):
            sign_str = '+ '
        else:
            sign_str = ''

        order_str = f'^{order - o}'
        if latex:
            order_str = f'${order_str}$'

        if o > 0:
            label += ' '

        if order - o == 0:
            label += f"{sign_str}{('{:' + res_str + '}').format(np.abs(par))}"
        elif order - o == 1:
            label += f"{sign_str}{('{:' + res_str + '}').format(np.abs(par))}{sep_str}{var_str}"
        else:
            label += f"{sign_str}{('{:' + res_str + '}').format(np.abs(par))}{sep_str}{var_str}" + order_str

    return label


def show_one(image: np.ndarray,
             figure_size: tuple[float, float] = (6.4, 4.8),
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
    plt.figure(figsize=figure_size)
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


if __name__ == '__main__':
    # pars = np.array([-2.3456, -1.2345, -6.789])
    pars = [-1.2345, -6.789]
    print(polyfit_label(pars, res=2, var_str='x', sep_str=' ', latex=False))
