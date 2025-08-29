"""
Minimal plotting utilities for binned scan data.

This module provides helper functions to plot aggregated (binned) scalar data
from scans using a frozen schema. It supports both single and multi-series
errorbar plots with optional asymmetric errors and index-based x-axes.

Functions
---------
_get_center_and_err
    Extract center values and optional asymmetric errors for a column.
_index_to_numeric
    Convert an index to numeric values, handling IntervalIndex specially.
plot_binned
    Plot a single y-series versus x from binned data.
plot_binned_multi
    Overlay multiple y-series versus the same x.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_center_and_err(
    binned: pd.DataFrame, col: str
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract center values and optional errors for a column.

    This function enforces a frozen schema:
    - For MultiIndex columns, expects subcolumns 'center', and optionally
      'err_low' and 'err_high'.
    - For flat columns, returns values directly without errors.

    Parameters
    ----------
    binned : pandas.DataFrame
        DataFrame of binned values. May have a MultiIndex or flat columns.
    col : str
        Column name to extract values for.

    Returns
    -------
    y : numpy.ndarray
        Center values for the requested column.
    yerr : numpy.ndarray or None
        Asymmetric errors with shape (2, N) if both 'err_low' and 'err_high'
        are present. Otherwise None.

    Raises
    ------
    KeyError
        If the requested column does not exist in the expected schema.
    """
    if isinstance(binned.columns, pd.MultiIndex):
        cols = binned.columns
        if (col, "center") in cols:
            y = binned[(col, "center")].to_numpy()
            if (col, "err_low") in cols and (col, "err_high") in cols:
                yerr = np.vstack(
                    [
                        binned[(col, "err_low")].to_numpy(),
                        binned[(col, "err_high")].to_numpy(),
                    ]
                )
            else:
                yerr = None
            return y, yerr
        # no fallback guessing when schema is frozen
        raise KeyError(f"'{col}' has no 'center' subcolumn in binned_scalars.")
    else:
        if col in binned.columns:
            return binned[col].to_numpy(), None
        raise KeyError(f"Column '{col}' not found in binned_scalars.")


def _index_to_numeric(idx: pd.Index) -> np.ndarray:
    """
    Convert a pandas Index into numeric values.

    - If the index is an IntervalIndex, use midpoints.
    - Otherwise, attempt direct float conversion.
    - On failure, fallback to sequential integers.

    Parameters
    ----------
    idx : pandas.Index
        Input index to convert.

    Returns
    -------
    values : numpy.ndarray
        Numeric representation of the index.
    """
    if isinstance(idx, pd.IntervalIndex):
        return ((idx.left.astype(float) + idx.right.astype(float)) / 2.0).to_numpy()
    try:
        return idx.to_numpy(dtype=float, copy=False)
    except Exception:
        return np.arange(len(idx), dtype=float)


def plot_binned(
    binned: pd.DataFrame,
    x_col: Optional[str],
    y_col: str,
    *,
    use_index_as_x: bool = False,
    ax: Optional[plt.Axes] = None,
    marker: str = "o",
    linestyle: str = "-",
    label: Optional[str] = None,
    xscale: str = "linear",
    yscale: str = "linear",
    grid: bool = True,
) -> plt.Axes:
    """
    Plot a single y-series versus x from binned data.

    Parameters
    ----------
    binned : pandas.DataFrame
        DataFrame of binned scalar values, with frozen schema.
    x_col : str or None
        Column to use for the x-axis. If None or `use_index_as_x=True`,
        use the DataFrame index.
    y_col : str
        Column to plot as y-axis values. Must exist as 'center' in MultiIndex
        schema or as a flat column.
    use_index_as_x : bool, default=False
        If True, use the DataFrame index for x-values regardless of `x_col`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, create a new figure and axes.
    marker : str, default="o"
        Marker style for the plot.
    linestyle : str, default="-"
        Line style for the plot.
    label : str, optional
        Label for the plotted series. If provided, a legend is added.
    xscale : {"linear", "log"}, default="linear"
        X-axis scale type.
    yscale : {"linear", "log"}, default="linear"
        Y-axis scale type.
    grid : bool, default=True
        Whether to draw a grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes containing the plot.

    Raises
    ------
    KeyError
        If requested columns are not found in the schema.
    """
    # y (required)
    y, yerr = _get_center_and_err(binned, y_col)

    # x
    if use_index_as_x or x_col is None:
        x = _index_to_numeric(binned.index)
        xerr = None
        xlabel = binned.index.name or "bin"
    else:
        x, xerr = _get_center_and_err(binned, x_col)
        xlabel = x_col

    # mask + sort
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if xerr is not None:
        xerr = xerr[:, mask]
    if yerr is not None:
        yerr = yerr[:, mask]

    order = np.argsort(x)
    x, y = x[order], y[order]
    if xerr is not None:
        xerr = xerr[:, order]
    if yerr is not None:
        yerr = yerr[:, order]

    if ax is None:
        _, ax = plt.subplots()
    ax.errorbar(
        x, y, xerr=xerr, yerr=yerr, fmt=marker, linestyle=linestyle, label=label
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_col)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if grid:
        ax.grid(True, alpha=0.3)
    if label:
        ax.legend()
    return ax


def plot_binned_multi(
    binned: pd.DataFrame,
    x_col: Optional[str],
    y_cols: Sequence[str],
    *,
    use_index_as_x: bool = False,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    markers: Optional[Sequence[str]] = None,
    linestyles: Optional[Sequence[str]] = None,
    xscale: str = "linear",
    yscale: str = "linear",
    grid: bool = True,
) -> plt.Axes:
    """
    Overlay multiple y-series versus the same x-axis.

    Parameters
    ----------
    binned : pandas.DataFrame
        DataFrame of binned scalar values, with frozen schema.
    x_col : str or None
        Column to use for the x-axis. If None or `use_index_as_x=True`,
        use the DataFrame index.
    y_cols : sequence of str
        List of column names to plot as y-axis values.
    use_index_as_x : bool, default=False
        If True, use the DataFrame index for x-values regardless of `x_col`.
    labels : sequence of str, optional
        Labels for the plotted series. Defaults to `y_cols`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, create a new figure and axes.
    markers : sequence of str, optional
        Markers for each series. Defaults to a set of common markers.
    linestyles : sequence of str, optional
        Linestyles for each series. Defaults to all solid lines.
    xscale : {"linear", "log"}, default="linear"
        X-axis scale type.
    yscale : {"linear", "log"}, default="linear"
        Y-axis scale type.
    grid : bool, default=True
        Whether to draw a grid.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes containing the plot.

    Raises
    ------
    KeyError
        If requested columns are not found in the schema.
    """
    if ax is None:
        _, ax = plt.subplots()
    if labels is None:
        labels = y_cols
    if markers is None:
        markers = ["o", "s", "D", "^", "v", "P", "X"]
    if linestyles is None:
        linestyles = ["-"] * len(y_cols)

    # common x
    if use_index_as_x or x_col is None:
        x = _index_to_numeric(binned.index)
        xerr = None
        xlabel = binned.index.name or "bin"
    else:
        x, xerr = _get_center_and_err(binned, x_col)
        xlabel = x_col
    order = np.argsort(x)

    for i, (yc, lab) in enumerate(zip(y_cols, labels)):
        y, yerr = _get_center_and_err(binned, yc)
        m = markers[i % len(markers)]
        ls = linestyles[i % len(linestyles)]
        ax.errorbar(
            x[order],
            y[order],
            xerr=None if xerr is None else xerr[:, order],
            yerr=None if yerr is None else yerr[:, order],
            fmt=m,
            linestyle=ls,
            label=lab,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(", ".join(y_cols) if len(y_cols) == 1 else "value")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if grid:
        ax.grid(True, alpha=0.3)
    ax.legend()
    return ax
