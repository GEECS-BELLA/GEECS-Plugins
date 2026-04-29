"""
Data cleaning utilities for tabular scan data.

This module provides reusable DataFrame-level cleaning helpers used by
dataset assembly and analysis workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# row filtering
# ---------------------------------------------------------------------------

_OPERATORS = {
    ">": lambda s, v: s > v,
    "<": lambda s, v: s < v,
    ">=": lambda s, v: s >= v,
    "<=": lambda s, v: s <= v,
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
}

RowFilterSpec = Tuple[str, str, Union[int, float]]


def apply_row_filters(
    df: pd.DataFrame,
    filters: List[RowFilterSpec],
) -> pd.DataFrame:
    """Filter rows using a sequence of ``(column, operator, value)`` conditions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to filter.
    filters : list of tuple[str, str, int | float]
        Conditions applied in order. Each tuple is
        ``(column_name, operator, threshold)`` where ``operator`` is one of
        ``">"``, ``"<"``, ``">="``, ``"<="``, ``"=="``, or ``"!="``.

    Returns
    -------
    pandas.DataFrame
        A filtered dataframe containing only rows that satisfy all conditions.

    Raises
    ------
    ValueError
        If any provided operator is not supported.
    """
    for column, operator, value in filters:
        fn = _OPERATORS.get(operator)
        if fn is None:
            raise ValueError(f"Unsupported filter operator: '{operator}'")
        df = df[fn(df[column], value)]
    return df


# ---------------------------------------------------------------------------
# outlier handling
# ---------------------------------------------------------------------------

OutlierMethod = Literal["clip", "nan"]


@dataclass
class OutlierConfig:
    """Configuration for outlier handling during dataset assembly.

    Attributes
    ----------
    method : ``"clip"`` | ``"nan"``
        ``"clip"`` removes entire rows outside the sigma band on each
        selected column (applied sequentially). ``"nan"`` replaces outlier
        values with ``NaN`` while keeping all rows.
    sigma : float
        Number of standard deviations from the column mean for the threshold.
    columns : list of str, optional
        Columns to apply outlier handling to. If ``None``, all numeric columns
        are used.
    """

    method: OutlierMethod = "nan"
    sigma: float = 5.0
    columns: Optional[List[str]] = None


def _get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Return names of columns with numeric dtypes in *df*.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.

    Returns
    -------
    list of str
        Column names selected by ``select_dtypes(include="number")``.
    """
    return df.select_dtypes(include="number").columns.tolist()


def _iter_outlier_bounds(
    df: pd.DataFrame,
    sigma: float,
    columns: Optional[Sequence[str]] = None,
) -> Iterator[Tuple[str, float, float]]:
    """
    Yield per-column sigma bounds for outlier detection.

    For each selected column, computes ``mean`` and ``std`` on *df* and
    yields ``(column, lower, upper)`` with ``lower = mean - sigma * std`` and
    ``upper = mean + sigma * std``. Columns with zero or NaN standard deviation
    are skipped (no bounds emitted).

    Parameters
    ----------
    df : pandas.DataFrame
        Data used to estimate mean and standard deviation per column.
    sigma : float
        Half-width of the band in units of standard deviation.
    columns : sequence of str, optional
        Columns to consider. If ``None``, all numeric columns in *df* are used.

    Yields
    ------
    tuple[str, float, float]
        ``(column_name, lower_bound, upper_bound)``.
    """
    cols = _get_numeric_columns(df) if columns is None else list(columns)

    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0 or np.isnan(std):
            continue
        lower = mean - sigma * std
        upper = mean + sigma * std

        yield col, lower, upper


def sigma_clip_frame(
    df: pd.DataFrame,
    sigma: float = 6.0,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Remove rows where any selected column lies outside a sigma band.

    For each column in turn, keeps only rows whose value lies within
    ``[mean - sigma * std, mean + sigma * std]`` for that column. Bounds are
    recomputed on the dataframe after each column's filter (so later columns
    use updated row statistics).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    sigma : float
        Number of standard deviations for the clipping threshold.
    columns : sequence of str, optional
        Columns to clip on. Defaults to all numeric columns.

    Returns
    -------
    pandas.DataFrame
        Filtered copy with outlier rows removed.

    See Also
    --------
    sigma_nan_frame : mask outliers with ``NaN`` instead of dropping rows.
    """
    out = df.copy()
    for col, lower, upper in _iter_outlier_bounds(out, sigma=sigma, columns=columns):
        out = out[(out[col] >= lower) & (out[col] <= upper)]
    return out


def sigma_nan_frame(
    df: pd.DataFrame,
    sigma: float = 6.0,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Replace values outside a sigma band with ``NaN``.

    Mean and standard deviation are computed once per column on the initial
    copy; outlier values are set to ``NaN`` without dropping rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    sigma : float
        Number of standard deviations for the outlier threshold.
    columns : sequence of str, optional
        Columns to apply to. Defaults to all numeric columns.

    Returns
    -------
    pandas.DataFrame
        Copy with outlier values set to ``NaN``.

    See Also
    --------
    sigma_clip_frame : drop rows outside the sigma band instead.
    """
    out = df.copy()
    for col, lower, upper in _iter_outlier_bounds(out, sigma=sigma, columns=columns):
        mask = (out[col] < lower) | (out[col] > upper)
        out.loc[mask, col] = np.nan
    return out


def apply_outlier_config(df: pd.DataFrame, cfg: OutlierConfig) -> pd.DataFrame:
    """Apply outlier handling according to *cfg*.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    cfg : OutlierConfig
        Method (``clip`` or ``nan``), sigma width, and optional column subset.

    Returns
    -------
    pandas.DataFrame
        Transformed copy of *df*.

    Raises
    ------
    ValueError
        If ``cfg.method`` is not ``"clip"`` or ``"nan"``.
    """
    if cfg.method == "clip":
        return sigma_clip_frame(df, sigma=cfg.sigma, columns=cfg.columns)
    if cfg.method == "nan":
        return sigma_nan_frame(df, sigma=cfg.sigma, columns=cfg.columns)
    raise ValueError(f"Unknown outlier method: '{cfg.method}'")
