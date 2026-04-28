"""
Data cleaning utilities for tabular scan data.

This module provides reusable DataFrame-level cleaning helpers used by
dataset assembly and analysis workflows.
"""

from typing import List, Tuple, Union

import pandas as pd

_OPERATORS = {
    ">": lambda s, v: s > v,
    "<": lambda s, v: s < v,
    ">=": lambda s, v: s >= v,
    "<=": lambda s, v: s <= v,
    "==": lambda s, v: s == v,
    "!=": lambda s, v: s != v,
}


def apply_row_filters(
    df: pd.DataFrame,
    filters: List[Tuple[str, str, Union[int, float]]],
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
