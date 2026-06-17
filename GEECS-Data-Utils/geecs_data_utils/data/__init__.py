"""Public API for shared data utilities (columns, cleaning, generic dataset assembly).

``RowFilterSpec`` names the tuple shape accepted by :func:`apply_row_filters` and
by :class:`~geecs_data_utils.data.dataset.DatasetBuilder` filter arguments.
"""

from geecs_data_utils.data.cleaning import (
    OutlierConfig,
    RowFilterSpec,
    apply_outlier_config,
    apply_row_filters,
    sigma_clip_frame,
    sigma_nan_frame,
)
from geecs_data_utils.data.columns import (
    ColumnMatchMode,
    ResolveColResult,
    find_cols,
    flatten_columns,
    resolve_col,
    resolve_col_detailed,
)
from geecs_data_utils.data.dataset import DatasetBuilder, DatasetFrame, LoadScansReport

__all__ = [
    "apply_row_filters",
    "OutlierConfig",
    "RowFilterSpec",
    "apply_outlier_config",
    "sigma_clip_frame",
    "sigma_nan_frame",
    "ColumnMatchMode",
    "ResolveColResult",
    "find_cols",
    "flatten_columns",
    "resolve_col",
    "resolve_col_detailed",
    "DatasetBuilder",
    "DatasetFrame",
    "LoadScansReport",
]
