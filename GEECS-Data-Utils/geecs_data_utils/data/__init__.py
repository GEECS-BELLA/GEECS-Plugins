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
from geecs_data_utils.data.binning import (
    BinnedFrame,
    BinningConfig,
    bin_frame,
    compute_bin_key,
)
from geecs_data_utils.data.row_filters import (
    FilterCondition,
    FilterGroup,
    RowFilters,
    apply_filters,
    filter_mask,
)
from geecs_data_utils.data.sfile import read_sfile, sfile_path_for_scan
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
    "BinnedFrame",
    "BinningConfig",
    "bin_frame",
    "compute_bin_key",
    "FilterCondition",
    "FilterGroup",
    "RowFilters",
    "apply_filters",
    "filter_mask",
    "read_sfile",
    "sfile_path_for_scan",
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
