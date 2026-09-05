"""Shared 1D data readers, with the schema ``data_loading`` boundary.

The file-loading implementation lives in :mod:`geecs_data_utils.io.array1d`
so Bluesky asset handling and ImageAnalysis use the same low-level readers.
:func:`read_1d_data` here additionally accepts the schema-side
:class:`~geecs_schemas.analysis.Data1DLoading` (what a diagnostic's
``image.data_loading`` validates to) and converts it to the reader's own
``Data1DConfig`` on the way through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from geecs_data_utils.io.array1d import (
    Data1DConfig,
    Data1DResult,
    Data1DType,
)
from geecs_data_utils.io.array1d import read_1d_data as _read_1d_data
from geecs_schemas.analysis import Data1DLoading

from image_analysis.config.array1d_processing import to_data1d_config

__all__ = [
    "Data1DConfig",
    "Data1DLoading",
    "Data1DResult",
    "Data1DType",
    "read_1d_data",
]


def read_1d_data(
    file_path: Union[Path, str], config: Union[Data1DLoading, Data1DConfig]
) -> Data1DResult:
    """Read one x-vs-y file per ``config`` (schema ``Data1DLoading`` or reader ``Data1DConfig``)."""
    return _read_1d_data(file_path, to_data1d_config(config))
