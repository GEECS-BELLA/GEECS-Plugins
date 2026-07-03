"""Compatibility imports for shared 1D data readers.

The 1D file-loading implementation lives in :mod:`geecs_data_utils.io.array1d`
so Bluesky asset handling and ImageAnalysis use the same low-level readers.
"""

from geecs_data_utils.io.array1d import (
    Data1DConfig,
    Data1DResult,
    Data1DType,
    read_1d_data,
)

__all__ = [
    "Data1DConfig",
    "Data1DResult",
    "Data1DType",
    "read_1d_data",
]
