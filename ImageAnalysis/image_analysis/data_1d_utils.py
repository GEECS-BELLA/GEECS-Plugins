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

__all__ = [
    "Data1DConfig",
    "Data1DLoading",
    "Data1DResult",
    "Data1DType",
    "read_1d_data",
    "to_data1d_config",
]


def to_data1d_config(loading: Union[Data1DLoading, Data1DConfig]) -> Data1DConfig:
    """Return the GEECS-Data-Utils reader config for a schema ``data_loading`` section.

    The two models are field-for-field mirrors (the schema package cannot
    import GEECS-Data-Utils; ``tests/test_data1d_loading_mirror.py`` pins
    it), so the conversion is a dump/validate.  A ``Data1DConfig`` passes
    through unchanged.
    """
    if isinstance(loading, Data1DConfig):
        return loading
    return Data1DConfig.model_validate(loading.model_dump(mode="json"))


def read_1d_data(
    file_path: Union[Path, str], config: Union[Data1DLoading, Data1DConfig]
) -> Data1DResult:
    """Read one x-vs-y file per ``config`` (schema ``Data1DLoading`` or reader ``Data1DConfig``)."""
    return _read_1d_data(file_path, to_data1d_config(config))
