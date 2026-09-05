"""Line (1D) processing models — re-exported from GEECS-Schemas under their historical names.

The models moved to :mod:`geecs_schemas.analysis.processing_1d` in
ImageAnalysis 2.0 (with a ``Line`` prefix there, so they never collide
with the camera models of the same role).  This module keeps the short
names the processing functions and analyzers have always used, plus the
one boundary helper the schema package cannot own:
:func:`to_data1d_config`, which hands a schema :class:`Data1DLoading` to
GEECS-Data-Utils' reader as its own :class:`Data1DConfig`.
"""

from __future__ import annotations

from typing import Union

from geecs_data_utils.io.array1d import Data1DConfig
from geecs_schemas.analysis.processing_1d import (
    Data1DLoading,
    Data1DType,
    Line1DConfig,
    LineBackgroundConfig as BackgroundConfig,
    LineBackgroundMethod as BackgroundMethod,
    LineFilteringConfig as FilteringConfig,
    LineFilterMethod as FilterMethod,
    LineInterpolationConfig as InterpolationConfig,
    LinePipelineStepType as PipelineStepType,
    LineROIConfig as ROI1DConfig,
    LineThresholdingConfig as ThresholdingConfig,
    LineThresholdMethod as ThresholdMethod,
)

__all__ = [
    "BackgroundConfig",
    "BackgroundMethod",
    "Data1DConfig",
    "Data1DLoading",
    "Data1DType",
    "FilterMethod",
    "FilteringConfig",
    "InterpolationConfig",
    "Line1DConfig",
    "PipelineStepType",
    "ROI1DConfig",
    "ThresholdMethod",
    "ThresholdingConfig",
    "to_data1d_config",
]


def to_data1d_config(loading: Union[Data1DLoading, Data1DConfig]) -> Data1DConfig:
    """Return the GEECS-Data-Utils reader config for a schema ``data_loading`` section.

    The two models are field-for-field mirrors (the schema package cannot
    import GEECS-Data-Utils), so the conversion is a dump/validate.  A
    ``Data1DConfig`` passes through unchanged.
    """
    if isinstance(loading, Data1DConfig):
        return loading
    return Data1DConfig.model_validate(loading.model_dump(mode="json"))
