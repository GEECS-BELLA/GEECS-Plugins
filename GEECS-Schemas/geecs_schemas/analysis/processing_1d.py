"""Line (1D trace) processing configuration — the ``image:`` section of a 1D diagnostic.

Covers scope traces, spectra, lineouts and other x-vs-y data.  A trace is
loaded per :class:`Data1DLoading`, scaled by the axis scale factors, and
then run through the ``pipeline`` steps in order — ROI (in x-value units,
not indices), interpolation onto a uniform grid, background, filtering,
thresholding.  As with cameras, a step runs only when listed.

Formerly ImageAnalysis' own 1D processing models (moved here in GEECS-Schemas
0.19.0).  The 1D sub-models carry a ``Line`` prefix here so they never
collide with the camera models of the same role; ImageAnalysis re-exports
them under their historical short names.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from geecs_schemas._base import SchemaModel

NumpyDtypeName = Literal[
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


class Data1DType(str, Enum):
    """The file formats the 1D reader understands."""

    TEK_SCOPE_HDF5 = "tek_scope_hdf5"
    TDMS_SCOPE = "tdms_scope"
    CSV = "csv"
    TSV = "tsv"
    NPY = "npy"


class Data1DLoading(SchemaModel):
    """How to read one trace file into an x-vs-y array.

    Mirrors ``geecs_data_utils.io.array1d.Data1DConfig`` field for field —
    the reader is handed ``Data1DConfig.model_validate(loading.model_dump())``
    — so the schema package stays pydantic-only while the reader keeps its
    own model.
    """

    data_type: Data1DType = Field(
        ...,
        description=(
            "File format: 'tek_scope_hdf5' or 'tdms_scope' for scope captures, "
            "'csv' / 'tsv' for delimited text, 'npy' for a saved array."
        ),
    )
    trace_index: int = Field(
        0, ge=0, description="Which trace / channel holds the y values (scope formats)."
    )
    x_trace_index: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "Which trace holds the x values; unset derives x from the waveform "
            "properties (scope formats)."
        ),
    )
    delimiter: Optional[str] = Field(
        None,
        description="Column delimiter for csv/tsv; unset uses the format's default.",
    )
    x_column: int = Field(0, ge=0, description="Column index of x (text formats).")
    y_column: int = Field(1, ge=0, description="Column index of y (text formats).")
    auxiliary_columns: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Extra named columns to load alongside y, for analyzers that need "
            "them (name -> column index)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_model_instances(cls, data: Any) -> Any:
        """Accept any pydantic model with the same fields (GEECS-Data-Utils' ``Data1DConfig``)."""
        if isinstance(data, BaseModel) and not isinstance(data, cls):
            return data.model_dump(mode="json")
        return data

    @model_validator(mode="after")
    def validate_auxiliary_columns(self) -> "Data1DLoading":
        """Validate named auxiliary column definitions (mirrors ``Data1DConfig``)."""
        seen: set[int] = set()
        for name, column in self.auxiliary_columns.items():
            if not name.strip():
                raise ValueError("auxiliary column names must be non-empty")
            if column < 0:
                raise ValueError("auxiliary column indices must be non-negative")
            if column in {self.x_column, self.y_column}:
                raise ValueError(
                    "auxiliary column indices must differ from x_column and y_column"
                )
            if column in seen:
                raise ValueError("auxiliary column indices must be unique")
            seen.add(column)
        return self


class LineBackgroundMethod(str, Enum):
    """How the trace background is obtained."""

    NONE = "none"
    CONSTANT = "constant"
    FROM_FILE = "from_file"


class LineFilterMethod(str, Enum):
    """Smoothing filters for traces."""

    NONE = "none"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"


class LineThresholdMethod(str, Enum):
    """How the trace threshold level is chosen."""

    NONE = "none"
    ABSOLUTE = "absolute"
    PERCENTILE = "percentile"


class LineROIConfig(SchemaModel):
    """Keep only the part of the trace between two x values.

    The bounds are in the trace's x units (after ``x_scale_factor``), not
    sample indices — a time window in seconds, a wavelength band in nm.
    Either bound may be left open.
    """

    x_min: Optional[float] = Field(
        None, description="Lowest x value kept (inclusive); unset = no lower bound."
    )
    x_max: Optional[float] = Field(
        None, description="Highest x value kept (inclusive); unset = no upper bound."
    )


class LinePipelineStepType(str, Enum):
    """The trace processing steps a pipeline may list."""

    ROI = "roi"
    INTERPOLATION = "interpolation"
    BACKGROUND = "background"
    FILTERING = "filtering"
    THRESHOLDING = "thresholding"


class LineBackgroundConfig(SchemaModel):
    """Subtract a background from the trace's y values.

    Field names match the camera :class:`~geecs_schemas.analysis.processing_2d.BackgroundConfig`
    (``constant_level``, ``file_path``) since 0.19.0; the older 1D spellings
    ``constant_value`` / ``background_file`` are lifted automatically.
    """

    method: LineBackgroundMethod = Field(
        LineBackgroundMethod.NONE,
        description=(
            "'constant' subtracts constant_level from y; 'from_file' subtracts "
            "the trace at file_path (any supported data_type); 'none' skips."
        ),
    )
    constant_level: Optional[float] = Field(
        None, description="Level subtracted from y for method 'constant'."
    )
    file_path: Optional[Path] = Field(
        None, description="Background trace file for method 'from_file'."
    )

    @model_validator(mode="after")
    def _method_requirements(self) -> "LineBackgroundConfig":
        """Require the field the chosen method needs."""
        if self.method == LineBackgroundMethod.CONSTANT and self.constant_level is None:
            raise ValueError("constant_level must be provided when method is CONSTANT")
        if self.method == LineBackgroundMethod.FROM_FILE and self.file_path is None:
            raise ValueError("file_path must be provided when method is FROM_FILE")
        return self


class LineFilteringConfig(SchemaModel):
    """Smooth the trace."""

    method: LineFilterMethod = Field(
        LineFilterMethod.NONE,
        description="'gaussian' (uses sigma), 'median' (uses kernel_size), 'bilateral', or 'none'.",
    )
    kernel_size: Optional[int] = Field(
        3, ge=1, description="Filter window in samples (odd), for the median filter."
    )
    sigma: Optional[float] = Field(
        1.0, gt=0, description="Gaussian width in samples, for the Gaussian filter."
    )

    @field_validator("kernel_size")
    @classmethod
    def validate_kernel_size(cls, v: Optional[int]) -> Optional[int]:
        """Ensure kernel size is odd."""
        if v is not None and v % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return v


class LineThresholdingConfig(SchemaModel):
    """Clip trace values below (or above) a level."""

    method: LineThresholdMethod = Field(
        LineThresholdMethod.NONE,
        description="'absolute' uses threshold_value, 'percentile' uses percentile, 'none' skips.",
    )
    threshold_value: Optional[float] = Field(
        None, description="Level in y units for method 'absolute'."
    )
    percentile: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Percentile of y (0-100) for method 'percentile'.",
    )
    clip_below: bool = Field(
        True, description="True clips values below the level; False clips values above."
    )

    @model_validator(mode="after")
    def _method_requirements(self) -> "LineThresholdingConfig":
        """Require the field the chosen method needs."""
        if self.method == LineThresholdMethod.ABSOLUTE and self.threshold_value is None:
            raise ValueError("threshold_value must be provided when method is ABSOLUTE")
        if self.method == LineThresholdMethod.PERCENTILE and self.percentile is None:
            raise ValueError("percentile must be provided when method is PERCENTILE")
        return self


class LineInterpolationConfig(SchemaModel):
    """Resample the trace onto a uniform x grid (so waterfall plots share an axis)."""

    num_points: int = Field(
        1500, ge=10, description="Number of points in the resampled trace."
    )
    x_min: Optional[float] = Field(
        None, description="Start of the grid; unset uses the data minimum."
    )
    x_max: Optional[float] = Field(
        None, description="End of the grid; unset uses the data maximum."
    )

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "LineInterpolationConfig":
        """Require x_max > x_min when both are given."""
        if (
            self.x_min is not None
            and self.x_max is not None
            and self.x_max <= self.x_min
        ):
            raise ValueError("x_max must be greater than x_min")
        return self


class Line1DConfig(SchemaModel):
    """How a device's traces are loaded and cleaned up before the analyzer measures them.

    The scale factors apply first, so ROI bounds and thresholds are written
    in the scaled units.  Each processing section is optional and runs only
    when listed in ``pipeline``, in that order.
    """

    type: Literal["line"] = Field(
        "line", description="Marks this as a line (1D trace) section."
    )
    description: str = Field("", description="Free-text note about this trace.")
    metadata: Optional[Dict[str, Any]] = Field(  # documentary free-form fields
        None,
        description=(
            "Free-form documentation (location, notes, calibration constants). "
            "Nothing in the pipeline reads it."
        ),
    )
    data_loading: Data1DLoading = Field(..., description="How to read one trace file.")
    label: str = Field(
        "x vs y",
        description=(
            "Human-readable description of what the trace is, e.g. 'charge "
            "density vs energy'. Shown on figures; not interpreted."
        ),
    )
    x_units: Optional[str] = Field(
        None,
        description="X-axis units (e.g. 'nm', 's'); overrides units read from the file.",
    )
    y_units: Optional[str] = Field(
        None,
        description="Y-axis units (e.g. 'V', 'counts'); overrides units read from the file.",
    )
    x_scale_factor: float = Field(
        1.0,
        description=(
            "Multiplier applied to x before any processing (1e9 turns seconds "
            "into nanoseconds). ROI bounds are in the scaled units."
        ),
    )
    y_scale_factor: float = Field(
        1.0,
        description=(
            "Multiplier applied to y before any processing. Thresholds are in "
            "the scaled units."
        ),
    )
    processing_dtype: NumpyDtypeName = Field(
        "float64", description="NumPy dtype used while processing."
    )
    storage_dtype: NumpyDtypeName = Field(
        "float32", description="NumPy dtype used when saving processed traces."
    )

    roi: Optional[LineROIConfig] = Field(None, description="X-range crop.")
    interpolation: Optional[LineInterpolationConfig] = Field(
        None, description="Resampling onto a uniform x grid."
    )
    background: Optional[LineBackgroundConfig] = Field(
        None, description="Background subtraction."
    )
    filtering: Optional[LineFilteringConfig] = Field(None, description="Smoothing.")
    thresholding: Optional[LineThresholdingConfig] = Field(
        None, description="Thresholding."
    )
    pipeline: List[LinePipelineStepType] = Field(
        default_factory=list,
        description=(
            "The steps that run, in order. A step runs only if listed here AND "
            "its section is present; empty means the raw trace is analyzed."
        ),
    )


__all__ = [
    "Data1DLoading",
    "Data1DType",
    "Line1DConfig",
    "LineBackgroundConfig",
    "LineBackgroundMethod",
    "LineFilterMethod",
    "LineFilteringConfig",
    "LineInterpolationConfig",
    "LinePipelineStepType",
    "LineROIConfig",
    "LineThresholdMethod",
    "LineThresholdingConfig",
    "NumpyDtypeName",
]
