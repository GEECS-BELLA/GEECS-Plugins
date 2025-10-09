"""Configuration models for 1D array processing.

This module provides Pydantic models for configuring 1D data processing pipelines,
including lineouts, spectra, and other 1D array data.

Data Format Convention:
    All 1D data is expected to be in Nx2 format where:
    - Column 0: x-values (independent variable - wavelength, time, position, etc.)
    - Column 1: y-values (dependent variable - intensity, signal, etc.)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, List

import numpy as np
from pydantic import BaseModel, Field, field_validator


class Data1DType(str, Enum):
    """Enumeration of supported 1D data formats.

    All formats return Nx2 arrays where column 0 is x values and column 1 is y values.
    """

    TEK_SCOPE_HDF5 = "tek_scope_hdf5"
    TDMS_SCOPE = "tdms_scope"
    CSV = "csv"
    TSV = "tsv"
    NPY = "npy"


class Data1DConfig(BaseModel):
    r"""Configuration for reading 1D data files.

    Parameters
    ----------
    data_type : Data1DType
        The type of data format to read
    trace_index : int, default=0
        Trace/channel index for scope files (Tek HDF5, TDMS)
    delimiter : str, optional
        Delimiter for CSV/TSV files (defaults to ',' for CSV, '\t' for TSV)
    x_column : int, default=0
        Column index for x values in delimited files
    y_column : int, default=1
        Column index for y values in delimited files
    """

    data_type: Data1DType
    trace_index: int = Field(default=0, ge=0, description="Trace/channel index")
    delimiter: Optional[str] = Field(
        default=None, description="Delimiter for CSV/TSV files"
    )
    x_column: int = Field(default=0, ge=0, description="Column index for x values")
    y_column: int = Field(default=1, ge=0, description="Column index for y values")


class BackgroundMethod(str, Enum):
    """Background subtraction methods for 1D data."""

    NONE = "none"
    CONSTANT = "constant"
    FROM_FILE = "from_file"


class FilterMethod(str, Enum):
    """Filtering methods for 1D data."""

    NONE = "none"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"


class ThresholdMethod(str, Enum):
    """Thresholding methods for 1D data."""

    NONE = "none"
    ABSOLUTE = "absolute"
    PERCENTILE = "percentile"


class PipelineStepType(str, Enum):
    """Types of processing steps available in the pipeline."""

    BACKGROUND = "background"
    FILTERING = "filtering"
    THRESHOLDING = "thresholding"


class BackgroundConfig(BaseModel):
    """Configuration for background subtraction on 1D data.

    Attributes
    ----------
    method : BackgroundMethod
        Background subtraction method to use
    constant_value : float, optional
        Constant value to subtract (for CONSTANT method)
    background_file : Path, optional
        Path to background file (for FROM_FILE method)
    """

    method: BackgroundMethod = BackgroundMethod.NONE
    constant_value: Optional[float] = Field(
        default=None, description="Constant value to subtract from y-values"
    )
    background_file: Optional[Path] = Field(
        default=None, description="Path to background file (NPY or NPZ format)"
    )

    @field_validator("constant_value")
    @classmethod
    def validate_constant_value(cls, v, info):
        """Validate that constant_value is provided when method is CONSTANT."""
        if info.data.get("method") == BackgroundMethod.CONSTANT and v is None:
            raise ValueError("constant_value must be provided when method is CONSTANT")
        return v

    @field_validator("background_file")
    @classmethod
    def validate_background_file(cls, v, info):
        """Validate that background_file is provided when method is FROM_FILE."""
        if info.data.get("method") == BackgroundMethod.FROM_FILE and v is None:
            raise ValueError(
                "background_file must be provided when method is FROM_FILE"
            )
        return v


class FilteringConfig(BaseModel):
    """Configuration for filtering 1D data.

    Attributes
    ----------
    method : FilterMethod
        Filtering method to use
    kernel_size : int, optional
        Size of the filter kernel (for applicable methods)
    sigma : float, optional
        Standard deviation for Gaussian filter
    """

    method: FilterMethod = FilterMethod.NONE
    kernel_size: Optional[int] = Field(
        default=3, ge=1, description="Filter kernel size (must be odd)"
    )
    sigma: Optional[float] = Field(
        default=1.0, gt=0, description="Gaussian filter sigma"
    )

    @field_validator("kernel_size")
    @classmethod
    def validate_kernel_size(cls, v):
        """Ensure kernel size is odd."""
        if v is not None and v % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return v


class ThresholdingConfig(BaseModel):
    """Configuration for thresholding 1D data.

    Attributes
    ----------
    method : ThresholdMethod
        Thresholding method to use
    threshold_value : float, optional
        Threshold value (for ABSOLUTE method)
    percentile : float, optional
        Percentile value 0-100 (for PERCENTILE method)
    clip_below : bool
        If True, clip values below threshold; if False, clip above
    """

    method: ThresholdMethod = ThresholdMethod.NONE
    threshold_value: Optional[float] = Field(
        default=None, description="Absolute threshold value"
    )
    percentile: Optional[float] = Field(
        default=None, ge=0, le=100, description="Percentile threshold (0-100)"
    )
    clip_below: bool = Field(
        default=True, description="Clip values below (True) or above (False) threshold"
    )

    @field_validator("threshold_value")
    @classmethod
    def validate_threshold_value(cls, v, info):
        """Validate threshold_value for ABSOLUTE method."""
        if info.data.get("method") == ThresholdMethod.ABSOLUTE and v is None:
            raise ValueError("threshold_value must be provided when method is ABSOLUTE")
        return v

    @field_validator("percentile")
    @classmethod
    def validate_percentile(cls, v, info):
        """Validate percentile for PERCENTILE method."""
        if info.data.get("method") == ThresholdMethod.PERCENTILE and v is None:
            raise ValueError("percentile must be provided when method is PERCENTILE")
        return v


class PipelineConfig(BaseModel):
    """Configuration for the 1D processing pipeline.

    Attributes
    ----------
    steps : List[PipelineStepType]
        Ordered list of processing steps to apply
    """

    steps: List[PipelineStepType] = Field(
        default_factory=lambda: [
            PipelineStepType.BACKGROUND,
            PipelineStepType.FILTERING,
            PipelineStepType.THRESHOLDING,
        ],
        description="Ordered list of processing steps",
    )


class Line1DConfig(BaseModel):
    """Main configuration for 1D line/spectrum processing.

    This configuration mirrors the CameraConfig structure from array2d but
    adapted for 1D data processing.

    Attributes
    ----------
    name : str
        Unique identifier for this configuration
    description : str
        Human-readable description
    data_loading : Data1DConfig
        Configuration for loading 1D data from files
    data_format : str
        Description of the data format (e.g., "wavelength vs intensity")
    processing_dtype : str
        Data type for processing (default: float64)
    storage_dtype : str
        Data type for storage (default: float32)
    background : BackgroundConfig, optional
        Background subtraction configuration
    filtering : FilteringConfig, optional
        Filtering configuration
    thresholding : ThresholdingConfig, optional
        Thresholding configuration
    pipeline : PipelineConfig, optional
        Pipeline orchestration configuration
    """

    name: str = Field(..., description="Configuration name/identifier")
    description: str = Field(
        default="", description="Human-readable description of this configuration"
    )

    # Data loading configuration
    data_loading: Data1DConfig = Field(
        ..., description="Configuration for loading 1D data from files"
    )

    data_format: str = Field(
        default="x vs y",
        description="Description of data format (e.g., 'wavelength vs intensity')",
    )
    processing_dtype: str = Field(
        default="float64", description="NumPy dtype for processing"
    )
    storage_dtype: str = Field(default="float32", description="NumPy dtype for storage")

    # Processing configurations
    background: Optional[BackgroundConfig] = None
    filtering: Optional[FilteringConfig] = None
    thresholding: Optional[ThresholdingConfig] = None
    pipeline: Optional[PipelineConfig] = Field(default_factory=PipelineConfig)

    @field_validator("processing_dtype", "storage_dtype")
    @classmethod
    def validate_dtype(cls, v):
        """Validate that dtype is a valid NumPy dtype."""
        try:
            np.dtype(v)
        except TypeError:
            raise ValueError(f"Invalid NumPy dtype: {v}")
        return v

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True
