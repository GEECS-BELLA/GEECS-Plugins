"""
Pydantic configuration models for image processing functions.

This module defines configuration models for all image processing operations
including background computation, masking, filtering, and geometric transforms.
All models use Pydantic for validation and automatic YAML/JSON serialization.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, Tuple, List, Union, Dict
from pathlib import Path
from enum import Enum


class BackgroundMethod(str, Enum):
    """Supported background computation methods."""

    CONSTANT = "constant"
    PERCENTILE_DATASET = "percentile_dataset"
    FROM_FILE = "from_file"
    MEDIAN = "median"  # Alias for temporal_median


class DynamicComputationConfig(BaseModel):
    """
    Configuration for dynamic background computation from image batches.

    This is used by Array2DScanAnalyzer to compute backgrounds from
    all images in a scan directory.

    Attributes
    ----------
    enabled : bool
        Whether dynamic background computation is enabled.
    method : BackgroundMethod
        Method to use for background computation.
    percentile : float
        Percentile value for percentile_dataset method (0-100).
    auto_save_path : Optional[Union[str, Path]]
        Path to save the computed background. Supports {scan_dir} placeholder.
    """

    enabled: bool = Field(True, description="Enable dynamic background computation")
    method: BackgroundMethod = Field(
        BackgroundMethod.PERCENTILE_DATASET, description="Background computation method"
    )
    percentile: float = Field(
        5.0, ge=0.0, le=100.0, description="Percentile for dataset background"
    )

    auto_save_path: Optional[Union[str, Path]] = Field(
        None, description="Path to save computed background (supports {scan_dir})"
    )

    @field_validator("percentile")
    def validate_percentile_range(cls, v):
        """Ensure percentile is in valid range."""
        if not 0.0 <= v <= 100.0:
            raise ValueError("percentile must be between 0 and 100")
        return v


class BackgroundConfig(BaseModel):
    """
    Simplified configuration for background subtraction.

    This configuration supports a two-stage workflow:
    1. Primary background source (from_file, constant, or None)
    2. Additional constant offset applied after primary background
    3. Optional dynamic computation (for batch processing only)

    Attributes
    ----------
    enabled : bool
        Whether background processing is enabled.
    method : Optional[BackgroundMethod]
        Primary background method. Can be None to skip primary background.
    file_path : Optional[Union[str, Path]]
        Path to background file (for from_file method). Supports {scan_dir} placeholder.
    constant_level : float
        Constant background level (for constant method, or fallback if file not found).
    additional_constant : float
        Additional constant to subtract AFTER primary background (default 0).
    dynamic_computation : Optional[DynamicComputationConfig]
        Configuration for dynamic background computation (batch processing only).
    """

    enabled: bool = Field(True, description="Enable background processing")
    method: Optional[BackgroundMethod] = Field(
        None, description="Primary background method (None to skip)"
    )
    file_path: Optional[Union[str, Path]] = Field(
        None, description="Path to background file (supports {scan_dir})"
    )
    constant_level: float = Field(
        0.0, ge=0.0, description="Constant background level or fallback"
    )
    additional_constant: float = Field(
        0.0, description="Additional constant offset applied after primary background"
    )
    dynamic_computation: Optional[DynamicComputationConfig] = Field(
        None, description="Dynamic background computation config (batch processing)"
    )

    @field_validator("file_path")
    def validate_file_path(cls, v, info):
        """Validate file path when method is from_file."""
        if hasattr(info, "data"):
            method = info.data.get("method")
            if method == BackgroundMethod.FROM_FILE and v is None:
                raise ValueError('file_path required when method is "from_file"')
        return v


class CrosshairConfig(BaseModel):
    """
    Configuration for a single crosshair.

    Attributes
    ----------
    center : Tuple[int, int]
        (x, y) pixel coordinates of crosshair center.
    width : int
        Width of the crosshair in pixels.
    height : int
        Height of the crosshair in pixels.
    thickness : int
        Thickness of the crosshair lines in pixels.
    angle : float
        Rotation angle of the crosshair in degrees.
    """

    center: Tuple[int, int] = Field(..., description="Center coordinates (x, y)")
    width: int = Field(..., gt=0, description="Crosshair width in pixels")
    height: int = Field(..., gt=0, description="Crosshair height in pixels")
    thickness: int = Field(..., gt=0, description="Crosshair thickness in pixels")
    angle: float = Field(0.0, description="Rotation angle in degrees")

    @field_validator("center")
    def validate_center_coordinates(cls, v):
        """Ensure center coordinates are non-negative."""
        x, y = v
        if x < 0 or y < 0:
            raise ValueError("Center coordinates must be non-negative")
        return v


class CrosshairMaskingConfig(BaseModel):
    """
    Configuration for crosshair masking operations.

    Supports sophisticated crosshair masking with multiple crosshairs,
    rotation, and custom dimensions.

    Attributes
    ----------
    enabled : bool
        Whether crosshair masking is enabled.
    crosshairs : List[CrosshairConfig]
        List of crosshair configurations.
    mask_value : float
        Value to use for masked pixels (typically 0).
    """

    enabled: bool = True
    crosshairs: List[CrosshairConfig] = Field(
        default_factory=list, description="List of crosshair configurations"
    )
    mask_value: float = Field(0.0, description="Value for masked pixels")

    def has_crosshairs(self) -> bool:
        """Check if any crosshairs are configured."""
        return len(self.crosshairs) > 0


class ROIConfig(BaseModel):
    """
    Configuration for region of interest cropping.

    Attributes
    ----------
    x_min : int
        Minimum X coordinate (inclusive).
    x_max : int
        Maximum X coordinate (exclusive).
    y_min : int
        Minimum Y coordinate (inclusive).
    y_max : int
        Maximum Y coordinate (exclusive).
    """

    x_min: int = Field(0, ge=0, description="Minimum X coordinate")
    x_max: int = Field(1024, gt=0, description="Maximum X coordinate")
    y_min: int = Field(0, ge=0, description="Minimum Y coordinate")
    y_max: int = Field(1024, gt=0, description="Maximum Y coordinate")

    @field_validator("x_max")
    def x_max_greater_than_min(cls, v, info):
        """Ensure x_max > x_min."""
        if hasattr(info, "data") and "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError("x_max must be greater than x_min")
        return v

    @field_validator("y_max")
    def y_max_greater_than_min(cls, v, info):
        """Ensure y_max > y_min."""
        if hasattr(info, "data") and "y_min" in info.data and v <= info.data["y_min"]:
            raise ValueError("y_max must be greater than y_min")
        return v

    @property
    def width(self) -> int:
        """Width of the ROI."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Height of the ROI."""
        return self.y_max - self.y_min


class FilteringConfig(BaseModel):
    """
    Configuration for image filtering operations.

    Attributes
    ----------
    gaussian_sigma : Optional[float]
        Standard deviation for Gaussian filter. If None, no Gaussian filtering.
    median_kernel_size : Optional[int]
        Kernel size for median filter. If None, no median filtering.
    """

    gaussian_sigma: Optional[float] = Field(
        None, gt=0.0, description="Gaussian filter sigma"
    )
    median_kernel_size: Optional[int] = Field(
        None, gt=0, description="Median filter kernel size"
    )

    @field_validator("median_kernel_size")
    def validate_median_kernel_odd(cls, v):
        """Ensure median kernel size is odd."""
        if v is not None and v % 2 == 0:
            raise ValueError("median_kernel_size must be odd")
        return v


class TransformConfig(BaseModel):
    """
    Configuration for geometric image transformations.

    Attributes
    ----------
    rotation_angle : float
        Rotation angle in degrees (positive = counterclockwise).
    flip_horizontal : bool
        Whether to flip image horizontally.
    flip_vertical : bool
        Whether to flip image vertically.
    distortion_correction : bool
        Whether to apply distortion correction.
    distortion_coeffs : Optional[List[float]]
        Distortion coefficients for correction.
    """

    rotation_angle: float = Field(0.0, description="Rotation angle in degrees")
    flip_horizontal: bool = Field(False, description="Flip image horizontally")
    flip_vertical: bool = Field(False, description="Flip image vertically")
    distortion_correction: bool = Field(
        False, description="Apply distortion correction"
    )
    distortion_coeffs: Optional[List[float]] = Field(
        None, description="Distortion coefficients"
    )

    @field_validator("rotation_angle")
    def normalize_rotation_angle(cls, v):
        """Normalize rotation angle to [-180, 180] range."""
        while v > 180:
            v -= 360
        while v <= -180:
            v += 360
        return v

    @field_validator("distortion_coeffs")
    def validate_distortion_coeffs(cls, v, info):
        """Validate distortion coefficients when correction is enabled."""
        if (
            hasattr(info, "data")
            and info.data.get("distortion_correction", False)
            and v is None
        ):
            raise ValueError(
                "distortion_coeffs required when distortion_correction is True"
            )
        return v


class CircularMaskConfig(BaseModel):
    """
    Configuration for circular masking operations.

    Attributes
    ----------
    enabled : bool
        Whether circular masking is enabled.
    center : Tuple[int, int]
        (x, y) pixel coordinates of circle center.
    radius : int
        Radius of the circular mask in pixels.
    mask_outside : bool
        If True, mask outside the circle. If False, mask inside.
    mask_value : float
        Value to use for masked pixels.
    """

    enabled: bool = False
    center: Tuple[int, int] = (512, 512)
    radius: int = Field(100, gt=0, description="Radius of circular mask")
    mask_outside: bool = Field(True, description="Mask outside circle if True")
    mask_value: float = Field(0.0, description="Value for masked pixels")

    @field_validator("center")
    def validate_center_coordinates(cls, v):
        """Ensure center coordinates are non-negative."""
        x, y = v
        if x < 0 or y < 0:
            raise ValueError("Center coordinates must be non-negative")
        return v


class ThresholdMethod(str, Enum):
    """Supported thresholding methods."""

    CONSTANT = "constant"
    PERCENTAGE_MAX = "percentage_max"


class ThresholdMode(str, Enum):
    """Supported thresholding modes."""

    BINARY = "binary"
    TO_ZERO = "to_zero"
    TRUNCATE = "truncate"
    TO_ZERO_INV = "to_zero_inv"
    TRUNCATE_INV = "truncate_inv"


class ThresholdingConfig(BaseModel):
    """
    Configuration for image thresholding operations.

    Attributes
    ----------
    enabled : bool
        Whether thresholding is enabled.
    method : ThresholdMethod
        Thresholding method to use.
    value : float
        Threshold value (absolute for constant, percentage 0-100 for percentage_max).
    mode : ThresholdMode
        Thresholding mode to apply.
    invert : bool
        Whether to invert the threshold operation.
    """

    enabled: bool = Field(False, description="Whether thresholding is enabled")
    method: ThresholdMethod = ThresholdMethod.CONSTANT
    value: float = Field(100.0, ge=0.0, description="Threshold value")
    mode: ThresholdMode = ThresholdMode.BINARY
    invert: bool = Field(False, description="Invert threshold operation")

    @field_validator("value")
    def validate_threshold_value(cls, v, info):
        """Validate threshold value based on method."""
        if hasattr(info, "data"):
            method = info.data.get("method")
            if method == ThresholdMethod.PERCENTAGE_MAX:
                if not 0.0 <= v <= 100.0:
                    raise ValueError(
                        "Percentage threshold value must be between 0 and 100"
                    )
            elif method == ThresholdMethod.CONSTANT:
                if v < 0:
                    raise ValueError("Constant threshold value must be non-negative")
        return v


class NormalizationMethod(str, Enum):
    """Supported normalization methods."""

    IMAGE_TOTAL = "image_total"
    IMAGE_MAX = "image_max"
    CONSTANT = "constant"
    DISTRIBUTE_VALUE = "distribute_value"


class NormalizationConfig(BaseModel):
    """
    Configuration for image normalization.

    Normalizes images by dividing by a normalization factor determined by
    the configured method. Useful for making images comparable across different
    exposure times or intensities.

    Attributes
    ----------
    enabled : bool
        Whether normalization is enabled.
    method : NormalizationMethod
        Normalization method to use.
        - IMAGE_TOTAL: Divide by sum of all pixel values
        - IMAGE_MAX: Divide by maximum pixel value
        - CONSTANT: Divide by constant_value
        - DISTRIBUTE_VALUE: Divide by sum of all pixel values, then multiply by constant_value
    constant_value : Optional[float]
        Divisor for CONSTANT method, or multiplier for DISTRIBUTE_VALUE method.
        Required if method is CONSTANT or DISTRIBUTE_VALUE.
    """

    enabled: bool = Field(False, description="Whether normalization is enabled")
    method: NormalizationMethod = Field(
        NormalizationMethod.IMAGE_TOTAL,
        description="Normalization method to use",
    )
    constant_value: Optional[float] = Field(
        None, description="Divisor for constant method"
    )

    @field_validator("constant_value")
    def validate_constant_value(cls, v, info):
        """Ensure constant_value is provided when method is CONSTANT or DISTRIBUTE_VALUE."""
        if hasattr(info, "data"):
            method = info.data.get("method")
            if (
                method == NormalizationMethod.CONSTANT
                or method == NormalizationMethod.DISTRIBUTE_VALUE
            ):
                if v is None or v == 0:
                    raise ValueError(
                        "constant_value must be non-zero when method is 'constant' or 'distribute_value'"
                    )
        return v


class ProcessingStepType(str, Enum):
    """Available processing step types for the pipeline."""

    BACKGROUND = "background"
    CROSSHAIR_MASKING = "crosshair_masking"
    ROI = "roi"
    CIRCULAR_MASK = "circular_mask"
    THRESHOLDING = "thresholding"
    FILTERING = "filtering"
    NORMALIZATION = "normalization"
    TRANSFORMS = "transforms"


class PipelineConfig(BaseModel):
    """
    Configuration for the processing pipeline execution order.

    This allows users to customize which processing steps are executed
    and in what order. If not specified, a default pipeline matching
    the original hardcoded order is used.

    Attributes
    ----------
    steps : List[ProcessingStepType]
        Ordered list of processing steps to execute. Steps are executed
        in the order specified. If a step's configuration is not provided
        or is disabled, it will be skipped automatically.
    """

    steps: List[ProcessingStepType] = Field(
        default_factory=lambda: [
            ProcessingStepType.BACKGROUND,
            ProcessingStepType.CROSSHAIR_MASKING,
            ProcessingStepType.ROI,
            ProcessingStepType.CIRCULAR_MASK,
            ProcessingStepType.THRESHOLDING,
            ProcessingStepType.FILTERING,
            ProcessingStepType.NORMALIZATION,
            ProcessingStepType.TRANSFORMS,
        ],
        description="Ordered list of processing steps to execute",
    )


class CameraConfig(BaseModel):
    """
    Complete camera configuration model.

    This is the top-level configuration model that encompasses all
    processing settings for a specific camera. It provides type safety
    and validation for the entire camera configuration.

    Attributes
    ----------
    name : str
        Camera identifier/name.
    description : Optional[str]
        Human-readable description of the camera.
    bit_depth : int
        Bit depth of camera images (typically 8, 12, 14, or 16).
    roi : Optional[ROIConfig]
        Region of interest configuration.
    background : Optional[BackgroundConfig]
        Background computation and subtraction configuration.
    crosshair_masking : Optional[CrosshairMaskingConfig]
        Crosshair masking configuration.
    circular_mask : Optional[CircularMaskConfig]
        Circular masking configuration.
    thresholding : Optional[ThresholdingConfig]
        Image thresholding configuration.
    filtering : Optional[FilteringConfig]
        Image filtering configuration.
    normalization : Optional[NormalizationConfig]
        Image normalization configuration.
    transforms : Optional[TransformConfig]
        Geometric transformation configuration.
    """

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    name: str = Field(..., description="Camera identifier/name")
    description: Optional[str] = Field(None, description="Camera description")
    bit_depth: int = Field(16, ge=8, le=32, description="Camera bit depth")

    # Processing configurations (all optional)
    roi: Optional[ROIConfig] = Field(None, description="Region of interest settings")
    background: Optional[BackgroundConfig] = Field(
        None, description="Background processing settings"
    )
    crosshair_masking: Optional[CrosshairMaskingConfig] = Field(
        None, description="Crosshair masking settings"
    )
    circular_mask: Optional[CircularMaskConfig] = Field(
        None, description="Circular masking settings"
    )
    thresholding: Optional[ThresholdingConfig] = Field(
        None, description="Image thresholding settings"
    )
    filtering: Optional[FilteringConfig] = Field(
        None, description="Image filtering settings"
    )
    normalization: Optional[NormalizationConfig] = Field(
        None, description="Image normalization settings"
    )
    transforms: Optional[TransformConfig] = Field(
        None, description="Geometric transform settings"
    )
    pipeline: Optional[PipelineConfig] = Field(
        None, description="Processing pipeline configuration"
    )

    @field_validator("bit_depth")
    def validate_bit_depth(cls, v):
        """Ensure bit depth is a common value."""
        valid_depths = [8, 10, 12, 14, 16, 32]
        if v not in valid_depths:
            raise ValueError(f"bit_depth must be one of {valid_depths}")
        return v

    def get_processing_configs(self) -> Dict[str, BaseModel]:
        """
        Get all non-None processing configurations as a dictionary.

        Returns
        -------
        Dict[str, BaseModel]
            Dictionary of processing configurations keyed by type.
        """
        configs = {}

        config_mapping = {
            "roi": self.roi,
            "background": self.background,
            "crosshair_masking": self.crosshair_masking,
            "circular_mask": self.circular_mask,
            "thresholding": self.thresholding,
            "filtering": self.filtering,
            "normalization": self.normalization,
            "transforms": self.transforms,
        }

        for key, config in config_mapping.items():
            if config is not None:
                configs[key] = config

        return configs

    @property
    def max_pixel_value(self) -> int:
        """Maximum pixel value for this camera's bit depth."""
        return (2**self.bit_depth) - 1

    @property
    def processing_dtype(self) -> str:
        """Recommended numpy dtype for processing (always float64 for precision)."""
        return "float64"

    @property
    def storage_dtype(self) -> str:
        """Recommended numpy dtype for storage based on bit depth."""
        if self.bit_depth <= 8:
            return "uint8"
        elif self.bit_depth <= 16:
            return "uint16"
        else:
            return "uint32"
