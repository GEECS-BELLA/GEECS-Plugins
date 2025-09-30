"""
Pydantic configuration models for image processing functions.

This module defines configuration models for all image processing operations
including background computation, masking, filtering, and geometric transforms.
All models use Pydantic for validation and automatic YAML/JSON serialization.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Tuple, List, Union, Dict
from pathlib import Path
from enum import Enum


class BackgroundType(str, Enum):
    """Background processing types."""

    STATIC = "static"  # Fixed background (from file or constant)
    DYNAMIC = "dynamic"  # Computed from image data


class BackgroundMethod(str, Enum):
    """Supported background computation methods."""

    CONSTANT = "constant"
    PERCENTILE_DATASET = "percentile_dataset"
    TEMPORAL_MEDIAN = "temporal_median"
    OUTLIER_REJECTION = "outlier_rejection"
    FROM_FILE = "from_file"
    MEDIAN = "median"  # Alias for temporal_median
    MEAN = "mean"  # Simple mean background


class FallbackBackgroundConfig(BaseModel):
    """
    Fallback background configuration for when dynamic background is unavailable.

    Attributes
    ----------
    method : BackgroundMethod
        Fallback method to use.
    level : float
        Constant background level (for constant method).
    file_path : Optional[Union[str, Path]]
        Path to fallback background file.
    """

    method: BackgroundMethod = BackgroundMethod.CONSTANT
    level: float = Field(0.0, ge=0.0, description="Constant background level")
    file_path: Optional[Union[str, Path]] = Field(
        None, description="Path to fallback background file"
    )

    @field_validator("file_path")
    def validate_fallback_file_path(cls, v, info):
        """Validate file path when method is from_file."""
        if (
            hasattr(info, "data")
            and info.data.get("method") == BackgroundMethod.FROM_FILE
            and v is None
        ):
            raise ValueError('file_path required when fallback method is "from_file"')
        return v


class BackgroundConfig(BaseModel):
    """
    Enhanced configuration for background computation and subtraction.

    Supports both static backgrounds (fixed from file/constant) and dynamic
    backgrounds (computed from image batches with optional fallback).

    Attributes
    ----------
    enabled : bool
        Whether background processing is enabled.
    type : BackgroundType
        Type of background processing (static or dynamic).
    method : BackgroundMethod
        Method to use for background computation.
    level : float
        Constant background level (used when method='constant').
    percentile : float
        Percentile value for dataset background computation (0-100).
    outlier_threshold : float
        Threshold for outlier rejection method (in standard deviations).
    file_path : Optional[Union[str, Path]]
        Path to background file (used when method='from_file').
    auto_save_path : Optional[Union[str, Path]]
        Path to automatically save computed dynamic backgrounds.
    fallback : Optional[FallbackBackgroundConfig]
        Fallback configuration when dynamic background is unavailable.
    """

    enabled: bool = Field(True, description="Enable background processing")
    type: BackgroundType = Field(
        BackgroundType.STATIC, description="Background processing type"
    )
    method: BackgroundMethod = BackgroundMethod.CONSTANT
    level: float = Field(0.0, ge=0.0, description="Constant background level")
    percentile: float = Field(
        5.0, ge=0.0, le=100.0, description="Percentile for dataset background"
    )
    outlier_threshold: float = Field(
        2.0, gt=0.0, description="Outlier rejection threshold (std devs)"
    )
    file_path: Optional[Union[str, Path]] = Field(
        None, description="Path to background file"
    )
    auto_save_path: Optional[Union[str, Path]] = Field(
        None, description="Auto-save path for computed backgrounds"
    )
    fallback: Optional[FallbackBackgroundConfig] = Field(
        None, description="Fallback background configuration"
    )

    @field_validator("percentile")
    def validate_percentile_range(cls, v):
        """Ensure percentile is in valid range."""
        if not 0.0 <= v <= 100.0:
            raise ValueError("percentile must be between 0 and 100")
        return v

    @field_validator("file_path")
    def validate_file_path(cls, v, info):
        """Validate file path when method is from_file."""
        if hasattr(info, "data"):
            method = info.data.get("method")
            bg_type = info.data.get("type")

            if (
                method == BackgroundMethod.FROM_FILE
                and bg_type == BackgroundType.STATIC
                and v is None
            ):
                raise ValueError(
                    'file_path required when type is "static" and method is "from_file"'
                )
        return v

    @field_validator("fallback")
    def validate_fallback_for_dynamic(cls, v, info):
        """Recommend fallback for dynamic backgrounds."""
        if hasattr(info, "data"):
            bg_type = info.data.get("type")
            if bg_type == BackgroundType.DYNAMIC and v is None:
                # This is just a warning, not an error - fallback is optional
                pass
        return v

    def is_static(self) -> bool:
        """Check if this is a static background configuration."""
        return self.type == BackgroundType.STATIC

    def is_dynamic(self) -> bool:
        """Check if this is a dynamic background configuration."""
        return self.type == BackgroundType.DYNAMIC

    def requires_image_batch(self) -> bool:
        """Check if this configuration requires an image batch for computation."""
        if not self.enabled:
            return False

        if self.is_static():
            return False

        # Dynamic backgrounds require image batches
        return self.method in [
            BackgroundMethod.PERCENTILE_DATASET,
            BackgroundMethod.TEMPORAL_MEDIAN,
            BackgroundMethod.MEDIAN,
            BackgroundMethod.MEAN,
            BackgroundMethod.OUTLIER_REJECTION,
        ]


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
    bilateral_d : Optional[int]
        Diameter for bilateral filter. If None, no bilateral filtering.
    bilateral_sigma_color : float
        Sigma for color space in bilateral filter.
    bilateral_sigma_space : float
        Sigma for coordinate space in bilateral filter.
    """

    gaussian_sigma: Optional[float] = Field(
        None, gt=0.0, description="Gaussian filter sigma"
    )
    median_kernel_size: Optional[int] = Field(
        None, gt=0, description="Median filter kernel size"
    )
    bilateral_d: Optional[int] = Field(
        None, gt=0, description="Bilateral filter diameter"
    )
    bilateral_sigma_color: float = Field(
        75.0, gt=0.0, description="Bilateral filter color sigma"
    )
    bilateral_sigma_space: float = Field(
        75.0, gt=0.0, description="Bilateral filter space sigma"
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

    enabled: bool = False
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
    transforms : Optional[TransformConfig]
        Geometric transformation configuration.
    """

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
    transforms: Optional[TransformConfig] = Field(
        None, description="Geometric transform settings"
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
