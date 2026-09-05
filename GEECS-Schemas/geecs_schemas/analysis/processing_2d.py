"""Camera (2D image) processing configuration — the ``image:`` section of a diagnostic.

These models describe how a raw camera frame is cleaned up before an
analyzer measures it: background subtraction, vignette correction,
masking, cropping, thresholding, filtering, normalization, and geometric
transforms.  The ``pipeline`` list on :class:`CameraConfig` is the single
source of truth for which steps run and in what order — a step runs if
and only if it appears there, and its matching section must be present.

Relocated from ``image_analysis.config.array2d_processing`` (GEECS-Schemas
0.19.0) so the whole diagnostic document validates with pydantic alone;
ImageAnalysis re-exports every name from here.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import Field, field_validator, model_validator

from geecs_schemas._base import SchemaModel


class BackgroundMethod(str, Enum):
    """How the primary background is obtained."""

    CONSTANT = "constant"
    FROM_FILE = "from_file"
    EDGE = "edge"


class BackgroundConfig(SchemaModel):
    """Subtract a background from every frame before analysis.

    Two stages: a primary background (a constant level, a saved frame, or
    the frame's own border), then an optional extra constant taken off
    afterwards.  Backgrounds that depend on the scan — another scan's
    dark frames, or an aggregate of this scan's own shots — are requested
    with ``scan.background_source`` on the diagnostic instead; the scan
    analyzer resolves that to a file and rewrites this section to
    ``from_file`` before shots are processed.
    """

    method: Optional[BackgroundMethod] = Field(
        None,
        description=(
            "Primary background: 'constant' subtracts constant_level, "
            "'from_file' subtracts the saved frame at file_path, 'edge' "
            "estimates the level from the frame border. Leave unset to "
            "apply only additional_constant."
        ),
    )
    file_path: Optional[Union[str, Path]] = Field(
        None,
        description=(
            "Saved background frame for method 'from_file'. May contain the "
            "{scan_dir} placeholder, filled in with the scan folder at run time."
        ),
    )
    constant_level: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Level subtracted for method 'constant'; also the fallback when a "
            "'from_file' background cannot be read."
        ),
    )
    additional_constant: float = Field(
        0.0,
        description="Extra constant subtracted after the primary background.",
    )
    edge_width: int = Field(
        1,
        ge=1,
        description="Border width in pixels averaged for method 'edge'.",
    )

    @model_validator(mode="after")
    def validate_background_source(self) -> "BackgroundConfig":
        """Require ``file_path`` when ``method`` is ``from_file``."""
        if self.method == BackgroundMethod.FROM_FILE and self.file_path is None:
            raise ValueError('file_path required when method is "from_file"')
        return self


class CrosshairConfig(SchemaModel):
    """One crosshair to mask out of the frame (a fiducial drawn on a screen, say)."""

    center: Tuple[int, int] = Field(
        ..., description="Pixel coordinates (x, y) of the crosshair centre."
    )
    width: int = Field(..., gt=0, description="Crosshair width in pixels.")
    height: int = Field(..., gt=0, description="Crosshair height in pixels.")
    thickness: int = Field(
        ..., gt=0, description="Thickness of the crosshair lines in pixels."
    )
    angle: float = Field(0.0, description="Rotation of the crosshair in degrees.")

    @field_validator("center")
    @classmethod
    def validate_center_coordinates(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Ensure center coordinates are non-negative."""
        x, y = v
        if x < 0 or y < 0:
            raise ValueError("Center coordinates must be non-negative")
        return v


class CrosshairMaskingConfig(SchemaModel):
    """Blank out one or more crosshairs so they do not count as signal."""

    crosshairs: List[CrosshairConfig] = Field(
        default_factory=list, description="The crosshairs to mask."
    )
    mask_value: float = Field(
        0.0, description="Pixel value written into the masked region."
    )

    def has_crosshairs(self) -> bool:
        """Whether any crosshairs are configured."""
        return len(self.crosshairs) > 0


class ROIConfig(SchemaModel):
    """Crop the frame to a rectangular region of interest, in pixels."""

    x_min: int = Field(0, ge=0, description="Left edge (inclusive), pixels.")
    x_max: int = Field(1024, gt=0, description="Right edge (exclusive), pixels.")
    y_min: int = Field(0, ge=0, description="Top edge (inclusive), pixels.")
    y_max: int = Field(1024, gt=0, description="Bottom edge (exclusive), pixels.")

    @field_validator("x_max")
    @classmethod
    def x_max_greater_than_min(cls, v: int, info) -> int:
        """Ensure x_max > x_min."""
        if "x_min" in info.data and v <= info.data["x_min"]:
            raise ValueError("x_max must be greater than x_min")
        return v

    @field_validator("y_max")
    @classmethod
    def y_max_greater_than_min(cls, v: int, info) -> int:
        """Ensure y_max > y_min."""
        if "y_min" in info.data and v <= info.data["y_min"]:
            raise ValueError("y_max must be greater than y_min")
        return v

    @property
    def width(self) -> int:
        """Width of the ROI in pixels."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Height of the ROI in pixels."""
        return self.y_max - self.y_min


class FilteringConfig(SchemaModel):
    """Smooth the frame with a Gaussian and/or a median filter."""

    gaussian_sigma: Optional[float] = Field(
        None,
        gt=0.0,
        description="Gaussian blur width in pixels; unset skips the Gaussian.",
    )
    median_kernel_size: Optional[int] = Field(
        None,
        gt=0,
        description="Median filter window (odd, in pixels); unset skips the median.",
    )

    @field_validator("median_kernel_size")
    @classmethod
    def validate_median_kernel_odd(cls, v: Optional[int]) -> Optional[int]:
        """Ensure median kernel size is odd."""
        if v is not None and v % 2 == 0:
            raise ValueError("median_kernel_size must be odd")
        return v


class TransformConfig(SchemaModel):
    """Rotate, flip, or undistort the frame."""

    rotation_angle: float = Field(
        0.0, description="Rotation in degrees, positive = counter-clockwise."
    )
    flip_horizontal: bool = Field(False, description="Mirror left-right.")
    flip_vertical: bool = Field(False, description="Mirror top-bottom.")
    distortion_correction: bool = Field(
        False, description="Apply the polynomial distortion correction."
    )
    distortion_coeffs: Optional[List[float]] = Field(
        None,
        description="Distortion coefficients; required when distortion_correction is on.",
    )

    @field_validator("rotation_angle")
    @classmethod
    def normalize_rotation_angle(cls, v: float) -> float:
        """Normalize rotation angle to the (-180, 180] range."""
        while v > 180:
            v -= 360
        while v <= -180:
            v += 360
        return v

    @field_validator("distortion_coeffs")
    @classmethod
    def validate_distortion_coeffs(
        cls, v: Optional[List[float]], info
    ) -> Optional[List[float]]:
        """Require coefficients when distortion correction is enabled."""
        if info.data.get("distortion_correction", False) and v is None:
            raise ValueError(
                "distortion_coeffs required when distortion_correction is True"
            )
        return v


class CircularMaskConfig(SchemaModel):
    """Keep (or discard) only the pixels inside a circle."""

    center: Tuple[int, int] = Field(
        (512, 512), description="Pixel coordinates (x, y) of the circle centre."
    )
    radius: int = Field(100, gt=0, description="Circle radius in pixels.")
    mask_outside: bool = Field(
        True,
        description="True masks everything outside the circle; False masks the inside.",
    )
    mask_value: float = Field(
        0.0, description="Pixel value written into the masked region."
    )

    @field_validator("center")
    @classmethod
    def validate_center_coordinates(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Ensure center coordinates are non-negative."""
        x, y = v
        if x < 0 or y < 0:
            raise ValueError("Center coordinates must be non-negative")
        return v


class VignetteMethod(str, Enum):
    """How the lens vignette is modelled."""

    RADIAL_POLYNOMIAL = "radial_polynomial"
    MAP_FILE = "map_file"


class VignetteConfig(SchemaModel):
    """Undo lens vignetting so the edges of the frame are not artificially dim.

    The radial-polynomial model follows the MATLAB-era convention: it is
    written in full-sensor coordinates, so a frame saved from a sensor
    sub-window needs the sensor size and the window's offset.
    """

    method: VignetteMethod = Field(
        VignetteMethod.RADIAL_POLYNOMIAL,
        description=(
            "'radial_polynomial' evaluates vgnt4/vgnt2/vgnt0 radially from the "
            "sensor centre; 'map_file' divides by a saved correction map."
        ),
    )
    full_width: Optional[int] = Field(
        None,
        gt=0,
        description="Full sensor width in pixels (required for radial_polynomial).",
    )
    full_height: Optional[int] = Field(
        None,
        gt=0,
        description="Full sensor height in pixels (required for radial_polynomial).",
    )
    x_offset: int = Field(
        0, ge=0, description="X offset of the saved frame within the full sensor."
    )
    y_offset: int = Field(
        0, ge=0, description="Y offset of the saved frame within the full sensor."
    )
    vgnt4: float = Field(0.0, description="4th-order radial coefficient.")
    vgnt2: float = Field(0.0, description="2nd-order radial coefficient.")
    vgnt0: float = Field(1.0, description="0th-order (centre) coefficient.")
    min_model_value: float = Field(
        1e-9,
        gt=0.0,
        description="Floor on the model value to avoid dividing by ~zero at the corners.",
    )
    map_file_path: Optional[Union[str, Path]] = Field(
        None, description="Saved .npy correction map for method 'map_file'."
    )

    @model_validator(mode="after")
    def validate_method_requirements(self) -> "VignetteConfig":
        """Validate required fields for the selected vignette method."""
        if self.method == VignetteMethod.MAP_FILE and self.map_file_path is None:
            raise ValueError("map_file_path required when method is 'map_file'")
        if self.method == VignetteMethod.RADIAL_POLYNOMIAL:
            if self.full_width is None or self.full_height is None:
                raise ValueError(
                    "full_width/full_height required when method is 'radial_polynomial'"
                )
        return self


class ThresholdMethod(str, Enum):
    """How the threshold level is chosen."""

    CONSTANT = "constant"
    PERCENTAGE_MAX = "percentage_max"


class ThresholdMode(str, Enum):
    """What happens to pixels on either side of the threshold."""

    BINARY = "binary"
    TO_ZERO = "to_zero"
    TRUNCATE = "truncate"
    TO_ZERO_INV = "to_zero_inv"
    TRUNCATE_INV = "truncate_inv"


class ThresholdingConfig(SchemaModel):
    """Suppress pixels below (or above) a level — the usual way to kill noise floor."""

    method: ThresholdMethod = Field(
        ThresholdMethod.CONSTANT,
        description=(
            "'constant' uses value as an absolute level; 'percentage_max' uses "
            "value as a percentage (0-100) of the frame maximum."
        ),
    )
    value: float = Field(
        100.0,
        ge=0.0,
        description="Threshold level: counts for 'constant', percent for 'percentage_max'.",
    )
    mode: ThresholdMode = Field(
        ThresholdMode.BINARY,
        description=(
            "'to_zero' zeroes pixels below the level (the common choice); "
            "'binary' makes a 0/1 mask; 'truncate' clips above the level; the "
            "_inv variants act on the other side."
        ),
    )
    invert: bool = Field(False, description="Invert the threshold operation.")

    @field_validator("value")
    @classmethod
    def validate_threshold_value(cls, v: float, info) -> float:
        """Validate threshold value based on method."""
        method = info.data.get("method")
        if method == ThresholdMethod.PERCENTAGE_MAX and not 0.0 <= v <= 100.0:
            raise ValueError("Percentage threshold value must be between 0 and 100")
        return v


class NormalizationMethod(str, Enum):
    """What the frame is divided by."""

    IMAGE_TOTAL = "image_total"
    IMAGE_MAX = "image_max"
    CONSTANT = "constant"
    DISTRIBUTE_VALUE = "distribute_value"


class NormalizationConfig(SchemaModel):
    """Rescale the frame so shots with different exposure or gain compare."""

    method: NormalizationMethod = Field(
        NormalizationMethod.IMAGE_TOTAL,
        description=(
            "'image_total' divides by the pixel sum, 'image_max' by the peak, "
            "'constant' by constant_value, 'distribute_value' divides by the "
            "sum then multiplies by constant_value."
        ),
    )
    constant_value: Optional[float] = Field(
        None,
        description="Divisor for 'constant' or multiplier for 'distribute_value'; required for those.",
    )

    @field_validator("constant_value")
    @classmethod
    def validate_constant_value(cls, v: Optional[float], info) -> Optional[float]:
        """Require a non-zero constant for the constant-based methods."""
        method = info.data.get("method")
        if method in (
            NormalizationMethod.CONSTANT,
            NormalizationMethod.DISTRIBUTE_VALUE,
        ):
            if v is None or v == 0:
                raise ValueError(
                    "constant_value must be non-zero when method is 'constant' "
                    "or 'distribute_value'"
                )
        return v


class ProcessingStepType(str, Enum):
    """The camera processing steps a pipeline may list."""

    BACKGROUND = "background"
    VIGNETTE = "vignette"
    CROSSHAIR_MASKING = "crosshair_masking"
    ROI = "roi"
    CIRCULAR_MASK = "circular_mask"
    THRESHOLDING = "thresholding"
    FILTERING = "filtering"
    NORMALIZATION = "normalization"
    TRANSFORMS = "transforms"


class CameraConfig(SchemaModel):
    """How a camera's frames are cleaned up before the analyzer measures them.

    Each processing section is optional and only runs when it is listed in
    ``pipeline``; the list order is the execution order.  An empty pipeline
    passes the raw frame straight to the analyzer.  Identity (which device,
    what to call the outputs) lives on the enclosing diagnostic, not here.
    """

    type: Literal["camera"] = Field(
        "camera", description="Marks this as a camera (2D image) section."
    )
    description: Optional[str] = Field(
        None, description="Free-text note about this camera / view."
    )
    metadata: Optional[Dict[str, Any]] = Field(  # documentary free-form fields
        None,
        description=(
            "Free-form documentation (location, notes, calibration constants). "
            "Nothing in the pipeline reads it; keep such notes here so the "
            "rest of the schema can stay strict."
        ),
    )
    bit_depth: int = Field(
        16, ge=8, le=32, description="Camera bit depth: 8, 10, 12, 14, 16 or 32."
    )

    roi: Optional[ROIConfig] = Field(None, description="Region-of-interest crop.")
    background: Optional[BackgroundConfig] = Field(
        None, description="Background subtraction."
    )
    crosshair_masking: Optional[CrosshairMaskingConfig] = Field(
        None, description="Crosshair masking."
    )
    circular_mask: Optional[CircularMaskConfig] = Field(
        None, description="Circular masking."
    )
    vignette: Optional[VignetteConfig] = Field(None, description="Vignette correction.")
    thresholding: Optional[ThresholdingConfig] = Field(
        None, description="Thresholding."
    )
    filtering: Optional[FilteringConfig] = Field(None, description="Smoothing filters.")
    normalization: Optional[NormalizationConfig] = Field(
        None, description="Intensity normalization."
    )
    transforms: Optional[TransformConfig] = Field(
        None, description="Rotation, flips, distortion correction."
    )
    pipeline: List[ProcessingStepType] = Field(
        default_factory=list,
        description=(
            "The steps that run, in order. A step runs only if listed here AND "
            "its section is present; empty means the raw frame is analyzed."
        ),
    )

    @field_validator("bit_depth")
    @classmethod
    def validate_bit_depth(cls, v: int) -> int:
        """Ensure bit depth is a common value."""
        valid_depths = [8, 10, 12, 14, 16, 32]
        if v not in valid_depths:
            raise ValueError(f"bit_depth must be one of {valid_depths}")
        return v

    def get_processing_configs(self) -> Dict[str, SchemaModel]:
        """Return every present processing section keyed by step name."""
        mapping = {
            "roi": self.roi,
            "background": self.background,
            "crosshair_masking": self.crosshair_masking,
            "circular_mask": self.circular_mask,
            "vignette": self.vignette,
            "thresholding": self.thresholding,
            "filtering": self.filtering,
            "normalization": self.normalization,
            "transforms": self.transforms,
        }
        return {key: cfg for key, cfg in mapping.items() if cfg is not None}

    @property
    def max_pixel_value(self) -> int:
        """Maximum pixel value for this camera's bit depth."""
        return (2**self.bit_depth) - 1

    @property
    def processing_dtype(self) -> str:
        """Recommended numpy dtype for processing (always float64)."""
        return "float64"

    @property
    def storage_dtype(self) -> str:
        """Recommended numpy dtype for storage, from the bit depth."""
        if self.bit_depth <= 8:
            return "uint8"
        if self.bit_depth <= 16:
            return "uint16"
        return "uint32"


__all__ = [
    "BackgroundConfig",
    "BackgroundMethod",
    "CameraConfig",
    "CircularMaskConfig",
    "CrosshairConfig",
    "CrosshairMaskingConfig",
    "FilteringConfig",
    "NormalizationConfig",
    "NormalizationMethod",
    "ProcessingStepType",
    "ROIConfig",
    "ThresholdMethod",
    "ThresholdMode",
    "ThresholdingConfig",
    "TransformConfig",
    "VignetteConfig",
    "VignetteMethod",
]
