"""Type definitions and TypedDicts for the ImageAnalysis package.

Defines NewType aliases for NumPy arrays and Pint quantities used throughout the
codebase, as well as the :class:`AnalyzerResultDict` TypedDict describing the
structure of results returned by analyzers.

Also provides the modern :class:`ImageAnalyzerResult` Pydantic model for new analyzers.
"""

from typing import NewType, TYPE_CHECKING, Any, Union, Optional, Callable, Dict, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

# exception to handle python 3.7
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

# Import numpy types for runtime use
try:
    from numpy.typing import NDArray
except ImportError:
    # Fallback for older numpy versions
    import numpy as np

    NDArray = np.ndarray

if TYPE_CHECKING:
    from pint import Quantity

    Array1D = NewType("Array1D", NDArray)  # Nx2 array for (x, y) pairs or 1D data
    Array2D = NewType("Array2D", NDArray)

    QuantityArray = NewType("QuantityArray", Quantity)
    QuantityArray2D = NewType("QuantityArray2D", Quantity)
else:
    # Runtime definitions for when TYPE_CHECKING is False
    Array1D = NDArray
    Array2D = NDArray
    QuantityArray = object
    QuantityArray2D = object


# ============================================================================
# LEGACY TYPES (LabVIEW Compatibility)
# ============================================================================


class AnalyzerResultDict(TypedDict):
    """TypedDict describing analyzer result dictionary.

    .. deprecated::
        Use :class:`ImageAnalyzerResult` for new analyzers.
        This TypedDict is maintained only for LabVIEW compatibility.
    """

    processed_image: Optional[NDArray]
    analyzer_return_dictionary: Optional[dict[str, Union[int, float]]]
    analyzer_return_lineouts: Optional[NDArray]
    analyzer_input_parameters: Optional[dict[str, Any]]


# ============================================================================
# MODERN TYPES (Pydantic Models)
# ============================================================================


class ImageAnalyzerResult(BaseModel):
    """Structured result from image/data analysis.

    This Pydantic model provides type-safe, validated results from image analyzers
    with support for three data types:

    - ``"scalars_only"`` (default): Returns only scalar metrics, no image data
    - ``"2d"``: Returns a 2D processed image plus scalars
    - ``"1d"``: Returns 1D line data (Nx2 array) plus scalars

    The model supports extensibility through:

    - ``render_data``: Dict for custom overlay/annotation data
    - ``extra="allow"``: Analyzers can add arbitrary custom fields

    Examples
    --------
    Scalars-only analyzer (metrics only):

    >>> result = ImageAnalyzerResult(
    ...     scalars={"peak_density": 1.5e19, "fwhm_mm": 2.3}
    ... )

    2D image analyzer (beam profile):

    >>> result = ImageAnalyzerResult(
    ...     data_type="2d",
    ...     processed_image=beam_image,
    ...     scalars={"centroid_x": 512, "centroid_y": 384}
    ... )
    >>> result.set_xy_projections(horiz_profile, vert_profile)

    1D line analyzer (spectrum):

    >>> result = ImageAnalyzerResult(
    ...     data_type="1d",
    ...     line_data=spectrum_array,
    ...     scalars={"peak_wavelength": 800.5}
    ... )

    Custom fields via extra="allow":

    >>> result = ImageAnalyzerResult(
    ...     data_type="2d",
    ...     processed_image=phase_map,
    ...     scalars={"rms_error": 0.05},
    ...     wavefront_3d=wavefront_data,  # Custom field
    ...     zernike_coeffs=coefficients    # Custom field
    ... )

    Attributes
    ----------
    data_type : Literal["1d", "2d", "scalars_only"]
        Type of primary data. Defaults to "scalars_only".
    processed_image : NDArray, optional
        Processed 2D image data. Required when data_type="2d".
    line_data : NDArray, optional
        Processed 1D data as Nx2 array [x, y]. Required when data_type="1d".
    scalars : Dict[str, float]
        Scalar analysis results (beam stats, peak positions, etc.).
    metadata : Dict[str, Any]
        Analysis configuration, parameters, and context.
    render_data : Dict[str, Any]
        Custom data for rendering (projections, overlays, annotations).
    render_function : Callable, optional
        Custom render function for this analyzer's results.
    """

    # === CORE FIELDS ===
    data_type: Literal["1d", "2d", "scalars_only"] = "scalars_only"

    # PRIMARY DATA (optional based on data_type)
    processed_image: Optional[NDArray] = None
    line_data: Optional[NDArray] = None

    # ANALYSIS OUTPUTS
    scalars: Dict[str, float] = Field(default_factory=dict)

    # METADATA
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # RENDERING
    render_data: Dict[str, Any] = Field(default_factory=dict)
    render_function: Optional[Callable] = None

    # === CONFIGURATION ===
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow numpy arrays, callables
        extra="allow",  # Allow custom fields without model modification
    )

    # === VALIDATION ===
    @field_validator("processed_image")
    @classmethod
    def validate_2d_data(cls, v, info):
        """Ensure 2D data is provided when data_type is '2d'."""
        if info.data.get("data_type") == "2d" and v is None:
            raise ValueError("processed_image required when data_type='2d'")
        return v

    @field_validator("line_data")
    @classmethod
    def validate_1d_data(cls, v, info):
        """Ensure 1D data is provided when data_type is '1d'."""
        if info.data.get("data_type") == "1d" and v is None:
            raise ValueError("line_data required when data_type='1d'")
        return v

    # === HELPER METHODS ===
    def get_primary_data(self) -> Optional[NDArray]:
        """Get primary data regardless of type.

        Returns
        -------
        NDArray or None
            - For "2d": returns processed_image
            - For "1d": returns line_data
            - For "scalars_only": returns None
        """
        if self.data_type == "2d":
            return self.processed_image
        elif self.data_type == "1d":
            return self.line_data
        else:  # scalars_only
            return None

    def has_image_data(self) -> bool:
        """Check if this result has renderable image/line data.

        Returns
        -------
        bool
            True if data_type is "1d" or "2d", False for "scalars_only"
        """
        return self.data_type in ["1d", "2d"]

    def set_xy_projections(self, horizontal: NDArray, vertical: NDArray) -> None:
        """Set horizontal and vertical projections for 2D image overlay.

        This is a standard pattern for 2D analyzers to provide projection
        data for rendering overlays on images.

        Parameters
        ----------
        horizontal : NDArray
            Horizontal (x-axis) projection of the image
        vertical : NDArray
            Vertical (y-axis) projection of the image
        """
        self.render_data["xy_projections"] = {
            "horizontal": horizontal,
            "vertical": vertical,
        }

    def get_xy_projections(self) -> Optional[tuple[NDArray, NDArray]]:
        """Get xy projections if they exist.

        Returns
        -------
        tuple of (horizontal, vertical) NDArrays, or None
            The projection data if set, otherwise None
        """
        if "xy_projections" in self.render_data:
            proj = self.render_data["xy_projections"]
            return proj["horizontal"], proj["vertical"]
        return None
