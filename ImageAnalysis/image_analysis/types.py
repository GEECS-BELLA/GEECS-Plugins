"""Type definitions and TypedDicts for the ImageAnalysis package.

Defines NewType aliases for NumPy arrays and Pint quantities used throughout the
codebase, as well as the :class:`AnalyzerResultDict` TypedDict describing the
structure of results returned by analyzers.

Also provides the modern :class:`ImageAnalyzerResult` Pydantic model for new analyzers.
"""

from typing import (
    NewType,
    TYPE_CHECKING,
    Any,
    Union,
    Optional,
    Callable,
    Dict,
    Literal,
    List,
)
from pydantic import BaseModel, Field, ConfigDict, field_validator
import logging

# exception to handle python 3.7
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

# Import numpy types for runtime use
try:
    from numpy.typing import NDArray
    import numpy as np
except ImportError:
    # Fallback for older numpy versions
    import numpy as np

    NDArray = np.ndarray

logger = logging.getLogger(__name__)

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

# Type alias for render_data values
RenderDataValue = Union[float, List[float], NDArray]


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
    render_data: Dict[str, RenderDataValue] = Field(default_factory=dict)
    """Rendering overlays and annotations. Flat dict of numeric data:

    - Scalars: {"threshold": 0.5, "roi_size": 100}
    - Lists: {"peak_positions": [120, 340, 560]}
    - Arrays: {"horizontal_projection": np.ndarray, "vertical_projection": np.ndarray}
    """
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
        self.render_data["horizontal_projection"] = horizontal
        self.render_data["vertical_projection"] = vertical

    def get_xy_projections(self) -> Optional[tuple[NDArray, NDArray]]:
        """Get xy projections if they exist.

        Returns
        -------
        tuple of (horizontal, vertical) NDArrays, or None
            The projection data if set, otherwise None
        """
        h = self.render_data.get("horizontal_projection")
        v = self.render_data.get("vertical_projection")
        if h is not None and v is not None:
            return h, v
        return None

    @classmethod
    def average(cls, results: List["ImageAnalyzerResult"]) -> "ImageAnalyzerResult":
        """Average multiple ImageAnalyzerResult objects.

        Averages all numerical data using nanmean to handle NaN values.
        Validates that render_data keys are consistent across results.

        Parameters
        ----------
        results : list[ImageAnalyzerResult]
            List of results to average (must all have same data_type)

        Returns
        -------
        ImageAnalyzerResult
            New result with averaged data

        Raises
        ------
        ValueError
            If results list is empty or data_types don't match

        Warns
        -----
        If render_data keys are inconsistent across results
        """
        if not results:
            raise ValueError("Cannot average empty list of results")

        # Verify all have same data_type
        data_types = {r.data_type for r in results}
        if len(data_types) > 1:
            raise ValueError(
                f"Cannot average results with different data_types: {data_types}"
            )

        data_type = results[0].data_type

        # Average primary data using nanmean
        if data_type == "2d":
            images = [
                r.processed_image for r in results if r.processed_image is not None
            ]
            avg_image = np.nanmean(images, axis=0) if images else None
            avg_line = None
        elif data_type == "1d":
            lines = [r.line_data for r in results if r.line_data is not None]
            avg_line = np.nanmean(lines, axis=0) if lines else None
            avg_image = None
        else:  # scalars_only
            avg_image = None
            avg_line = None

        # Average scalars using nanmean
        all_scalars = [r.scalars for r in results if r.scalars]
        avg_scalars = {}
        if all_scalars:
            for key in all_scalars[0].keys():
                values = [s[key] for s in all_scalars if key in s]
                if not values:
                    # Key missing from all results - skip it
                    continue
                # Check if all values are NaN - if so, return NaN without warning
                if all(np.isnan(v) for v in values):
                    avg_scalars[key] = np.nan
                else:
                    avg_scalars[key] = float(np.nanmean(values))

        # Check render_data key consistency and warn if inconsistent
        all_keys_sets = [set(r.render_data.keys()) for r in results]
        expected_keys = all_keys_sets[0] if all_keys_sets else set()
        inconsistent = any(keys != expected_keys for keys in all_keys_sets)

        if inconsistent:
            all_unique_keys = set().union(*all_keys_sets)
            logger.warning(
                f"Inconsistent render_data keys across results. "
                f"Expected {expected_keys}, found union of {all_unique_keys}. "
                f"This may indicate a poorly implemented analyzer."
            )

        # Average render_data using union of all keys
        avg_render_data = {}
        all_keys = set().union(*all_keys_sets) if all_keys_sets else set()

        for key in all_keys:
            values = [r.render_data[key] for r in results if key in r.render_data]

            if not values:
                continue

            first_val = values[0]
            if isinstance(first_val, np.ndarray):
                avg_render_data[key] = np.nanmean(values, axis=0)
            elif isinstance(first_val, list):
                avg_render_data[key] = np.nanmean(values, axis=0).tolist()
            elif isinstance(first_val, (int, float)):
                avg_render_data[key] = float(np.nanmean(values))

        return cls(
            data_type=data_type,
            processed_image=avg_image,
            line_data=avg_line,
            scalars=avg_scalars,
            metadata=results[0].metadata.copy() if results[0].metadata else {},
            render_data=avg_render_data,
            render_function=results[0].render_function,
        )
