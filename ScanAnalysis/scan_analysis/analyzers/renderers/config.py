"""Configuration models for scan analysis renderers.

This module provides Pydantic models for configuring renderer behavior,
including colormap options, visualization modes, and rendering parameters.
It also provides RenderContext for bundling data with metadata.
"""

from __future__ import annotations

from typing import Optional, Literal, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field

from image_analysis.types import ImageAnalyzerResult


class BaseRendererConfig(BaseModel):
    """Base configuration shared by all renderers.

    This class defines common rendering options applicable to both 1D and 2D
    visualizations, including colormap normalization, animation settings, and labels.

    Attributes
    ----------
    colormap_mode : str
        Colormap normalization mode:
        - "sequential": Standard 0 to max (default)
        - "diverging": Symmetric around zero for bipolar data
        - "custom": User-defined vmin/vmax
    cmap : str, optional
        Matplotlib colormap name (e.g., 'plasma', 'RdBu_r', 'coolwarm').
        If not provided, defaults are used based on colormap_mode.
    vmin : float, optional
        Minimum value for colormap normalization.
        - For "sequential": defaults to 0
        - For "diverging": computed as -vmax
        - For "custom": user-defined
    vmax : float, optional
        Maximum value for colormap normalization.
        - For "sequential": defaults to data max
        - For "diverging": computed as max(abs(data))
        - For "custom": user-defined
        Note: In Image2DRenderer, this replaces the legacy 'plot_scale' parameter.
    duration : float
        Animation frame duration in milliseconds (default: 100)
    dpi : int
        DPI for rendered figures (default: 150)
    xlabel : str, optional
        X-axis label for plots
    ylabel : str, optional
        Y-axis label for plots
    colorbar_label : str
        Colorbar label (default: "Intensity")

    """

    colormap_mode: Literal["sequential", "diverging", "custom"] = Field(
        default="sequential", description="Colormap normalization mode"
    )

    cmap: Optional[str] = Field(default=None, description="Matplotlib colormap name")

    vmin: Optional[float] = Field(
        default=None, description="Minimum value for colormap normalization"
    )

    vmax: Optional[float] = Field(
        default=None, description="Maximum value for colormap normalization"
    )

    duration: float = Field(
        default=100, description="Animation frame duration in milliseconds", gt=0
    )

    dpi: int = Field(default=150, description="DPI for rendered figures", gt=0)

    xlabel: Optional[str] = Field(default=None, description="X-axis label")

    ylabel: Optional[str] = Field(default=None, description="Y-axis label")

    colorbar_label: str = Field(default="Intensity", description="Colorbar label")


class Line1DRendererConfig(BaseRendererConfig):
    """Configuration for Line1DRenderer.

    Extends BaseRendererConfig with 1D-specific visualization options,
    including the choice between waterfall, overlay, and grid modes.

    Attributes
    ----------
    mode : str
        Visualization mode for summary figures:
        - "waterfall": Heatmap plot (x-axis: data x-values, y-axis: scan parameter)
        - "overlay": All bins plotted as lines on same axes
        - "grid": Subplot grid with one plot per bin
    cmap : str, optional
        Matplotlib colormap name. Defaults to 'plasma' for 1D visualizations.

    """

    mode: Literal["waterfall", "overlay", "grid"] = Field(
        default="waterfall", description="Visualization mode for summary figure"
    )

    cmap: Optional[str] = Field(
        default="plasma",
        description="Matplotlib colormap name (default: plasma for 1D)",
    )


class Image2DRendererConfig(BaseRendererConfig):
    """Configuration for Image2DRenderer.

    Extends BaseRendererConfig with 2D-specific visualization options,
    including figure sizing for grid montages and animations.

    Attributes
    ----------
    figsize : tuple of float
        Panel width and height in inches for grid montages (default: (6, 6))
    figsize_inches : float
        Width/height for square animation frames (default: 4.0)
    cmap : str, optional
        Matplotlib colormap name. Defaults to 'plasma' for 2D visualizations.

    Notes
    -----
    The `vmax` parameter from BaseRendererConfig replaces the legacy `plot_scale`
    parameter for backward compatibility.

    """

    figsize: Tuple[float, float] = Field(
        default=(6, 6), description="Panel width and height in inches for grid montages"
    )

    figsize_inches: float = Field(
        default=4.0, description="Width/height for square animation frames", gt=0
    )

    cmap: Optional[str] = Field(
        default="plasma",
        description="Matplotlib colormap name (default: plasma for 2D)",
    )


@dataclass
class RenderContext:
    """Complete context for rendering a single dataset.

    Bundles the complete ImageAnalyzerResult with scan metadata.
    Separates "what to render" (context) from "how to render" (config).

    This design preserves all analyzer output (render_data, custom fields, etc.)
    by passing the complete result object instead of extracting pieces.

    Attributes
    ----------
    result : ImageAnalyzerResult
        The complete analyzer result with all data, scalars, metadata, and render_data
    device_name : str
        Name of the device being analyzed
    identifier : int or str
        Unique identifier (bin_key, shot_number, etc.)
    scan_parameter : str, optional
        Name of the scan parameter (for scan context)
    parameter_value : float, optional
        Value of the scan parameter for this data
    """

    result: ImageAnalyzerResult
    device_name: str
    identifier: Union[int, str]
    scan_parameter: Optional[str] = None
    parameter_value: Optional[float] = None

    @classmethod
    def from_bin_result(
        cls,
        bin_key: int,
        bin_entry: Dict[str, Any],
        device_name: str,
        scan_parameter: Optional[str] = None,
        render_function: Optional[Callable[..., Any]] = None,
    ) -> "RenderContext":
        """Create RenderContext from binned_data entry.

        Passes the complete ImageAnalyzerResult to preserve all data
        including render_data and custom fields.

        Parameters
        ----------
        bin_key : int
            The bin number/key
        bin_entry : dict
            Entry from binned_data dict containing 'result' and 'value'
        device_name : str
            Name of the device
        scan_parameter : str, optional
            Name of the scan parameter
        render_function : Callable, optional
            Custom render function (overrides result.render_function if provided)

        Returns
        -------
        RenderContext
            Initialized context ready for rendering
        """
        result = bin_entry["result"]

        # Allow override of render function if provided
        if render_function is not None:
            result.render_function = render_function

        return cls(
            result=result,
            device_name=device_name,
            identifier=bin_key,
            scan_parameter=scan_parameter,
            parameter_value=bin_entry.get("value"),
        )

    @classmethod
    def from_analyzer_result(
        cls,
        shot_number: int,
        result: ImageAnalyzerResult,
        device_name: str,
        render_function: Optional[Callable[..., Any]] = None,
    ) -> "RenderContext":
        """Create RenderContext from single shot result.

        Passes the complete ImageAnalyzerResult to preserve all data
        including render_data and custom fields.

        Parameters
        ----------
        shot_number : int
            The shot number
        result : ImageAnalyzerResult
            Analyzer result object
        device_name : str
            Name of the device
        render_function : Callable, optional
            Custom render function (overrides result.render_function if provided)

        Returns
        -------
        RenderContext
            Initialized context ready for rendering
        """
        # Allow override of render function if provided
        if render_function is not None:
            result.render_function = render_function

        return cls(
            result=result,
            device_name=device_name,
            identifier=shot_number,
        )

    def get_metadata_kwargs(self) -> Dict[str, str]:
        """Extract visualization metadata (labels, units) from result.metadata.

        Returns
        -------
        dict
            Dictionary containing x_label, y_label, x_units, y_units if present
        """
        return {
            k: self.result.metadata.get(k)
            for k in ["x_label", "y_label", "x_units", "y_units"]
            if k in self.result.metadata
        }

    def get_filename(self, suffix: str, extension: str = "png") -> str:
        """Generate consistent filename for this context.

        Parameters
        ----------
        suffix : str
            Suffix to add to filename (e.g., 'processed', 'visual')
        extension : str, default='png'
            File extension

        Returns
        -------
        str
            Formatted filename
        """
        return f"{self.device_name}_{self.identifier}_{suffix}.{extension}"
