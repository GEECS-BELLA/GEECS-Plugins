"""Configuration models for scan analysis renderers.

This module provides Pydantic models for configuring renderer behavior,
including colormap options, visualization modes, and rendering parameters.
It also provides RenderContext for bundling data with metadata.
"""

from typing import Optional, Literal, Tuple, Dict, Any, Union, List, Callable
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field


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

    panel_size: Tuple[float, float] = Field(
        default=(4.0, 4.0),
        description="Desired panel width/height (inches) before clamping to max figure width",
    )

    figsize_inches: float = Field(
        default=4.0, description="Width/height for square animation frames", gt=0
    )

    cmap: Optional[str] = Field(
        default="plasma",
        description="Matplotlib colormap name (default: plasma for 2D)",
    )

    max_figure_width: float = Field(
        default=7.0,
        description="Maximum summary figure width in inches (Word-friendly)",
        gt=0,
    )

    downsample_factor: Optional[int] = Field(
        default=None,
        description="Integer factor to downsample images before plotting (>=1)",
        ge=1,
    )

    summary_dpi: Optional[int] = Field(
        default=None,
        description="Override DPI for summary figure (otherwise BaseRendererConfig.dpi)",
        gt=0,
    )

    bin_stride: int = Field(
        default=1,
        description="Include every Nth bin in the summary grid (1 = use all bins)",
        ge=1,
    )

    max_columns: Optional[int] = Field(
        default=None,
        description="Maximum number of panels per row in summary grids",
        ge=1,
    )

    min_panel_width: float = Field(
        default=2.5,
        description="Minimum acceptable panel width (inches) before adding extra rows",
        gt=0,
    )

    font_size: float = Field(
        default=10.0,
        description="Base font size (points) for titles, labels, and ticks",
        gt=0,
    )

    analyzer_render_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments forwarded to analyzer-specific render functions",
    )


@dataclass
class RenderContext:
    """Complete context for rendering a single dataset.

    Bundles data, metadata, and identification info needed for rendering.
    Separates "what to render" (context) from "how to render" (config).

    Attributes
    ----------
    data : np.ndarray
        The processed data to render (1D or 2D array)
    input_parameters : dict
        Analyzer input parameters including labels, units, etc.
    device_name : str
        Name of the device being analyzed
    identifier : int or str
        Unique identifier (bin_key, shot_number, etc.)
    scan_parameter : str, optional
        Name of the scan parameter (for scan context)
    parameter_value : float, optional
        Value of the scan parameter for this data
    overlay_lineouts : list[np.ndarray], optional
        Lineouts to overlay on rendered images
    render_function : callable, optional
        Custom render function (e.g., ImageAnalyzer.render_image) to use when
        creating visualizations
    """

    data: np.ndarray
    input_parameters: Dict[str, Any]
    device_name: str
    identifier: Union[int, str]
    scan_parameter: Optional[str] = None
    parameter_value: Optional[float] = None
    overlay_lineouts: Optional[List[np.ndarray]] = None
    render_function: Optional[Callable[..., Any]] = None

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

        Handles both 2D data (in processed_image) and 1D data (in lineouts).

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

        Returns
        -------
        RenderContext
            Initialized context ready for rendering
        """
        result = bin_entry["result"]

        # Extract data from either processed_image (2D) or lineouts (1D)
        data = result.get("processed_image")
        lineouts = result.get("analyzer_return_lineouts")

        if data is None and lineouts is not None:
            # Reconstruct Nx2 array from lineouts [x_array, y_array]
            data = np.column_stack([lineouts[0], lineouts[1]])

        return cls(
            data=data,
            input_parameters=result.get("analyzer_input_parameters", {}),
            device_name=device_name,
            identifier=bin_key,
            scan_parameter=scan_parameter,
            parameter_value=bin_entry.get("value"),
            overlay_lineouts=lineouts,
            render_function=render_function,
        )

    @classmethod
    def from_analyzer_result(
        cls,
        shot_number: int,
        result: Dict[str, Any],
        device_name: str,
        render_function: Optional[Callable[..., Any]] = None,
    ) -> "RenderContext":
        """Create RenderContext from single shot result.

        Parameters
        ----------
        shot_number : int
            The shot number
        result : dict
            Analyzer result dict
        device_name : str
            Name of the device

        Returns
        -------
        RenderContext
            Initialized context ready for rendering
        """
        data = result.get("processed_image")
        lineouts = result.get("analyzer_return_lineouts")
        if data is None and lineouts is not None:
            data = np.column_stack([lineouts[0], lineouts[1]])

        return cls(
            data=data,
            input_parameters=result.get("analyzer_input_parameters", {}),
            device_name=device_name,
            identifier=shot_number,
            overlay_lineouts=lineouts,
            render_function=render_function,
        )

    def get_metadata_kwargs(self) -> Dict[str, str]:
        """Extract visualization metadata (labels, units) from input_parameters.

        Returns
        -------
        dict
            Dictionary containing x_label, y_label, x_units, y_units if present
        """
        return {
            k: self.input_parameters.get(k)
            for k in ["x_label", "y_label", "x_units", "y_units"]
            if k in self.input_parameters
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
