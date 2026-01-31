"""Standard 1D Data Analyzer using configurable processing pipeline.

This module provides a general-purpose foundation for 1D data analysis (spectra,
lineouts, traces, etc.) using:
- Type-safe line configuration via Pydantic models
- Externalized configuration in YAML files
- Unified processing pipeline
- Clean separation of concerns

This class is designed to be inherited by specialized analyzers that add
domain-specific analysis capabilities (e.g., peak finding, spectral analysis).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

from image_analysis.base import ImageAnalyzer
from image_analysis.config_loader import load_line_config
from image_analysis.data_1d_utils import read_1d_data
from image_analysis.processing.array1d.pipeline import apply_line_processing_pipeline
from image_analysis.types import Array1D, ImageAnalyzerResult

logger = logging.getLogger(__name__)


class Standard1DAnalyzer(ImageAnalyzer):
    """
    Standard 1D data analyzer with configurable processing pipeline.

    This analyzer provides a general-purpose foundation for 1D data analysis using:
    - Type-safe line configuration via Pydantic models
    - Externalized configuration in YAML files
    - Unified processing pipeline
    - Clean separation of concerns

    This class is designed to be inherited by specialized analyzers that add
    domain-specific analysis capabilities (e.g., spectral analysis, peak finding).

    Parameters
    ----------
    line_config_name : str
        Name of the line configuration to load (e.g., "example_spectrum_1d")
    """

    def __init__(
        self,
        line_config_name: str,
    ):
        """Initialize the standard 1D analyzer with external configuration."""
        # Load line configuration
        try:
            self.line_config = load_line_config(line_config_name)
            logger.info("Loaded configuration for line: %s", self.line_config.name)
        except Exception as e:
            raise ValueError(
                f"Failed to load line configuration '{line_config_name}': {e}"
            ) from e

        # Store analyzer state
        self.line_config_name = line_config_name
        self.run_analyze_image_asynchronously = False

        # Storage for metadata from read_1d_data
        self.data_metadata: Optional[Dict[str, str]] = None

        # Initialize base class
        super().__init__()

    def load_image(self, file_path: Path) -> Array1D:
        """Load 1D data from file using configured data loader.

        This method overrides the base class to use read_1d_data instead of
        read_imaq_image. It uses the data_loading configuration stored in
        self.line_config and preserves metadata for later use.

        Parameters
        ----------
        file_path : Path
            Path to the data file

        Returns
        -------
        Array1D
            Nx2 array where column 0 is x values and column 1 is y values
        """
        # Extract Data1DConfig from line_config
        data_config = self.line_config.data_loading

        # Load data using read_1d_data
        result = read_1d_data(file_path, data_config)

        # Store metadata for use in _build_input_parameters
        self.data_metadata = {
            "x_units": result.x_units,
            "y_units": result.y_units,
            "x_label": result.x_label,
            "y_label": result.y_label,
        }

        logger.info(
            "Loaded 1D data from %s (type: %s, shape: %s)",
            file_path,
            data_config.data_type,
            result.data.shape,
        )

        return result.data  # Return Nx2 array

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Apply processing pipeline to 1D data, including optional axis scaling.

        Parameters
        ----------
        data : np.ndarray
            Input Nx2 array (column 0: x, column 1: y)

        Returns
        -------
        np.ndarray
            Processed Nx2 array with optional axis scaling applied
        """
        # Apply scale factors first (before other processing)
        # This way all downstream processing uses the scaled values
        scaled_data = data.copy()

        if self.line_config.x_scale_factor is not None:
            scaled_data[:, 0] *= self.line_config.x_scale_factor

        if self.line_config.y_scale_factor is not None:
            scaled_data[:, 1] *= self.line_config.y_scale_factor

        # Then apply the rest of the processing pipeline
        return apply_line_processing_pipeline(scaled_data, self.line_config)

    def analyze_image(
        self, image: Array1D, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """Analyze 1D data.

        Note: Method is named 'analyze_image' for compatibility with base class,
        but it processes 1D data.

        Parameters
        ----------
        image : Array1D
            Input Nx2 array (column 0: x, column 1: y)
        auxiliary_data : dict, optional
            Additional data for analysis

        Returns
        -------
        ImageAnalyzerResult
            Structured result containing processed 1D data and metadata
        """
        # If metadata not already cached AND file_path is available, load it
        if (
            self.data_metadata is None
            and auxiliary_data
            and "file_path" in auxiliary_data
        ):
            self.load_image(auxiliary_data["file_path"])  # Populates self.data_metadata

        # Apply processing pipeline
        processed = self.preprocess_data(image)

        # Build input parameters
        input_params = self._build_input_parameters(auxiliary_data)

        # Build and return result with 1D data
        result = ImageAnalyzerResult(
            data_type="1d",
            line_data=processed,  # Nx2 array
            scalars={},  # No scalars by default, subclasses can add them
            metadata=input_params,
        )

        return result

    def _build_input_parameters(
        self, auxiliary_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build input parameters dictionary including metadata from data loading.

        This method implements unit resolution where config units take precedence
        over file metadata units, allowing for override/fallback functionality.

        Parameters
        ----------
        auxiliary_data : dict, optional
            Additional auxiliary data to include

        Returns
        -------
        dict
            Input parameters dictionary with metadata and resolved units
        """
        params = {
            "line_name": self.line_config.name,
            "data_type": self.line_config.data_loading.data_type,
            "config_name": self.line_config_name,
            "data_format": self.line_config.data_format,
        }

        # Add metadata from read_1d_data if available, with unit resolution
        if self.data_metadata is not None:
            # Unit resolution: config units override file metadata
            # This allows config to provide fallback units (for NPY files)
            # or override incorrect file metadata
            x_units = self.line_config.x_units or self.data_metadata.get("x_units")
            y_units = self.line_config.y_units or self.data_metadata.get("y_units")

            params.update(
                {
                    "x_units": x_units,
                    "y_units": y_units,
                    "x_label": self.data_metadata.get("x_label"),
                    "y_label": self.data_metadata.get("y_label"),
                }
            )
        else:
            # Fallback to config-only units if file wasn't loaded yet
            params.update(
                {
                    "x_units": self.line_config.x_units,
                    "y_units": self.line_config.y_units,
                }
            )

        if auxiliary_data:
            params.update(auxiliary_data)

        return params

    def render_image(
        self,
        result: ImageAnalyzerResult,
        figsize: Tuple[float, float] = (8, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
        **plot_kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render 1D data from ImageAnalyzerResult.

        Parameters
        ----------
        result : ImageAnalyzerResult
            Complete analysis result with line_data and metadata
        figsize : tuple, default=(8, 4)
            Figure size in inches (width, height)
        dpi : int, default=150
            Figure DPI
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure.
        **plot_kwargs
            Additional keyword arguments passed to ax.plot()

        Returns
        -------
        tuple
            (figure, axes) matplotlib objects

        Raises
        ------
        ValueError
            If result.data_type is not "1d" or line_data is None.
        """
        # Validate input
        if result.data_type != "1d":
            raise ValueError(
                f"render_image requires data_type='1d', got '{result.data_type}'"
            )

        if result.line_data is None:
            raise ValueError("line_data cannot be None for 1d data_type")

        # Create figure/axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
        else:
            fig = ax.figure

        # Extract data
        x_data = result.line_data[:, 0]
        y_data = result.line_data[:, 1]

        # Plot
        ax.plot(x_data, y_data, **plot_kwargs)

        # Set labels from metadata
        metadata = result.metadata or {}

        # Build axis labels with units
        x_label = metadata.get("x_label", "X")
        x_units = metadata.get("x_units")
        if x_units:
            x_label = f"{x_label} ({x_units})"

        y_label = metadata.get("y_label", "Y")
        y_units = metadata.get("y_units")
        if y_units:
            y_label = f"{y_label} ({y_units})"

        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

        # Optional title from line name
        line_name = metadata.get("line_name")
        if line_name:
            ax.set_title(line_name)

        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def render_data(
        data: np.ndarray,
        analysis_results_dict: Optional[Dict] = None,
        input_params_dict: Optional[Dict] = None,
        figsize: Tuple[float, float] = (8, 4),
        ax: Optional[plt.Axes] = None,
        **plot_kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render 1D data as a line plot.

        Parameters
        ----------
        data : np.ndarray
            Nx2 array to plot (column 0: x, column 1: y)
        analysis_results_dict : dict, optional
            Analysis results (currently unused, for future extensions)
        input_params_dict : dict, optional
            Input parameters including labels and format info
        figsize : tuple, default=(8, 4)
            Figure size in inches (width, height)
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure.
        **plot_kwargs
            Additional keyword arguments passed to ax.plot()

        Returns
        -------
        tuple
            (figure, axes) matplotlib objects
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Extract x and y data
        x_data = data[:, 0]
        y_data = data[:, 1]

        # Plot
        ax.plot(x_data, y_data, **plot_kwargs)

        # Set labels if available
        if input_params_dict:
            # Try to use metadata labels first (from read_1d_data)
            x_label = input_params_dict.get("x_label")
            y_label = input_params_dict.get("y_label")
            x_units = input_params_dict.get("x_units")
            y_units = input_params_dict.get("y_units")

            # Build axis labels with units if available
            if x_label:
                xlabel = f"{x_label} ({x_units})" if x_units else x_label
                ax.set_xlabel(xlabel)
            elif "data_format" in input_params_dict:
                # Fallback to parsing data_format string
                data_format = input_params_dict.get("data_format", "x vs y")
                if " vs " in data_format:
                    parts = data_format.split(" vs ")
                    ax.set_xlabel(parts[0].strip())
                else:
                    ax.set_xlabel("X")
            else:
                ax.set_xlabel("X")

            if y_label:
                ylabel = f"{y_label} ({y_units})" if y_units else y_label
                ax.set_ylabel(ylabel)
            elif "data_format" in input_params_dict:
                # Fallback to parsing data_format string
                data_format = input_params_dict.get("data_format", "x vs y")
                if " vs " in data_format:
                    parts = data_format.split(" vs ")
                    ax.set_ylabel(parts[1].strip())
                else:
                    ax.set_ylabel("Y")
            else:
                ax.set_ylabel("Y")

            # Add title if line name is available
            line_name = input_params_dict.get("line_name")
            if line_name:
                ax.set_title(line_name)

        # Grid
        ax.grid(True, alpha=0.3)

        return fig, ax

    @property
    def camera_name(self) -> str:
        """Return line config name for ScanAnalyzer compatibility.

        This property provides compatibility with the ScanAnalyzer module which
        expects a camera_name attribute.

        Returns
        -------
        str
            The line configuration name
        """
        return self.line_config.name
