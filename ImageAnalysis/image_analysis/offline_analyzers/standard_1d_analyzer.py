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
from image_analysis.types import Array1D, AnalyzerResultDict

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
    config_overrides : dict, optional
        Runtime overrides for configuration parameters
    """

    def __init__(
        self,
        line_config_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
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

        # Apply runtime overrides if provided
        if config_overrides:
            self._apply_config_overrides(config_overrides)
            logger.info("Applied configuration overrides: %s", config_overrides)

        # Store analyzer state
        self.line_config_name = line_config_name
        self.run_analyze_image_asynchronously = False

        # Initialize base class
        super().__init__()

    def _apply_config_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply runtime configuration overrides.

        Parameters
        ----------
        overrides : dict
            Dictionary of configuration overrides. Can contain Pydantic model
            instances or dictionaries for any configuration section.
        """
        for key, value in overrides.items():
            if hasattr(self.line_config, key):
                # Get the current attribute
                current_attr = getattr(self.line_config, key)

                # If value is a dict and current attr is a Pydantic model, update it
                if isinstance(value, dict) and hasattr(current_attr, "model_validate"):
                    # Merge with existing config
                    current_dict = current_attr.model_dump() if current_attr else {}
                    current_dict.update(value)
                    # Validate and set
                    model_class = type(current_attr)
                    setattr(
                        self.line_config, key, model_class.model_validate(current_dict)
                    )
                else:
                    # Direct assignment
                    setattr(self.line_config, key, value)

                logger.debug("Updated config.%s", key)
            else:
                logger.warning("Unknown config key in overrides: %s", key)

    def load_image(self, file_path: Path) -> Array1D:
        """Load 1D data from file using configured data loader.

        This method overrides the base class to use read_1d_data instead of
        read_imaq_image. It uses the data_loading configuration stored in
        self.line_config.

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

        logger.info(
            "Loaded 1D data from %s (type: %s, shape: %s)",
            file_path,
            data_config.data_type,
            result.data.shape,
        )

        return result.data  # Return Nx2 array

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Apply processing pipeline to 1D data.

        Parameters
        ----------
        data : np.ndarray
            Input Nx2 array (column 0: x, column 1: y)

        Returns
        -------
        np.ndarray
            Processed Nx2 array
        """
        return apply_line_processing_pipeline(data, self.line_config)

    def analyze_image(
        self, image: Array1D, auxiliary_data: Optional[Dict] = None
    ) -> AnalyzerResultDict:
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
        AnalyzerResultDict
            Dictionary containing processed data and metadata
        """
        # Apply processing pipeline
        processed = self.preprocess_data(image)

        # Build input parameters
        input_params = self._build_input_parameters(auxiliary_data)

        # Build and return result dictionary
        return self.build_return_dictionary(
            return_image=processed,
            input_parameters=input_params,
        )

    def _build_input_parameters(
        self, auxiliary_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Build input parameters dictionary.

        Parameters
        ----------
        auxiliary_data : dict, optional
            Additional auxiliary data to include

        Returns
        -------
        dict
            Input parameters dictionary
        """
        params = {
            "line_name": self.line_config.name,
            "data_type": self.line_config.data_loading.data_type,
            "config_name": self.line_config_name,
            "data_format": self.line_config.data_format,
        }

        if auxiliary_data:
            params.update(auxiliary_data)

        return params

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
            data_format = input_params_dict.get("data_format", "x vs y")
            # Try to parse format string like "wavelength (nm) vs intensity (a.u.)"
            if " vs " in data_format:
                parts = data_format.split(" vs ")
                ax.set_xlabel(parts[0].strip())
                ax.set_ylabel(parts[1].strip())
            else:
                ax.set_xlabel("X")
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
