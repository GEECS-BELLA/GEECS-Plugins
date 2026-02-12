"""Beam Profile Analyzer using the StandardAnalyzer framework.

This module provides a specialized analyzer for beam profile analysis that inherits
from StandardAnalyzer. It adds beam-specific capabilities:
- Beam statistics calculation (centroid, width, height, FWHM)
- Gaussian fitting parameters
- Beam quality metrics
- Specialized beam rendering with overlays
- Lineout generation and analysis

The BeamAnalyzer focuses purely on beam-specific analysis while leveraging
the StandardAnalyzer for all image processing pipeline functionality.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the StandardAnalyzer parent class
from image_analysis.offline_analyzers.standard_analyzer import StandardAnalyzer

from image_analysis.algorithms.frog_dll_retrieval import FrogDllRetrieval

from image_analysis.types import ImageAnalyzerResult

logger = logging.getLogger(__name__)


class GrenouilleAnalyzer(StandardAnalyzer):
    """
    Beam profile analyzer using the StandardAnalyzer framework.

    Parameters
    ----------
    camera_config_name : str
        Name of the camera configuration to load (e.g., "undulator_exit_cam")
    """

    def __init__(
        self,
        camera_config_name: str,
    ):
        """Initialize the beam analyzer with external configuration.

        Parameters
        ----------
        camera_config_name : str
            Name of the camera configuration to load (e.g., "UC_ALineEBeam3")
        """
        # Initialize parent class
        super().__init__(camera_config_name)

        self.retrieval = FrogDllRetrieval.from_config()

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """
        Run complete beam analysis using the processing pipeline.

        This method extends the StandardAnalyzer's analyze_image method to add
        beam-specific analysis including statistics calculation and lineouts.

        Parameters
        ----------
        image : np.ndarray
            Input image to analyze
        auxiliary_data : dict, optional
            Additional data including file path and preprocessing flags

        Returns
        -------
        ImageAnalyzerResult
            Structured result containing processed image, beam statistics, and metadata
        """
        initial_result: ImageAnalyzerResult = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        processed_image = initial_result.processed_image

        result = self.retrieval.retrieve_pulse(
            # gr_trace, delt=0.89368, dellam=-0.0798546, lam0=400., N=512, target_error=.001, max_time_seconds=5
            processed_image,
            delt=0.85,
            dellam=-0.085,
            lam0=400.0,
            N=512,
            target_error=0.001,
            max_time_seconds=5,
        )

        scalar_results = {
            f"{self.camera_name}_temporal_fwhm": result.temporal_fwhm,
            f"{self.camera_name}_spectral_fwhm": result.spectral_fwhm,
            f"{self.camera_name}_frog_error": result.frog_error,
        }

        # Pad shorter array with NaN to match lengths
        time_len = len(result.time)
        wave_len = len(result.wavelength)
        max_len = max(time_len, wave_len)

        time_padded = np.pad(result.time, (0, max_len - time_len), constant_values=np.nan)
        temporal_int_padded = np.pad(result.temporal_intensity, (0, max_len - time_len), constant_values=np.nan)
        temporal_phase_padded = np.pad(result.temporal_phase, (0, max_len - time_len), constant_values=np.nan)

        wave_padded = np.pad(result.wavelength, (0, max_len - wave_len), constant_values=np.nan)
        spectral_int_padded = np.pad(result.spectral_intensity, (0, max_len - wave_len), constant_values=np.nan)
        spectral_phase_padded = np.pad(result.spectral_phase, (0, max_len - wave_len), constant_values=np.nan)

        df = pd.DataFrame({
            'time_fs': time_padded,
            'temporal_intensity': temporal_int_padded,
            'temporal_phase': temporal_phase_padded,
            'wavelength_nm': wave_padded,
            'spectral_intensity': spectral_int_padded,
            'spectral_phase': spectral_phase_padded,
        })

        file_path = auxiliary_data.get("file_path", None)
        if file_path is not None:
            # Generate output filename
            file_stem = Path(file_path).stem
            output_filename = f"{file_stem}_retrieved_lineouts.tsv"
            output_path = Path(file_path).parent / output_filename

            # Save to TSV
            df.to_csv(output_path, sep="\t", index=False)

        # Build result with beam-specific data
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=result.retrieved_trace,
            scalars=scalar_results,
            metadata=initial_result.metadata,
        )

        # Add projection overlays for rendering
        if processed_image is not None:
            result.render_data = {
                "horizontal_projection": processed_image.sum(axis=0),
                "vertical_projection": processed_image.sum(axis=1),
            }

        return result

    @staticmethod
    def render_image(
        result: ImageAnalyzerResult,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "plasma",
        figsize: Tuple[float, float] = (4, 4),
        dpi: int = 150,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render beam image with beam-specific overlays.

        This method provides specialized rendering for beam analysis including
        XY projection lineouts and beam centroid markers using composable
        overlay functions.
        """
        from image_analysis.tools.rendering import (
            base_render_image,
            add_xy_projections,
            add_marker,
        )

        # Base rendering
        fig, ax = base_render_image(
            result=result,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
        )

        # Add beam-specific overlays
        add_xy_projections(ax, result)
        add_marker(ax, (100, 100), size=1, color="blue")

        return fig, ax

    def visualize(
        self,
        results: ImageAnalyzerResult,
        *,
        show: bool = True,
        close: bool = True,
        ax: Optional[plt.Axes] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "plasma",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render a visualization of the analyzed image with beam overlays.

        This is a simple convenience wrapper that calls :meth:`render_image`
        and optionally shows or closes the figure.
        """
        # Call render_image with all parameters
        fig, ax = self.render_image(
            result=results,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
        )

        if show:
            plt.show()
        if close:
            plt.close(fig)

        return fig, ax
