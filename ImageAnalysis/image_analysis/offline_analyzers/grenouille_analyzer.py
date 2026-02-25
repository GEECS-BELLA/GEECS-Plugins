"""Grenouille (FROG) Analyzer using the StandardAnalyzer framework.

This module provides a specialized analyzer for FROG pulse retrieval that
inherits from StandardAnalyzer. It adds Grenouille-specific capabilities:
- Pulse retrieval via FROG DLL
- Temporal and spectral FWHM extraction
- Retrieved trace and lineout export

The GrenouilleAnalyzer focuses purely on FROG-specific analysis while
leveraging the StandardAnalyzer for all image processing pipeline
functionality.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict
from pathlib import Path

import numpy as np
import pandas as pd

# Import the StandardAnalyzer parent class
from image_analysis.offline_analyzers.standard_analyzer import StandardAnalyzer

from image_analysis.algorithms.frog_dll_retrieval import (
    FrogDllRetrieval,
    FrogRetrievalConfig,
)

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

        # Validate analysis config (if present) into a typed model
        self.analysis_config = FrogRetrievalConfig.model_validate(
            self.camera_config.analysis or {}
        )

        logger.info(
            "Initialized GrenouilleAnalyzer with config '%s'", camera_config_name
        )

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
            processed_image,
            delt=self.analysis_config.delt,
            dellam=self.analysis_config.dellam,
            lam0=self.analysis_config.lam0,
            N=self.analysis_config.N,
            target_error=self.analysis_config.target_error,
            max_time_seconds=self.analysis_config.max_time_seconds,
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

        time_padded = np.pad(
            result.time, (0, max_len - time_len), constant_values=np.nan
        )
        temporal_int_padded = np.pad(
            result.temporal_intensity, (0, max_len - time_len), constant_values=np.nan
        )
        temporal_phase_padded = np.pad(
            result.temporal_phase, (0, max_len - time_len), constant_values=np.nan
        )

        wave_padded = np.pad(
            result.wavelength, (0, max_len - wave_len), constant_values=np.nan
        )
        spectral_int_padded = np.pad(
            result.spectral_intensity, (0, max_len - wave_len), constant_values=np.nan
        )
        spectral_phase_padded = np.pad(
            result.spectral_phase, (0, max_len - wave_len), constant_values=np.nan
        )

        df = pd.DataFrame(
            {
                "time_fs": time_padded,
                "temporal_intensity": temporal_int_padded,
                "temporal_phase": temporal_phase_padded,
                "wavelength_nm": wave_padded,
                "spectral_intensity": spectral_int_padded,
                "spectral_phase": spectral_phase_padded,
            }
        )

        file_path = auxiliary_data.get("file_path", None)
        if file_path is not None:
            # Generate output filename
            file_stem = Path(file_path).stem
            output_filename = f"{file_stem}_retrieved_lineouts.tsv"
            output_path = Path(file_path).parent / output_filename

            # Save to TSV
            df.to_csv(output_path, sep="\t", index=False)

        processed_image = result.retrieved_trace

        # Build result with beam-specific data
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=processed_image,
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
