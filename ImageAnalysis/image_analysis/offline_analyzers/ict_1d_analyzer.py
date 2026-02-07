"""ICT (Integrated Current Transformer) specialized 1D analyzer.

This module provides ICT1DAnalyzer, a specialized analyzer for measuring charge
from oscilloscope traces using the ICT signal processing pipeline.

The analyzer inherits from Standard1DAnalyzer and adds:
- Automatic dt extraction from TDMS metadata
- ICT-specific signal processing (Butterworth filter, sinusoid removal)
- Charge calculation in picocoulombs
- Single output value (charge_pC) for downstream scan analyzer
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np

from image_analysis.offline_analyzers.standard_1d_analyzer import Standard1DAnalyzer
from image_analysis.types import Array1D, ImageAnalyzerResult
from image_analysis.algorithms import ict_algorithms

logger = logging.getLogger(__name__)


class ICT1DAnalyzer(Standard1DAnalyzer):
    """Specialized 1D analyzer for ICT charge measurements.

    Inherits from Standard1DAnalyzer and adds ICT-specific functionality:
    - Automatic dt extraction from TDMS waveform properties
    - ICT signal processing pipeline (Butterworth filter, sinusoid removal)
    - Charge calculation in picocoulombs
    - Single scalar output (charge_pC) for scan analyzer

    The analyzer requires a configuration with an 'ict_analysis' section
    that specifies Butterworth filter parameters and calibration factor.

    Parameters
    ----------
    line_config_name : str
        Name of the line configuration to load (must include ict_analysis section)

    Raises
    ------
    ValueError
        If configuration missing ict_analysis section

    Examples
    --------
    >>> analyzer = ICT1DAnalyzer("U_BCaveICT")
    >>> file_path = Path("shot_001.tdms")
    >>> data = analyzer.load_image(file_path)
    >>> result = analyzer.analyze_image(data, auxiliary_data={"file_path": file_path})
    >>> charge_pC = result.scalars['charge_pC']
    """

    def __init__(self, line_config_name: str):
        """Initialize ICT analyzer with configuration validation.

        Parameters
        ----------
        line_config_name : str
            Name of configuration file (must include ict_analysis section)

        Raises
        ------
        ValueError
            If configuration missing ict_analysis section
        """
        super().__init__(line_config_name)

        # Validate ICT configuration exists
        if self.line_config.ict_analysis is None:
            raise ValueError(
                f"Configuration '{line_config_name}' must include 'ict_analysis' section"
            )

        self.ict_config = self.line_config.ict_analysis
        self.dt = None  # Will be set during load_image()

        logger.info(
            f"Initialized ICT1DAnalyzer with config '{line_config_name}' "
            f"(detector: {self.ict_config.calibration.detector_type})"
        )

    def load_image(self, file_path: Path) -> Array1D:
        """Load TDMS data and extract dt from metadata.

        Overrides parent to extract time step (dt) from TDMS waveform properties.
        Supports automatic extraction, override, and fallback to default.

        Parameters
        ----------
        file_path : Path
            Path to TDMS file

        Returns
        -------
        Array1D
            Nx2 array [time, voltage]
        """
        # Call parent to load data and populate metadata
        data = super().load_image(file_path)

        # Extract dt from TDMS metadata if configured
        if self.ict_config.extract_dt_from_metadata:
            extracted_dt = self._extract_dt_from_metadata(file_path)
            if extracted_dt is not None:
                self.dt = extracted_dt
                logger.debug(f"Extracted dt={self.dt:.2e} s from TDMS metadata")

        # Use override if provided
        if self.ict_config.dt_override is not None:
            self.dt = self.ict_config.dt_override
            logger.debug(f"Using dt override: {self.dt:.2e} s")

        # Fallback to default
        if self.dt is None:
            self.dt = 4e-9  # 250 MHz default
            logger.warning(f"Using default dt={self.dt:.2e} s (250 MHz)")

        return data

    def analyze_image(
        self, image: Array1D, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """Analyze ICT waveform and calculate charge.

        Overrides parent to add ICT-specific charge calculation.
        Returns only charge_pC as scalar output to downstream scan analyzer.

        NOTE: ICT analysis requires raw, unprocessed voltage data. The ICT analysis
        pipeline has its own signal processing steps (Butterworth filter, sinusoid
        removal, etc.) that should not be preceded by general preprocessing.
        Therefore, this method skips the standard preprocessing pipeline and passes
        raw voltage data directly to apply_ict_analysis().

        Parameters
        ----------
        image : Array1D
            Nx2 array [time, voltage]
        auxiliary_data : dict, optional
            Additional data (e.g., file_path for metadata)

        Returns
        -------
        ImageAnalyzerResult
            Result with charge_pC scalar and original waveform
        """
        # Load metadata if needed
        if (
            self.data_metadata is None
            and auxiliary_data
            and "file_path" in auxiliary_data
        ):
            self.load_image(auxiliary_data["file_path"])

        # Extract raw voltage trace (column 1) WITHOUT preprocessing
        # ICT analysis requires raw data - it has its own signal processing pipeline
        voltage_trace = image[:, 1]

        # Apply ICT analysis pipeline (includes Butterworth filter, sinusoid removal, etc.)
        try:
            charge_pC = ict_algorithms.apply_ict_analysis(
                data=voltage_trace,
                dt=self.dt,
                butterworth_order=self.ict_config.butterworth.order,
                butterworth_crit_f=self.ict_config.butterworth.critical_frequency,
                calibration_factor=self.ict_config.calibration.calibration_factor,
            )
        except Exception as e:
            logger.error(f"ICT analysis failed: {e}")
            charge_pC = 0.0

        # Build input parameters
        input_params = self._build_input_parameters(auxiliary_data)
        input_params.update(
            {
                "dt": self.dt,
                "butterworth_order": self.ict_config.butterworth.order,
                "butterworth_crit_f": self.ict_config.butterworth.critical_frequency,
                "calibration_factor": self.ict_config.calibration.calibration_factor,
                "detector_type": self.ict_config.calibration.detector_type,
            }
        )

        # Return result with ONLY charge_pC scalar for scan analyzer
        result = ImageAnalyzerResult(
            data_type="1d",
            line_data=image,  # Return original data for visualization/debugging
            scalars={"charge_pC": charge_pC},  # ONLY output to scan analyzer
            metadata=input_params,  # For logging/debugging
        )

        logger.debug(f"ICT analysis complete: charge={charge_pC:.2f} pC")

        return result

    def _extract_dt_from_metadata(self, file_path: Path) -> Optional[float]:
        """Extract time step from TDMS waveform properties.

        TDMS files store wf_increment in channel properties, which represents
        the time step between samples. This is typically extracted from the
        oscilloscope's waveform metadata.

        Parameters
        ----------
        file_path : Path
            Path to TDMS file

        Returns
        -------
        float or None
            Time step in seconds, or None if not found or error occurs
        """
        try:
            from nptdms import TdmsFile

            tdms_file = TdmsFile.read(file_path)
            groups = tdms_file.groups()

            if not groups:
                logger.debug("No groups found in TDMS file")
                return None

            # Get first channel from first group
            channels = groups[0].channels()
            if not channels:
                logger.debug("No channels found in TDMS file")
                return None

            channel = channels[0]
            wf_increment = channel.properties.get("wf_increment")

            if wf_increment is not None:
                dt = float(wf_increment)
                logger.debug(f"Found wf_increment in TDMS: {dt:.2e} s")
                return dt

            logger.debug("wf_increment not found in TDMS channel properties")
            return None

        except Exception as e:
            logger.warning(f"Failed to extract dt from TDMS metadata: {e}")
            return None
