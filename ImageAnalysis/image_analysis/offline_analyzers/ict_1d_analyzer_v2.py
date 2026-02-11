"""ICT (Integrated Current Transformer) specialized 1D analyzer (v2).

This module provides ICT1DAnalyzer, a child of Standard1DAnalyzer that runs
the ICT charge-measurement algorithm on oscilloscope voltage traces. The
analyze_image method leverages the ict_algrotihms module. Parameters needed
by the algorithm can be added to the config as model_extras called
ict_analysis_params.

YAML config example
-------------------
Add an ``ict_analysis_params`` block to the device's config file::

    ict_analysis_params:
      butterworth_order: 1
      butterworth_crit_f: 0.125
      calibration_factor: 0.1
      dt: null            # null → derive from time column
"""

from __future__ import annotations

import logging
from typing import Optional, Dict


from image_analysis.offline_analyzers.standard_1d_analyzer import Standard1DAnalyzer
from image_analysis.types import Array1D, ImageAnalyzerResult
from image_analysis.algorithms.ict_algorithms import apply_ict_analysis

logger = logging.getLogger(__name__)


class ICT1DAnalyzer(Standard1DAnalyzer):
    """Specialized 1D analyzer for ICT charge measurements.

    ICT analysis parameters are plain instance attributes that can be set:

    1. From the YAML config file (``ict_analysis_params`` section), or
    2. At runtime by assigning to the attribute directly.

    Parameters
    ----------
    line_config_name : str
        Name of the line configuration to load (must define ``data_loading``).

    Attributes
    ----------
    butterworth_order : int
        Butterworth filter order (default: 1).
    butterworth_crit_f : float
        Normalised critical frequency for Butterworth filter (default: 0.125).
    calibration_factor : float
        Calibration factor in V·s/C (default: 0.1).
    dt : float or None
        Time step override in seconds.  If ``None`` (the default), ``dt`` is
        derived from the time column of the loaded data.
    """

    def __init__(self, line_config_name: str):
        super().__init__(line_config_name)

        # Read ICT-specific params from the config's extras, with defaults
        # that match the algorithm function signature.
        extras = self.line_config.model_extra or {}
        ict_params = extras.get("ict_analysis_params", {})

        self.butterworth_order: int = ict_params.get("butterworth_order", 1)
        self.butterworth_crit_f: float = ict_params.get("butterworth_crit_f", 0.125)
        self.calibration_factor: float = ict_params.get("calibration_factor", 0.1)
        self.dt: Optional[float] = ict_params.get("dt", None)

        logger.info("Initialized ICT1DAnalyzer with config '%s'", line_config_name)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_image(
        self, image: Array1D, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """Run the ICT analysis pipeline on a voltage trace."""
        # run preprocessing pipelin
        initial_result: ImageAnalyzerResult = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        line_data = initial_result.line_data

        # Resolve dt: use self.dt if provided, otherwise derive from time column
        # dt = self.dt
        if self.dt is None:
            self.dt = float(line_data[1, 0] - line_data[0, 0])

        # Extract voltage trace (column 1) — algorithm expects 1D voltage array
        voltage_trace = line_data[:, 1]

        # Run ICT analysis
        try:
            charge_pC = apply_ict_analysis(
                data=voltage_trace,
                dt=self.dt,
                butterworth_order=self.butterworth_order,
                butterworth_crit_f=self.butterworth_crit_f,
                calibration_factor=self.calibration_factor,
            )
        except Exception as e:
            logger.error("ICT analysis failed: %s", e)
            charge_pC = 0.0

        result = ImageAnalyzerResult(
            data_type="1d",
            line_data=image,
            scalars={"charge_pC": charge_pC},
        )

        logger.debug("ICT analysis complete: charge=%.2f pC", charge_pC)

        return result
