"""
Magnetic spectrometer analyzer with energy calibration.

Provides analysis for magnetic spectrometer images with pixel-to-energy
conversion using device-specific calibrations.
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt

from image_analysis.tools.rendering import base_render_image
from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
from image_analysis.types import ImageAnalyzerResult
from image_analysis.algorithms.axis_interpolation import interpolate_image_axis

logger = logging.getLogger(__name__)


class MagSpecManualCalibAnalyzer(BeamAnalyzer):
    """
    Analyzer for magnetic spectrometer with device-specific calibrations.

    Supports multiple calibration formats:
    - 'polynomial': Polynomial coefficients [p0, p1, p2, ...]
    - 'array': Pre-computed pixel-to-energy array
    - 'callable': Custom calibration function
    """

    # Device-specific configurations
    DEVICE_CONFIGS: Dict[str, Dict[str, Any]] = {
        "UC_HiResMagCam": {
            "default_mag_field": "825mT",
            "calibrations": {
                "800mT": {
                    "type": "polynomial",
                    "coeffs": [8.66599527e01, 1.78007126e-02, 1.10546749e-06],
                },
                "825mT": {
                    "type": "polynomial",
                    "coeffs": [8.93681013e01, 1.83568540e-02, 1.14012869e-06],
                },
            },
            "default_energy_range": (20, 150),
        },
    }

    def __init__(
        self,
        camera_config_name,
        mag_field: Optional[str] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        num_energy_points: int = 500,
    ):
        """
        Initialize magnetic spectrometer analyzer.

        Parameters
        ----------
        camera_config_name : CameraConfig
            Standard camera configuration (includes device name).
        mag_field : str, optional
            Magnetic field setting. If None, uses device default.
        energy_range : tuple of float, optional
            Energy range in GeV (min, max). If None, uses device default.
        num_energy_points : int, default=500
            Number of points in uniform energy grid.

        Raises
        ------
        ValueError
            If device is not found in DEVICE_CONFIGS or mag_field is invalid.
        """
        super().__init__(camera_config_name)

        # Get device name from config
        device_name = self.camera_name

        # Look up device configuration
        if device_name not in self.DEVICE_CONFIGS:
            raise ValueError(
                f"No calibration found for device '{device_name}'. "
                f"Available devices: {list(self.DEVICE_CONFIGS.keys())}"
            )

        self.device_config = self.DEVICE_CONFIGS[device_name]

        # Use provided mag_field or device default
        self.mag_field = mag_field or self.device_config["default_mag_field"]

        # Validate mag field for this device
        if self.mag_field not in self.device_config["calibrations"]:
            raise ValueError(
                f"Mag field '{self.mag_field}' not available for {device_name}. "
                f"Available: {list(self.device_config['calibrations'].keys())}"
            )

        # Use provided energy_range or device default
        self.energy_range = energy_range or self.device_config["default_energy_range"]
        self.num_energy_points = num_energy_points

    def _build_energy_calibration(self, image: np.ndarray):
        """Build pixel-to-energy mapping for current configuration."""
        cal = self.device_config["calibrations"][self.mag_field]

        # Get image width from config
        image_width = image.shape[1]

        # Create pixel axis
        pixel_axis = np.arange(image_width)

        cal_type = cal["type"]

        if cal_type == "polynomial":
            # Polynomial calibration: sum(coeffs[i] * x^i)
            coeffs = cal["coeffs"]
            self.pixel_to_energy = np.zeros_like(pixel_axis, dtype=float)
            for i, coeff in enumerate(coeffs):
                self.pixel_to_energy += coeff * np.power(pixel_axis, i)
            logger.debug(
                f"Built polynomial calibration with {len(coeffs)} coefficients"
            )

        elif cal_type == "array":
            # Pre-computed array calibration
            if "file" in cal:
                # Load from file
                cal_path = Path(cal["file"])
                if not cal_path.is_absolute():
                    # Try to resolve relative to package
                    cal_path = Path(__file__).parent.parent / cal_path
                self.pixel_to_energy = np.load(cal_path)
                logger.debug(f"Loaded calibration array from {cal_path}")
            elif "values" in cal:
                # Use inline array
                self.pixel_to_energy = np.array(cal["values"])
                logger.debug("Using inline calibration array")
            else:
                raise ValueError("Array calibration must have 'file' or 'values'")

            # Validate length
            if len(self.pixel_to_energy) != image_width:
                raise ValueError(
                    f"Calibration array length ({len(self.pixel_to_energy)}) "
                    f"doesn't match image width ({image_width})"
                )

        elif cal_type == "callable":
            # Custom function calibration
            func = cal["function"]
            self.pixel_to_energy = func(pixel_axis)
            logger.debug("Applied callable calibration function")

        else:
            raise ValueError(f"Unknown calibration type: {cal_type}")

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """
        Analyze magnetic spectrometer image with energy calibration.

        Parameters
        ----------
        image : NDArray
            Path to the image file to analyze.
        auxiliary_data : Dict
            Additional arguments passed to parent analyze method.

        Returns
        -------
        ImageAnalyzerResult
            Analysis result with energy-calibrated image and metadata.
        """
        initial_result: ImageAnalyzerResult = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        # Apply threshold at 10 counts (specific to this analyzer)
        final_image = initial_result.processed_image.copy()

        self._build_energy_calibration(image=final_image)

        # Apply energy axis interpolation
        interp_image, energy_axis = interpolate_image_axis(
            final_image,
            self.pixel_to_energy,
            axis=1,  # Horizontal = energy
            num_points=self.num_energy_points,
            physical_min=self.energy_range[0],
            physical_max=self.energy_range[1],
        )

        # Compute analysis metrics
        scalars = self._compute_metrics(interp_image, energy_axis)

        # Create result
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=interp_image,
            scalars=scalars,
            metadata={
                "energy_units": "GeV",
                "mag_field": self.mag_field,
            },
        )

        result.render_data = {
            "energy_axis": energy_axis,
        }

        return result

    def _compute_metrics(
        self, image: np.ndarray, energy_axis: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute energy spectrum metrics.

        Parameters
        ----------
        image : np.ndarray
            Energy-calibrated image.
        energy_axis : np.ndarray
            Energy axis values.

        Returns
        -------
        dict
            Dictionary of scalar metrics.
        """
        # Vertical projection (sum over vertical to get energy spectrum)
        energy_spectrum = np.sum(image, axis=0)

        # Find peak energy
        if len(energy_spectrum) > 0 and np.max(energy_spectrum) > 0:
            peak_idx = np.argmax(energy_spectrum)
            peak_energy = energy_axis[peak_idx]
        else:
            peak_energy = np.nan

        # Total charge (integrated signal)
        total_charge = np.sum(image)

        # Mean energy (weighted by intensity)
        if total_charge > 0:
            mean_energy = np.sum(energy_axis * energy_spectrum) / np.sum(
                energy_spectrum
            )
        else:
            mean_energy = np.nan

        metrics = {
            "peak_energy_GeV": peak_energy,
            "mean_energy_GeV": mean_energy,
            "total_charge_au": total_charge,
            "max_intensity": np.max(image),
        }

        return metrics

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
        """Custom render function for energy-calibrated spectra."""
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

        # Get energy axis from render_data
        energy_axis = result.render_data.get("energy_axis")

        if energy_axis is not None:
            # Don't mess with extent - keep image in pixel coordinates
            # Just update the tick labels to show energy values

            # Create a few evenly-spaced ticks across the image width
            num_ticks = 5  # or 6, whatever looks good
            tick_positions = np.linspace(0, len(energy_axis) - 1, num_ticks)
            tick_labels = [f"{energy_axis[int(pos)]:.2f}" for pos in tick_positions]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("Energy (GeV)")
            ax.set_ylabel("Vertical Position (pixels)")

        return fig, ax
