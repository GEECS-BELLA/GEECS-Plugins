"""
Magnetic spectrometer analyzer with energy calibration.

Provides analysis for magnetic spectrometer images with pixel-to-energy
conversion using device-specific calibrations.
"""

import numpy as np
import logging
from typing import Optional, Tuple, Dict, List, Callable
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt

from image_analysis.tools.rendering import base_render_image
from image_analysis.offline_analyzers.beam_analyzer import BeamAnalyzer
from image_analysis.types import ImageAnalyzerResult
from image_analysis.algorithms.axis_interpolation import interpolate_image_axis

logger = logging.getLogger(__name__)


@dataclass
class MagSpecCalibration:
    """
    Calibration specification for pixel-to-energy mapping.

    Supports three calibration types:
    - polynomial: Uses polynomial coefficients
    - array: Uses pre-computed array (from file or inline)
    - callable: Uses custom function
    """

    type: str  # "polynomial", "array", or "callable"
    coeffs: Optional[List[float]] = None  # For polynomial
    values: Optional[np.ndarray] = None  # For inline array
    file: Optional[str] = None  # For array from file
    function: Optional[Callable[[np.ndarray], np.ndarray]] = None  # For callable

    def __post_init__(self):
        """Validate calibration configuration."""
        if self.type not in ["polynomial", "array", "callable"]:
            raise ValueError(
                f"Calibration type must be 'polynomial', 'array', or 'callable', "
                f"got '{self.type}'"
            )

        if self.type == "polynomial" and self.coeffs is None:
            raise ValueError("Polynomial calibration requires 'coeffs'")

        if self.type == "array" and self.values is None and self.file is None:
            raise ValueError("Array calibration requires 'values' or 'file'")

        if self.type == "callable" and self.function is None:
            raise ValueError("Callable calibration requires 'function'")


@dataclass
class MagSpecConfig:
    """
    Configuration for magnetic spectrometer analysis.

    Parameters
    ----------
    mag_field : str
        Magnetic field setting identifier (e.g., "825mT")
    calibration : MagSpecCalibration
        Pixel-to-energy calibration specification (always in canonical orientation)
    energy_range : tuple of (float, float)
        Energy range in MeV (min, max)
    num_energy_points : int, default=500
        Number of points in uniform energy grid
    flip_horizontal : bool, default=False
        Flip raw image horizontally to align with canonical calibration.
        Set to True if camera is mounted backwards relative to calibration.
    flip_vertical : bool, default=False
        Flip raw image vertically to align with canonical calibration.
        Set to True if camera is mounted upside-down.

    Notes
    -----
    The calibration defines the canonical orientation (energy increasing
    left-to-right). Flip parameters correct for camera mounting, ensuring
    the raw image aligns with this canonical orientation before interpolation.
    The output is always in canonical orientation.
    """

    mag_field: str
    calibration: MagSpecCalibration
    energy_range: Tuple[float, float]
    num_energy_points: int = 500
    flip_horizontal: bool = False
    flip_vertical: bool = False

    @classmethod
    def for_uc_hiresmag(
        cls,
        mag_field: str = "825mT",
        energy_range: Optional[Tuple[float, float]] = None,
        num_energy_points: int = 500,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
    ) -> "MagSpecConfig":
        """
        Preset configuration for UC_HiResMagCam device.

        Parameters
        ----------
        mag_field : str, default="825mT"
            Magnetic field setting. Options: "800mT", "825mT"
        energy_range : tuple of (float, float), optional
            Energy range in MeV. Defaults to (20, 150)
        num_energy_points : int, default=500
            Number of points in uniform energy grid
        flip_horizontal : bool, default=False
            Whether to flip image left-right
        flip_vertical : bool, default=False
            Whether to flip image up-down

        Returns
        -------
        MagSpecConfig
            Configuration for UC_HiResMagCam

        Raises
        ------
        ValueError
            If mag_field is not recognized
        """
        # Calibration coefficients for UC_HiResMagCam
        calibrations = {
            "800mT": [8.66599527e01, 1.78007126e-02, 1.10546749e-06],
            "825mT": [8.93681013e01, 1.83568540e-02, 1.14012869e-06],
        }

        if mag_field not in calibrations:
            raise ValueError(
                f"Mag field '{mag_field}' not available for UC_HiResMagCam. "
                f"Available: {list(calibrations.keys())}"
            )

        return cls(
            mag_field=mag_field,
            calibration=MagSpecCalibration(
                type="polynomial", coeffs=calibrations[mag_field]
            ),
            energy_range=energy_range or (20, 150),
            num_energy_points=num_energy_points,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        )


class MagSpecManualCalibAnalyzer(BeamAnalyzer):
    """
    Analyzer for magnetic spectrometer with device-specific calibrations.

    Supports multiple calibration formats through MagSpecConfig:
    - Polynomial coefficients
    - Pre-computed arrays (from file or inline)
    - Custom calibration functions
    """

    def __init__(
        self,
        camera_config_name: str,
        magspec_config: Optional[MagSpecConfig] = None,
    ):
        """
        Initialize magnetic spectrometer analyzer.

        Parameters
        ----------
        camera_config_name : str
            Camera configuration name (loaded from config system)
        magspec_config : MagSpecConfig, optional
            Magnetic spectrometer configuration. If None, attempts to use
            UC_HiResMagCam preset with default settings.

        Raises
        ------
        ValueError
            If magspec_config is None and camera is not UC_HiResMagCam
        """
        super().__init__(camera_config_name)

        # Use provided config or create default for UC_HiResMagCam
        if magspec_config is None:
            if self.camera_name == "UC_HiResMagCam":
                self.magspec_config = MagSpecConfig.for_uc_hiresmag()
                logger.info("Using default UC_HiResMagCam configuration")
            else:
                raise ValueError(
                    f"magspec_config required for camera '{self.camera_name}'. "
                    f"Use MagSpecConfig.for_uc_hiresmag() or create custom config."
                )
        else:
            self.magspec_config = magspec_config

        logger.info(
            f"Initialized {self.camera_name} with mag_field={self.magspec_config.mag_field}, "
            f"energy_range={self.magspec_config.energy_range} MeV"
        )

    def _build_energy_calibration(self, image: np.ndarray) -> np.ndarray:
        """
        Build pixel-to-energy mapping for current configuration.

        Parameters
        ----------
        image : np.ndarray
            Image to build calibration for (used to get width)

        Returns
        -------
        np.ndarray
            Array mapping pixel index to energy value
        """
        cal = self.magspec_config.calibration

        # Get image width
        image_width = image.shape[1]

        # Create pixel axis
        pixel_axis = np.arange(image_width)

        if cal.type == "polynomial":
            # Polynomial calibration: sum(coeffs[i] * x^i)
            pixel_to_energy = np.zeros_like(pixel_axis, dtype=float)
            for i, coeff in enumerate(cal.coeffs):
                pixel_to_energy += coeff * np.power(pixel_axis, i)
            logger.debug(
                f"Built polynomial calibration with {len(cal.coeffs)} coefficients"
            )

        elif cal.type == "array":
            # Pre-computed array calibration
            if cal.file is not None:
                # Load from file
                cal_path = Path(cal.file)
                if not cal_path.is_absolute():
                    # Try to resolve relative to package
                    cal_path = Path(__file__).parent.parent / cal_path
                pixel_to_energy = np.load(cal_path)
                logger.debug(f"Loaded calibration array from {cal_path}")
            else:
                # Use inline array
                pixel_to_energy = cal.values
                logger.debug("Using inline calibration array")

            # Validate length
            if len(pixel_to_energy) != image_width:
                raise ValueError(
                    f"Calibration array length ({len(pixel_to_energy)}) "
                    f"doesn't match image width ({image_width})"
                )

        elif cal.type == "callable":
            # Custom function calibration
            pixel_to_energy = cal.function(pixel_axis)
            logger.debug("Applied callable calibration function")

        else:
            raise ValueError(f"Unknown calibration type: {cal.type}")

        return pixel_to_energy

    def analyze_image(
        self, image: np.ndarray, auxiliary_data: Optional[Dict] = None
    ) -> ImageAnalyzerResult:
        """
        Analyze magnetic spectrometer image with energy calibration.

        Parameters
        ----------
        image : np.ndarray
            Image array to analyze
        auxiliary_data : dict, optional
            Additional data (unused)

        Returns
        -------
        ImageAnalyzerResult
            Analysis result with energy-calibrated image and metadata

        Notes
        -----
        The calibration is always defined in canonical orientation (increasing
        left-to-right). The flip parameters correct for camera mounting:
        - flip_horizontal: Camera is mounted backwards relative to calibration
        - flip_vertical: Camera is mounted upside-down

        Flips are applied to the raw image BEFORE interpolation to align it
        with the canonical calibration. The output is always in canonical
        orientation.
        """
        # Standard processing from BeamAnalyzer
        initial_result: ImageAnalyzerResult = super().analyze_image(
            image=image, auxiliary_data=auxiliary_data
        )

        # Get processed image
        final_image = initial_result.processed_image.copy()

        # Apply flips to align image with canonical calibration orientation
        # These correct for camera mounting - NOT post-processing effects
        if self.magspec_config.flip_horizontal:
            final_image = final_image[:, ::-1]
            logger.debug("Applied horizontal flip to align with canonical calibration")

        if self.magspec_config.flip_vertical:
            final_image = final_image[::-1, :]
            logger.debug("Applied vertical flip to align with canonical calibration")

        # Build energy calibration (always canonical/increasing)
        pixel_to_energy = self._build_energy_calibration(image=final_image)

        # Apply energy axis interpolation
        # Both image and calibration are now in canonical orientation
        interp_image, energy_axis = interpolate_image_axis(
            final_image,
            pixel_to_energy,
            axis=1,  # Horizontal = energy
            num_points=self.magspec_config.num_energy_points,
            physical_min=self.magspec_config.energy_range[0],
            physical_max=self.magspec_config.energy_range[1],
        )

        # Compute analysis metrics
        scalars = self._compute_metrics(interp_image, energy_axis)

        # Create result
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=interp_image,
            scalars=scalars,
            metadata={
                "energy_units": "MeV",
                "mag_field": self.magspec_config.mag_field,
                "flip_horizontal": self.magspec_config.flip_horizontal,
                "flip_vertical": self.magspec_config.flip_vertical,
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
            Energy-calibrated image
        energy_axis : np.ndarray
            Energy axis values in MeV

        Returns
        -------
        dict
            Dictionary of scalar metrics
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
            "peak_energy_MeV": peak_energy,
            "mean_energy_MeV": mean_energy,
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
        """
        Custom render function for energy-calibrated spectra.

        Parameters
        ----------
        result : ImageAnalyzerResult
            Analysis result containing energy-calibrated image
        vmin : float, optional
            Minimum value for colormap
        vmax : float, optional
            Maximum value for colormap
        cmap : str, default="plasma"
            Colormap name
        figsize : tuple, default=(4, 4)
            Figure size in inches
        dpi : int, default=150
            Figure DPI
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure

        Returns
        -------
        tuple of (Figure, Axes)
            The matplotlib figure and axes objects
        """
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
            # Update tick labels to show energy values (not pixels)
            num_ticks = 5
            tick_positions = np.linspace(0, len(energy_axis) - 1, num_ticks)
            tick_labels = [f"{energy_axis[int(pos)]:.1f}" for pos in tick_positions]

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("Energy (MeV)")
            ax.set_ylabel("Vertical Position (pixels)")

        return fig, ax
