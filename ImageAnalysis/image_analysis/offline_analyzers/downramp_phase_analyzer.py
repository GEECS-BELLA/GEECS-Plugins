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

from typing import Optional, Dict, Union, TYPE_CHECKING, Tuple
from pathlib import Path
if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


# Import the StandardAnalyzer parent class
from image_analysis.offline_analyzers.standard_analyzer import StandardAnalyzer
from image_analysis.types import ImageAnalyzerResult
from image_analysis.algorithms.beam_slopes import compute_beam_slopes

logger = logging.getLogger(__name__)

def apply_gaussian_mask(
        image: np.ndarray,
        center_x: Optional[int] = None,
        center_y: Optional[int] = None,
        sigma_x: float = 100,
        sigma_y: float = 20,
        binarize: bool = False,
        threshold_factor: float = 0.01,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Apply a separable 2D Gaussian mask and optionally binarize.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image.
    center_x : int, optional
        X-coordinate of Gaussian center; defaults to image center.
    center_y : int, optional
        Y-coordinate of Gaussian center; defaults to image center.
    sigma_x : float, default=100
        Standard deviation of Gaussian along x.
    sigma_y : float, default=20
        Standard deviation of Gaussian along y.
    binarize : bool, default=False
        If True, return ``binary_mask * image`` using `threshold_factor`.
    threshold_factor : float, default=0.01
        Fraction of the max masked value used as a binarization threshold.

    Returns
    -------
    numpy.ndarray
        Masked image if `binarize` is False.
    tuple of numpy.ndarray
        ``(binary_mask * image)`` if `binarize` is True.
    """
    rows, cols = image.shape
    if center_x is None:
        center_x = cols // 2
    if center_y is None:
        center_y = rows // 2

    y, x = np.mgrid[0:rows, 0:cols]
    gaussian_mask = np.exp(
        -(
                ((x - center_x) ** 2) / (2 * sigma_x ** 2)
                + ((y - center_y) ** 2) / (2 * sigma_y ** 2)
        )
    )
    masked_image = image * gaussian_mask

    if not binarize:
        return masked_image
    else:
        threshold = threshold_factor * np.nanmax(masked_image)
        binary_mask = (masked_image > threshold).astype(np.float32)
        return binary_mask * image

class DownrampPhaseAnalyzer(StandardAnalyzer):
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
        name_suffix: Optional[str] = None,
        metric_suffix: Optional[str] = None,
    ):
        """Initialize the beam analyzer with external configuration.

        Parameters
        ----------
        camera_config_name : str
            Name of the camera configuration to load (e.g., "UC_ALineEBeam3")
        """
        # Initialize parent class
        super().__init__(
            camera_config_name=camera_config_name,
            name_suffix=name_suffix,
            metric_suffix=metric_suffix,
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

        self.file_path = (
            Path(auxiliary_data["file_path"])
            if auxiliary_data and "file_path" in auxiliary_data
            else None
        )

        processed_image = initial_result.processed_image

        processed_image = -(processed_image - np.nanmin(processed_image))
        processed_image = abs(processed_image - np.nanmin(processed_image))

        processed_image = processed_image - np.nanmedian(processed_image)

        gauss_masked_array = apply_gaussian_mask(
            image=processed_image, binarize=True)
        processed_image = gauss_masked_array

        slopes_result = compute_beam_slopes(processed_image)
        print(slopes_result)

        scalar_results_dict = self.compile_shock_analysis(processed_image)

        phase_converted = processed_image
        phase_scale_16bit_scale = (2 ** 16 - 1) / (2 * np.pi)
        phase_converted = phase_scale_16bit_scale * phase_converted
        phase_converted = phase_converted.astype(np.uint16)

        # Build result with beam-specific data
        result = ImageAnalyzerResult(
            data_type="2d",
            processed_image=phase_converted,
            scalars=scalar_results_dict,
            metadata=initial_result.metadata,
        )

        # Add projection overlays for rendering
        if processed_image is not None:
            result.render_data = {
                "horizontal_projection": processed_image.sum(axis=0),
                "vertical_projection": processed_image.sum(axis=1),
            }

        # Apply metric suffix to final scalars dict (no-op if empty or no suffix)
        if getattr(result, "scalars", None):
            result.scalars = self.apply_metric_suffix(result.scalars)

        return result

    def compile_shock_analysis(
            self, phase_array: NDArray, window_size: int = 20
    ) -> dict:
        """Compute shock metrics, assemble composite figure, and return results.

        Workflow
        --------
        - Rotate phase for analysis.
        - Compute shock angle.
        - Compute max gradient and location.
        - Compute plateau average and peak-to-plateau delta.
        - Save a composite of the three diagnostic plots when `file_path` is set.

        Parameters
        ----------
        phase_array : NDArray
            Input phase map for analysis.
        window_size : int, default=20
            Window size used for gradient and plateau calculations.

        Returns
        -------
        dict
            Dictionary of metrics with keys:
            'Plasma downramp shock_angle', 'Plasma downramp shock slope (phase/pixel)',
            'Plasma downramp shock location (pixel)', 'Plasma downramp plateau avg (phase)',
            'Plasma downramp peak to plateau (phase)', and figure metadata from rotation.
        """
        slopes = compute_beam_slopes(phase_array, prefix=self.camera_name)
        laser_angle = np.arctan(slopes[f'{self.camera_name}_image_com_slope_x'])
        laser_angle_deg = np.degrees(laser_angle)
        rotated_data = ndimage.rotate(
            phase_array, angle=laser_angle_deg, reshape=True, order=1
        )

        rotated_phase = rotated_data

        shock_angle = self.get_shock_angle(rotated_phase)
        max_slope, best_center = self.get_shock_gradient_and_position(
            rotated_phase, window_size=window_size
        )
        plateau_value, delta = self.calculate_delta_plateau(
            rotated_phase, best_center, window_size=window_size
        )

        results = {
            "Plasma downramp shock_angle": shock_angle,
            "Plasma downramp shock slope (phase/pixel)": max_slope,
            "Plasma downramp shock location (pixel)": best_center,
            "Plasma downramp plateau avg (phase)": plateau_value,
            "Plasma downramp peak to plateau (phase)": delta,
        }

        merged = {**results, **slopes}

        def fig_to_array(fig, keep_alpha: bool = False) -> np.ndarray:
            """Return H×W×3 (RGB) or H×W×4 (RGBA) uint8 from a Matplotlib figure, all backends."""
            fig.canvas.draw()  # ensure renderer exists/updated
            rgba = np.asarray(
                fig.canvas.renderer.buffer_rgba()
            )  # shape (H, W, 4), dtype=uint8
            return rgba if keep_alpha else rgba[:, :, :3]

        fig1 = self.shock_angle_fig
        fig2 = self.shock_grad_fig
        fig3 = self.delta_plateau_fig

        img1 = fig_to_array(fig1)
        plt.close(fig1)
        img2 = fig_to_array(fig2)
        plt.close(fig2)
        img3 = fig_to_array(fig3)
        plt.close(fig3)

        combined_fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img1)
        axs[0].set_title("Shock Angle Determination")
        axs[0].axis("off")

        axs[1].imshow(img2)
        axs[1].set_title("Shock Gradient & Position")
        axs[1].axis("off")

        axs[2].imshow(img3)
        axs[2].set_title("Plateau estimation")
        axs[2].axis("off")

        plt.tight_layout()
        if self.file_path is not None:
            combined_save_path = (
                    self.file_path.parent
                    / f"{self.file_path.stem}_combined_shock_analysis.pdf"
            )
            combined_fig.savefig(combined_save_path)
            plt.close(combined_fig)
            logger.info(
                "Combined shock analysis figure saved to %s", combined_save_path
            )

        results["combined_fig"] = combined_fig
        return merged

    def get_shock_angle(self, image: np.ndarray, window_size: int = 20) -> float:
        """
        Estimate the shock angle in an image by finding the local slope per row
        and fitting a line to these positions.

        Parameters
        ----------
        image : np.ndarray
            2D phase or intensity map to analyze.
        window_size : int, default=20
            Window width for local linear fits when finding row-wise maxima.

        Returns
        -------
        float
            Shock angle in radians.

        Side Effects
        ------------
        Stores a matplotlib figure showing the image with the fitted shock angle line
        in `self.shock_angle_fig`.
        """
        fig = plt.figure(figsize=(6, 4))

        H, W = image.shape
        max_slope_positions = []

        for row_idx in range(H):
            row = image[row_idx, :]
            _, best_center, _, _ = self.find_max_local_slope(row, window_size)
            max_slope_positions.append(best_center)

        max_slope_positions = np.array(max_slope_positions, dtype=float)
        rows = np.arange(H)

        # Mask invalid entries
        valid = np.isfinite(max_slope_positions)
        rows_valid = rows[valid]
        positions_valid = max_slope_positions[valid]
        weights = np.nansum(image[valid, :], axis=1)  # only valid rows

        # Fit a line to get shock angle using only valid rows
        if len(rows_valid) >= 2:
            coeffs = np.polyfit(rows_valid, positions_valid, 1,w=weights)
            slope_line = coeffs[0]
            angle_rad = np.arctan(slope_line)
        else:
            slope_line = 0.0
            angle_rad = np.nan

        plt.imshow(image, origin="lower", cmap="plasma")
        plt.title("Shock angle estimation")

        # Only plot if we have a valid fit
        if len(rows_valid) >= 2:
            intercept = coeffs[1]
            plt.plot(
                slope_line * rows_valid + intercept,  # y-values along valid rows
                rows_valid,  # x-axis = row indices
                color="cyan",
                linewidth=2,
                linestyle="--",  # makes it dashed; use ':' for dotted
                alpha=0.7,  # 0 = transparent, 1 = fully opaque
                label="Shock angle fit",
            )

        plt.legend()
        self.shock_angle_fig = plt.gcf()

        return angle_rad

    def find_max_local_slope(
            self, profile: np.ndarray,
            window_size: int = 20
    ) -> Tuple[float, int, np.ndarray, np.ndarray]:
        """
        Find the maximum local slope and its position in a 1D array using sliding window.

        Returns:
            max_slope : float
                Maximum slope found (phase units per pixel)
            best_center : int
                Center index of the window with maximum slope
            best_fit_x : np.ndarray
                X values for the best window (for optional plotting)
            best_fit_y : np.ndarray
                Linear fit values for the best window (for optional plotting)
        """

        if np.nansum(profile) <= 0:
            # Invalid profile: total counts are zero or negative
            return np.nan, None, None, None

        x = np.arange(len(profile))
        max_slope = 0.0
        best_center = None
        best_fit_x = None
        best_fit_y = None

        for start in range(0, len(profile) - window_size + 1):
            end = start + window_size
            x_window = x[start:end]
            y_window = profile[start:end]

            # Skip window if it contains only non-positive values
            if np.nansum(y_window) <= 0:
                continue

            coeffs = np.polyfit(x_window, y_window, 1)
            slope = coeffs[0]
            center = (start + end) // 2

            if abs(slope) > abs(max_slope):
                max_slope = slope
                best_center = center
                best_fit_x = x_window
                best_fit_y = slope * x_window + coeffs[1]

        # If no valid window was found, return NaN/None
        if best_center is None:
            return np.nan, None, None, None

        return max_slope, best_center, best_fit_x, best_fit_y

    def get_shock_gradient_and_position(
            self, image: np.ndarray, window_size: int = 20
    ) -> tuple[float, int]:
        """Find the maximum local slope and its position from a vertical sum profile.

        Parameters
        ----------
        image : numpy.ndarray
            Phase map for analysis.
        window_size : int, default=20
            Window width for local linear fits.

        Returns
        -------
        float
            Maximum local slope (phase units per pixel).
        int
            Center column index of the window with maximum slope.
        """
        fig = plt.figure(figsize=(6, 4))

        profile = np.sum(image, axis=0)
        max_slope, best_center, best_x, best_y = self.find_max_local_slope(profile, window_size)

        # Plot
        plt.plot(np.arange(len(profile)), profile, label="Summed phase", color="blue")
        if best_x is not None and best_y is not None:
            plt.plot(best_x, best_y, "r-", linewidth=2, label="Max local slope")
        plt.xlabel("pixel (10.1 um per pixel)")
        plt.ylabel("Vertical sum of phase")
        plt.title("Shock gradient and position estimator")
        plt.legend()
        self.shock_grad_fig = plt.gcf()

        return max_slope, best_center

    def calculate_delta_plateau(
            self, image: np.ndarray, best_center: int, window_size: int = 20
    ) -> tuple[float, float]:
        """Compute plateau average and peak-to-plateau delta from a vertical sum profile.

        Parameters
        ----------
        image : numpy.ndarray
            Phase map.
        best_center : int
            Center column for the steepest gradient window.
        window_size : int, default=20
            Window width used by the gradient step.

        Returns
        -------
        float
            Plateau average.
        float
            Peak minus plateau value.
        """
        profile = np.sum(image, axis=0)
        x = np.arange(len(profile))

        overall_max = np.max(profile)

        start_index = best_center - (1 + 3) * window_size
        end_index = start_index + 3 * window_size
        start_index = max(start_index, 0)
        end_index = min(end_index, len(profile))

        plateau_region = profile[start_index:end_index]
        plateau_value = np.mean(plateau_region)
        delta = overall_max - plateau_value

        fig = plt.figure(figsize=(6, 4))
        plt.plot(x, profile, label="Summed phase", color="blue")
        plt.xlabel("pixel (10.1 um per pixel")
        plt.ylabel("Vertically summed phase")
        plt.title("Plateau and Peak to Plateau estimate")
        plt.axvspan(
            start_index, end_index, color="green", alpha=0.3, label="Plateau Region"
        )
        max_index = np.argmax(profile)
        plt.axvline(
            x=max_index, color="red", linestyle="--", label=f"Max (col {max_index})"
        )
        plt.axhline(
            y=plateau_value,
            color="purple",
            linestyle="--",
            label=f"Plateau Avg: {plateau_value:.2f}",
        )
        plt.legend()
        self.delta_plateau_fig = fig
        logger.info(
            "Overall max: %f, Plateau avg: %f, Delta: %f",
            overall_max,
            plateau_value,
            delta,
        )
        return plateau_value, delta

