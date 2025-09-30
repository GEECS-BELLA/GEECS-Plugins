"""Plasma density analysis from phase data using preprocessing and Abel inversion.

This module provides a class-based framework for processing phase maps (e.g., from
wavefront sensors) to estimate plasma density. It includes utilities for background
removal, cropping, rotation alignment, Gaussian masking, and thresholding. Density
estimation can be performed using PyAbel. The primary entry point for HTU-style
processing is `PhaseDownrampProcessor`, which implements a focused downramp workflow.

Dependencies
------------
numpy, scipy, matplotlib, opencv-python, pyabel, pint

Notes
-----
- Phase preprocessing typically follows: subtract background → crop → polynomial
  background fit → Gaussian mask → threshold.
- Rotation alignment uses a weighted line fit to column centroids.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, TYPE_CHECKING, NewType

from dataclasses import dataclass
import scipy.ndimage as ndimage
import logging

from image_analysis.base import ImageAnalyzer
from image_analysis.utils import read_imaq_image

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pint import Quantity

    SpatialFrequencyQuantity = NewType(
        "SpatialFrequencyQuantity", Quantity
    )  # [length]**-1
    LengthQuantity = NewType("LengthQuantity", Quantity)  # [length]
    DensityQuantity = NewType("DensityQuantity", Quantity)  # [length]**-3


@dataclass
class PhaseAnalysisConfig:
    """Configuration parameters for phase data analysis.

    Parameters
    ----------
    pixel_scale : float
        Spatial calibration in µm per pixel (vertical).
    wavelength_nm : float
        Probe laser wavelength in nanometers.
    threshold_fraction : float, default=0.5
        Fraction of the maximum used for thresholding the phase data.
    roi : tuple of int, optional
        Region of interest as (x_min, x_max, y_min, y_max) for cropping.
        Negative bounds are interpreted as Python slice semantics (from end).
    background_path : pathlib.Path, optional
        Path to a background image/TSV to subtract from the phase data.
    """

    pixel_scale: float
    wavelength_nm: float
    threshold_fraction: float = 0.5
    roi: Optional[Tuple[int, int, int, int]] = None
    background_path: Optional[Path] = None


def threshold_data(data: np.ndarray, threshold_frac: float) -> np.ndarray:
    """Zero values below a fraction of the maximum while preserving NaNs.

    The threshold is defined as ``threshold = threshold_frac * nanmax(data)`` and
    only non-NaN values below this threshold are set to zero.

    Parameters
    ----------
    data : numpy.ndarray
        Input 2D array to threshold.
    threshold_frac : float
        Fraction of the maximum used to define the threshold.

    Returns
    -------
    numpy.ndarray
        Thresholded array with NaNs preserved.
    """
    data_max: float = np.nanmax(data)
    thresh_val: float = data_max * threshold_frac
    thresh: np.ndarray = np.copy(data)
    mask = (thresh < thresh_val) & ~np.isnan(thresh)
    thresh[mask] = 0
    return thresh


class PhasePreprocessor:
    """Preprocessing utilities for phase data (crop, background, rotation)."""

    def __init__(self, config: PhaseAnalysisConfig, debug_mode: bool = False) -> None:
        """Initialize the preprocessor with configuration and debug flag."""
        self.config = config
        self.phase_array = None
        self.debug_mode = debug_mode

    @staticmethod
    def subtract_background(phase_data: NDArray, background: np.ndarray) -> NDArray:
        """Subtract a background array from the phase map.

        The background must be broadcastable to the shape of the phase array.

        Parameters
        ----------
        phase_data : NDArray
            Phase data array.
        background : numpy.ndarray
            Background array to subtract.

        Returns
        -------
        NDArray
            Background-subtracted phase array (offset-shifted to be positive).
        """
        bkg_subtracted = phase_data - background
        bkg_subtracted = -(bkg_subtracted - np.nanmin(bkg_subtracted))
        bkg_subtracted = abs(bkg_subtracted - np.nanmin(bkg_subtracted))
        return bkg_subtracted

    @staticmethod
    def crop(
        phase_data: NDArray, x_min: int, x_max: int, y_min: int, y_max: int
    ) -> NDArray:
        """Crop the phase array to a rectangular ROI."""
        return phase_data[y_min:y_max, x_min:x_max]

    @staticmethod
    def compute_column_centroids(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute weighted centroids and sums for each column, ignoring NaNs.

        Parameters
        ----------
        data : numpy.ndarray
            2D array of phase values.

        Returns
        -------
        numpy.ndarray
            Weighted centroids per column (NaN or None where invalid).
        numpy.ndarray
            Column sums used as weights.
        """
        if data.size == 0:
            return np.array([]), np.array([])

        centroids = []
        col_sums = []
        for col in data.T:
            valid = ~np.isnan(col)
            col_valid = col[valid]
            if col_valid.size == 0:
                centroids.append(None)
                col_sums.append(None)
                continue
            total = np.sum(col_valid)
            if total != 0:
                indices = np.arange(len(col))[valid]
                centroid = np.sum(indices * col_valid) / total
            else:
                centroid = 0
            centroids.append(centroid)
            col_sums.append(total)

        return np.array(centroids), np.array(col_sums)

    @staticmethod
    def fit_line_to_centroids(
        centroids: np.ndarray, weights: np.ndarray
    ) -> Tuple[float, float]:
        """Fit a weighted line to centroids vs. column index, ignoring NaNs.

        Parameters
        ----------
        centroids : numpy.ndarray
            Per-column centroid positions.
        weights : numpy.ndarray
            Per-column weights (e.g., column sums).

        Returns
        -------
        float
            Slope of the best-fit line.
        float
            Intercept of the best-fit line.

        Raises
        ------
        ValueError
            If fewer than two valid points are available.
        """
        x = np.arange(len(centroids))
        valid = ~np.isnan(centroids) & ~np.isnan(weights)
        if np.sum(valid) < 2:
            raise ValueError("Not enough valid points to perform a linear fit.")
        x_valid = x[valid]
        centroids_valid = centroids[valid]
        weights_valid = weights[valid]
        p = np.polyfit(x_valid, centroids_valid, deg=1, w=weights_valid)
        return p[0], p[1]

    @staticmethod
    def compute_rotation_angle(slope: float) -> float:
        """Convert slope to rotation angle in radians."""
        return np.arctan(slope)

    def rotate_phase_data(self, phase_data: NDArray) -> Tuple[NDArray, dict]:
        """Rotate phase data to align features using weighted column centroids.

        Steps
        -----
        1. Compute weighted column centroids and sums.
        2. Fit a weighted line to obtain slope and intercept.
        3. Convert slope to rotation angle in radians/degrees.
        4. Rotate the image with ``scipy.ndimage.rotate(reshape=True)``.

        Parameters
        ----------
        phase_data : NDArray
            Phase data to be rotated.

        Returns
        -------
        NDArray
            Rotated phase data.
        dict
            Fit parameters with keys: 'Laser Plasma Slope', 'Laser Plasma Intercept',
            'Laser Plasma angle_radians', 'Laser Plasma angle_degrees'.
        """
        phase_array = phase_data
        centroids, col_sums = self.compute_column_centroids(phase_array)
        slope, intercept = self.fit_line_to_centroids(centroids, col_sums)
        angle_rad = self.compute_rotation_angle(slope)
        angle_deg = np.degrees(angle_rad)
        fit_params = {
            "Laser Plasma Slope": slope,
            "Laser Plasma Intercept": intercept,
            "Laser Plasma angle_radians": angle_rad,
            "Laser Plasma angle_degrees": angle_deg,
        }
        rotated_data = ndimage.rotate(
            phase_array, angle=angle_deg, reshape=True, order=1
        )
        phase_array = self.crop(rotated_data, 20, -20, 35, -35)
        return (phase_array, fit_params)

    # ---------------- New Background Removal Workflow ----------------
    @staticmethod
    def poly_design_matrix(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
        """Build a design matrix for 2D polynomials with total degree ≤ order."""
        terms = []
        for total_degree in range(order + 1):
            for i in range(total_degree + 1):
                j = total_degree - i
                terms.append((x**i) * (y**j))
        return np.column_stack(terms)

    def remove_background_polyfit(
        self, phase_data: NDArray, threshold_frac: float = 1, poly_order: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subtract a smooth background using a masked 2D polynomial fit.

        Pixels above a threshold are excluded from the fit to avoid plasma regions.

        Parameters
        ----------
        phase_data : NDArray
            Input phase data.
        threshold_frac : float, default=1
            Fraction of the data dynamic range used to mask bright regions; a value
            of 1 applies no mask.
        poly_order : int, default=2
            Maximum total degree of the 2D polynomial.

        Returns
        -------
        numpy.ndarray
            Background-subtracted phase data.
        numpy.ndarray
            2D polynomial fit evaluated over the full grid.
        numpy.ndarray
            Fitted polynomial coefficients.

        Raises
        ------
        ValueError
            If no valid pixels are available or insufficient points to fit the model.
        """
        data = phase_data
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        thresh_val = data_min + (data_max - data_min) * threshold_frac
        masked_data = data.copy()
        masked_data[masked_data > thresh_val] = np.nan

        nrows, ncols = masked_data.shape
        X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = masked_data.flatten()

        valid = ~np.isnan(Z_flat)
        if not np.any(valid):
            raise ValueError(
                "No valid background pixels available for polynomial fitting."
            )

        X_valid = X_flat[valid]
        Y_valid = Y_flat[valid]
        Z_valid = Z_flat[valid]

        num_terms = (poly_order + 1) * (poly_order + 2) // 2
        if len(Z_valid) < num_terms:
            raise ValueError(
                f"Not enough valid background pixels ({len(Z_valid)}) to fit order {poly_order} "
                f"(requires at least {num_terms} points)."
            )

        G = self.poly_design_matrix(X_valid, Y_valid, poly_order)
        coeffs, residuals, rank, s = np.linalg.lstsq(G, Z_valid, rcond=None)

        G_full = self.poly_design_matrix(X.flatten(), Y.flatten(), poly_order)
        background_fit = (G_full @ coeffs).reshape(nrows, ncols)

        data = data - background_fit
        return data, background_fit, coeffs

    @staticmethod
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
                ((x - center_x) ** 2) / (2 * sigma_x**2)
                + ((y - center_y) ** 2) / (2 * sigma_y**2)
            )
        )
        masked_image = image * gaussian_mask

        if not binarize:
            return masked_image
        else:
            threshold = threshold_factor * np.nanmax(masked_image)
            binary_mask = (masked_image > threshold).astype(np.float32)
            return binary_mask * image


class PhaseDownrampProcessor(ImageAnalyzer):
    """HTU-specific processor for phase downramp features with plotting outputs."""

    def __init__(self, debug_mode: bool = False, **config):
        """Initialize with `PhaseAnalysisConfig` derived from kwargs.

        Parameters
        ----------
        debug_mode : bool, default=False
            If True, show intermediate plots.
        **config
            Keyword arguments for `PhaseAnalysisConfig`.

        Raises
        ------
        ValueError
            If configuration is invalid for `PhaseAnalysisConfig`.
        """
        try:
            self.config = PhaseAnalysisConfig(**config)
        except TypeError as e:
            logging.error(
                "Failed to create PhaseAnalysisConfig from provided config dict."
            )
            logging.error(f"Provided config: {config}")
            raise ValueError(f"Invalid config for PhaseAnalysisConfig: {e}") from e

        self.debug_mode = debug_mode
        self.processor = PhasePreprocessor(self.config)

        if self.config.background_path is not None:
            self.bg_data = read_imaq_image(self.config.background_path)

        super().__init__()

        self.run_analyze_image_asynchronously = False
        self.use_interactive = False
        self.shock_angle_fig = None
        self.shock_grad_fig = None
        self.delta_plateau_fig = None

    def __getstate__(self):
        """Make instance picklable by clearing figure handles."""
        state = self.__dict__.copy()
        state["shock_angle_fig"] = None
        state["shock_grad_fig"] = None
        state["delta_plateau_fig"] = None
        return state

    def process_phase(self, phase_array: NDArray) -> NDArray:
        """Apply HTU phase preprocessing: background, crop, polyfit, mask, threshold."""
        if self.use_interactive:
            plt.imshow(phase_array, cmap="plasma", origin="lower")
            plt.title("first phase")
            plt.show()

        phase_array = self.processor.subtract_background(phase_array, self.bg_data)
        if self.use_interactive:
            plt.imshow(phase_array, cmap="plasma", origin="lower")
            plt.title("bkg subtracted phase")
            plt.show()

        if self.config.roi is not None:
            x_min, x_max, y_min, y_max = self.config.roi
            phase_array = self.processor.crop(phase_array, x_min, x_max, y_min, y_max)

        polynomial_subtraction_result = self.processor.remove_background_polyfit(
            phase_array, threshold_frac=1
        )
        phase_array = polynomial_subtraction_result[0]
        if self.use_interactive:
            plt.imshow(phase_array, cmap="plasma", origin="lower")
            plt.title("bkg fit subtracted step 1")
            plt.show()

        gauss_masked_array = self.processor.apply_gaussian_mask(
            phase_array, binarize=True
        )

        phase_array = gauss_masked_array
        if self.use_interactive:
            plt.imshow(phase_array, cmap="plasma", origin="lower")
            plt.title("gauss masked array")
            plt.show()

        phase_array = threshold_data(phase_array, self.config.threshold_fraction)
        if self.use_interactive:
            plt.imshow(phase_array, cmap="plasma", origin="lower")
            plt.title("bkg fit subtracted and threshed")
            plt.show()

        return phase_array

    def analyze_image(
        self, image: np.array, auxiliary_data: Optional[dict] = None
    ) -> dict[str, Union[float, int, str, np.ndarray]]:
        """Analyze a phase map for downramp features and return results dictionary.

        Parameters
        ----------
        image : numpy.ndarray
            Phase map to analyze.
        auxiliary_data : dict, optional
            Optional metadata; if present, `auxiliary_data['file_path']` is used for saving plots.

        Returns
        -------
        dict
            Standard `ImageAnalyzer` return dictionary with processed image and scalars.
        """
        self.file_path = (
            Path(auxiliary_data["file_path"])
            if auxiliary_data and "file_path" in auxiliary_data
            else None
        )
        try:
            processed_phase = self.process_phase(image)
            logging.info(f"processed {self.file_path}")
        except Exception:
            logging.warning(f"could not process {self.file_path}")
            raise

        scalar_results_dict = self.compile_shock_analysis(processed_phase)

        phase_converted = processed_phase
        phase_scale_16bit_scale = (2**16 - 1) / (2 * np.pi)
        phase_converted = phase_scale_16bit_scale * phase_converted
        phase_converted = phase_converted.astype(np.uint16)

        return_dictionary = self.build_return_dictionary(
            return_image=phase_converted, return_scalars=scalar_results_dict
        )

        return return_dictionary

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
        rotation_result = self.processor.rotate_phase_data(phase_array)
        rotated_phase = rotation_result[0]
        if self.use_interactive:
            plt.imshow(rotated_phase, cmap="plasma", origin="lower")
            plt.title("rotated phase to use for analysis")
            plt.show()
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

        merged = {**results, **rotation_result[1]}

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
            logging.info(
                "Combined shock analysis figure saved to %s", combined_save_path
            )

        results["combined_fig"] = combined_fig
        return merged

    def get_shock_angle(self, phase_array: NDArray):
        """Estimate shock angle by weighted line fit to rotated peak positions."""
        fig = plt.figure()
        threshed = threshold_data(
            phase_array, threshold_frac=self.config.threshold_fraction
        )
        rotated = np.rot90(threshed)
        num_cols = rotated.shape[1]
        col_indices = np.arange(num_cols)
        max_rows = np.array([np.argmax(rotated[:, j]) for j in range(num_cols)])
        col_weights = np.array([np.sum(rotated[:, j]) for j in range(num_cols)])

        try:
            coeffs = np.polyfit(col_indices, max_rows, 1, w=col_weights)
            slope, intercept = coeffs
            logging.info(
                "Weighted fit (rotated): slope = %f, intercept = %f", slope, intercept
            )
        except Exception as e:
            logging.error("Error during weighted fit: %s", e)
            slope, intercept = 0, 0

        W = rotated.shape[0]
        orig_fit_rows = col_indices
        orig_fit_cols = W - 1 - (slope * col_indices + intercept)

        plt.clf()
        plt.imshow(threshed, cmap="plasma", origin="lower")
        plt.title("Shock Angle")
        plt.plot(
            orig_fit_cols,
            orig_fit_rows,
            color="cyan",
            linewidth=2,
            label="Weighted Fit to peaks",
        )
        plt.legend()

        self.shock_angle_fig = fig
        angle_rotated_rad = np.arctan(slope)
        angle_rotated_deg = np.degrees(angle_rotated_rad)
        angle_original_deg = angle_rotated_deg - 90.0
        logging.info(
            "Computed line angle (rotated): %.2f deg; original: %.2f deg",
            angle_rotated_deg,
            angle_original_deg,
        )
        self.shock_angle_fig = fig
        return angle_original_deg

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
        x = np.arange(len(profile))

        plt.plot(x, profile, label="Summed phase", color="blue")
        plt.xlabel("pixel (10.1 um per pixel)")
        plt.ylabel("Vertical sum of phase")
        plt.title("Shock gradient and position estimator")

        window_centers = []
        window_slopes = []

        max_slope = 0.0
        best_center = None
        best_fit_line_x = None
        best_fit_line_y = None

        for start in range(0, len(profile) - window_size + 1):
            end = start + window_size
            x_window = x[start:end]
            y_window = profile[start:end]
            coeffs = np.polyfit(x_window, y_window, 1)
            slope = coeffs[0]
            center = (start + end) // 2
            window_centers.append(center)
            window_slopes.append(10 * slope)
            if abs(slope) > abs(max_slope):
                max_slope = slope
                best_center = center
                best_fit_line_x = x_window
                best_fit_line_y = slope * x_window + coeffs[1]

        if best_fit_line_x is not None and best_fit_line_y is not None:
            plt.plot(
                best_fit_line_x,
                best_fit_line_y,
                "r-",
                linewidth=2,
                label="Max local slope",
            )

        plt.legend()
        self.shock_grad_fig = fig
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
        logging.info(
            "Overall max: %f, Plateau avg: %f, Delta: %f",
            overall_max,
            plateau_value,
            delta,
        )
        return plateau_value, delta


if __name__ == "__main__":
    bkg_path: Path = Path(
        "/Volumes/hdna2/data/Undulator/Y2025/03-Mar/25_0306/scans/Scan015/U_HasoLift/average_phase.tsv"
    )
    phase_file_path = Path(
        "/Volumes/hdna2/data/Undulator/Y2025/03-Mar/25_0306/scans/Scan016/U_HasoLift/Scan016_U_HasoLift_002_postprocessed.tsv"
    )

    config: PhaseAnalysisConfig = PhaseAnalysisConfig(
        pixel_scale=10.1,  # um per pixel (vertical)
        wavelength_nm=800,  # Probe laser wavelength in nm
        threshold_fraction=0.2,  # Threshold fraction for pre-processing
        roi=(10, -10, 10, -100),  # Example ROI: (x_min, x_max, y_min, y_max)
        background_path=bkg_path,  # Background is now a Path
    )
    from dataclasses import asdict

    config_dict = asdict(config)
    print(phase_file_path)
    image_analyzer: PhaseDownrampProcessor = PhaseDownrampProcessor(**config_dict)
    image_analyzer.use_interactive = False
    image_analyzer.analyze_image_file(phase_file_path)

    # # --- Using the PyAbel inversion technique ---
    # pyabel_lineout, pyabel_density = image_analyzer.get_density(technique='pyabel')
    # image_analyzer.plot_density(pyabel_density, pyabel_lineout)
