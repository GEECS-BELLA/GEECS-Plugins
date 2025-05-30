"""
DensityAnalysis Module
========================

This module provides a class-based framework for performing plasma density analysis
from phase data. It encapsulates functionality for loading phase data (from TSV files or images),
preprocessing the data (background subtraction and cropping), and computing the plasma density
map using different Abel inversion techniques. It was originally designed to use the tsv phase output
from the haso_analysis class in this same package.

Classes and Components:
-------------------------
- **PhaseAnalysisConfig**: A dataclass that groups configuration parameters such as the pixel scale,
  wavelength (in nm), threshold fraction, region-of-interest (ROI), and background. This centralizes
  configuration and makes it easy to pass settings around.
  
- **PhaseDataLoader**: Loads the raw phase data from a file. It supports both TSV files (as numerical data)
  and image files (using OpenCV).
  
- **PhasePreprocessor**: Provides methods for preprocessing the phase data, including background subtraction
  and cropping to a specified ROI. Note that background subtraction is applied prior to cropping
  to ensure consistency.
  
- **InversionTechnique** (Abstract Base Class): Defines the common interface for all inversion techniques.
  
- **CustomAbelInversion**: Implements a manual Abel inversion routine using custom numerical methods.
  
- **PyAbelInversion**: Implements an inversion using the external PyAbel package.
  
- **DensityAnalysis**: The main class that ties together data loading, preprocessing, and inversion.
  It uses a `PhaseAnalysisConfig` object for configuration and provides methods to compute the density map
  and plot the results.

Usage Example:
--------------
Below is an example of how you might use this module:

    from density_analysis import DensityAnalysis, PhaseAnalysisConfig

    # Define the analysis configuration
    config = PhaseAnalysisConfig(
        pixel_scale=4.64,            # µm per pixel (vertical)
        wavelength_nm=800,           # Laser probe wavelength in nm
        threshold_fraction=0.5,      # Fraction to threshold the phase data
        roi=(50, 300, 30, 250),      # Optional ROI: (x_min, x_max, y_min, y_max)
        background=None             # Optional background array for subtraction
    )

    # Create an analyzer for a given phase data file
    analyzer = DensityAnalysis("path/to/phase_data.tsv", config)

    # Compute the density using the custom inversion technique
    vertical_lineout, density_map = analyzer.get_density(technique='custom')

    # Plot the density map and vertical lineout
    analyzer.plot_density(density_map, vertical_lineout)

Dependencies:
-------------
- numpy
- opencv-python (cv2)
- matplotlib
- pyabel

Additional Notes:
-----------------
- This module uses type hints and (optionally) the future import `from __future__ import annotations`
  to improve type annotation clarity and allow forward references.
- The module is designed to be flexible and extensible; additional inversion techniques or preprocessing
  steps can be added as needed.
"""


from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import abel  # pip install pyabel
import abc
from typing import Optional, Tuple, Union, TYPE_CHECKING, NewType

from dataclasses import dataclass
import scipy.ndimage as ndimage
import logging

from image_analysis import ureg, Q_
from image_analysis.base import ImageAnalyzer
from image_analysis.utils import read_imaq_image

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pint import Quantity
    SpatialFrequencyQuantity = NewType('SpatialFrequencyQuantity', Quantity) # [length]**-1
    LengthQuantity = NewType('LengthQuantity', Quantity) # [length]
    DensityQuantity = NewType('DensityQuantity', Quantity) # [length]**-3

@dataclass
class PhaseAnalysisConfig:
    """
    Configuration parameters for phase data analysis.
    
    Attributes:
      pixel_scale: float
          Spatial calibration in µm per pixel (vertical).
      wavelength_nm: float
          Probe laser wavelength in nanometers.
      threshold_fraction: float
          Fraction used to threshold the phase data.
      roi: Optional[Tuple[int, int, int, int]]
          Region of interest (x_min, x_max, y_min, y_max) to crop the phase data.
      background: Optional[np.ndarray]
          A background array to subtract from the phase data.
    """
    pixel_scale: float
    wavelength_nm: float
    threshold_fraction: float = 0.5
    roi: Optional[Tuple[int, int, int, int]] = None
    background_path: Optional[Path] = None

def threshold_data(data: np.ndarray, threshold_frac: float) -> np.ndarray:
    """
    Apply a simple threshold to a 2D array.
    Pixels with values below:
         data_min + threshold_frac*(data_max - data_min)
    are set to zero.
    NaNs are ignored in computing the threshold and remain unchanged.
    """
    data_max: float = np.nanmax(data)
    thresh_val: float = data_max * threshold_frac
    thresh: np.ndarray = np.copy(data)
    # Create a mask that only selects non-NaN values below the threshold.
    mask = (thresh < thresh_val) & ~np.isnan(thresh)
    thresh[mask] = 0

    return thresh

class PhasePreprocessor:
    """
    Handles some basic pre-processing like cropping and background subtraction for phase data.
    """
    def __init__(self, config: PhaseAnalysisConfig, debug_mode: bool = False) -> None:
        self.config = config
        self.phase_array = None
        self.debug_mode = debug_mode


    @staticmethod
    def subtract_background(phase_data:NDArray, background: np.ndarray) -> NDArray:
        """
        Subtract a background array from the phase array.
        The background array must be broadcastable to the shape of the phase array.
        """
        bkg_subtracted = phase_data - background
        bkg_subtracted = -(bkg_subtracted - np.nanmin(bkg_subtracted))
        bkg_subtracted = abs(bkg_subtracted - np.nanmin(bkg_subtracted))

        return bkg_subtracted

    @staticmethod
    def crop(phase_data:NDArray, x_min: int, x_max: int, y_min: int, y_max: int) -> NDArray:
        """
        Crop the phase array to the specified region of interest (ROI).
        """
        return phase_data[y_min:y_max, x_min:x_max]

    @staticmethod
    def compute_column_centroids(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute weighted center-of-mass and column sums for each column in data,
        ignoring any NaN values. If the input array is empty, return empty arrays.

        Returns:
            centroids: np.ndarray of weighted centroids (one per column).
            col_sums: np.ndarray of column sums.
        """
        # If the input data is empty, return empty arrays.
        if data.size == 0:
            return np.array([]), np.array([])

        centroids = []
        col_sums = []
        for col in data.T:  # iterate over columns
            # Optionally, ignore NaNs in the column.
            valid = ~np.isnan(col)
            col_valid = col[valid]

            # If the column is completely invalid, set defaults.
            if col_valid.size == 0:
                centroids.append(None)
                col_sums.append(None)
                continue

            total = np.sum(col_valid)
            if total != 0:
                # Use only valid indices.
                indices = np.arange(len(col))[valid]
                centroid = np.sum(indices * col_valid) / total
            else:
                centroid = 0
            centroids.append(centroid)
            col_sums.append(total)

        return np.array(centroids), np.array(col_sums)

    @staticmethod
    def fit_line_to_centroids(centroids: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
        """
        Fit a weighted linear model to the centroids versus column indices,
        ignoring any NaN values in centroids or weights.

        Returns:
            slope, intercept of the best-fit line.
        """
        x = np.arange(len(centroids))
        # Create a mask for valid (non-NaN) values in both centroids and weights.
        valid = ~np.isnan(centroids) & ~np.isnan(weights)

        # Ensure there are enough valid points (at least 2 for a linear fit).
        if np.sum(valid) < 2:
            raise ValueError("Not enough valid points to perform a linear fit.")

        x_valid = x[valid]
        centroids_valid = centroids[valid]
        weights_valid = weights[valid]

        p = np.polyfit(x_valid, centroids_valid, deg=1, w=weights_valid)
        return p[0], p[1]

    @staticmethod
    def compute_rotation_angle(slope: float) -> float:
        """
        Compute the rotation angle (in radians) from the slope of the fitted line.
        """
        return np.arctan(slope)

    def rotate_phase_data(self, phase_data: NDArray) -> Tuple[NDArray,dict]:
        """
        Rotate the phase data to correct for any tilt based on the weighted column centers.

        This workflow assumes that the phase data in self.phase_array has already been
        background-subtracted, cropped and appropriately thresholded as desired.

        Steps:
          1. Compute weighted column centroids and column sums.
          2. Fit a weighted linear model (centroid vs. column index) to obtain the slope.
          3. Compute the rotation angle (in radians and degrees) from the slope.
          4. Rotate the original phase data using scipy.ndimage.rotate with reshape=True.

        Parameters:
            phase_data: NDArray, phase data to be rotated and analyzed

        Returns:
            final_data: The rotated phase data (2D array).
            fit_params: Dictionary with keys 'Laser Plasma Slope', 'Laser Plasma Intercept',
                'Laser Plasma angle_radians', 'Laser Plasma angle_degrees'.
        """
        phase_array = phase_data
        # Step 1: Compute weighted column centroids and column sums.
        centroids, col_sums = self.compute_column_centroids(phase_array)

        # Step 2: Fit a weighted line to the centroids.
        slope, intercept = self.fit_line_to_centroids(centroids, col_sums)

        # Step 3: Compute rotation angle.
        angle_rad = self.compute_rotation_angle(slope)
        angle_deg = np.degrees(angle_rad)
        fit_params = {
            'Laser Plasma Slope': slope,
            'Laser Plasma Intercept': intercept,
            'Laser Plasma angle_radians': angle_rad,
            'Laser Plasma angle_degrees': angle_deg
        }

        # Step 4: Rotate the original phase data.
        # Using reshape=True so that the output shape is adjusted to contain the entire rotated image.
        rotated_data = ndimage.rotate(phase_array, angle=angle_deg, reshape=True, order=1)

        phase_array = self.crop(rotated_data,20, -20, 35, -35)

        return (phase_array,fit_params)

    # ---------------- New Background Removal Workflow ----------------
    @staticmethod
    def poly_design_matrix(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
        """
        Construct a design matrix for a 2D polynomial in x and y of total degree <= order.

        Parameters:
            x, y: 1D arrays (flattened coordinate arrays for each data point).
            order: Maximum total degree of polynomial terms.

        Returns:
            A 2D array where each column corresponds to one monomial term x^i * y^j,
            with i+j <= order.
        """
        terms = []
        for total_degree in range(order + 1):
            for i in range(total_degree + 1):
                j = total_degree - i
                terms.append((x ** i) * (y ** j))
        return np.column_stack(terms)

    def remove_background_polyfit(self, phase_data:NDArray, threshold_frac: float = 1, poly_order: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        utility to do some more background subtraction using 2D polynomial fits.

        Parameters:
            threshold_frac: Fraction of the peak value, above which is set to nan. Meant to be used
                to 'mask out' the part of the phase data that contains the plasma. Default value of
                1 applys no mask.
            poly_order: The maximum order of the polynomial to fit.

        Returns:
            background_fit: The 2D background (from the polynomial fit).
            coeffs: The polynomial coefficients as a 1D array.
        """
        data = phase_data
        # Compute the dynamic range of the data.
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        thresh_val = data_min + (data_max - data_min) * threshold_frac
        # Create a masked copy: set pixels above the threshold to NaN.
        masked_data = data.copy()
        masked_data[masked_data > thresh_val] = np.nan
        # plt.imshow(masked_data, cmap='plasma', origin='lower')
        # plt.title("threshed data for bkg fitting")
        # plt.show()

        # Build coordinate grids.
        nrows, ncols = masked_data.shape
        X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = masked_data.flatten()

        # Remove NaNs.
        valid = ~np.isnan(Z_flat)
        if not np.any(valid):
            raise ValueError("No valid background pixels available for polynomial fitting.")

        X_valid = X_flat[valid]
        Y_valid = Y_flat[valid]
        Z_valid = Z_flat[valid]

        # Check if the number of valid points is sufficient.
        # Number of terms in a 2D polynomial of order n is (n+1)(n+2)/2.
        num_terms = (poly_order + 1) * (poly_order + 2) // 2
        if len(Z_valid) < num_terms:
            raise ValueError(f"Not enough valid background pixels ({len(Z_valid)}) "
                             f"to fit a polynomial of order {poly_order} (requires at least {num_terms} points).")

        # Build the design matrix for valid points.
        G = self.poly_design_matrix(X_valid, Y_valid, poly_order)
        coeffs, residuals, rank, s = np.linalg.lstsq(G, Z_valid, rcond=None)

        # Evaluate the polynomial over the full grid.
        G_full = self.poly_design_matrix(X.flatten(), Y.flatten(), poly_order)
        background_fit = (G_full @ coeffs).reshape(nrows, ncols)

        # Subtract the background fit from the original phase data.
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
            threshold_factor: float = 0.01
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Applies a Gaussian mask to a 2D image.

        Parameters:
            image (np.ndarray): 2D grayscale image.
            center_x (int, optional): X-coordinate of the Gaussian center (default: image center).
            center_y (int, optional): Y-coordinate of the Gaussian center (default: image center).
            sigma_x (float): Standard deviation in the x-direction.
            sigma_y (float): Standard deviation in the y-direction.
            binarize (bool): If True, binarizes the masked image based on threshold_factor.
                             If False, returns the Gaussian masked image.
            threshold_factor (float): Fraction of the maximum masked value to use as threshold.

        Returns:
            If binarize is False:
                np.ndarray: The masked image.
            If binarize is True:
                tuple[np.ndarray, np.ndarray]: A tuple where the first element is the final image (binary mask multiplied by the image)
                and the second element is the binary mask.
        """
        rows, cols = image.shape
        if center_x is None:
            center_x = cols // 2
        if center_y is None:
            center_y = rows // 2

        y, x = np.mgrid[0:rows, 0:cols]

        # Compute the Gaussian mask.
        gaussian_mask = np.exp(-(((x - center_x) ** 2) / (2 * sigma_x ** 2) +
                                 ((y - center_y) ** 2) / (2 * sigma_y ** 2)))
        masked_image = image * gaussian_mask

        if not binarize:
            return masked_image
        else:
            threshold = threshold_factor * np.nanmax(masked_image)
            binary_mask = (masked_image > threshold).astype(np.float32)
            return binary_mask * image

class PhaseDownrampProcessor(ImageAnalyzer):
    """
    Handles cropping and background subtraction for phase data.
    """

    def __init__(self, debug_mode: bool = False, **config):
        # Validate or document with the dataclass
        try:
            # This will raise TypeError if required fields are missing or types don't match
            self.config = PhaseAnalysisConfig(**config)
        except TypeError as e:
            logging.error("Failed to create PhaseAnalysisConfig from provided config dict.")
            logging.error(f"Provided config: {config}")
            raise ValueError(f"Invalid config for PhaseAnalysisConfig: {e}") from e

        self.debug_mode = debug_mode
        self.processor  = PhasePreprocessor(self.config)

        # If background is provided, treat it as a Path.
        # Assumption here is that background is loaded only at
        # initialization.
        if self.config.background_path is not None:
            self.bg_data  = read_imaq_image(self.config.background_path)

        super().__init__()


        self.run_analyze_image_asynchronously = False

        self.use_interactive = False

        self.shock_angle_fig = None
        self.shock_grad_fig = None
        self.delta_plateau_fig = None

    def __getstate__(self):
        # Make a copy of the instance's dictionary.
        state = self.__dict__.copy()
        # Remove or clear non-pickleable attributes. Necessary for use with parallel processing
        # in Array2DScanAnalysis
        state['shock_angle_fig'] = None
        state['shock_grad_fig'] = None
        state['delta_plateau_fig'] = None
        return state

    def process_phase(self, phase_array: NDArray) -> NDArray:
        """
        Parameters: file_path
        """
        # logging.info(f'loader file path: {file_path}')
        # loader: PhaseDataLoader = PhaseDataLoader(file_path)
        # phase_array = loader.load_data()
        if self.use_interactive:
            plt.imshow(phase_array, cmap='plasma', origin='lower')
            plt.title("first phase")
            plt.show()

        phase_array = self.processor.subtract_background(phase_array, self.bg_data)
        if self.use_interactive:
            plt.imshow(phase_array, cmap='plasma', origin='lower')
            plt.title("bkg subtracted phase")
            plt.show()

        if self.config.roi is not None:
            x_min, x_max, y_min, y_max = self.config.roi
            phase_array = self.processor.crop(phase_array, x_min, x_max, y_min, y_max)

        polynomial_subtraction_result = self.processor.remove_background_polyfit(phase_array, threshold_frac=1)
        phase_array = polynomial_subtraction_result[0]
        if self.use_interactive:
            plt.imshow(phase_array, cmap='plasma', origin='lower')
            plt.title("bkg fit subtracted step 1")
            plt.show()

        gauss_masked_array = self.processor.apply_gaussian_mask(phase_array, binarize=True)

        phase_array = gauss_masked_array
        if self.use_interactive:
            plt.imshow(phase_array, cmap='plasma', origin='lower')
            plt.title("gauss masked array")
            plt.show()

        phase_array = threshold_data(phase_array, self.config.threshold_fraction)
        if self.use_interactive:
            plt.imshow(phase_array, cmap='plasma', origin='lower')
            plt.title("bkg fit subtracted and threshed")
            plt.show()

        return phase_array

    def analyze_image(self, image: np.array, auxiliary_data: Optional[dict] = None) -> dict[
        str, Union[float, int, str, np.ndarray]]:

        """
        Apply some HTU specific processing of a phase map showing a density down ramp feature.

        Parameters:
            image (np.array): image

        Returns:
            A dictionary containing results (e.g., phase map and/or related parameters).
        """
        self.file_path = Path(auxiliary_data['file_path']) if auxiliary_data and 'file_path' in auxiliary_data else None
        try:
            processed_phase = self.process_phase(image)
            logging.info(f'processed {self.file_path}')

        except Exception as e:
            logging.warning(f'could not process {self.file_path}')
            raise

        scalar_results_dict = self.compile_shock_analysis(processed_phase)

        phase_converted = processed_phase
        phase_scale_16bit_scale = (2 ** 16 - 1) / (2 * np.pi)
        phase_converted = phase_scale_16bit_scale * phase_converted
        phase_converted = phase_converted.astype(np.uint16)

        return_dictionary = self.build_return_dictionary(return_image=phase_converted,
                                                         return_scalars=scalar_results_dict)

        return return_dictionary

    def compile_shock_analysis(self, phase_array: NDArray, window_size: int = 20) -> dict:
        """
        Runs all shock analysis methods, collects computed metrics,
        and compiles a composite figure (1 row × 3 columns) of intermediary plots.

        It does the following:
          1. Calls get_shock_angle to compute the shock angle.
          2. Calls get_shock_gradient_and_position to compute the local slope and its position.
          3. Calls calculate_delta_plateau to compute the plateau average and delta.
          4. Retrieves the corresponding figures from each method.
          5. Converts each figure to an image array and compiles them into a single composite figure.
          6. Returns a dictionary with all computed metrics and the composite figure.

        Parameters:
          phase_array : NDArray
              The input phase image.
          window_size : int, optional
              The window size used for gradient and plateau calculations (default is 20).

        Returns:
          dict: A dictionary with keys:
                'shock_angle': computed shock angle (in degrees),
                'max_slope': the maximum local slope,
                'best_center': the column index corresponding to the steepest gradient,
                'plateau_value': the average intensity of the plateau region,
                'delta': the difference between the overall max and the plateau average,
                'combined_fig': the composite matplotlib figure.
        """
        # Run the individual analysis methods.
        rotation_result = self.processor.rotate_phase_data(phase_array)
        rotated_phase  =rotation_result[0]
        if self.use_interactive:
            plt.imshow(rotated_phase, cmap='plasma', origin='lower')
            plt.title("rotated phase to use for analysis")
            plt.show()
        shock_angle = self.get_shock_angle(rotated_phase)
        max_slope, best_center = self.get_shock_gradient_and_position(rotated_phase, window_size=window_size)
        plateau_value, delta = self.calculate_delta_plateau(rotated_phase, best_center, window_size=window_size)

        results = {
            "Plasma downramp shock_angle": shock_angle,
            "Plasma downramp shock slope (phase/pixel)": max_slope,
            "Plasma downramp shock location (pixel)": best_center,
            "Plasma downramp plateau avg (phase)": plateau_value,
            "Plasma downramp peak to plateau (phase)": delta,
        }

        merged = {**results, **rotation_result[1]}

        def fig_to_array(fig):
            """
            Converts a matplotlib figure to a NumPy array using its true pixel dimensions.
            """
            # Force a draw on the canvas.
            fig.canvas.draw()
            # Get the figure size in pixels.
            w, h = fig.get_size_inches() * fig.dpi
            w, h = int(w), int(h)
            # Get the RGB buffer from the canvas.
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            # Reshape to an image array.
            return buf.reshape(h, w, 3)

        # Retrieve the saved figure objects.
        fig1 = self.shock_angle_fig  # from get_shock_angle
        fig2 = self.shock_grad_fig  # from get_shock_gradient_and_position
        fig3 = self.delta_plateau_fig  # from calculate_delta_plateau

        # Convert each figure to an image array.
        img1 = fig_to_array(fig1)
        plt.close(fig1)  # Close the figure object.
        img2 = fig_to_array(fig2)
        plt.close(fig2)  # Close the figure object.
        img3 = fig_to_array(fig3)
        plt.close(fig3)  # Close the figure object.


        # Create a composite figure with 1 row and 3 columns.
        combined_fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img1)
        axs[0].set_title("Shock Angle Determination")
        axs[0].axis('off')

        axs[1].imshow(img2)
        axs[1].set_title("Shock Gradient & Position")
        axs[1].axis('off')

        axs[2].imshow(img3)
        axs[2].set_title("Plateau estimation")
        axs[2].axis('off')

        plt.tight_layout()
        if self.file_path is not None:
            combined_save_path = self.file_path.parent / f'{self.file_path.stem}_combined_shock_analysis.pdf'
            # combined_save_path = "combined_shock_analysis.pdf"

            combined_fig.savefig(combined_save_path)
            plt.close(combined_fig)  # Correct way to close the figure.
            logging.info("Combined shock analysis figure saved to %s", combined_save_path)

        results["combined_fig"] = combined_fig

        return merged

    def get_shock_angle(self, phase_array: NDArray):

        # Create a new figure and store it.
        # fig = plt.figure(figsize=(6, 4))
        fig = plt.figure()

        # Threshold the data.
        threshed = threshold_data(phase_array, threshold_frac=self.config.threshold_fraction)

        # Rotate the image by 90 degrees (counter-clockwise).
        # np.rot90 rotates the array counter-clockwise by 90°.
        rotated = np.rot90(threshed)

        # Now treat columns of the rotated image as "rows" for analysis.
        num_cols = rotated.shape[1]
        col_indices = np.arange(num_cols)

        # For each column in the rotated image, find the row index of the maximum value.
        max_rows = np.array([np.argmax(rotated[:, j]) for j in range(num_cols)])

        # Optionally, compute weights (for example, the sum of each column).
        col_weights = np.array([np.sum(rotated[:, j]) for j in range(num_cols)])

        # Fit a line to the points (col index, max row index) using a weighted linear fit.
        try:
            # We want to fit max_rows as a function of col_indices.
            coeffs = np.polyfit(col_indices, max_rows, 1, w=col_weights)
            slope, intercept = coeffs
            logging.info("Weighted fit (rotated): slope = %f, intercept = %f", slope, intercept)
        except Exception as e:
            logging.error("Error during weighted fit: %s", e)
            slope, intercept = 0, 0

        # Compute the fitted line on the rotated image.
        fitted_line = slope * col_indices + intercept

        # Now, convert the markers and fitted line from the rotated system back to the original.
        # Let W = rotated.shape[0] (this equals the original image width).
        W = rotated.shape[0]

        # For each column j in the rotated image, the corresponding original coordinates are:
        #   row_orig = j,  col_orig = (W - 1) - (value from rotated)
        orig_marker_rows = col_indices  # because for each column j in rotated, original row = j.
        orig_marker_cols = W - 1 - max_rows

        # For the fitted line, for each j (rotated column), the fitted line gives a rotated row coordinate.
        orig_fit_rows = col_indices  # same as above.
        orig_fit_cols = W - 1 - (slope * col_indices + intercept)

        # Now clear the current figure and plot the original thresholded image.
        plt.clf()  # Clear the figure.
        plt.imshow(threshed, cmap='plasma', origin='lower')
        plt.title("Shock Angle")

        # Plot the fitted line.
        plt.plot(orig_fit_cols, orig_fit_rows, color='cyan', linewidth=2, label='Weighted Fit to peaks')
        plt.legend()

        self.shock_angle_fig = fig

        # Compute the angle from the rotated slope.
        angle_rotated_rad = np.arctan(slope)
        angle_rotated_deg = np.degrees(angle_rotated_rad)
        # Adjust back to the original orientation.
        angle_original_deg = angle_rotated_deg - 90.0

        logging.info("Computed line angle (rotated): %.2f deg; original: %.2f deg", angle_rotated_deg,
                     angle_original_deg)
        self.shock_angle_fig = fig
        return angle_original_deg

    def get_shock_gradient_and_position(self, image: np.ndarray, window_size: int = 20) -> tuple[float, int]:
        """
        Analyzes the phase image by:
          1. Summing the image over rows to produce a 1D profile (assumed to represent intensity vs. column).
          2. Visualizing the profile.
          3. Computing the local slope over a moving window of a specified size.
          4. Finding the window with the largest (absolute) local slope.
          5. Overlaying the corresponding fitted line segment on the plot.
          6. Also plotting the local slopes as a function of window center.
          7. Returning the maximum local slope and the center column index of that window.

        Parameters:
          image : np.ndarray
              The original (unrotated) image.
          window_size : int, optional
              The number of pixels in the moving window for the local fit (default is 20).

        Returns:
          max_slope : float
              The slope (in intensity units per pixel) of the window with the largest (absolute) slope.
          best_center : int
              The column index corresponding to the center of the window with the maximum slope.
        """

        # Create a new figure and store it.
        fig = plt.figure(figsize=(6, 4))

        # Sum over rows to produce a 1D profile (intensity vs. column)
        profile = np.sum(image, axis=0)
        x = np.arange(len(profile))

        plt.plot(x, profile, label="Summed phase", color="blue")
        plt.xlabel("pixel (10.1 um per pixel)")
        plt.ylabel("Vertical sum of phase")
        plt.title("Shock gradient and position estimator")

        # Lists to store the slope of each window and its center position.
        window_centers = []
        window_slopes = []

        max_slope = 0.0
        best_center = None
        best_fit_line_x = None
        best_fit_line_y = None

        # Slide over the profile with a window of size 'window_size'
        for start in range(0, len(profile) - window_size + 1):
            end = start + window_size
            x_window = x[start:end]
            y_window = profile[start:end]

            # Fit a line: y = m*x + b using np.polyfit
            coeffs = np.polyfit(x_window, y_window, 1)  # coeffs[0] is slope, coeffs[1] is intercept
            slope = coeffs[0]
            center = (start + end) // 2

            window_centers.append(center)
            window_slopes.append(10*slope)

            # Check if the absolute slope is larger than the current best
            if abs(slope) > abs(max_slope):
                max_slope = slope
                best_center = center
                best_fit_line_x = x_window
                best_fit_line_y = slope * x_window + coeffs[1]

        # Overlay the best-fit line segment if available.
        if best_fit_line_x is not None and best_fit_line_y is not None:
            plt.plot(best_fit_line_x, best_fit_line_y, 'r-', linewidth=2, label="Max local slope")

        plt.legend()
        self.shock_grad_fig = fig

        return max_slope, best_center

    def calculate_delta_plateau(self, image: np.ndarray, best_center: int, window_size: int = 20) -> tuple[
        float, float]:
        """
        Calculates the plateau value and the delta between the maximum intensity and the plateau.

        The plateau is defined as the average intensity over a region starting 2 window sizes
        to the left of the steepest slope (best_center) and spanning 3 window sizes. The delta is
        defined as the overall maximum intensity in the profile minus the plateau value.

        Parameters:
          image : np.ndarray
              The original (unrotated) image.
          best_center : int
              The column index where the steepest slope was found.
          window_size : int, optional
              The size of the window (in pixels) used in earlier analysis (default is 20).

        Returns:
          plateau_value : float
              The average intensity of the plateau region.
          delta : float
              The difference between the overall maximum intensity and the plateau average.
        """

        # Sum the image over rows to produce a 1D profile.
        profile = np.sum(image, axis=0)
        x = np.arange(len(profile))

        overall_max = np.max(profile)

        # Define the plateau region.
        # Start: best_center minus 2 window sizes.
        start_index = best_center - (1 + 3) * window_size
        # End: start + 5 window sizes.
        end_index = start_index + 3 * window_size

        # Ensure the indices are within bounds.
        start_index = max(start_index, 0)
        end_index = min(end_index, len(profile))

        # Compute the plateau average.
        plateau_region = profile[start_index:end_index]
        plateau_value = np.mean(plateau_region)

        # Compute delta as overall maximum minus plateau.
        delta = overall_max - plateau_value

        # Plot the intensity profile.
        fig = plt.figure(figsize=(6, 4))
        plt.plot(x, profile, label="Summed phase", color="blue")
        plt.xlabel("pixel (10.1 um per pixel")
        plt.ylabel("Vertically summed phase")
        plt.title("Plateau and Peak to Plateau estimate")

        # Mark the plateau region on the plot.
        plt.axvspan(start_index, end_index, color="green", alpha=0.3, label="Plateau Region")
        # Mark the overall maximum.
        max_index = np.argmax(profile)
        plt.axvline(x=max_index, color="red", linestyle="--", label=f"Max (col {max_index})")

        # Also, draw a horizontal line at the plateau value.
        plt.axhline(y=plateau_value, color="purple", linestyle="--", label=f"Plateau Avg: {plateau_value:.2f}")

        plt.legend()
        self.delta_plateau_fig = fig # from calculate_delta_plateau

        logging.info("Overall max: %f, Plateau avg: %f, Delta: %f", overall_max, plateau_value, delta)

        return plateau_value, delta

class InversionTechnique(abc.ABC):
    """
    Abstract base class for an inversion technique.
    """
    def __init__(self, phase_data: np.ndarray, image_resolution: float, wavelength_nm: float) -> None:
        self.phase_data: np.ndarray = phase_data
        self.image_resolution: float = image_resolution  # µm per pixel (vertical)
        self.wavelength_nm: float = wavelength_nm

    @abc.abstractmethod
    def invert(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the inversion.
        Should return a tuple: (vertical_lineout, density_map)
        """
        pass

class PyabelInversion(InversionTechnique):
    def __init__(self, phase_data: np.ndarray, image_resolution: float, wavelength_nm: float) -> None:
        super().__init__(phase_data, image_resolution, wavelength_nm)

    def invert(self):

        # Assuming `wavefront_raw` is a NumPy array in nanometers:
        wavefront_quantity = - self.phase_data * self.wavelength_nm
        wavefront_quantity = Q_(wavefront_quantity, 'nanometer')
        density = self.calculate_density(wavefront=wavefront_quantity, image_resolution = Q_(self.image_resolution,'micrometer'))
        density = density.to("1/cm**3")

        center_index = (density.shape[0] - 1) // 2

        # Define how many rows to average (e.g. 10) and calculate half that number.
        num_lines = 40
        half_lines = num_lines // 2

        # Extract the rows around the center.
        # This will take 5 rows above and 5 rows below the center (if num_lines is even, adjust as needed).
        rows_to_average = density[center_index - half_lines: center_index + half_lines + 1, :]

        # Compute the mean on the magnitude, then reattach the units.
        averaged_lineout_magnitude = np.mean(rows_to_average.magnitude, axis=0)
        averaged_lineout = Q_(averaged_lineout_magnitude, density.units)

        return averaged_lineout, density

    def calculate_density(self,
                          wavefront: np.ndarray,
                          wavelength=Q_(800, 'nm'),
                          image_resolution: Optional[LengthQuantity] = None
                          ) -> DensityQuantity:
        """ Convert the wavefront into a density map, using equation for plasma refraction.

        Parameters
        ----------
        wavefront : 2d array, units [length]
            A wavefront, with [length] units, as produced by QWLSIImageAnalyzer.calculate_wavefront().
            Should be background-subtracted, and baselined, i.e. wavefront should be 0 where there's no plasma
        wavelength : [length] Quantity
        image_resolution: [length] Quantity or None
            the real-world length represented by a pixel, which can be different from CAMERA_RESOLUTION if there
            are optics between object and grating/camera. If None (default), same as self.CAMERA_RESOLUTION.

        Returns
        -------
        density : 2d array, units [length]^-3
            Slice of the 3D (cylindrically symmetric) electron density profile.
            Note that because of the centering in the axis=0 direction, the number of rows may be one less
            than the given wavefront array.
            The center row in this array (row = (numrows - 1) // 2) represents the cylinder axis.

        """


        # unit-aware version of abel.Transform()
        @ureg.wraps('nm/pixel', 'nm')
        def abel_transform_ua(wavefront):
            return abel.Transform(
                wavefront.T,  # transpose because the Abel inverse is done around axis = 0
                direction='inverse',
                method='onion_bordas',
                origin='convolution', center_options={'axes': 1},  # only center in the perpendicular-to-laser direction
                symmetry_axis=0,  # assume above and below laser axis have a symmetric profile
            ).transform.T  # transpose back

        # a slice of the cylindrically symmetric optical path length disturbance profile
        # As the optical path disturbance is given in nanometers, and this slice gives the contribution of one
        # pixel's distance to the total wavefront shift, this is in units of [length]/[length], or dimensionless.
        self.optical_path_change_per_distance = abel_transform_ua(wavefront) / (image_resolution / ureg.pixel)

        # from https://www.ipp.mpg.de/2882460/anleitung.pdf:
        #   phi = -lambda*e^2/(4 pi c^2 e0 me) integrate(n(z) dz)
        # with phi/2pi = wavefront/lambda, and C = -e^2/(4 pi c^2 e0 me)
        #   2pi/lambda * d(wavefront)/dz = C * lambda * density
        #   density = 2pi/lambda^2 / C * d(wavefront)/dz

        C = -(ureg.elementary_charge ** 2 / (
                    4 * np.pi * ureg.speed_of_light ** 2 * ureg.vacuum_permittivity * ureg.electron_mass))
        self.density = 2 * np.pi / wavelength ** 2 / C * self.optical_path_change_per_distance

        return self.density

# class DensityAnalysis:
#     """
#     Main class for performing density analysis.
#
#     This class loads and pre-processes phase data and then uses a specified
#     inversion technique (custom or PyAbel) to compute a plasma density map.
#     """
#
#     def __init__(self, phase_file: Path, config: PhaseAnalysisConfig) -> None:
#         """
#         Parameters:
#           phase_file: Path
#               Path to the phase data file (TSV or image).
#           config: PhaseAnalysisConfig
#               Configuration parameters (pixel scale, wavelength, threshold fraction, ROI, background).
#         """
#         self.phase_file: Path = phase_file
#         self.config: PhaseAnalysisConfig = config
#
#         # Pre-process: subtract background first, then crop if specified.
#         processor  = PhasePreprocessor(phase_file_path, config)
#         processor.process_phase()
#
#
#     def get_density(self, technique: str = 'pyabel') -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Compute the density map and vertical lineout using a chosen inversion technique.
#
#         Parameters:
#           technique: str
#               'pyabel' for the PyAbel method.
#
#         Returns:
#           (vertical_lineout, density_map)
#         """
#         if technique == 'pyabel':
#             inverter = PyabelInversion(self.thresh_data, self.config.pixel_scale, self.config.wavelength_nm)
#         else:
#             raise ValueError("Technique must be 'pyabel'.")
#         return inverter.invert()
#
#     def plot_density(self, density_map: Quantity, vertical_lineout: Quantity) -> None:
#         """
#         Plot the 2D density map and the vertical lineout using units.
#
#         Parameters:
#             density_map: A Pint Quantity (2D array) representing the density map.
#             vertical_lineout: A Pint Quantity (1D array) representing the vertical lineout.
#         """
#         import matplotlib.pyplot as plt
#
#         fig, (ax_map, ax_line) = plt.subplots(1, 2, figsize=(12, 5))
#
#         # Plot the density map (use .magnitude for the numerical values)
#         im = ax_map.imshow(density_map.magnitude, cmap='jet', origin='lower')
#         ax_map.set_title("Density Map")
#         plt.colorbar(im, ax=ax_map, label=f"Density ({(density_map.units)})")
#
#         # Plot the vertical lineout (again, extract .magnitude)
#         ax_line.plot(vertical_lineout.magnitude)
#         ax_line.set_title("Vertical Lineout")
#         ax_line.set_xlabel("Pixel Position")
#         ax_line.set_ylabel(f"Density ({(vertical_lineout.units)}")
#
#         plt.tight_layout()
#         plt.show()
#
if __name__ == '__main__':
    bkg_path: Path = Path('/Volumes/hdna2/data/Undulator/Y2025/03-Mar/25_0306/scans/Scan015/U_HasoLift/average_phase.tsv')
    phase_file_path=Path('/Volumes/hdna2/data/Undulator/Y2025/03-Mar/25_0306/scans/Scan016/U_HasoLift/Scan016_U_HasoLift_002_postprocessed.tsv')

    config: PhaseAnalysisConfig = PhaseAnalysisConfig(
        pixel_scale=10.1,            # um per pixel (vertical)
        wavelength_nm=800,           # Probe laser wavelength in nm
        threshold_fraction=0.2,      # Threshold fraction for pre-processing
        roi=(10, -10, 10, -100),      # Example ROI: (x_min, x_max, y_min, y_max)
        background_path=bkg_path,  # Background is now a Path
    )
    from dataclasses import asdict
    config_dict = asdict(config)
    print(phase_file_path)
    analyzer: PhaseDownrampProcessor = PhaseDownrampProcessor(**config_dict)
    analyzer.use_interactive = True
    analyzer.analyze_image_file(phase_file_path)

    # # --- Using the PyAbel inversion technique ---
    # pyabel_lineout, pyabel_density = analyzer.get_density(technique='pyabel')
    # analyzer.plot_density(pyabel_density, pyabel_lineout)