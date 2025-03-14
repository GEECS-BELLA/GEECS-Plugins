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
from image_analysis.offline_analyzers.basic_image_analysis import BasicImageAnalyzer

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
    background: Optional[Path] = None

def threshold_data(data: np.ndarray, threshold_frac: float) -> np.ndarray:
    """
    Apply a simple threshold to a 2D array.
    Pixels with values below:
         data_min + threshold_frac*(data_max - data_min)
    are set to zero.
    NaNs are ignored in computing the threshold and remain unchanged.
    """
    data_min: float = np.nanmin(data)
    data_max: float = np.nanmax(data)
    thresh_val: float = data_min + (data_max - data_min) * threshold_frac
    thresh: np.ndarray = np.copy(data)
    # Create a mask that only selects non-NaN values below the threshold.
    mask = (thresh < thresh_val) & ~np.isnan(thresh)
    thresh[mask] = 0

    return thresh

class PhaseDataLoader:
    """
    Loads phase data from a file.
    If the file extension is '.tsv', the file is read as tab‐separated values.
    Otherwise, it is assumed to be an image file.
    """
    def __init__(self, file_path: Path) -> None:
        self.file_path: Path = file_path

    def load_data(self) -> np.ndarray:
        ext: str = self.file_path.suffix.lower()
        if ext == '.tsv':
            data: np.ndarray = np.genfromtxt(self.file_path, delimiter='\t')
            return data.astype(np.float64)
        else:
            image: Optional[np.ndarray] = cv2.imread(str(self.file_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load file: {self.file_path}")
            info = np.iinfo(image.dtype)
            return image.astype(np.float64) / info.max

class PhasePreprocessor:
    """
    Handles cropping and background subtraction for phase data.
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
        return phase_data - background

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

    # ---------------- Main Rotation Workflow ----------------

    def rotate_phase_data(self, phase_data: NDArray, threshold_frac: float = 0.2) -> Tuple[NDArray,dict]:
        """
        Rotate the phase data to correct for any tilt based on the weighted column centers.

        This workflow assumes that the phase data in self.phase_array has already been
        background-subtracted and cropped as desired.

        Steps:
          1. Compute the dynamic range (min, max) of the data.
          2. Create a thresholded copy for centroid computation.
          3. Compute weighted column centroids and column sums.
          4. Fit a weighted linear model (centroid vs. column index) to obtain the slope.
          5. Compute the rotation angle (in radians and degrees) from the slope.
          6. Rotate the original phase data using scipy.ndimage.rotate with reshape=True.
          7. Clip the rotated data back to the original range.

        Parameters:
            threshold_frac: Fraction of the dynamic range to use as a threshold.

        Returns:
            final_data: The rotated phase data (2D array).
            fit_params: Dictionary with keys 'slope', 'intercept', 'angle_radians', 'angle_degrees'.
        """
        phase_array = phase_data
        # Step 1: Determine dynamic range.
        data_min, data_max = np.nanmin(phase_array), np.nanmax(phase_array)
        phase_array = -(phase_array - data_min)
        phase_array = np.abs(phase_array - np.nanmin(phase_array))

        # Step 2: Create a thresholded version for centroid computation.
        thresh_data = threshold_data(phase_array, threshold_frac)

        # Step 3: Compute weighted column centroids and column sums.
        centroids, col_sums = self.compute_column_centroids(thresh_data)

        # Step 4: Fit a weighted line to the centroids.
        slope, intercept = self.fit_line_to_centroids(centroids, col_sums)

        # Step 5: Compute rotation angle.
        angle_rad = self.compute_rotation_angle(slope)
        angle_deg = np.degrees(angle_rad)
        fit_params = {
            'slope': slope,
            'intercept': intercept,
            'angle_radians': angle_rad,
            'angle_degrees': angle_deg
        }

        # Step 6: Rotate the original phase data.
        # Using reshape=True so that the output shape is adjusted to contain the entire rotated image.
        rotated_data = ndimage.rotate(phase_array, angle=angle_deg, reshape=True, order=1)

        phase_array = self.crop(phase_array,20, -20, 35, -35)

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

    def remove_background_polyfit(self, phase_data:NDArray, threshold_frac: float = 0.15, poly_order: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        After rotating the phase array, mask out the region that is usually kept by the threshold
        (i.e. the high-signal area) by replacing those pixel values with NaN. Then perform a 2D
        polynomial fit (of arbitrary order) to the remaining background data and subtract the fitted
        background from the full data.

        Parameters:
            threshold_frac: Fraction of the dynamic range used to determine the threshold.
            poly_order: The maximum order of the polynomial to fit.

        Returns:
            background_fit: The 2D background (from the polynomial fit).
            coeffs: The polynomial coefficients as a 1D array.
        """
        data = phase_data
        # Compute the dynamic range of the data.
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        data = -np.abs((data - data_min))
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        thresh_val = data_min + (data_max - data_min) * threshold_frac
        # Create a masked copy: set pixels above the threshold to NaN.
        masked_data = data.copy()
        masked_data[masked_data > thresh_val] = np.nan

        # plt.imshow(masked_data, cmap='viridis', origin='lower')
        # plt.title("maske phase")
        # plt.colorbar()
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

        # plt.imshow(data, cmap='viridis', origin='lower')
        # plt.title("maske phase")
        # plt.colorbar()
        # plt.show()

        return data, background_fit, coeffs

class PhaseDownrampProcessor(BasicImageAnalyzer):
    """
    Handles cropping and background subtraction for phase data.
    """

    def __init__(self, config: PhaseAnalysisConfig, debug_mode:bool = False) -> None:
        self.config = config

        self.processor  = PhasePreprocessor(config)

        # If background is provided, treat it as a Path.
        # Assumption here is that background is loaded only at
        # initialization.
        if self.config.background is not None:
            bg_loader = PhaseDataLoader(self.config.background)
            self.bg_data = bg_loader.load_data()

        super().__init__()

    def process_phase(self, file_path: Path) -> Tuple[NDArray,dict[str,float]]:
        """
        Parameters: file_path
        """
        loader: PhaseDataLoader = PhaseDataLoader(file_path)
        phase_array = loader.load_data()
        plt.imshow(phase_array, cmap='viridis', origin='lower')
        plt.title("first phase")
        plt.show()

        phase_array = self.processor.subtract_background(phase_array, self.bg_data)

        plt.imshow(phase_array, cmap='viridis', origin='lower')
        plt.title("bkg subtracted phase")
        plt.show()
        if self.config.roi is not None:
            x_min, x_max, y_min, y_max = self.config.roi
            phase_array = self.processor.crop(phase_array, x_min, x_max, y_min, y_max)

        # plt.imshow(phase_array, cmap='viridis', origin='lower')
        # plt.title("processed phase")
        # plt.show()

        rotation_result = self.processor.rotate_phase_data(phase_array)
        plt.imshow(rotation_result[0], cmap='viridis', origin='lower')
        plt.title("processed phase")
        plt.show()

        polynomial_subtraction_result = self.processor.remove_background_polyfit(phase_array)

        phase_array = polynomial_subtraction_result[0]
        thresh_data = threshold_data(phase_array, self.config.threshold_fraction)

        plt.imshow(thresh_data, cmap='viridis', origin='lower')
        plt.title("processed phase")
        plt.show()

        return phase_array, rotation_result[1]

    def analyze_image(self, image: NDArray = None, file_path: Path = None) -> dict[str, Union[float, int, str, np.ndarray]]:

        """
        Apply some HTU specific processing of a phase map showing a density down ramp feature.

        Parameters:
            image (NDArray): None. This part of the signature for the base class, but this analyzer
                requires loading of an already post processed file
            file_path (Path): Path to the image file.

        Returns:
            A dictionary containing results (e.g., phase map and/or related parameters).

        """

        try:
            processed_phase = self.process_phase(file_path)
            logging.info(f'processed {image}')

        except Exception as e:
            logging.warning(f'could not process {image}')
            raise

        # self.downramp_phase_analysis(processed_phase[1])

        plt.imshow(processed_phase[0], cmap='viridis', origin='lower')
        plt.title("processed phase")
        plt.show()

        # Append the flattened result dictionary to your list
        return_dictionary = self.build_return_dictionary(return_image=processed_phase[0], return_scalars=processed_phase[1])
        return return_dictionary

    def downramp_phase_analysis(self, phase_array: NDArray):
        # Find the index of the maximum value in the array.
        max_index = np.unravel_index(np.argmax(phase_array), phase_array.shape)
        row, col = max_index
        logging.info("Max value at:", row, col)

        # Define a window of ±20 pixels around the max point, ensuring we don't go out of bounds.
        row_min = max(row - 40, 0)
        row_max = min(row + 40, phase_array.shape[0])
        col_min = max(col - 40, 0)
        col_max = min(col + 40, phase_array.shape[1])

        # Crop the region.
        cropped_region = phase_array[row_min:row_max, col_min:col_max]
        # Display the cropped region.
        plt.imshow(cropped_region, cmap='viridis', origin='lower')
        plt.title("Cropped Region with Row-wise Max Markers")

        # For each row in the cropped region, overlay a black marker at the max value.
        for i in range(cropped_region.shape[0]):
            # Get the column index for the maximum value in this row.
            max_col_index = np.argmax(cropped_region[i, :])
            # Plot the marker. Note: x corresponds to column, y to row.
            plt.plot(max_col_index, i, marker='o', markersize=5, color='black')

        plt.show()

        return

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
# if __name__ == '__main__':
#     phase_file_path: Path = Path('../Scan003_U_HasoLift_001_postprocessed.tsv')  # or an image file path
#
#     config: PhaseAnalysisConfig = PhaseAnalysisConfig(
#         pixel_scale=10.1,            # um per pixel (vertical)
#         wavelength_nm=800,           # Probe laser wavelength in nm
#         threshold_fraction=0.2,      # Threshold fraction for pre-processing
#         roi=(10, -10, 10, -100),      # Example ROI: (x_min, x_max, y_min, y_max)
#         background=Path('../average_phase.tsv')  # Background is now a Path
#
#     )
#
#     analyzer: DensityAnalysis = DensityAnalysis(phase_file_path, config)
#
#     # --- Using the PyAbel inversion technique ---
#     pyabel_lineout, pyabel_density = analyzer.get_density(technique='pyabel')
#     analyzer.plot_density(pyabel_density, pyabel_lineout)

if __name__ == '__main__':

    from geecs_python_api.analysis.scans.scan_data import ScanData, ScanTag
    from image_analysis.analyzers.density_from_phase_analysis import PhaseAnalysisConfig, PhasePreprocessor, \
        PhaseDownrampProcessor


    def get_path_to_phase_file():
        st = ScanTag(2025, 3, 6, 16, experiment='Undulator')
        s_data = ScanData(tag=st)
        path_to_file = Path(
            s_data.get_analysis_folder() / "U_HasoLift" / "HasoAnalysis" / 'Scan016_U_HasoLift_002_postprocessed.tsv')
        return path_to_file


    def get_path_to_bkg_file():
        st = ScanTag(2025, 3, 6, 15, experiment='Undulator')
        s_data = ScanData(tag=st)
        path_to_file = Path(s_data.get_analysis_folder() / "U_HasoLift" / "HasoAnalysis" / 'average_phase2.tsv')
        return path_to_file


    def test_phase_processing():
        phase_file_path: Path = get_path_to_phase_file()
        bkg_file_path = get_path_to_bkg_file()

        config: PhaseAnalysisConfig = PhaseAnalysisConfig(
            pixel_scale=10.1,            # um per pixel (vertical)
            wavelength_nm=800,           # Probe laser wavelength in nm
            threshold_fraction=0.2,      # Threshold fraction for pre-processing
            roi=(10, -10, 10, -100),      # Example ROI: (x_min, x_max, y_min, y_max)
            background=Path('../average_phase.tsv')  # Background is now a Path
        )

        processor = PhaseDownrampProcessor(config)
        processor.analyze_image(phase_file_path)


    test_phase_processing()

