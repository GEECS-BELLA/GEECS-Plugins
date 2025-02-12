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
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import abel  # pip install pyabel
import abc
from typing import Optional, Tuple, List
from dataclasses import dataclass


# =============================================================================
# Configuration Dataclass
# =============================================================================
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
    background: Optional[np.ndarray] = None


# =============================================================================
# Phase Data Loader
# =============================================================================
class PhaseDataLoader:
    """
    Loads phase data from a file.
    If the file extension is '.tsv', the file is read as tab‐separated values.
    Otherwise, it is assumed to be an image file.
    """
    def __init__(self, file_path: str) -> None:
        self.file_path: Path = Path(file_path)

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


def threshold_data(data: np.ndarray, threshold_frac: float) -> np.ndarray:
    """
    Apply a simple threshold to a 2D array.
    Pixels with values below:
         data_min + threshold_frac*(data_max - data_min)
    are set to zero.
    """
    data_min: float = np.min(data)
    data_max: float = np.max(data)
    thresh_val: float = data_min + (data_max - data_min) * threshold_frac
    thresh: np.ndarray = np.copy(data)
    thresh[thresh < thresh_val] = 0
    return thresh


# =============================================================================
# Phase Preprocessor
# =============================================================================
class PhasePreprocessor:
    """
    Handles cropping and background subtraction for phase data.
    """
    def __init__(self, phase_array: np.ndarray) -> None:
        self.phase_array: np.ndarray = phase_array.copy()

    def subtract_background(self, background: np.ndarray) -> None:
        """
        Subtract a background array from the phase array.
        The background array must be broadcastable to the shape of the phase array.
        """
        self.phase_array = self.phase_array - background

    def subtract_constant(self, constant: float) -> None:
        """
        Subtract a constant background value from the phase array.
        """
        self.phase_array = self.phase_array - constant

    def crop(self, x_min: int, x_max: int, y_min: int, y_max: int) -> None:
        """
        Crop the phase array to the specified region of interest (ROI).
        """
        self.phase_array = self.phase_array[y_min:y_max, x_min:x_max]

    def get_processed_data(self) -> np.ndarray:
        return self.phase_array


# =============================================================================
# Inversion Technique Base Class
# =============================================================================
class InversionTechnique(abc.ABC):
    """
    Abstract base class for an inversion technique.
    """
    def __init__(self, phase_data: np.ndarray, pixel_scale: float, wavelength_nm: float) -> None:
        self.phase_data: np.ndarray = phase_data
        self.pixel_scale: float = pixel_scale  # µm per pixel (vertical)
        self.wavelength_nm: float = wavelength_nm

    @abc.abstractmethod
    def invert(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the inversion.
        Should return a tuple: (vertical_lineout, density_map)
        """
        pass


# =============================================================================
# Custom Abel Inversion (Manual Routines)
# =============================================================================
class CustomAbelInversion(InversionTechnique):
    def __init__(self, phase_data: np.ndarray, pixel_scale: float, wavelength_nm: float) -> None:
        super().__init__(phase_data, pixel_scale, wavelength_nm)

    def _center_of_mass(self, vector: np.ndarray) -> int:
        x: np.ndarray = np.arange(1, len(vector) + 1)
        mass: float = np.trapz(vector, x)
        moment: float = np.trapz(x * vector, x)
        return int(np.rint(moment / mass - 1))

    def _find_vertical_center(self, data: np.ndarray) -> int:
        vertical_profile: np.ndarray = np.sum(data, axis=1)
        return self._center_of_mass(vertical_profile)

    def _symmetrize(self, data: np.ndarray, center_row: int) -> np.ndarray:
        n_rows, _ = data.shape
        half: np.ndarray = data[center_row, :][np.newaxis, :]
        offset: int = 1
        while (center_row - offset) >= 0 and (center_row + offset) < n_rows:
            top: np.ndarray = data[center_row - offset, :]
            bottom: np.ndarray = data[center_row + offset, :]
            avg: np.ndarray = np.mean(np.stack((top, bottom)), axis=0)
            half = np.vstack((half, avg))
            offset += 1
        return half

    def _reconstruct_full(self, half_array: np.ndarray) -> np.ndarray:
        n_rows, n_cols = half_array.shape
        full: np.ndarray = np.zeros((2 * n_rows - 1, n_cols))
        full[:n_rows, :] = half_array
        full[n_rows:, :] = np.flipud(half_array[:-1, :])
        return full

    def _central_derivative(self, vector: np.ndarray, dt: float) -> np.ndarray:
        n: int = len(vector)
        deriv: List[float] = []
        for i in range(n):
            if i == 0:
                d = (vector[i + 1] - vector[i]) / dt
            elif i == n - 1:
                d = (vector[i] - vector[i - 1]) / dt
            else:
                d = (vector[i + 1] - vector[i - 1]) / (2 * dt)
            deriv.append(d)
        return np.array(deriv)

    def _invert_line(self, phase_line: np.ndarray, dt: float) -> np.ndarray:
        n_points: int = len(phase_line)
        y: np.ndarray = np.linspace(0, (n_points - 1) * dt, n_points)
        dphase_dy: np.ndarray = self._central_derivative(phase_line, dt)
        inverted: List[float] = []
        for i, r in enumerate(y):
            sub_y: np.ndarray = y[i:]
            sub_dphase: np.ndarray = dphase_dy[i:]
            denom: np.ndarray = np.sqrt(np.maximum(sub_y**2 - r**2, 1e-12))
            integrand: np.ndarray = sub_dphase / denom
            integral: float = np.trapz(integrand, dx=dt)
            inverted.append(-integral / math.pi)
        return np.array(inverted)

    def _wavelength_to_angular_freq(self, wavelength_nm: float) -> float:
        c_nm_fs: float = 299792.458  # nm/fs
        return 2 * math.pi * c_nm_fs / wavelength_nm

    def _angular_freq_to_critical_density(self, omega: float) -> float:
        c: float = 299792458  # m/s
        epsilon0: float = 1 / (c**2 * 4 * math.pi * 1e-7)
        m_e: float = 9.10938215e-31  # kg
        e: float = 1.602176487e-19   # C
        scaling: float = (epsilon0 * m_e) / (e**2) * 1e24 * 1e-18
        return omega**2 * scaling

    def _compute_density(self, inverted_map: np.ndarray, wavelength_nm: float) -> np.ndarray:
        speed_factor: float = 0.299792458  # µm/fs
        omega: float = self._wavelength_to_angular_freq(wavelength_nm)
        m_scaled: np.ndarray = inverted_map * (self.pixel_scale / (omega * 1) * speed_factor)
        density: np.ndarray = (1 - (1 + m_scaled) ** 2) * self._angular_freq_to_critical_density(omega)
        return density

    def invert(self) -> Tuple[np.ndarray, np.ndarray]:
        center_row: int = self._find_vertical_center(self.phase_data)
        half_phase: np.ndarray = self._symmetrize(self.phase_data, center_row)
        n_half, n_cols = half_phase.shape
        inverted_half: np.ndarray = np.zeros_like(half_phase)
        for col in range(n_cols):
            inverted_half[:, col] = self._invert_line(half_phase[:, col], self.pixel_scale)
        full_inverted: np.ndarray = self._reconstruct_full(np.flipud(inverted_half))
        density_map: np.ndarray = self._compute_density(full_inverted, self.wavelength_nm)
        center_col: int = density_map.shape[1] // 2
        vertical_lineout: np.ndarray = density_map[:, center_col]
        return vertical_lineout, density_map


# =============================================================================
# PyAbel Inversion
# =============================================================================
class PyAbelInversion(InversionTechnique):
    def __init__(self, phase_data: np.ndarray, pixel_scale: float, wavelength_nm: float, origin: Optional[Tuple[float, float]] = None, inversion_method: str = 'hansenlaw') -> None:
        super().__init__(phase_data, pixel_scale, wavelength_nm)
        self.origin: Optional[Tuple[float, float]] = origin
        self.inversion_method: str = inversion_method

    def invert(self) -> Tuple[np.ndarray, np.ndarray]:
        transformed: np.ndarray = abel.Transform(
            np.transpose(self.phase_data),
            direction='inverse',
            method=self.inversion_method,
            origin=self.origin if self.origin is not None else abel.tools.center.find_center(self.phase_data, center='convolution'),
            symmetry_axis=0,
            symmetrize_method='fourier'
        ).transform
        inverted: np.ndarray = np.transpose(transformed)
        scale_factor: float = self.pixel_scale * (self.wavelength_nm * 1e-9) * 2.817e-15
        density_map: np.ndarray = inverted / scale_factor
        center_col: int = density_map.shape[1] // 2
        vertical_lineout: np.ndarray = density_map[:, center_col]
        return vertical_lineout, density_map


# =============================================================================
# Main Density Analysis Class
# =============================================================================
class DensityAnalysis:
    """
    Main class for performing density analysis.
    
    This class loads and pre-processes phase data and then uses a specified
    inversion technique (custom or PyAbel) to compute a plasma density map.
    """
    def __init__(self, phase_file: str, config: PhaseAnalysisConfig) -> None:
        """
        Parameters:
          phase_file: str
              Path to the phase data file (TSV or image).
          config: PhaseAnalysisConfig
              Configuration parameters (pixel scale, wavelength, threshold fraction, ROI, background).
        """
        self.phase_file: str = phase_file
        self.config: PhaseAnalysisConfig = config

        loader: PhaseDataLoader = PhaseDataLoader(phase_file)
        raw_phase: np.ndarray = loader.load_data()

        # Pre-process: subtract background first, then crop if specified.
        preprocessor: PhasePreprocessor = PhasePreprocessor(raw_phase)
        if config.background is not None:
            preprocessor.subtract_background(config.background)
        if config.roi is not None:
            x_min, x_max, y_min, y_max = config.roi
            preprocessor.crop(x_min, x_max, y_min, y_max)
        self.phase_data: np.ndarray = preprocessor.get_processed_data()
        self.thresh_data: np.ndarray = threshold_data(self.phase_data, config.threshold_fraction)

    def get_density(self, technique: str = 'custom', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the density map and vertical lineout using a chosen inversion technique.
        
        Parameters:
          technique: str
              'custom' for the manual inversion or 'pyabel' for the PyAbel method.
          kwargs: additional keyword arguments passed to the inversion class.
        
        Returns:
          (vertical_lineout, density_map)
        """
        if technique == 'custom':
            inverter: InversionTechnique = CustomAbelInversion(self.phase_data, self.config.pixel_scale, self.config.wavelength_nm)
        elif technique == 'pyabel':
            inverter = PyAbelInversion(self.phase_data, self.config.pixel_scale, self.config.wavelength_nm, **kwargs)
        else:
            raise ValueError("Technique must be 'custom' or 'pyabel'.")
        return inverter.invert()

    def plot_density(self, density_map: np.ndarray, vertical_lineout: np.ndarray) -> None:
        """
        Plot the 2D density map and the vertical lineout.
        """
        fig, (ax_map, ax_line) = plt.subplots(1, 2, figsize=(12, 5))
        im = ax_map.imshow(density_map, cmap='jet', origin='lower')
        ax_map.set_title("Density Map")
        plt.colorbar(im, ax=ax_map, label="Density (arb. units)")
        ax_line.plot(vertical_lineout, np.arange(len(vertical_lineout)))
        ax_line.invert_yaxis()  # Adjust if vertical axis increases downward
        ax_line.set_title("Vertical Lineout")
        ax_line.set_xlabel("Density (arb. units)")
        ax_line.set_ylabel("Vertical Pixel")
        plt.tight_layout()
        plt.show()


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == '__main__':
    phase_file_path: str = 'path/to/your/phase_data.tsv'  # or image file
    config: PhaseAnalysisConfig = PhaseAnalysisConfig(
        pixel_scale=4.64,            # µm per pixel (vertical)
        wavelength_nm=800,           # Probe laser wavelength in nm
        threshold_fraction=0.5,      # Threshold fraction for pre-processing
        roi=(50, 300, 30, 250),      # Example ROI: (x_min, x_max, y_min, y_max)
        background=None             # Replace with a background array if available
    )

    analyzer: DensityAnalysis = DensityAnalysis(phase_file_path, config)

    # --- Using the custom inversion technique ---
    custom_lineout, custom_density = analyzer.get_density(technique='custom')
    analyzer.plot_density(custom_density, custom_lineout)

    # --- Using the PyAbel inversion technique ---
    pyabel_lineout, pyabel_density = analyzer.get_density(technique='pyabel', origin=None, inversion_method='hansenlaw')
    analyzer.plot_density(pyabel_density, pyabel_lineout)
