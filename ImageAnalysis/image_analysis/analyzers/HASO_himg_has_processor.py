from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
from pathlib import Path

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..types import Array2D, QuantityArray2D

try:
    import image_analysis.third_party_sdks.wavekit_43.wavekit_py as wkpy
except Exception as e:
    print("could not import wkpy, e.g. might be running on non windows machine")
    raise

import logging
import numpy as np

from image_analysis.analyzers.basic_image_analysis import BasicImageAnalyzer
from image_analysis.utils import ROI

from dataclasses import dataclass

@dataclass
class FilterParameters:
    apply_tiltx_filter: bool = False
    apply_tilty_filter: bool = False
    apply_curv_filter: bool = False
    apply_astig0_filter: bool = False
    apply_astig45_filter: bool = False
    apply_others_filter: bool = False

class HASOHimgHasProcessor(BasicImageAnalyzer):

    # Default filter parameters as a class attribute
    default_filter_params = FilterParameters()

    def __init__(self, 
                 roi: ROI = ROI(top=None, bottom=None, left=None, right=None),
                 medium: str = 'plasma',
                 background_path: Path = None,
                 on_no_background: str = 'warn',

                 laser_wavelength: float = 800, #in nanmeter
                ):
        """
        Parameters
        ----------
        roi : ROI
            Region of interest, as top, bottom (where top < bottom), left, right.
        medium : str
            One of 'plasma', 'gas/He', 'gas/N', for calculating density from Abel-
            inverted wavefront.
        background_path : Path
            A file or folder containing interferograms to use as background.
        on_no_background : str
            What to do if no background is set explicitly and no background path is
            given. 
                'raise': raise ValueError
                'warn': return wavefront with no background subtraction and issue warning
                'ignore': return wavefrtont with no background subtraction and don't
                          issue warning.

        laser_wavelength : [length] Quantity
            of imaging laser

        """

        self.roi = roi
        self.medium = medium
        self.background: Optional[QuantityArray2D] = None
        self.on_no_background: str = on_no_background

        # for loading backgrounds on the fly. 
        self.background_path: Path = background_path
        self.background_cache: dict[tuple[Path, ROI], Array2D] = {}

        self.laser_wavelength = laser_wavelength

        # Use default filter parameters from class attribute
        self.filter_params = HASOHimgHasProcessor.default_filter_params

        self.flag_logging = True

        self.raw_slopes: wkpy.HasoSlopes = None
        self.processed_slopes: wkpy.HasoSlopes = None

        super().__init__()

        config_file_path = Path('C:/GEECS/Developers Version/source/GEECS-Plugins/ImageAnalysis/image_analysis/third_party_sdks/wavekit_43/WFS_HASO4_LIFT_680_8244_gain_enabled.dat')

        self.instantiate_wavekit_resources(config_file_path=config_file_path)

    def _log_info(self, message: str, *args, **kwargs):
        """Log an info message if logging is enabled."""
        if self.flag_logging:
            logging.info(message, *args, **kwargs)

    def _log_warning(self, message: str, *args, **kwargs):
        """Log a warning message if logging is enabled."""
        if self.flag_logging:
            logging.warning(message, *args, **kwargs)

    def instantiate_wavekit_resources(self, config_file_path: Path):
        """
        attempt to instantiate necessary wavekit resources

        Args:
            config_file_path (Path): Path to the config file.

        """

        self._log_info(f"instantiating wavekit resources: HasoEngine etc.")

        try:
            # Create the necessary Wavekit objects.
            self.hasoengine = wkpy.HasoEngine(config_file_path=str(config_file_path))
            self.hasoengine.set_lift_enabled(True, self.laser_wavelength)
            self.hasoengine.set_lift_option(True, self.laser_wavelength)

            # Set preferences with an arbitrary subpupil and denoising strength.
            start_subpupil = wkpy.uint2D(87, 64)
            denoising_strength = 0.0
            self.hasoengine.set_preferences(start_subpupil, denoising_strength, False)

            self.compute_phase_set = wkpy.ComputePhaseSet(type_phase=wkpy.E_COMPUTEPHASESET.ZONAL)
            self.compute_phase_set.set_zonal_prefs(100, 500, 1e-6)

            self.post_processor = wkpy.SlopesPostProcessor()

        except Exception:
            self._log_warning(
                f"Not able to create necessary Wavekit objects, likely a result of Wavekit not being installed or missing/incorrect license file")
            raise

    def analyze_image(self, image: NDArray = None, file_path: Path = None) -> Tuple[wkpy.HasoSlopes, wkpy.HasoSlopes, NDArray, NDArray, NDArray]:
        """
        Create phase map from a .himg or .has file.

        Parameters:
            image (NDArray): None. This part of the signature for the base class, but this analyzer
                requires loading of the image and processing with a third party SDK
            file_path: Path to the image file.

        Returns:
            A dictionary containing results (e.g., phase map and/or related parameters).

        Raises:
            ValueError: If the file type is not supported.
        """
        self.file_path = file_path
        ext = file_path.suffix.lower()
        self._log_info(f'extension is {ext}')
        if ext == ".himg":
            self.raw_slopes = self.create_slopes_object_from_himg(file_path)
        elif ext == ".has":
            self.raw_slopes = self.load_slopes_from_has_file(file_path)
        else:
            msg = f"Unsupported file extension '{ext}'. Supported file types are .himg and .has."
            logging.error(msg)
            raise ValueError(msg)

        self.post_process_slopes()

        haso_intensity = wkpy.Intensity(hasoslopes=self.raw_slopes)
        haso_intensity = haso_intensity.get_data()[0]

        raw_phase = self.compute_phase_from_slopes(self.raw_slopes)
        processed_phase = self.compute_phase_from_slopes(self.processed_slopes)

        result = self.raw_slopes, self.processed_slopes, raw_phase, processed_phase, haso_intensity

        self.save_individual_results(result)

        return

    def create_slopes_object_from_himg(self, image_file_path: Path) -> wkpy.HasoSlopes:
        """
        Compute and save the slopes file (.has) from the provided image file.

        Args:
            image_file_path (Path): Path to the .himg file.

        Returns:
            Path: The path to the created slopes file (.has).
        """
        self._log_info(f"Creating slopes file for image: {image_file_path}")
        image_file_str = str(image_file_path)

        try:
            # Create the necessary Wavekit objects.
            image = wkpy.Image(image_file_path=image_file_str)
        except Exception:
            self._log_warning(
                "Not able to create necessary Wavekit objects, likely a result of Wavekit not being installed or missing/incorrect license file")
            raise

        # Compute slopes
        learn_from_trimmer = False
        _, hasoslopes = self.hasoengine.compute_slopes(image, learn_from_trimmer)

        return hasoslopes

    @staticmethod
    def load_slopes_from_has_file(slopes_file_path: Path) -> wkpy.HasoSlopes:

        hasoslopes = wkpy.HasoSlopes(has_file_path=str(slopes_file_path))
        return hasoslopes

    def post_process_slopes(self):
        self.processed_slopes = self.raw_slopes
        self.reference_subtract(self.background_path)
        self.apply_filter_wrapper(self.filter_params)

    def reference_subtract(self, background_path: Optional[Path] = None):
        if background_path:
            #TODO: add a check that the background path is to a .has file
            bkg_data = wkpy.HasoSlopes(has_file_path=str(background_path))
            self.processed_slopes = self.post_processor.apply_substractor(self.processed_slopes, bkg_data)
        else:
            self.processed_slopes = self.processed_slopes

    def apply_filter_wrapper(self, filter_params: FilterParameters):
        self.processed_slopes = self.post_processor.apply_filter(
            self.processed_slopes,
            filter_params.apply_tiltx_filter,
            filter_params.apply_tilty_filter,
            filter_params.apply_curv_filter,
            filter_params.apply_astig0_filter,
            filter_params.apply_astig45_filter,
            filter_params.apply_others_filter
        )

    def compute_phase_from_slopes(self, slopes_data: wkpy.HasoSlopes) -> float|NDArray:
        """
        Compute phase data from the provided slopes file (.has) and save the result as a TSV.

        Args:
            slopes_data (wkpy.HasoSlopes): slopes data

        Returns:
            DataFrame: The computed phase data.
        """

        hasodata = wkpy.HasoData(hasoslopes = slopes_data)
        phase = wkpy.Compute.phase_zonal(self.compute_phase_set, hasodata)
        phase_values = phase.get_data()[0]
        return phase_values

    def save_individual_results(self, result):
        # Unpack the returned tuple.
        base_file_path = self.file_path.parent
        raw_slopes, processed_slopes, raw_phase, processed_phase, intensity = result

        self.slopes_file_path_raw = self.path_dict['save'] / f"{base_file_path}_raw.has"
        self.slopes_file_path_postprocessed = self.path_dict['save'] / f"{base_file_path}_postprocessed.has"

        self.save_slopes_file(slopes_data=raw_slopes, save_path=self.slopes_file_path_raw)
        self.save_slopes_file(slopes_data=processed_slopes, save_path=self.slopes_file_path_postprocessed)

        self.raw_phase_file_path = self.path_dict['save'] / f"{base_file_path}_raw.tsv"
        self.processed_phase_file_path = self.path_dict['save'] / f"{base_file_path}_postprocessed.tsv"
        self.intensity_file_path = self.path_dict['save'] / f"{base_file_path}_intensity.tsv"

        self.save_phase_file(phase_values=raw_phase, save_path=self.raw_phase_file_path)
        self.save_phase_file(phase_values=processed_phase, save_path=self.processed_phase_file_path)
        self.save_phase_file(phase_values=intensity, save_path=self.intensity_file_path)

    def save_slopes_file(self, slopes_data: wkpy.HasoSlopes, save_path: Path):
        slopes_data.save_to_file(str(save_path), '', '')

    def save_phase_file(phase_values: float | NDArray, save_path: Path) -> None:

        # Convert phase_values to a numpy array (if it's a scalar, it'll become a 0-d array).
        arr = np.array(phase_values)

        # If you want to ensure it is at least 1D, you can do:
        if arr.ndim == 0:
            arr = np.array([[arr]])

        # Save the array to the specified path using tab-delimited format.
        np.savetxt(save_path, arr, delimiter="\t", fmt="%s")

if __name__ == "__main__":
    has  = HASOHimgHasProcessor()
    path_to_has = Path('Z:/data/Undulator/Y2025/02-Feb/25_0219/scans/Scan002/U_HasoLift/Scan002_U_HasoLift_001_raw.has')
    haso_processor = HASOHimgHasProcessor()
    print(haso_processor.roi)