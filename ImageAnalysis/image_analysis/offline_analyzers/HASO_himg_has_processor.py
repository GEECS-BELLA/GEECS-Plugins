from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, NamedTuple
from pathlib import Path

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from image_analysis.types import Array2D

try:
    import image_analysis.third_party_sdks.wavekit_43.wavekit_py as wkpy
except ModuleNotFoundError as e:
    errmsg = "could not import wkpy, e.g. might be running on non windows machine"
    e.args += (errmsg,)
    raise

import logging
import numpy as np

from image_analysis.base import ImageAnalyzer

class SlopesMask(NamedTuple):
    top: Optional[int]
    bottom: Optional[int]
    left: Optional[int]
    right: Optional[int]

from dataclasses import dataclass

@dataclass
class FilterParameters:
    apply_tiltx_filter: bool = True
    apply_tilty_filter: bool = True
    apply_curv_filter: bool = True
    apply_astig0_filter: bool = True
    apply_astig45_filter: bool = True
    apply_others_filter: bool = False

@dataclass
class HasoHimgHasConfig:
    """
    Configuration parameters for HasoHimg processor.

    Attributes:
        mask: SlopesMask
        background_path: Path
          path to a slopes file for bkg subtraction.
        laser_wavelength: float
          Probe laser wavelength in nanometers.
    """
    mask: SlopesMask = SlopesMask(top=1, bottom=-1, left=1, right=-1)
    background_path: Path = None
    laser_wavelength: float = 800  # in nanometer

    # global path to the wavekit config file for the specific serial number of HASO
    wakekit_config_file_path: Path = Path('C:/GEECS/Developers Version/source/GEECS-Plugins/ImageAnalysis/image_analysis/third_party_sdks/wavekit_43/WFS_HASO4_LIFT_680_8244_gain_enabled.dat')

class HASOHimgHasProcessor(ImageAnalyzer):

    # Default filter parameters as a class attribute
    default_filter_params = FilterParameters()

    def __init__(self, config: HasoHimgHasConfig = HasoHimgHasConfig()):
        """
        Parameters
        ----------
        config : HasoHimgHasConfig
            configuration for processing the .himg and .has files. contains mask, bkg and laser wavelength
            information
        """

        self.mask = config.mask

        # for loading backgrounds on the fly. 
        self.background_path = config.background_path

        self.laser_wavelength = config.laser_wavelength

        # Use default filter parameters from class attribute
        self.filter_params = HASOHimgHasProcessor.default_filter_params

        self.flag_logging = True

        self.raw_slopes: wkpy.HasoSlopes = None
        self.processed_slopes: wkpy.HasoSlopes = None

        super().__init__(config = config)

        self.wakekit_config_file_path = config.wakekit_config_file_path

        # self.instantiate_wavekit_resources(config_file_path=self.wakekit_config_file_path)

        self.run_analyze_image_asynchronously = False

        self.image_file_path: Path = None

        self.wavekit_resources_instantiated = False

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

    def load_image(self, file_path:Path) -> Array2D:
        """
         Create phase map from a .himg or .has file.

         Parameters:
             file_path: Path to the image file.

         Returns:
             image: Array2D.

         Raises:
             ValueError: If the file type is not supported.
         """
        if not self.wavekit_resources_instantiated:
            self.instantiate_wavekit_resources(config_file_path = self.wakekit_config_file_path)
            self.wavekit_resources_instantiated = True

        self.image_file_path = file_path
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

        # image: Array2D = Array2D(processed_phase)
        # for some reason the above throws an error that Array2D is undefined...
        image = processed_phase

        return image

    def analyze_image(self, image: Array2D, auxiliary_data: Optional[dict] = None) -> dict[str, Union[float, int, str, np.ndarray]]:
        """
        Create phase map from a .himg or .has file.

        Parameters:
            image (NDArray): None. This part of the signature for the base class, but this analyzer
                requires loading of the image and processing with a third party SDK

        Returns:
            A dictionary containing results (e.g., phase map and/or related parameters).

        Raises:
            ValueError: If the file type is not supported.
        """

        return  self.build_return_dictionary(return_lineouts=image)

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
        self.apply_mask()
        self.apply_filter_wrapper(self.filter_params)

    def apply_mask(self):
        """
        mask a hasoslopes object with a rectangular mask.
        The pupil objects are not well documented in the SDK manual
        so it's not straightforward to generate an arbitray pupil and apply it.
        Here, we read the pupil buffer from the existing hasoslopes object,
        which is a 2D array of bools. We set everythign to false, then use
        get_mask to set a specificed rectangular mask to true. This then gets
        applied to the slopes using the sdk method apply_pupil
        """

        self.pupil = wkpy.Pupil(hasoslopes=self.processed_slopes)
        pupil_buffer = self.pupil.get_data()
        new_mask = self.get_mask(pupil_buffer)
        self.pupil.set_data(datas=new_mask)
        self.processed_slopes = self.post_processor.apply_pupil(self.processed_slopes, self.pupil)

    def get_mask(self, bool_array):
        """
        Takes a 2D boolean NumPy array, resets it to False, and applies a mask
        where values are set to True within the given x and y bounds.

        Parameters:
        - bool_array (np.ndarray): Input 2D boolean array.


        Returns:
        - np.ndarray: Updated boolean array with the applied mask.
        """

        # Ensure input is a NumPy array
        bool_array = np.asarray(bool_array, dtype=bool)

        # Get the size of the array
        rows, cols = bool_array.shape

        # Reset all values to False
        bool_array.fill(False)

        # Apply the mask, ensuring indices are within bounds
        x_start, x_end = max(0, self.mask.left), min(cols, self.mask.right)
        y_start, y_end = max(0, self.mask.top), min(rows, self.mask.bottom)

        bool_array[y_start:y_end, x_start:x_end] = True

        return bool_array

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

    def compute_phase_from_slopes(self, slopes_data: wkpy.HasoSlopes) -> NDArray:
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

        base_file_path = self.image_file_path.parent
        file_stem = self.image_file_path.stem
        logging.info(f'base file path is {base_file_path}')
        # Unpack the returned tuple.
        raw_slopes, processed_slopes, raw_phase, processed_phase, intensity = result

        self.slopes_file_path_raw = base_file_path / f"{file_stem}_raw.has"
        self.slopes_file_path_postprocessed = base_file_path / f"{file_stem}_postprocessed.has"

        self.save_slopes_file(slopes_data=raw_slopes, save_path=self.slopes_file_path_raw)
        self.save_slopes_file(slopes_data=processed_slopes, save_path=self.slopes_file_path_postprocessed)

        self.raw_phase_file_path = base_file_path / f"{file_stem}_raw.tsv"
        self.processed_phase_file_path = base_file_path/ f"{file_stem}_postprocessed.tsv"
        self.intensity_file_path = base_file_path / f"{file_stem}_intensity.tsv"

        self.save_phase_file(phase_values=raw_phase, save_path=self.raw_phase_file_path)
        self.save_phase_file(phase_values=processed_phase, save_path=self.processed_phase_file_path)
        self.save_phase_file(phase_values=intensity, save_path=self.intensity_file_path)

    def save_slopes_file(self, slopes_data: wkpy.HasoSlopes, save_path: Path):
        slopes_data.save_to_file(str(save_path), '', '')

    def save_phase_file(self, phase_values: NDArray, save_path: Path) -> None:

        # Convert phase_values to a numpy array (if it's a scalar, it'll become a 0-d array).
        arr = np.array(phase_values)

        # If you want to ensure it is at least 1D, you can do:
        if arr.ndim == 0:
            arr = np.array([[arr]])

        # Save the array to the specified path using tab-delimited format.
        np.savetxt(save_path, arr, delimiter="\t", fmt="%s")

if __name__ == "__main__":
    has  = HASOHimgHasProcessor()
    path_to_himg = Path('Z:/data/Undulator/Y2025/02-Feb/25_0219/scans/Scan002/U_HasoLift/Scan002_U_HasoLift_001.himg')
    # path_to_himg = Path('Z:/data/Undulator/Y2025/03-Mar/25_0306/scans/Scan055/U_HasoLift/Scan055_U_HasoLift_061.himg')
    path_to_has = Path('Z:/data/Undulator/Y2025/02-Feb/25_0219/scans/Scan002/U_HasoLift/Scan002_U_HasoLift_001_raw.has')

    mask = SlopesMask(top=75, bottom=246, left=10, right=670)
    analysis_config = HasoHimgHasConfig()
    analysis_config.mask = mask
    # haso_processor = HASOHimgHasProcessor(config = analysis_config)
    haso_processor = HASOHimgHasProcessor()

    haso_processor.analyze_image_file(image_filepath = path_to_himg)