# """
# Camera Image Analysis

# General camera image analyzer.
# Child to ScanAnalysis (./scan_analysis/base.py)
# """

# from __future__ import annotations
# from typing import TYPE_CHECKING, Optional
# if TYPE_CHECKING:
    # from geecs_python_api.controls.api_defs import ScanTag

# from pathlib import Path
# import logging
# import pandas as pd

# from scan_analysis.base import ScanAnalysis
# import scan_analysis.third_party_sdks.wavekit_43.wavekit_py as wkpy


# class HasoAnalysis(ScanAnalysis):
    # """
    # HasoAnalysis performs camera image analysis using a HASO device.
    # It processes images from a specified scan directory and computes slopes
    # and phase data using the Wavekit API.
    # """

    # def __init__(self, scan_tag: ScanTag, device_name: str,
                 # skip_plt_show: bool = True,
                 # flag_logging: bool = True,
                 # flag_save_images: bool = True):
        # """
        # Initialize the HasoAnalysis instance.

        # Args:
            # scan_tag (ScanTag): Path to the scan directory containing data.
            # device_name (str): Device name used to construct subdirectory paths.
            # skip_plt_show (bool): If True, skips plotting via matplotlib.
            # flag_logging (bool): If True, enables logging of warnings and errors.
            # flag_save_images (bool): If True, enables saving images to disk.
        # """
        # if not device_name:
            # raise ValueError("HasoAnalysis requires a device_name.")

        # super().__init__(scan_tag, device_name=device_name, skip_plt_show=skip_plt_show)

        # self.flag_logging = flag_logging
        # self.flag_save_images = flag_save_images

        # # Organize paths: one for image data and one for analysis output.
        # self.path_dict = {
            # 'data_img': Path(self.scan_directory) / device_name,
            # 'save': self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / device_name / "HasoAnalysis",
        # }

        # if self.flag_logging:
            # logging.info(f"Initialized HasoAnalysis for device '{device_name}' with scan directory '{self.scan_directory}'")

    # def run_analysis(self, config_options: Optional[str] = None) -> Optional:
        # """
        # Execute the analysis workflow.

        # Args:
            # config_options (Optional[str]): Configuration options file (not implemented).

        # Returns:
            # The display contents if analysis is successful, or None.
        # """
        # if self.flag_logging:
            # logging.info("Starting analysis workflow.")

        # if self.path_dict['data_img'] is None or self.auxiliary_data is None:
            # if self.flag_logging:
                # logging.info("Skipping analysis due to missing data or auxiliary file.")
            # return None

        # if config_options is not None:
            # # Future implementation: load configuration from the provided file.
            # raise NotImplementedError("Loading configuration options is not implemented.")

        # if self.flag_save_images:
            # if not self.path_dict['save'].exists():
                # self.path_dict['save'].mkdir(parents=True)
                # if self.flag_logging:
                    # logging.info(f"Created save directory: {self.path_dict['save']}")
            # else:
                # if self.flag_logging:
                    # logging.info(f"Using existing save directory: {self.path_dict['save']}")

        # try:
            # if self.noscan:
                # if self.flag_logging:
                    # logging.info("Running no-scan analysis.")
                # self.run_noscan_analysis()
            # else:
                # if self.flag_logging:
                    # logging.info("Running scan analysis.")
                # self.run_scan_analysis()
            # if self.flag_logging:
                # logging.info("Analysis workflow completed successfully.")
            # return self.display_contents
        # except Exception as e:
            # if self.flag_logging:
                # logging.warning(f"Image analysis failed due to: {e}")
            # return None

    # def run_noscan_analysis(self) -> None:
        # """
        # Perform analysis for a no-scan scenario.
        # Processes images based on shot numbers from auxiliary data.
        # """
        # for shot_num in self.auxiliary_data['Shotnumber'].values:
            # if self.flag_logging:
                # logging.info(f"Processing shot number: {shot_num}")
            # image_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}.himg'), None)
            # if image_file:
                # if self.flag_logging:
                    # logging.info(f"Found image file for shot {shot_num}: {image_file}")
                # self.get_phase_from_himg(image_file)
            # else:
                # if self.flag_logging:
                    # logging.warning(f"Missing data for shot {shot_num}.")

    # def run_scan_analysis(self) -> None:
        # """
        # Perform analysis for a scan scenario.
        # Currently, this method delegates to the no-scan analysis.
        # """
        # if self.flag_logging:
            # logging.info("Delegating scan analysis to no-scan analysis.")
        # self.run_noscan_analysis()

    # def create_slopes_file(self, image_file_path: Path) -> Path:
        # """
        # Compute and save the slopes file (.has) from the provided image file.

        # Args:
            # image_file_path (Path): Path to the .himg file.

        # Returns:
            # Path: The path to the created slopes file (.has).
        # """
        # if self.flag_logging:
            # logging.info(f"Creating slopes file for image: {image_file_path}")
        # config_file_path = 'scan_analysis/third_party_sdks/wavekit_43/WFS_HASO4_LIFT_680_8244_gain_enabled.dat'
        # image_file_str = str(image_file_path)

        # # Create the necessary Wavekit objects.
        # image = wkpy.Image(image_file_path=image_file_str)
        # hasoengine = wkpy.HasoEngine(config_file_path=config_file_path)
        # hasoengine.set_lift_enabled(True, 800)
        # hasoengine.set_lift_option(True, 800)

        # # Set preferences with an arbitrary subpupil and denoising strength.
        # start_subpupil = wkpy.uint2D(16, 16)
        # denoising_strength = 1.0
        # hasoengine.set_preferences(start_subpupil, denoising_strength, False)

        # base_name = image_file_path.stem
        # slopes_file_path = self.path_dict['save'] / f"{base_name}.has"

        # # Compute slopes and save the slopes file.
        # learn_from_trimmer = False
        # _, hasoslopes = hasoengine.compute_slopes(image, learn_from_trimmer)
        # hasoslopes.save_to_file(str(slopes_file_path), '', '')
        # if self.flag_logging:
            # logging.info(f"Slopes file saved: {slopes_file_path}")
        # return slopes_file_path

    # def compute_phase_from_slopes(self, slopes_file_path: Path) -> pd.DataFrame:
        # """
        # Compute phase data from the provided slopes file (.has) and save the result as a TSV.

        # Args:
            # slopes_file_path (Path): Path to the slopes (.has) file.

        # Returns:
            # DataFrame: The computed phase data.
        # """
        # if self.flag_logging:
            # logging.info(f"Computing phase data from slopes file: {slopes_file_path}")
        # base_name = slopes_file_path.stem
        # tsv_file_path = self.path_dict['save'] / f"{base_name}.tsv"

        # compute_phase_set = wkpy.ComputePhaseSet(type_phase=wkpy.E_COMPUTEPHASESET.ZONAL)
        # hasodata = wkpy.HasoData(has_file_path=str(slopes_file_path))
        # phase = wkpy.Compute.phase_zonal(compute_phase_set, hasodata)

        # # Save phase data to a TSV file.
        # phase_values = phase.get_data()[0]
        # df = pd.DataFrame(phase_values)
        # df.to_csv(tsv_file_path, sep="\t", index=False, header=False)
        # if self.flag_logging:
            # logging.info(f"Phase data saved to TSV file: {tsv_file_path}")
        # return df

    # def get_phase_from_himg(self, image_file_path: Path) -> pd.DataFrame:
        # """
        # Process a .himg file by computing the slopes file and then the phase data.

        # Args:
            # image_file_path (Path): Path to the .himg file.

        # Returns:
            # DataFrame: The computed phase data.
        # """
        # if self.flag_logging:
            # logging.info(f"Starting phase analysis for image file: {image_file_path}")
        # slopes_file = self.create_slopes_file(image_file_path)
        # df = self.compute_phase_from_slopes(slopes_file)
        # if self.flag_logging:
            # logging.info(f"Completed phase analysis for image file: {image_file_path}")
        # return df


# if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    # from geecs_python_api.analysis.scans.scan_data import ScanData
    # tag = ScanData.get_scan_tag(year=2025, month=2, day=11, number=11, experiment_name='Undulator')
    # analyzer = HasoAnalysis(scan_tag=tag, device_name="U_HasoLift", skip_plt_show=True)
    # analyzer.run_analysis()
    
    
"""
Camera Image Analysis

General camera image analyzer.
Child to ScanAnalysis (./scan_analysis/base.py)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag

from pathlib import Path
import logging
import pandas as pd

from scan_analysis.base import ScanAnalysis
import scan_analysis.third_party_sdks.wavekit_43.wavekit_py as wkpy


class HasoAnalysis(ScanAnalysis):
    """
    HasoAnalysis performs camera image analysis using a HASO device.
    It processes images from a specified scan directory and computes slopes
    and phase data using the Wavekit API.
    """

    def __init__(self, scan_tag: ScanTag, device_name: str,
                 skip_plt_show: bool = True,
                 flag_logging: bool = True,
                 flag_save_images: bool = True):
        """
        Initialize the HasoAnalysis instance.

        Args:
            scan_tag (ScanTag): Path to the scan directory containing data.
            device_name (str): Device name used to construct subdirectory paths.
            skip_plt_show (bool): If True, skips plotting via matplotlib.
            flag_logging (bool): If True, enables logging of warnings and errors.
            flag_save_images (bool): If True, enables saving images to disk.
        """
        if not device_name:
            raise ValueError("HasoAnalysis requires a device_name.")

        super().__init__(scan_tag, device_name=device_name, skip_plt_show=skip_plt_show)

        self.flag_logging = flag_logging
        self.flag_save_images = flag_save_images

        # Organize paths: one for image data and one for analysis output.
        self.path_dict = {
            'data_img': Path(self.scan_directory) / device_name,
            'save': self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / device_name / "HasoAnalysis",
        }

        self._log_info(f"Initialized HasoAnalysis for device '{device_name}' with scan directory '{self.scan_directory}'")

    def _log_info(self, message: str, *args, **kwargs):
        """Log an info message if logging is enabled."""
        if self.flag_logging:
            logging.info(message, *args, **kwargs)

    def _log_warning(self, message: str, *args, **kwargs):
        """Log a warning message if logging is enabled."""
        if self.flag_logging:
            logging.warning(message, *args, **kwargs)

    def run_analysis(self, config_options: Optional[str] = None) -> Optional:
        """
        Execute the analysis workflow.

        Args:
            config_options (Optional[str]): Configuration options file (not implemented).

        Returns:
            The display contents if analysis is successful, or None.
        """
        self._log_info("Starting analysis workflow.")

        if self.path_dict['data_img'] is None or self.auxiliary_data is None:
            self._log_info("Skipping analysis due to missing data or auxiliary file.")
            return None

        if config_options is not None:
            raise NotImplementedError("Loading configuration options is not implemented.")

        if self.flag_save_images:
            if not self.path_dict['save'].exists():
                self.path_dict['save'].mkdir(parents=True)
                self._log_info(f"Created save directory: {self.path_dict['save']}")
            else:
                self._log_info(f"Using existing save directory: {self.path_dict['save']}")

        try:
            if self.noscan:
                self._log_info("Running no-scan analysis.")
                self.run_noscan_analysis()
            else:
                self._log_info("Running scan analysis.")
                self.run_scan_analysis()
            self._log_info("Analysis workflow completed successfully.")
            return self.display_contents
        except Exception as e:
            self._log_warning(f"Image analysis failed due to: {e}")
            return None

    def run_noscan_analysis(self) -> None:
        """
        Perform analysis for a no-scan scenario.
        For each shot, first check for a .himg file; if it exists, process it.
        Otherwise, if a .has file exists, compute phase data directly.
        """
        for shot_num in self.auxiliary_data['Shotnumber'].values:
            self._log_info(f"Processing shot number: {shot_num}")

            # Try to locate the .himg file.
            himg_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}.himg'), None)
            if himg_file:
                self._log_info(f"Found .himg file for shot {shot_num}: {himg_file}")
                self.get_phase_from_himg(himg_file)
            else:
                # If no .himg file, check for a pre-existing .has file.
                has_file = next(self.path_dict['save'].glob(f'*_{shot_num:03d}.has'), None)
                if has_file:
                    self._log_info(f"No .himg file found for shot {shot_num}. Using existing slopes file: {has_file}")
                    self.compute_phase_from_slopes(has_file)
                else:
                    self._log_warning(f"Missing data for shot {shot_num}. Neither .himg nor .has file exists.")

    def run_scan_analysis(self) -> None:
        """
        Perform analysis for a scan scenario.
        Delegates to the no-scan analysis.
        """
        self._log_info("Delegating scan analysis to no-scan analysis.")
        self.run_noscan_analysis()

    def create_slopes_file(self, image_file_path: Path) -> Path:
        """
        Compute and save the slopes file (.has) from the provided image file.

        Args:
            image_file_path (Path): Path to the .himg file.

        Returns:
            Path: The path to the created slopes file (.has).
        """
        self._log_info(f"Creating slopes file for image: {image_file_path}")
        config_file_path = 'scan_analysis/third_party_sdks/wavekit_43/WFS_HASO4_LIFT_680_8244_gain_enabled.dat'
        image_file_str = str(image_file_path)

        # Create the necessary Wavekit objects.
        image = wkpy.Image(image_file_path=image_file_str)
        hasoengine = wkpy.HasoEngine(config_file_path=config_file_path)
        hasoengine.set_lift_enabled(True, 800)
        hasoengine.set_lift_option(True, 800)

        # Set preferences with an arbitrary subpupil and denoising strength.
        start_subpupil = wkpy.uint2D(16, 16)
        denoising_strength = 1.0
        hasoengine.set_preferences(start_subpupil, denoising_strength, False)

        base_name = image_file_path.stem
        slopes_file_path = self.path_dict['save'] / f"{base_name}.has"

        # Compute slopes and save the slopes file.
        learn_from_trimmer = False
        _, hasoslopes = hasoengine.compute_slopes(image, learn_from_trimmer)
        hasoslopes.save_to_file(str(slopes_file_path), '', '')
        self._log_info(f"Slopes file saved: {slopes_file_path}")
        return slopes_file_path

    def compute_phase_from_slopes(self, slopes_file_path: Path) -> pd.DataFrame:
        """
        Compute phase data from the provided slopes file (.has) and save the result as a TSV.

        Args:
            slopes_file_path (Path): Path to the slopes (.has) file.

        Returns:
            DataFrame: The computed phase data.
        """
        self._log_info(f"Computing phase data from slopes file: {slopes_file_path}")
        base_name = slopes_file_path.stem
        tsv_file_path = self.path_dict['save'] / f"{base_name}.tsv"

        compute_phase_set = wkpy.ComputePhaseSet(type_phase=wkpy.E_COMPUTEPHASESET.ZONAL)
        hasodata = wkpy.HasoData(has_file_path=str(slopes_file_path))
        phase = wkpy.Compute.phase_zonal(compute_phase_set, hasodata)

        phase_values = phase.get_data()[0]
        df = pd.DataFrame(phase_values)
        df.to_csv(tsv_file_path, sep="\t", index=False, header=False)
        self._log_info(f"Phase data saved to TSV file: {tsv_file_path}")
        return df

    def get_phase_from_himg(self, image_file_path: Path) -> pd.DataFrame:
        """
        Process a .himg file by computing the slopes file and then the phase data.

        Args:
            image_file_path (Path): Path to the .himg file.

        Returns:
            DataFrame: The computed phase data.
        """
        self._log_info(f"Starting phase analysis for image file: {image_file_path}")
        slopes_file = self.create_slopes_file(image_file_path)
        df = self.compute_phase_from_slopes(slopes_file)
        self._log_info(f"Completed phase analysis for image file: {image_file_path}")
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from geecs_python_api.analysis.scans.scan_data import ScanData
    tag = ScanData.get_scan_tag(year=2025, month=2, day=11, number=11, experiment_name='Undulator')
    analyzer = HasoAnalysis(scan_tag=tag, device_name="U_HasoLift", skip_plt_show=True)
    analyzer.run_analysis()
