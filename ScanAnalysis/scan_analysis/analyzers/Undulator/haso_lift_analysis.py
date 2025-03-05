"""
Haso4_3 phase retrieval Analysis

Child to ScanAnalysis (./scan_analysis/base.py)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag
    from numpy.typing import NDArray

from pathlib import Path
import logging
import pandas as pd

from scan_analysis.base import ScanAnalysis
from image_analysis.analyzers.HASO_himg_has_processor import HASOHimgHasProcessor, FilterParameters

class HasoAnalysis(ScanAnalysis):
    """
    HasoAnalysis performs phase retrieval analysis using a HASO device.
    It processes data from a specified scan directory and computes slopes
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
        
        # Add a list to store phase DataFrames
        self.phase_dfs = []

        # Organize paths: one for image data and one for analysis output.
        self.path_dict = {
            'data_img': Path(self.scan_directory) / device_name,
            'save': self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / device_name / "HasoAnalysis",
        }

        self.path_to_bkg_has_file = None
        self.haso_processor = HASOHimgHasProcessor(background_path=self.path_to_bkg_has_file)
        self.haso_processor.filter_params = FilterParameters(apply_tiltx_filter=True, apply_tilty_filter=True,
                                                        apply_curv_filter=True)
                
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
        
        #use an accumulator to sum the individual phases files to create an average at the end
        accumulator = None
        count = 0

        for shot_num in self.auxiliary_data['Shotnumber'].values:
            self._log_info(f"Processing shot number: {shot_num}")
            # Try to locate the .himg file.
            himg_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}.himg'), None)
            has_file = next(self.path_dict['data_img'].glob(f'*_{shot_num:03d}.has'), None)

            if himg_file:
                self._log_info(f"Found .himg file for shot {shot_num}: {himg_file}")
                result = self.haso_processor.analyze_image(himg_file)
                base_file_path = himg_file.stem
            else:
                if has_file:
                    self._log_info(f"No .himg file found for shot {shot_num}. Using existing slopes file: {has_file}")
                    result = self.haso_processor.analyze_image(has_file)
                    base_file_path = has_file.stem
                else:
                    self._log_warning(f"Missing data for shot {shot_num}. Neither .himg nor .has file exists.")
                    continue  # Skip to next shot
            if result:
                df = self.process_haso_data(result=result, base_file_path=base_file_path)


            # Convert DataFrame to numpy array for efficient arithmetic.
            data = df.to_numpy()
            if accumulator is None:
                accumulator = data
            else:
                accumulator += data
            count += 1

        if count > 0:
            average = accumulator / count
            avg_df = pd.DataFrame(average)
            avg_file_path = self.path_dict['save'] / 'average_phase.tsv'
            avg_df.to_csv(avg_file_path, sep="\t", index=False, header=False)
            self._log_info(f"Average phase data saved to TSV file: {avg_file_path}")
        else:
            self._log_warning("No phase data available to average.")

    def process_haso_data(self, result, base_file_path):
        # Unpack the returned tuple.
        raw_slopes, processed_slopes, raw_phase, processed_phase, intensity = result
        self.save_individual_results(result, base_file_path)
        return pd.DataFrame(raw_phase)

    def save_individual_results(self,result, base_file_path):
        # Unpack the returned tuple.
        raw_slopes, processed_slopes, raw_phase, processed_phase, intensity = result

        self.slopes_file_path_raw = self.path_dict['save'] / f"{base_file_path}_raw.has"
        self.slopes_file_path_postprocessed = self.path_dict['save'] / f"{base_file_path}_postprocessed.has"

        self.save_slopes_file(slopes_data = raw_slopes, save_path = self.slopes_file_path_raw)
        self.save_slopes_file(slopes_data = processed_slopes, save_path = self.slopes_file_path_postprocessed)

        self.raw_phase_file_path = self.path_dict['save'] / f"{base_file_path}_raw.tsv"
        self.processed_phase_file_path = self.path_dict['save'] / f"{base_file_path}_postprocessed.tsv"
        self.intensity_file_path = self.path_dict['save'] / f"{base_file_path}_intensity.tsv"

        self.save_phase_file(phase_values = raw_phase, save_path = self.raw_phase_file_path)
        self.save_phase_file(phase_values = processed_phase, save_path = self.processed_phase_file_path)
        self.save_phase_file(phase_values = intensity, save_path = self.intensity_file_path)

    def save_slopes_file(self, slopes_data: HasoSlopes, save_path: Path):
        slopes_data.save_to_file(str(save_path), '', '')

    def save_phase_file(self, phase_values: float|NDArray, save_path: Path):
        df = pd.DataFrame(phase_values)
        df.to_csv(save_path, sep="\t", index=False, header=False)

    def run_scan_analysis(self) -> None:
        """
        Perform analysis for a scan scenario.
        Delegates to the no-scan analysis.
        """
        self._log_info("Delegating scan analysis to no-scan analysis.")
        self.run_noscan_analysis()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from geecs_python_api.analysis.scans.scan_data import ScanData
    tag = ScanData.get_scan_tag(year=2025, month=2, day=19, number=2, experiment_name='Undulator')
    analyzer = HasoAnalysis(scan_tag=tag, device_name="U_HasoLift", skip_plt_show=True)
    analyzer.run_analysis()
