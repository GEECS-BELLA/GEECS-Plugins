# -*- coding: utf-8 -*-
"""
M3Near Gaussian Fit analyzer

Child to ScanAnalysis (./scan_analysis/base.py)
"""
# %% imports
from __future__ import annotations

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from scan_analysis.base import ScanAnalysis
from image_analysis.utils import read_imaq_png_image
from image_analysis.analyzers.online_analysis_modules.image_processing_funcs import threshold_reduction

from geecs_python_api.analysis.scans.scan_data import ScanData


# %% classes
class Gaussian_Fit(ScanAnalysis):
    """
    Analyzer for fitting a gaussian fit to a camera image
    """
    def __init__(self, scan_tag: ScanTag, device_name: Optional[str] = None, skip_plt_show: bool = True):
        super().__init__(scan_tag, device_name=None, skip_plt_show=skip_plt_show)

        self.device_list = ['CAM-HPD-M3Near']
        self.background_tag = ScanData.get_scan_tag(year=2025, month=4, day=28, number=3, experiment='N:\data')
        self.backgrounds = {}

        # Check if data directory exists and is not empty
        for device in self.device_list:
            device_path = self.scan_directory / device
            if not device_path.exists() or not any(device_path.iterdir()):
                msg = f"Data directory 'device_path' does not exist or is empty."
                logging.warning(msg)
                raise NotADirectoryError(msg)

        self.save_path = self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / "ScAnalyzer"
        self.load_backgrounds()

    def load_backgrounds(self):
        """ From the background tag, loads 10 images from each device and averages them.  Saved into a dict """
        background_shots = list(range(1, 11))
        for device in self.device_list:
            average_image = None
            for shot_num in background_shots:
                image_file = ScanData.get_device_shot_path(tag=self.background_tag, device_name=device,
                                                           shot_number=shot_num)
                image = read_imaq_png_image(image_file) * 1.0
                average_image = image if average_image is None else average_image + image
            average_image /= 10
            self.backgrounds[device] = average_image

    def run_analysis(self, config_options: Optional[str] = None):
        """ Main function to run the analysis and generate plots. """
        # For HTU we grab these from files
        # energy_values, charge_density_matrix

        # For each shot, stitch together the three images and project it onto the x-axis.  Save the final lineout
        all_projections = []
        for shot_num in self.auxiliary_data['Shotnumber'].values:
            # print(shot_num)
            # stitched_projection = None
            # for device in self.device_list:
            #     # For more complex analysis, the actual `image analysis` code within this `for` block could be moved
            #     #  to a dedicated ImageAnalysis class, and one could multithread the analysis so that each thread gets
            #     #  a single image to analyze.  But for this simple example I am opting to keep it all within this file
            #     image_file = ScanData.get_device_shot_path(tag=self.tag, device_name=device,
            #                                                shot_number=int(float(shot_num)))
            #     image = read_imaq_png_image(image_file)*1.0
            #     image -= self.backgrounds[device]
            #     image[np.where(image < 0)] = 0
            #     image = image[1:, :]
            #     image = median_filter(image, size=3)
            #     image = threshold_reduction(image=image, threshold=2)

            #     projection = np.sum(image, axis=0)
            #     if stitched_projection is None:
            #         stitched_projection = projection
            #     else:
            #         stitched_projection = np.concatenate((stitched_projection, projection))

            #     if device == 'HTT-C23_1_MagSpec1':  # Add on some zeros
            #         stitched_projection = np.concatenate((stitched_projection, np.zeros(200)))

            #     #plt.imshow(image, aspect='auto', vmin = 0, vmax = 15)
            #     #plt.plot(projection)
            #     #plt.show()

            # all_projections.append(stitched_projection)

            #plt.plot(stitched_projection)
            #plt.show()

        all_projections = np.vstack(all_projections)

        plt.figure(figsize=(10, 6))
        plt.imshow(all_projections, aspect='auto')
        plt.xlabel('Stitched Image Horizontal Pixel')
        plt.ylabel('Shotnumber')
        plt.title(f"{self.tag.experiment}: {self.tag.month:02d}/{self.tag.day:02d}/{self.tag.year} Scan{self.tag.number:03d} Raw Magspec Waterfall")

        save_path = self.save_path / 'RawMagspecWaterfall'
        save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        plt.savefig(save_path, bbox_inches='tight')

        self.close_or_show_plot()

        # Could from here do additional analysis and append scalars to the sfile using:
        # self.append_to_sfile(self, dict_to_append={'key': list_of_data}):

        self.display_contents.append(str(save_path))
        return self.display_contents
    
    def load_data(self, filename):
        self._update_hole_in(filename)
        if not self._ref_img_paths:
            top_dir, _, _, _ = pygc.get_top_dir_from_sfilename(self.reference_sfilename, print_data=False)
            scan_data, _, _ = pygc.load_scan_data(top_dir, self.reference_sfilename, 
                                                  diagnostic=self.cam_name, file_ext=".png")
            self._ref_img_paths = list(scan_data[f"{self.cam_name} file_list"])
        return super().load_data(filename)

    def analyze_data(self, data, bg=None):
        """
        If hole is in, fit a gaussian. Otherwise, only calculate the energy transmission
        """
        if self.hole_in:
            # image is fitted image
            # This function already subtracts bg
            image, columns = super().analyze_data(data, bg=bg)
        else:
            # image is original image
            # Still fit gaussian but with 0 hole radius to get normal gaussian fit
            # image = data
            # columns = {}
            if bg:
                bg_data = super().load_data(bg)
                data = data - bg_data
            image, _, columns = self._fit_gaussian_with_hole_and_roi(data, self.hole_x, self.hole_y, 0, self.beam_r)
        
        # Calculating energy transmission
        # Calculate the average mean counts for the reference scan
        # Only does this once to save time
        if not self._mean_counts_ref:
            sum_mean_counts = 0
            for im_path in self._ref_img_paths:
                ref_image = super().load_data(im_path)
                if bg:
                    bg_data = super().load_data(bg)
                    ref_image = ref_image - bg_data
                # Filter out hot pixels
                ref_image = scipy.ndimage.median_filter(ref_image, 5)
                sum_mean_counts += np.mean(ref_image)
            self._mean_counts_ref = sum_mean_counts / len(self._ref_img_paths)

        # Median filter to eliminate hot pixels
        # if not self.hole_in:
        image = scipy.ndimage.median_filter(image, 5)

        energy_trans = np.mean(image) / self._mean_counts_ref * 100

        columns["Energy transmission [%]"] = energy_trans
        return image, columns
        
    def write_analyzed_data(self, save_path, data):
        if self.hole_in:
            super().write_analyzed_data(save_path, data)

    def _update_hole_in(self, filename):
        scan_data = self._get_scan_data_from_filename(filename)
        if self.turn_on_name in scan_data.columns:
            hole_in_entry = scan_data[self.turn_on_name].to_numpy()[0]
            if isinstance(hole_in_entry, str):
                self.hole_in = (hole_in_entry.lower() == "on")
            elif isinstance(hole_in_entry, (np.float64, float, int)):
                self.hole_in = bool(hole_in_entry)
        else:
            self.hole_in = self.default_on

        # Update analysis diagnostic
        if self.hole_in:
            self.analysis_options["analysis_diagnostic"] = "CAM-HPD-M3Near-fitted"
            self.analysis_options["write_analyzed"] = True
        else:
            self.analysis_options["analysis_diagnostic"] = self.diagnostic
            self.analysis_options["write_analyzed"] = False


if __name__ == "__main__":
    from geecs_python_api.analysis.scans.scan_data import ScanData
    tag = ScanData.get_scan_tag(year=2025, month=4, day=28, number=68, experiment='N:\data')
    ScanData.reload_paths_config(default_experiment="Bella")
    analyzer = HTTMagSpecAnalysis(scan_tag=tag, skip_plt_show=False)
    analyzer.run_analysis()
