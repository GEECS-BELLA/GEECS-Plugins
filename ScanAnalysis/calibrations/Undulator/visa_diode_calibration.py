# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:46:43 2024

@author: kjensen11 (kjensen@lbl.gov)
"""
# =============================================================================
# %% imports
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geecs_paths_utils.scan_paths import ScanTag

from geecs_scan_data_utils.scan_data import ScanData
from scan_analysis.analyzers.Undulator.visa_ebeam_analysis import VisaEBeamAnalysis
from image_analysis.utils import read_imaq_image
from image_analysis.tools.general import image_signal_thresholding, find_beam_properties

import matplotlib.pyplot as plt
# =============================================================================
# %% classes

class VisaBlueDiodeCalibration(VisaEBeamAnalysis):

    def __init__(self, scan_tag: ScanTag):
        """
        Initialize the CameraImageAnalysis class.

        Args:
            scan_directory (str or Path): Path to the scan directory containing data.
            device_name (str): Name of the device to construct the subdirectory path.
        """
        super().__init__(scan_tag)

    @staticmethod
    def plot_calibration_result(image, centroidx, centroidy):

        plt.figure()
        plt.imshow(image)
        plt.plot(centroidx, centroidy, 'ro')
        plt.title(f"Centroid: X = {centroidx}, Y = {centroidy}")
        plt.show()

    def run_analysis(self, show=True):

        # check for existing data
        if self.path_dict['data_img'] is None or self.auxiliary_data is None:
            raise Exception("Warning: No calibration data exists. Skipping calibration.")
            return

        # average calibration images
        image_sum = None
        shot_list = self.auxiliary_data['Shotnumber'].values
        for shot_num in shot_list:
            image_file = next(self.path_dict['data_img'].glob(f"*_{shot_num:03d}.png"), None)

            if image_file is None:
                continue

            if image_sum is None:
                image_sum = read_imaq_image(image_file)
            else:
                image_sum += read_imaq_image(image_file)

        avg_image = image_sum / len(shot_list)

        # image thresholding
        processed_image = image_signal_thresholding(avg_image,
                                                    median_filter_size=2,
                                                    threshold_coeff=0.25)

        # extract beam properties
        beam_properties = find_beam_properties(processed_image)
        centx = round(beam_properties['centroid'][1])
        centy = round(beam_properties['centroid'][0])

        # display image
        if show:
            self.plot_calibration_result(processed_image, centx, centy)

        # organize output
        output = {'centroidx': centx,
                  'centroidy': centy
                  }

        return output

# =============================================================================
# %% functions

# =============================================================================
# %% routines

def testing_VisaEBeamAnalysis():

    # get scan tag
    scan_tag = ScanData.get_scan_tag(year=2024, month=10, day=31, number=33,
                                     experiment_name='Undulator')

    # initialize analysis class
    analysis_class = VisaBlueDiodeCalibration(scan_tag)

    output = analysis_class.run_analysis()

    print(f"Device: {analysis_class.device_name}")
    print(f"Centroid X = {output['centroidx']}")
    print(f"Centroid Y = {output['centroidy']}")

    return
# =============================================================================
# %% execute

if __name__=="__main__":

    testing_VisaEBeamAnalysis()
