# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:47:05 2024

@author: kjensen11 (kjensen@lbl.gov)
"""
# =============================================================================
# %% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scan_analysis.analyzers.Undulator.camera_image_analysis import CameraImageAnalyzer
from image_analysis.tools.general import find_beam_properties, image_signal_thresholding
from geecs_data_utils import ScanData
from calibrations.Undulator.calibration_data.experimental_config import get_experimental_config_distance
from calibrations.Undulator.utils import get_calibration_location

# =============================================================================
# %% classes


class MagnetTrajectoryCalibration(CameraImageAnalyzer):

    def __init__(self, device_name,
                 use_gui=True, flag_logging=False, flag_save_images=False):
        """
        Initialize the CameraImageAnalyzer class.

        Args:
            device_name (str): Name of the device to construct the subdirectory path.
        """
        super().__init__(device_name, skip_plt_show=use_gui,
                         flag_logging=flag_logging, flag_save_images=flag_save_images)

    def get_binned_centroids(self, binned_data,
                             median_filter_size=2, threshold_coeff=0.1):

        # loop through bins, get centroids
        for key, item in binned_data.items():

            # copy for convenience
            image = item['image'].copy()

            # threshold image for higher fidelity centroid analysis
            image = image_signal_thresholding(image,
                                              median_filter_size=median_filter_size,
                                              threshold_coeff=threshold_coeff)

            # extract beam properties
            beam_properties = find_beam_properties(image)
            binned_data[key]['centroid'] = [beam_properties['centroid'][1],
                                            beam_properties['centroid'][0]] # [x, y]

        return binned_data

    def perform_calibration(self, binned_data):

        # get spatial calibration value [m/pix]
        spatial_cal = self.camera_analysis_settings['Calibration']

        # get experimental configuration distance
        start_loc = self.scan_parameter.split(' ')[0][:-1]
        end_loc = self.device_name
        distance = get_experimental_config_distance(start_loc, end_loc)

        # determine zero current bin
        scan_values = np.array([binned_data[key]['value'] for key in binned_data.keys()]).round(3)
        ind_zero = np.argwhere(scan_values == 0.).flatten()[0]

        # calculate zero current adjusted centroids
        for key, item in binned_data.items():

            # calcualte centroid offset
            centroid_offset = np.array(item['centroid']) - np.array(binned_data[ind_zero]['centroid'])
            binned_data[key]['centroid_offset'] = centroid_offset.tolist()

            # convert centroid offset to spatial distance
            spatial_offset = (centroid_offset * spatial_cal)
            binned_data[key]['spatial_offset'] = spatial_offset.tolist()

            # calculate angle [rad]
            binned_data[key]['angle_offset'] = np.arctan(spatial_offset / distance)

        return binned_data

    def export_calibration_data(self, binned_data):

        df = pd.DataFrame({'Scan Value': [item['value']
                                          for _, item in binned_data.items()],
                           'Angle Offset X': [item['angle_offset'][0]
                                              for _, item in binned_data.items()],
                           'Angle Offset Y': [item['angle_offset'][1]
                                              for _, item in binned_data.items()]}
                          )

        filepath = (self.path_dict['calibration_data'] /
                    f"deflection_calibration_{self.scan_parameter.split(' ')[0]}.csv")
        df.to_csv(filepath, index=False)

        return

    def plot_calibration(self, binned_data):

        # extract data
        scan_values = [item['value'] for _, item in binned_data.items()]
        spatial_cal = np.array([item['spatial_offset'] for _, item in binned_data.items()]) * 10**3
        angle_cal = np.array([item['angle_offset'] for _, item in binned_data.items()])

        scan_num = int(str(self.scan_directory).split('n')[-1])

        # plot
        fig, ax1 = plt.subplots()

        # plot spatial calibration
        spatx, = ax1.plot(scan_values, spatial_cal[:,0], label='spatial cal (x)',
                          marker='x', linestyle='-', color='blue')
        spaty, = ax1.plot(scan_values, spatial_cal[:,1], label='spatial cal (y)',
                          marker='o', linestyle='-', color='blue')

        ax1.set_xlabel('EMQ current (A)')
        ax1.set_ylabel('Spatial calibration (mm)')
        ax1.tick_params(axis='y', labelcolor='blue')

        # create second axis on right (angle calibration)
        ax2 = ax1.twinx()

        angx, = ax2.plot(scan_values, angle_cal[:,0], label='angle cal (x)',
                         marker='x', linestyle='--', color='red')
        angy, = ax2.plot(scan_values, angle_cal[:,1], label='angle cal (y)',
                         marker='o', linestyle='--', color='red')

        ax2.set_ylabel('Angle calibration (rad)')
        ax2.tick_params(axis='y', labelcolor='red')

        # set title
        plt.title(f"Scan: {scan_num}, Parameter: {self.scan_parameter}, Device: {self.device_name}")

        # include comprehensive legend
        lines = [spatx, spaty, angx, angy]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        # show plot
        plt.show()

        return

    def _run_analysis_core(self, analysis_settings=None, flag_save=True):

        super()._run_analysis_core()

        self.path_dict['calibration_data'] = get_calibration_location() / 'calibration_data'

        if flag_save is None:
            flag_save = self.flag_save_images
        if analysis_settings is None:
            analysis_settings = self.camera_analysis_settings

        # perform bulk image analysis
        binned_data = self.binned_data

        # get centroid for each binned image
        binned_data = self.get_binned_centroids(binned_data)

        # perform calibration
        binned_data = self.perform_calibration(binned_data)

        # export calibration
        if flag_save:
            self.export_calibration_data(binned_data)

        # plot calibration
        self.plot_calibration(binned_data)

# =============================================================================
# %% executable


if __name__ == "__main__":
    scan = {'year': '2024',
            'month': 'Nov',
            'day': '14',
            'num': 35}
    device_name = "UC_ALineEBeam3"
    scan_tag = ScanData.get_scan_tag(year=scan['year'], month=scan['month'], day=scan['day'],
                                     number=scan['num'], experiment_name='Undulator')
    analysis_class = MagnetTrajectoryCalibration(device_name)
    analysis_class.run_analysis(scan_tag=scan_tag)
