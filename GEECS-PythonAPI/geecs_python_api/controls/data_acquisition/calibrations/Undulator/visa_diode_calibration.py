# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:46:43 2024

@author: kjensen11 (kjensen@lbl.gov)
"""
# =============================================================================
# %% imports
from skimage.measure import regionprops, label
from scipy.ndimage import median_filter

from geecs_python_api.controls.data_acquisition.scan_analysis import CameraImageAnalysis
from image_analysis.utils import read_imaq_image

import matplotlib.pyplot as plt
# =============================================================================
# %% classes

class VisaBlueDiodeCalibration(CameraImageAnalysis):

    def __init__(self, scan_directory, device_name, use_gui=True, experiment_dir = 'Undulator'):
        """
        Initialize the CameraImageAnalysis class.

        Args:
            scan_directory (str or Path): Path to the scan directory containing data.
            device_name (str): Name of the device to construct the subdirectory path.
        """
        super().__init__(scan_directory, device_name, use_gui=use_gui)

    def image_signal_thresholding(self, image, thresh_median=2, thresh_coeff=0.25):
        '''
        Function exists within image_analysis.analyzers.generic_beam_analyser.py
        Should consolidate duplicated code
        '''

        data_type = image.dtype
        image = image.astype('float64')

        # perform median filtering
        blurred = median_filter(image, size=thresh_median)

        # threshold with respect to the blurred image max
        image[blurred < blurred.max() * thresh_coeff] = 0

        return image.astype(data_type)

    @staticmethod
    def find_beam_properties(image):
        '''
        Function exists within image_analysis.analyzers.generic_beam_analyser.py
        Should consolidate duplicated code
        '''

        # initialize beam properties dict
        beam_properties = {}

        # construct binary and label images
        image_binary = image.copy()
        image_binary[image_binary > 0] = 1
        image_binary = image_binary.astype(int)
        image_label = label(image_binary)

        # get beam properties and reduce to the largest region
        props = regionprops(image_label, image)
        areas = [i.area for i in props]
        props = props[areas.index(max(areas))]

        # extract centroid
        beam_properties['centroid'] = props.centroid_weighted

        return beam_properties

    @staticmethod
    def plot_calibration_result(image, centroidx, centroidy):

        fig = plt.figure()
        plt.imshow(image)
        plt.plot(centroidx, centroidy, 'ro')
        plt.title(f"Centroid: X = {centroidx}, Y = {centroidy}")
        plt.show()

    def run_analysis(self, show=True):

        # check for existing data
        if self.data_subdirectory is None or self.auxiliary_data is None:
            raise Exception("Warning: No calibration data exists. Skipping calibration.")
            return

        # average calibration images
        image_sum = None
        shot_list = self.auxiliary_data['Shotnumber'].values
        for shot_num in shot_list:
            image_file = next(self.data_subdirectory.glob(f"*_{shot_num:03d}.png"), None)

            if image_file is None:
                continue

            if image_sum is None:
                image_sum = read_imaq_image(image_file)
            else:
                image_sum += read_imaq_image(image_file)

        avg_image = image_sum / len(shot_list)

        # image thresholding
        processed_image = self.image_signal_thresholding(avg_image)

        # extract beam properties
        beam_properties = self.find_beam_properties(processed_image)
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

    # imports
    from geecs_python_api.controls.data_acquisition.data_acquisition import DataInterface

    # define scan information
    scan_dict = {'year': '2024',
            'month': 'Oct',
            'day': '31',
            'num': 33}
    device_name = "UC_VisaEBeam8"

    # initialize and configure data interface
    data_interface = DataInterface()
    data_interface.year = scan_dict['year']
    data_interface.month = scan_dict['month']
    data_interface.day = scan_dict['day']
    (raw_data_path,
     analysis_data_path) = data_interface.create_data_path(scan_dict['num'])

    # initialize analysis class
    scan_directory = raw_data_path / f"Scan{scan_dict['num']:03d}"
    analysis_class = VisaBlueDiodeCalibration(scan_directory, device_name)

    output = analysis_class.run_analysis()

    print(f"Centroid X = {output['centroidx']}")
    print(f"Centroid Y = {output['centroidy']}")

    return
# =============================================================================
# %% execute

if __name__=="__main__":

    testing_VisaEBeamAnalysis()