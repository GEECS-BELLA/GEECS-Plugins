"""

Multi-scan Analyzer

Contributors:
Kyle Jensen, kjensen@lbl.gov

"""
# =============================================================================
# %% imports
import os
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geecs_python_api.controls.data_acquisition.data_acquisition import DataInterface
from geecs_python_api.controls.data_acquisition.scan_analysis import VisaEBeamAnalysis
from image_analysis.utils import read_imaq_image

# =============================================================================
# %% classes

class MultiScanAnalysis():
    """
    Intended to be a parent class for generalized multi-scan analysis.
    Not sure how to generalize at this point.
    """

    def __init__(self):
        self.data_interface = DataInterface()

    def parse_scans(self, input_scans):
        """
        Parse input scans and organize.
        """

        # parse input
        scan_numbers = list(input_scans['scan_num'])
        scan_years = ([input_scans['year']] * len(scan_numbers)
                      if isinstance(input_scans['year'], str)
                      else list(input_scans['year']))
        scan_months = ([input_scans['month']] * len(scan_numbers)
                      if isinstance(input_scans['month'], str)
                      else list(input_scans['month']))
        scan_days = ([input_scans['day']] * len(scan_numbers)
                      if isinstance(input_scans['day'], str)
                      else list(input_scans['day']))

        # check compatibility of scan dates
        if (len(scan_years) != len(scan_numbers)
            or len(scan_months) != len(scan_numbers)
            or len(scan_days) != len(scan_numbers)):
            raise Exception("Error: Incompatible dates specifications.")

        # organize isolated scan dicts
        scan_dict = {}
        for ind, scan in enumerate(scan_numbers):
            scan_dict[scan] = {'day': scan_days[ind],
                               'month': scan_months[ind],
                               'year': scan_years[ind],
                               'num': scan}

            (scan_dict[scan]['path_scan'],
             scan_dict[scan]['path_analysis'],
             scan_dict[scan]['path_sfile']) = self.get_scan_paths(scan_dict[scan])

        return scan_dict

    def get_scan_paths(self, scan_info):

        self.data_interface.year = scan_info['year']
        self.data_interface.month = scan_info['month']
        self.data_interface.day = scan_info['day']

        raw_data_path, analysis_path = self.data_interface.create_data_path(scan_info['num'])
        scan_directory = raw_data_path / f"Scan{scan_info['num']:03d}"
        analysis_directory = analysis_path / f"Scan{scan_info['num']:03d}"
        sfile_path = analysis_path / f"s{scan_info['num']}.txt"

        return scan_directory, analysis_directory, sfile_path

    @staticmethod
    def load_sfile(filepath):
        dataframe = pd.read_table(filepath, header=0)
        return dataframe

class MultiScanAnalysis_Visa(MultiScanAnalysis):

    def __init__(self, input_scans):

        # initialize data interface object
        self.data_interface = DataInterface()

        self.scans = self.parse_scans(input_scans)

    def parse_scans(self, input_scans):
        """
        Parse input scans and organize.
        """

        # use super parse function
        scan_dict = super().parse_scans(input_scans)

        # identify scan specific device
        template = "UC_VisaEBeam*"
        for key, item in scan_dict.items():

            # get list of paths satisfying template
            template_dir = os.path.join(item['path_scan'], template)
            path_list = list(glob.glob(template_dir))

            # iterate through path list for item of interest
            for val in os.listdir(item['path_scan']):
                test_path = os.path.join(item['path_scan'], val)

                # if path is a directory and satisfies template
                # then record to scan dict and break loop
                if (os.path.isdir(test_path) and test_path in path_list):
                    scan_dict[key]['device'] = val
                    break

        return scan_dict

    def scan_analysis_backup(self, scan, scan_parameter=None):

        # load sfile
        sfile = super().load_sfile(scan['path_sfile'])

        # identify scan parameter bin values
        bins = sfile[scan_parameter].unique()

        # identify shots for each bin
        binned_shots = {val: sfile.loc[sfile[scan_parameter] == val, 'Shotnumber'].tolist() for val in bins}

        # for each bin, load images and average

        pass

    def scan_analysis(self, scan):

        # initialize VisaEBeamAnalysis
        analyzer = VisaEBeamAnalysis(scan['path_scan'], scan['device'])
        self.scan_parameter = analyzer.scan_parameter

        # get blue diode coordinates
        analysis_settings = analyzer.camera_analysis_configs[scan['device']]
        scan['diode_coords'] = (analysis_settings['Blue Centroid X'] - analysis_settings['Left ROI'],
                                analysis_settings['Blue Centroid Y'] - analysis_settings['Top ROI'])

        # perform visa e beam analysis
        binned_images = analyzer.perform_bulk_image_analysis()

        # record in scan dict
        scan['binned_data'] = {ind+1: {'value': val,
                                       'image': binned_images[ind]}
                               for ind, val in enumerate(analyzer.binned_param_values)}

        return scan

    def load_analysis_results(self, scan):
        """
        Load existing analysis images rather than perform full analysis.

        Needs to be revisited once merged.
        """

        # initialize VisaEBeamAnalysis
        analyzer = VisaEBeamAnalysis(scan['path_scan'], scan['device'])
        self.scan_parameter = analyzer.scan_parameter

        # get bin numbers and values (assumes correlated bin number and value!!)
        bin_num = np.unique(analyzer.bins)
        bin_vals = analyzer.binned_param_values

        # load already processed images
        img = cv2.imread('Z:/data/Undulator/Y2024/10-Oct/24_1031/scans/Scan053/UC_VisaEBeam1_1_processed.png', cv2.IMREAD_UNCHANGED)
        scan['binned_data'] = {val: {'value': bin_vals[ind],
                                     'image': read_imaq_image(analyzer.scan_directory / f"{scan['device']}_{val}_processed.png")}
                               for ind, val in enumerate(bin_num)}

        return scan

    def compile_image_matrix(self, scans, ref_coords=True, save_figure=True):
        """
        Arrange averaged images from multiple scans in matrixed grid.

        Assumes each scan has the same number of scan parameter bins.
        Can implement checks/solutions for failure to meet this criteria in the future.
        """

        # set save path
        first_scan = np.min(list(scans.keys()))
        if save_figure:
            save_path = scans[first_scan]['path_analysis'] / f"multiscan_scan_matrix.png"

        # get grid size
        grid_row = len(scans.keys())
        grid_col = len(scans[first_scan]['binned_data'].keys())

        # get global color scale
        all_pixels = np.concatenate([scans[snum]['binned_data'][bnum]['image'].ravel()
                                     for snum in list(scans.keys())
                                     for bnum in list(scans[snum]['binned_data'].keys())
                                     if scans[snum]['binned_data'][bnum]['image'] is not None])
        vmin, vmax = 0, all_pixels.max()

        # create figure with appropriate number of subplots
        fig, ax = plt.subplots(grid_row, grid_col, figsize=(grid_col*3, grid_row*3))

        fig.suptitle(f"MultiScan Visualizer\n{self.scan_parameter}")

        # loop through images
        for scan_ind, scan_num in enumerate(scans.keys()):
            scan_item = scans[scan_num]['binned_data']
            for bin_ind, bin_num in enumerate(scan_item.keys()):
                bin_item = scan_item[bin_num]

                # display image with adjusted scale
                ax[scan_ind, bin_ind].imshow(bin_item['image'],
                                             cmap='plasma', vmin=vmin, vmax=vmax)

                # include scan parameter value as title
                ax[scan_ind, bin_ind].set_title(f"{bin_item['value']:.2f}")

                # hide axes for cleaner display
                # ax[scan_ind, bin_ind].axis('off')
                ax[scan_ind, bin_ind].set_yticklabels([])
                ax[scan_ind, bin_ind].set_xticklabels([])
                ax[scan_ind, bin_ind].set_yticks([])
                ax[scan_ind, bin_ind].set_xticks([])

                # add row labels
                if bin_ind == 0:
                    ax[scan_ind, 0].set_ylabel(f"Scan {scan_num}\n{scans[scan_num]['device']}")

                # include reference coordinate (needs modified for each visa screen)
                if ref_coords:
                    ax[scan_ind, bin_ind].plot(scans[scan_num]['diode_coords'][0],
                                               scans[scan_num]['diode_coords'][1],
                                               color='g', marker='o')

        # for scan_ind, scan_num in enumerate(scans.keys()):
        #     ax[scan_ind, 0].set_ylabel(f"Scan {scan_num}")

        # plt.tight_layout()

        if save_figure:
            plt.savefig(save_path, bbox_inches='tight')

        plt.close()

        return

    def run_routine(self, run_analysis=False):

        # run analysis for each scan
        for key, item in self.scans.items():
            if run_analysis:
                self.scans[key] = self.scan_analysis(item)
            else:
                self.scans[key] = self.load_analysis_results(item)

        # compile image matrix
        self.compile_image_matrix(self.scans)

        return

# =============================================================================
# %% functions

# =============================================================================
# %% routines

def run_multiscan_visa():

    # initialize data set info
    scans = {'year': '2024',
             'month': 'Nov',
             'day': '05',
             'scan_num': [37, 38, 39, 40]}

    # initialize analyzer
    analyzer = MultiScanAnalysis_Visa(scans)

    # run routine
    analyzer.run_routine(run_analysis=True)

# =============================================================================
# %% execute
if __name__ == "__main__":

    run_multiscan_visa()
