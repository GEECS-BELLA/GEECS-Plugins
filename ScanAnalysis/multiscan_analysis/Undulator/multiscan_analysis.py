"""

Multi-scan Analyzer

Contributors:
Kyle Jensen, kjensen@lbl.gov

"""
# =============================================================================
# %% imports
from __future__ import annotations
from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import ScanTag

import os
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from geecs_python_api.analysis.scans.scan_data import ScanData
from scan_analysis.analyzers.Undulator.VisaEBeamAnalysis import VisaEBeamAnalysis
from image_analysis.utils import read_imaq_image

# =============================================================================
# %% classes

class MultiScanAnalysis():
    """
    Intended to be a parent class for generalized multi-scan analysis.
    Not sure how to generalize at this point.
    """

    def __init__(self, scan_tags: ScanTag) -> None:
        """

        Parameters
        ----------
        input_scans : TYPE
            Needs structure documentation! Example in executable routine below.

        Returns
        -------
        None.

        """
        # compile scan information
        self.scans = self.compile_scan_info(scan_tags)

    def compile_scan_info(self, scan_tags: ScanTag) -> dict[int: dict[str: ScanTag]]:

        # initialize storage dict
        scans = {item.number: {} for item in scan_tags}

        # iterate through scans, compiling useful information
        for scan_tag in scan_tags:
            scans[scan_tag.number]['tag'] = scan_tag

        return scans

    @staticmethod
    def load_sfile(filepath) -> pd.DataFrame:
        dataframe = pd.read_table(filepath, header=0)
        return dataframe

class MultiScanAnalysis_Visa(MultiScanAnalysis):

    def __init__(self, scan_tags: ScanTag) -> None:
        super().__init__(scan_tags)

    @staticmethod
    def get_diode_coords(analysis_settings: dict) -> tuple:

        coords = (analysis_settings['Blue Centroid X'] - analysis_settings['Left ROI'],
                  analysis_settings['Blue Centroid Y'] - analysis_settings['Top ROI'])

        return coords

    def scan_analysis(self, scan: dict[str: ScanTag]) -> dict[str: Union[ScanTag, tuple, dict[str: np.ndarray]]]:

        # initialize VisaEBeamAnalysis
        analyzer = VisaEBeamAnalysis(scan['tag'])

        analysis_settings = analyzer.camera_analysis_settings
        self.scan_parameter = analyzer.scan_parameter

        scan['device'] = analyzer.device_name
        scan['save_path'] = analyzer.path_dict['save']

        # get blue diode coordinates
        scan['diode_coords'] = self.get_diode_coords(analysis_settings)

        # perform visa e beam analysis
        analyzer.run_analysis()

        # record in scan dict
        scan['binned_data'] = {key: {'value': item['value'],
                                     'image': item['image']}
                               for key, item in analyzer.binned_data.items()}

        return scan

    def load_analysis_results(self, scan: dict[str: ScanTag]) -> dict[str: Union[ScanTag, str, tuple, dict[str: np.ndarray]]]:
        """
        Load existing analysis images rather than perform full analysis.

        Needs to be revisited once merged.
        """

        # initialize VisaEBeamAnalysis
        # assumes scan parameter is the same for each scan
        analyzer = VisaEBeamAnalysis(scan['tag'])

        analysis_settings = analyzer.camera_analysis_settings
        self.scan_parameter = analyzer.scan_parameter

        scan['device'] = analyzer.device_name
        scan['save_path'] = analyzer.path_dict['save']

        # get blue diode coordinates
        scan['diode_coords'] = self.get_diode_coords(analysis_settings)

        # get bin numbers and values (assumes correlated bin number and value!!)
        bin_num = np.unique(analyzer.bins)
        bin_vals = analyzer.binned_param_values

        # load already processed images
        scan['binned_data'] = {val: {'value': bin_vals[ind],
                                     'image': cv2.imread(analyzer.path_dict['save']
                                                         / f"{scan['device']}_{val}_processed.png",
                                                         cv2.IMREAD_UNCHANGED)}
                               for ind, val in enumerate(bin_num)}

        return scan

    def compile_image_matrix(self, scans: dict[int: dict[str: Union[ScanTag, str, tuple, dict[str: np.ndarray]]]],
                             ref_coords: bool =True, save_figure: bool =True) -> None:
        """
        Arrange averaged images from multiple scans in matrixed grid.

        Assumes each scan has the same number of scan parameter bins.
        Can implement checks/solutions for failure to meet this criteria in the future.
        """

        # set save path
        first_scan = np.min(list(scans.keys()))
        if save_figure:
            save_path = scans[first_scan]['save_path'] / "multiscan_scan_matrix.png"

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

    def run_routine(self, run_analysis: bool = False) -> None:

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
# %% routines

def run_multiscan_visa():

    experiment = 'Undulator'
    year = 2024
    month = 12
    day = 5
    scans = [20, 21, 22]

    # generate list of scan tags
    scan_tags = [ScanData.get_scan_tag(year=year, month=month, day=day,
                                       number=num, experiment_name=experiment)
                 for num in scans]

    # initialize analyzer
    analyzer = MultiScanAnalysis_Visa(scan_tags)

    # run routine
    analyzer.run_routine(run_analysis=False)

# =============================================================================
# %% execute
if __name__ == "__main__":

    run_multiscan_visa()
