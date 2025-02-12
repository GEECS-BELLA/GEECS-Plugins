"""
HamaSpectro Analysis

Analyzer for spectrometer data. 
Could likely be generalized to any fiber-based spectrometer.

authors:
    Kyle Jensen, kjensen@lbl.gov
"""
# %% imports
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scan_analysis.base import ScanAnalysis
from geecs_python_api.analysis.scans.scan_data import ScanTag

# %% classes

class FiberSpectrometerAnalysis(ScanAnalysis):

    def __init__(self, scan_tag: ScanTag, device_name: str,
                 skip_plt_show: bool = True,
                 flag_logging: bool = True,
                 flag_save_images: bool = True) -> None:

        super().__init__(scan_tag, device_name=device_name, skip_plt_show=skip_plt_show)

        # define flags
        self.flags = {'logging': flag_logging,
                      'save_image': flag_save_images,
                      'noscan': self.noscan}

        # organize paths
        self.path_dict = {'data': Path(self.scan_directory) / f"{device_name}",
                          'save': (self.scan_directory.parents[1] / 'analysis'
                                   / self.scan_directory.name / f"{device_name}" / "FiberSpectrometerAnalysis")
                          }

        # initialize future variables
        self.wavelength = None
        self.spectra = None
        self.spectra_binned = None

    def run_analysis(self) -> None:
        try:
            # load  data
            self.wavelength, self.spectra = self.load_spectrometer_data()

            # ensure save location exists
            if self.flags['save_image'] and not self.path_dict['save'].exists():
                self.path_dict['save'].mkdir(parents=True)

            if self.flags['noscan']:
                self.run_noscan_analysis()
            else:
                self.run_scan_analysis()

            return self.display_contents

        except Exception as e:
            raise Exception(f""""
                            Error: FiberSpectrometerAnalysis for {self.device_name} failed. 
                            Error Message: {e}.
                            """)

    def run_scan_analysis(self):

        # bin data
        self.spectra_binned = self.bin_data(self.spectra)

        # plot unbinned waterfall plot
        self.plot_waterfall_unbinned(self.wavelength, self.spectra, save=self.flags['save_image'])

        # plot binned waterfall plot
        self.plot_waterfall_binned(self.wavelength, self.spectra_binned, save=self.flags['save_image'])

    def run_noscan_analysis(self):

        # plot waterfall
        self.plot_waterfall_noscan(self.wavelength, self.spectra, save=self.flags['save_image'])

    def load_spectrometer_data(self) -> [np.ndarray, np.ndarray]:
        """
        Load spectrometer data from txt file.
        Currently specifically for U_HamaSpectro. Can be generalized as needed.

        Returns
        -------
        wavelength: np.ndarray
            Wavelength array from spectrometer.
        spectra: np.ndarray
            Signal arrays from spectrometer for each shot in scan.
        """
        # get list of expected shot numbers
        shot_numbers = self.auxiliary_data['Shotnumber'].values

        # initialize storage
        wavelength = None
        spectra = [[]] * len(shot_numbers)

        # iterate shot numbers
        for ind, shot_num in enumerate(shot_numbers):
            try:
                # get file for shot number
                file = next(self.path_dict['data'].glob(f"*_{shot_num:03d}.txt"), None)

                # read data
                data = pd.read_csv(file, delimiter='\t', header=None)
                spectra[ind] = data[1].values.tolist()
                if wavelength is None:
                    wavelength = data[0].values

            except Exception as e:
                logging.error(f"Error reading data for shot {shot_num}: {e}")
                spectra[ind] = None

        # replace all missing shots with zero arrays
        for ind in [i for i, x in enumerate(spectra) if x is None]:
            spectra[ind] = np.zeros_like(wavelength).tolist()

        return wavelength, np.array(spectra)

    def bin_data(self, data: np.ndarray) -> np.ndarray:

        # initialize storage
        unique_bins = np.unique(self.bins)
        binned_matrix = [[]] * len(unique_bins)

        for ind, bin_num in enumerate(unique_bins):
            binned_matrix[ind] = np.mean(data[self.bins == bin_num], axis=0)

        return np.array(binned_matrix)

    def plot_waterfall_noscan(self, x_data: np.ndarray, yz_data: np.ndarray,
                                save: bool = False, display: bool = True) -> None:

        def set_xlabels(ax):
            ind_minima = self.find_local_minima(x_data%100)
            ax.set_xticks(ind_minima)
            ax.set_xticklabels(x_data[ind_minima].astype(int))

        display = False if not save else None

        im_kwargs = {'cmap': 'viridis',
                     'aspect': 'auto'}

        fig, ax = plt.subplots()

        im = ax.imshow(yz_data, **im_kwargs)

        plt.colorbar(im)

        ax.set_title(f"{self.device_name}")
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel("Shot Number")
    
        set_xlabels(ax)

        plt.tight_layout()
    
        if self.flags['save_image']:
            filename = "noscan.png"
            filepath = self.path_dict['save'] / filename
            plt.savefig(filepath, bbox_inches='tight')
            logging.info(f"Saved no scan image as {filename}.")
    
            if display:
                self.display_contents.append(str(filepath))

        self.close_or_show_plot()

    def plot_waterfall_unbinned(self, x_data: np.ndarray, yz_data: np.ndarray,
                                save: bool = False, display: bool = True) -> None:

        def find_first_index(array, value):
            indices = np.where(array == value)[0]
            return indices[0] if len(indices) > 0 else -1

        def set_xlabels(ax):
            ind_minima = self.find_local_minima(x_data%100)
            ax.set_xticks(ind_minima)
            ax.set_xticklabels(x_data[ind_minima].astype(int))
    
        def set_ylabels(ax):
            ax.set_yticks([find_first_index(self.bins, value) for value in np.unique(self.bins)])
            ax.set_yticklabels(self.binned_param_values.round(2))

        display = False if not save else None

        im_kwargs = {'cmap': 'viridis',
                     'aspect': 'auto'}

        fig, ax = plt.subplots()

        im = ax.imshow(yz_data, **im_kwargs)

        plt.colorbar(im)

        ax.set_title(f"{self.device_name}")
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(f"{self.scan_parameter}")
    
        set_xlabels(ax)
        set_ylabels(ax)

        plt.tight_layout()
    
        if self.flags['save_image']:
            filename = "unbinned_scan.png"
            filepath = self.path_dict['save'] / filename
            plt.savefig(filepath, bbox_inches='tight')
            logging.info(f"Saved unbinned scan image as {filename}.")
    
            if display:
                self.display_contents.append(str(filepath))

        self.close_or_show_plot()

    def plot_waterfall_binned(self, x_data: np.ndarray, yz_data: np.ndarray,
                              save: bool = False, display: bool = True) -> None:

        def set_xlabels(ax):
            ind_minima = self.find_local_minima(x_data%100)
            ax.set_xticks(ind_minima)
            ax.set_xticklabels(x_data[ind_minima].astype(int))

        def set_ylabels(ax):
            ax.set_yticks(np.arange(len(self.binned_param_values)))
            ax.set_yticklabels(self.binned_param_values.round(2))

        display = False if not save else None

        im_kwargs = {'cmap': 'viridis',
                     'aspect': 'auto'}

        fig, ax = plt.subplots()

        im = ax.imshow(yz_data, **im_kwargs)

        plt.colorbar(im)

        ax.set_title(f"{self.device_name}")
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel(f"{self.scan_parameter}")

        set_xlabels(ax)
        set_ylabels(ax)

        plt.tight_layout()

        if self.flags['save_image']:
            filename = "binned_scan.png"
            filepath = self.path_dict['save'] / filename
            plt.savefig(filepath, bbox_inches='tight')
            logging.info(f"Saved binned scan image as {filename}.")

            if display:
                self.display_contents.append(str(filepath))

        self.close_or_show_plot()

    @staticmethod
    def find_local_minima(array):
        minima = (np.r_[True, array[1:] < array[:-1]] &
                  np.r_[array[:-1] < array[1:], True])
        return np.where(minima)[0]

# %% routines

def testing():
    # define scan
    kwargs = {'year': 2025, 'month': 2, 'day': 10,
              'number': 5, 'experiment': 'Undulator'}
    tag = ScanTag(**kwargs)

    # initialize analyzer, run analysis
    analyzer = FiberSpectrometerAnalysis(scan_tag=tag, device_name="U_HamaSpectro")
    analyzer.run_analysis()

# %% execute
if __name__ == "__main__":
    testing()
