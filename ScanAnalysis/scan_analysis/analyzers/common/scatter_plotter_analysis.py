from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional, List
if TYPE_CHECKING:
    from geecs_data_utils import ScanTag
    from matplotlib.pyplot import Axes

from pathlib import Path
import logging
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter, gaussian_filter
import imageio as io

from scan_analysis.base import ScanAnalyzer

import traceback
PRINT_TRACEBACK = True


class ScatterPlotterAnalysis(ScanAnalyzer):
    def __init__(self, scan_tag: ScanTag, scatter_plot: bool, use_median: bool,
                 data_key_1: str, label_1: str, ylabel_1: str,
                 data_key_2: Optional[str] = None, label_2: Optional[str] = None, ylabel_2: Optional[str] = None,
                 skip_plt_show: bool = True,
                 device_name: Optional[str] = None,
                 flag_logging: bool = True,
                 ):
        super().__init__(scan_tag, skip_plt_show, device_name)
        self.flag_logging = flag_logging

        # Set up key names to load from auxiliary data
        found_keys = self.auxiliary_data.keys()
        if data_key_1 not in found_keys:
            raise KeyError(f"f{data_key_1} not found in sfile for Scan {scan_tag.number}")
        if data_key_2 is not None and data_key_2 not in found_keys:
            raise KeyError(f"f{data_key_2} not found in sfile for Scan {scan_tag.number}")

        self.data_key_1: str = data_key_1
        self.data_key_2: Optional[str] = data_key_2

        self.label_1: str = label_1
        self.label_2: Optional[str] = label_2
        self.ylabel_1: str = ylabel_1
        self.ylabel_2: Optional[str] = ylabel_2

        # Plotting options
        self.scatter_plot: bool = scatter_plot
        self.use_median: bool = use_median

    def run_analysis(self) -> Optional[list[Union[Path, str]]]:
        try:
            # Call the function to generate scatter plot
            self.generate_plots()

            # Return from analysis method with display contents
            return self.display_contents

        except Exception as e:
            if PRINT_TRACEBACK:
                print(traceback.format_exc())
            if self.flag_logging:
                logging.warning(f"Warning: Scatter Plotter analysis failed due to: {e}")
            return

    def generate_plots(self) -> None:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        self._plot_data(axis=ax1, key_name=self.data_key_1, c='g', label=self.label_1, ylabel=self.ylabel_1)
        ax1.set_xlabel(self.scan_parameter)

        if self.data_key_2 is not None:
            ax2 = ax1.twinx()
            self._plot_data(axis=ax2, key_name=self.data_key_2, c='g', label=self.label_2, ylabel=self.ylabel_2)

        self.close_or_show_plot()

    def _plot_data(self, axis: Axes, key_name: str, c: str, label: str, ylabel: str):
        binned_data = self.process_to_bins(key_name)
        if self.use_median:
            stat_type = 'median'
        else:
            stat_type = 'average'

        if self.scatter_plot:
            axis.scatter(binned_data['bin'], binned_data[stat_type], c=c, label=label)
        else:
            axis.plot(binned_data['bin'], binned_data[stat_type], c=c, ls='-', label=label)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis='y', labelcolor=c)

    def process_to_bins(self, key_name: str) -> dict[str, np.ndarray]:
        # Load data from sfile contents for key name
        data_array = self.auxiliary_data[key_name]

        # Identify unique parameter bins
        unique_bins = np.unique(self.bins)
        if self.flag_logging:
            logging.info(f"unique_bins: {unique_bins}")

        # Preallocate storage
        bins = np.zeros(len(unique_bins))
        average = np.zeros(len(unique_bins))
        sigma = np.zeros(len(unique_bins))
        median = np.zeros(len(unique_bins))

        # Iterate parameter bins
        for bin_ind, bin_val in enumerate(unique_bins):
            # Sort by shots within this bin
            shots_in_bin = self.auxiliary_data[self.auxiliary_data['Bin #'] == bin_val]['Shotnumber'].values - 1
            data_in_bin = data_array[shots_in_bin]

            # Calculate stats on the data in this bin
            bins[bin_ind] = self.binned_param_values[bin_ind]
            average[bin_ind] = np.average(data_in_bin)
            sigma[bin_ind] = np.std(data_in_bin)
            median[bin_ind] = np.median(data_in_bin)

        return {
            'bin': bins,
            'average': average,
            'sigma': sigma,
            'median': median
        }
