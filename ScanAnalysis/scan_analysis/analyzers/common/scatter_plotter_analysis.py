from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional
if TYPE_CHECKING:
    from geecs_data_utils import ScanTag
    from matplotlib.pyplot import Axes

from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

from scan_analysis.base import ScanAnalyzer

import traceback
PRINT_TRACEBACK = True


class ScatterPlotterAnalysis(ScanAnalyzer):
    def __init__(self, scan_tag: ScanTag, use_median: bool,
                 title: str,
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
        self.title: str = title

        if use_median:
            self.stat_type = 'median'
        else:
            self.stat_type = 'average'

    def run_analysis(self) -> Optional[list[Union[Path, str]]]:
        try:
            if self.noscan:
                self.run_noscan_analysis()
            else:
                self.run_scan_analysis()

            plt.legend()
            plt.grid()
            plt.title(self.title)

            self.close_or_show_plot()

            return self.display_contents

        except Exception as e:
            if PRINT_TRACEBACK:
                print(traceback.format_exc())
            if self.flag_logging:
                logging.warning(f"Warning: Image analysis failed due to: {e}")
            return None

    def run_noscan_analysis(self) -> None:
        fig, ax1 = plt.subplots(figsize=(7, 5))
        self._generate_noscan_plot(axis=ax1, key_name=self.data_key_1, c='g', label=self.label_1, ylabel=self.ylabel_1)
        ax1.set_xlabel("Shotnumber")

    def _generate_noscan_plot(self, axis: Axes, key_name: str, c: str, label: str, ylabel: str):
        data_array = self.auxiliary_data[key_name].values
        shotnumber = self.auxiliary_data['Shotnumber'].values

        average = np.average(data_array)
        sigma = np.std(data_array)
        if self.stat_type == 'median':
            center = np.median(data_array)
        else:
            center = average

        axis.scatter(shotnumber, data_array, c=c, label=label)

        axis.hlines(y=center, colors=c, linestyles='solid', xmin=shotnumber[0], xmax=shotnumber[-1])
        axis.hlines(y=average-sigma, colors=c, linestyles='dashed', xmin=shotnumber[0], xmax=shotnumber[-1])
        axis.hlines(y=average+sigma, colors=c, linestyles='dashed', xmin=shotnumber[0], xmax=shotnumber[-1])

        axis.set_ylabel(ylabel)
        axis.tick_params(axis='y', labelcolor=c)

    def run_scan_analysis(self) -> None:
        fig, ax1 = plt.subplots(figsize=(7, 5))

        self._generate_scan_plot(axis=ax1, key_name=self.data_key_1, c='g', label=self.label_1, ylabel=self.ylabel_1)
        ax1.set_xlabel(self.scan_parameter)

        if self.data_key_2 is not None:
            ax2 = ax1.twinx()
            self._generate_scan_plot(axis=ax2, key_name=self.data_key_2, c='g', label=self.label_2, ylabel=self.ylabel_2)

    def _generate_scan_plot(self, axis: Axes, key_name: str, c: str, label: str, ylabel: str):
        binned_data = self.process_to_bins(key_name)

        axis.scatter(binned_data['bin'], binned_data[self.stat_type], c=c, s=10)
        axis.plot(binned_data['bin'], binned_data[self.stat_type], c=c, ls='-', label=label)
        axis.errorbar(binned_data['bin'], binned_data['average'], yerr=binned_data['sigma'], c=c, ls='none', capsize=3)

        axis.set_ylabel(ylabel)
        axis.tick_params(axis='y', labelcolor=c)

    def process_to_bins(self, key_name: str) -> dict[str, np.ndarray]:
        # Load data from sfile contents for key name
        data_array = self.auxiliary_data[key_name].values

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
            shots_in_bin = self.auxiliary_data[self.auxiliary_data['Bin #'] == bin_val]['Shotnumber'].values-1

            data_in_bin = data_array[shots_in_bin]

            # Calculate stats on the data in this bin
            bins[bin_ind] = self.binned_param_values[bin_ind]
            average[bin_ind] = np.average(data_in_bin)
            sigma[bin_ind] = np.std(data_in_bin)
            median[bin_ind] = np.median(data_in_bin)

            if self.flag_logging:
                logging.info(f"Bin Index: {bin_ind}")
                logging.info(f"Bin Value: {self.binned_param_values[bin_ind]}")
                logging.info(f"Shots: {shots_in_bin[0]} - {shots_in_bin[-1]}")

        return {
            'bin': bins,
            'average': average,
            'sigma': sigma,
            'median': median
        }
