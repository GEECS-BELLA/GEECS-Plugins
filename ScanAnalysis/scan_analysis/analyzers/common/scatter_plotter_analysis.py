from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional, NamedTuple
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

class PlotParameter(NamedTuple):
    key_name: str
    legend_label: str
    axis_label: str
    color: str


class ScatterPlotterAnalysis(ScanAnalyzer):
    def __init__(self,
                 use_median: bool,
                 title: str,
                 parameters: Union[PlotParameter, list[PlotParameter]],
                 filename: str,
                 skip_plt_show: bool = True,
                 device_name: Optional[str] = None,
                 flag_logging: bool = True,
                 ):
        super().__init__(skip_plt_show=skip_plt_show, device_name=device_name)

        self.flag_logging = flag_logging
        self.filename = filename
        self.base_title = title
        self.title: str

        if use_median:
            self.stat_type = 'median'
        else:
            self.stat_type = 'average'

        self.parameters: list[PlotParameter]
        if not isinstance(parameters, list):
            self.parameters = [parameters]
        else:
            self.parameters = parameters

    def _check_parameters(self):
        # Set up key names to load from auxiliary data
        found_keys = self.auxiliary_data.keys()

        for param in self.parameters:
            if param.key_name not in found_keys:
                raise KeyError(f"f{param.key_name} not found in sfile for Scan {self.scan_tag.number}")

    def _set_plotting_options(self):
        # Plotting options
        self.title: str = (f"{self.scan_tag.month:02d}/{self.scan_tag.day:02d}/{self.scan_tag.year%1000:02d} "
                           f"Scan{self.scan_tag.number:03d}: {self.base_title}")

        self.save_path = self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / "ParameterPlots"

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        self._check_parameters()
        self._set_plotting_options()

        try:
            if self.noscan:
                self.run_noscan_analysis()
            else:
                self.run_scan_analysis()

            # Perform some final plot touch-ups
            plt.legend()
            plt.grid()
            plt.title(self.title)
            plt.tight_layout()

            # Save the plot
            save_path = Path(self.save_path) / f"{self.filename}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            if self.flag_logging:
                logging.info(f"Image saved at {save_path}")
            self.display_contents.append(str(save_path))

            # Closeout the plot and return list of images saved
            self.close_or_show_plot()
            return self.display_contents

        except Exception as e:
            if PRINT_TRACEBACK:
                print(traceback.format_exc())
            if self.flag_logging:
                logging.warning(f"Warning: Image analysis failed due to: {e}")
            return None

    @staticmethod
    def _shift_axis_spine(axis: Axes, plot_number: int) -> None:
        if plot_number < 2:
            return

        if plot_number % 2 == 0:
            side = 'left'
        else:
            side = 'right'
        outward_pixels = int(plot_number/2) * 50

        axis.spines[side].set_position(('outward', outward_pixels))
        axis.spines[side].set_visible(True)
        axis.yaxis.set_ticks_position(side)
        axis.yaxis.set_label_position(side)

    # # # # # # Routine for Noscan Analysis # # # # # #

    def run_noscan_analysis(self) -> None:
        fig, ax1 = plt.subplots(figsize=(7, 5))

        counter = 0
        for param in self.parameters:
            if counter > 0:
                new_ax = ax1.twinx()
            else:
                new_ax = ax1
            self._generate_noscan_plot(axis=new_ax, plot_parameter = param)
            self._shift_axis_spine(axis=new_ax, plot_number=counter)
            counter += 1

        ax1.set_xlabel("Shotnumber")

    def _generate_noscan_plot(self, axis: Axes, plot_parameter: PlotParameter):
        data_array = self.auxiliary_data[plot_parameter.key_name].values
        shotnumber = self.auxiliary_data['Shotnumber'].values

        average = np.average(data_array)
        sigma = np.std(data_array)
        if self.stat_type == 'median':
            center = np.median(data_array)
        else:
            center = average

        c = plot_parameter.color
        axis.scatter(shotnumber, data_array, c=c, label=plot_parameter.legend_label)

        axis.hlines(y=center, colors=c, linestyles='solid', xmin=shotnumber[0], xmax=shotnumber[-1])
        axis.hlines(y=average-sigma, colors=c, linestyles='dashed', xmin=shotnumber[0], xmax=shotnumber[-1])
        axis.hlines(y=average+sigma, colors=c, linestyles='dashed', xmin=shotnumber[0], xmax=shotnumber[-1])

        axis.set_ylabel(plot_parameter.axis_label)
        axis.tick_params(axis='y', labelcolor=c)

    # # # # # # Routine for Scan Analysis # # # # # #

    def run_scan_analysis(self) -> None:
        fig, ax1 = plt.subplots(figsize=(7, 5))

        counter = 0
        for param in self.parameters:
            if counter > 0:
                new_ax = ax1.twinx()
            else:
                new_ax = ax1
            self._generate_scan_plot(axis=new_ax, plot_parameter = param)
            self._shift_axis_spine(axis=new_ax, plot_number=counter)
            counter += 1

        ax1.set_xlabel(self.scan_parameter)

    def _generate_scan_plot(self, axis: Axes, plot_parameter:PlotParameter):
        binned_data = self.process_to_bins(plot_parameter.key_name)

        c = plot_parameter.color
        axis.scatter(binned_data['bin'], binned_data[self.stat_type], c=c, s=10)
        axis.plot(binned_data['bin'], binned_data[self.stat_type], c=c, ls='-', label=plot_parameter.legend_label)
        axis.errorbar(binned_data['bin'], binned_data['average'], yerr=binned_data['sigma'], c=c, ls='none', capsize=3)

        axis.set_ylabel(plot_parameter.axis_label)
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
