"""
ScatterPlotterAnalysis

Generic analyzer to load and plot scalar data from auxiliary sfiles.
Child to ScanAnalyzer (./scan_analysis/base.py)

Provides a quick framework for 2-parameter-style plots that can be run through
the ScAnalyzer framework and GUI.  Intended to be used by child classes that
specify the data to be plotted.

Parameters are specified using the `PlotParameter` class defined below.  Each
PlotParameter contains a key that points to the scalar data in the sfile, along with
strings for how to label the y-axis, how to label the legend, and what color to
make the line.

In a 1D scan, this analyzer will bin the data and plot the median (or mean)
of each given PlotParameter.  Errorbars for each bin are plotted as +/- 1 std from the
mean.

In a noscan, this analyzer will instead plot a scatter plot of the PlotParameter vs
shot number.  The median is plotted as a solid, horizontal line.  Two dashed lines at
the mean +/- 1 std represent the spread in the data.

For a good example of usage, see `analyzers/Undulator/ict_plot_analysis.py`

FUTURE WORK:
 - Add in an option to allow child classes to specify if they want the scanned parameter
   to be the x-axis, or if scatter plots should instead be between two other scalars.  This
   may make the binning of data in a 1D scan a bit strange, so some thought is needed.

"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional, NamedTuple
if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt

from scan_analysis.base import ScanAnalyzer

import traceback
PRINT_TRACEBACK = True


class PlotParameter(NamedTuple):
    """
    Class that defines all required information to load and plot data from the sfile
    """
    key_name: str  # Key name from the sfile to load for the given plot
    legend_label: str  # String to appear on plot's y-axis for given parameter
    axis_label: str  # String to appear on plot's legend for given parameter
    color: str  # String that determines the color for the plot (see matplotlib)


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
        """
        Initializes class variables for the ScatterPlotter instance

        Args:
            use_median (bool): If true, the median is plotted instead of the mean for a
                representative value in a given scan bin or noscan.  Defaults to True for median
            title (str): The top title for the full final figure
            parameters: Either a single PlotParameter or list of PlotParameter's.  Can be any length
            filename (str): Name of the resulting png file of the exported figure
            skip_plt_show (bool): Flag that sets if matplotlib is tried to use for plotting
            device_name (str): Name of the device to construct the subdirectory path.
            flag_logging (bool): Flag that sets if error and warning messages are displayed
        """
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

    def _check_parameters(self) -> None:
        """
        Checks the keys contained in the scan's sfile against those in the list of parameters for this analyzer

        Raises:
            KeyError: if a given key is not in the sfile for the scan
        """
        found_keys = self.auxiliary_data.keys()

        for param in self.parameters:
            if param.key_name not in found_keys:
                raise KeyError(f"f{param.key_name} not found in sfile for Scan {self.scan_tag.number}")

    def _set_plotting_options(self) -> None:
        """
        Sets plot options before analysis script using the given scan tag to the base analyzer
        """
        self.title: str = (f"{self.scan_tag.month:02d}/{self.scan_tag.day:02d}/{self.scan_tag.year%1000:02d} "
                           f"Scan{self.scan_tag.number:03d}: {self.base_title}")

        self.save_path = self.scan_directory.parents[1] / 'analysis' / self.scan_directory.name / "ParameterPlots"

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        """
        Core analysis script.  Calls a different routine depending on if the scan is a noscan or not, and
        after the routine this method packages up the plot, saves to the analysis directory, and optionally
        displays the plot in matplotlib

        :return: Optional - the filename for the saved figure
        """
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
        """
        Assigns an axis to be either on the left or right side of a figure, and shifts this
        axis outwards from the figure.  The orientation and shift is calculated by using the
        counter for this specific axis.

        IE:  Every other axis uses the left-hand side of the figure.  Axes 3 and 4 are shifted
        50 pixels outwards (on the left and right side, respectively).  Axes 5 and 6 are shifted
        100 pixels outwards, and so forth.

        Args
            axis (Axes): matplotlib Axes instance
            plot_number (int): sequentially, which plot this Axes corresponds to in the figure
        """
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
        """
        Loops over all PlotParameters and generates a plot for each.  The resulting collection
        of axes are organized together into the full figure.  This method is for noscans
        """
        fig, ax1 = plt.subplots(figsize=(7, 5))

        counter = 0
        for param in self.parameters:
            if counter > 0:
                new_ax = ax1.twinx()
            else:
                new_ax = ax1
            self._generate_noscan_plot(axis=new_ax, plot_parameter=param)
            self._shift_axis_spine(axis=new_ax, plot_number=counter)
            counter += 1

        ax1.set_xlabel("Shotnumber")

    def _generate_noscan_plot(self, axis: Axes, plot_parameter: PlotParameter):
        """
        Calculates statistics for the given PlotParameter and generates a matplotlib plot

        Args:
            axis: matplotlib Axes instance (either generated through "subplots" or "twinx")
            plot_parameter: instance of PlotParameter containing the key name and plot options
        """
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
        """
        Loops over all PlotParams and generates a plot for each.  The resulting collection
        of axes are organized together into the full figure.  This method is for 1D scans
        """
        fig, ax1 = plt.subplots(figsize=(7, 5))

        counter = 0
        for param in self.parameters:
            if counter > 0:
                new_ax = ax1.twinx()
            else:
                new_ax = ax1
            self._generate_scan_plot(axis=new_ax, plot_parameter=param)
            self._shift_axis_spine(axis=new_ax, plot_number=counter)
            counter += 1

        ax1.set_xlabel(self.scan_parameter)

    def _generate_scan_plot(self, axis: Axes, plot_parameter: PlotParameter):
        """
        Bins the data for the given PlotParameter and generates a matplotlib plot using the
        statistics of the bins.  Errorbars are plotted as the mean +/- 1 standard deviation

        Args:
            axis: matplotlib Axes instance (either generated through "subplots" or "twinx")
            plot_parameter: instance of PlotParameter containing the key name and plot options
        """
        binned_data = self.process_to_bins(plot_parameter.key_name)

        c = plot_parameter.color
        axis.scatter(binned_data['bin'], binned_data[self.stat_type], c=c, s=10)
        axis.plot(binned_data['bin'], binned_data[self.stat_type], c=c, ls='-', label=plot_parameter.legend_label)
        axis.errorbar(binned_data['bin'], binned_data['average'], yerr=binned_data['sigma'], c=c, ls='none', capsize=3)

        axis.set_ylabel(plot_parameter.axis_label)
        axis.tick_params(axis='y', labelcolor=c)

    def process_to_bins(self, key_name: str) -> dict[str, np.ndarray]:
        """
        Args:
            key_name (str): key name to extract from sfile and bin across the 1D scan variable

        Returns:
            dict containing numpy arrays organized by type:
                'bin' : numpy array with the scan parameter value at each bin
                'average' : mean value for the plot parameter at each bin
                'sigma' : standard deviation for the plot parameter at each bin
                'median' : median for the plot parameter at each bin
        """
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
