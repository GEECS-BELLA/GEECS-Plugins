"""
ScatterPlotterAnalysis module.

Provides a generic analyzer for loading and plotting scalar data from auxiliary sfiles.
This module defines `ScatterPlotterAnalysis`, which extends `ScanAnalyzer` to create
two-parameter-style plots that integrate with the ScanAnalyzer framework and GUI. It is
intended to be subclassed by analyzers that specify the particular data to be plotted.

The `PlotParameter` named tuple collects the metadata required for each plotted parameter,
including the sfile key, axis label, legend label, and color. One or more `PlotParameter`
instances configure the plots.

Behavior differs by scan type:
- **1D scan**: Data are binned by the scanned parameter. The median (or mean) is plotted
  per bin, and error bars indicate ±1 standard deviation about the mean.
- **Noscan**: Each parameter is plotted as a scatter versus shot number. A solid line
  marks the representative statistic (median or mean); dashed lines indicate mean ±1σ.

Classes
-------
PlotParameter
    Named tuple specifying how a scalar quantity is plotted.
ScatterPlotterAnalysis
    Analyzer that generates plots for scalar data across scans or noscans.

Examples
--------
For a usage example, see:
`analyzers/Undulator/ict_plot_analysis.py`

Notes
-----
Future work may allow child classes to choose whether the scanned parameter appears on
the x-axis or whether scatter plots should relate two arbitrary scalars, which could
complicate binning logic for 1D scans.
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
    Plot configuration for a scalar quantity.

    Parameters
    ----------
    key_name : str
        Key name in the sfile from which to load the data for this plot.
    legend_label : str
        Text to use in the legend for this parameter.
    axis_label : str
        Label to use for the y-axis corresponding to this parameter.
    color : str
        Matplotlib-compatible color specifier for plotting this parameter.

    Notes
    -----
    The legend label and axis label are intentionally separate to support concise legends
    while preserving detailed axis labeling.
    """

    key_name: str  # Key name from the sfile to load for the given plot
    legend_label: str  # String to appear on plot's y-axis for given parameter
    axis_label: str  # String to appear on plot's legend for given parameter
    color: str  # String that determines the color for the plot (see matplotlib)


class ScatterPlotterAnalysis(ScanAnalyzer):
    """
    Analyzer for plotting scalar data from sfiles across scans or noscans.

    This class extends `ScanAnalyzer` and produces parameter plots either by binning a
    1D scan or by plotting per-shot scatter for noscans. The representative statistic
    for each parameter can be the median or the mean.

    Parameters
    ----------
    use_median : bool
        If ``True``, plot the median as the representative statistic; otherwise, plot
        the mean. The mean is still used to compute error bars (±1 standard deviation)
        in 1D scans.
    title : str
        Title to display at the top of the final figure.
    parameters : PlotParameter or list of PlotParameter
        One or more plot specifications describing which scalars to plot and how.
    filename : str
        Base filename (without extension) for the exported figure (PNG).
    skip_plt_show : bool, optional
        If ``True``, avoid calling ``plt.show()``; typically used in headless or batch
        environments. Default is ``True``.
    device_name : str, optional
        Optional device name used to construct subdirectory paths for outputs.
    flag_logging : bool, optional
        If ``True``, log informative messages, warnings, and errors. Default is ``True``.

    Notes
    -----
    The save path is created under:
    ``<scan_root>/analysis/<scan_directory_name>/ParameterPlots/<filename>.png``.
    """

    def __init__(
        self,
        use_median: bool,
        title: str,
        parameters: Union[PlotParameter, list[PlotParameter]],
        filename: str,
        skip_plt_show: bool = True,
        device_name: Optional[str] = None,
        flag_logging: bool = True,
    ):
        """
        Initialize the `ScatterPlotterAnalysis` instance.

        Parameters
        ----------
        use_median : bool
            If ``True``, the median is used as the representative statistic; otherwise,
            the mean is used.
        title : str
            Figure title to display on the output plot.
        parameters : PlotParameter or list of PlotParameter
            Single plot parameter or a list of parameters to include.
        filename : str
            Output PNG filename stem (without extension).
        skip_plt_show : bool, optional
            If ``True``, do not display the plot interactively. Default is ``True``.
        device_name : str, optional
            Name of the device for constructing subdirectory paths.
        flag_logging : bool, optional
            If ``True``, enable logging for progress and warnings. Default is ``True``.
        """
        super().__init__(skip_plt_show=skip_plt_show, device_name=device_name)

        self.flag_logging = flag_logging
        self.filename = filename
        self.base_title = title
        self.title: str

        if use_median:
            self.stat_type = "median"
        else:
            self.stat_type = "average"

        self.parameters: list[PlotParameter]
        if not isinstance(parameters, list):
            self.parameters = [parameters]
        else:
            self.parameters = parameters

    def _check_parameters(self) -> None:
        """
        Validate that required sfile keys exist for all configured parameters.

        Raises
        ------
        KeyError
            If any `PlotParameter.key_name` is not present in the loaded sfile data.
        """
        found_keys = self.auxiliary_data.keys()

        for param in self.parameters:
            if param.key_name not in found_keys:
                raise KeyError(
                    f"f{param.key_name} not found in sfile for Scan {self.scan_tag.number}"
                )

    def _set_plotting_options(self) -> None:
        """
        Configure plot titles and output paths.

        Notes
        -----
        The plot title is formatted with the scan date and number. The save path is set
        to ``<scan_root>/analysis/<scan_dir>/ParameterPlots``.
        """
        self.title: str = (
            f"{self.scan_tag.month:02d}/{self.scan_tag.day:02d}/{self.scan_tag.year % 1000:02d} "
            f"Scan{self.scan_tag.number:03d}: {self.base_title}"
        )

        self.save_path = (
            self.scan_directory.parents[1]
            / "analysis"
            / self.scan_directory.name
            / "ParameterPlots"
        )

    def _run_analysis_core(self) -> Optional[list[Union[Path, str]]]:
        """
        Execute the analysis and generate the plot.

        This method dispatches to either noscan or scan analysis paths, finalizes plot
        styling, saves the figure, and optionally displays it.

        Returns
        -------
        list of Path or list of str or None
            A list containing the saved figure path(s) as `Path` or `str`. Returns
            ``None`` if analysis fails.

        Notes
        -----
        The figure is saved as a PNG with ``bbox_inches='tight'`` and small padding.
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
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
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
        Offset and position an axis spine to avoid overlap in multi-axis figures.

        Every second axis is placed on alternating sides (left/right) and pushed outward
        by multiples of 50 pixels to improve readability when stacking several twinx axes.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            The axis to offset and place on the left or right.
        plot_number : int
            Zero-based index of the axis in the figure; determines side and offset.

        Notes
        -----
        If ``plot_number < 2``, no offset is applied and the axis remains in place.
        """
        if plot_number < 2:
            return

        if plot_number % 2 == 0:
            side = "left"
        else:
            side = "right"
        outward_pixels = int(plot_number / 2) * 50

        axis.spines[side].set_position(("outward", outward_pixels))
        axis.spines[side].set_visible(True)
        axis.yaxis.set_ticks_position(side)
        axis.yaxis.set_label_position(side)

    # # # # # # Routine for Noscan Analysis # # # # # #

    def run_noscan_analysis(self) -> None:
        """
        Generate per-shot scatter plots for all configured parameters (noscan mode).

        Notes
        -----
        A new twinx axis is created for each additional parameter, and axis spines are
        shifted outward to mitigate overlap. The x-axis is shot number.
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
        Create a scatter plot and reference lines for a parameter in noscan mode.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Axis on which to render the scatter and reference lines.
        plot_parameter : PlotParameter
            Plot specification containing key name, labels, and color.

        Notes
        -----
        The solid horizontal line indicates the representative statistic
        (median if configured, otherwise mean). Dashed lines show mean ±1σ.
        """
        data_array = self.auxiliary_data[plot_parameter.key_name].values
        shotnumber = self.auxiliary_data["Shotnumber"].values

        average = np.average(data_array)
        sigma = np.std(data_array)
        if self.stat_type == "median":
            center = np.median(data_array)
        else:
            center = average

        c = plot_parameter.color
        axis.scatter(shotnumber, data_array, c=c, label=plot_parameter.legend_label)

        axis.hlines(
            y=center,
            colors=c,
            linestyles="solid",
            xmin=shotnumber[0],
            xmax=shotnumber[-1],
        )
        axis.hlines(
            y=average - sigma,
            colors=c,
            linestyles="dashed",
            xmin=shotnumber[0],
            xmax=shotnumber[-1],
        )
        axis.hlines(
            y=average + sigma,
            colors=c,
            linestyles="dashed",
            xmin=shotnumber[0],
            xmax=shotnumber[-1],
        )

        axis.set_ylabel(plot_parameter.axis_label)
        axis.tick_params(axis="y", labelcolor=c)

    # # # # # # Routine for Scan Analysis # # # # # #

    def run_scan_analysis(self) -> None:
        """
        Generate per-bin plots for all configured parameters (1D scan mode).

        Notes
        -----
        Data are grouped by the scanned parameter into unique bins. The representative
        statistic (median or mean) is plotted per bin. Error bars reflect mean ±1σ.
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
        Plot per-bin statistics and error bars for a parameter in 1D scan mode.

        Parameters
        ----------
        axis : matplotlib.axes.Axes
            Axis on which to render the per-bin statistics.
        plot_parameter : PlotParameter
            Plot specification containing key name, labels, and color.

        Notes
        -----
        The representative statistic is controlled by the `use_median` flag passed at
        initialization. Error bars are computed as mean ±1 standard deviation.
        """
        binned_data = self.process_to_bins(plot_parameter.key_name)

        c = plot_parameter.color
        axis.scatter(binned_data["bin"], binned_data[self.stat_type], c=c, s=10)
        axis.plot(
            binned_data["bin"],
            binned_data[self.stat_type],
            c=c,
            ls="-",
            label=plot_parameter.legend_label,
        )
        axis.errorbar(
            binned_data["bin"],
            binned_data["average"],
            yerr=binned_data["sigma"],
            c=c,
            ls="none",
            capsize=3,
        )

        axis.set_ylabel(plot_parameter.axis_label)
        axis.tick_params(axis="y", labelcolor=c)

    def process_to_bins(self, key_name: str) -> dict[str, np.ndarray]:
        """
        Compute per-bin statistics for a parameter across the scanned variable.

        Parameters
        ----------
        key_name : str
            Key in the sfile from which to extract the scalar data to bin.

        Returns
        -------
        dict of str to numpy.ndarray
            Dictionary containing per-bin statistics:
            - ``'bin'``: float array of scanned parameter values for each bin.
            - ``'average'``: float array of mean values per bin.
            - ``'sigma'``: float array of standard deviations per bin.
            - ``'median'``: float array of medians per bin.

        Notes
        -----
        Bins are determined by unique values in ``self.bins``. The corresponding
        scanned parameter values are taken from ``self.binned_param_values``.
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
            shots_in_bin = (
                self.auxiliary_data[self.auxiliary_data["Bin #"] == bin_val][
                    "Shotnumber"
                ].values
                - 1
            )

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

        return {"bin": bins, "average": average, "sigma": sigma, "median": median}
