"""
Quick ICT charge plotting analysis.

This module defines `ICTPlotAnalysis`, a thin wrapper around
`ScatterPlotterAnalysis` that produces quick plots of the BCave ICT
(Integrated Current Transformer) charge for each scan. It configures a single
`PlotParameter` for the BCave ICT charge and delegates all plotting, binning,
and statistic selection (mean/median) to the parent analyzer.
"""

from __future__ import annotations
from typing import Optional

from scan_analysis.analyzers.common.scatter_plotter_analysis import (
    ScatterPlotterAnalysis,
    PlotParameter,
)


class ICTPlotAnalysis(ScatterPlotterAnalysis):
    """
    Quick analysis for plotting BCave ICT charge across a scan.

    This analyzer configures a single `PlotParameter` pointing to the BCave ICT
    charge and uses the generic `ScatterPlotterAnalysis` to compute per-bin
    statistics (mean or median) and render a simple line plot with optional
    error bars.

    Parameters
    ----------
    skip_plt_show : bool, default=True
        If True, suppress calls to `matplotlib.pyplot.show()`. Set to False to
        display figures interactively.
    device_name : str or None, default=None
        Device name used to construct the subdirectory path for outputs. If not
        provided, the analyzer uses `'U_BCaveICT'`.
    use_median : bool, default=True
        If True, compute the median per bin; otherwise compute the mean.
    flag_logging : bool, default=False
        If True, enable verbose logging of warnings and errors.

    Notes
    -----
    - The underlying data key is configured as:
      `'U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC'`.
    - The parent `ScatterPlotterAnalysis` handles binning, statistic
      computation, and plotting.

    See Also
    --------
    scan_analysis.analyzers.common.scatter_plotter_analysis.ScatterPlotterAnalysis
        Generic analyzer for line/scatter-style plots from scalar data.
    scan_analysis.analyzers.common.scatter_plotter_analysis.PlotParameter
        Parameter specification for selecting and labeling plotted data.

    Examples
    --------
    Create and run the analysis for a specific scan tag:

    >>> from geecs_data_utils import ScanData
    >>> tag = ScanData.get_scan_tag(year=2025, month=3, day=6, number=8, experiment='Undulator')
    >>> analyzer = ICTPlotAnalysis(skip_plt_show=False, flag_logging=True, use_median=True)
    >>> analyzer.run_analysis(scan_tag=tag)
    """

    def __init__(
        self,
        skip_plt_show: bool = True,
        device_name: Optional[str] = None,
        use_median: bool = True,
        flag_logging: bool = False,
    ) -> None:
        """
        Initialize the ICT analyzer and configure the BCave ICT `PlotParameter`.

        Parameters
        ----------
        skip_plt_show : bool, default=True
            If True, suppress calls to `matplotlib.pyplot.show()`. Set to False
            to display figures interactively.
        device_name : str or None, default=None
            Device name used to construct the subdirectory path for outputs. If
            not provided, the analyzer uses `'U_BCaveICT'`.
        use_median : bool, default=True
            If True, compute the median per bin; otherwise compute the mean.
        flag_logging : bool, default=False
            If True, enable verbose logging of warnings and errors.
        """
        ict_plot_parameter = [
            PlotParameter(
                key_name="U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC",
                color="b",
                legend_label="BCave ICT Charge",
                axis_label="Charge (pC)",
            ),
        ]

        super().__init__(
            device_name=device_name or "U_BCaveICT",
            use_median=use_median,
            title="BCave ICT Charge Value",
            parameters=ict_plot_parameter,
            filename="bcaveict",
            skip_plt_show=skip_plt_show,
            flag_logging=flag_logging,
        )


if __name__ == "__main__":
    from geecs_data_utils import ScanData

    tag = ScanData.get_scan_tag(
        year=2025, month=3, day=6, number=8, experiment="Undulator"
    )
    analyzer = ICTPlotAnalysis(skip_plt_show=False, flag_logging=True, use_median=True)
    analyzer.run_analysis(scan_tag=tag)
