"""
ICTPlotAnalysis

This utilizes the generic ScatterPlotterAnalysis to make quick plots of the BCave ICT charge for each scan.
"""

from __future__ import annotations
from typing import Optional

from scan_analysis.analyzers.common.scatter_plotter_analysis import ScatterPlotterAnalysis, PlotParameter


class ICTPlotAnalysis(ScatterPlotterAnalysis):
    def __init__(self, skip_plt_show: bool = True, device_name: Optional[str] = None,
                 use_median: bool = True,  flag_logging: bool = False):
        """
        Initializes settings and the PlotParameter for the parent ScatterPlotterAnalysis

        Args:
            skip_plt_show (bool): Flag that sets if matplotlib is tried to use for plotting.  Defaults True
            device_name (str): Name of the device to construct the subdirectory path.
            use_median (bool): If true, uses the median instead of the mean.  Defaults to True
            flag_logging (bool): Flag that sets if error and warning messages are displayed
        """

        ict_plot_parameter = [
            PlotParameter(key_name='U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC',
                          color='b',
                          legend_label='BCave ICT Charge',
                          axis_label='Charge (pC)'),
        ]

        super().__init__(device_name='U_BCaveICT',
                         use_median=use_median,
                         title='BCave ICT Charge Value',
                         parameters=ict_plot_parameter,
                         filename='bcaveict',
                         skip_plt_show=skip_plt_show,
                         flag_logging=flag_logging)


if __name__ == "__main__":
    from geecs_data_utils import ScanData

    tag = ScanData.get_scan_tag(year=2025, month=3, day=6, number=8, experiment='Undulator')
    analyzer = ICTPlotAnalysis(skip_plt_show=False, flag_logging=True, use_median=True)
    analyzer.run_analysis(scan_tag=tag)
