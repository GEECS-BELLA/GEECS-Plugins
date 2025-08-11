from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geecs_data_utils import ScanTag

from scan_analysis.analyzers.common.scatter_plotter_analysis import ScatterPlotterAnalysis, PlotParameter


class ICTPlotAnalysis(ScatterPlotterAnalysis):
    def __init__(self, scan_tag: ScanTag, use_median: bool = True, skip_plt_show: bool = True, flag_logging: bool = False):

        ict_plot_parameter = [
            PlotParameter(key_name='U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC',
                          color = 'b',
                          legend_label='BCave ICT Charge',
                          axis_label='Charge (pC)'),
        ]

        super().__init__(scan_tag=scan_tag,
                         device_name='U_BCaveICT',
                         use_median=use_median,
                         title='BCave ICT Charge Value',
                         parameters=ict_plot_parameter,
                         skip_plt_show=skip_plt_show,
                         flag_logging=flag_logging)


if __name__ == "__main__":
    from geecs_data_utils import ScanData

    tag = ScanData.get_scan_tag(year=2025, month=3, day=6, number=8, experiment='Undulator')
    analyzer = ICTPlotAnalysis(scan_tag=tag, skip_plt_show=False, flag_logging=True, use_median=True)
    analyzer.run_analysis()
