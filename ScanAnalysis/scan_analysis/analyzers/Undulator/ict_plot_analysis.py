from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geecs_data_utils import ScanTag

from scan_analysis.analyzers.common.scatter_plotter_analysis import ScatterPlotterAnalysis


class ICTPlotAnalysis(ScatterPlotterAnalysis):
    def __init__(self, scan_tag: ScanTag, use_median: bool = True, skip_plt_show: bool = True, flag_logging: bool = False):
        super().__init__(scan_tag=scan_tag,
                         device_name='U_BCaveICT',
                         scatter_plot=False,
                         use_median=use_median,
                         data_key_1='U_BCaveICT Python Results.ChA Alias:U_BCaveICT Charge pC',
                         label_1='BCave ICT Charge',
                         ylabel_1='Charge (pC)',
                         skip_plt_show=skip_plt_show,
                         flag_logging=flag_logging)


if __name__ == "__main__":
    from geecs_data_utils import ScanData

    tag = ScanData.get_scan_tag(year=2025, month=7, day=3, number=1, experiment='Undulator')
    analyzer = ICTPlotAnalysis(scan_tag=tag, skip_plt_show=False, flag_logging=True, use_median=True)
    analyzer.run_analysis()
