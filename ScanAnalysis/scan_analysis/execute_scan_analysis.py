"""
Module containing the mapping of specific analyzers to their respective classes.  Gives an analysis command to the
specified analyzer with the scan folder location
"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scan_analysis.base import AnalyzerInfo
    from geecs_python_api.controls.api_defs import ScanTag
    from logmaker_4_googledocs import docgen
import logging


def analyze_scan(tag: 'ScanTag', analyzer_list: list['AnalyzerInfo'], debug_mode: bool = False):
    for analyzer_info in analyzer_list:
        device = analyzer_info.device_name if analyzer_info.device_name else ''
        print(tag, ":", analyzer_info.analyzer_class.__name__, device)
        if not debug_mode:
            try:
                analyzer_class = analyzer_info.analyzer_class
                analyzer = analyzer_class(scan_tag=tag, device_name=analyzer_info.device_name, skip_plt_show=True)
                index_of_files = analyzer.run_analysis(config_options=analyzer_info.config_file)

                # if index_of_files:  # TODO And if a Google doc procedure is defined for the given experiment
                #     # TODO Append the images to the appropriate location in the daily experiment log on Google
                #     pass

            except Exception as err:
                logging.error(f"Error in analyze_scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}): {err}")
