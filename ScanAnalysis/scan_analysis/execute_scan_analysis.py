"""
Module containing the mapping of specific analyzers to their respective classes.  Gives an analysis command to the
specified analyzer with the scan folder location
"""
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from scan_analysis.base import AnalyzerInfo
    from geecs_python_api.controls.api_defs import ScanTag


def analyze_scan(tag: 'ScanTag', analyzer_list: list['AnalyzerInfo'], debug_mode: bool = False):
    for analyzer_info in analyzer_list:
        device = analyzer_info.device_name if analyzer_info.device_name else ''
        print(tag, ":", analyzer_info.analyzer_class.__name__, device)
        if not debug_mode:
            analyzer_class = analyzer_info.analyzer_class
            analyzer = analyzer_class(scan_tag=tag, device_name=analyzer_info.device_name, use_gui=True)
            index_of_files = analyzer.run_analysis(config_options=analyzer_info.config_file)

            if index_of_files:  # TODO And if a Google doc procedure is defined for the given experiment
                # TODO Append the images to the appropriate location in the daily experiment log on Google
                pass


if __name__ == '__main__':  # TODO move to a unit test
    from geecs_python_api.controls.api_defs import ScanTag
    from scan_analysis.mapping.map_Undulator import undulator_analyzers

    # Given scan tag and string for analysis:
    test_tag = ScanTag(year=2024, month=11, day=5, number=5, experiment='Undulator')
    test_analyzer = undulator_analyzers[0]  # MagSpec

    # Convert string to analysis class and call analysis with scan tag
    analyze_scan(test_tag, [test_analyzer])
    print("Done with MagSpec")

    ######################

    test_tag = ScanTag(year=2024, month=11, day=26, number=19, experiment='Undulator')
    test_analyzer = undulator_analyzers[1]  # Visa Undulator

    # Convert string to analysis class and call analysis with scan tag
    analyze_scan(test_tag, [test_analyzer])
    print("Done with Visa")
