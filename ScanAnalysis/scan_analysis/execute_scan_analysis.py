"""
Module containing the mapping of specific analyzers to their respective classes.  Gives an analysis command to the
specified analyzer with the scan folder location
"""
from typing import List, NamedTuple, Type, Optional
from scan_analysis.base import ScanAnalysis
from geecs_python_api.controls.api_defs import ScanTag

from scan_analysis.analyzers.Undulator.CameraImageAnalysis import CameraImageAnalysis
from scan_analysis.analyzers.Undulator.MagSpecStitcherAnalysis import MagSpecStitcherAnalysis
from scan_analysis.analyzers.Undulator.VisaEBeamAnalysis import VisaEBeamAnalysis


class AnalyzerSetup(NamedTuple):
    analyzer: Type[ScanAnalysis]
    device_name: Optional[str] = None
    config_file: Optional[str] = None


ANALYSIS_DICT = {
    'MagSpec': AnalyzerSetup(analyzer=MagSpecStitcherAnalysis, device_name='U_BCaveMagSpec'),
    'VISAEBeam': AnalyzerSetup(analyzer=VisaEBeamAnalysis),
    'Aline3': AnalyzerSetup(analyzer=CameraImageAnalysis, device_name='UC_ALineEBeam3')
}


def analyze_scan(tag: ScanTag, analyzer_list: List[str], debug_mode: bool = False):
    for analyzer_name in analyzer_list:
        if analyzer_name not in ANALYSIS_DICT:
            print(f"Error:  '{analyzer_name}' not defined in analysis_dict within execute_scan_analysis.py")
        else:
            print(tag, ":", analyzer_name)
            if not debug_mode:
                analyzer_info = ANALYSIS_DICT.get(analyzer_name)
                analyzer_class = analyzer_info.analyzer
                analyzer = analyzer_class(scan_tag=tag, device_name=analyzer_info.device_name, use_gui=True)
                index_of_files = analyzer.run_analysis(config_options=analyzer_info.config_file)

                if index_of_files:  # TODO And if a Google doc procedure is defined for the given experiment
                    # TODO Append the images to the appropriate location in the daily experiment log on Google
                    pass


if __name__ == '__main__':
    # Given scan tag and string for analysis:
    test_tag = ScanTag(year=2024, month=11, day=5, number=5, experiment='Undulator')
    test_analyzer = 'MagSpec'

    # Convert string to analysis class and call analysis with scan tag
    analyze_scan(test_tag, [test_analyzer])
    print("Done with MagSpec")

    ######################

    test_tag = ScanTag(year=2024, month=11, day=26, number=19, experiment='Undulator')
    test_analyzer = 'VISAEBeam'

    # Convert string to analysis class and call analysis with scan tag
    analyze_scan(test_tag, [test_analyzer])
    print("Done with Visa")
