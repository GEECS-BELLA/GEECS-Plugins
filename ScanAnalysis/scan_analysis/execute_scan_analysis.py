"""
Module containing the mapping of specific analyzers to their respective classes.  Gives an analyze command to the
specified analyzer with the scan folder location
"""

# TODO Add in type hinting for the analyzers

from typing import List, NamedTuple, Type, Optional
from .base import ScanAnalysis
from geecs_python_api.controls.api_defs import ScanTag

from scan_analysis.analyzers.Undulator.CameraImageAnalysis import CameraImageAnalysis
from scan_analysis.analyzers.Undulator.MagSpecStitcherAnalysis import MagSpecStitcherAnalysis
from scan_analysis.analyzers.Undulator.VisaEBeamAnalysis import VisaEBeamAnalysis


class AnalyzerSetup(NamedTuple):
    analyzer: Type[ScanAnalysis]
    device_name: Optional[str] = None
    config_file: Optional[str] = None


ANALYSIS_DICT = {
    'MagSpec': AnalyzerSetup(analyzer=MagSpecStitcherAnalysis),
    'VISAEBeam': AnalyzerSetup(analyzer=VisaEBeamAnalysis),
    'Aline3': AnalyzerSetup(analyzer=CameraImageAnalysis, device_name='UC_ALineEBeam3')
}


def analyze_scan(experiment_name: str, tag: ScanTag, analyzer_list: List[str]):
    for analyzer_name in analyzer_list:
        if analyzer_name not in ANALYSIS_DICT:
            print(f"Error:  '{analyzer_name}' not defined in analysis_dict within execute_scan_analysis.py")
        else:
            analyzer_class = ANALYSIS_DICT.get(analyzer_name).analyzer
            analyzer = analyzer_class()


if __name__ == '__main__':
    print("test")
    # Given scan tag and string for analysis:

    # Convert string to analysis class

    # Call analysis with scan tag
