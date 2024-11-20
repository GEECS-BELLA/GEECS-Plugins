"""
Module containing the mapping of specific analyzers to their respective classes.  Gives an analyze command to the
specified analyzer with the scan folder location
"""

# TODO Add in type hinting for the analyzers

from scan_analysis.analyzers.Undulator import example_analyzer as Example

ANALYSIS_DICT = {
    'MagSpec': Example,


}


def analyze_scan(analyzer_name, scan_folder):
    if analyzer_name not in ANALYSIS_DICT:
        print(f"Error:  '{analyzer_name}' not defined in analysis_dict within execute_scan_analysis.py")
        return
    analyzer_class = ANALYSIS_DICT.get(analyzer_name)
    analyzer = analyzer_class()

