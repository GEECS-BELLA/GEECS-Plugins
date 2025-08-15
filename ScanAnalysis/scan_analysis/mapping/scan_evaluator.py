from typing import Union
from pathlib import Path
from scan_analysis.base import ScanAnalyzerInfo, ScanAnalyzer
from scan_analysis.execute_scan_analysis import instantiate_scan_analyzer


def check_for_analysis_match(scan_folder: Union[Path, str], analyzer_list: list[ScanAnalyzerInfo]) -> list[ScanAnalyzer]:
    """
    Checks list of potential analyzers against what is actually saved for a given scan

    :param scan_folder: scan data folder
    :param analyzer_list: list of analyzers, as defined in
    :return: List of all analyses that can be performed
    """
    scan_folder = Path(scan_folder)

    saved_devices = get_available_directories(scan_folder)
    valid_analyzers = []
    for analyzer_info in analyzer_list:
        if evaluate_condition(analyzer_info.requirements, saved_devices) and analyzer_info.is_active:
            valid_analyzers.append(instantiate_scan_analyzer(analyzer_info))
    return valid_analyzers


def evaluate_condition(condition: Union[dict[str, list], set, str], saved_devices: list[str]) -> bool:
    """
    Recursive function to evaluate the conditions defined in `undulator_scan_analyzers` for the given list of devices

    :param condition: either a dict of 'AND/OR' with a list of requirements, or just a single name. TODO check type hint
    :param saved_devices: list of saved devices
    :return: True if the list of saved devices matches the requirements in the given condition
    """
    if 'AND' in condition:
        and_conditions = condition['AND']
        return all(
            evaluate_condition(cond, saved_devices) if isinstance(cond, dict) else cond in saved_devices
            for cond in and_conditions)
    elif 'OR' in condition:
        or_conditions = condition['OR']
        return any(
            evaluate_condition(cond, saved_devices) if isinstance(cond, dict) else cond in saved_devices
            for cond in or_conditions)
    else:
        return all(directory in saved_devices for directory in condition)


def get_available_directories(root: Path) -> list[str]:
    """
    :param root:  Path of scan folder
    :return: All folder names in root path, which correspond to saved devices for that scan
    """
    try:
        return [d.name for d in root.iterdir() if d.is_dir()]
    except FileNotFoundError:
        print(f"Folder '{root}' does not exist.")
        return []
