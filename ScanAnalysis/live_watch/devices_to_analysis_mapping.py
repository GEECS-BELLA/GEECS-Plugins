"""
Here we map analysis classes to combinations of cameras
"""
from pathlib import Path
from typing import Union, List

undulator_scan_analyzers = {
    'MagSpec': {'AND': ['U_BCaveICT', 'U_BCaveMagSpec']},
    'VISAEBeam': {'OR': ['UC_VisaEBeam1', 'UC_VisaEBeam2', 'UC_VisaEBeam3', 'UC_VisaEBeam4',
                         'UC_VisaEBeam5', 'UC_VisaEBeam6', 'UC_VisaEBeam7', 'UC_VisaEBeam8']},
    'Rad2Spec': {'AND': ['U_BCaveICT', 'UC_UndulatorRad2',
                         {'OR': ['UC_VisaEBeam1', 'UC_VisaEBeam2', 'UC_VisaEBeam3', 'UC_VisaEBeam4',
                                 'UC_VisaEBeam5', 'UC_VisaEBeam6', 'UC_VisaEBeam7', 'UC_VisaEBeam8']}]},
    'Aline3': {'UC_ALineEBeam3'}
}


def check_for_analysis_match(scan_folder: Union[Path, str]) -> List[str]:
    """
    Checks list of potential analyzers against what is actually saved for a given scan

    :param scan_folder: scan data folder
    :return: List of all analyses that can be performed
    """
    scan_folder = Path(scan_folder)
    saved_devices = get_available_directories(scan_folder)
    valid_analyzers = []
    for analyzer in undulator_scan_analyzers:
        if evaluate_condition(undulator_scan_analyzers[analyzer], saved_devices):
            valid_analyzers.append(analyzer)
    return valid_analyzers


def evaluate_condition(condition, saved_devices):
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
        return all(dir in saved_devices for dir in condition)


def get_available_directories(root):
    try:
        return [d.name for d in root.iterdir() if d.is_dir()]
    except FileNotFoundError:
        print(f"Folder '{root}' does not exist.")
        return []


if __name__ == '__main__':
    print("Scan 21 should just be the VISA, Rad2")
    folder = "Z:\\data\\Undulator\\Y2024\\11-Nov\\24_1105\\scans\\Scan021"
    results = check_for_analysis_match(folder)
    print(results)

    print("Scan 7 should just be the Mag Spec")
    folder = "Z:\\data\\Undulator\\Y2024\\11-Nov\\24_1105\\scans\\Scan007"
    results = check_for_analysis_match(folder)
    print(results)

    print("Scan 12 should just be the ALine3")
    folder = "Z:\\data\\Undulator\\Y2024\\11-Nov\\24_1105\\scans\\Scan012"
    results = check_for_analysis_match(folder)
    print(results)

    print("Previously using Master Control, nearly all would be flagged every time")
    folder = "Z:\\data\\Undulator\\Y2024\\06-Jun\\24_0606\\scans\\Scan003"
    results = check_for_analysis_match(folder)
    print(results)