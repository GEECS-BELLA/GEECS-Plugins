"""
Functions for opening file explorer on the given folders.  TODO make a Mac-compatible version if needed (or throw error)
"""
import re
import subprocess
from pathlib import Path
from geecs_python_api.analysis.scans.scan_data import ScanData


def open_folder(path_to_folder: Path):
    """ Opens Windows file explorer to the specified location """
    subprocess.Popen(f'explorer "{path_to_folder}"')


def open_daily_data_folder(experiment: str):
    """ Uses ScanData to find the server's save data location for today, and opens in Windows file explorer """
    latest = ScanData.get_daily_scan_folder(experiment=experiment)
    latest.mkdir(parents=True, exist_ok=True)
    open_folder(path_to_folder=latest)


def get_latest_scan_number(experiment: str) -> int:
    """ Finds the latest scan number using regular expressions rather than iterating through ScanData scan tags

    :return: int for latest scan number, 0 if no scans that day
    """
    scan_folder = ScanData.get_daily_scan_folder(experiment=experiment)
    pattern = re.compile(r'^Scan(\d+)$')

    max_scan_folder: Path = max(filter(lambda p: pattern.match(p.name), scan_folder.iterdir()), default=None)
    if max_scan_folder:
        return int(pattern.match(max_scan_folder.name).group(1))
    else:
        return 0


def reload_scan_data_paths():
    """ Calls ScanData's function to reset static variables pointing to experiment-specific folders """
    ScanData.reload_paths_config()
