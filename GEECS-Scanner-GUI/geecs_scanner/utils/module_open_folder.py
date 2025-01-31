"""
Functions for opening file explorer on the given folders.  TODO make a Mac-compatible version if needed (or throw error)
"""

import subprocess
from pathlib import Path
from geecs_python_api.analysis.scans.scan_data import ScanData


def open_folder(path_to_folder: Path):
    """ Opens Windows file explorer to the specified location """
    subprocess.Popen(f'explorer "{path_to_folder}"')


def open_daily_data_folder(experiment: str):
    """ Uses ScanData to find the server's save data location for today, and opens in Windows file explorer """
    latest = ScanData.get_next_scan_folder(experiment=experiment).parent
    latest.mkdir(parents=True, exist_ok=True)
    open_folder(path_to_folder=latest)


def reload_scan_data_paths():
    """ Calls ScanData's function to reset static variables pointing to experiment-specific folders """
    ScanData.reload_paths_config()
