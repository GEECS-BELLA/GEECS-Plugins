"""
Functions for opening file explorer on the given folders.  TODO make a Mac-compatible version if needed (or throw error)
"""

import subprocess
from pathlib import Path
from geecs_python_api.analysis.scans.scan_data import ScanData, reload_paths_config


def open_folder(path_to_folder: Path):
    subprocess.Popen(f'explorer "{path_to_folder}"')


def open_daily_data_folder(experiment: str):
    latest = ScanData.get_next_scan_folder(experiment=experiment).parent
    latest.mkdir(parents=True, exist_ok=True)
    open_folder(path_to_folder=latest)


def reload_scan_data_paths():
    """ Calls ScanData's function to reset static variables pointing to experiment-specific folders """
    reload_paths_config()
