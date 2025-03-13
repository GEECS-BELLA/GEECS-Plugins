"""
Functions for opening file explorer on the given folders.  TODO make a Mac-compatible version if needed (or throw error)
"""
from typing import Optional
import re
import subprocess
import configparser
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


def iterate_scan_numbers(scan_folder: Path):
    """
    :param scan_folder: Daily scan folder path
    :return: a generated list of scan numbers within the folder
    """
    pattern = re.compile(r'^Scan(\d+)$')
    if not scan_folder.exists():
        return
    for p in scan_folder.iterdir():
        if p.is_dir() and (m := pattern.match(p.name)):
            yield int(m.group(1))


def get_latest_scan_number(experiment: str) -> int:
    """ Finds the latest scan number using regular expressions rather than iterating through ScanData scan tags

    :return: int for latest scan number, 0 if no scans that day
    """
    scan_folder = ScanData.get_daily_scan_folder(experiment=experiment)
    return max(iterate_scan_numbers(scan_folder), default=0)


def get_experiment_scanlog_url(experiment: str) -> Optional[str]:
    """
    Maps an experiment name to a routine used to read the current day's experiment log

    :param experiment: The experiment name
    :return: a string pointing to the
    """
    experiment_to_func = {
        'Undulator': get_scanlog_undulator,
    }
    if func := experiment_to_func.get(experiment, None):
        if url := func():
            return url


def get_scanlog_undulator() -> Optional[str]:
    """
    :return: the Google doc id for HTU's most recent daily experimental scan log
    """
    base_folder = Path("Z:\software\control-all-loasis\HTU\Active Version\GEECS-Plugins\LogMaker4GoogleDocs\logmaker_4_googledocs")
    if not base_folder.exists():
        return

    config_file = 'HTUparameters.ini'
    config_path = base_folder / config_file

    # Load the configuration file
    experiment_config = configparser.ConfigParser()
    experiment_config.read(config_path)

    # Verify the loaded sections
    if not experiment_config.sections():
        return

    # Build the url string
    document_id = experiment_config['DEFAULT']['logid']
    url = f"https://docs.google.com/document/d/{document_id}"
    return url


def reload_scan_data_paths():
    """ Calls ScanData's function to reset static variables pointing to experiment-specific folders """
    ScanData.reload_paths_config()
