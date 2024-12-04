"""
This was originally created by Reinier van Mourik LiveImageProcessing, later copied by Chris Doss to test for use with
ScanAnalysis.  If this remains the case, then TODO should make a shared version in geecs-python-api
"""
from __future__ import annotations

import time
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Optional, Union
import re
from queue import Queue

from geecs_python_api.analysis.scans.scan_data import ScanData
from scan_analysis.execute_scan_analysis import analyze_scan
from devices_to_analysis_mapping import check_for_analysis_match

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

if TYPE_CHECKING:
    from watchdog.events import DirCreatedEvent, FileCreatedEvent
    from geecs_python_api.controls.api_defs import ScanTag

import logging.config
import json
import yaml

print(Path(__file__).parents[1] / "logging_config.json")

logging.config.dictConfig(
    json.load(
        (Path(__file__).parents[1] / "logging_config.json")
        .open()
    )
)
logger = logging.getLogger("scan_analyzer")


def extract_s_file_number(filename: str) -> Optional[int]:
    """ Tries to match a given file to a valid s-file, if applicable """
    s_filename_regex = re.compile(r"s(?P<scan_number>\d+).txt")
    m = s_filename_regex.match(filename)
    if m is None:
        return None
    else:
        return int(m['scan_number'])


class AnalysisFolderEventHandler(FileSystemEventHandler):
    def __init__(self, scan_watch_queue: 'Queue', base_tag: 'ScanTag'):
        super().__init__()
        self.queue: Queue = scan_watch_queue
        self.base_tag = base_tag

    def on_created(self, event: Union[DirCreatedEvent, FileCreatedEvent]):
        """ If the event is a new file and a valid s-file, add the scan tag to the queue """
        # ignore new directories
        if event.is_directory:
            return

        date_folder_name, analysis_literal, filename = Path(event.src_path).parts[-3:]
        assert analysis_literal == 'analysis'
        scan_number = extract_s_file_number(filename)
        # ignore anything other than sXX.txt files
        if scan_number is None:
            return

        logger.info(f"Found new s-file: {filename}")
        tag = ScanData.get_scan_tag(self.base_tag.year, self.base_tag.month, self.base_tag.day,
                                    scan_number, experiment_name=self.base_tag.experiment)
        self.queue.put(tag)


class ScanWatch:
    def __init__(self, experiment_name: str, year: int, month: Union[int, str], day: int,
                 ignore_list: list[int] = None, overwrite_previous: bool = False, perform_initial_search: bool = True):
        """
        Parameters
        ----------
        experiment_name : str
            A GEECS experiment name, IE 'Undulator'
        year : int
            Target directory year
        month : Union[int, str]
            Target directory month
        day: int
            Target directory day
        ignore_list: list[int]
            List of scan numbers to disregard for the given day
        overwrite_previous: bool
            Flag to ignore previously analyzed scan directories if True
        perform_initial_search: bool
            Flag to search through the target directory for existing scans to analyze

        """
        self.tag = ScanData.get_scan_tag(year, month, day, number=0, experiment_name=experiment_name)
        self.watch_folder = ScanData.build_scan_folder_path(tag=self.tag).parents[1] / "analysis"

        self.analysis_queue = Queue()

        self.processed_list_filename = Path(f"./processed_scans_{experiment_name}.yaml")
        self.processed_list = []
        if ignore_list is not None:
            self.processed_list = ignore_list
        if not overwrite_previous:
            self._read_processed_list()

        self.observer = PollingObserver()
        self.observer.schedule(AnalysisFolderEventHandler(self.analysis_queue, self.tag), str(self.watch_folder))

        # Initial check of scan folder
        if perform_initial_search:
            self.initial_search_of_watch_folder()

    def _check_watch_folder_exists(self, watch_folder_not_exist: str = 'raise'):
        """
        Check if the watch folder exists, and take action if it doesn't.
        
        Parameters
        ----------
        watch_folder_not_exist : str
            what to do if the watch folder, <date_folder_name>/analysis, does not exist. (observer requires
            an existing folder)
            
            one of the following:
                'wait': wait for the watch folder to be created by the control system
                'create': create the folder ourselves
                'raise': raise FileNotFoundError
        
        """
        if not self.watch_folder.exists():
            if watch_folder_not_exist == 'raise':
                raise FileNotFoundError(f"Watch folder {self.watch_folder} does not exist. "
                                        "Either create it, or set watch_folder_not_exist "
                                        "to 'wait' or 'create'"
                                        )

            elif watch_folder_not_exist == 'create':
                logger.info(f"Creating {self.watch_folder}.")
                self.watch_folder.mkdir(parents=True)

            elif watch_folder_not_exist == 'wait':
                logger.info(f"Waiting for {self.watch_folder} to be created.")
                while not self.watch_folder.exists():
                    sleep(10)
            else:
                raise ValueError(f"Unknown value for watch_folder_not_exist: {watch_folder_not_exist}")

    def start(self, watch_folder_not_exist: str = 'raise'):
        """
        Parameters
        ----------
        watch_folder_not_exist : str
            what to do if the watch folder, <date_folder_name>/analysis, does not exist. (observer requires
            an existing folder)
            
            one of the following:
                'wait': wait for the watch folder to be created by the control system
                'create': create the folder ourselves
                'raise': raise FileNotFoundError
        
        """

        self._check_watch_folder_exists(watch_folder_not_exist)

        self.observer.start()
        logger.info(f"Started watching folder {self.watch_folder} for sXX.txt files.")

    def stop(self):
        """ Stop the polling thread """
        self.observer.stop()
        self.observer.join()

    def process_queue(self):
        """ Grab the next scan tag from the queue and attempt analysis """
        if self.analysis_queue.empty():
            return

        tag: ScanTag = self.analysis_queue.get()
        self.processed_list.append(tag.number)
        scan_folder = ScanData.build_scan_folder_path(tag=tag)
        self._evaluate_folder(tag, scan_folder)

    def initial_search_of_watch_folder(self):
        """ Look for existing scans in the watch folder that have not been previously analyzed """
        logger.info(f"Searching for untouched scans on {self.tag.month}/{self.tag.day}/{self.tag.year}:")
        valid_scan_numbers = []
        if self.watch_folder.exists():
            for item in self.watch_folder.iterdir():
                if item.is_file():
                    scan_number = extract_s_file_number(str(item.name))
                    if scan_number is not None and scan_number not in self.processed_list:
                        valid_scan_numbers.append(scan_number)
        sorted_scans = sorted(valid_scan_numbers)
        for scan_number in sorted_scans:
            tag = ScanData.get_scan_tag(self.tag.year, self.tag.month, self.tag.day, scan_number,
                                        experiment_name=self.tag.experiment)
            self.analysis_queue.put(tag)
        logger.info(f"Found {self.analysis_queue.qsize()} untouched scans.")

    def _evaluate_folder(self, tag: 'ScanTag', scan_folder: Path):
        """ If there is a match for an analysis routine, perform the respective analysis(es) """
        logger.info(
            f"Starting analysis on scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}")
        valid_analyzers = check_for_analysis_match(scan_folder)

        try:
            analyze_scan(tag, valid_analyzers, debug_mode=False)
        except Exception as err:
            logger.error(f"Error in analyze_scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}): {err}")

        self._write_processed_list()

        logger.info(f"Finished analysis on scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}")

    def _read_processed_list(self):
        """ Add all scans numbers found in the yaml to the processed list """
        contents = self._read_yaml_file()

        year_data = contents.get(str(self.tag.year)) if contents is not None else None
        month_data = year_data.get(str(self.tag.month)) if year_data is not None else None
        day_data = month_data.get(str(self.tag.day)) if month_data is not None else None

        if day_data is not None:
            for scan in day_data:
                self.processed_list.append(scan)

    def _read_yaml_file(self) -> dict:
        """ Read yaml file for the dictionary """
        contents = None
        if self.processed_list_filename.exists():
            with open(self.processed_list_filename, 'r') as file:
                contents = yaml.safe_load(file) or []
        return contents

    def _write_processed_list(self):
        """ Update yaml file with the new processed list """
        data = self._read_yaml_file()
        new_contents = {str(self.tag.year): {str(self.tag.month): {str(self.tag.day): self.processed_list}}}
        if data is None:
            data = new_contents
        else:
            data = recursive_update(data, new_contents)

        with open(self.processed_list_filename, 'w') as file:
            yaml.safe_dump(data, file)


def recursive_update(base: dict, new: dict) -> dict:
    """ Recursively update a dictionary with new information """
    for key, value in new.items():
        if isinstance(value, dict):
            base[key] = recursive_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base


if __name__ == '__main__':
    """ap = ArgumentParser()
    ap.add_argument("--watch_folder_not_exist",
                    type=str,
                    choices=['wait', 'create', 'raise'],
                    default='raise',
                    help=("What to do if date_folder/analysis does not exist. "
                          "'wait': wait for the watch folder to be created by the control system; "
                          "'create': create the folder ourselves; "
                          "'raise': raise FileNotFoundError; "
                          )
                    )"""

    exp = 'Undulator'
    test_year = 2024
    test_month = 11
    test_day = 26

    scan_watch = ScanWatch(experiment_name=exp, year=test_year, month=test_month, day=test_day)
    print("Starting...")
    scan_watch.start(watch_folder_not_exist='wait')

    while True:
        time.sleep(0.5)
        scan_watch.process_queue()
        pass
