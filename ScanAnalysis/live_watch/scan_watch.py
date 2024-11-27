"""
This was originally created by Reinier van Mourik LiveImageProcessing, later copied by Chris Doss to test for use with
ScanAnalysis.  If this remains the case, then TODO should make a shared version in geecs-python-api
"""
from __future__ import annotations

import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Optional, Union
import re
from multiprocessing import Process
from queue import Queue

from geecs_python_api.analysis.scans.scan_data import ScanData
from scan_analysis.execute_scan_analysis import test_command
from devices_to_analysis_mapping import check_for_analysis_match

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

if TYPE_CHECKING:
    from watchdog.events import DirCreatedEvent, FileCreatedEvent
    from geecs_python_api.controls.api_defs import ScanTag

import logging
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


class AnalysisFolderEventHandler(FileSystemEventHandler):
    s_filename_regex = re.compile(r"s(?P<scan_number>\d+).txt")

    def __init__(self, scan_watch_queue):
        super().__init__()
        self.queue: Queue = scan_watch_queue

    def on_created(self, event: Union[DirCreatedEvent, FileCreatedEvent]):
        # ignore new directories
        if event.is_directory:
            return

        date_folder_name, analysis_literal, filename = Path(event.src_path).parts[-3:]
        assert analysis_literal == 'analysis'
        m = self.s_filename_regex.match(filename)
        # ignore anything other than sXX.txt files
        if m is None:
            return

        logger.info(f"Found new s-file: {filename}")
        scan_number = int(m['scan_number'])
        tag = ScanData.get_scan_tag(scan_watch.tag.year, scan_watch.tag.month, scan_watch.tag.day, scan_number)
        self.queue.put(tag)


class ScanWatch:
    def __init__(self, experiment_name: str, year, month, day,
                 ignore_list=None, overwrite_previous=False):
        """
        Parameters
        ----------
        date_folder_name : str
            a HTU date folder name in the format 'yy_mmdd'
        scan_analyzer : Optional[ScanAnalyzer]
            use an existing ScanAnalyzer instance, or (default) create a new one.
        """
        self.experiment_name = experiment_name
        self.tag = ScanData.get_scan_tag(year, month, day, number=0)
        self.watch_folder = ScanData.build_scan_folder_path(tag=self.tag,
                                                            experiment=experiment_name).parents[1] / "analysis"

        self.analysis_queue = Queue()

        self.processed_list_filename = Path(f"./processed_scans_{experiment_name}.yaml")
        self.processed_list = []
        if ignore_list is not None:
            self.processed_list = ignore_list
        if not overwrite_previous:
            self.read_processed_list()

        self.observer = PollingObserver()
        self.observer.schedule(AnalysisFolderEventHandler(self.analysis_queue), str(self.watch_folder))

        # Initial check of scan folder
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
        self.observer.stop()
        self.observer.join()

    def process_queue(self):
        if self.analysis_queue.empty():
            return

        tag: ScanTag = self.analysis_queue.get()
        self.processed_list.append(tag.number)
        scan_folder = ScanData.build_scan_folder_path(tag=tag, experiment=self.experiment_name)
        self.evaluate_folder(tag, scan_folder)

    def initial_search_of_watch_folder(self):
        logger.info(f"Searching for untouched scans on {self.tag.month}/{self.tag.day}/{self.tag.year}:")
        s_filename_regex = re.compile(r"s(?P<scan_number>\d+).txt")
        if self.watch_folder.exists():
            for item in self.watch_folder.iterdir():
                if item.is_file():
                    m = s_filename_regex.match(str(item.name))
                    if m is not None:
                        scan_number = int(m['scan_number'])
                        if scan_number not in self.processed_list:
                            tag = ScanData.get_scan_tag(self.tag.year, self.tag.month,
                                                        self.tag.day, scan_number)
                            self.analysis_queue.put(tag)
        logger.info(f"Found {self.analysis_queue.qsize()} untouched scans.")

    def evaluate_folder(self, tag: ScanTag, scan_folder):
        logger.info(
            f"Starting analysis on scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}")
        valid_analyzers = check_for_analysis_match(scan_folder)

        try:
            test_command(tag, valid_analyzers)
        except Exception as err:
            logger.error(f"Error in analyze_scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}): {err}")

        self.write_processed_list()

        logger.info(f"Finished analysis on scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}")

    def read_processed_list(self):
        contents = self._read_yaml_file()

        year_data = contents.get(str(self.tag.year)) if contents is not None else None
        month_data = year_data.get(str(self.tag.month)) if year_data is not None else None
        day_data = month_data.get(str(self.tag.day)) if month_data is not None else None

        if day_data is not None:
            for scan in day_data:
                self.processed_list.append(scan)

    def _read_yaml_file(self):
        contents = None
        if self.processed_list_filename.exists():
            with open(self.processed_list_filename, 'r') as file:
                contents = yaml.safe_load(file) or []
        return contents

    def write_processed_list(self):
        data = self._read_yaml_file()
        new_contents = {str(self.tag.year): {str(self.tag.month): {str(self.tag.day): self.processed_list}}}
        if data is None:
            data = new_contents
        else:
            data = recursive_update(data, new_contents)

        with open(self.processed_list_filename, 'w') as file:
            yaml.safe_dump(data, file)


def recursive_update(base, new):
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
    year = 2024
    month = 11
    day = 26

    scan_watch = ScanWatch(experiment_name=exp, year=year, month=month, day=day)
    print("Starting...")
    scan_watch.start(watch_folder_not_exist='wait')

    while True:
        time.sleep(0.5)
        scan_watch.process_queue()
        pass
