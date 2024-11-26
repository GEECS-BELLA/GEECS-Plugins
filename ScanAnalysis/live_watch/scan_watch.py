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

from geecs_python_api.analysis.scans.scan_data import ScanData
from scan_analysis.execute_scan_analysis import test_command
from devices_to_analysis_mapping import check_for_analysis_match

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

if TYPE_CHECKING:
    from watchdog.events import DirCreatedEvent, FileCreatedEvent

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
logger = logging.getLogger("live_run_analysis")


class AnalysisFolderEventHandler(FileSystemEventHandler):
    s_filename_regex = re.compile(r"s(?P<scan_number>\d+).txt")

    def __init__(self, scan_watch_instance):
        super().__init__()
        self.scan_watch: ScanWatch = scan_watch_instance

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

        scan_number = int(m['scan_number'])

        valid_analyzers = check_for_analysis_match(Path(event.src_path).parents[1] / 'scans' / f"Scan{scan_number:03d}")
        tag = ScanData.get_scan_tag(scan_watch.tag.year, scan_watch.tag.month, scan_watch.tag.day, scan_number)
        self.scan_watch.processed_list.append(scan_number)

        logger.info(f"Found s-file {filename}, starting analysis on scan {date_folder_name}:Scan{scan_number:03d}")

        try:
            test_command(tag, valid_analyzers)
        except Exception as err:
            logger.error(f"Error in analyze_scan({date_folder_name}, {scan_number:d}): {err}")

        self.scan_watch.write_processed_list()

        # try:
        #     self.scan_analyzer.save_scan_metrics()
        # except Exception as err:
        #     logger.error(f"Error in save_scan_metrics() on scan {date_folder_name}:Scan{scan_number:03d}: {err}")

        # try:
        #     self.scan_analyzer.upload_scan_metrics()
        # except Exception as err:
        #     logger.error(f"Error in upload_scan_metrics() on scan {date_folder_name}:Scan{scan_number:03d}: {err}")

        logger.info(f"Finished analysis on scan {date_folder_name}:Scan{scan_number:03d}")


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
        self.tag = ScanData.get_scan_tag(year, month, day, number=0)
        self.watch_folder = ScanData.build_scan_folder_path(tag=self.tag,
                                                            experiment=experiment_name).parents[1] / "analysis"

        self.processed_list_filename = Path(f"./processed_scans_{experiment_name}.yaml")
        self.processed_list = []
        if ignore_list is not None:
            self.processed_list = ignore_list
        if not overwrite_previous:
            self.read_processed_list()

        self.observer = PollingObserver()
        self.observer.schedule(AnalysisFolderEventHandler(self), str(self.watch_folder))

        # Initial check of scan folder

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

    def run(self, watch_folder_not_exist: str = 'raise'):
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
    scan_watch.run(watch_folder_not_exist='wait')

    while True:
        time.sleep(10)
        pass
    #time.sleep(10)
    #scan_watch.stop()
