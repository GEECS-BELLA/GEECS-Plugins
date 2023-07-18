from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Optional
import re
from multiprocessing import Process

import logging
import logging.config
import json
logging.config.dictConfig(
    json.load(
        (Path(__file__).parents[1] / "logging_config.json")
            .open()
    )
)
logger = logging.getLogger("live_run_analysis")

from .utils import get_run_folder
from .scan_analyzer import ScanAnalyzer

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
if TYPE_CHECKING:
    from watchdog.events import DirCreatedEvent, FileCreatedEvent


class AnalysisFolderEventHandler(FileSystemEventHandler):

    s_filename_regex = re.compile(r"s(?P<scan_number>\d+).txt")

    def __init__(self, scan_analyzer: ScanAnalyzer):
        self.scan_analyzer = scan_analyzer
        super().__init__()

    def on_created(self, event: DirCreatedEvent|FileCreatedEvent):
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
        logger.info(f"Found s-file {filename}, starting analysis on scan {date_folder_name}:Scan{scan_number:03d}")

        try:
            self.scan_analyzer.analyze_scan(date_folder_name, scan_number)
        except Exception as err:
            logger.error(f"Error in analyze_scan({date_folder_name}, {scan_number:d}): {err}")

        try:
            self.scan_analyzer.save_scan_metrics()
        except Exception as err:
            logger.error(f"Error in save_scan_metrics() on scan {date_folder_name}:Scan{scan_number:03d}: {err}")

        try:
            self.scan_analyzer.upload_scan_metrics()
        except Exception as err:
            logger.error(f"Error in upload_scan_metrics() on scan {date_folder_name}:Scan{scan_number:03d}: {err}")

        logger.info(f"Finished analysis on scan {date_folder_name}:Scan{scan_number:03d}")


class ScanWatch:
    def __init__(self, date_folder_name: str, 
                 scan_analyzer: Optional[ScanAnalyzer] = None, 
                ):
        """
        Parameters
        ----------
        date_folder_name : str
            a HTU date folder name in the format 'yy_mmdd'
        scan_analyzer : Optional[ScanAnalyzer]
            use an existing ScanAnalyzer instance, or (default) create a new one.
        """
        self.watch_folder: Path = get_run_folder(date_folder_name)/"analysis"
        if scan_analyzer is None:
            self.scan_analyzer = ScanAnalyzer()
        else:
            self.scan_analyzer = scan_analyzer
        self.observer = PollingObserver()
        self.observer.schedule(AnalysisFolderEventHandler(self.scan_analyzer), self.watch_folder)

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


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--date_folder_name",
                    type=str,
                    default=datetime.now().strftime("%y_%m%d"),
                    help="date folder to watch for scans, in yy_mmdd format", 
                   )
    ap.add_argument("--watch_folder_not_exist",
                    type=str,
                    choices=['wait', 'create', 'raise'],
                    default='raise',
                    help=("What to do if date_folder/analysis does not exist. "
                          "'wait': wait for the watch folder to be created by the control system; "
                          "'create': create the folder ourselves; "
                          "'raise': raise FileNotFoundError; "
                         ) 
                   )

    args = ap.parse_args()

    scan_watch = ScanWatch(args.date_folder_name)
    scan_watch.run(watch_folder_not_exist=args.watch_folder_not_exist)
