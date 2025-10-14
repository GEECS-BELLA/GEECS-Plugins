"""
File watcher that auto-runs scan analyses when new s-files appear.

This module monitors a day's ``analysis/`` folder for newly created s-files
(e.g., ``s123.txt``). When a new s-file is detected, it enqueues a matching
:class:`~geecs_data_utils.ScanTag` and invokes the appropriate analyzer(s)
via :func:`scan_analysis.execute_scan_analysis.analyze_scan`. It also
persists a YAML record of processed scans to avoid duplicate work.

Notes
-----
- Originally prototyped for LiveImageProcessing by Reinier van Mourik and
  adapted by Chris Doss for integration with :class:`~scan_analysis.base.ScanAnalyzer`.
- Consider extracting shared logic into ``geecs-python-api`` if reuse continues.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from scan_analysis.base import ScanAnalyzerInfo

import time
from pathlib import Path
from time import sleep
import re
from queue import Queue

from geecs_data_utils import ScanPaths, ScanTag
from scan_analysis.execute_scan_analysis import analyze_scan
from scan_analysis.mapping.scan_evaluator import check_for_analysis_match

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

if TYPE_CHECKING:
    from watchdog.events import DirCreatedEvent, FileCreatedEvent
    from geecs_data_utils import ScanTag

import logging.config
import json
import yaml

print(Path(__file__).parents[1] / "logging_config.json")

logging.config.dictConfig(
    json.load((Path(__file__).parents[1] / "logging_config.json").open())
)
logger = logging.getLogger("scan_analyzer")


def extract_s_file_number(filename: str) -> Optional[int]:
    """
    Extract the scan number from an s-file name.

    Matches names like ``s42.txt`` and returns the integer scan number.

    Parameters
    ----------
    filename : str
        File name to check (not a full path required).

    Returns
    -------
    int or None
        The scan number if the filename matches an s-file pattern, otherwise ``None``.
    """
    s_filename_regex = re.compile(r"s(?P<scan_number>\d+).txt")
    m = s_filename_regex.match(filename)
    if m is None:
        return None
    else:
        return int(m["scan_number"])


class AnalysisFolderEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler that queues scans when new s-files are created.

    Attributes
    ----------
    queue : Queue
        Thread-safe queue to receive :class:`~geecs_data_utils.ScanTag` items.
    base_tag : ScanTag
        Base tag containing date/experiment info; scan number is replaced per file.
    """

    def __init__(self, scan_watch_queue: Queue, base_tag: ScanTag):
        """
        Initialize the event handler.

        Parameters
        ----------
        scan_watch_queue : Queue
            Queue to which new :class:`~geecs_data_utils.ScanTag` items are pushed.
        base_tag : ScanTag
            Base tag providing ``year``, ``month``, ``day``, and ``experiment``.
        """
        super().__init__()
        self.queue: Queue = scan_watch_queue
        self.base_tag = base_tag

    def on_created(self, event: Union[DirCreatedEvent, FileCreatedEvent]):
        """
        React to filesystem create events.

        If a newly created file matches the s-file pattern (``sNNN.txt``) in
        the day's ``analysis/`` folder, enqueue a corresponding scan tag.

        Parameters
        ----------
        event : DirCreatedEvent or FileCreatedEvent
            Watchdog event describing the creation.

        Notes
        -----
        Directory creations are ignored.
        """
        # ignore new directories
        if event.is_directory:
            return

        date_folder_name, analysis_literal, filename = Path(event.src_path).parts[-3:]
        assert analysis_literal == "analysis"
        scan_number = extract_s_file_number(filename)
        # ignore anything other than sXX.txt files
        if scan_number is None:
            return

        logger.info(f"Found new s-file: {filename}")
        tag = ScanTag(
            year=self.base_tag.year,
            month=self.base_tag.month,
            day=self.base_tag.day,
            number=scan_number,
            experiment=self.base_tag.experiment,
        )

        self.queue.put(tag)


class ScanWatch:
    """
    File watcher for daily scan analysis.

    This class monitors a day's ``analysis/`` directory for new s-files,
    determines which analyzers apply to each scan, and runs them. It
    maintains a YAML record of processed scans to prevent reprocessing.

    Parameters
    ----------
    experiment_name : str
        GEECS experiment name (e.g., ``"Undulator"``).
    year : int
        Year of the run day.
    month : int or str
        Month of the run day.
    day : int
        Day of the run day.
    analyzer_list : list of ScanAnalyzerInfo
        Candidate analyzers to match against discovered scans.
    ignore_list : list of int, optional
        Scan numbers to treat as already processed.
    overwrite_previous : bool, default=False
        If True, ignore any persisted processed-set and re-analyze everything.
    perform_initial_search : bool, default=True
        If True, populate the queue with any unprocessed s-files already present.
    upload_to_scanlog : bool, default=True
        If True, upload analyzer outputs to the Scan Log (Google Doc).
    documentID : str, optional
        Explicit Google Doc ID to receive uploads; if omitted, defaults to today's scan log.

    Attributes
    ----------
    tag : ScanTag
        Base tag for day/experiment context; scan number is updated per job.
    watch_folder : Path
        Path to the day's ``analysis/`` directory being watched.
    analysis_queue : Queue
        Queue of :class:`~geecs_data_utils.ScanTag` to process.
    processed_list_filename : Path
        YAML file that persists processed scan numbers for the day.
    processed_list : list of int
        In-memory set of processed scan numbers (duplicates avoided externally).
    observer : PollingObserver
        Watchdog polling observer.
    upload_to_scanlog : bool
        Whether outputs should be uploaded to the scan log.
    documentID : str or None
        Target Google Doc ID when uploading is enabled.
    """

    def __init__(
        self,
        experiment_name: str,
        year: int,
        month: Union[int, str],
        day: int,
        analyzer_list: list[ScanAnalyzerInfo],
        ignore_list: list[int] = None,
        overwrite_previous: bool = False,
        perform_initial_search: bool = True,
        upload_to_scanlog: bool = True,
        documentID: Optional[str] = None,
    ):
        """
        Construct a ScanWatch object.

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
        upload_to_scanlog: bool
            Flag to upload returned list of files to google doc
        documentID: str
            If given, will use this documentID when uploading data.  Otherwise, defaults to today's scanlog

        """
        self.tag = ScanTag(
            year=year, month=month, day=day, number=0, experiment=experiment_name
        )
        self.watch_folder = (
            ScanPaths.get_scan_folder_path(tag=self.tag).parents[1] / "analysis"
        )

        self.analyzer_list = analyzer_list

        self.analysis_queue = Queue()

        self.processed_list_filename = Path(
            f"{self.watch_folder}/processed_scans_{experiment_name}.yaml"
        )
        self.processed_list = []
        if ignore_list is not None:
            self.processed_list = ignore_list
        if not overwrite_previous:
            self._read_processed_list()

        self.observer = PollingObserver()
        self.observer.schedule(
            AnalysisFolderEventHandler(self.analysis_queue, self.tag),
            str(self.watch_folder),
        )

        # Initial check of scan folder
        if perform_initial_search:
            self.initial_search_of_watch_folder()

        self.upload_to_scanlog = upload_to_scanlog
        self.documentID = documentID

    def _check_watch_folder_exists(self, watch_folder_not_exist: str = "raise"):
        """
        Ensure the watch folder exists, or handle its absence.

        Parameters
        ----------
        watch_folder_not_exist : {'wait', 'create', 'raise'}, default='raise'
            Strategy if the watch folder does not yet exist:
            - 'wait' : Block until the folder appears.
            - 'create' : Create the folder.
            - 'raise' : Raise :class:`FileNotFoundError`.

        Raises
        ------
        FileNotFoundError
            If the folder does not exist and ``watch_folder_not_exist='raise'``.
        ValueError
            If an unknown strategy string is provided.
        """
        if not self.watch_folder.exists():
            if watch_folder_not_exist == "raise":
                raise FileNotFoundError(
                    f"Watch folder {self.watch_folder} does not exist. "
                    "Either create it, or set watch_folder_not_exist "
                    "to 'wait' or 'create'"
                )

            elif watch_folder_not_exist == "create":
                logger.info(f"Creating {self.watch_folder}.")
                self.watch_folder.mkdir(parents=True)

            elif watch_folder_not_exist == "wait":
                logger.info(f"Waiting for {self.watch_folder} to be created.")
                while not self.watch_folder.exists():
                    sleep(10)
            else:
                raise ValueError(
                    f"Unknown value for watch_folder_not_exist: {watch_folder_not_exist}"
                )

    def start(self, watch_folder_not_exist: str = "raise"):
        """
        Start watching the day's analysis folder.

        Parameters
        ----------
        watch_folder_not_exist : {'wait', 'create', 'raise'}, default='raise'
            Strategy if the watch folder does not yet exist (see
            :meth:`_check_watch_folder_exists`).

        Raises
        ------
        FileNotFoundError
            If the folder does not exist and ``watch_folder_not_exist='raise'``.
        ValueError
            If an unknown strategy string is provided.
        """
        self._check_watch_folder_exists(watch_folder_not_exist)

        self.observer.start()
        logger.info(f"Started watching folder {self.watch_folder} for sXX.txt files.")

    def stop(self):
        """
        Stop the polling observer.

        Notes
        -----
        Blocks until the observer thread has joined.
        """
        self.observer.stop()
        self.observer.join()

    def process_queue(self):
        """
        Process the next queued scan, if any.

        Dequeues one :class:`~geecs_data_utils.ScanTag`, marks it as processed
        for the day, and runs matching analyzers on the corresponding scan
        folder.

        Notes
        -----
        If the queue is empty, this method returns immediately.
        """
        if self.analysis_queue.empty():
            return

        tag: ScanTag = self.analysis_queue.get()
        self.processed_list.append(tag.number)
        scan_folder = ScanPaths.get_scan_folder_path(tag=tag)
        self._evaluate_folder(tag, scan_folder)

    def initial_search_of_watch_folder(self):
        """
        Seed the queue with any unprocessed s-files already present.

        Scans the day's ``analysis/`` folder for s-files not listed in
        ``processed_list`` (or ``ignore_list``), queues them in ascending
        scan-number order, and logs a summary of findings.
        """
        logger.info(
            f"Searching for untouched scans on {self.tag.month}/{self.tag.day}/{self.tag.year}:"
        )
        valid_scan_numbers = []
        if self.watch_folder.exists():
            for item in self.watch_folder.iterdir():
                if item.is_file():
                    scan_number = extract_s_file_number(str(item.name))
                    if (
                        scan_number is not None
                        and scan_number not in self.processed_list
                    ):
                        valid_scan_numbers.append(scan_number)
        sorted_scans = sorted(valid_scan_numbers)
        for scan_number in sorted_scans:
            tag = ScanTag(
                year=self.tag.year,
                month=self.tag.month,
                day=self.tag.day,
                number=scan_number,
                experiment=self.tag.experiment,
            )
            self.analysis_queue.put(tag)
        logger.info(f"Found {self.analysis_queue.qsize()} untouched scans.")

    def _evaluate_folder(self, tag: ScanTag, scan_folder: Path):
        """
        Run matching analyzers for a given scan folder.

        Parameters
        ----------
        tag : ScanTag
            Scan tag representing the specific scan to analyze.
        scan_folder : Path
            Path to the scan's folder.

        Side Effects
        ------------
        - Executes matched analyzers via :func:`analyze_scan`.
        - Updates the YAML record of processed scans.
        - Emits log records before/after analysis.
        """
        logger.info(
            f"Starting analysis on scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}"
        )
        valid_analyzers = check_for_analysis_match(
            scan_folder=scan_folder, analyzer_list=self.analyzer_list
        )

        analyze_scan(
            tag,
            valid_analyzers,
            debug_mode=False,
            upload_to_scanlog=self.upload_to_scanlog,
            documentID=self.documentID,
        )

        self._write_processed_list()

        logger.info(
            f"Finished analysis on scan {tag.month}/{tag.day}/{tag.year}:Scan{tag.number:03d}"
        )

    def _read_processed_list(self):
        """
        Load the processed-scan list from YAML.

        Reads the per-experiment YAML at ``processed_list_filename`` and
        extends :attr:`processed_list` with any scans recorded for the
        current day.

        Notes
        -----
        Missing files are treated as empty; no exception is raised here.
        """
        contents = self._read_yaml_file()

        year_data = contents.get(str(self.tag.year)) if contents is not None else None
        month_data = (
            year_data.get(str(self.tag.month)) if year_data is not None else None
        )
        day_data = month_data.get(str(self.tag.day)) if month_data is not None else None

        if day_data is not None:
            for scan in day_data:
                self.processed_list.append(scan)

    def _read_yaml_file(self) -> dict:
        """
        Read and parse the processed-scans YAML file.

        Returns
        -------
        dict
            Parsed YAML contents; empty dict if file is missing or unreadable.

        Raises
        ------
        FileNotFoundError
            If repeated attempts fail beyond the allowed maximum.
        """
        attempts = 0
        while attempts < 3:
            if self.processed_list_filename.exists():
                try:
                    with open(self.processed_list_filename, "r") as file:
                        contents = yaml.safe_load(file) or {}
                    logging.info(f"Read scans from '{self.processed_list_filename}'.")
                    return contents

                except FileNotFoundError:  # Race condition or network issue
                    logging.warning(
                        f"Could not read {self.processed_list_filename}: temporarily unavailable."
                    )
                    time.sleep(secs=1)
            attempts += 1
        logging.warning(
            f"Max attempts reached, did not write to '{self.processed_list_filename}'."
        )
        if attempts > 3:
            raise FileNotFoundError(
                f"Max attempts reached, did not write to '{self.processed_list_filename}'."
            )
        else:
            return {}

    def _write_processed_list(self):
        """
        Persist the processed-scan list to YAML.

        Merges the current day's :attr:`processed_list` into the on-disk YAML
        structure, creating or updating nested year/month/day keys as needed.

        Raises
        ------
        FileNotFoundError
            If repeated write attempts fail beyond the allowed maximum.
        """
        data = self._read_yaml_file()
        new_contents = {
            str(self.tag.year): {
                str(self.tag.month): {str(self.tag.day): self.processed_list}
            }
        }
        if data is None:
            data = new_contents
        else:
            data = recursive_update(data, new_contents)

        attempts = 0
        while attempts < 3:
            try:
                with open(self.processed_list_filename, "w") as file:
                    yaml.safe_dump(data, file)
                logging.info(f"Wrote scans to '{self.processed_list_filename}'.")
                return

            except FileNotFoundError:  # Race condition or network issue
                logging.warning(
                    f"Failed attempt writing to '{self.processed_list_filename}': temporarily unavailable."
                )
                attempts += 1
                time.sleep(secs=1)
        logging.warning(
            f"Max attempts reached, did not write to '{self.processed_list_filename}'."
        )
        raise FileNotFoundError(
            f"Max attempts reached, did not write to '{self.processed_list_filename}'."
        )


def recursive_update(base: dict, new: dict) -> dict:
    """
    Recursively update a nested dictionary.

    Keys present in ``new`` replace or merge into ``base``. Nested dictionaries
    are updated recursively.

    Parameters
    ----------
    base : dict
        Original dictionary to be updated (modified in place).
    new : dict
        New values to merge into ``base``.

    Returns
    -------
    dict
        The updated dictionary (same object as ``base``).
    """
    for key, value in new.items():
        if isinstance(value, dict):
            base[key] = recursive_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base


if __name__ == "__main__":
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
    from scan_analysis.mapping.map_Undulator import undulator_analyzers

    exp = "Undulator"
    test_year = 2025
    test_month = 2
    test_day = 13

    scan_watch = ScanWatch(
        experiment_name=exp,
        year=test_year,
        month=test_month,
        day=test_day,
        analyzer_list=undulator_analyzers,
        overwrite_previous=True,
    )
    print("Starting...")
    scan_watch.start(watch_folder_not_exist="wait")

    while True:
        time.sleep(0.5)
        scan_watch.process_queue()
        pass
