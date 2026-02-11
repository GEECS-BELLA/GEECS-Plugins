"""
Data logging and file management module for GEECS experimental scans.

This module defines the `DataLogger` and `FileMover` classes, which coordinate
data acquisition, device synchronization, and file organization during scans.
These classes are intended to be used exclusively by within this project and
not standalone objects. In other words, creation and use of DataLogger should
be handled exclusively by a ScanManager object.

Classes
-------
DataLogger
    Handles event-driven and asynchronous logging from GEECS devices.
    Manages timestamps, synchronization, and log data organization.
FileMover
    Processes file move tasks based on timestamp matching and device type.
    Supports background thread workers, orphan file handling, and variant processing.

Functionality
-------------
- Logs data via TCP subscriptions using elapsed time.
- Synchronizes multiple devices using standby/trigger detection.
- Automatically renames and moves primary and derived files.
- Supports composite variables, asynchronous polling, and audio feedback.

Dependencies
------------
- DeviceManager
- GeecsDevice (GEECS Python API)
- SoundPlayer (utils)
- pandas, threading, pathlib, logging
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Callable, List, Union

import time
from time import time as _now
import threading
from datetime import datetime
import logging
import pandas as pd
import queue
import shutil
from pathlib import Path
from dataclasses import dataclass

from . import DeviceManager
from geecs_scanner.utils import SoundPlayer
from geecs_scanner.logging_setup import update_context

from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.tools.files.timestamping import extract_timestamp_from_file
import geecs_python_api.controls.interface.message_handling as mh

DeviceSavePaths = Dict[str, Dict[str, Union[Path, str]]]

logger = logging.getLogger(__name__)


@dataclass
class FileMoveTask:
    """
    Task definition for moving and renaming files during data acquisition.

    Attributes
    ----------
    source_dir : Path
        Directory where the original file is expected to be located.
    target_dir : Path
        Destination directory where the file should be moved.
    device_name : str
        Name of the device that generated the file.
    device_type : str
        Type of device (e.g., 'UC_UndulatorRad2', etc.).
    expected_timestamp : float
        Timestamp to match when identifying the correct file.
    shot_index : int
        Shot number corresponding to this file.
    random_part : Optional[str], optional
        Unique identifier extracted from the file name which is generated
        using the system timestamp after 'successful' acquire loop execution
    suffix : Optional[str], optional
        Suffix to identify file variants (e.g., '-interpSpec').
    new_name : Optional[str], optional
        New standardized file name (without extension).
    """

    source_dir: Path
    target_dir: Path
    device_name: str
    device_type: (
        str  # TODO Device type is not required, consider making an Optional[str]
    )
    expected_timestamp: float
    shot_index: int
    random_part: Optional[str] = None  # Unique identifier extracted from primary file.
    suffix: Optional[str] = (
        None  # For variant processing (e.g. "-interpSpec" or "-interpDiv")
    )
    new_name: Optional[str] = (
        None  # The standardized file stem (e.g., "Scan001_DeviceA_005")
    )


class FileMover:
    """
    Class for managing file renaming and relocation during or after scan acquisition.

    Uses background worker threads to handle file matching and movement based on
    acquisition timestamps and scan metadata.

    When a file is created after a successful acquisition, the file name contains
    the system timestamp. The same timestamp is recorded in the scalar data (aka
    scan_paths, sfile, etc.) during acquisition. This information is used to infer
    the shot number and change the name to match the shot number based naming
    convention established by Master Control.

    save_local = True is the more reliable mode. This requires the device computers
    to have a folder in the C: drive called SharedData that is shared with the
    appropriate domain users. Local saving prevents network bottlenecks of saving
    all data to single location at the same time.

    Attributes
    ----------
    task_queue : queue.Queue
        Queue to hold FileMoveTask instances.
    stop_event : threading.Event
        Event used to signal shutdown to worker threads.
    workers : list[threading.Thread]
        Pool of worker threads for parallel processing.
    file_check_counts : dict
        Counts of how many times a file was examined. When this exceeds a defined
        limit, the file is no longer checked. It's possible that a file and the
        metadata don't produce an immediate match and this prevents inifinitely
        checking files. When the threshold is met, the file gets added to the
        orphaned_files set.
    orphaned_files : set
        Files checked multiple times without being moved.
    orphan_tasks : list
        Tasks that could not be completed due to missing files.
    processed_files : set
        Set of files already moved.
    scan_is_live : bool
        Indicates if acquisition is still in progress.
    save_local : bool
        If True, disables network delay sleep.
    scan_number : Optional[int]
        The scan number used in renaming files.
    """

    def __init__(self, num_workers: int = 16) -> None:
        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.workers = []
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_func, daemon=True)
            worker.start()
            self.workers.append(worker)

        # Track how many times a file has been checked.
        self.file_check_counts = {}
        # Files checked more than twice without moving will be marked as orphaned.
        self.orphaned_files = set()
        self.orphan_tasks = []
        self.processed_files = set()
        self.scan_is_live = True
        self.save_local = True

        self.scan_number: Optional[int] = (
            None  # TODO Should have a `set` function instead of editing externally
        )
        logger.info("FileMover worker started.")

    def _worker_func(self) -> None:
        """
        Continuously process file move tasks from the queue until stopped.

        This background thread worker runs in a loop, pulling tasks from the queue.
        It handles graceful shutdown when `stop_event` is set and the queue is empty.
        """
        while True:
            try:
                task: Optional[FileMoveTask] = self.task_queue.get(timeout=1)
            except queue.Empty:
                if self.stop_event.is_set() and self.task_queue.empty():
                    break
                continue

            if task is None:
                continue

            try:
                self._process_task(task)
            except Exception:
                logger.exception("Error processing task")
            finally:
                self.task_queue.task_done()
        logger.info("FileMover worker stopped.")

    def _process_task(self, task: FileMoveTask) -> None:
        """
        Process a FileMoveTask by locating, renaming, and moving matching files.

        This method searches the appropriate variant directories for files that match the
        expected acquisition timestamp. Once a match is found, the file is renamed using
        a standardized naming convention and moved to the corresponding target directory.
        If the device type is a MagSpec variant, additional related files (e.g., "-interpSpec")
        are also processed.

        Parameters
        ----------
        task : FileMoveTask
            The task containing source/target paths, device info, and expected timestamp.

        Notes
        -----
        - If `save_local` is False, adds a short delay to allow time for files to be written.
        - Files that cannot be matched after repeated checks are marked as orphaned.
        - For MagSpec (and others devices), looks for and processes additional variant files
         with known suffixes.
        """
        source_dir = task.source_dir
        target_dir = task.target_dir
        device_name = task.device_name
        device_type = task.device_type
        expected_timestamp = task.expected_timestamp
        shot_index = task.shot_index

        home_dir = source_dir.parent

        # For MagSpec devices, only search the primary folder (exact match).
        if device_type in ["MagSpecStitcher", "MagSpecCamera"]:
            variant_dirs = [
                d for d in home_dir.iterdir() if d.is_dir() and d.name == device_name
            ]
        else:
            variant_dirs = [
                d
                for d in home_dir.iterdir()
                if d.is_dir() and d.name.startswith(device_name)
            ]

        # Determine expected file count.
        if device_type in [
            "PicoscopeV2",
            "FROG",
            "Thorlabs CCS175 Spectrometer",
            "RohdeSchwarz_RTA4000",
            "ThorlabsWFS"
        ]:
            expected_file_count = 2
        elif device_type in ["MagSpecStitcher", "MagSpecCamera"]:
            expected_file_count = 1
        else:
            expected_file_count = 1

        # there is some time overhead in saving to the netapp rather than local. It's possible
        # that the FileMoveTask for a given event is created and queued before the file is
        # successfully written to the network drive. In that case, no match is found but then
        # the file is written shortly after. As it stands, each new time stamp is checked
        # against existing files once, so it's quite possible that a files build up without
        # getting moved/renamed. In the future, could consider re-queueing a task a fixed number
        # of times rather than adding a sleep here.
        if not self.save_local:
            time.sleep(0.1)

        task_success = False
        for variant in variant_dirs:
            task_success = False
            adjusted_target_dir = target_dir.parent / variant.name
            adjusted_target_dir.mkdir(parents=True, exist_ok=True)

            found_files_count = 0
            for file in variant.glob("*"):
                if not file.is_file():
                    continue

                # if the file has been checked already e.g. orphaned, skip when the scan is live.
                # If scan is not live, allow cleanup tasks.
                if file in self.orphaned_files:
                    if self.scan_is_live:
                        continue

                if file in self.processed_files:
                    continue

                # is scan is live, make sure you don't get stuck in a loop iterating through the
                # same files over and over by adding it to the orphaned files. iF scan is not live
                # allow the task processing to continue one more time
                if self.scan_is_live:
                    self.file_check_counts[file] = (
                        self.file_check_counts.get(file, 0) + 1
                    )
                    if self.file_check_counts[file] > 1:
                        logger.info(
                            "File %s checked >1 times; marking as orphaned.", file
                        )
                        self.orphaned_files.add(file)
                        continue

                file_ts = extract_timestamp_from_file(file, device_type)
                if abs(file_ts - expected_timestamp) < 0.0011:
                    found_files_count += 1

                    # Extract the unique random part from the primary file.
                    # Expects filename format: "{device_name}_{random}.png"
                    random_part = file.stem.replace(f"{device_name}_", "")
                    task.random_part = random_part

                    # Generate the new standardized file stem.
                    task.new_name = self._generate_device_shot_filename(
                        self.scan_number, device_name, shot_index
                    )

                    # Move the primary file.
                    self._move_file(task, file, variant.name)

                    # For MagSpec devices, process additional variant files.
                    if device_type in ["MagSpecStitcher", "MagSpecCamera"]:
                        task.suffix = "-interp"
                        self._process_variant_file(task)
                        task.suffix = "-interpSpec"
                        self._process_variant_file(task)
                        task.suffix = "-interpDiv"
                        self._process_variant_file(task)

                    if found_files_count == expected_file_count:
                        task_success = True
                        break

        if not task_success:
            # Task failed to find a match, log it as an orphaned task
            logger.info(
                "failed to find a file for %s with timestamp %s",
                task.device_name,
                task.expected_timestamp,
            )
            self.orphan_tasks.append(task)

    def _move_file(
        self, task: FileMoveTask, source_file: Path, new_device_name: str
    ) -> bool:
        """
        Rename and move a file based on the given FileMoveTask and device variant.

        Constructs a standardized file name using the scan number, device name, and shot index.
        The file is then moved to the appropriate target directory, which is created if needed.

        Parameters
        ----------
        task : FileMoveTask
            The task containing directory paths, shot index, and naming metadata.
        source_file : Path
            The original file to be moved and renamed.
        new_device_name : str
            Name of the device (or variant) used in the target path and naming.

        Returns
        -------
        bool
            True if the file was moved successfully; False if an error occurred.
        """
        target_dir = task.target_dir.parent / new_device_name
        target_dir.mkdir(parents=True, exist_ok=True)
        new_file_stem = (
            task.new_name
            if task.new_name
            else self._generate_device_shot_filename(
                self.scan_number, new_device_name, task.shot_index
            )
        )
        new_filename = new_file_stem + source_file.suffix
        dest_file = target_dir / new_filename
        try:
            shutil.move(str(source_file), str(dest_file))
            self.processed_files.add(dest_file)
            return True
        except Exception:
            logger.exception("Error moving %s to %s", source_file, dest_file)
            return False

    def _process_variant_file(self, task: FileMoveTask) -> None:
        """
        Process and move a file from a variant directory based on the task's suffix and random part.

        Constructs a variant directory path using the device name and suffix. Searches the directory
        for a file whose name contains the task's `random_part`, then renames and moves the file
        using the standardized name defined in `task.new_name`.

        Parameters
        ----------
        task : FileMoveTask
            The task containing device name, suffix, random part, and target directory information.
        """
        if task.suffix is None or task.random_part is None:
            logger.info(
                "No suffix or random_part in task; skipping variant processing."
            )
            return

        variant_dir = task.source_dir.parent / f"{task.device_name}{task.suffix}"
        if not variant_dir.exists():
            logger.info(
                "Variant directory %s does not exist; skipping processing for %s.",
                variant_dir,
                task.suffix,
            )
            return

        candidate = None
        for file in variant_dir.glob("*"):
            if file.is_file() and task.random_part in file.stem:
                candidate = file
                break

        if candidate is None:
            logger.warning(
                "No file found in %s containing %s.", variant_dir, task.random_part
            )
            return

        new_device_name = f"{task.device_name}{task.suffix}"
        self._move_file(task, candidate, new_device_name)

    @staticmethod
    def _generate_device_shot_filename(
        scan_number: int, device_name: str, shot_index: int
    ) -> str:
        """
        Generate a standardized file stem using scan number, device name, and shot index.

        The output format is: "ScanXXX_Device_YYY", where:
        - XXX is the zero-padded scan number
        - YYY is the zero-padded shot index

        Parameters
        ----------
        scan_number : int
            Scan number to include in the filename (zero-padded to 3 digits).
        device_name : str
            Name of the device generating the file.
        shot_index : int
            Shot number (zero-padded to 3 digits).

        Returns
        -------
        str
            Standardized file stem for the device shot.
        """
        scan_number_str = str(scan_number).zfill(3)
        shot_number_str = str(shot_index).zfill(3)
        return f"Scan{scan_number_str}_{device_name}_{shot_number_str}"

    def move_files_by_timestamp(self, task: FileMoveTask) -> None:
        """
        Enqueue a file move task to be processed by worker threads.

        Parameters
        ----------
        task : FileMoveTask
            The task containing file movement parameters such as source and target directories,
            device name, expected timestamp, and shot index.
        """
        self.task_queue.put(task)

    def _post_process_orphaned_files(
        self, log_df: pd.DataFrame, device_save_paths_mapping: dict
    ) -> None:
        """
        Attempt to recover and process orphaned files based on device timestamps and log data.

        For each device listed in the `device_save_paths_mapping`, recursively search the source
        directory for files containing the device name. For each file found, extract its timestamp
        and compare it to acquisition timestamps in `log_df`. If a match is found within a tolerance
        window, determine the corresponding shot number, construct a `FileMoveTask`, and enqueue it
        for processing.

        Parameters
        ----------
        log_df : pd.DataFrame
            DataFrame containing shot metadata with a 'Shotnumber' column and per-device acquisition
            timestamps (e.g., `"DeviceA acq_timestamp"`).

        device_save_paths_mapping : dict
            Mapping of device names to a dictionary with keys:
            - 'source_dir': directory where the raw files are located
            - 'target_dir': destination directory for processed files
            - 'device_type': type of the device, used for timestamp parsing

        Notes
        -----
        - Matching is done using a configurable timestamp tolerance (`0.0011` seconds).
        - Orphan files are those not processed during live acquisition but present on disk.
        - This method assumes a filename format of `{device_name}_{random}.ext`.
        """
        logger.info("looking to handle orphaned data files")
        tolerance = 0.0011  # Adjust as needed
        for device_name, device_info in device_save_paths_mapping.items():
            source_dir = Path(device_info["source_dir"])
            target_dir = Path(device_info["target_dir"])
            device_type = device_info["device_type"]

            # Create a list of (shotnumber, timestamp) pairs from the df. Ensure the df columns are named appropriately.
            # *NOTE* Using `acq_timestamp` for data that was logged
            shot_timestamp_pairs = [
                (row["Shotnumber"], row[f"{device_name} acq_timestamp"])
                for _, row in log_df.iterrows()
                if pd.notnull(row[f"{device_name} acq_timestamp"])
            ]

            # Recursively find orphaned files that include the device name.
            orphan_files = [
                f
                for f in source_dir.rglob("*")
                if f.is_file() and device_name in f.name
            ]

            for file in orphan_files:
                file_ts = extract_timestamp_from_file(file, device_type)
                logger.info("Found orphan file %s with timestamp %s", file, file_ts)
                matched_shot = None

                # Find the matching shot number using the pairs from the DataFrame.
                for shot_number, ts in shot_timestamp_pairs:
                    if abs(file_ts - ts) < tolerance:
                        matched_shot = int(shot_number)
                        break

                if matched_shot is not None:
                    # Extract unique random part from the filename.
                    random_part = file.stem.replace(f"{device_name}_", "")

                    # Create a FileMoveTask for this orphan file.
                    task = FileMoveTask(
                        source_dir=file.parent,
                        target_dir=target_dir,
                        device_name=device_name,
                        device_type=device_type,
                        expected_timestamp=file_ts,
                        shot_index=matched_shot,
                        random_part=random_part,
                    )
                    logger.info(
                        "Enqueuing orphan task for {file} with shot number %s",
                        matched_shot,
                    )
                    # Process the task using your FileMover's method.

                    self.move_files_by_timestamp(task)
                else:
                    logger.warning(
                        "No matching shot number found for orphan file %s (timestamp %s)",
                        file,
                        file_ts,
                    )

    def _post_process_orphan_task(self):
        """
        Enqueue all previously failed `FileMoveTask` entries for reprocessing.

        This method iterates through the list of orphaned tasks (i.e., tasks that failed
        to find matching files during initial processing) and re-adds them to the queue
        for another attempt at moving files.
        """
        for task in self.orphan_tasks:
            self.move_files_by_timestamp(task)

    def shutdown(self, wait: bool = True) -> None:
        """
        Shut down all worker threads gracefully.

        This method signals that no new tasks will be added by setting the stop event.
        If `wait` is True, it blocks until all currently queued tasks are processed.
        It then sends shutdown signals to all worker threads and waits for them to exit.

        Parameters
        ----------
        wait : bool, optional
            Whether to wait for all currently queued tasks to finish before shutdown.
            Defaults to True.
        """
        self.stop_event.set()
        if wait:
            self.task_queue.join()
        for _ in self.workers:
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join()
        logger.info("FileMover has been shut down gracefully.")


class DataLogger:
    """
    Handle logging of device data during scans.

    This class manages both event-driven (synchronous) and polled (asynchronous) observables.
    It coordinates synchronization, state tracking, file handling, sound feedback, and metadata
    logging across devices. It integrates with a `DeviceManager` and a `FileMover` to enable
    hardware-aware acquisition and post-acquisition file management.

    Attributes
    ----------
    device_manager : DeviceManager
        Manages connections to GEECS devices and tracks observables.
    file_mover : FileMover or None
        Handles renaming and organizing of data files per shot.
    log_entries : dict
        Stores logged data per elapsed time during a scan.
    scan_number : int or None
        Identifier for the scan; used in filenames and metadata.
    bin_num : int
        Bin number index; set externally via the ScanManager.
    shot_index : int
        Counter for the number of shots logged during the current scan.
    polling_interval : float
        Polling frequency (seconds) for asynchronous observables.
    repetition_rate : float
        Repetition rate (Hz) used for timestamp rounding in elapsed time.
    virtual_variable_name : str or None
        Optional scan variable name to record with each shot.
    virtual_variable_value : float
        The current value of the virtual scan variable.
    data_recording : bool
        Flag indicating if logging is actively recording data.
    idle_time : float
        Duration of inactivity during polling.
    lock : threading.Lock
        Lock to synchronize access to log entries.
    stop_event : threading.Event
        Used to signal the stop of polling and logging threads.
    shot_save_event : threading.Event
        Signaled when a shot has been successfully logged.
    warning_timeout_sync : float
        Timeout for determining if a synchronous device is idle.
    warning_timeout_async_factor : float
        Multiplier of polling_interval for async timeout threshold.
    last_log_time_sync : dict
        Tracks last successful logging times for sync devices.
    last_log_time_async : dict
        Tracks last successful logging times for async devices.
    last_timestamps : dict
        Most recent timestamps received from each device.
    initial_timestamps : dict
        Initial timestamps used to compute elapsed times.
    synced_timestamps : dict
        Timestamps received post-synchronization shot.
    standby_mode_device_status : dict
        Status flags indicating whether each sync device is in standby.
    device_save_paths_mapping : dict
        Maps device names to file save and source directory information.
    synchronous_device_names : list of str
        Names of devices that report data via event triggers.
    event_driven_observables : list of str
        Observable strings (device:variable) monitored via event.
    all_devices_in_standby : bool
        True if all sync devices have entered standby state.
    devices_synchronized : bool
        True once all sync devices have received a trigger post-standby.
    save_local : bool
        Whether to store file data locally.
    sound_player : SoundPlayer
        Manages audio feedback for new shots and session completion.
    """

    def __init__(
        self, experiment_dir: Optional[str], device_manager: DeviceManager = None
    ):
        """
        Initialize the DataLogger with the experiment directory and a device manager.

        Parameters
        ----------
        experiment_dir : str, optional
            Path to the experiment directory, typically identifying the experiment name.
        device_manager : DeviceManager, optional
            An instance of DeviceManager to use for controlling devices.
            If None, a new one is created using the experiment_dir.
        """
        self.device_manager = device_manager or DeviceManager(experiment_dir)
        self.global_sync_tol_ms = 0

        self.stop_event = threading.Event()  # Event to control polling thread
        self.warning_timeout_sync = 2  # Timeout for synchronous devices (seconds)
        self.warning_timeout_async_factor = (
            1  # Factor of polling interval for async timeout
        )
        self.last_log_time_sync = {}  # Dictionary to track last log time for synchronous devices
        self.last_log_time_async = {}  # Dictionary to track last log time for async devices
        self.polling_interval = 0.5
        self.results = {}  # Store results for later processing

        self.log_entries = {}

        # Initialize the sound player
        self.sound_player = SoundPlayer()
        self.shot_index = 0

        # Note: bin_num and scan_number are updated in ScanManager
        self.bin_num = 0  # Initialize bin as 0
        self.scan_number = None

        self.virtual_variable_name = None
        self.virtual_variable_value = 0

        self.data_recording = False
        self.idle_time = 0

        self.lock = threading.Lock()

        self.repetition_rate = 1.0  # Gets updated upon scan manager's reinitialization

        self.shot_save_event = threading.Event()

        # Dictionaries for tracking timestamps and statuses
        self.last_timestamps: Dict[str, float] = {}  # Maps device names to timestamps
        self.initial_timestamps: Dict[str, Optional[float]] = {}
        self.synced_timestamps: Dict[str, float] = {}
        self.standby_mode_device_status: Dict[str, Optional[bool]] = {}
        self.device_save_paths_mapping: DeviceSavePaths = {}
        # Uses Optional[bool] because status can be True, False, or None

        # File management and observables
        self.file_mover: Optional[FileMover] = (
            None  # Will be assigned a FileMover instance later
        )
        self.synchronous_device_names: List[str] = []
        self.event_driven_observables: List[str] = []

        # Boolean flags
        self.all_devices_in_standby: bool = False
        self.devices_synchronized: bool = False
        self.save_local = True

    def set_device_save_paths_mapping(self, mapping: DeviceSavePaths) -> None:
        """
        Set the mapping of device names to their corresponding save path configurations.

        This method allows external components (e.g., ScanManager) to define where
        each device's data should be sourced and saved during logging.

        Parameters
        ----------
        mapping : DeviceSavePaths
            A dictionary mapping device names to their respective save path settings,
            typically including 'source_dir', 'target_dir', and 'device_type' for each device.
        """
        self.device_save_paths_mapping = mapping

    def start_logging(self) -> Dict[str, Any]:
        """
        Start logging data for all devices.

        This method initializes the internal state for a new logging session,
        including timestamp tracking, standby mode detection, and event registrations.
        It configures a `FileMover` instance, starts the sound queue for shot notifications,
        and registers event-driven observables for real-time data logging.

        Devices marked as event-driven will trigger logging upon updates. Asynchronous devices
        will be polled periodically, and their values will be included in log entries.

        Returns
        -------
        dict
            A dictionary mapping elapsed time (float) to dictionaries of logged device data.
            Each entry contains device variables, optional virtual scan parameters, and metadata
            such as scan number and bin number.
        """
        self.last_timestamps = {}
        self.initial_timestamps = {}
        self.synced_timestamps = {}
        self.standby_mode_device_status = {}
        self.log_entries = {}

        # Create a FileMover instance specific to this thread, e.g. scan
        self.file_mover = FileMover()

        self.all_devices_in_standby = False
        self.devices_synchronized = False
        self.synchronous_device_names = []

        # Start the sound player
        self.sound_player.start_queue()

        # Scan number in datalogger updates in scan_manager
        self.file_mover.scan_number = self.scan_number
        self.file_mover.save_local = self.save_local

        # Access event-driven and async observables from DeviceManager
        self.event_driven_observables = self.device_manager.event_driven_observables

        for observable in self.event_driven_observables:
            device_name = observable.split(":")[0]
            if device_name not in self.synchronous_device_names:
                self.synchronous_device_names.append(device_name)

        # Register the logging function for event-driven observables
        self._register_event_logging(self._handle_TCP_message_from_device)

        logger.info("Logging has started for all event-driven devices.")

        return self.log_entries

    def _handle_TCP_message_from_device(
        self, message: str, device: GeecsDevice
    ) -> None:
        """
        Process incoming TCP messages from a device and log new data when a valid timestamp is detected.

        This method is registered as a callback for event-driven devices. It parses the device-specific
        acquisition timestamp from the incoming message and determines whether the device is in standby mode
        or has received a new hardware trigger. Once all devices are confirmed to be in standby and then
        synchronized, new data is logged only if the timestamp is unique and the synchronization shot has passed.

        Parameters
        ----------
        message : str
            The raw TCP message string received from the device.
        device : GeecsDevice
            The device instance that produced the message.

        Notes
        -----
        - This method is central to event-driven logging and controls both synchronization state
          detection and real-time data acquisition.
        - Standby mode is inferred by detecting unchanged timestamps across multiple events.
        - Data is only logged after synchronization is confirmed and duplicate timestamps are filtered out.
        - The synchronization shot itself is ignored to prevent polluting the data logs.
        """
        # TCP messages and events get generated constantly from a GEECS device. We need to
        # determine if a message and event are originating from a triggered event or from a
        # timeout event. This involves parsing out the device specific timestamp on the first
        # TCP event. Then, we need to compare it to the timestamp from the following event.
        # During a timeout event, the device timestamp is not updated as it represents the
        # timestamp from the last successful acquisition. If we see that the timestamp is not
        # updating, that means we know that the device has timed out and can be considered
        # to be in standby mode. It is awaiting the next hardware trigger. So, the process
        # here is to interpret the first few messages from each device to conclude everything
        # is in standby mode and can be then be synchronized through a dedicated timing shot

        timestamp_from_device = self._extract_timestamp_from_tcp_message(
            message, device
        )

        # TODO: I think this needs to be changed to raise some kind of error, as not getting a timestamp
        # is a pretty fatal error
        if timestamp_from_device is None:
            return

        if not self.all_devices_in_standby:
            self._check_device_standby_mode_status(device, timestamp_from_device)
            self.all_devices_in_standby = self._check_all_standby_status()
            return

        # This part is a little messy as all the logic and execution that will synchronize devices
        # occurs in ScanManager
        if not self.devices_synchronized:
            self._check_device_standby_mode_status(device, timestamp_from_device)
            self.devices_synchronized = self._check_all_exited_standby_status()
            if not self.devices_synchronized:
                return
            if self.devices_synchronized:
                self.initial_timestamps = self.synced_timestamps

        elapsed_time = self._calculate_elapsed_time(device, timestamp_from_device)

        # Ensure that the synchronization shot is not included in the logging
        if elapsed_time > 0:
            if self._check_duplicate_timestamp(device, timestamp_from_device):
                return

            self._log_device_data(device, elapsed_time)

    def _check_all_standby_status(self) -> bool:
        """
        Check if all synchronous devices are in standby mode.

        This method verifies that:
        - Each device listed in `synchronous_device_names` is present in the `standby_mode_device_status` dictionary.
        - All such devices have a status of `True`, indicating they are in standby mode (i.e., not receiving new triggers).

        Returns
        -------
        bool
            True if all synchronous devices are present in the status dictionary and are confirmed to be in standby mode.
            False otherwise.

        Notes
        -----
        - This method is typically used before initiating synchronization to confirm that all relevant devices are idle.
        - Devices not yet added to the status dictionary or with `False`/`None` status are considered not in standby.
        """
        device_names = set(self.synchronous_device_names)
        standby_keys = self.standby_mode_device_status.keys()

        all_in_dict = device_names.issubset(standby_keys)
        all_on = all(
            self.standby_mode_device_status.get(device, False)
            for device in self.synchronous_device_names
        )

        if all_in_dict and all_on:
            logger.info(
                "All device names are present in standby_mode_device_status dict and all have True status."
            )
            return True
        else:
            logger.info(
                "Not all devices are in standby: %s", self.standby_mode_device_status
            )

            return False

    def _check_all_exited_standby_status(self) -> bool:
        """
        Check if all synchronous devices have exited standby mode.

        This method verifies that:
        - Each device listed in `synchronous_device_names` is present in the `standby_mode_device_status` dictionary.
        - All such devices have a status of `False`, indicating they have received a hardware trigger
          and are no longer in standby mode.

        Returns
        -------
        bool
            True if all synchronous devices are present in the status dictionary and are confirmed to have
            exited standby mode. False otherwise.

        Notes
        -----
        - This check is used after all devices are confirmed to have entered standby,
          to detect whether they have subsequently received a valid trigger.
        - Devices with `True` or `None` status are considered still in or indeterminate standby.
        """
        device_names = set(self.synchronous_device_names)
        standby_keys = self.standby_mode_device_status.keys()

        all_in_dict = device_names.issubset(standby_keys)
        all_off = all(
            not self.standby_mode_device_status.get(device, True)
            for device in self.synchronous_device_names
        )

        if all_in_dict and all_off:
            logger.info(
                "All device names are present in standby_mode_device_status dict and all "
                "have False status meaning they have exited standby mode."
            )
            return True
        else:
            return False

    def _check_device_standby_mode_status(
        self, device: GeecsDevice, timestamp: float
    ) -> None:
        """
        Update the standby mode status of a device based on repeated timestamp checks.

        This method tracks whether a device is in standby mode (i.e., has not received new data)
        by checking if its device-reported timestamp remains unchanged across TCP events.
        Devices are considered:
        - `None`: Unknown/initial state; not enough information to determine status.
        - `True`: Device is in standby (repeated identical timestamps).
        - `False`: Device has exited standby mode (received a hardware trigger and updated timestamp).

        Parameters
        ----------
        device : GeecsDevice
            The device whose standby status is being evaluated.
        timestamp : float
            The latest timestamp extracted from the device's TCP message.

        Notes
        -----
        - This method initializes `standby_mode_device_status` and `initial_timestamps` on first call.
        - If this is the first timestamp seen for the device, it sets the initial value but makes no status change.
        - On subsequent calls, if the timestamp is unchanged, the device is considered in standby.
        - If the timestamp has updated, the device is assumed to have exited standby and cannot return to `True`
          until explicitly reset.
        - Status tracking is critical for determining when all devices are idle and ready to synchronize.
        """
        # TODO: statuses are bit wonky and could be cleaned up. Right now, 'None'
        #  really means the status is unknown. "True" indicates the device verifiably
        #  went into standby mode. "False" indicates that device originally went into
        #  standby mode, and has since received a verifiable hardware trigger. So,
        #  once a "False" status has been flagged, it should stay that way until something
        #  explicitly turns it back to "None".

        # if this is the first call to this method, add the device a standby_mode
        # dict to be tracked. When it is entered, we don't know anything about its
        # status, so set it to none.
        if device.get_name() not in self.standby_mode_device_status:
            self.standby_mode_device_status[device.get_name()] = None
            self.initial_timestamps[device.get_name()] = None

        if self.standby_mode_device_status[device.get_name()] is False:
            return  # TODO may be worth to instead have a custom variable than can be one of three states

        # check if there has been a timestamp added to the dict for a given device
        t0 = self.initial_timestamps.get(device.get_name(), None)

        # if this is the first logged timestamp, return None because we can't say for
        # certain if the device is in standby mode
        if t0 is None:
            self.initial_timestamps[device.get_name()] = timestamp
            logger.info(
                "First TCP event received from %s. Initial dummy timestamp set to %s.",
                device.get_name(),
                timestamp,
            )
            return

        logger.info("checking standby status of %s", device.get_name())

        # update the timestamp in this dict each call. Once all devices have verifiably
        # entered standby and exited standby synchronously, we will overwrite the
        # initial_timestamps dict to reset t0 for each device
        self.synced_timestamps[device.get_name()] = timestamp

        # handle the case that this isn't the first call. If the passed timestamp
        # is equal to the timestamp in the dict, that means we've received two
        # TCP events from the device without the device timestamp updating, which
        # means the device has timed out and can be considered to be in standby mode

        # *NOTE* This uses `timestamp` from `_extract_timestamp_from_tcp_message` for synchronization check
        if t0 == timestamp:
            self.standby_mode_device_status[device.get_name()] = True
            logger.info("%s is in standby", device.get_name())
            return
        else:
            self.standby_mode_device_status[device.get_name()] = False
            logger.info("%s has exited in standby", device.get_name())
            return

    def update_repetition_rate(self, new_repetition_rate) -> None:
        """
        Update the repetition rate for the scan.

        Parameters
        ----------
        new_repetition_rate : float
            The new repetition rate in Hz (shots per second).
        """
        self.repetition_rate = new_repetition_rate

    @staticmethod
    def _extract_timestamp_from_tcp_message(message: str, device: GeecsDevice) -> float:
        """
        Extract the acquisition timestamp from a TCP message sent by a GEECS device.

        This function wraps the message into a `NetworkMessage`, passes it through the device's
        subscription handler, and extracts the `'acq_timestamp'` field from the parsed result.
        If no timestamp is present, the current system time is used as a fallback.

        Parameters
        ----------
        message : str
            The raw TCP message received from the device.
        device : GeecsDevice
            The device object responsible for parsing and interpreting the message.

        Returns
        -------
        float
            The extracted timestamp (in seconds). If unavailable, falls back to the system time.

        Notes
        -----
        - The `acq_timestamp` is critical for synchronizing and logging device data.
        - If `None` is returned by the parser, the fallback to system time is logged as a warning.
        - Fallback values may affect synchronization accuracy and should be used with caution.
        """
        stamp_str = datetime.now().__str__()
        net_msg = mh.NetworkMessage(tag=device.get_name(), stamp=stamp_str, msg=message)
        parsed = device.handle_subscription(net_msg)
        current_timestamp = parsed[2].get("acq_timestamp")

        if current_timestamp is None:
            logger.warning(
                "No timestamp found for %s. Using system time instead.",
                device.get_name(),
            )
            current_timestamp = _now()

        return float(current_timestamp)

    def _register_event_logging(
        self, log_update: Callable[[str, GeecsDevice], None]
    ) -> None:
        """
        Register a logging callback for all event-driven observables.

        This sets up the `log_update` function to be triggered whenever an update
        event occurs on any registered event-driven observable.

        Parameters
        ----------
        log_update : Callable[[str, GeecsDevice], None]
            The callback function to invoke on observable updates. It must accept
            a string message and a GeecsDevice instance.
        """
        for device_name, device in self.device_manager.devices.items():
            for observable in self.event_driven_observables:
                if observable.startswith(device_name):
                    logger.info(
                        "Registering logging for event-driven observable: %s",
                        observable,
                    )
                    # Ensure the device wires its TCP → event publisher callback
                    device.register_update_listener(
                        "logger", lambda msg, dev=device: log_update(msg, dev)
                    )

    def _calculate_elapsed_time(
        self, device: GeecsDevice, current_timestamp: float
    ) -> float:
        """
        Calculate the elapsed time since the initial timestamp for a given device.

        The result is rounded to the nearest multiple of 1/repetition_rate to ensure
        consistency with scan timing intervals.

        Parameters
        ----------
        device : GeecsDevice
            The device for which to calculate elapsed time.
        current_timestamp : float
            The current acquisition timestamp reported by the device.

        Returns
        -------
        float
            The rounded elapsed time (in seconds) since the initial timestamp.
        """
        t0 = self.initial_timestamps[device.get_name()]
        elapsed_time = current_timestamp - t0
        return round(elapsed_time * self.repetition_rate) / self.repetition_rate

    def _check_duplicate_timestamp(
        self, device: GeecsDevice, current_timestamp: float
    ) -> bool:
        """
        Determine whether the given timestamp has already been logged for the device.

        If the timestamp matches the last recorded timestamp for the device, it is
        considered a duplicate and should not be logged again.

        Parameters
        ----------
        device : GeecsDevice
            The device to check for duplicate timestamp.
        current_timestamp : float
            The timestamp to compare against the last logged timestamp.

        Returns
        -------
        bool
            True if the timestamp is a duplicate and should be skipped; False otherwise.
        """
        if (
            device.get_name() in self.last_timestamps
            and self.last_timestamps[device.get_name()] == current_timestamp
        ):
            logger.info(
                "Timestamp hasn't changed for %s. Skipping log.", device.get_name()
            )
            return True
        self.last_timestamps[device.get_name()] = current_timestamp
        return False

    def _update_async_observables(
        self, async_observables: list, elapsed_time: float
    ) -> None:
        """
        Update the log entries for asynchronous observables at the specified elapsed time.

        This method retrieves the latest values for each asynchronous observable from the
        associated device and stores them in the `log_entries` dictionary. It handles both
        standard and composite devices, including their sub-components.

        Parameters
        ----------
        async_observables : list of str
            List of asynchronous observables to update, specified as
            "device_name" or "device_name:variable_name".
        elapsed_time : float
            Time (in seconds) since the start of logging, used as the key
            in the `log_entries` dictionary.
        """
        for observable in async_observables:
            # Check if the observable has a variable specified (e.g., 'Dev1:var1')
            if ":" in observable:
                device_name, var_name = observable.split(":")
            else:
                device_name = observable
                var_name = None

            device = self.device_manager.devices.get(device_name)

            if device:
                if device.is_composite:
                    # Handle composite devices
                    composite_value = device.state.get("composite_var", "N/A")
                    self.log_entries[elapsed_time][f"{device_name}:composite_var"] = (
                        composite_value
                    )
                    logger.info(
                        "Updated composite var %s:composite_var to %s for elapsed time %s.",
                        device_name,
                        composite_value,
                        elapsed_time,
                    )

                    # Log sub-component states
                    for comp in device.components:
                        sub_device_name = comp["device"]
                        sub_var_name = comp["variable"]
                        sub_device = device.sub_devices[sub_device_name]["instance"]

                        if sub_device:
                            sub_value = sub_device.state.get(sub_var_name, "N/A")
                            self.log_entries[elapsed_time][
                                f"{sub_device_name}:{sub_var_name}"
                            ] = sub_value
                            logger.info(
                                "Updated sub-component %s:%s to %s for elapsed time %s.",
                                sub_device_name,
                                sub_var_name,
                                sub_value,
                                elapsed_time,
                            )
                        else:
                            logger.warning(
                                "Sub-device %s not found for %s.",
                                sub_device_name,
                                device_name,
                            )
                else:
                    # Handle regular devices
                    if var_name is None:
                        logger.warning(
                            "No variable specified for device %s. Skipping.",
                            device_name,
                        )
                        continue

                    value = device.state.get(var_name, "N/A")
                    self.log_entries[elapsed_time][f"{device_name}:{var_name}"] = value
                    logger.info(
                        "Updated async var %s:%s to %s for elapsed time %s.",
                        device_name,
                        var_name,
                        value,
                        elapsed_time,
                    )

                    # Handle file movement for async devices that save non-scalar data
                    if device_name in self.device_save_paths_mapping:
                        acq_timestamp = device.state.get("acq_timestamp")
                        if acq_timestamp is not None:
                            cfg = self.device_save_paths_mapping[device_name]
                            task = FileMoveTask(
                                source_dir=cfg["source_dir"],
                                target_dir=cfg["target_dir"],
                                device_name=device_name,
                                device_type=cfg["device_type"],
                                expected_timestamp=float(acq_timestamp),
                                shot_index=self.shot_index,
                            )
                            self.file_mover.move_files_by_timestamp(task)
                            logger.info(
                                "Created FileMoveTask for async device %s with timestamp %s for shot %s.",
                                device_name,
                                acq_timestamp,
                                self.shot_index,
                            )
            else:
                logger.warning(
                    "Device %s not found in DeviceManager. Skipping %s.",
                    device_name,
                    observable,
                )

    def _log_device_data(self, device: GeecsDevice, elapsed_time: float) -> None:
        """
        Log the data for a device during an event-driven observation.

        If the elapsed time is not yet recorded in `log_entries`, a new entry is created,
        including metadata like scan number, bin number, virtual variable (if present),
        and asynchronous observables. A beep sound is played, and the shot index is incremented.

        Then, data from the current device's observables is added to the entry. If the
        device is configured for file movement, a `FileMoveTask` is constructed and enqueued.

        Parameters
        ----------
        device : GeecsDevice
            The device whose state is being logged.
        elapsed_time : float
            The time (in seconds) since logging began, used as the key in `log_entries`.

        Side Effects
        ------------
        - Updates `log_entries` with device data and metadata.
        - Sets `shot_save_event` to signal that a shot occurred.
        - Triggers an audible beep via `sound_player`.
        - Increments `shot_index`.
        - Enqueues a file move task to the `FileMover` (if configured).
        """
        with self.lock:
            observables_data = {
                observable.split(":")[1]: device.state.get(observable.split(":")[1], "")
                for observable in self.event_driven_observables
                if observable.startswith(device.get_name())
            }
            if elapsed_time not in self.log_entries:
                logger.info(
                    "elapsed time in sync devices %s. reported by %s",
                    elapsed_time,
                    device.get_name(),
                )
                self.log_entries[elapsed_time] = {"Elapsed Time": elapsed_time}
                # Log configuration variables (such as 'bin') only when a new entry is created
                # bin number is updated in scan_manager
                self.log_entries[elapsed_time]["Bin #"] = self.bin_num
                self.log_entries[elapsed_time]["scan"] = self.scan_number

                if self.virtual_variable_name is not None:
                    self.log_entries[elapsed_time][self.virtual_variable_name] = (
                        self.virtual_variable_value
                    )

                # Increment shot index and update logging context before processing devices
                # This ensures both sync and async devices use the same shot number
                self.shot_index += 1
                update_context({"shot_id": str(self.shot_index)})

                # Update with async observable values
                self._update_async_observables(
                    self.device_manager.async_observables, elapsed_time
                )

                # TODO move the on-shot tdms writer functionality from scan manager to here

                # Set a flag to tell scan manager that a shot occurred
                self.shot_save_event.set()

                # Trigger the beep in the background
                self.sound_player.play_beep()  # Play the beep sound

            self.log_entries[elapsed_time].update(
                {
                    f"{device.get_name()}:{key}": value
                    for key, value in observables_data.items()
                }
            )

            if device.get_name() in self.device_save_paths_mapping:
                device_name = device.get_name()
                cfg = self.device_save_paths_mapping[device_name]
                task = FileMoveTask(
                    source_dir=cfg["source_dir"],
                    target_dir=cfg["target_dir"],
                    device_name=device_name,
                    device_type=cfg["device_type"],
                    expected_timestamp=observables_data[
                        "acq_timestamp"
                    ],  # *NOTE* `acq_timestamp` for data logging
                    shot_index=self.shot_index,
                )
                self.file_mover.move_files_by_timestamp(task)

    def stop_logging(self) -> None:
        """
        Stop the logging process and clean up resources.

        This method performs the following tasks:
        - Unregisters event handlers from all devices to stop event-driven logging.
        - Plays a sound to indicate that logging has stopped.
        - Signals the polling thread to stop asynchronous logging.
        - Stops the sound player.
        - Clears the stop event to allow future logging sessions to start cleanly.

        Note
        ----
        The shutdown of the file_mover happens in scan_manager for an order of operations
        issue that could be resolved.
        """
        # Unregister all event-driven logging
        for device_name, device in self.device_manager.devices.items():
            device.unregister_update_listener("logger")

        self.sound_player.play_toot()

        # Signal to stop the polling thread
        self.stop_event.set()

        self.sound_player.stop()
        # TODO check if this needs to be moved.  It might be cleared before the stop is registered
        # Reset the stop_event for future logging sessions

        self.stop_event.clear()

    def reinitialize_sound_player(self, options: Optional[dict] = None) -> None:
        """
        Reinitialize the sound player with optional settings.

        This stops the current `SoundPlayer` instance and replaces it with a new one
        using the provided options. The `shot_index` is also reset to 0.

        Parameters
        ----------
        options : dict, optional
            Dictionary of options to pass to the `SoundPlayer` constructor.

        Returns
        -------
        None
        """
        self.sound_player.stop()
        self.sound_player = SoundPlayer(options=options)
        self.shot_index = 0

    def get_current_shot(self) -> int:
        """
        Get the current shot index.

        Used for progress tracking during data acquisition.

        Returns
        -------
        int
            The current shot index.
        """
        return self.shot_index

    def synchronize_devices_global_time(self) -> bool:
        """
        Attempt to synchronize devices using global time synchronization.

        This method leverages improved Windows domain time synchronization to check
        if devices are already synchronized based on their current acquisition timestamps.
        If all device timestamps are within tolerance, synchronization is considered
        successful and the timeout-based method can be skipped entirely.

        Returns
        -------
        bool
            True if global time synchronization was successful, False if fallback
            to timeout method is needed.

        Notes
        -----
        - Requires that Windows domain time sync is working well (~10ms accuracy)
        - Uses configurable tolerance for timestamp comparison
        - Sets initial_timestamps and devices_synchronized flags on success
        - Provides significant time savings by avoiding timeout waits
        """
        logger.info("Attempting global time synchronization")

        # Collect current timestamps from all synchronous devices
        current_timestamps = {}
        for device_name in self.synchronous_device_names:
            timestamp = self._get_current_device_timestamp(device_name)
            if timestamp is None:
                logger.warning(
                    "Failed to get timestamp from device %s. Falling back to timeout method.",
                    device_name,
                )
                return False
            current_timestamps[device_name] = timestamp
            logger.debug("Device %s current timestamp: %s", device_name, timestamp)

        # Check if all timestamps are within tolerance
        if self._timestamps_within_tolerance(current_timestamps):
            # Devices are already synchronized!
            self.initial_timestamps = current_timestamps.copy()
            self.synced_timestamps = current_timestamps.copy()
            self.devices_synchronized = True
            logger.info(
                "Global time sync successful. Devices already synchronized with timestamps: %s",
                current_timestamps,
            )
            return True
        else:
            logger.info(
                "Device timestamps not within tolerance. Falling back to timeout method."
            )
            logger.debug("Timestamps were: %s", current_timestamps)
            return False

    def _get_current_device_timestamp(self, device_name: str) -> Optional[float]:
        """
        Get the current acquisition timestamp from a specific device.

        This method retrieves the latest timestamp from a device by accessing
        its current state or triggering a status update.

        Parameters
        ----------
        device_name : str
            Name of the device to get timestamp from.

        Returns
        -------
        float or None
            The current acquisition timestamp from the device, or None if
            the timestamp could not be retrieved.

        Notes
        -----
        - Uses the device manager to access device instances
        - Attempts to get 'acq_timestamp' from device state
        - Returns None if device not found or timestamp unavailable
        """
        device = self.device_manager.devices.get(device_name)
        if device is None:
            logger.warning("Device %s not found in device manager", device_name)
            return None

        try:
            # Try to get the acquisition timestamp from device state
            timestamp = device.state.get("acq_timestamp")
            if timestamp is not None:
                return float(timestamp)

            # If not available in state, try to trigger an update
            # This might involve calling a device-specific method to get current status
            logger.debug(
                "No acq_timestamp in state for %s, attempting to get current value",
                device_name,
            )

            # For now, return None if timestamp not readily available
            # This could be enhanced to actively query the device
            return None

        except Exception:
            logger.exception("Error getting timestamp from device %s", device_name)
            return None

    def _timestamps_within_tolerance(self, timestamps: Dict[str, float]) -> bool:
        """
        Check if all device timestamps are within acceptable tolerance of each other.

        This method determines if devices are synchronized by comparing their
        acquisition timestamps. Static offsets between devices are acceptable
        as long as they are consistent.

        Parameters
        ----------
        timestamps : dict
            Dictionary mapping device names to their current timestamps.

        Returns
        -------
        bool
            True if all timestamps are within tolerance, False otherwise.

        Notes
        -----
        - Uses configurable tolerance (default 50ms)
        - Handles the case where devices have consistent static offsets
        - Compares all timestamps against the first device's timestamp
        """
        if len(timestamps) < 2:
            logger.info("Only one or no devices to synchronize")
            return True

        tolerance_seconds = self.global_sync_tol_ms / 1000.0

        # Get reference timestamp (first device)
        reference_timestamp = next(iter(timestamps.values()))
        reference_device = next(iter(timestamps.keys()))

        logger.debug(
            "Checking timestamp tolerance with reference device %s (timestamp: %s), tolerance: %s ms",
            reference_device,
            reference_timestamp,
            self.global_sync_tol_ms,
        )

        # Check all other timestamps against reference
        for device_name, timestamp in timestamps.items():
            if device_name == reference_device:
                continue

            time_diff = abs(timestamp - reference_timestamp)
            logger.debug(
                "Device %s timestamp difference from reference: %s ms",
                device_name,
                time_diff * 1000,
            )

            if time_diff > tolerance_seconds:
                logger.info(
                    "Device %s timestamp differs by %s ms (> %s ms tolerance)",
                    device_name,
                    time_diff * 1000,
                    self.global_sync_tol_ms,
                )
                return False

        logger.info(
            "All device timestamps within %s ms tolerance", self.global_sync_tol_ms
        )
        return True
