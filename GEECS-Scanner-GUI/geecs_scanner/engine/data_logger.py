"""Per-shot data logging, device synchronization, and file management for ScanManager."""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time as _now
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from . import DeviceManager
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.tools.files.timestamping import extract_timestamp_from_file
import geecs_python_api.controls.interface.message_handling as mh
from geecs_scanner.logging_setup import update_context
from geecs_scanner.utils import SoundPlayer
from geecs_scanner.utils.exceptions import DataFileError
from geecs_scanner.utils.retry import retry
from geecs_scanner.engine.models.scan_options import ScanOptions

DeviceSavePaths = Dict[str, Dict[str, Union[Path, str]]]

logger = logging.getLogger(__name__)


@dataclass
class FileMoveTask:
    """Task definition for moving and renaming a device file after acquisition."""

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
    retry_count: int = 0  # Number of times this task has been re-queued
    max_retries: int = 2  # Maximum number of re-queue attempts before orphaning
    files_found_so_far: int = (
        0  # Cumulative count of files matched and moved across all retries
    )


class FileMover:
    """Rename and relocate device files using background worker threads.

    Files are matched by acquisition timestamp: when a device saves a file after
    a shot, the filename encodes the system timestamp, which is also recorded in
    scalar data.  Workers extract that timestamp, compare it against the expected
    value, and rename the file to the scan's shot-number convention.

    ``save_local=True`` (the default) points devices at ``C:/SharedData`` on
    their host rather than the network share, avoiding write-time bottlenecks.

    Attributes
    ----------
    task_queue : queue.Queue
    stop_event : threading.Event
    workers : list[threading.Thread]
    file_check_counts : dict
        How many times each file has been examined; files exceeding the limit
        are moved to ``orphaned_files`` to prevent infinite re-checking.
    orphaned_files : set
    orphan_tasks : list
    processed_files : set
    scan_is_live : bool
    save_local : bool
    scan_number : int or None
    """

    def __init__(self, num_workers: int = 16) -> None:
        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.workers = []
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_func, daemon=True)
            worker.start()
            self.workers.append(worker)

        self.file_check_counts = {}
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
        """Drain the task queue until stopped."""
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
        """Locate the file matching *task*'s timestamp and move it to the target dir."""
        # Give the device time to finish writing before retrying.  The sleep
        # lives here (in the worker) rather than in move_files_by_timestamp so
        # that _post_process_orphan_task can queue all orphan tasks at once and
        # let the 16 workers drain them in parallel instead of serially.
        if task.retry_count > 0:
            time.sleep(0.5)

        source_dir = task.source_dir
        target_dir = task.target_dir
        device_name = task.device_name
        device_type = task.device_type
        expected_timestamp = task.expected_timestamp
        shot_index = task.shot_index

        home_dir = source_dir.parent

        # For MagSpec devices, only search the primary folder (exact match).
        try:
            if device_type in ["MagSpecStitcher", "MagSpecCamera"]:
                variant_dirs = [
                    d
                    for d in home_dir.iterdir()
                    if d.is_dir() and d.name == device_name
                ]
            else:
                variant_dirs = [
                    d
                    for d in home_dir.iterdir()
                    if d.is_dir() and d.name.startswith(device_name)
                ]
        except OSError as exc:
            raise DataFileError(
                f"Cannot read source directory {home_dir} for device {device_name}"
            ) from exc

        # Determine expected file count.
        if device_type in [
            "PicoscopeV2",
            "FROG",
            "Thorlabs CCS175 Spectrometer",
            "RohdeSchwarz_RTA4000",
            "ThorlabsWFS",
        ]:
            expected_file_count = 2
        elif device_type in ["MagSpecStitcher", "MagSpecCamera"]:
            expected_file_count = 1
        else:
            expected_file_count = 1

        # There is some time overhead in saving to the netapp rather than local. It's possible
        # that the FileMoveTask for a given event is created and queued before the file is
        # successfully written to disk. Tasks that fail to find a matching file are now
        # re-queued up to `task.max_retries` times before being marked as orphan tasks,
        # which are retried during post-scan cleanup.
        if not self.save_local:
            time.sleep(0.1)

        # Track found files across ALL variant directories, not per-variant.
        # For devices like FROG, the expected files are spread across multiple
        # variant directories (e.g., -Temporal and -Spatial), so we need to
        # accumulate the count across all of them.
        task_success = False
        found_files_count = 0
        for variant in variant_dirs:
            adjusted_target_dir = target_dir.parent / variant.name
            adjusted_target_dir.mkdir(parents=True, exist_ok=True)

            for file in variant.glob("*"):
                try:
                    if not file.is_file():
                        continue
                except OSError:
                    # File is locked for writing by the device; skip and let
                    # the retry or orphan sweep pick it up later.
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
                        logger.debug(
                            "File %s checked >1 times; marking as orphaned.", file
                        )
                        self.orphaned_files.add(file)
                        continue

                file_ts = extract_timestamp_from_file(file, device_type)
                if abs(file_ts - expected_timestamp) < 0.0011:
                    found_files_count += 1
                    task.files_found_so_far += 1

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

                    if task.files_found_so_far >= expected_file_count:
                        task_success = True
                        break

            if task_success:
                break

        if not task_success:
            if self.scan_is_live and task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.debug(
                    "Re-queuing task for %s with timestamp %s (retry %d/%d)",
                    task.device_name,
                    task.expected_timestamp,
                    task.retry_count,
                    task.max_retries,
                )
                self.move_files_by_timestamp(task)
            else:
                logger.warning(
                    "Failed to find a file for %s with timestamp %s — will orphan.",
                    task.device_name,
                    task.expected_timestamp,
                )
                self.orphan_tasks.append(task)

    def _move_file(
        self, task: FileMoveTask, source_file: Path, new_device_name: str
    ) -> None:
        """Move *source_file* to the target dir, retrying on transient network-share errors.

        Raises
        ------
        DataFileError
            If the file cannot be moved after all retry attempts.
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
            retry(
                lambda: shutil.move(str(source_file), str(dest_file)),
                attempts=3,
                delay=0.5,
                backoff=2.0,
                catch=(OSError,),
                on_retry=lambda exc, n: logger.debug(
                    "Retry %d moving %s → %s: %s", n, source_file.name, dest_file, exc
                ),
            )
            self.processed_files.add(dest_file)
        except OSError as exc:
            raise DataFileError(
                f"Failed to move {source_file.name} → {dest_file} after retries"
            ) from exc

    def _process_variant_file(self, task: FileMoveTask) -> None:
        """Find and move the variant file (e.g. ``-interpSpec``) matching *task.random_part*."""
        if task.suffix is None or task.random_part is None:
            logger.debug(
                "No suffix or random_part in task; skipping variant processing."
            )
            return

        variant_dir = task.source_dir.parent / f"{task.device_name}{task.suffix}"
        if not variant_dir.exists():
            logger.debug(
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
        """Return ``Scan{NNN}_{device}_{SSS}`` — the shot-number naming convention."""
        scan_number_str = str(scan_number).zfill(3)
        shot_number_str = str(shot_index).zfill(3)
        return f"Scan{scan_number_str}_{device_name}_{shot_number_str}"

    def move_files_by_timestamp(self, task: FileMoveTask) -> None:
        """Enqueue *task* for processing by a worker thread."""
        self.task_queue.put(task)

    def _post_process_orphaned_files(
        self, log_df: pd.DataFrame, device_save_paths_mapping: dict
    ) -> None:
        """Match unprocessed files on disk against log timestamps and re-queue them."""
        logger.info("looking to handle orphaned data files")
        tolerance = 0.0011
        for device_name, device_info in device_save_paths_mapping.items():
            source_dir = Path(device_info["source_dir"])
            target_dir = Path(device_info["target_dir"])
            device_type = device_info["device_type"]

            # *NOTE* Using `acq_timestamp` for data that was logged
            acq_col = f"{device_name} acq_timestamp"
            if acq_col not in log_df.columns:
                logger.warning(
                    "No acq_timestamp column for %s in log_df; skipping orphan sweep.",
                    device_name,
                )
                continue

            shot_timestamp_pairs = [
                (row["Shotnumber"], row[acq_col])
                for _, row in log_df.iterrows()
                if pd.notnull(row[acq_col])
            ]

            if not shot_timestamp_pairs:
                logger.debug(
                    "No valid timestamps for %s; skipping orphan sweep.", device_name
                )
                continue

            home_dir = source_dir.parent
            try:
                variant_dirs = [
                    d
                    for d in home_dir.iterdir()
                    if d.is_dir() and d.name.startswith(device_name)
                ]
            except OSError:
                logger.warning(
                    "Cannot iterate %s for orphan sweep of %s; skipping.",
                    home_dir,
                    device_name,
                )
                continue

            orphan_files = []
            for variant_dir in variant_dirs:
                for f in variant_dir.glob("*"):
                    if f.is_file() and f not in self.processed_files:
                        orphan_files.append(f)

            logger.debug(
                "Found %d candidate orphan files for %s across %d variant dirs.",
                len(orphan_files),
                device_name,
                len(variant_dirs),
            )

            for file in orphan_files:
                file_ts = extract_timestamp_from_file(file, device_type)
                matched_shot = None

                for shot_number, ts in shot_timestamp_pairs:
                    if abs(file_ts - ts) < tolerance:
                        matched_shot = int(shot_number)
                        break

                if matched_shot is not None:
                    logger.debug(
                        "Found orphan file %s matching shot %s", file, matched_shot
                    )
                    random_part = file.stem.replace(f"{device_name}_", "")
                    task = FileMoveTask(
                        source_dir=file.parent,
                        target_dir=target_dir,
                        device_name=device_name,
                        device_type=device_type,
                        expected_timestamp=file_ts,
                        shot_index=matched_shot,
                        random_part=random_part,
                    )
                    self.move_files_by_timestamp(task)
                else:
                    logger.warning(
                        "No matching shot number found for orphan file %s (timestamp %s)",
                        file,
                        file_ts,
                    )

    def _post_process_orphan_task(self):
        """Re-queue all previously failed tasks for a final move attempt."""
        orphan_snapshot = list(self.orphan_tasks)
        self.orphan_tasks.clear()
        for task in orphan_snapshot:
            task.retry_count = (
                0  # No new files are being written at scan end; skip the retry sleep.
            )
            self.move_files_by_timestamp(task)

    def shutdown(self, wait: bool = True) -> None:
        """Signal workers to stop; if *wait*, drain the queue first."""
        self.stop_event.set()
        if wait:
            self.task_queue.join()
        for _ in self.workers:
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join()
        logger.debug("FileMover has been shut down gracefully.")


class DataLogger:
    """Handle event-driven and polled data logging from GEECS devices during scans.

    Owned and reinitialized by ``ScanManager`` for each scan; not intended for
    standalone use.

    Attributes
    ----------
    device_manager : DeviceManager
    file_mover : FileMover or None
    log_entries : dict
        ``{elapsed_time: {column: value, ...}}`` — the live in-memory log.
    scan_number : int or None
    bin_num : int
        Set externally by ScanManager before each step.
    shot_index : int
    polling_interval : float
    repetition_rate : float
        Used to round elapsed times to the nearest shot interval.
    virtual_variable_name : str or None
    virtual_variable_value : float
    data_recording : bool
    idle_time : float
    lock : threading.Lock
    stop_event : threading.Event
    warning_timeout_sync : float
    warning_timeout_async_factor : float
    last_log_time_sync : dict
    last_log_time_async : dict
    last_timestamps : dict
    initial_timestamps : dict
    synced_timestamps : dict
    standby_mode_device_status : dict
        Per-device standby flag: ``None`` = unknown, ``True`` = in standby,
        ``False`` = exited standby (received trigger).
    device_save_paths_mapping : DeviceSavePaths
    synchronous_device_names : list[str]
    event_driven_observables : list[str]
    all_devices_in_standby : bool
    devices_synchronized : bool
    save_local : bool
    sound_player : SoundPlayer
    global_sync_tol_ms : float
    """

    def __init__(
        self, experiment_dir: Optional[str], device_manager: DeviceManager = None
    ):
        self.device_manager = device_manager or DeviceManager(experiment_dir)
        self.global_sync_tol_ms = 0

        self.stop_event = threading.Event()
        self.warning_timeout_sync = 2
        self.warning_timeout_async_factor = 1
        self.last_log_time_sync = {}
        self.last_log_time_async = {}
        self.polling_interval = 0.5
        self.results = {}

        self.log_entries = {}

        self.sound_player = SoundPlayer()
        self.shot_index = 0

        # Note: bin_num and scan_number are updated in ScanManager
        self.bin_num = 0
        self.scan_number = None

        self.virtual_variable_name = None
        self.virtual_variable_value = 0

        self.data_recording = False
        self.idle_time = 0

        self.lock = threading.Lock()

        self.repetition_rate = 1.0

        self.last_timestamps: Dict[str, float] = {}
        self.initial_timestamps: Dict[str, Optional[float]] = {}
        self.synced_timestamps: Dict[str, float] = {}
        self.standby_mode_device_status: Dict[str, Optional[bool]] = {}
        self.device_save_paths_mapping: DeviceSavePaths = {}

        self.file_mover: Optional[FileMover] = None
        self.synchronous_device_names: List[str] = []
        self.event_driven_observables: List[str] = []

        self.all_devices_in_standby: bool = False
        self.devices_synchronized: bool = False
        self.save_local = True

    def set_device_save_paths_mapping(self, mapping: DeviceSavePaths) -> None:
        """Inject the per-device save-path config from ScanManager."""
        self.device_save_paths_mapping = mapping

    def start_logging(self) -> Dict[str, Any]:
        """Reset state, create a FileMover, and register TCP callbacks.

        Returns
        -------
        dict
            Reference to ``self.log_entries`` (populated live during the scan).
        """
        self.last_timestamps = {}
        self.initial_timestamps = {}
        self.synced_timestamps = {}
        self.standby_mode_device_status = {}
        self.log_entries = {}

        self.file_mover = FileMover()

        self.all_devices_in_standby = False
        self.devices_synchronized = False
        self.synchronous_device_names = []

        self.sound_player.start_queue()

        # Scan number in datalogger updates in scan_manager
        self.file_mover.scan_number = self.scan_number
        self.file_mover.save_local = self.save_local

        self.event_driven_observables = self.device_manager.event_driven_observables

        for observable in self.event_driven_observables:
            device_name = observable.split(":")[0]
            if device_name not in self.synchronous_device_names:
                self.synchronous_device_names.append(device_name)

        self._register_event_logging(self._handle_TCP_message_from_device)

        logger.info("Logging has started for all event-driven devices.")

        return self.log_entries

    def _handle_TCP_message_from_device(
        self, message: str, device: GeecsDevice
    ) -> None:
        """Parse a TCP message and log a new shot if the timestamp advanced.

        TCP messages and events get generated constantly from a GEECS device. We need to
        determine if a message and event are originating from a triggered event or from a
        timeout event. This involves parsing out the device specific timestamp on the first
        TCP event. Then, we need to compare it to the timestamp from the following event.
        During a timeout event, the device timestamp is not updated as it represents the
        timestamp from the last successful acquisition. If we see that the timestamp is not
        updating, that means we know that the device has timed out and can be considered
        to be in standby mode. It is awaiting the next hardware trigger. So, the process
        here is to interpret the first few messages from each device to conclude everything
        is in standby mode and can be then be synchronized through a dedicated timing shot
        """
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
        """Return True if every synchronous device has ``standby_mode_device_status == True``."""
        device_names = set(self.synchronous_device_names)
        standby_keys = self.standby_mode_device_status.keys()

        all_in_dict = device_names.issubset(standby_keys)
        all_on = all(
            self.standby_mode_device_status.get(device, False)
            for device in self.synchronous_device_names
        )

        if all_in_dict and all_on:
            logger.debug(
                "All device names are present in standby_mode_device_status dict and all have True status."
            )
            return True
        else:
            logger.debug(
                "Not all devices are in standby: %s", self.standby_mode_device_status
            )

            return False

    def _check_all_exited_standby_status(self) -> bool:
        """Return True if every synchronous device has ``standby_mode_device_status == False``."""
        device_names = set(self.synchronous_device_names)
        standby_keys = self.standby_mode_device_status.keys()

        all_in_dict = device_names.issubset(standby_keys)
        all_off = all(
            not self.standby_mode_device_status.get(device, True)
            for device in self.synchronous_device_names
        )

        if all_in_dict and all_off:
            logger.debug(
                "All device names are present in standby_mode_device_status dict and all "
                "have False status meaning they have exited standby mode."
            )
            return True
        else:
            return False

    def _check_device_standby_mode_status(
        self, device: GeecsDevice, timestamp: float
    ) -> None:
        """Update *device*'s standby flag based on whether its timestamp changed.

        Parameters
        ----------
        device : GeecsDevice
        timestamp : float
            Latest timestamp from the device's TCP message.
        """
        # TODO: statuses are bit wonky and could be cleaned up. Right now, 'None'
        #  really means the status is unknown. "True" indicates the device verifiably
        #  went into standby mode. "False" indicates that device originally went into
        #  standby mode, and has since received a verifiable hardware trigger. So,
        #  once a "False" status has been flagged, it should stay that way until something
        #  explicitly turns it back to "None".

        if device.get_name() not in self.standby_mode_device_status:
            self.standby_mode_device_status[device.get_name()] = None
            self.initial_timestamps[device.get_name()] = None

        if self.standby_mode_device_status[device.get_name()] is False:
            return  # TODO may be worth to instead have a custom variable than can be one of three states

        t0 = self.initial_timestamps.get(device.get_name(), None)

        if t0 is None:
            self.initial_timestamps[device.get_name()] = timestamp
            logger.debug(
                "First TCP event received from %s. Initial dummy timestamp set to %s.",
                device.get_name(),
                timestamp,
            )
            return

        logger.debug("checking standby status of %s", device.get_name())

        # update the timestamp in this dict each call. Once all devices have verifiably
        # entered standby and exited standby synchronously, we will overwrite the
        # initial_timestamps dict to reset t0 for each device
        self.synced_timestamps[device.get_name()] = timestamp

        # *NOTE* This uses `timestamp` from `_extract_timestamp_from_tcp_message` for synchronization check
        if t0 == timestamp:
            self.standby_mode_device_status[device.get_name()] = True
            logger.debug("%s is in standby", device.get_name())
            return
        else:
            self.standby_mode_device_status[device.get_name()] = False
            logger.debug("%s has exited in standby", device.get_name())
            return

    def update_repetition_rate(self, new_repetition_rate) -> None:
        """Set the repetition rate (Hz) used to round elapsed times."""
        self.repetition_rate = new_repetition_rate

    @staticmethod
    def _extract_timestamp_from_tcp_message(message: str, device: GeecsDevice) -> float:
        """Parse ``acq_timestamp`` from *message*; falls back to system time if absent."""
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
        """Wire *log_update* as the TCP-event callback for all event-driven observables."""
        for device_name, device in self.device_manager.devices.items():
            for observable in self.event_driven_observables:
                if observable.startswith(device_name):
                    logger.debug(
                        "Registering logging for event-driven observable: %s",
                        observable,
                    )
                    device.register_update_listener(
                        "logger", lambda msg, dev=device: log_update(msg, dev)
                    )

    def _calculate_elapsed_time(
        self, device: GeecsDevice, current_timestamp: float
    ) -> float:
        """Return elapsed time since sync, rounded to the nearest shot interval."""
        t0 = self.initial_timestamps[device.get_name()]
        elapsed_time = current_timestamp - t0
        return round(elapsed_time * self.repetition_rate) / self.repetition_rate

    def _check_duplicate_timestamp(
        self, device: GeecsDevice, current_timestamp: float
    ) -> bool:
        """Return True if *current_timestamp* was already logged for *device*."""
        if (
            device.get_name() in self.last_timestamps
            and self.last_timestamps[device.get_name()] == current_timestamp
        ):
            logger.debug(
                "Timestamp hasn't changed for %s. Skipping log.", device.get_name()
            )
            return True
        self.last_timestamps[device.get_name()] = current_timestamp
        return False

    def _update_async_observables(
        self, async_observables: list, elapsed_time: float
    ) -> None:
        """Poll the current state of async devices and write values into the log entry."""
        for observable in async_observables:
            if ":" in observable:
                device_name, var_name = observable.split(":")
            else:
                device_name = observable
                var_name = None

            device = self.device_manager.devices.get(device_name)

            if device:
                if device.is_composite:
                    composite_value = device.state.get("composite_var", "N/A")
                    self.log_entries[elapsed_time][f"{device_name}:composite_var"] = (
                        composite_value
                    )
                    logger.debug(
                        "Updated composite var %s:composite_var to %s for elapsed time %s.",
                        device_name,
                        composite_value,
                        elapsed_time,
                    )

                    for comp in device.components:
                        sub_device_name = comp["device"]
                        sub_var_name = comp["variable"]
                        sub_device = device.sub_devices[sub_device_name]["instance"]

                        if sub_device:
                            sub_value = sub_device.state.get(sub_var_name, "N/A")
                            self.log_entries[elapsed_time][
                                f"{sub_device_name}:{sub_var_name}"
                            ] = sub_value
                            logger.debug(
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
                    if var_name is None:
                        logger.warning(
                            "No variable specified for device %s. Skipping.",
                            device_name,
                        )
                        continue

                    value = device.state.get(var_name, "N/A")
                    self.log_entries[elapsed_time][f"{device_name}:{var_name}"] = value
                    logger.debug(
                        "Updated async var %s:%s to %s for elapsed time %s.",
                        device_name,
                        var_name,
                        value,
                        elapsed_time,
                    )

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
                            logger.debug(
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
        """Write device observables into ``log_entries[elapsed_time]``.

        Creates a new entry on the first device report for a given *elapsed_time*,
        populating bin, scan, virtual-variable, and async-observable fields, then
        increments ``shot_index`` and enqueues a file move task if configured.
        Subsequent devices for the same shot merge their observables into the same entry.
        """
        with self.lock:
            observables_data = {
                observable.split(":")[1]: device.state.get(observable.split(":")[1], "")
                for observable in self.event_driven_observables
                if observable.startswith(device.get_name())
            }
            if elapsed_time not in self.log_entries:
                logger.debug(
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
                logger.info("shot %s", self.shot_index)

                self._update_async_observables(
                    self.device_manager.async_observables, elapsed_time
                )

                # TODO move the on-shot tdms writer functionality from scan manager to here

                self.sound_player.play_beep()

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
        """Unregister TCP callbacks, play completion sound, and signal the stop event.

        Note
        ----
        FileMover shutdown is intentionally deferred to ScanManager to preserve
        the correct shutdown order (scalar data must be written before the workers
        drain, so the orphan sweep has a complete DataFrame to match against).
        """
        for device_name, device in self.device_manager.devices.items():
            device.unregister_update_listener("logger")

        self.sound_player.play_toot()

        self.stop_event.set()

        self.sound_player.stop()
        # TODO check if this needs to be moved.  It might be cleared before the stop is registered
        self.stop_event.clear()

    def reinitialize_sound_player(self, options: Optional[ScanOptions] = None) -> None:
        """Replace the SoundPlayer with a fresh instance and reset ``shot_index``."""
        self.sound_player.stop()
        randomized = options.randomized_beeps if options is not None else False
        self.sound_player = SoundPlayer(randomized_beeps=randomized)
        self.shot_index = 0

    def get_current_shot(self) -> int:
        """Return the current shot index for progress tracking."""
        return self.shot_index

    def synchronize_devices_global_time(self) -> bool:
        """Check if devices are already synchronized via Windows domain time.

        Windows domain time sync gives ~10ms accuracy, which is enough to skip the
        timeout-based standby/trigger handshake and save several seconds per scan.

        Returns
        -------
        bool
            True if all device timestamps are within ``global_sync_tol_ms`` and
            synchronization flags have been set; False if the timeout fallback is needed.
        """
        logger.debug("Attempting global time synchronization")

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
            logger.info("Device %s current timestamp: %s", device_name, timestamp)

        if self._timestamps_within_tolerance(current_timestamps):
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
            logger.info("Timestamps were: %s", current_timestamps)
            return False

    def _get_current_device_timestamp(self, device_name: str) -> Optional[float]:
        """Return ``acq_timestamp`` from *device_name*'s current state, or None."""
        device = self.device_manager.devices.get(device_name)
        if device is None:
            logger.warning("Device %s not found in device manager", device_name)
            return None

        try:
            timestamp = device.state.get("acq_timestamp")
            if timestamp is not None:
                return float(timestamp)

            logger.debug(
                "No acq_timestamp in state for %s, attempting to get current value",
                device_name,
            )
            return None

        except Exception:
            logger.exception("Error getting timestamp from device %s", device_name)
            return None

    def _timestamps_within_tolerance(self, timestamps: Dict[str, float]) -> bool:
        """Return True if all timestamps in *timestamps* are within ``global_sync_tol_ms``."""
        if len(timestamps) < 2:
            logger.debug("Only one or no devices to synchronize")
            return True

        tolerance_seconds = self.global_sync_tol_ms / 1000.0

        reference_timestamp = next(iter(timestamps.values()))
        reference_device = next(iter(timestamps.keys()))

        logger.debug(
            "Checking timestamp tolerance with reference device %s (timestamp: %s), tolerance: %s ms",
            reference_device,
            reference_timestamp,
            self.global_sync_tol_ms,
        )

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
                logger.debug(
                    "Device %s timestamp differs by %s ms (> %s ms tolerance)",
                    device_name,
                    time_diff * 1000,
                    self.global_sync_tol_ms,
                )
                return False

        logger.debug(
            "All device timestamps within %s ms tolerance", self.global_sync_tol_ms
        )
        return True
