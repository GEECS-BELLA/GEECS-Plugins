from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Union

DeviceSavePaths = Dict[str, Dict[str, Union[Path,str]]]

import time
import threading
from datetime import datetime
import logging
import pandas as pd

from . import DeviceManager
from geecs_scanner.utils import SoundPlayer

from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.tools.files.timestamping import extract_timestamp_from_file
from geecs_python_api.controls.interface.geecs_errors import ErrorAPI
import geecs_python_api.controls.interface.message_handling as mh

import queue
import shutil

from dataclasses import dataclass


@dataclass
class FileMoveTask:
    source_dir: Path
    target_dir: Path
    device_name: str
    device_type: str
    expected_timestamp: float
    shot_index: int
    random_part: Optional[str] = None  # Unique identifier extracted from primary file.
    suffix: Optional[str] = None  # For variant processing (e.g. "-interpSpec" or "-interpDiv")
    new_name: Optional[str] = None  # The standardized file stem (e.g., "Scan001_DeviceA_005")


class FileMover:
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

        self.scan_number: Optional[int] = None
        logging.info("FileMover worker started.")

    def _worker_func(self) -> None:
        """Process file move tasks from the queue until stopped."""
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
            except Exception as e:
                logging.error(f"Error processing task: {e}")
            finally:
                self.task_queue.task_done()
        logging.info("FileMover worker stopped.")

    def _process_task(self, task: FileMoveTask) -> None:
        """
        Process a FileMoveTask:
          1. Identify the relevant variant directories under the parent of source_dir.
          2. For each file in those directories, check if its timestamp matches the expected timestamp.
          3. For matching primary files, extract the unique random part and generate a new name.
          4. Move the primary file, and for MagSpec devices, process variant files using the random part.
        """
        source_dir = task.source_dir
        target_dir = task.target_dir
        device_name = task.device_name
        device_type = task.device_type
        expected_timestamp = task.expected_timestamp
        shot_index = task.shot_index

        home_dir = source_dir.parent

        # For MagSpec devices, only search the primary folder (exact match).
        if device_type in ['MagSpecStitcher', 'MagSpecCamera']:
            variant_dirs = [d for d in home_dir.iterdir() if d.is_dir() and d.name == device_name]
        else:
            variant_dirs = [d for d in home_dir.iterdir() if d.is_dir() and d.name.startswith(device_name)]

        # Determine expected file count.
        if device_type in ['PicoscopeV2', 'FROG', 'Thorlabs CCS175 Spectrometer','RohdeSchwarz_RTA4000']:
            expected_file_count = 2
        elif device_type in ['MagSpecStitcher', 'MagSpecCamera']:
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
            time.sleep(.1)

        task_success = False
        for variant in variant_dirs:
            task_success = False
            adjusted_target_dir = target_dir.parent / variant.name
            adjusted_target_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Processing variant '{variant.name}' with target '{adjusted_target_dir}'")

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
                    self.file_check_counts[file] = self.file_check_counts.get(file, 0) + 1
                    if self.file_check_counts[file] > 1:
                        logging.info(f"File {file} checked >1 times; marking as orphaned.")
                        self.orphaned_files.add(file)
                        continue

                file_ts = extract_timestamp_from_file(file, device_type)
                logging.info(f"Checking {file} with timestamp {file_ts} against expected {expected_timestamp}")
                if abs(file_ts - expected_timestamp) < 0.0011:
                    found_files_count += 1

                    # Extract the unique random part from the primary file.
                    # Expects filename format: "{device_name}_{random}.png"
                    random_part = file.stem.replace(f"{device_name}_", "")
                    task.random_part = random_part

                    # Generate the new standardized file stem.
                    task.new_name = self.rename_file(self.scan_number, device_name, shot_index)

                    # Move the primary file.
                    self._move_file(task, file, variant.name)

                    # For MagSpec devices, process additional variant files.
                    if device_type in ['MagSpecStitcher', 'MagSpecCamera']:
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
            #task failed to find a match, log it as an orpahned task
            logging.info(f'failed to find a file for {task.device_name} with timestamp {task.expected_timestamp}')
            self.orphan_tasks.append(task)

    def _move_file(self, task: FileMoveTask, source_file: Path, new_device_name: str) -> bool:
        """
        Rename and move a file using information from task.

        Args:
            task (FileMoveTask): The task containing parameters.
            source_file (Path): The file to move.
            new_device_name (str): Device name (or variant name) to use in renaming.

        Returns:
            bool: True if moved successfully, False otherwise.
        """
        target_dir = task.target_dir.parent / new_device_name
        target_dir.mkdir(parents=True, exist_ok=True)
        new_file_stem = task.new_name if task.new_name else self.rename_file(self.scan_number, new_device_name,
                                                                             task.shot_index)
        new_filename = new_file_stem + source_file.suffix
        dest_file = target_dir / new_filename
        try:
            shutil.move(str(source_file), str(dest_file))
            logging.info(f"Moved {source_file} to {dest_file}")
            self.processed_files.add(dest_file)
            return True
        except Exception as e:
            logging.error(f"Error moving {source_file} to {dest_file}: {e}")
            return False

    def _process_variant_file(self, task: FileMoveTask) -> None:
        """
        Process a file in a variant directory based on task.suffix.

        The variant directory is constructed as: f"{device_name}{task.suffix}".
        Searches for a file whose name contains task.random_part,
        renames it using task.new_name (with task.suffix appended),
        and moves it to the corresponding target directory.
        """
        if task.suffix is None or task.random_part is None:
            logging.info("No suffix or random_part in task; skipping variant processing.")
            return

        variant_dir = task.source_dir.parent / f"{task.device_name}{task.suffix}"
        if not variant_dir.exists():
            logging.info(f"Variant directory {variant_dir} does not exist; skipping processing for {task.suffix}.")
            return

        candidate = None
        for file in variant_dir.glob("*"):
            if file.is_file() and task.random_part in file.stem:
                candidate = file
                break

        if candidate is None:
            logging.warning(f"No file found in {variant_dir} containing '{task.random_part}'.")
            return

        new_device_name = f"{task.device_name}{task.suffix}"
        self._move_file(task, candidate, new_device_name)

    @staticmethod
    def rename_file(scan_number: int, device_name: str, shot_index: int) -> str:
        """
        Generate a new file stem based on scan number, device name, and shot index.

        Args:
            scan_number (int): Scan number (zero-padded to 3 digits).
            device_name (str): Device name.
            shot_index (int): Shot number (zero-padded to 3 digits).

        Returns:
            str: New file name stem.
        """
        scan_number_str = str(scan_number).zfill(3)
        shot_number_str = str(shot_index).zfill(3)
        return f"Scan{scan_number_str}_{device_name}_{shot_number_str}"

    def move_files_by_timestamp(self, task: FileMoveTask) -> None:
        """Enqueue a file move task."""
        self.task_queue.put(task)

    def post_process_orphaned_files(self, log_df: pd.DataFrame, device_save_paths_mapping: dict) -> None:
        """
        Process orphaned files for each device using the saved device mapping and logged timestamps.
        For each device, we look recursively through the source directory for files whose names
        contain the device name, extract their timestamp, and then match that timestamp with the
        one in the DataFrame to determine the correct shot number. With that information, we then
        create a FileMoveTask and process the file accordingly.

        Args:
            log_df (pd.DataFrame): DataFrame containing a 'shotnumber' column and a column for each device's timestamp,
                                   e.g. "DeviceA SysTimestamp".
            device_save_paths_mapping (dict): Mapping of device names to their save path information.
        """
        logging.info(f'looking to handle orphaned data files')
        tolerance = 0.0011  # Adjust as needed
        for device_name, device_info in device_save_paths_mapping.items():
            source_dir = Path(device_info['source_dir'])
            target_dir = Path(device_info['target_dir'])
            device_type = device_info['device_type']

            # Create a list of (shotnumber, timestamp) pairs from the df. Ensure the df columns are named appropriately.
            # *NOTE* Using `SysTimestamp` for data that was logged
            shot_timestamp_pairs = [
                (row['Shotnumber'], row[f'{device_name} acq_timestamp'])
                for _, row in log_df.iterrows()
                if pd.notnull(row[f'{device_name} acq_timestamp'])
            ]

            # Recursively find orphaned files that include the device name.
            orphan_files = [f for f in source_dir.rglob("*") if f.is_file() and device_name in f.name]

            for file in orphan_files:
                file_ts = extract_timestamp_from_file(file, device_type)
                logging.info(f"Found orphan file {file} with timestamp {file_ts}")
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
                        random_part=random_part
                    )
                    logging.info(f"Enqueuing orphan task for {file} with shot number {matched_shot}")
                    # Process the task using your FileMover's method.

                    self.move_files_by_timestamp(task)
                else:
                    logging.warning(f"No matching shot number found for orphan file {file} (timestamp {file_ts})")

    def post_process_orphan_taks(self):
        for task in self.orphan_tasks:
            self.move_files_by_timestamp(task)

    def shutdown(self, wait: bool = True) -> None:
        """
        Signal that no new tasks will be added, wait for current tasks to finish,
        and then shut down all worker threads.
        """
        self.stop_event.set()
        if wait:
            self.task_queue.join()
        for _ in self.workers:
            self.task_queue.put(None)
        for worker in self.workers:
            worker.join()
        logging.info("FileMover has been shut down gracefully.")

class DataLogger:
    """
    Handles the logging of data from devices during a scan, supporting both event-driven
    and asynchronous data acquisition. This class manages polling, logging, and state management
    for various devices in the experimental setup.
    """

    def __init__(self, experiment_dir:str, device_manager: DeviceManager = None):

        """
        Initialize the DataLogger with the experiment directory and a device manager.

        Args:
            experiment_dir (str): Directory where the experiment's data is stored.
            device_manager (DeviceManager, optional): The manager responsible for handling devices.
                                                      If not provided, a new one is initialized.
        """

        self.device_manager = device_manager or DeviceManager(experiment_dir)

        self.stop_event = threading.Event()  # Event to control polling thread
        self.warning_timeout_sync = 2  # Timeout for synchronous devices (seconds)
        self.warning_timeout_async_factor = 1  # Factor of polling interval for async timeout
        self.last_log_time_sync = {}  # Dictionary to track last log time for synchronous devices
        self.last_log_time_async = {}  # Dictionary to track last log time for async devices
        self.polling_interval = .5
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
        self.initial_timestamps: Dict[str, float] = {}
        self.synced_timestamps: Dict[str, float] = {}
        self.standby_mode_device_status: Dict[str, Optional[bool]] = {}
        self.device_save_paths_mapping: DeviceSavePaths = {}
        # Uses Optional[bool] because status can be True, False, or None

        # File management and observables
        self.file_mover: Optional[FileMover] = None  # Will be assigned a FileMover instance later
        self.synchronous_device_names: List[str] = []
        self.event_driven_observables: List[str] = []

        # Boolean flags
        self.all_devices_in_standby: bool = False
        self.devices_synchronized: bool = False
        self.save_local = True

    def set_device_save_paths_mapping(self, mapping: DeviceSavePaths) -> None:
        """Set the device_save_paths_mapping externally."""
        self.device_save_paths_mapping = mapping

    def start_logging(self) -> Dict[str, Any]:
        """
        Start logging data for all devices. Event-driven observables trigger logs, and asynchronous
        observables are polled at regular intervals.

        Returns:
            dict: A dictionary storing the logged entries.
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

        #scan number in datalogger updates in scan_manager
        self.file_mover.scan_number = self.scan_number
        self.file_mover.save_local = self.save_local

        # Access event-driven and async observables from DeviceManager
        self.event_driven_observables = self.device_manager.event_driven_observables

        for observable in self.event_driven_observables:
            device_name = observable.split(':')[0]
            if device_name not in self.synchronous_device_names:
                self.synchronous_device_names.append(device_name)

        # Register the logging function for event-driven observables
        self._register_event_logging(self._handle_TCP_message_from_device)

        logging.info(
            'waiting for all devices to go to standby mode. Note, device standby status not checked, just waiting 4 seconds for all devices to timeout')
        time.sleep(1)

        logging.info("Logging has started for all event-driven devices.")

        return self.log_entries

    def _handle_TCP_message_from_device(self, message: str, device: GeecsDevice) -> None:
        """
        Handle updates from TCP subscribers and log them when a new timestamp is detected.

        args:
            message (str): Message from the device containing data to be logged.
            device (GeecsDevice): The device object from which the message originated.
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

        timestamp_from_device = self._extract_timestamp_from_tcp_message(message, device)

        #TODO: I think this needs to be changed to raise some kind of error, as not getting a timestamp
        # is a pretty fatal error
        if timestamp_from_device is None:
            return

        if not self.all_devices_in_standby:
            self.check_device_standby_mode_status(device, timestamp_from_device)
            self.all_devices_in_standby = self.check_all_standby_status()
            return

        # This part is a little messy as all the logic and execution that will synchronize devices
        # occurs in ScanManager
        if not self.devices_synchronized:
            self.check_device_standby_mode_status(device, timestamp_from_device)
            self.devices_synchronized = self.check_all_exited_standby_status()
            if not self.devices_synchronized:
                return
            if self.devices_synchronized:
                self.initial_timestamps = self.synced_timestamps

        elapsed_time = self._calculate_elapsed_time(device, timestamp_from_device)

        #ensure that the synchronization shot is not included in the logging
        if elapsed_time>0:
            if self._check_duplicate_timestamp(device, timestamp_from_device):
                return

            self._log_device_data(device, elapsed_time)

    def check_all_standby_status(self) -> bool:
        device_names = set(self.synchronous_device_names)
        standby_keys = self.standby_mode_device_status.keys()

        all_in_dict = device_names.issubset(standby_keys)
        all_on = all(self.standby_mode_device_status.get(device,False) for device in self.synchronous_device_names)

        if all_in_dict and all_on:
            logging.info("All device names are present in standby_mode_device_status dict and all have True status.")
            return True
        else:
            logging.info(f'Not all devices are in standby: {self.standby_mode_device_status}')

            return False

    def check_all_exited_standby_status(self) -> bool:
        device_names = set(self.synchronous_device_names)
        standby_keys = self.standby_mode_device_status.keys()

        all_in_dict = device_names.issubset(standby_keys)
        all_off = all(not self.standby_mode_device_status.get(device,True) for device in self.synchronous_device_names)

        if all_in_dict and all_off:
            logging.info("All device names are present in standby_mode_device_status dict and all "
                         "have False status meaning they have exited standby mode.")
            return True
        else:
            return False

    def check_device_standby_mode_status(self, device: GeecsDevice, timestamp: float) -> None:
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

        if self.standby_mode_device_status[device.get_name()] == False:
            return

        # check if there has been a timestamp added to the dict for a given device
        t0 = self.initial_timestamps.get(device.get_name(),None)

        # if this is the first logged timestamp, return None because we can't say for
        # certain if the device is in standby mode
        if t0 is None:
            self.initial_timestamps[device.get_name()] = timestamp
            logging.info(
                f"First TCP event received from {device.get_name()}. Initial dummy timestamp set to {timestamp}.")
            return

        logging.info(f'checking standby status of {device.get_name()}')

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
            logging.info(f'{device.get_name()} is in standby')
            return
        else:
            self.standby_mode_device_status[device.get_name()] = False
            logging.info(f'{device.get_name()} has exited in standby')
            return

    def update_repetition_rate(self, new_repetition_rate) -> None:
        self.repetition_rate = new_repetition_rate

    @staticmethod
    def _extract_timestamp_from_tcp_message(message: str, device: GeecsDevice) -> float:

        """
        Extract the timestamp from a device message.

        Args:
            message (str): The message containing device data.
            device (GeecsDevice): The device sending the message.

        Returns:
            float: The extracted timestamp, or the current system time if no timestamp is found.
        """

        stamp = datetime.now().__str__()
        err = ErrorAPI()
        net_msg = mh.NetworkMessage(tag=device.get_name(), stamp=stamp, msg=message, err=err)
        parsed_data = device.handle_subscription(net_msg)
        current_timestamp = parsed_data[2].get('acq_timestamp')  # *NOTE* `timestamp` for synchronizing
        if current_timestamp is None:
            logging.warning(f"No timestamp found for {device.get_name()}. Using system time instead.")
            current_timestamp = float(stamp)
        return float(current_timestamp)

    def _register_event_logging(self, log_update: Callable[[str, GeecsDevice], None]) -> None:
        """
        Register event-driven observables for logging.

        Args:
            log_update (Callable[[str, GeecsDevice], None]):
                Function to call when an event update occurs.
                Expects a `str` message and a `GeecsDevice` object.
                return type for log_update is None
        """

        for device_name, device in self.device_manager.devices.items():
            for observable in self.event_driven_observables:
                if observable.startswith(device_name):
                    logging.info(f"Registering logging for event-driven observable: {observable}")
                    device.event_handler.register('update', 'logger', lambda msg, dev=device: log_update(msg, dev))

    def _calculate_elapsed_time(self, device: GeecsDevice, current_timestamp: float) -> float:
        """
        Calculate the elapsed time for a device based on its initial timestamp.

        Args:
            device (GeecsDevice): The device being monitored.
            current_timestamp (float): The current timestamp from the device.

        Returns:
            int: The elapsed time (rounded) since the device's initial timestamp.
        """

        t0 = self.initial_timestamps[device.get_name()]
        elapsed_time = current_timestamp - t0
        return round(elapsed_time * self.repetition_rate)/self.repetition_rate

    def _check_duplicate_timestamp(self, device: GeecsDevice, current_timestamp: float) -> bool:
        """
        Check if the current timestamp for a device is a duplicate.

        Args:
            device (GeecsDevice): The device being monitored.
            current_timestamp (float): The current timestamp from the device.

        Returns:
            bool: True if the timestamp is a duplicate, False otherwise.
        """

        if device.get_name() in self.last_timestamps and self.last_timestamps[device.get_name()] == current_timestamp:
            logging.info(f"Timestamp hasn't changed for {device.get_name()}. Skipping log.")
            return True
        self.last_timestamps[device.get_name()] = current_timestamp
        return False

    def update_async_observables(self, async_observables: list, elapsed_time: float)-> None:
        """
        Update log entries with the latest values for asynchronous observables.

        Args:
            async_observables (list): List of asynchronous observables to update.
            elapsed_time (int): The time elapsed since the logging started.
        """
        for observable in async_observables:
            # Check if the observable has a variable specified (e.g., 'Dev1:var1')
            if ':' in observable:
                device_name, var_name = observable.split(':')
            else:
                device_name = observable
                var_name = None

            device = self.device_manager.devices.get(device_name)

            if device:
                if device.is_composite:
                    # Handle composite devices
                    composite_value = device.state.get("composite_var", "N/A")
                    self.log_entries[elapsed_time][f"{device_name}:composite_var"] = composite_value
                    logging.info(
                        f"Updated composite var {device_name}:composite_var to {composite_value} for elapsed time {elapsed_time}.")

                    # Log sub-component states
                    for comp in device.components:
                        sub_device_name = comp['device']
                        sub_var_name = comp['variable']
                        sub_device = device.sub_devices[sub_device_name]['instance']

                        if sub_device:
                            sub_value = sub_device.state.get(sub_var_name, "N/A")
                            self.log_entries[elapsed_time][f"{sub_device_name}:{sub_var_name}"] = sub_value
                            logging.info(
                                f"Updated sub-component {sub_device_name}:{sub_var_name} to {sub_value} for elapsed time {elapsed_time}.")
                        else:
                            logging.warning(f"Sub-device {sub_device_name} not found for {device_name}.")
                else:
                    # Handle regular devices
                    if var_name is None:
                        logging.warning(f"No variable specified for device {device_name}. Skipping.")
                        continue

                    value = device.state.get(var_name, "N/A")
                    self.log_entries[elapsed_time][f"{device_name}:{var_name}"] = value
                    logging.info(
                        f"Updated async var {device_name}:{var_name} to {value} for elapsed time {elapsed_time}.")
            else:
                logging.warning(f"Device {device_name} not found in DeviceManager. Skipping {observable}.")

    def _log_device_data(self, device:GeecsDevice, elapsed_time:float) -> None:

        """
        Log the data for a device during an event-driven observation.

        Args:
            device (GeecsDevice): The device being logged.
            elapsed_time (int): The time elapsed since the logging started.

        Logs:
            - Device variable data and its elapsed time.
            - Plays a beep sound on a new log entry.
        """

        with self.lock:
            observables_data = {
                observable.split(':')[1]: device.state.get(observable.split(':')[1], '')
                for observable in self.event_driven_observables if observable.startswith(device.get_name())
            }
            if elapsed_time not in self.log_entries:
                logging.info(f'elapsed time in sync devices {elapsed_time}. reported by {device.get_name()}')
                self.log_entries[elapsed_time] = {'Elapsed Time': elapsed_time}
                # Log configuration variables (such as 'bin') only when a new entry is created
                # bin number is updated in scan_manager
                self.log_entries[elapsed_time]['Bin #'] = self.bin_num
                self.log_entries[elapsed_time]['scan'] = self.scan_number

                if self.virtual_variable_name is not None:
                    self.log_entries[elapsed_time][self.virtual_variable_name] = self.virtual_variable_value

                # Update with async observable values
                self.update_async_observables(self.device_manager.async_observables, elapsed_time)

                # TODO move the on-shot tdms writer functionality from scan manager to here

                # Set a flag to tell scan manager that a shot occurred
                self.shot_save_event.set()

                # Trigger the beep in the background
                self.sound_player.play_beep()  # Play the beep sound
                self.shot_index += 1

            self.log_entries[elapsed_time].update({
                f"{device.get_name()}:{key}": value for key, value in observables_data.items()
            })

            if device.get_name() in self.device_save_paths_mapping:
                device_name = device.get_name()
                cfg = self.device_save_paths_mapping[device_name]
                task = FileMoveTask(
                    source_dir=cfg['source_dir'],
                    target_dir=cfg['target_dir'],
                    device_name=device_name,
                    device_type=cfg['device_type'],
                    expected_timestamp=observables_data['acq_timestamp'],  # *NOTE* `SysTimestamp` for data logging
                    shot_index=self.shot_index
                )
                self.file_mover.move_files_by_timestamp(task)

    def stop_logging(self) -> None:
        """
        Stop both event-driven and asynchronous logging, unregister all event handlers, and reset states.
        """
        # Unregister all event-driven logging
        for device_name, device in self.device_manager.devices.items():
            device.event_handler.unregister('update', 'logger')

        self.sound_player.play_toot()

        # Signal to stop the polling thread
        self.stop_event.set()

        self.sound_player.stop()
        # TODO check if this needs to be moved.  It might be cleared before the stop is registered
        # Reset the stop_event for future logging sessions

        # # Shut down the file mover gracefully.
        # self.file_mover.shutdown(wait=True)

        self.stop_event.clear()

    def reinitialize_sound_player(self, options: Optional[dict] = None) -> None:
        """
        Reinitialize the sound player, stopping the current one and creating a new instance.
        """

        self.sound_player.stop()
        self.sound_player = SoundPlayer(options=options)
        self.shot_index = 0

    def get_current_shot(self) -> int:
        """
        Get the current shot index. used for progress bar tracking

        Returns:
            float: The current shot index.
        """
        return self.shot_index
