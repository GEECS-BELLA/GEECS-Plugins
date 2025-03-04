from __future__ import annotations
from typing import Optional

import time
import threading
from datetime import datetime
import logging

from . import DeviceManager
from geecs_scanner.utils import SoundPlayer

from geecs_python_api.controls.devices.geecs_device import GeecsDevice
from geecs_python_api.controls.interface.geecs_errors import ErrorAPI
import geecs_python_api.controls.interface.message_handling as mh

import queue
import shutil
from pathlib import Path
import datetime


class FileMover:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_func, daemon=True)
        self.worker.start()
        logging.info("FileMover worker started.")

    def _extract_timestamp(self, file: Path) -> datetime.datetime:
        """Extract timestamp from file. Here we use the file's modification time."""
        return datetime.datetime.fromtimestamp(file.stat().st_mtime)

    def _worker_func(self):
        """
        Process file move tasks from the queue. When the stop_event is set and the queue is empty,
        the worker exits.
        """
        while True:
            try:
                # Wait for a task (timeout allows periodic check of stop_event)
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                # If no tasks are waiting and the stop_event is set, exit the loop.
                if self.stop_event.is_set() and self.task_queue.empty():
                    break
                continue

            # If a sentinel is encountered, skip further processing.
            if task is None:
                continue

            try:
                source_dir, target_dir, expected_timestamp, elapsed_time = task
                self._process_task(source_dir, target_dir, expected_timestamp, elapsed_time)
            except Exception as e:
                logging.error(f"Error processing task: {e}")
            finally:
                self.task_queue.task_done()
        logging.info("FileMover worker stopped.")

    def _process_task(self, source_dir: Path, target_dir: Path,
                      expected_timestamp: datetime.datetime, elapsed_time: float):
        """
        Search for files in source_dir with a timestamp close to expected_timestamp,
        rename them with a standard naming convention (including the timestamp and elapsed time),
        and move them to target_dir.
        """
        tolerance = datetime.timedelta(minutes=1)
        for file in source_dir.glob("*"):
            if file.is_file():
                file_ts = self._extract_timestamp(file)
                if abs(file_ts - expected_timestamp) <= tolerance:
                    # Create a new filename using the expected timestamp and elapsed time.
                    new_filename = f"data_{expected_timestamp.strftime('%Y%m%dT%H%M%S')}_{int(elapsed_time)}{file.suffix}"
                    dest_file = target_dir / new_filename
                    try:
                        shutil.move(str(file), str(dest_file))
                        logging.info(f"Moved {file} to {dest_file}")
                    except Exception as e:
                        logging.error(f"Error moving {file} to {dest_file}: {e}")

    def move_files_by_timestamp(self, source_dir: Path, target_dir: Path,
                                expected_timestamp: datetime.datetime, elapsed_time: float):
        """
        Enqueue a file move task.
        """
        self.task_queue.put((source_dir, target_dir, expected_timestamp, elapsed_time))

    def shutdown(self, wait: bool = True):
        """
        Signal that no new tasks will be added, wait for current tasks to finish,
        and then shut down the worker thread.
        """
        self.stop_event.set()
        if wait:
            # Wait for all tasks in the queue to be processed.
            self.task_queue.join()
        # Optionally put a sentinel to unblock the worker if needed.
        self.task_queue.put(None)
        self.worker.join()
        logging.info("FileMover has been shut down gracefully.")


class DataLogger:
    """
    Handles the logging of data from devices during an scan, supporting both event-driven
    and asynchronous data acquisition. This class manages polling, logging, and state management
    for various devices in the experimental setup.
    """

    def __init__(self, experiment_dir, device_manager=None):

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

        # Create a FileMover instance
        self.file_mover = FileMover()

        self.log_entries = {}


        # Initialize the sound player
        self.sound_player = SoundPlayer()
        self.shot_index = 0

        self.bin_num = 0  # Initialize bin as 0

        self.virtual_variable_name = None
        self.virtual_variable_value = 0

        self.data_recording = False
        self.idle_time = 0

        self.lock = threading.Lock()

        self.repetition_rate = 1.0  # Gets updated upon scan manager's reinitialization
        
        self.shot_save_event = threading.Event()

    def start_logging(self):
        """
        Start logging data for all devices. Event-driven observables trigger logs, and asynchronous
        observables are polled at regular intervals.

        Returns:
            dict: A dictionary storing the logged entries.
        """

        last_timestamps = {}
        initial_timestamps = {}
        standby_mode = {}
        self.log_entries = {}

        # Start the sound player
        self.sound_player.start_queue()

        # Access event-driven and async observables from DeviceManager
        event_driven_observables = self.device_manager.event_driven_observables
        async_observables = self.device_manager.async_observables

        def log_update(message, device):
            """
            Handle updates from TCP subscribers and log them when a new timestamp is detected.

            Args:
                message (str): Message from the device containing data to be logged.
                device (GeecsDevice): The device object from which the message originated.
            """

            current_timestamp = self._extract_timestamp(message, device)

            if current_timestamp is None:
                return

            if self._initialize_standby_mode(device, standby_mode, initial_timestamps, current_timestamp):
                return

            elapsed_time = self._calculate_elapsed_time(device, initial_timestamps, current_timestamp)

            if self._check_duplicate_timestamp(device, last_timestamps, current_timestamp):
                return

            self._log_device_data(device, event_driven_observables, elapsed_time)

        # Register the logging function for event-driven observables
        self._register_event_logging(event_driven_observables, log_update)

        logging.info(
            'waiting for all devices to go to standby mode. Note, device standby status not checked, just waiting 4 seconds for all devices to timeout')
        time.sleep(4)

        logging.info("Logging has started for all event-driven devices.")

        # # Start a thread to monitor device warnings
        # self.warning_thread = threading.Thread(target=self._monitor_warnings, args=(event_driven_observables, async_observables))
        # self.warning_thread.start()

        return self.log_entries

    def update_repetition_rate(self, new_repetition_rate):
        self.repetition_rate = new_repetition_rate

    def _extract_timestamp(self, message, device):

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
        current_timestamp = parsed_data[2].get('timestamp')
        if current_timestamp is None:
            logging.warning(f"No timestamp found for {device.get_name()}. Using system time instead.")
            current_timestamp = float(stamp)
        return float(current_timestamp)

    def _register_event_logging(self, event_driven_observables, log_update):
        """
        Register event-driven observables for logging.

        Args:
            event_driven_observables (list): A list of event-driven observables to monitor.
            log_update (function): Function to call when an event update occurs.
        """

        for device_name, device in self.device_manager.devices.items():
            for observable in event_driven_observables:
                if observable.startswith(device_name):
                    logging.info(f"Registering logging for event-driven observable: {observable}")
                    device.event_handler.register('update', 'logger', lambda msg, dev=device: log_update(msg, dev))

    def _initialize_standby_mode(self, device, standby_mode, initial_timestamps, current_timestamp):
        """
        Initialize and manage the standby mode for a device based on its timestamp. This is an
        essential component for getting synchronization correct

        Args:
            device (GeecsDevice): The device being monitored.
            standby_mode (dict): A dictionary tracking which devices are in standby mode.
            initial_timestamps (dict): A dictionary storing initial timestamps for devices.
            current_timestamp (float): The current timestamp from the device.

        Returns:
            bool: True if the device is still in standby mode, False otherwise.
        """
        if device.get_name() not in standby_mode:
            standby_mode[device.get_name()] = True
        if device.get_name() not in initial_timestamps:
            initial_timestamps[device.get_name()] = current_timestamp
            logging.info(
                f"Initial dummy timestamp for {device.get_name()} set to {current_timestamp}. Standby mode enabled.")
        t0 = initial_timestamps[device.get_name()]
        if standby_mode[device.get_name()] and current_timestamp != t0:
            standby_mode[device.get_name()] = False
            logging.info(f"Device {device.get_name()} exiting standby mode. First real timestamp: {current_timestamp}")
            initial_timestamps[device.get_name()] = current_timestamp
        elif standby_mode[device.get_name()]:
            logging.info(f"Device {device.get_name()} is still in standby mode.")
            return True
        return False

    def _calculate_elapsed_time(self, device, initial_timestamps, current_timestamp):
        """
        Calculate the elapsed time for a device based on its initial timestamp.

        Args:
            device (GeecsDevice): The device being monitored.
            initial_timestamps (dict): A dictionary storing initial timestamps for devices.
            current_timestamp (float): The current timestamp from the device.

        Returns:
            int: The elapsed time (rounded) since the device's initial timestamp.
        """

        t0 = initial_timestamps[device.get_name()]
        elapsed_time = current_timestamp - t0
        return round(elapsed_time * self.repetition_rate)/self.repetition_rate

    def _check_duplicate_timestamp(self, device, last_timestamps, current_timestamp):
        """
        Check if the current timestamp for a device is a duplicate.

        Args:
            device (GeecsDevice): The device being monitored.
            last_timestamps (dict): A dictionary storing the last timestamps for devices.
            current_timestamp (float): The current timestamp from the device.

        Returns:
            bool: True if the timestamp is a duplicate, False otherwise.
        """

        if device.get_name() in last_timestamps and last_timestamps[device.get_name()] == current_timestamp:
            logging.info(f"Timestamp hasn't changed for {device.get_name()}. Skipping log.")
            return True
        last_timestamps[device.get_name()] = current_timestamp
        return False

    def update_async_observables(self, async_observables, elapsed_time):
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

    def _log_device_data(self, device, event_driven_observables, elapsed_time):

        """
        Log the data for a device during an event-driven observation.

        Args:
            device (GeecsDevice): The device being logged.
            event_driven_observables (list): A list of event-driven observables to monitor.
            elapsed_time (int): The time elapsed since the logging started.

        Logs:
            - Device variable data and its elapsed time.
            - Plays a beep sound on a new log entry.
        """

        with self.lock:
            observables_data = {
                observable.split(':')[1]: device.state.get(observable.split(':')[1], '')
                for observable in event_driven_observables if observable.startswith(device.get_name())
            }
            if elapsed_time not in self.log_entries:
                logging.info(f'elapsed time in sync devices {elapsed_time}')
                self.log_entries[elapsed_time] = {'Elapsed Time': elapsed_time}
                # Log configuration variables (such as 'bin') only when a new entry is created
                self.log_entries[elapsed_time]['Bin #'] = self.bin_num
                if self.virtual_variable_name is not None:
                    self.log_entries[elapsed_time][self.virtual_variable_name] = self.virtual_variable_value

                # Update with async observable values
                self.update_async_observables(self.device_manager.async_observables, elapsed_time)

                # TODO move the on-shot tdms writer functionality from scan manager to here

                # Set a flag to tell scan manager that a shot occured
                self.shot_save_event.set()

                # Trigger the beep in the background
                self.sound_player.play_beep()  # Play the beep sound
                self.shot_index += 1

            self.log_entries[elapsed_time].update({
                f"{device.get_name()}:{key}": value for key, value in observables_data.items()
            })

            # Enqueue a file-moving task.
            # For demonstration, use fixed source and target directories.
            # Note, below is not fucntional yet. The source dir will need to be determined
            # using the device name and the IP for the host computer of that device. This
            # information needs to be parsed from the database. Makes sense that this information
            # gets parsed once upon intialization of the start_logging method.
            # The target directory would have to be passed from the ScanManager to account for the
            # Scan number. Again, this probably should be passed once at the top of the scanning.
            # Scan number will also be need in renaming the moved files.
            # Note, the implementation of using timestamps is wrong.
            source_dir = Path(r"C:\LocalData")
            target_dir = Path(r"\\CentralServer\SharedData")
            # Use current time as expected timestamp.
            expected_timestamp = datetime.datetime.now()
            self.file_mover.move_files_by_timestamp(source_dir, target_dir, expected_timestamp, elapsed_time)

    def _monitor_warnings(self, event_driven_observables, async_observables):
        """
        Monitor the last log time for each device and issue warnings if a device hasn't updated within the threshold.
        """
        while not self.stop_event.is_set():
            current_time = time.time()

            if self.data_recording:
                # Check synchronous devices (event-driven observables)
                for observable in event_driven_observables:
                    device_name = observable.split(':')[0]
                    last_log_time = self.last_log_time_sync.get(device_name, None)

                    if last_log_time and (current_time - (last_log_time + self.idle_time)) > self.warning_timeout_sync:
                        logging.warning(
                            f"Synchronous device {device_name} hasn't updated in over {self.warning_timeout_sync} seconds.")

                # Check asynchronous devices
                async_timeout = self.polling_interval * self.warning_timeout_async_factor
                for observable in async_observables:
                    device_name = observable.split(':')[0]
                    last_log_time = self.last_log_time_async.get(device_name, None)

                    if last_log_time and (current_time - (last_log_time + self.idle_time)) > async_timeout:
                        logging.warning(
                            f"Asynchronous device {device_name} hasn't updated in over {async_timeout} seconds.")

                time.sleep(1)  # Monitor the warnings every second

    def stop_logging(self):
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

        # Shut down the file mover gracefully.
        self.file_mover.shutdown(wait=True)

        self.stop_event.clear()

    def reinitialize_sound_player(self, options: Optional[dict] = None):
        """
        Reinitialize the sound player, stopping the current one and creating a new instance.
        """

        self.sound_player.stop()
        self.sound_player = SoundPlayer(options=options)
        self.shot_index = 0

    def get_current_shot(self):
        """
        Get the current shot index. used for progress bar tracking

        Returns:
            float: The current shot index.
        """
        return float(self.shot_index)
