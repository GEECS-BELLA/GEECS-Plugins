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


class DataLogger:
    """
    Handles the logging of data from devices during a scan, supporting both event-driven
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

        self.last_timestamps = {}
        self.initial_timestamps = {}
        self.synced_timestamps = {}
        self.standby_mode_device_status = {}
        self.log_entries = {}

        self.all_devices_in_standby = False
        self.devices_synchronized = False
        self.synchronous_device_names = []

        # Start the sound player
        self.sound_player.start_queue()

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

        # # Start a thread to monitor device warnings
        # self.warning_thread = threading.Thread(target=self._monitor_warnings, args=(self.event_driven_observables, async_observables))
        # self.warning_thread.start()

        return self.log_entries

    def _handle_TCP_message_from_device(self, message, device):
        """
        Handle updates from TCP subscribers and log them when a new timestamp is detected.

        Args:
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

        timestamp_from_device = self._extract_timestamp(message, device)

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

    def check_all_standby_status(self):
        device_names = set(self.synchronous_device_names)
        standby_keys = self.standby_mode_device_status.keys()

        all_in_dict = device_names.issubset(standby_keys)
        all_on = all(self.standby_mode_device_status.get(device,False) for device in self.synchronous_device_names)

        if all_in_dict and all_on:
            logging.info("All device names are present in standby_mode_device_status dict and all have True status.")
            return True
        else:
            logging.info(f'NOt all devices are in standby: {self.standby_mode_device_status}')

            return False

    def check_all_exited_standby_status(self):
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

    def check_device_standby_mode_status(self, device, timestamp: float):
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
            return None

        logging.info(f'checking standby status of {device.get_name()}')

        # update the timestamp in this dict each call. Once all devices have verifiably
        # entered standby and exited standby synchronously, we will overwrite the
        # initial_timestamps dict to reset t0 for each device
        self.synced_timestamps[device.get_name()] = timestamp

        # handle the case that this isn't the first call. If the passed timestamp
        # is equal to the timestamp in the dict, that means we've received two
        # TCP events from the device without the device timestamp updating, which
        # means the device has timed out and can be considered to be in standby mode
        if t0 == timestamp:
            self.standby_mode_device_status[device.get_name()] = True
            logging.info(f'{device.get_name()} is in standby')

            return True
        else:
            self.standby_mode_device_status[device.get_name()] = False
            logging.info(f'{device.get_name()} has exited in standby')
            return False

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

    def _register_event_logging(self, log_update):
        """
        Register event-driven observables for logging.

        Args:
            log_update (function): Function to call when an event update occurs.
        """

        for device_name, device in self.device_manager.devices.items():
            for observable in self.event_driven_observables:
                if observable.startswith(device_name):
                    logging.info(f"Registering logging for event-driven observable: {observable}")
                    device.event_handler.register('update', 'logger', lambda msg, dev=device: log_update(msg, dev))

    def _calculate_elapsed_time(self, device, current_timestamp):
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

    def _check_duplicate_timestamp(self, device, current_timestamp):
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

    def _log_device_data(self, device, elapsed_time):

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
                self.log_entries[elapsed_time]['Bin #'] = self.bin_num
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

    def _monitor_warnings(self, async_observables):
        """
        Monitor the last log time for each device and issue warnings if a device hasn't updated within the threshold.
        """
        while not self.stop_event.is_set():
            current_time = time.time()

            if self.data_recording:
                # Check synchronous devices (event-driven observables)
                for observable in self.event_driven_observables:
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
