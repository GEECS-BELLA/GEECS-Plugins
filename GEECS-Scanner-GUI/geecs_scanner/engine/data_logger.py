"""Per-shot data logging and device synchronization for ScanManager."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from time import time as _now
from typing import Any, Callable, Dict, List, Optional, Union

from . import DeviceManager
from geecs_python_api.controls.devices.geecs_device import GeecsDevice
import geecs_python_api.controls.interface.message_handling as mh
from geecs_scanner.logging_setup import update_context
from geecs_scanner.utils import SoundPlayer
from geecs_scanner.engine.file_mover import FileMover, FileMoveTask
from geecs_scanner.engine.models.scan_options import ScanOptions

DeviceSavePaths = Dict[str, Dict[str, Union[Path, str]]]

logger = logging.getLogger(__name__)


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

    def start_logging(self, file_mover: FileMover) -> Dict[str, Any]:
        """Reset state, wire the injected FileMover, and register TCP callbacks.

        Parameters
        ----------
        file_mover : FileMover
            Pre-configured worker created and owned by ``ScanManager``.

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

        self.file_mover = file_mover

        self.all_devices_in_standby = False
        self.devices_synchronized = False
        self.synchronous_device_names = []

        self.sound_player.start_queue()

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
