"""Central scan orchestrator: device setup, trigger control, data acquisition, and cleanup."""

from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
import warnings
from dataclasses import fields
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from . import (
    ActionManager,
    DataLogger,
    DatabaseDictLookup,
    DeviceManager,
    ScanDataManager,
    ScanStepExecutor,
)
from geecs_data_utils import ScanConfig, ScanMode
from geecs_python_api.controls.devices.scan_device import ScanDevice
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceInstantiationError,
)
from geecs_scanner.data_acquisition.dialog_request import (
    DEVICE_COMMAND_ERRORS,
    DialogRequest,
)
from geecs_scanner.logging_setup import scan_log
from geecs_scanner.optimization.base_optimizer import BaseOptimizer
from geecs_scanner.utils.exceptions import (
    DeviceSynchronizationError,
    DeviceSynchronizationTimeout,
    OrphanProcessingTimeout,
    ScanAbortedError,
    TriggerError,
)
from geecs_scanner.utils.retry import retry

logger = logging.getLogger(__name__)

database_dict = DatabaseDictLookup()


def get_database_dict():
    """Return the current database dictionary."""
    return database_dict.get_database()


class ScanManager:
    """Coordinate all aspects of a scan: devices, trigger, logging, and cleanup.

    Attributes
    ----------
    device_manager : DeviceManager
    action_manager : ActionManager
    optimizer : BaseOptimizer or None
    initialization_success : bool
    scan_config : ScanConfig
    dialog_queue : queue.Queue[DialogRequest]
        Worker threads post dialog requests here; the main-thread 200 ms timer drains it.
    restore_failures : list[str]
        Collected by restore_initial_state(); shown once by the main thread after the
        scan thread exits (blocking in the scan thread would deadlock with join()).
    last_reinit_error : str or None
        Device name from the most recent GeecsDeviceInstantiationError, for the GUI dialog.
    """

    def __init__(
        self,
        experiment_dir: str,
        shot_control_information: dict,
        options_dict: Optional[dict] = None,
        device_manager=None,
        scan_data=None,
    ):
        database_dict.reload(experiment_name=experiment_dir)
        self.device_manager = device_manager or DeviceManager(
            experiment_dir=experiment_dir
        )
        self.action_manager = ActionManager(experiment_dir=experiment_dir)
        self.optimizer: Optional[BaseOptimizer] = None
        self.initialization_success = False

        self.MC_ip = ""

        self.scan_data_manager = ScanDataManager(
            self.device_manager, scan_data, database_dict
        )

        self.data_logger = DataLogger(experiment_dir, self.device_manager)

        self.shot_control: Optional[ScanDevice] = None
        self.shot_control_variables = None
        shot_control_device = shot_control_information.get("device", None)
        if shot_control_device:
            self.shot_control = ScanDevice(shot_control_information["device"])
            self.shot_control_variables = shot_control_information["variables"]

        self.results = {}

        self.stop_scanning_thread_event = threading.Event()

        # Queue for worker threads to request GUI dialogs on the main thread.
        # See gui_dialogs.py and GEECSScannerWindow.update_gui_status().
        self.dialog_queue: queue.Queue[DialogRequest] = queue.Queue()

        # Failures collected by restore_initial_state() after the scan ends.
        # Cannot show a blocking dialog there (scan thread would deadlock with
        # stop_scanning_thread().join()).  The main thread reads this list once
        # is_active() transitions to False and shows a one-shot warning.
        self.restore_failures: list[str] = []

        # Set by reinitialize() when a GeecsDeviceInstantiationError occurs so
        # the GUI can include the device name in the error dialog.
        self.last_reinit_error: Optional[str] = None

        self.virtual_variable_list = []
        self.virtual_variable_name = None

        self.acquisition_time = 0

        self.scanning_thread = None

        self.scan_step_start_time = 0
        self.scan_step_end_time = 0

        self.initial_state = None
        self.scan_steps = []

        self.pause_scan_event = threading.Event()
        self.pause_scan_event.set()  # Set to 'running' by default
        self.pause_time = 0

        self.options_dict: dict = {} if options_dict is None else options_dict
        self.save_local = True

        self.scan_config: ScanConfig

        self.executor = ScanStepExecutor(
            device_manager=self.device_manager,
            data_logger=self.data_logger,
            scan_data_manager=self.scan_data_manager,
            optimizer=self.optimizer,
            shot_control=self.shot_control,
            options_dict=self.options_dict,
            stop_scanning_thread_event=self.stop_scanning_thread_event,
            pause_scan_event=self.pause_scan_event,
        )
        self.executor.trigger_on_fn = self.trigger_on
        self.executor.trigger_off_fn = self.trigger_off
        self.executor.on_device_error = self.request_user_dialog
        self.action_manager.on_user_prompt = self.request_user_dialog
        self.scan_data_manager.on_device_error = self.request_user_dialog

    # ------------------------------------------------------------------
    # Dialog bridge (worker → main thread)
    # ------------------------------------------------------------------

    def request_user_dialog(
        self, exc: Exception, context: Optional[str] = None
    ) -> bool:
        """Submit a device-error dialog request and block until the user responds.

        Called from worker threads.  Puts a :class:`DialogRequest` on
        ``self.dialog_queue`` and blocks until the main-thread GUI timer
        drains it and the user clicks Continue or Abort.

        Parameters
        ----------
        exc : Exception
            The device exception that triggered the dialog.
        context : str, optional
            Extra information shown in the dialog body.

        Returns
        -------
        bool
            ``True`` if the user chose Abort; ``False`` to continue.
        """
        request = DialogRequest(exc=exc, context=context)
        self.dialog_queue.put(request)
        request.response_event.wait()
        return request.abort[0]

    def pause_scan(self):
        """Pause the scan by clearing the pause event."""
        if self.pause_scan_event.is_set():
            self.pause_scan_event.clear()
            logger.info("Scanning paused.")

    def resume_scan(self):
        """Resume the scan by setting the pause event."""
        if not self.pause_scan_event.is_set():
            self.pause_scan_event.set()
            logger.info("Scanning resumed.")

    def reinitialize(
        self, config_path=None, config_dictionary=None, scan_data=None
    ) -> bool:
        """Reset and reload device configuration.

        Parameters
        ----------
        config_path : str, optional
        config_dictionary : dict, optional
        scan_data : ScanData, optional
            If given, scan_data_manager will use an alternative scan folder.

        Returns
        -------
        bool
            True if successful and all devices connected.

        Raises
        ------
        GeecsDeviceInstantiationError
            If device reinitialization fails.
        """
        self.initial_state = None
        self.initialization_success = False
        self.optimizer: Optional[BaseOptimizer] = None
        self.executor.optimizer: Optional[BaseOptimizer] = None

        try:
            self.device_manager.reinitialize(
                config_path=config_path, config_dictionary=config_dictionary
            )
        except GeecsDeviceInstantiationError as e:
            self.last_reinit_error = str(e)
            logger.exception(
                "Device reinitialization failed during initialization of device manager. check "
                "that all devices are on and available"
            )
            return False

        self.scan_data_manager = ScanDataManager(
            self.device_manager, scan_data, database_dict
        )

        logger.debug("config dictionary in reinitialize: %s", config_dictionary)

        if config_dictionary is not None and "options" in config_dictionary:
            self.options_dict = config_dictionary["options"]
            self.save_local = not self.options_dict.get("Save Direct on Network", False)

        new_mc_ip = self.options_dict.get("master_control_ip", "")
        if self.shot_control and new_mc_ip and self.MC_ip != new_mc_ip:
            self.MC_ip = new_mc_ip
            self.enable_live_ECS_dump(client_ip=self.MC_ip)

        self.data_logger.reinitialize_sound_player(options=self.options_dict)
        self.data_logger.last_log_time_sync = {}
        self.data_logger.update_repetition_rate(self.options_dict.get("rep_rate_hz", 1))
        self.data_logger.global_sync_tol_ms = self.options_dict.get(
            "global_time_tolerance_ms", 0
        )

        self.initialization_success = True
        return self.initialization_success

    def _set_trigger(self, state: str) -> list:
        """Set the trigger device to *state* for all shot-control variables.

        Parameters
        ----------
        state : str
            One of ``'OFF'``, ``'SCAN'``, ``'STANDBY'``, ``'SINGLESHOT'``.

        Returns
        -------
        list
            Return values from each ``shot_control.set()`` call.

        Raises
        ------
        TriggerError
            If any trigger variable fails after retries.  Trigger failures are
            always scan-fatal — do not catch and continue.
        """
        if self.shot_control is None or self.shot_control_variables is None:
            logger.debug("No shot control device, skipping 'set state %s'", state)
            return []

        valid_states = ["OFF", "SCAN", "STANDBY", "SINGLESHOT"]
        if state not in valid_states:
            logger.error("Invalid trigger state: %s", state)
            return []

        device_name = self.shot_control.get_name()
        results = []
        for variable, variable_settings in self.shot_control_variables.items():
            set_value = variable_settings.get(state, "")
            if not set_value:
                continue
            try:
                result = retry(
                    lambda v=variable, sv=set_value: self.shot_control.set(
                        v, sv, exec_timeout=0.5
                    ),
                    attempts=2,
                    delay=0.25,
                    catch=DEVICE_COMMAND_ERRORS,
                    on_retry=lambda exc, n, var=variable: logger.debug(
                        "Trigger variable '%s' retry %d: %s", var, n, exc
                    ),
                )
                results.append(result)
                logger.debug("Trigger: set %s → %s", variable, set_value)
            except DEVICE_COMMAND_ERRORS as exc:
                raise TriggerError(
                    device_name,
                    f"set trigger state '{state}' on variable '{variable}'",
                    variable=variable,
                ) from exc

        logger.info("Trigger state → %s", state)
        return results

    def trigger_off(self):
        """Set trigger state to OFF."""
        self._set_trigger("OFF")

    def trigger_on(self):
        """Set trigger state to SCAN."""
        self._set_trigger("SCAN")

    def is_scanning_active(self):
        """Return True if the scan thread is currently alive."""
        return bool(self.scanning_thread and self.scanning_thread.is_alive())

    def start_scan_thread(self, scan_config: Union[ScanConfig, dict] = None) -> None:
        """Start the scan in a background thread.

        Parameters
        ----------
        scan_config : ScanConfig or dict
            Passing a dict is deprecated — use :class:`~geecs_data_utils.ScanConfig` directly.
        """
        if not self.initialization_success:
            logger.error("Initialization unsuccessful, cannot start a new scan session")
            return

        if self.is_scanning_active():
            logger.warning(
                "Scanning is already active, cannot start a new scan session."
            )
            return

        # Backward compatibility: allow dict input, with warning
        if isinstance(scan_config, dict):
            valid_keys = {f.name for f in fields(ScanConfig)}
            unknown_keys = set(scan_config) - valid_keys
            if unknown_keys:
                logger.warning(
                    "Unexpected keys in scan_config dict: %s — they will be ignored.",
                    unknown_keys,
                )

            warnings.warn(
                "Passing scan_config as a dict is deprecated. Please migrate to using ScanConfig dataclass.",
                DeprecationWarning,
            )
            scan_config = ScanConfig(
                **{k: v for k, v in scan_config.items() if k in valid_keys}
            )

        self.scan_config = scan_config

        self.stop_scanning_thread_event.clear()

        logger.debug("Scan config: %s", self.scan_config)
        self.scanning_thread = threading.Thread(target=self._start_scan)
        self.scanning_thread.start()
        logger.info("Scan thread started.")

    def stop_scanning_thread(self):
        """Signal the scan thread to stop; returns immediately without blocking.

        Safe to call when no scan is active.  Use ``is_scanning_active()`` to
        poll for completion.  The scan thread clears ``self.scanning_thread`` itself.
        """
        if not self.is_scanning_active():
            logger.warning("No active scanning thread to stop.")
            return

        logger.info("Stopping the scanning thread...")
        self.stop_scanning_thread_event.set()

    def _start_scan(self) -> pd.DataFrame:
        """Run the full scan lifecycle in the scan thread."""
        if not self.initialization_success:
            logger.error("initialization unsuccessful, cannot start a new scan session")
            return pd.DataFrame()

        log_df = pd.DataFrame()

        # ------------------------------------------------------------------ #
        # Phase 1: before scan paths exist                                    #
        # Disable the trigger and create the scan directory.  Any failure     #
        # here means nothing has been committed — abort cleanly.              #
        # ------------------------------------------------------------------ #
        logger.debug(
            "scan config getting sent to pre logging is this: %s", self.scan_config
        )
        try:
            logger.debug("Turning off the trigger.")
            self.trigger_off()
            self.scan_data_manager.initialize_scan_data_and_output_files()
        except Exception:
            logger.exception("Pre-logging setup failed. Aborting scan.")
            return pd.DataFrame()

        if self.stop_scanning_thread_event.is_set():
            logger.info("Stop requested before scan directory was used; aborting.")
            return pd.DataFrame()

        scan_dir = str(self.scan_data_manager.scan_paths.get_folder())
        scan_id = getattr(self.scan_data_manager, "parsed_scan_string", None)
        if not scan_id:
            num = getattr(self.scan_data_manager, "scan_number_int", None)
            scan_id = f"Scan{int(num):03d}" if num is not None else "Scan-UNKNOWN"

        # ------------------------------------------------------------------ #
        # Phase 2: remaining prelogging + full scan lifecycle, all captured   #
        # inside the per-scan log.  Aborts during setup are now visible in    #
        # scan.log, not only in the global geecs_scanner.log.                 #
        # stop_scan() is in the finally block so cleanup is always logged.    #
        # ------------------------------------------------------------------ #
        with scan_log(scan_id=scan_id, scan_dir=scan_dir):
            try:
                logger.info("scan %s: starting (dir=%s)", scan_id, scan_dir)
                self.pre_logging_setup()

                if self.scan_config:
                    self.estimate_acquisition_time()
                    logger.debug(
                        "Estimated acquisition time based on scan config: %s seconds.",
                        self.acquisition_time,
                    )
                logger.debug("options dict: %s", self.options_dict)

                if self.stop_scanning_thread_event.is_set():
                    raise ScanAbortedError("Stop requested after prelogging")

                self.results = self.data_logger.start_logging()

                if self.shot_control is not None:
                    self.synchronize_devices()

                self.scan_data_manager.purge_all_local_save_dir()

                self.executor.execute_scan_loop(self.scan_steps)

                logger.info("scan %s: completed normally", scan_id)

            except ScanAbortedError:
                logger.info("Scan aborted; running cleanup.")

            except TriggerError as trig_err:
                logger.critical("Scan aborted: trigger device failed — %s", trig_err)

            except GeecsDeviceInstantiationError:
                logger.exception(
                    "Scan aborted: one or more devices could not be instantiated."
                )

            except DeviceSynchronizationError as sync_err:
                logger.error(
                    "Scan aborted due to device synchronization failure: %s", sync_err
                )

            except Exception:
                logger.exception("Error during scan execution")

            finally:
                logger.info("Executing scan cleanup...")
                try:
                    log_df = self.stop_scan()
                except Exception:
                    logger.exception(
                        "Error during scan cleanup - attempting to continue"
                    )

                logger.info("scan %s: finished", scan_id)

        self.scanning_thread = None
        return log_df

    def check_devices_in_standby_mode(self) -> bool:
        """Wait up to 8 s for all devices to enter standby; return False on timeout."""
        timeout = 8
        start_time = time.time()
        while not self.data_logger.all_devices_in_standby:
            if time.time() - start_time > timeout:
                logger.error(
                    "Timeout reached while waiting for all devices to be go into standby."
                )
                return False
            time.sleep(1)
        return True

    def synchronize_devices(self) -> None:
        """Synchronize all devices via global-time or timeout-based fallback.

        Raises
        ------
        DeviceSynchronizationTimeout
            If devices do not synchronize within 15.5 seconds.
        """
        if self.options_dict.get("enable_global_time_sync", False):
            logger.debug("Attempting global time synchronization")
            if self.data_logger.synchronize_devices_global_time():
                logger.info(
                    "Global time synchronization successful. Skipping timeout method."
                )
                self.data_logger.all_devices_in_standby = True
                return
            else:
                logger.debug(
                    "Global time synchronization failed. Falling back to timeout method."
                )

        logger.debug("Using timeout-based synchronization method")
        timeout = 15.5
        start_time = time.time()
        while not self.data_logger.devices_synchronized:
            if time.time() - start_time > timeout:
                logger.error(
                    "Timeout reached while waiting for devices to synchronize."
                )
                raise DeviceSynchronizationTimeout(
                    "Devices failed to synchronize within the allowed timeout."
                )

            if self.data_logger.all_devices_in_standby:
                logger.debug("Sending single-shot trigger to synchronize devices.")

                res = self._set_trigger("SINGLESHOT")
                logger.debug("Result of single shot command: %s", res)
                # wait 3.5 seconds after the test fire to allow time for shot to execute and for devices to respond
                time.sleep(3.5)
                if self.data_logger.devices_synchronized:
                    logger.info("Devices synchronized using timeout method.")
                    break
                else:
                    logger.warning("Not all devices exited standby after single shot.")
                    devices_still_in_standby = [
                        device
                        for device, status in self.data_logger.standby_mode_device_status.items()
                        if status is True
                    ]
                    logger.warning(
                        "Devices still in standby: %s", devices_still_in_standby
                    )
                    logger.debug("Resetting standby status to none for all devices.")
                    self.data_logger.standby_mode_device_status = {
                        key: None for key in self.data_logger.standby_mode_device_status
                    }
                    logger.debug(
                        "Resetting initial timestamp to None for each device to enforce rechecking of stanby mode."
                    )
                    self.data_logger.initial_timestamps = {
                        key: None for key in self.data_logger.initial_timestamps
                    }
                    logger.debug("Waiting for devices to re-enter standby mode.")
                    self.data_logger.all_devices_in_standby = False
            time.sleep(0.1)

    def _join_file_mover_queue(self, timeout: float = 30.0) -> None:
        """Wait for the FileMover queue to drain, with a hard *timeout*.

        Raises
        ------
        OrphanProcessingTimeout
            If the queue has not drained within *timeout* seconds.
        """
        join_thread = threading.Thread(
            target=self.data_logger.file_mover.task_queue.join, daemon=True
        )
        join_thread.start()
        join_thread.join(timeout=timeout)
        if join_thread.is_alive():
            raise OrphanProcessingTimeout(
                f"FileMover task queue did not drain within {timeout:.0f} s. "
                "Some files may not have been moved."
            )

    def stop_scan(self):
        """Stop the scan, persist all data, and restore device state.

        Shutdown order:

        1. Trigger → STANDBY (no new shots)
        2. stop_logging() seals self.results; process_results() + _make_sFile()
        3. _stop_saving_devices() in background thread (parallel save=off + sleep)
        4. file-mover drain  (overlaps with step 3)
        5. restore initial state, closeout actions
        6. join background thread  ← must precede device_manager.reset()
        7. device_manager.reset()

        Returns
        -------
        pd.DataFrame
            Processed scan data.
        """
        log_df = pd.DataFrame()

        # Step 1: Trigger to standby — no new shots from here on.
        self._set_trigger("STANDBY")

        # Step 2: Seal self.results and write scalar files before any device
        # interaction that could fail (restore, closeout).
        self.data_logger.stop_logging()
        log_df = self.scan_data_manager.process_results(self.results)
        self.scan_data_manager._make_sFile(log_df)

        # Step 3: Dispatch camera save=off in a background thread so its 2 s
        # sleep overlaps with file-mover, restore, and closeout below.
        # Must join before device_manager.reset().
        stop_saving_thread = threading.Thread(
            target=self._stop_saving_devices,
            name="stop-saving-devices",
            daemon=True,
        )
        stop_saving_thread.start()

        if self.data_logger.file_mover is not None:
            # Signal that the scan is no longer live so orphaned files
            # are no longer skipped during task processing.
            self.data_logger.file_mover.scan_is_live = False

            # Re-queue tasks that failed during live acquisition (file may
            # not have been on disk yet when the worker first checked).
            self.data_logger.file_mover._post_process_orphan_task()
            try:
                self._join_file_mover_queue(timeout=30.0)
            except OrphanProcessingTimeout:
                logger.error(
                    "Orphan task queue did not drain within 30 s. "
                    "Some files may not have been moved — check the scan directory manually."
                )

            if self.save_local:
                # Sweep the filesystem for any remaining unmatched files
                # and create new tasks for them based on the log DataFrame.
                self.data_logger.file_mover._post_process_orphaned_files(
                    log_df=log_df,
                    device_save_paths_mapping=self.scan_data_manager.device_save_paths_mapping,
                )
                try:
                    self._join_file_mover_queue(timeout=30.0)
                except OrphanProcessingTimeout:
                    logger.error(
                        "Filesystem sweep tasks did not drain within 30 s. "
                        "Some files may not have been moved — check the scan directory manually."
                    )

            self.data_logger.file_mover.shutdown(wait=False)  # queue already drained
            self.data_logger.file_mover = None
        else:
            logger.debug("Logging never started; skipping file-mover cleanup.")

        # Step 4: Wait for save=off to complete before restore or closeout,
        # since closeout actions may talk to the same camera devices.
        stop_saving_thread.join()
        logger.debug("Non-scalar device save states reset.")

        # Step 5: Restore the initial state of devices.
        if self.initial_state is not None:
            self.restore_initial_state(self.initial_state)

        # Step 6: Execute closeout actions.  Wrapped so that a device error
        # (e.g. GeecsDeviceCommandRejected) cannot prevent data already
        # written above from being preserved.
        if self.device_manager.scan_closeout_action is not None:
            logger.debug("Attempting to execute closeout actions.")
            logger.debug("Action list %s", self.device_manager.scan_closeout_action)
            try:
                self.action_manager.add_action(
                    action_name="closeout_action",
                    action_seq=self.device_manager.scan_closeout_action,
                )
                self.action_manager.execute_action("closeout_action")
            except Exception:
                logger.exception(
                    "Closeout action failed — scan data has already been written."
                )

        if self.scan_config.scan_mode == ScanMode.OPTIMIZATION:
            scan_dir = self.scan_data_manager.data_txt_path.parent
            xopt_dump_str = str(scan_dir / "xopt_dump.yaml")
            self.optimizer.xopt.dump(xopt_dump_str)

        self.scan_step_start_time = 0
        self.scan_step_end_time = 0
        self.data_logger.idle_time = 0

        # Step 7: Reset the device manager.
        self.device_manager.reset()

        # Step 8: Require reinitialization before the next scan.
        self.initialization_success = False

        return log_df

    def _stop_saving_devices(self):
        """Dispatch save=off and path-reset to all camera devices in parallel.

        Fire-and-forget (sync=False) so the thread returns as soon as all UDP
        packets are queued.  stop_logging() is *not* called here — the caller
        seals self.results first, allowing this thread to overlap with the
        file-mover drain and restore steps.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _reset_device(device_name, device):
            logger.debug("Setting save to off for %s", device_name)
            try:
                device.set("save", "off", sync=False)
            except Exception:
                logger.warning(
                    "Failed to set save=off for %s", device_name, exc_info=True
                )
            try:
                device.set("localsavingpath", "c:\\temp", sync=False)
            except Exception:
                logger.warning(
                    "Failed to reset localsavingpath for %s", device_name, exc_info=True
                )
            logger.debug("Save state reset for %s", device_name)

        devices = {
            name: dev
            for name in self.device_manager.non_scalar_saving_devices
            if (dev := self.device_manager.devices.get(name)) is not None
        }
        for name in set(self.device_manager.non_scalar_saving_devices) - set(devices):
            logger.warning("Device %s not found in DeviceManager.", name)

        if devices:
            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                futures = {
                    executor.submit(_reset_device, name, dev): name
                    for name, dev in devices.items()
                }
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        logger.exception(
                            "Error resetting save state for %s", futures[future]
                        )

        logger.debug("scanning has stopped for all devices.")

    def save_hiatus(self, hiatus_period: float):
        """Disable saving on all non-scalar devices, wait *hiatus_period* seconds, then re-enable."""
        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                logger.debug("Setting save to off for %s", device_name)
                device.set("save", "off", sync=False)
            else:
                logger.warning("Device %s not found in DeviceManager.", device_name)

        logger.debug("All devices with Save OFF for %s s", hiatus_period)
        time.sleep(hiatus_period)

        for device_name in self.device_manager.non_scalar_saving_devices:
            device = self.device_manager.devices.get(device_name)
            if device:
                logger.debug("Setting save to on for %s", device_name)
                device.set("save", "on", sync=False)
            else:
                logger.warning("Device %s not found in DeviceManager.", device_name)

    def pre_logging_setup(self):
        """Configure devices and files for data acquisition.

        Called from within the per-scan log context, after the scan directory
        has been created and the trigger disabled.  Raises ``ScanAbortedError``
        at each cooperative check-point so that a stop request is honoured
        between steps rather than waiting for the entire setup to finish.

        Setup steps (trigger-off and path initialisation are done by the
        caller before this method is invoked):

        1. Generate scan steps
        2. Configure device save paths
        3. Initialize scan variables through DeviceManager
        4. Settle delay (interruptible)
        5. Capture initial device states
        6. Execute pre-scan setup actions
        7. Generate live experiment configuration dump

        Raises
        ------
        ScanAbortedError
            If ``stop_scanning_thread_event`` is set between steps.
        GeecsDeviceInstantiationError
            If device initialization or configuration fails.
        """
        self.scan_steps = self._generate_scan_steps()
        logger.debug("steps for the scan are : %s", self.scan_steps)

        if self.stop_scanning_thread_event.is_set():
            raise ScanAbortedError("Stop requested during prelogging")

        self.scan_data_manager.configure_device_save_paths(
            save_local=self.save_local,
            stop_event=self.stop_scanning_thread_event,
        )
        self.data_logger.save_local = self.save_local
        self.data_logger.set_device_save_paths_mapping(
            self.scan_data_manager.device_save_paths_mapping
        )
        self.data_logger.scan_number = (
            self.scan_data_manager.scan_number_int
        )  # TODO replace with a `set` func.

        if self.stop_scanning_thread_event.is_set():
            raise ScanAbortedError("Stop requested during prelogging")

        self.scan_data_manager.write_scan_info_ini(self.scan_config)

        logger.debug("scan config in pre logging is this: %s", self.scan_config)
        try:
            self.device_manager.handle_scan_variables(self.scan_config)
        except GeecsDeviceInstantiationError:
            logger.exception(
                "Device instantiation failed during handling of scan devices"
            )
            raise

        # Interruptible settle delay — returns immediately if stop is signalled.
        if self.stop_scanning_thread_event.wait(timeout=2.5):
            raise ScanAbortedError("Stop requested during prelogging settle delay")

        device_var = self.scan_config.device_var
        if device_var:
            self.initial_state = self.get_initial_state()

        if self.stop_scanning_thread_event.is_set():
            raise ScanAbortedError("Stop requested during prelogging")

        if self.device_manager.scan_setup_action is not None:
            logger.debug("Attempting to execute pre-scan actions.")
            logger.debug("Action list %s", self.device_manager.scan_setup_action)

            self.action_manager.add_action(
                action_name="setup_action",
                action_seq=self.device_manager.scan_setup_action,
            )
            self.action_manager.execute_action("setup_action")

        if self.stop_scanning_thread_event.is_set():
            raise ScanAbortedError("Stop requested during prelogging")

        logger.debug("attempting to generate ECS live dump using %s", self.MC_ip)
        if self.MC_ip is not None and self.shot_control is not None:
            self.generate_live_ECS_dump(self.MC_ip)

        logger.debug("Pre-logging setup completed.")

    def enable_live_ECS_dump(self, client_ip: str = "192.168.0.1"):
        """Send UDP commands to MC to enable remote ECS dumps for this scan."""
        if self.shot_control is None:
            logger.error("Cannot enable live ECS dump without shot control device")
            return

        steps = ["enable remote scan ECS dumps", "Main: Check scans path>>None"]

        for step in steps:
            success = self.shot_control.dev_udp.send_scan_cmd(step, client_ip=client_ip)
            time.sleep(0.5)
            logger.debug("enable live ecs dumps step %s complete", step)
            if not success:
                logger.warning(
                    "Failed to enable live ECS dumps on MC on computer: %s", client_ip
                )
                break

    def generate_live_ECS_dump(self, client_ip: str = "192.168.0.1"):
        """Trigger a live ECS configuration dump at scan start."""
        if self.shot_control is None:
            logger.error("Cannot enable live ECS dump without shot control device")
            return
        logger.debug("sending comands to MC to generate ECS live dump")

        steps = ["Save Live Expt Devices Configuration>>ScanStart"]

        for step in steps:
            success = self.shot_control.dev_udp.send_scan_cmd(step, client_ip=client_ip)
            time.sleep(0.5)
            if not success:
                logger.warning("Failed to generate an ECS live dump")
                break

    def _generate_scan_steps(self) -> List[Dict[str, Any]]:
        """Build the list of scan steps from the current scan configuration.

        Returns
        -------
        list[dict]
            Each dict has keys ``'variables'``, ``'wait_time'``, ``'is_composite'``.
            NOSCAN → one step with empty variables; OPTIMIZATION → placeholder
            steps filled in dynamically; STANDARD → linear sweep.
        """
        self.data_logger.bin_num = 0
        steps: List[Dict[str, Any]] = []
        self.optimizer = None

        mode = self.scan_config.scan_mode
        wait_time = self.scan_config.wait_time

        if mode == ScanMode.NOSCAN:
            steps.append(
                {"variables": {}, "wait_time": wait_time, "is_composite": False}
            )

        elif mode == ScanMode.OPTIMIZATION:
            self._setup_optimizer_from_config()

            num_steps = (
                int(
                    abs(
                        (self.scan_config.end - self.scan_config.start)
                        / self.scan_config.step
                    )
                )
                + 1
            )
            for _ in range(num_steps):
                steps.append(
                    {
                        "variables": {},  # to be filled in dynamically later
                        "wait_time": wait_time,
                        "is_composite": False,
                    }
                )

        elif mode == ScanMode.STANDARD:
            current_value = self.scan_config.start
            end = self.scan_config.end
            step = abs(self.scan_config.step)
            device_var = self.scan_config.device_var

            positive = current_value < end
            while (positive and current_value <= end) or (
                not positive and current_value >= end
            ):
                steps.append(
                    {
                        "variables": {device_var: current_value},
                        "wait_time": wait_time,
                        "is_composite": False,
                    }
                )
                current_value += step if positive else -step
        logger.debug("scan steps generate: %s", steps)
        return steps

    def _setup_optimizer_from_config(self):
        """Instantiate the optimizer from ``scan_config.optimizer_config_path``.

        Raises
        ------
        ValueError
            If ``optimizer_config_path`` is not set.
        """
        if not self.scan_config.optimizer_config_path:
            raise ValueError(
                "optimizer_config_path must be set in ScanConfig for optimization scans"
            )

        self.optimizer = BaseOptimizer.from_config_file(
            config_path=self.scan_config.optimizer_config_path,
            scan_data_manager=self.scan_data_manager,
            data_logger=self.data_logger,
        )

        self.device_manager.load_from_dictionary(self.optimizer.device_requirements)

        from collections import defaultdict

        device_variables = defaultdict(list)
        for key in self.optimizer.vocs.variables.keys():
            device, variable = key.split(":", 1)
            device_variables[device].append(variable)

        for device, variables in device_variables.items():
            self.device_manager.add_scan_device(device, variables)

        self.executor.optimizer = self.optimizer
        self.executor.data_logger = self.data_logger

    def estimate_acquisition_time(self):
        """Compute and store the estimated total scan duration in ``self.acquisition_time``."""
        total_time = 0

        if (
            self.scan_config.scan_mode is ScanMode.NOSCAN
            or self.scan_config.scan_mode is ScanMode.BACKGROUND
        ):
            total_time = self.scan_config.wait_time - 0.5
        else:
            start = self.scan_config.start
            end = self.scan_config.end
            step = self.scan_config.step
            wait_time = self.scan_config.wait_time

            steps = abs((end - start) / step) + 1
            total_time += steps * wait_time

        logger.debug("Estimated scan time: %s", total_time)
        self.acquisition_time = total_time

    def estimate_current_completion(self):
        """Return the fraction of the scan completed (0.0–1.0)."""
        if self.acquisition_time == 0:
            return 0
        completion = self.data_logger.get_current_shot() / self.acquisition_time
        return 1 if completion > 1 else completion

    def get_initial_state(self):
        """Return ``{device_var: current_value}`` for the scan variable before manipulation."""
        device_var = self.scan_config.device_var

        if self.device_manager.is_composite_variable(device_var):
            initial_state = {
                f"{device_var}:composite_var": self.device_manager.devices[
                    device_var
                ].state.get("composite_var")
            }
        else:
            device_name, var_name = device_var.split(":")
            initial_state = {
                device_var: self.device_manager.devices[device_name].state.get(var_name)
            }

        logger.debug("Initial scan variable state: %s", initial_state)
        return initial_state

    def restore_initial_state(self, initial_state):
        """Restore each device variable in *initial_state* to its pre-scan value.

        Failures are collected into ``self.restore_failures`` rather than shown
        immediately — blocking a dialog here would deadlock with
        ``stop_scanning_thread().join()`` on the main thread.
        """
        self.restore_failures = []

        for device_var, value in initial_state.items():
            device_name, variable_name = device_var.split(":", 1)

            if device_name in self.device_manager.devices:
                device = self.device_manager.devices[device_name]
                try:
                    device.set(variable_name, value)
                    logger.debug(
                        "Restored %s:%s to %s.", device_name, variable_name, value
                    )
                except DEVICE_COMMAND_ERRORS as e:
                    logger.error(
                        "Failed to restore %s:%s to %s (%s)",
                        device_name,
                        variable_name,
                        value,
                        type(e).__name__,
                    )
                    # Cannot block here — we're in the scan thread, and
                    # stop_scanning_thread().join() on the main thread would
                    # deadlock.  Collect the failure; the main thread will
                    # show a summary once the scan thread has exited.
                    self.restore_failures.append(
                        f"  • {device_name}:{variable_name} → {value}"
                        f"  ({type(e).__name__})"
                    )
                except Exception:
                    logger.exception(
                        "Failed to restore %s:%s to %s",
                        device_name,
                        variable_name,
                        value,
                    )
            else:
                logger.warning(
                    "Device %s not found. Skipping restoration for %s.",
                    device_name,
                    device_var,
                )


if __name__ == "__main__":
    print("testing")
    module_and_class = (
        "scan_analysis.analyzers.Undulator.CameraImageAnalysis.CameraImageAnalysis"
    )
    if module_and_class:
        module_name, class_name = module_and_class.rsplit(".", 1)
        print(module_name, class_name)
        module = importlib.import_module(module_name)
        analysis_class = getattr(module, class_name)
        print(analysis_class)
