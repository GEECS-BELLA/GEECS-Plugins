"""Core GEECS device abstraction with DB lookups, UDP/TCP I/O, scan helpers, and state management."""

from __future__ import annotations

import ast
import configparser
import inspect
import os
import queue
import re
import shutil
import time
from datetime import datetime as dtime
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Optional, Union, Callable

import numpy as np
import numpy.typing as npt

import geecs_python_api.controls.interface.message_handling as mh
from geecs_python_api.controls.api_defs import ThreadInfo, VarAlias
from geecs_python_api.controls.interface import (
    EventHandler,
    GeecsDatabase,
    TcpSubscriber,
    UdpHandler,
)
from geecs_python_api.controls.interface.geecs_errors import (
    GeecsDeviceInstantiationError,
)

if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import AsyncResult, ExpDict, VarDict

import logging

logger = logging.getLogger(__name__)


class GeecsDevice:
    """Device wrapper providing variable access, subscriptions, and scan orchestration."""

    # Static
    threads_lock = Lock()
    all_threads: list[ThreadInfo] = []

    appdata_path: Path
    if os.name == "nt":
        appdata_path = Path(os.getenv("LOCALAPPDATA", "")) / "GEECS"
    elif os.name == "posix":
        appdata_path = Path(os.getenv("TMPDIR", "/tmp")) / "GEECS"
    elif "USERPROFILE" in os.environ:
        appdata_path = Path(os.getenv("USERPROFILE", "")) / "GEECS"
    elif "HOME" in os.environ:
        appdata_path = Path(os.getenv("HOME", "")) / "GEECS"
    else:
        appdata_path = Path.cwd() / "GEECS"

    scan_file_path: Path = appdata_path / "geecs_scan.txt"
    exp_info: dict[str, Any] = {}

    def __init__(self, name: str, virtual: bool = False):
        """Initialize device metadata, I/O endpoints, and subscriptions."""
        # Identity
        self.__dev_name: str = name.strip()
        self.__dev_virtual = virtual or not self.__dev_name
        self.__class_name = re.search(r"\w+\'>$", str(self.__class__))[0][:-2]

        # Variables & state
        self.dev_vars: dict[str, Any] = {}
        self.var_spans: dict[VarAlias, tuple[Optional[float], Optional[float]]] = {}
        self.var_names_by_index: dict[int, tuple[str, VarAlias]] = {}
        self.var_aliases_by_name: dict[str, tuple[VarAlias, int]] = {}
        self.use_alias_in_TCP_subscription: bool = True

        self.setpoints: dict[VarAlias, Any] = {}
        self.state: dict[VarAlias, Any] = {
            "fresh": False,
            "shot number": None,
            "GEECS device error": False,
            "Device alive on instantiation": False,
        }

        self.generic_vars = ["Device Status", "device error", "device preset"]

        if not self.__dev_virtual:
            self.list_variables(GeecsDevice.exp_info.get("devices"))

        # Messaging infra
        self.queue_cmds: queue.Queue = queue.Queue()
        self.own_threads: list[ThreadInfo] = []

        self.notify_on_udp = False
        self.queue_udp_msgs: queue.Queue = queue.Queue()
        self.notifier_udp_msgs = Condition()

        self.notify_on_tcp = False
        self.queue_tcp_msgs: queue.Queue = queue.Queue()
        self.notifier_tcp_msgs = Condition()

        # Comms endpoints
        self.mc_port: int = GeecsDevice.exp_info.get("MC_port", -1)

        self.dev_ip: str = ""
        self.dev_port: int = 0
        if not self.__dev_virtual:
            self.dev_ip, self.dev_port = GeecsDatabase.find_device(self.__dev_name)

        self.dev_tcp: Optional[TcpSubscriber] = None
        self.dev_udp: Optional[UdpHandler] = None

        try:
            self.init_resources()
        except Exception as e:
            raise GeecsDeviceInstantiationError(
                f"Failed to initialize device {name}: {e}"
            ) from e

        # Data roots
        self.data_root_path: Path = GeecsDevice.exp_info.get("data_path", Path("."))

        try:
            GeecsDevice.appdata_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug(
                "failed to create appdata path %s",
                GeecsDevice.appdata_path,
                exc_info=True,
            )

        # Event handler for generic notifications
        self.event_handler = EventHandler(["update"])

        # # Wire TCP callback
        # if self.dev_tcp:
        #     self.dev_tcp.set_message_callback(self.handle_tcp_update)

    # ---- Callbacks & resource setup -------------------------------------------------
    def handle_tcp_update(self, message: str) -> None:
        """Handle raw TCP message and publish an 'update' event."""
        self.state["last_message"] = message
        self.event_handler.publish("update", message)

    def enable_update_events(self) -> None:
        """Enable in-process 'update' events fed from the TCP stream."""
        if self.dev_tcp and getattr(self.dev_tcp, "message_callback", None) is None:
            self.dev_tcp.set_message_callback(self.handle_tcp_update)

    def disable_update_events(self) -> None:
        """Disable in-process 'update' events; TCP stream/state continue."""
        if self.dev_tcp:
            self.dev_tcp.set_message_callback(None)

    def register_update_listener(self, name: str, fn: Callable[[str], None]) -> None:
        """Register a listener for 'update' events and ensure events are enabled."""
        self.event_handler.register("update", name, fn)
        if self.dev_tcp and getattr(self.dev_tcp, "message_callback", None) is None:
            self.dev_tcp.set_message_callback(self.handle_tcp_update)

    def unregister_update_listener(self, name: str) -> None:
        """Unregister a listener and disable events if no listeners remain."""
        self.event_handler.unregister("update", name)
        subs = getattr(self.event_handler, "events", {}).get("update", {})
        if self.dev_tcp and (not subs):  # no remaining subscribers
            self.dev_tcp.set_message_callback(None)

    def has_update_listeners(self) -> bool:
        """Check if listeners are attached."""
        return bool(getattr(self.event_handler, "events", {}).get("update"))

    def init_resources(self) -> None:
        """Create UDP server/client and probe TCP connectivity (if not virtual)."""
        if not self.__dev_virtual:
            if self.dev_udp is None:
                self.dev_udp = UdpHandler(owner=self)
                self.register_cmd_executed_handler()

                if self.is_valid():
                    if self.dev_tcp is None:
                        try:
                            self.dev_tcp = TcpSubscriber(owner=self)
                        except Exception as e:
                            logger.error(
                                "failed to create TcpSubscriber for %s",
                                self.__dev_name,
                                exc_info=True,
                            )
                            raise GeecsDeviceInstantiationError(
                                f"TCP subscriber initialization failed for {self.__dev_name}: {e}"
                            ) from e
                else:
                    logger.error('device "%s" not found in database', self.__dev_name)
                    raise GeecsDeviceInstantiationError(
                        f"Device {self.__dev_name} not found in database"
                    )

                is_device_on = self.dev_tcp.connect() if self.dev_tcp else False
                if not is_device_on:
                    logger.error("TCP connection test failed for %s", self.__dev_name)
                    self.close()
                    raise GeecsDeviceInstantiationError(
                        f"TCP connection test failed for {self.__dev_name}"
                    )
                else:
                    self.dev_tcp.close()
                    self.state["Device alive on instantiation"] = True
        else:
            try:
                self.close()
            except Exception:
                pass
            self.dev_udp = None
            self.dev_tcp = None

    def close(self) -> None:
        """Flush queues and close sockets."""
        mh.flush_queue(self.queue_udp_msgs)
        mh.flush_queue(self.queue_tcp_msgs)
        self.stop_waiting_for_all_cmds()
        if self.dev_udp is not None:
            self.dev_udp.close()
            self.dev_udp = None
        if self.dev_tcp is not None:
            self.dev_tcp.close()
            self.dev_tcp = None

    def reconnect(self) -> None:
        """Tear down and rebuild UDP/TCP endpoints."""
        try:
            self.close()
        except Exception:
            pass

        if not self.__dev_virtual:
            self.dev_udp = UdpHandler(owner=self)
            self.register_cmd_executed_handler()

            if self.is_valid():
                try:
                    self.dev_tcp = TcpSubscriber(owner=self)
                except Exception:
                    logger.exception("failed creating TCP subscriber in reconnect()")
            else:
                logger.warning(
                    'device "%s" not found in DB during reconnect', self.__dev_name
                )
        else:
            self.dev_udp = None
            self.dev_tcp = None

    def is_valid(self) -> bool:
        """Return True if device has a valid DB entry and a TCP port."""
        return (not self.__dev_virtual) and bool(self.dev_ip) and (self.dev_port > 0)

    # ---- Accessors ----------------------------------------------------------
    def get_name(self) -> str:
        """Return device name."""
        return self.__dev_name

    def get_class(self) -> str:
        """Return class name (string)."""
        return self.__class_name

    # ---- Registrations ------------------------------------------------------
    def is_var_listener_connected(self) -> bool:
        """Return True if TCP subscriber is connected."""
        return bool(self.dev_tcp and self.dev_tcp.is_connected())

    def register_var_listener_handler(self) -> bool:
        """Enable TCP subscription callbacks if valid."""
        return bool(self.dev_tcp and self.dev_tcp.register_handler())

    def unregister_var_listener_handler(self) -> None:
        """Disable TCP subscription callbacks."""
        if self.dev_tcp:
            self.dev_tcp.unregister_handler()

    def register_cmd_executed_handler(self) -> bool:
        """Enable UDP executed-command notifications if valid."""
        return bool(self.dev_udp and self.dev_udp.register_handler())

    def unregister_cmd_executed_handler(self) -> None:
        """Disable UDP executed-command notifications."""
        if self.dev_udp:
            self.dev_udp.unregister_handler()

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        """Subscribe the device's TCP stream; keeps state updated quietly."""
        subscribed = False

        if self.is_valid() and variables is None:
            if self.var_names_by_index:
                variables = [var[0] for var in self.var_names_by_index.values()]
            else:
                variables = list(self.dev_vars.keys())

        variables = self.generic_vars + (variables or [])

        if self.is_valid() and variables:
            try:
                self.unregister_var_listener_handler()
                subscribed = bool(
                    self.dev_tcp and self.dev_tcp.subscribe(",".join(variables))
                )
                if subscribed:
                    self.register_var_listener_handler()
            except Exception:
                logger.exception(
                    'subscribe_var_values failed for "%s"', self.__dev_name
                )

        return subscribed

    def unsubscribe_var_values(self) -> None:
        """Stop TCP subscription if connected."""
        if self.is_var_listener_connected() and self.dev_tcp:
            self.dev_tcp.unsubscribe()

    # ---- Variables ----------------------------------------------------------
    def list_variables(
        self, exp_devs: Optional["ExpDict"] = None
    ) -> tuple["VarDict", "ExpDict"]:
        """Populate device variable metadata from `exp_devs` or the database."""
        try:
            if exp_devs is None:
                GeecsDevice.exp_info = GeecsDatabase.collect_exp_info()
                exp_devs = GeecsDevice.exp_info["devices"]

            self.dev_vars = exp_devs[self.__dev_name]
        except Exception:
            if exp_devs is None or self.__dev_name not in exp_devs:
                logger.warning('device "%s" not found in database', self.__dev_name)
            self.dev_vars = {}
        return self.dev_vars, exp_devs  # type: ignore[return-value]

    def find_var_by_alias(self, alias: VarAlias = VarAlias("")) -> str:
        """Return variable name given an alias (falls back to alias-as-name)."""
        if not self.dev_vars:
            self.list_variables()
        if not self.dev_vars:
            return ""
        for attributes in self.dev_vars.values():
            if attributes["alias"] == alias:
                return attributes["variablename"]
        if alias in self.dev_vars:
            return str(alias)
        return ""

    def find_alias_by_var(self, var: str = "") -> VarAlias:
        """Return VarAlias given a variable name (falls back to var)."""
        if not self.dev_vars:
            self.list_variables()
        if not self.dev_vars:
            return VarAlias("")
        for attributes in self.dev_vars.values():
            if attributes["variablename"] == var:
                return VarAlias(attributes["alias"])
        if var in self.dev_vars:
            return VarAlias(var)
        return VarAlias("")

    def build_var_dicts(self) -> None:
        """Build bidirectional maps between index/name and alias."""
        self.var_names_by_index = {
            index: (self.find_var_by_alias(var_alias), var_alias)
            for index, var_alias in enumerate(self.var_spans.keys())
        }
        self.var_aliases_by_name = {
            self.find_var_by_alias(var_alias): (var_alias, index)
            for index, var_alias in enumerate(self.var_spans.keys())
        }

    def _state_value(self, var_name: str) -> Any:
        """Return current state value for a variable or None."""
        if var_name in self.generic_vars:
            var_alias = VarAlias(var_name)
        elif var_name in self.var_aliases_by_name:
            var_alias = self.var_aliases_by_name[var_name][0]
        else:
            var_alias = self.find_alias_by_var(var_name)
        return self.state.get(var_alias, None)

    # ---- Operations ---------------------------------------------------------
    def set(
        self,
        variable: str,
        value: Any,
        exec_timeout: Optional[float] = 120.0,
        attempts_max: int = 5,
        sync: bool = True,
    ) -> Any:
        """Set a variable; returns state if sync else async tuple."""
        ret = self._execute(variable, value, exec_timeout, attempts_max, sync)
        return self._state_value(variable) if (ret and sync) else ret

    def get(
        self,
        variable: str,
        exec_timeout: Optional[float] = 5.0,
        attempts_max: int = 5,
        sync: bool = True,
    ) -> Any:
        """Get a variable; returns state if sync else async tuple."""
        ret = self._execute(variable, None, exec_timeout, attempts_max, sync)
        return self._state_value(variable) if (ret and sync) else ret

    def _execute(
        self,
        variable: str,
        value: Any,
        exec_timeout: Optional[float] = 10.0,
        attempts_max: int = 5,
        sync: bool = True,
    ) -> Optional["AsyncResult"]:
        """Internal execution path: build command, send/ack, and optionally queue async wait."""
        if isinstance(value, (int, str)):
            cmd_str = f"set{variable}>>{value}"
            cmd_label = f"set({variable}, {value})"
        elif isinstance(value, float):
            cmd_str = f"set{variable}>>{value:.12f}"
            cmd_label = f"set({variable}, {value:.12f})"
        elif isinstance(value, bool):
            cmd_str = f"set{variable}>>{int(value)}"
            cmd_label = f"set({variable}, {value})"
        else:
            cmd_str = f"get{variable}>>"
            cmd_label = f"get({variable})"

        if not self.is_valid():
            logger.warning('failed to execute "%s": device not connected', cmd_label)
            return None

        stamp = re.sub(r"[\s.:]", "-", dtime.now().__str__())
        cmd_label += f" @ {stamp}"

        queued: bool = False
        async_thread: ThreadInfo = (None, None)

        self._cleanup_threads()

        if sync:
            self.wait_for_all_cmds(timeout=120.0)
            with GeecsDevice.threads_lock:
                self._process_command(
                    cmd_str,
                    cmd_label,
                    thread_info=(None, None),
                    attempts_max=attempts_max,
                )
                assert self.dev_udp is not None
                self.dev_udp.cmd_checker.wait_for_exe(
                    cmd_tag=cmd_label, timeout=exec_timeout, sync=sync
                )
        elif exec_timeout and exec_timeout > 0:
            with GeecsDevice.threads_lock:
                assert self.dev_udp is not None
                async_thread = self.dev_udp.cmd_checker.wait_for_exe(
                    cmd_tag=cmd_label, timeout=exec_timeout, sync=sync
                )
                if (not self.own_threads) and self.queue_cmds.empty():
                    self._process_command(
                        cmd_str,
                        cmd_label,
                        thread_info=async_thread,
                        attempts_max=attempts_max,
                    )
                else:
                    self.queue_cmds.put(
                        (cmd_str, cmd_label, async_thread, attempts_max)
                    )
                    queued = True

        return queued, cmd_label, async_thread  # type: ignore[return-value]

    # ---- Scans --------------------------------------------------------------
    def scan(
        self,
        var_alias: VarAlias,
        start_value: float,
        end_value: float,
        step_size: float,
        var_span: Optional[tuple[Optional[float], Optional[float]]] = None,
        shots_per_step: int = 10,
        comment: str = "",
        use_alias: bool = True,
        timeout: float = 60.0,
    ) -> tuple[Path, int, bool, bool]:
        """Build a 1D scan file and initiate a file-based scan."""
        var_values = self._scan_values(
            var_alias, start_value, end_value, step_size, var_span
        )
        target_var = var_alias if use_alias else self.find_var_by_alias(var_alias)
        GeecsDevice.write_1D_scan_file(
            self.get_name(), target_var, var_values, shots_per_step
        )
        if not comment:
            comment = f"{var_alias} scan"
        return GeecsDevice.file_scan(self, comment, timeout)

    @staticmethod
    def no_scan(
        monitoring_device: Optional["GeecsDevice"] = None,
        comment: str = "no scan",
        shots: int = 10,
        timeout: float = 300.0,
    ) -> tuple[Path, int, bool, bool]:
        """Trigger a no-scan acquisition via UDP command."""
        cmd = f"ScanStart>>{comment}>>{shots}"
        return GeecsDevice._process_scan(cmd, comment, monitoring_device, timeout)

    @staticmethod
    def file_scan(
        monitoring_device: Optional["GeecsDevice"] = None,
        comment: str = "no scan",
        timeout: float = 300.0,
    ) -> tuple[Path, int, bool, bool]:
        """Trigger a file-backed scan using the prewritten scan file."""
        cmd = f"FileScan>>{GeecsDevice.scan_file_path}"
        return GeecsDevice._process_scan(cmd, comment, monitoring_device, timeout)

    @staticmethod
    def _process_scan(
        cmd: str,
        comment: str = "no scan",
        monitoring_device: Optional["GeecsDevice"] = None,
        timeout: float = 300.0,
    ) -> tuple[Path, int, bool, bool]:
        """Internal helper to send scan command, wait for INI/TDMS, and return final status."""
        if monitoring_device is None:
            dev = GeecsDevice("tmp", virtual=True)
            dev.dev_udp = UdpHandler(owner=dev)
        else:
            dev = monitoring_device

        next_folder, next_scan = dev.next_scan_folder()
        command_accepted = timed_out = False

        txt_file_name = f"ScanData{os.path.basename(next_folder)}.txt"
        txt_file_path: Path = next_folder / txt_file_name

        try:
            assert dev.dev_udp is not None
            command_accepted = dev.dev_udp.send_scan_cmd(cmd)
            ini_file_name = f"ScanInfo{next_folder.name}.ini"
            ini_file_path = next_folder / ini_file_name
            ini_found = False
            if command_accepted:
                # wait for .ini file creation
                t0 = time.monotonic()
                while True:
                    if (time.monotonic() - t0) > 10.0:
                        break
                    if ini_file_path.is_file():
                        ini_found = True
                        break
                    time.sleep(0.2)

                if ini_found:
                    GeecsDevice.update_ini_file(ini_file_path, comment)

            timed_out = GeecsDevice.wait_for_scan_start(
                next_folder, next_scan, timeout=60.0
            )
            if not timed_out:
                if not dev.is_valid():
                    time.sleep(2.0)  # buffer since cannot verify in 'scan' mode

                # wait for 'no scan' status or .txt file as end-of-scan
                t0 = time.monotonic()
                while True:
                    timed_out = (time.monotonic() - t0) > timeout
                    if txt_file_path.is_file() or timed_out:
                        break
                    time.sleep(1.0)
        except Exception:
            logger.exception("scan process failed")
        finally:
            if monitoring_device is None:
                try:
                    dev.close()
                except Exception:
                    pass

        return next_folder, next_scan, command_accepted, timed_out

    @staticmethod
    def update_ini_file(ini_file_path: Union[str, Path], comment: str) -> None:
        """Patch ScanInfo INI with the scan number, comment, and parameter label."""
        ini_file_path = Path(ini_file_path)
        backup_file_path = str(ini_file_path) + "~"

        match = re.search(r"ScanInfoScan(\d{3})\.ini$", str(ini_file_path))
        if not match:
            logger.warning(
                "INI filename does not match expected format: %s", ini_file_path
            )
            return
        scan_number = int(match.group(1))

        try:
            shutil.copy2(ini_file_path, backup_file_path)
            config = configparser.ConfigParser()
            config.optionxform = lambda option: option  # preserve case
            config.read(backup_file_path)

            if "Scan Info" not in config:
                config["Scan Info"] = {}

            config["Scan Info"]["Scan No"] = str(scan_number)
            config["Scan Info"]["ScanStartInfo"] = comment
            config["Scan Info"]["Scan Parameter"] = "Shotnumber"

            with open(ini_file_path, "w") as configfile:
                config.write(configfile)
            logger.debug("updated INI %s", ini_file_path)
        except Exception:
            logger.exception("error updating INI file %s", ini_file_path)
            # Restore original
            try:
                if Path(ini_file_path).exists():
                    os.remove(ini_file_path)
                shutil.move(backup_file_path, ini_file_path)
            except Exception:
                logger.exception("failed to restore original INI after error")
        else:
            try:
                os.remove(backup_file_path)
            except Exception:
                logger.debug(
                    "failed to remove backup INI %s", backup_file_path, exc_info=True
                )

    # ---- Value access helpers ----------------------------------------------
    def get_status(
        self, exec_timeout: float = 2.0, sync: bool = True
    ) -> Optional[Union[float, "AsyncResult"]]:
        """Return 'Device Status' state if sync else async tuple."""
        return self.get("Device Status", exec_timeout=exec_timeout, sync=sync)

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        """Convert a string to float by default (override in subclasses)."""
        try:
            return float(val_string)
        except Exception:
            return val_string

    def interpret_generic_variables(self, var: str, val: str) -> None:
        """Store generic variables as-is in state."""
        self.state[VarAlias(var)] = val

    # ---- Command queueing ---------------------------------------------------
    def dequeue_command(self) -> None:
        """Process all queued commands sequentially."""
        self._cleanup_threads()
        with GeecsDevice.threads_lock:
            while not self.queue_cmds.empty():
                try:
                    cmd_str, cmd_label, async_thread, attempts_max = (
                        self.queue_cmds.get_nowait()
                    )
                    self._process_command(
                        cmd_str,
                        cmd_label,
                        thread_info=async_thread,
                        attempts_max=attempts_max,
                    )
                    time.sleep(0.5)
                except queue.Empty:
                    break

    def _process_command(
        self,
        cmd_str: str,
        cmd_label: str,
        thread_info: ThreadInfo = (None, None),
        attempts_max: int = 5,
    ) -> None:
        """Send command and optionally start its listener thread upon ack."""
        accepted = False
        try:
            for attempt in range(attempts_max):
                assert self.dev_udp is not None
                sent = self.dev_udp.send_cmd(
                    ipv4=(self.dev_ip, self.dev_port), msg=cmd_str
                )
                if sent:
                    accepted = self.dev_udp.ack_cmd(timeout=5.0)
                else:
                    time.sleep(0.1)
                    continue
                if accepted:
                    break
                time.sleep(0.1)
        except Exception:
            logger.exception('command processing failed for "%s"', cmd_label)

        # Raise exception if command was never acknowledged after all attempts
        if not accepted:
            from geecs_python_api.controls.interface.geecs_errors import (
                GeecsDeviceCommandRejected,
            )

            raise GeecsDeviceCommandRejected(
                device_name=self.get_name(),
                command=cmd_str,
                ipv4=(self.dev_ip, self.dev_port),
            )

        # Command accepted - start listener thread if needed
        if accepted and (thread_info[0] is not None):
            thread_info[0].start()
            self.own_threads.append(thread_info)
            GeecsDevice.all_threads.append(thread_info)

    # ---- Message handlers ---------------------------------------------------
    def handle_response(self, net_msg: mh.NetworkMessage) -> tuple[str, str, str, bool]:
        """Parse UDP response, update state, and raise exception on hardware errors."""
        try:
            response = GeecsDevice._response_parser(net_msg.msg)
            if len(response) == 4:
                dev_name, cmd_received, dev_val, err_status = response
            else:
                logger.warning("unexpected response structure: %r", response)
                dev_name = cmd_received = dev_val = ""
                err_status = True

            if self.notify_on_udp:
                self.queue_udp_msgs.put((dev_name, cmd_received, dev_val, err_status))
                with self.notifier_udp_msgs:
                    self.notifier_udp_msgs.notify_all()

            if dev_name != self.__dev_name:
                logger.warning(
                    'mismatch in device name: got "%s", expected "%s"',
                    dev_name,
                    self.__dev_name,
                )

            if (
                dev_name == self.get_name()
                and (not err_status)
                and cmd_received[:3] == "get"
            ):
                if cmd_received[3:] in self.generic_vars:
                    self.interpret_generic_variables(cmd_received[3:], dev_val)
                elif cmd_received[3:] in self.var_aliases_by_name:
                    var_alias = self.var_aliases_by_name[cmd_received[3:]][0]
                    self.state[var_alias] = self.interpret_value(var_alias, dev_val)
                else:
                    var_alias = self.find_alias_by_var(cmd_received[3:])
                    try:
                        self.state[var_alias] = float(dev_val)
                    except Exception:
                        self.state[var_alias] = dev_val

            if (
                dev_name == self.get_name()
                and (not err_status)
                and cmd_received[:3] == "set"
            ):
                if cmd_received[3:] in self.var_aliases_by_name:
                    var_alias = self.var_aliases_by_name[cmd_received[3:]][0]
                else:
                    var_alias = self.find_alias_by_var(cmd_received[3:])
                coerced = self.interpret_value(var_alias, dev_val)
                # fallback to safe literal evaluation if interpret_value returns a string
                if isinstance(coerced, str):
                    try:
                        coerced = ast.literal_eval(coerced)
                    except Exception:
                        pass
                self.setpoints[var_alias] = coerced
                self.state[var_alias] = coerced

            self.state["GEECS device error"] = err_status

            # Raise exception if hardware reported an error
            if err_status:
                from geecs_python_api.controls.interface.geecs_errors import (
                    GeecsDeviceCommandFailed,
                )

                # Extract error detail from message
                error_detail = "Unknown error"
                if ">>" in net_msg.msg:
                    try:
                        _, err_msg = net_msg.msg.rsplit(">>", 1)
                        if "," in err_msg:
                            _, error_detail = err_msg.split(",", 1)
                    except Exception:
                        error_detail = net_msg.msg

                raise GeecsDeviceCommandFailed(
                    device_name=dev_name,
                    command=cmd_received,
                    error_detail=error_detail.strip(),
                    actual_value=dev_val,
                )

            return dev_name, cmd_received, dev_val, err_status
        except GeecsDeviceCommandFailed:
            # Let hardware errors propagate
            raise
        except Exception:
            logger.exception("handle_response failed")
            return "", "", "", True

    def handle_subscription(
        self, net_msg: mh.NetworkMessage
    ) -> tuple[str, int, dict[str, str]]:
        """Parse TCP subscription update, update state, and return components."""
        try:
            dev_name, shot_nb, dict_vals = GeecsDevice._subscription_parser(net_msg.msg)

            if self.notify_on_tcp:
                self.queue_tcp_msgs.put((dev_name, shot_nb, dict_vals))
                with self.notifier_tcp_msgs:
                    self.notifier_tcp_msgs.notify_all()

            if dev_name == self.get_name() and dict_vals:
                self.state["shot number"] = shot_nb
                for var, val in dict_vals.items():
                    if var in self.generic_vars:
                        self.interpret_generic_variables(var, val)
                    if var in self.var_aliases_by_name:
                        var_alias: VarAlias = self.var_aliases_by_name[var][0]
                        self.state[var_alias] = self.interpret_value(var_alias, val)
                        self.state["fresh"] = True
                    else:
                        if self.use_alias_in_TCP_subscription:
                            var_alias = self.find_alias_by_var(var)
                        else:
                            var_alias = VarAlias(var)  # keep as provided
                        try:
                            fval: Any = float(val)
                        except Exception:
                            fval = val
                        self.state[var_alias] = fval
                        self.state["fresh"] = True

            return dev_name, shot_nb, dict_vals
        except Exception:
            logger.exception("handle_subscription failed")
            return "", 0, {}

    # ---- Parsers ------------------------------------------------------------
    @staticmethod
    def _subscription_parser(msg: str = "") -> tuple[str, int, dict[str, str]]:
        """Parse subscription message into (device, shot, {var: val})."""
        pattern = re.compile(r"[^,]+nval,[^,]+nvar")
        blocks = msg.split(">>")
        dev_name = blocks[0]
        shot_nb = int(blocks[1])
        vars_vals = pattern.findall(blocks[-1])
        dict_vals = {
            s.split(",")[0][:-5].strip(): s.split(",")[1][:-5] for s in vars_vals
        }
        return dev_name, shot_nb, dict_vals

    @staticmethod
    def _response_parser(msg: str = "") -> tuple[str, str, str, bool]:
        """Parse response message into (device, command, value, is_error)."""
        dev_name, cmd_received, dev_val, err_msg = msg.split(">>")
        err_status_str, err_detail = err_msg.split(",", 1)
        err_status = err_status_str == "error"
        if err_status:
            logger.error('control-system error for "%s": %s', cmd_received, err_detail)
        return dev_name, cmd_received, dev_val, err_status

    # ---- Coercion & ranges --------------------------------------------------
    def coerce_float(
        self,
        var_alias: VarAlias,
        method: str,
        value: float,
        var_span: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> float:
        """Clamp value to configured span; log if coercion occurs."""
        try:
            if var_span is None:
                var_span = self.var_spans.get(var_alias, (None, None))

            lo, hi = var_span
            if (lo is not None) and value < lo:
                if method:
                    logger.warning(
                        "%s value coerced from %s to %s", var_alias, value, lo
                    )
                value = lo
            if (hi is not None) and value > hi:
                if method:
                    logger.warning(
                        "%s value coerced from %s to %s", var_alias, value, hi
                    )
                value = hi
        except Exception:
            logger.exception("failed to coerce value")
        return value

    def _scan_values(
        self,
        var_alias: VarAlias,
        start_value: float,
        end_value: float,
        step_size: float,
        var_span: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> npt.ArrayLike:
        """Return an inclusive numpy arange respecting sign of step and bounds."""
        start_value = self.coerce_float(
            var_alias, inspect.stack()[0][3], start_value, var_span
        )
        end_value = self.coerce_float(
            var_alias, inspect.stack()[0][3], end_value, var_span
        )
        step = -abs(step_size) if end_value < start_value else abs(step_size)
        return np.arange(start_value, end_value + step, step)

    # ---- Scan file I/O ------------------------------------------------------
    @staticmethod
    def write_1D_scan_file(
        devices: Union[list[str], str],
        variables: Union[list[str], str],
        values_by_row: Union[np.ndarray, list],
        shots_per_step: int = 10,
    ) -> None:
        """Write a simple 1D scan definition to GeecsDevice.scan_file_path."""
        scan_number = 1
        with open(GeecsDevice.scan_file_path, "w+", encoding="utf-8") as f:
            f.write(f"[Scan{scan_number}]\n")
            if isinstance(devices, list):
                f.write('Device = "' + ",".join(devices) + '"\n')
            else:
                f.write(f'Device = "{devices}"\n')

            if isinstance(variables, list):
                f.write('Variable = "' + ",".join(variables) + '"\n')
            else:
                f.write(f'Variable = "{variables}"\n')

            f.write('Values:#shots = "')
            arr = (
                np.array(values_by_row)
                if isinstance(values_by_row, list)
                else values_by_row
            )
            if getattr(arr, "ndim", 1) > 1:
                for col in range(arr.shape[1]):
                    f.write(f"({str(list(arr[:, col]))[1:-1]}):{shots_per_step}|")
            else:
                for col in range(arr.size):
                    f.write(f"({arr[col]}):{shots_per_step}|")
            f.write('"')

    # ---- Data locations -----------------------------------------------------
    def today_data_folder(self) -> Path:
        """Return today's data folder path."""
        stamp = dtime.now()
        date_folders = os.path.join(
            stamp.strftime("Y%Y"),
            stamp.strftime("%m-%B")[:6],
            stamp.strftime("%y_%m%d"),
        )
        return self.data_root_path / date_folders

    def last_scan_number(self) -> int:
        """Return last scan index for today or -1 if none."""
        data_folder: Path = self.today_data_folder()
        scans_dir = data_folder / "scans"
        if not scans_dir.is_dir() or not next(os.walk(scans_dir))[1]:
            return -1
        scan_folders: list[str] = next(os.walk(scans_dir))[1]
        scan_folders = [
            x for x in scan_folders if re.match(r"^Scan(?P<scan>\d{3})$", x)
        ]
        return int(scan_folders[-1][-3:]) if scan_folders else -1

    def next_scan_folder(self) -> tuple[Path, int]:
        """Return (path, index) for the next scan directory."""
        last_scan: int = self.last_scan_number()
        data_folder: Path = self.today_data_folder()
        next_folder = f"Scan{last_scan + 1:03d}" if last_scan > 0 else "Scan001"
        return data_folder / "scans" / next_folder, last_scan + 1

    @staticmethod
    def wait_for_scan_start(
        next_folder: Path, next_scan: int, timeout: float = 60.0
    ) -> bool:
        """Wait until the TDMS file for the upcoming scan appears or timeout; return True if timed out."""
        t0 = time.monotonic()
        while True:
            if (time.monotonic() - t0) > timeout:
                return True
            tdms_filepath = next_folder / f"Scan{next_scan:03d}.tdms"
            if next_folder.is_dir() and tdms_filepath.is_file():
                return False
            time.sleep(0.1)

    # ---- Synchronization ----------------------------------------------------
    @staticmethod
    def cleanup_all_threads() -> None:
        """Remove finished threads from the global registry."""
        with GeecsDevice.threads_lock:
            for it in range(len(GeecsDevice.all_threads)):
                if not GeecsDevice.all_threads[-1 - it][0].is_alive():
                    GeecsDevice.all_threads.pop(-1 - it)

    def _cleanup_threads(self) -> None:
        """Remove finished threads from device and global registries."""
        with GeecsDevice.threads_lock:
            for it in range(len(self.own_threads)):
                try:
                    if not self.own_threads[-1 - it][0].is_alive():
                        self.own_threads.pop(-1 - it)
                except Exception:
                    continue
            for it in range(len(GeecsDevice.all_threads)):
                try:
                    if not GeecsDevice.all_threads[-1 - it][0].is_alive():
                        GeecsDevice.all_threads.pop(-1 - it)
                except Exception:
                    continue

    @staticmethod
    def wait_for_all_devices(timeout: Optional[float] = None) -> bool:
        """Join all device threads with an optional timeout; return True if all finished."""
        GeecsDevice.cleanup_all_threads()
        synced = True
        with GeecsDevice.threads_lock:
            for thread in GeecsDevice.all_threads:
                thread[0].join(timeout)
                synced &= not thread[0].is_alive()
        return synced

    @staticmethod
    def stop_waiting_for_all_devices() -> None:
        """Signal all device threads to stop."""
        GeecsDevice.cleanup_all_threads()
        with GeecsDevice.threads_lock:
            for thread in GeecsDevice.all_threads:
                thread[1].set()

    def wait_for_all_cmds(self, timeout: Optional[float] = None) -> bool:
        """Join all of this device's threads; return True when none remain alive."""
        self._cleanup_threads()
        any_alive = False
        with GeecsDevice.threads_lock:
            for thread in self.own_threads:
                thread[0].join(timeout)
                any_alive |= thread[0].is_alive()
        return not any_alive

    def stop_waiting_for_all_cmds(self) -> None:
        """Signal this device's threads to stop."""
        self._cleanup_threads()
        with GeecsDevice.threads_lock:
            for thread in self.own_threads:
                thread[1].set()

    def wait_for_cmd(self, thread: Thread, timeout: Optional[float] = None) -> bool:
        """Join a single thread with an optional timeout; return True if finished."""
        with GeecsDevice.threads_lock:
            if self.is_valid() and thread.is_alive():
                thread.join(timeout)
            alive = thread.is_alive()
        return not alive

    def stop_waiting_for_cmd(self, thread: Thread, stop: Event) -> None:
        """Signal a single thread to stop."""
        if self.is_valid() and thread.is_alive():
            stop.set()


# Demo block intentionally omitted: library code should not configure logging or perform I/O.
