""" @author: Guillaume Plateau, TAU Systems """

from __future__ import annotations
from typing import Optional, Any, Union, TYPE_CHECKING
from geecs_python_api.controls.api_defs import VarAlias
if TYPE_CHECKING:
    from geecs_python_api.controls.api_defs import VarDict, ExpDict, AsyncResult, ThreadInfo
import queue
import re
import inspect
import time
import os
import shutil
import numpy as np
from queue import Queue
import numpy.typing as npt
from pathlib import Path
from threading import Thread, Condition, Event, Lock
from datetime import datetime as dtime
import geecs_python_api.controls.interface.message_handling as mh
from geecs_python_api.controls.interface import GeecsDatabase, UdpHandler, TcpSubscriber, ErrorAPI, api_error


class GeecsDevice:
    # Static
    threads_lock = Lock()
    all_threads: list[ThreadInfo] = []

    appdata_path: Path
    if os.name == 'nt':
        appdata_path = Path(os.getenv('LOCALAPPDATA')) / 'GEECS'
    elif os.name == 'posix':
        appdata_path = Path(os.getenv('TMPDIR')) / 'GEECS'
    elif 'USERPROFILE' in os.environ:
        appdata_path = Path(os.getenv('USERPROFILE')) / 'GEECS'
    elif 'HOME' in os.environ:
        appdata_path = Path(os.getenv('HOME')) / 'GEECS'
    else:
        appdata_path = Path(os.getenv('LOCALAPPDATA')) / 'GEECS'

    scan_file_path: Path = appdata_path / 'geecs_scan.txt'
    exp_info: dict[str, Any] = {}

    def __init__(self, name: str, virtual: bool = False):

        # Static properties
        self.__dev_name: str = name.strip()
        self.__dev_virtual = virtual or not self.__dev_name
        self.__class_name = re.search(r'\w+\'>$', str(self.__class__))[0][:-2]

        # Communications
        self.dev_tcp: Optional[TcpSubscriber] = None
        self.dev_udp: Optional[UdpHandler]
        self.mc_port: int = 0
        self.mc_port = GeecsDevice.exp_info['MC_port']  # needed to initialize dev_udp

        self.dev_udp: Optional[UdpHandler]
        if not self.__dev_virtual:
            self.dev_udp = UdpHandler(owner=self)
        else:
            self.dev_udp = None

        self.dev_ip: str = ''
        self.dev_port: int = 0

        # Variables
        self.dev_vars = {}
        self.var_spans: dict[VarAlias, tuple[Optional[float], Optional[float]]] = {}
        self.var_names_by_index: dict[int, tuple[str, VarAlias]] = {}
        self.var_aliases_by_name: dict[str, tuple[VarAlias, int]] = {}

        self.setpoints: dict[VarAlias, Any] = {}
        self.state: dict[VarAlias, Any] = {}
        self.generic_vars = ['device status', 'device error', 'device preset']

        # Message handling
        self.queue_cmds = Queue()
        self.own_threads: list[(Thread, Event)] = []

        self.notify_on_udp = False
        self.queue_udp_msgs = Queue()
        self.notifier_udp_msgs = Condition()

        self.notify_on_tcp = False
        self.queue_tcp_msgs = Queue()
        self.notifier_tcp_msgs = Condition()

        if not self.__dev_virtual:
            self.dev_ip, self.dev_port = GeecsDatabase.find_device(self.__dev_name)
            self.register_cmd_executed_handler()

            if self.is_valid():
                # print(f'Device "{self.dev_name}" found: {self.dev_ip}, {self.dev_port}')
                try:
                    self.dev_tcp = TcpSubscriber(owner=self)
                    self.connect_var_listener()
                    self.register_var_listener_handler()
                except Exception:
                    api_error.error('Failed creating TCP subscriber', 'GeecsDevice class, method "__init__"')
            else:
                api_error.warning(f'Device "{self.__dev_name}" not found', 'GeecsDevice class, method "__init__"')

            self.list_variables(GeecsDevice.exp_info['devices'])

        # Data
        self.data_root_path: Path = GeecsDevice.exp_info['data_path']

        if not GeecsDevice.appdata_path.is_dir():
            os.makedirs(GeecsDevice.appdata_path)

    def close(self):
        mh.flush_queue(self.queue_udp_msgs)
        mh.flush_queue(self.queue_tcp_msgs)

        self.stop_waiting_for_all_cmds()

        if self.dev_udp:
            self.dev_udp.close()

        if self.dev_tcp:
            self.dev_tcp.close()

    def reconnect(self):
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
                    self.connect_var_listener()
                    self.register_var_listener_handler()
                except Exception:
                    api_error.error('Failed creating TCP subscriber', 'GeecsDevice class, method "__init__"')
            else:
                api_error.warning(f'Device "{self.__dev_name}" not found', 'GeecsDevice class, method "__init__"')
        else:
            self.dev_udp = None
            self.dev_tcp = None

    def is_valid(self) -> bool:
        return not self.__dev_virtual and self.dev_ip and self.dev_port > 0

    # Accessors
    # -----------------------------------------------------------------------------------------------------------
    def get_name(self):
        return self.__dev_name

    def get_class(self):
        return self.__class_name

    # Registrations
    # -----------------------------------------------------------------------------------------------------------
    def connect_var_listener(self):
        if self.is_valid() and not self.is_var_listener_connected():
            self.dev_tcp.connect((self.dev_ip, self.dev_port))

            if not self.dev_tcp.is_connected():
                api_error.warning('Failed to connect TCP subscriber', f'GeecsDevice "{self.__dev_name}"')

    def is_var_listener_connected(self):
        return self.dev_tcp and self.dev_tcp.is_connected()

    def register_var_listener_handler(self):
        return self.dev_tcp.register_handler()  # happens only if is_valid()

    def unregister_var_listener_handler(self):
        return self.dev_tcp.unregister_handler()

    def register_cmd_executed_handler(self):
        return self.dev_udp.register_handler()  # happens only if is_valid()

    def unregister_cmd_executed_handler(self):
        return self.dev_udp.unregister_handler()

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        subscribed = False

        if self.is_valid() and variables is None:
            if self.var_names_by_index:
                variables = [var[0] for var in self.var_names_by_index.values()]
            else:
                variables = list(self.dev_vars.keys())

        variables = self.generic_vars + variables

        if self.is_valid() and variables:
            try:
                subscribed = self.dev_tcp.subscribe(','.join(variables))
            except Exception as ex:
                api_error.error(str(ex), 'Class GeecsDevice, method "subscribe_var_values"')

        return subscribed

    def unsubscribe_var_values(self):
        if self.is_var_listener_connected():
            self.dev_tcp.unsubscribe()

    # Variables
    # -----------------------------------------------------------------------------------------------------------
    def list_variables(self, exp_devs: Optional[ExpDict] = None) -> tuple[VarDict, ExpDict]:
        try:
            if exp_devs is None:
                GeecsDevice.exp_info = GeecsDatabase.collect_exp_info()
                exp_devs = GeecsDevice.exp_info['devices']

            self.dev_vars = exp_devs[self.__dev_name]

        except Exception:
            if self.__dev_name not in exp_devs:
                api_error.warning(f'Device "{self.__dev_name}" not found in database',
                                  'GeecsDevice class, method "list_variables"')
            self.dev_vars = {}

        return self.dev_vars, exp_devs

    def find_var_by_alias(self, alias: VarAlias = VarAlias('')) -> str:
        if not self.dev_vars:
            self.list_variables()

        if not self.dev_vars:
            return ''

        var_name = ''
        for attributes in self.dev_vars.values():
            if attributes['alias'] == alias:
                var_name = attributes['variablename']
                break

        if not var_name and alias in self.dev_vars:
            var_name = str(alias)

        return var_name

    def find_alias_by_var(self, var: str = '') -> VarAlias:
        if not self.dev_vars:
            self.list_variables()

        if not self.dev_vars:
            return VarAlias('')

        var_alias = ''
        for attributes in self.dev_vars.values():
            if attributes['variablename'] == var:
                var_alias = attributes['alias']
                break

        if not var_alias and var in self.dev_vars:
            var_alias = var

        return VarAlias(var_alias)

    def build_var_dicts(self):
        self.var_names_by_index: dict[int, tuple[str, VarAlias]] = {
            index: (self.find_var_by_alias(var_alias), var_alias)
            for index, var_alias in enumerate(self.var_spans.keys())}

        self.var_aliases_by_name: dict[str, tuple[VarAlias, int]] = {
            self.find_var_by_alias(var_alias): (var_alias, index)
            for index, var_alias in enumerate(self.var_spans.keys())}

    def _state_value(self, var_name: str) -> Any:
        var_alias: VarAlias

        if var_name in self.generic_vars:
            var_alias = VarAlias(var_name)

        elif var_name in self.var_aliases_by_name:
            var_alias = self.var_aliases_by_name[var_name][0]

        else:
            var_alias = self.find_alias_by_var(var_name)

        if var_alias in self.state:
            return self.state[var_alias]
        else:
            return None

    # Operations
    # -----------------------------------------------------------------------------------------------------------
    def set(self, variable: str, value, exec_timeout: Optional[float] = 120.0, attempts_max: int = 5, sync=True) -> Any:
        """
        if sync=True (default), returns state value after execution
        else, returns AsyncResult (tuple of "queued", "cmd_label", "async_thread")
        """
        ret = self._execute(variable, value, exec_timeout, attempts_max, sync)

        if ret and sync:
            return self._state_value(variable)
        else:
            return ret

    def get(self, variable: str, exec_timeout: Optional[float] = 5.0, attempts_max: int = 5, sync=True) -> Any:
        """
        if sync=True (default), returns updated state value
        else, returns AsyncResult (tuple of "queued", "cmd_label", "async_thread")
        """
        ret = self._execute(variable, None, exec_timeout, attempts_max, sync)

        if ret and sync:
            return self._state_value(variable)
        else:
            return ret

    def _execute(self, variable: str, value, exec_timeout: Optional[float] = 10.0,
                 attempts_max: int = 5, sync=True) -> Optional[AsyncResult]:
        if api_error.is_error:
            return None

        if isinstance(value, (int, str)):
            cmd_str = f'set{variable}>>{value}'
            cmd_label = f'set({variable}, {value})'
        elif isinstance(value, float):
            cmd_str = f'set{variable}>>{value:.6f}'
            cmd_label = f'set({variable}, {value:.6f})'
        elif isinstance(value, bool):
            cmd_str = f'set{variable}>>{int(value)}'
            cmd_label = f'set({variable}, {value})'
        else:
            cmd_str = f'get{variable}>>'
            cmd_label = f'get({variable})'

        if not self.is_valid():
            api_error.warning(f'Failed to execute "{cmd_label}"',
                              f'GeecsDevice "{self.__dev_name}" not connected')
            return None

        stamp = re.sub(r'[\s.:]', '-', dtime.now().__str__())
        cmd_label += f' @ {stamp}'

        queued: bool = False
        async_thread: ThreadInfo = (None, None)

        self._cleanup_threads()

        if sync:
            self.wait_for_all_cmds(timeout=120.)

            with GeecsDevice.threads_lock:
                self._process_command(cmd_str, cmd_label, thread_info=(None, None), attempts_max=attempts_max)
                self.dev_udp.cmd_checker.wait_for_exe(cmd_tag=cmd_label, timeout=exec_timeout, sync=sync)

        elif exec_timeout > 0:
            with GeecsDevice.threads_lock:
                # create listening thread (only)
                async_thread: ThreadInfo = \
                    self.dev_udp.cmd_checker.wait_for_exe(cmd_tag=cmd_label, timeout=exec_timeout, sync=sync)

                # if nothing running and no commands in queue
                if (not self.own_threads) and self.queue_cmds.empty():
                    self._process_command(cmd_str, cmd_label, thread_info=async_thread, attempts_max=attempts_max)
                else:
                    self.queue_cmds.put((cmd_str, cmd_label, async_thread, attempts_max))
                    queued = True

        return queued, cmd_label, async_thread

    def scan(self, var_alias: VarAlias, start_value: float, end_value: float, step_size: float,
             var_span: Optional[tuple[Optional[float], Optional[float]]] = None, shots_per_step: int = 10,
             use_alias: bool = True, timeout: float = 60.) -> tuple[Path, int, bool, bool]:
        var_values = self._scan_values(var_alias, start_value, end_value, step_size, var_span)

        if use_alias:
            # noinspection PyTypeChecker
            GeecsDevice.write_1D_scan_file(self.get_name(), var_alias, var_values, shots_per_step)
        else:
            var_name = self.find_var_by_alias(var_alias)
            # noinspection PyTypeChecker
            GeecsDevice.write_1D_scan_file(self.get_name(), var_name, var_values, shots_per_step)

        comment = f'{var_alias} scan'
        return GeecsDevice.file_scan(self, comment, timeout)

    @staticmethod
    def no_scan(monitoring_device: Optional[GeecsDevice] = None, comment: str = 'no scan',
                shots: int = 10, timeout: float = 300.) -> tuple[Path, int, bool, bool]:
        cmd = f'ScanStart>>{comment}>>{shots}'
        return GeecsDevice._process_scan(cmd, comment, monitoring_device, timeout)

    @staticmethod
    def file_scan(monitoring_device: Optional[GeecsDevice] = None, comment: str = 'no scan', timeout: float = 300.)\
            -> tuple[Path, int, bool, bool]:
        cmd = f'FileScan>>{GeecsDevice.scan_file_path}'
        return GeecsDevice._process_scan(cmd, comment, monitoring_device, timeout)

    @staticmethod
    def _process_scan(cmd: str, comment: str = 'no scan', monitoring_device: Optional[GeecsDevice] = None,
                      timeout: float = 300.) -> tuple[Path, int, bool, bool]:
        if monitoring_device is None:
            dev = GeecsDevice('tmp', virtual=True)
            dev.dev_udp = UdpHandler(owner=dev)
        else:
            dev = monitoring_device

        next_folder, next_scan = dev.next_scan_folder()
        accepted = timed_out = False

        txt_file_name = f'ScanData{os.path.basename(next_folder)}.txt'
        txt_file_path: Path = next_folder / txt_file_name

        try:
            accepted = dev.dev_udp.send_scan_cmd(cmd)

            # format ini file
            ini_file_name = f'ScanInfo{next_folder.name}.ini'
            # txt_file_name = f'ScanData{os.path.basename(next_folder)}.txt'

            ini_file_path = next_folder / ini_file_name
            # txt_file_path = os.path.join(next_folder, txt_file_name)
            ini_found = False

            if accepted:
                # wait for .ini file creation
                t0 = time.monotonic()
                while True:
                    timed_out = (time.monotonic() - t0 > 10.)
                    if timed_out:
                        break

                    if ini_file_path.is_file():
                        ini_found = True
                        break

                if ini_found:
                    try:
                        # make a copy and write content to it
                        shutil.copy2(ini_file_path, Path(str(ini_file_path) + '~'))

                        destination = open(ini_file_path, 'w')
                        source = open(Path(str(ini_file_path) + '~'), 'r')

                        info_line_found = False
                        par_line_found = False
                        for line in source:
                            if line.startswith('ScanStartInfo'):
                                destination.write(f'ScanStartInfo = "{comment}"\n')
                                info_line_found = True
                            elif line.startswith('Scan Parameter'):
                                destination.write('Scan Parameter = "Shotnumber"\n')
                                par_line_found = True
                            else:
                                destination.write(line)

                        source.close()
                        destination.close()

                        #  add lines if missing
                        if not info_line_found or not par_line_found:
                            destination = open(ini_file_path, 'a')
                            if not info_line_found:
                                destination.write(f'ScanStartInfo = "{comment}"\n')
                            if not par_line_found:
                                destination.write('Scan Parameter = "Shotnumber"\n')
                            destination.close()

                    except Exception as ex:
                        api_error.error(str(ex), f'Could not update "{ini_file_name}" with scan comment')
                        try:
                            # restore files
                            os.remove(ini_file_path)
                            shutil.move(str(ini_file_path) + '~', ini_file_path)
                        except Exception:
                            pass
                    else:
                        # remove original if successful
                        try:
                            os.remove(str(ini_file_path) + "~")
                        except Exception:
                            pass

            timed_out = GeecsDevice.wait_for_scan_start(next_folder, next_scan, timeout=60.)
            if not timed_out:
                if not dev.is_valid():
                    time.sleep(2.)  # buffer since cannot verify in 'scan' mode

                # wait for 'no scan' status (if valid device) or .txt file to be created = end of scan
                t0 = time.monotonic()
                while True:
                    timed_out = (time.monotonic() - t0 > timeout)
                    if os.path.isfile(txt_file_path) \
                            or (dev.is_valid() and dev.state[VarAlias('device status')] == 'no scan') \
                            or timed_out:
                        break
                    time.sleep(1.)
        except Exception:
            pass
        finally:
            if monitoring_device is None:
                try:
                    dev.close()
                except Exception:
                    pass

        return next_folder, next_scan, accepted, timed_out

    def get_status(self, exec_timeout: float = 2.0, sync=True) -> Optional[Union[float, AsyncResult]]:
        return self.get('device status', exec_timeout=exec_timeout, sync=sync)

    def interpret_value(self, var_alias: VarAlias, val_string: str) -> Any:
        return float(val_string)

    def interpret_generic_variables(self, var: str, val: str):
        # ['device status', 'device error', 'device preset']
        self.state[VarAlias(var)] = val

    def dequeue_command(self):
        self._cleanup_threads()

        with GeecsDevice.threads_lock:
            # if nothing running and commands in queue
            if (not self.own_threads) and (not self.queue_cmds.empty()):
                try:
                    cmd_str, cmd_label, async_thread, attempts_max = self.queue_cmds.get_nowait()
                    self._process_command(cmd_str, cmd_label, thread_info=async_thread, attempts_max=attempts_max)
                except queue.Empty:
                    pass

    def _process_command(self, cmd_str: str, cmd_label: str,
                         thread_info: ThreadInfo = (None, None), attempts_max: int = 5):
        accepted = False
        try:
            for _ in range(attempts_max):
                sent = self.dev_udp.send_cmd(ipv4=(self.dev_ip, self.dev_port), msg=cmd_str)
                if sent:
                    accepted = self.dev_udp.ack_cmd(timeout=5.0)
                else:
                    time.sleep(0.1)
                    continue

                if accepted or api_error.is_error:
                    break
                else:
                    time.sleep(0.1)

        except Exception as ex:
            api_error.error(str(ex), f'GeecsDevice "{self.__dev_name}", method "{cmd_label}"')

        if accepted and (thread_info[0] is not None):
            thread_info[0].start()
            self.own_threads.append(thread_info)
            GeecsDevice.all_threads.append(thread_info)

    def handle_response(self, net_msg: mh.NetworkMessage) -> tuple[str, str, str, bool]:
        try:
            dev_name, cmd_received, dev_val, err_status = GeecsDevice._response_parser(net_msg.msg)

            # Queue & notify
            if self.notify_on_udp:
                self.queue_udp_msgs.put((dev_name, cmd_received, dev_val, err_status))
                self.notifier_udp_msgs.notify_all()

            # Error handling
            if net_msg.err.is_error or net_msg.err.is_warning:
                print(net_msg.err)

            if dev_name != self.__dev_name:
                warn = ErrorAPI('Mismatch in device name', f'Class {self.__class_name}, method "handle_response"')
                print(warn)

            # Update dictionaries
            if dev_name == self.get_name() and not err_status and cmd_received[:3] == 'get':
                if cmd_received[3:] in self.generic_vars:
                    self.interpret_generic_variables(cmd_received[3:], dev_val)

                elif cmd_received[3:] in self.var_aliases_by_name:
                    var_alias = self.var_aliases_by_name[cmd_received[3:]][0]
                    dev_val = self.interpret_value(var_alias, dev_val)
                    self.state[var_alias] = dev_val

                else:
                    var_alias = self.find_alias_by_var(cmd_received[3:])
                    try:
                        dev_val = float(dev_val)
                    except Exception:
                        pass
                    self.state[var_alias] = dev_val

            if dev_name == self.get_name() and not err_status and cmd_received[:3] == 'set':
                if cmd_received[3:] in self.var_aliases_by_name:
                    var_alias = self.var_aliases_by_name[cmd_received[3:]][0]
                    dev_val = self.interpret_value(var_alias, dev_val)
                    self.setpoints[var_alias] = dev_val
                    self.state[var_alias] = dev_val
                else:
                    var_alias = self.find_alias_by_var(cmd_received[3:])
                    try:
                        dev_val = float(dev_val)
                    except Exception:
                        pass
                    try:
                        dev_val = np.safe_eval(dev_val)
                    except Exception:
                        pass
                    self.setpoints[var_alias] = dev_val
                    self.state[var_alias] = dev_val

            return dev_name, cmd_received, dev_val, err_status

        except Exception as ex:
            err = ErrorAPI(str(ex), f'Class {self.__class_name}, method "{inspect.stack()[0][3]}"')
            print(err)
            return '', '', '', True

    def handle_subscription(self, net_msg: mh.NetworkMessage) -> tuple[str, int, dict[str, str]]:
        try:
            dev_name, shot_nb, dict_vals = GeecsDevice._subscription_parser(net_msg.msg)

            # Queue & notify
            if self.notify_on_tcp:
                self.queue_tcp_msgs.put((dev_name, shot_nb, dict_vals))
                self.notifier_tcp_msgs.notify_all()

            # Error handling
            if net_msg.err.is_error or net_msg.err.is_warning:
                print(net_msg.err)

            # Update dictionaries
            if dev_name == self.get_name() and dict_vals:
                for var, val in dict_vals.items():
                    if var in self.generic_vars:
                        self.interpret_generic_variables(var, val)

                    if var in self.var_aliases_by_name:
                        var_alias: VarAlias = self.var_aliases_by_name[var][0]
                        self.state[var_alias] = self.interpret_value(var_alias, val)
                    else:
                        var_alias = self.find_alias_by_var(var)
                        try:
                            val = float(val)
                        except Exception:
                            pass
                        self.state[var_alias] = val

            return dev_name, shot_nb, dict_vals

        except Exception as ex:
            err = ErrorAPI(str(ex), f'Class {self.__class_name}, method "{inspect.stack()[0][3]}"')
            print(err)
            return '', 0, {}

    @staticmethod
    def _subscription_parser(msg: str = '') -> tuple[str, int, dict[str, str]]:
        """ General parser to be called when messages are received. """

        # msg = 'U_S2V>>0>>Current nval, -0.000080 nvar, Voltage nval,0.002420 nvar,'
        pattern = re.compile(r'[^,]+nval,[^,]+nvar')
        blocks = msg.split('>>')
        dev_name = blocks[0]
        shot_nb = int(blocks[1])
        vars_vals = pattern.findall(blocks[-1])

        dict_vals = {vars_vals[i].split(',')[0][:-5].strip(): vars_vals[i].split(',')[1][:-5]
                     for i in range(len(vars_vals))}

        return dev_name, shot_nb, dict_vals

    @staticmethod
    def _response_parser(msg: str = '') -> tuple[str, str, str, bool]:
        """ General parser to be called when messages are received. """

        # Examples:
        # 'U_ESP_JetXYZ>>getJet_X (mm)>>>>error,Error occurred during access CVT -  "jet_x (mm)" variable not found'
        # 'U_ESP_JetXYZ>>getPosition.Axis 1>>7.600390>>no error,'

        dev_name, cmd_received, dev_val, err_msg = msg.split('>>')
        err_status, err_msg = err_msg.split(',')
        err_status = (err_status == 'error')
        if err_status:
            api_error.error(err_msg, f'Failed to execute command "{cmd_received}", error originated in control system')

        return dev_name, cmd_received, dev_val, err_status

    def coerce_float(self, var_alias: VarAlias, method: str, value: float,
                     var_span: Optional[tuple[Optional[float], Optional[float]]] = None) -> float:
        try:
            if var_span is None:
                if var_alias in self.var_spans:
                    var_span = self.var_spans[var_alias]
                else:
                    var_span = (None, None)

            if var_span[0] and value < var_span[0]:
                api_error.warning(f'{var_alias} value coerced from {value} to {var_span[0]}',
                                  f'Class {self.__class_name}, method "{method}"')
                value = var_span[0]
            if var_span[1] and value > var_span[1]:
                api_error.warning(f'{var_alias} value coerced from {value} to {var_span[1]}',
                                  f'Class {self.__class_name}, method "{method}"')
                value = var_span[1]
        except Exception:
            api_error.error('Failed to coerce value')

        return value

    def _scan_values(self, var_alias: VarAlias, start_value: float, end_value: float, step_size: float,
                     var_span:  Optional[tuple[Optional[float], Optional[float]]] = None) -> npt.ArrayLike:
        start_value = self.coerce_float(var_alias, inspect.stack()[0][3], start_value, var_span)
        end_value = self.coerce_float(var_alias, inspect.stack()[0][3], end_value, var_span)

        if end_value < start_value:
            step_size = -abs(step_size)
        else:
            step_size = abs(step_size)

        return np.arange(start_value, end_value + step_size, step_size)

    @staticmethod
    def write_1D_scan_file(devices: Union[list[str], str], variables: Union[list[str], str],
                           values_by_row: Union[np.ndarray, list], shots_per_step: int = 10):
        scan_number = 1

        with open(GeecsDevice.scan_file_path, 'w+') as f:
            f.write(f'[Scan{scan_number}]\n')
            if isinstance(devices, list):
                f.write('Device = "' + ','.join(devices) + '"\n')
            else:
                f.write(f'Device = "{devices}"\n')

            if isinstance(variables, list):
                f.write('Variable = "' + ','.join(variables) + '"\n')
            else:
                f.write(f'Variable = "{variables}"\n')
            f.write('Values:#shots = "')

            if isinstance(values_by_row, list):
                values_by_row = np.array(values_by_row)

            if values_by_row.ndim > 1:
                for col in range(values_by_row.shape[1]):
                    f.write(f'({str(list(values_by_row[:, col]))[1:-1]}):{shots_per_step}|')
            else:
                for col in range(values_by_row.size):
                    f.write(f'({values_by_row[col]}):{shots_per_step}|')
            f.write('"')

    def today_data_folder(self) -> Path:
        stamp = dtime.now()
        date_folders = os.path.join(stamp.strftime('Y%Y'), stamp.strftime('%m-%B')[:6], stamp.strftime('%y_%m%d'))

        return self.data_root_path / date_folders

    def last_scan_number(self) -> int:
        data_folder: Path = self.today_data_folder()
        # if not os.path.isdir(os.path.join(data_folder, 'scans'))\
        #         or not next(os.walk(os.path.join(data_folder, 'scans')))[1]:  # no 'scans' or no 'ScanXXX' folders
        #     return -1

        # no 'scans' or no 'ScanXXX' folders
        if not (data_folder/'scans').is_dir() or not next(os.walk(data_folder/'scans'))[1]:
            return -1

        # scan_folders: list[SysPath] = next(os.walk(os.path.join(data_folder, 'scans')))[1]
        scan_folders: list[Path] = next(os.walk(data_folder/'scans'))[1]
        # noinspection PyTypeChecker
        scan_folders = [x for x in scan_folders if re.match(r'^Scan(?P<scan>\d{3})$', x)]
        if scan_folders:
            return int(scan_folders[-1][-3:])
        else:
            return -1

    def next_scan_folder(self) -> tuple[Path, int]:
        last_scan: int = self.last_scan_number()
        data_folder: Path = self.today_data_folder()

        if last_scan > 0:
            next_folder = f'Scan{last_scan + 1:03d}'
        else:
            next_folder = 'Scan001'

        return data_folder/'scans'/next_folder, last_scan + 1

    @staticmethod
    def wait_for_scan_start(next_folder: Path, next_scan: int, timeout: float = 60.) -> bool:
        t0 = time.monotonic()
        while True:
            timed_out = (time.monotonic() - t0 > timeout)
            if timed_out:
                break

            tdms_filepath = next_folder/f'Scan{next_scan:03d}.tdms'
            # if os.path.isdir(next_folder) and os.path.isfile(tdms_filepath) \
            #         and (not self.is_valid() or
            #              (('device status' in self.state) and (self.state[VarAlias('device status')] == 'scan'))):
            #     break
            if next_folder.is_dir() and tdms_filepath.is_file():
                break
            time.sleep(0.1)

        return timed_out

    # Synchronization
    # -----------------------------------------------------------------------------------------------------------
    @staticmethod
    def cleanup_all_threads():
        with GeecsDevice.threads_lock:
            for it in range(len(GeecsDevice.all_threads)):
                if not GeecsDevice.all_threads[-1 - it][0].is_alive():
                    GeecsDevice.all_threads.pop(-1 - it)

    def _cleanup_threads(self):
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
    def wait_for_all_devices(timeout: Optional[float] = None):
        GeecsDevice.cleanup_all_threads()
        synced = True

        with GeecsDevice.threads_lock:
            for thread in GeecsDevice.all_threads:
                thread[0].join(timeout)
                synced &= thread[0].is_alive()

        return synced

    @staticmethod
    def stop_waiting_for_all_devices():
        GeecsDevice.cleanup_all_threads()

        with GeecsDevice.threads_lock:
            for thread in GeecsDevice.all_threads:
                thread[1].set()

    def wait_for_all_cmds(self, timeout: Optional[float] = None) -> bool:
        self._cleanup_threads()
        any_alive = False

        with GeecsDevice.threads_lock:
            for thread in self.own_threads:
                thread[0].join(timeout)
                any_alive |= thread[0].is_alive()

        return not any_alive

    def stop_waiting_for_all_cmds(self):
        self._cleanup_threads()

        with GeecsDevice.threads_lock:
            for thread in self.own_threads:
                thread[1].set()

    def wait_for_cmd(self, thread: Thread, timeout: Optional[float] = None):
        with GeecsDevice.threads_lock:
            if self.is_valid() and thread.is_alive():
                thread.join(timeout)

            alive = thread.is_alive()

        return not alive

    def stop_waiting_for_cmd(self, thread: Thread, stop: Event):
        if self.is_valid() and thread.is_alive():
            stop.set()


if __name__ == '__main__':
    GeecsDevice.exp_info = GeecsDatabase.collect_exp_info('Undulator')

    # example for 1d scan (no object instantiated)
    GeecsDevice.write_1D_scan_file('U_S3H', 'Current', np.linspace(-1., 1., 5), shots_per_step=20)
    _next_folder, _next_scan, _accepted, _timed_out = \
        GeecsDevice.file_scan(comment='U_S3H current scan (GeecsDevice demo)', timeout=300.)

    print(f'Scan folder (#{_next_scan}): {_next_folder}')
    print(f'Scan{"" if _accepted else " not"} accepted{"; Scan timed out!" if _timed_out else ""}')

    # example for 1d scan (using existing device object)
    s3h = GeecsDevice('U_S3H')

    GeecsDevice.write_1D_scan_file(s3h.get_name(), 'Current', np.linspace(-1., 1., 5), shots_per_step=20)
    _next_folder, _next_scan, _accepted, _timed_out = \
        GeecsDevice.file_scan(comment=f'{s3h.get_name()} current scan (GeecsDevice demo)', timeout=300.)

    print(f'Scan folder (#{_next_scan}): {_next_folder}')
    print(f'Scan{"" if _accepted else " not"} accepted{"; Scan timed out!" if _timed_out else ""}')

    # example for variables subscription (user defined)
    # s3h = GeecsDevice('U_S3H')
    s3h.subscribe_var_values(['Current', 'Voltage'])
    time.sleep(1.)
    print(f'{s3h.get_name()} state:\n\t{s3h.state}')

    # asynchronous example
    # s3h = GeecsDevice('U_S3H')
    s3v = GeecsDevice('U_S3V')

    h_queued, h_cmd_label, (h_thread, h_stop) = s3h.get('Current', sync=False)
    thread_v = s3v.get('Current', sync=False)

    # all-device synchronization example
    devs_synced = GeecsDevice.wait_for_all_devices(timeout=10.)
    if not devs_synced:
        GeecsDevice.stop_waiting_for_all_devices()

    # device-level synchronization example
    dev_h_synced = s3h.wait_for_cmd(thread=h_thread, timeout=1.)
    dev_v_synced = s3h.wait_for_cmd(thread=thread_v[2][0], timeout=1.)
    if not dev_h_synced or not dev_v_synced:
        s3h.stop_waiting_for_all_cmds()
        s3v.stop_waiting_for_cmd(thread=thread_v[2][0], stop=thread_v[2][1])

    # close objects
    s3h.close()
