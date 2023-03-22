import re
import inspect
from queue import Queue
from threading import Thread, Condition, Event
from typing import Optional, Any
from datetime import datetime as dtime
import geecs_api.interface.message_handling as mh
from geecs_api.interface import GeecsDatabase, UdpHandler, TcpSubscriber, ErrorAPI, api_error


class GeecsDevice:
    def __init__(self, name: str, exp_vars: Optional[dict[str, dict[str, dict[str, Any]]]], virtual=False):
        self.__dev_name: str = name.strip()  # cannot be changed after initialization
        self.__dev_virtual = virtual or not self.__dev_name
        self.__class_name = re.search(r'\w+\'>$', str(self.__class__))[0][:-2]

        self.dev_tcp: Optional[TcpSubscriber] = None
        self.dev_udp: Optional[UdpHandler]
        if not self.__dev_virtual:
            self.dev_udp = UdpHandler(owner=self)
        else:
            self.dev_udp = None

        self.dev_ip: str = ''
        self.dev_port: int = 0

        self.dev_vars = {}
        self.var_names_by_index = {}
        self.var_aliases_by_name = {}

        self.setpoints = {}
        self.state = {}

        if not self.__dev_virtual:
            self.dev_ip, self.dev_port = GeecsDatabase.find_device(self.__dev_name)
            if self.is_valid():
                # print(f'Device "{self.dev_name}" found: {self.dev_ip}, {self.dev_port}')
                try:
                    self.dev_tcp = TcpSubscriber(owner=self)
                    self.connect_var_listener()
                except Exception:
                    api_error.error('Failed creating TCP subscriber', 'GeecsDevice class, method "__init__"')
            else:
                api_error.warning(f'Device "{self.__dev_name}" not found', 'GeecsDevice class, method "__init__"')

            self.list_variables(exp_vars)

    def get_name(self):
        return self.__dev_name

    def cleanup(self):
        if self.dev_udp:
            self.dev_udp.cleanup()

        if self.dev_tcp:
            self.dev_tcp.cleanup()

    def is_valid(self):
        return not self.__dev_virtual and self.dev_ip and self.dev_port > 0

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
            variables = [var[0] for var in self.var_names_by_index.values()]

        if self.is_valid() and variables:
            try:
                subscribed = self.dev_tcp.subscribe(','.join(variables))
            except Exception as ex:
                api_error.error(str(ex), 'Class GeecsDevice, method "subscribe_var_values"')

        return subscribed

    def unsubscribe_var_values(self):
        if self.is_var_listener_connected():
            self.dev_tcp.unsubscribe()

    def list_variables(self, exp_vars: Optional[dict[str, dict[str, dict[str, Any]]]] = None,
                       exp_name: str = 'Undulator') \
            -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
        try:
            if exp_vars is None:
                exp_vars = GeecsDatabase.find_experiment_variables(exp_name)

            self.dev_vars = exp_vars[self.__dev_name]

        except Exception:
            self.dev_vars = {}

        return self.dev_vars, exp_vars

    def find_var_by_alias(self, alias: str = '') -> str:
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
            var_name = alias

        return var_name

    def get_var_dicts(self, aliases: tuple[str]):
        self.var_names_by_index: dict[int, tuple[str, str]] = \
            {index: (self.find_var_by_alias(aliases[index]), aliases[index]) for index in range(len(aliases))}

        self.var_aliases_by_name: dict[str, tuple[str, int]] = \
            {self.find_var_by_alias(aliases[index]): (aliases[index], index) for index in range(len(aliases))}

    def set(self, variable: str, value, exec_timeout: float = 120.0,
            attempts_max: int = 5, sync=False) -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self._execute(variable, value, exec_timeout, attempts_max, sync)

    def get(self, variable: str, exec_timeout: float = 5.0,
            attempts_max: int = 5, sync=False) -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        return self._execute(variable, None, exec_timeout, attempts_max, sync)

    def wait_for_all_cmds(self, timeout: Optional[float] = None) -> bool:
        return self.dev_udp.cmd_checker.wait_for_all(timeout)

    def wait_for_last_cmd(self, timeout: Optional[float] = None) -> bool:
        return self.dev_udp.cmd_checker.wait_for_last(timeout)

    def wait_for(self, thread: Thread, timeout: Optional[float] = None) -> bool:
        return self.dev_udp.cmd_checker.wait_for(thread, timeout)

    def _execute(self, variable: str, value, exec_timeout: float = 10.0,
                 attempts_max: int = 5, sync=False) -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if api_error.is_error:
            return False, '', (None, None)

        if isinstance(value, float):
            cmd_str = f'set{variable}>>{value:.6f}'
            cmd_label = f'set({variable}, {value:.6f})'
        elif isinstance(value, str):
            cmd_str = f'set{variable}>>{value}'
            cmd_label = f'set({variable}, {value})'
        else:
            cmd_str = f'get{variable}>>'
            cmd_label = f'get({variable})'

        if not self.is_valid():
            api_error.warning(f'Failed to execute "{cmd_label}"',
                              f'GeecsDevice "{self.__dev_name}" not connected')
            return False, '', (None, None)

        executed = False
        async_thread: tuple[Optional[Thread], Optional[Event]] = (None, None)

        try:
            accepted = False
            for _ in range(attempts_max):
                sent = self.dev_udp.send_cmd(ipv4=(self.dev_ip, self.dev_port), msg=cmd_str)
                if sent:
                    accepted = self.dev_udp.ack_cmd(timeout=5.0)
                if accepted or api_error.is_error:
                    break

            if accepted:
                stamp = re.sub(r'[\s.:]', '-', dtime.now().__str__())
                cmd_label += f' @ {stamp}'
                async_thread = \
                    self.dev_udp.cmd_checker.wait_for_exe(cmd_tag=cmd_label, timeout=exec_timeout, sync=sync)
                executed = True

        except Exception as ex:
            api_error.error(str(ex), f'GeecsDevice "{self.__dev_name}", method "{cmd_label}"')

        return executed, cmd_label, async_thread

    def handle_response(self, net_msg: mh.NetworkMessage,
                        notifier: Optional[Condition] = None,
                        queue_msgs: Optional[Queue] = None) -> tuple[str, str, str, str]:
        try:
            dev_name, cmd_received, dev_val, err_status = GeecsDevice._response_parser(net_msg.msg)

            if net_msg.err.is_error or net_msg.err.is_warning:
                print(net_msg.err)

            if dev_name != self.__dev_name:
                warn = ErrorAPI('Mismatch in device name', f'Class {self.__class_name}, method "handle_response"')
                print(warn)

            if dev_name == self.get_name() and cmd_received[:3] == 'get':
                var_alias = self.var_aliases_by_name[cmd_received[3:]][0]
                dev_val = self.interpret_value(var_alias, dev_val)
                self.state[var_alias] = dev_val
                dev_val = f'"{dev_val}"' if isinstance(dev_val, str) else dev_val
                print(f'{self.__class_name} [{self.__dev_name}]: {var_alias} = {dev_val}')

            if dev_name == self.get_name() and cmd_received[:3] == 'set':
                var_alias = self.var_aliases_by_name[cmd_received[3:]][0]
                dev_val = self.interpret_value(var_alias, dev_val)
                self.setpoints[var_alias] = dev_val
                dev_val = f'"{dev_val}"' if isinstance(dev_val, str) else dev_val
                print(f'{self.__class_name} [{self.__dev_name}]: {var_alias} set to {dev_val}')

            return dev_name, cmd_received, dev_val, err_status

        except Exception as ex:
            err = ErrorAPI(str(ex), f'Class {self.__class_name}, method "{inspect.stack()[0][3]}"')
            print(err)
            return '', '', '', ''

    def handle_subscription(self, net_msg: mh.NetworkMessage,
                            notifier: Optional[Condition] = None,
                            queue_msgs: Optional[Queue] = None) -> tuple[str, int, dict[str, str]]:
        try:
            dev_name, shot_nb, dict_vals = GeecsDevice._subscription_parser(net_msg.msg)

            if net_msg.err.is_error or net_msg.err.is_warning:
                print(net_msg.err)

            if dev_name == self.get_name() and dict_vals:
                for var, val in dict_vals.items():
                    if var in self.var_aliases_by_name:
                        var_alias: str = self.var_aliases_by_name[var][0]
                        self.state[var_alias] = self.interpret_value(var_alias, val)

            return dev_name, shot_nb, dict_vals

        except Exception as ex:
            err = ErrorAPI(str(ex), f'Class {self.__class_name}, method "{inspect.stack()[0][3]}"')
            print(err)
            return '', 0, {}

    def interpret_value(self, var_alias: str, val_string: str) -> Any:
        return float(val_string)

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
    def _response_parser(msg: str = '') -> tuple[str, str, str, str]:
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

    def coerce_float(self, var_name: str, method: str, value: float,
                     span: tuple[Optional[float], Optional[float]]) -> float:
        try:
            if span[0] and value < span[0]:
                api_error.warning(f'{var_name} value coerced from {value} to {span[0]}',
                                  f'Class {self.__class_name}, method "{method}"')
                value = span[0]
            if span[1] and value > span[1]:
                api_error.warning(f'{var_name} value coerced from {value} to {span[1]}',
                                  f'Class {self.__class_name}, method "{method}"')
                value = span[1]
        except Exception:
            api_error.error('Failed to coerce value')

        return value


if __name__ == '__main__':
    api_error.clear()

    devs = GeecsDatabase.find_experiment_variables('Undulator')
    # _dev_name = 'U_ESP_JetXYZ'
    _dev_name = 'U_Hexapod'
    dev = GeecsDevice(_dev_name, devs)
    # dev.list_variables(devs)
    dev.register_var_listener_handler()
    dev.register_cmd_executed_handler()

    # var_x = dev.find_var_by_alias('Jet_X (mm)')
    # var_y = dev.find_var_by_alias('Jet_Y (mm)')
    var_y = dev.find_var_by_alias('ypos')

    # dev.get(var_y, sync=False)
    # dev.set(var_x, 7.6, sync=False)
    _, _, exe_thread = dev.set(var_y, -10.0, sync=False)
    print('main thread not blocked!')

    # dev.subscribe_var_values([var_x, var_y])
    # dev.subscribe_var_values([var_y])
    # time.sleep(1.0)
    # dev.unsubscribe_var_values()
    # time.sleep(1.0)

    if exe_thread[0]:
        is_done = dev.wait_for(exe_thread[0], 120.0)
    else:
        is_done = dev.wait_for_all_cmds(120.0)
        # is_done = dev.wait_for_last_cmd(120.0)
    print(f'thread terminated: {is_done}')
    print(api_error)
