import re
import time
from queue import Queue
from threading import Thread, Condition, Event
from typing import Optional, Any
from datetime import datetime as dtime
import geecs_api.interface.message_handling as mh
from geecs_api.interface import GeecsDatabase, UdpHandler, TcpSubscriber, ErrorAPI, api_error


class GeecsDevice:
    def __init__(self, name=''):
        self.dev_name: str = name

        self.dev_tcp: Optional[TcpSubscriber] = None
        self.dev_udp: Optional[UdpHandler]
        if self.dev_name:
            self.dev_udp = UdpHandler(owner=self)
        else:
            self.dev_udp = None

        self.dev_ip: str = ''
        self.dev_port: int = 0

        self.dev_vars = {}
        self.var_names = {}
        self.var_aliases = {}

        self.sets = {}
        self.gets = {}

        if self.dev_name:
            self.dev_ip, self.dev_port = GeecsDatabase.find_device(self.dev_name)
            if self.is_valid():
                # print(f'Device "{self.dev_name}" found: {self.dev_ip}, {self.dev_port}')
                try:
                    self.dev_tcp = TcpSubscriber(owner=self)
                    self.connect_var_listener()
                except Exception:
                    api_error.error('Failed creating TCP subscriber', 'GeecsDevice class, method "__init__"')
            else:
                # print(f'Device "{self.dev_name}" not found')
                api_error.warning(f'Device "{self.dev_name}" not found', 'GeecsDevice class, method "__init__"')

    def cleanup(self):
        if self.dev_udp:
            self.dev_udp.cleanup()

        if self.dev_tcp:
            self.dev_tcp.cleanup()

    def is_valid(self):
        return self.dev_name and self.dev_ip and self.dev_port > 0

    def connect_var_listener(self):
        if not self.is_var_listener_connected():
            self.dev_tcp.connect((self.dev_ip, self.dev_port))

            if not self.dev_tcp.is_connected():
                api_error.warning('Failed to connect TCP subscriber', f'GeecsDevice "{self.dev_name}"')

    def is_var_listener_connected(self):
        return self.dev_tcp and self.dev_tcp.is_connected()

    def register_var_listener_handler(self):
        return self.dev_tcp.register_handler()

    def unregister_var_listener_handler(self):
        return self.dev_tcp.unregister_handler()

    def register_cmd_executed_handler(self):
        return self.dev_udp.register_handler()

    def unregister_cmd_executed_handler(self):
        return self.dev_udp.unregister_handler()

    def subscribe_var_values(self, variables: Optional[list[str]] = None) -> bool:
        subscribed = False

        if variables is None:
            variables = [var[0] for var in self.var_names.values()]

        if variables:
            try:
                subscribed = self.dev_tcp.subscribe(','.join(variables))
            except Exception as ex:
                api_error.error(str(ex), 'Class GeecsDevice, method "subscribe_var_values"')

        return subscribed

    def unsubscribe_var_values(self):
        self.dev_tcp.unsubscribe()

    def list_variables(self, exp_vars: Optional[dict[str, dict[str, dict[str, Any]]]] = None,
                       exp_name: str = 'Undulator') \
            -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
        try:
            if exp_vars is None:
                exp_vars = GeecsDatabase.find_experiment_variables(exp_name)

            self.dev_vars = exp_vars[self.dev_name]

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

    def get_var_dicts(self, aliases: list[str]):
        self.var_names: dict[int, tuple[str, str]] = \
            {index: (self.find_var_by_alias(aliases[index]), aliases[index]) for index in range(len(aliases))}

        self.var_aliases: dict[str, tuple[str, int]] = \
            {self.find_var_by_alias(aliases[index]): (aliases[index], index) for index in range(len(aliases))}

    def set(self, variable: str, value: float, exec_timeout: float = 120.0,
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

    def _execute(self, variable: str, value: Optional[float] = None, exec_timeout: float = 10.0,
                 attempts_max: int = 5, sync=False) -> tuple[bool, str, tuple[Optional[Thread], Optional[Event]]]:
        if api_error.is_error:
            return False, '', (None, None)

        if value:
            cmd_str = f'set{variable}>>{value:.6f}'
            cmd_label = f'set({variable}, {value:.6f})'
        else:
            cmd_str = f'get{variable}>>'
            cmd_label = f'get({variable})'

        if not self.is_valid():
            api_error.warning(f'Failed to execute "{cmd_label}"',
                              f'GeecsDevice "{self.dev_name}" not connected')
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
            api_error.error(str(ex), f'GeecsDevice "{self.dev_name}", method "{cmd_label}"')

        return executed, cmd_label, async_thread

    def handle_response(self, net_msg: mh.NetworkMessage,
                        notifier: Optional[Condition] = None,
                        queue_msgs: Optional[Queue] = None) -> tuple[str, str, str, str]:
        try:
            dev_name, cmd_received, dev_val, err_status = GeecsDevice._response_parser(net_msg.msg)

            if net_msg.err.is_error or net_msg.err.is_warning:
                print(net_msg.err)

            if err_status:
                print(api_error)

            if dev_name != self.dev_name:
                warn = ErrorAPI('Mismatch in device name', f'Class {self.__class__}, method "handle_response"')
                print(warn)

            # if not net_msg.stamp:
            #     net_msg.stamp = 'no timestamp'
            #
            # msg_str = f'Command message:\n\tStamp: {net_msg.stamp}\n\tDevice: {dev_name}\n\tCommand: {cmd_received}'
            # if dev_val:
            #     msg_str += f'\n\tValue: {dev_val}'
            #
            # print(msg_str)
            return dev_name, cmd_received, dev_val, err_status

        except Exception as ex:
            err = ErrorAPI(str(ex), 'Class GeecsDevice, method "subscription_handler"')
            print(err)
            return '', '', '', ''

    def handle_subscription(self, net_msg: mh.NetworkMessage,
                            notifier: Optional[Condition] = None,
                            queue_msgs: Optional[Queue] = None) -> tuple[str, int, dict[str, float]]:
        try:
            dev_name, shot_nb, dict_vals = GeecsDevice._subscription_parser(net_msg.msg)

            if net_msg.err.is_error or net_msg.err.is_warning:
                print(net_msg.err)

            # if not net_msg.stamp:
            #     net_msg.stamp = 'no timestamp'
            #
            # msg_str = f'Subscription message:\n\tStamp: {net_msg.stamp}\n\tDevice: {dev_name}\n\tShot: {shot_nb}'
            # for var, val in dict_vals.items():
            #     if var in self.dev_vars:
            #         msg_str += f'\n\t{self.dev_vars[var]["alias"]}: {val}'
            #     else:
            #         msg_str += f'\n\t{var}: {val}'
            # print(msg_str)
            return dev_name, shot_nb, dict_vals

        except Exception as ex:
            err = ErrorAPI(str(ex), 'Class GeecsDevice, method "subscription_handler"')
            print(err)
            return '', 0, {}

    @staticmethod
    def _subscription_parser(msg: str = '') -> tuple[str, int, dict[str, float]]:
        """ General parser to be called when messages are received. """

        # msg = 'U_S2V>>0>>Current nval, -0.000080 nvar, Voltage nval,0.002420 nvar,'
        pattern = re.compile(r'[^,]+nval,[^,]+nvar')
        blocks = msg.split('>>')
        dev_name = blocks[0]
        shot_nb = int(blocks[1])
        vars_vals = pattern.findall(blocks[-1])
        dict_vals = {vars_vals[i].split(',')[0][:-5].strip(): float(vars_vals[i].split(',')[1][:-5])
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


if __name__ == '__main__':
    api_error.clear()

    devs = GeecsDatabase.find_experiment_variables('Undulator')
    # _dev_name = 'U_ESP_JetXYZ'
    _dev_name = 'U_Hexapod'
    dev = GeecsDevice(_dev_name)
    dev.list_variables(devs)
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
