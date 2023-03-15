import re
import time
import socket
from queue import Queue
from threading import Thread, Condition
from typing import Optional, Any
from datetime import datetime as dtime
import geecs_api.interface as gi
import geecs_api.interface.message_handling as mh
from geecs_api.interface.geecs_errors import ErrorAPI, api_error


class GeecsDevice:  # add listing of all vars available in __init__
    def __init__(self, name=''):
        self.dev_name: str = name
        self.dev_tcp: Optional[gi.TcpSubscriber] = None
        self.dev_udp: gi.UdpHandler = gi.UdpHandler()
        self.dev_ip: str = ''
        self.dev_port: int = 0
        self.dev_vars = {}

        self.config = {}
        self.state = {}

        if self.dev_name:
            self.dev_ip, self.dev_port = gi.GeecsDatabase.find_device(self.dev_name)
            if self.is_valid():
                print(f'Device "{self.dev_name}" found: {self.dev_ip}, {self.dev_port}')

                try:
                    self.dev_tcp = gi.TcpSubscriber()
                    self.connect_var_listener()
                except Exception:
                    api_error.error('Failed creating TCP subscriber', 'GeecsDevice class, method "__init__"')
            else:
                print(f'Device "{self.dev_name}" not found')

    def __del__(self):
        try:
            self.dev_udp.__del__()
            if self.dev_tcp:
                self.dev_tcp.__del__()

        except Exception:
            pass

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
        return self.dev_tcp.register_handler(subscriber=self)

    def unregister_var_listener_handler(self):
        return self.dev_tcp.unregister_handler()

    def register_cmd_executed_handler(self):
        return self.dev_udp.register_handler(subscriber=self)

    def unregister_cmd_executed_handler(self):
        return self.dev_udp.unregister_handler()

    def subscribe_var_values(self, variables: Optional[list[str]] = None):
        # if variables is None:
        #     # list and subscribe to all variables for this device
        #     if self.dev_vars:

        subscribed = False
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
                exp_vars = gi.GeecsDatabase.find_experiment_variables(exp_name)

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

    def _execute(self, variable: str, value: Optional[float] = None, attempts_max: int = 5, sync=False) \
            -> tuple[bool, Optional[Thread]]:
        if api_error.is_error:
            return False, None

        if value:
            cmd_str = f'set{variable}>>{value:.6f}'
            cmd_label = f'set({variable}, {value:.6f})'
        else:
            cmd_str = f'get{variable}>>'
            cmd_label = f'get({variable})'

        if not self.is_valid():
            api_error.warning(f'Failed to execute "{cmd_label}"',
                              f'GeecsDevice "{self.dev_name}" not connected')
            return False, None

        executed = False
        async_thread: Optional[Thread] = None

        try:
            accepted = False
            for _ in range(attempts_max):
                sent = self.dev_udp.send_cmd(ipv4=(self.dev_ip, self.dev_port), msg=cmd_str)
                if sent:
                    accepted = self.dev_udp.ack_cmd(ready_timeout_sec=5.0)
                if accepted or api_error.is_error:
                    break

            if accepted:
                stamp = re.sub(r'[\s.:]', '-', dtime.now().__str__())
                cmd_label += f' @ {stamp}'
                async_thread = \
                    self.dev_udp.cmd_checker.wait_for_exe(cmd_tag=cmd_label, ready_timeout_sec=120.0, sync=sync)
                executed = True

        except Exception as ex:
            api_error.error(str(ex), f'GeecsDevice "{self.dev_name}", method "{cmd_label}"')

        return executed, async_thread

    def set(self, variable: str, value: float, attempts_max: int = 5, sync=False) -> bool:
        done, _ = self._execute(variable, value, attempts_max, sync)
        return done

    def get(self, variable: str, attempts_max: int = 5, sync=False) -> bool:
        done, _ = self._execute(variable, None, attempts_max, sync)
        return done

    def wait_for_all_cmds(self, timeout: Optional[float] = None) -> bool:
        return self.dev_udp.cmd_checker.wait_for_all(timeout)

    def wait_for_last_cmd(self, timeout: Optional[float] = None) -> bool:
        return self.dev_udp.cmd_checker.wait_for_last(timeout)

    def handle_subscription(self, net_msg: mh.NetworkMessage,
                            notifier: Optional[Condition] = None,
                            queue_msgs: Optional[Queue] = None):
        try:
            dev_name, shot_nb, dict_vals = GeecsDevice.subscription_parser(net_msg.msg)

            if net_msg.err.is_error or net_msg.err.is_warning:
                print(net_msg.err)

            if not net_msg.stamp:
                net_msg.stamp = 'no timestamp'

            msg_str = f'Subscription message:\n\tStamp: {net_msg.stamp}\n\tDevice: {dev_name}\n\tShot: {shot_nb}'
            for var, val in dict_vals.items():
                if var in self.dev_vars:
                    msg_str += f'\n\t{self.dev_vars[var]["alias"]}: {val}'
                else:
                    msg_str += f'\n\t{var}: {val}'
            print(msg_str)

        except Exception as ex:
            err = ErrorAPI(str(ex), 'Class GeecsDevice, method "subscription_handler"')
            print(err)

    def handle_response(self, net_msg: mh.NetworkMessage,
                        notifier: Optional[Condition] = None,
                        queue_msgs: Optional[Queue] = None):
        try:
            dev_name, cmd_received, dev_val, err_status = GeecsDevice.response_parser(net_msg.msg)

            if net_msg.err.is_error or net_msg.err.is_warning:
                print(net_msg.err)

            if err_status:
                print(api_error)

            if not net_msg.stamp:
                net_msg.stamp = 'no timestamp'

            msg_str = f'Command message:\n\tStamp: {net_msg.stamp}\n\tDevice: {dev_name}\n\tCommand: {cmd_received}'
            if dev_val:
                msg_str += f'\n\tValue: {dev_val}'

            print(msg_str)

        except Exception as ex:
            err = ErrorAPI(str(ex), 'Class GeecsDevice, method "subscription_handler"')
            print(err)

    @staticmethod
    def subscription_parser(msg: str = '') -> tuple[str, int, dict[str, float]]:
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
    def response_parser(msg: str = ''):
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

    @staticmethod  # placeholder, will be moved to system class, 2 levels up
    def send_preset(preset=''):
        MCAST_GRP = '234.5.6.8'
        MCAST_PORT = 58432
        MULTICAST_TTL = 4

        sock = None

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
            sock.sendto(f'preset>>{preset}>>{socket.gethostbyname(socket.gethostname())}'.encode(),
                        (MCAST_GRP, MCAST_PORT))

        except Exception:
            api_error.error(f'Failed to send preset "{preset}"', 'UdpHandler class, method "send_preset"')

        finally:
            try:
                sock.close()
            except Exception:
                pass


if __name__ == '__main__':
    api_error.clear()

    devs = gi.GeecsDatabase.find_experiment_variables('Undulator')
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
    dev.set(var_y, -22.0, sync=False)
    print('main thread not blocked!')

    # dev.subscribe_var_values([var_x, var_y])
    # dev.subscribe_var_values([var_y])
    # time.sleep(1.0)
    # dev.unsubscribe_var_values()
    # time.sleep(1.0)

    # dev.wait_for_all_cmds(120.0)
    is_done = dev.wait_for_last_cmd(120.0)  # add event to kill thread(s) and method "stop_thread"
    print(f'thread terminated: {is_done}')
    print(api_error)
