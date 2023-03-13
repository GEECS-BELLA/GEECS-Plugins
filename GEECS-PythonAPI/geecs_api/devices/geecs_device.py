import re
from typing import Optional
from datetime import datetime as dtime
import geecs_api.interface as gi
from geecs_api.interface.geecs_errors import api_error


class GeecsDevice:  # add listing of all vars available in __init__
    def __init__(self, name=''):
        self.dev_name: str = name
        self.dev_tcp: Optional[gi.TcpSubscriber] = None
        self.dev_udp: gi.UdpHandler = gi.UdpHandler()
        self.dev_ip: str = ''
        self.dev_port: int = 0

        if self.dev_name:
            self.dev_ip, self.dev_port = gi.GeecsDatabase.find_device(self.dev_name)
            if self.is_valid():
                print(f'Device "{self.dev_name}" found: {self.dev_ip}, {self.dev_port}')

                try:
                    self.dev_tcp = gi.TcpSubscriber(self.dev_name)
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

    def register_var_listener_handler(self, callback=None):
        return self.dev_tcp.register_handler(self.dev_name, callback=callback)

    def unregister_var_listener_handler(self):
        return self.dev_tcp.unregister_handler(self.dev_name)

    def register_cmd_executed_handler(self, callback=None):
        return self.dev_udp.register_handler(self.dev_name, callback=callback)

    def unregister_cmd_executed_handler(self):
        return self.dev_udp.unregister_handler(self.dev_name)

    def subscribe_var_values(self, variables: Optional[list[str]] = None):
        # if variables is None:
        # list and subscribe to all variables for this device

        subscribed = False

        try:
            subscribed = self.dev_tcp.subscribe(','.join(variables))
        except Exception as ex:
            api_error.error(str(ex), 'Class GeecsDevice, method "subscribe_var_values"')

        return subscribed

    def unsubscribe_var_values(self):
        self.dev_tcp.unsubscribe()

    def _execute(self, variable: str, value: Optional[float] = None, attempts_max: int = 5, sync=False) -> bool:
        if api_error.is_error:
            return False

        if value:
            cmd_str = f'set{variable}>>{value:.6f}'
            cmd_label = f'set({variable}, {value:.6f})'
        else:
            cmd_str = f'get{variable}>>'
            cmd_label = f'get({variable})'

        if not self.is_valid():
            api_error.warning(f'Failed to execute "{cmd_label}"',
                              f'GeecsDevice "{self.dev_name}" not connected')
            return False

        executed = False
        try:
            accepted = False

            for _ in range(attempts_max):
                sent = self.dev_udp.send_cmd(ipv4=(self.dev_ip, self.dev_port), msg=cmd_str)
                if sent:
                    accepted = self.dev_udp.ack_cmd(ready_timeout_sec=1.0)
                if accepted or api_error.is_error:
                    break

            if accepted:
                stamp = re.sub(r'[\s.:]', '-', dtime.now().__str__())
                cmd_label += f'_{stamp}'
                self.dev_udp.cmd_checker.wait_for_exe(cmd_tag=cmd_label, ready_timeout_sec=1.0, sync=sync)
                executed = True

        except Exception as ex:
            api_error.error(str(ex), f'GeecsDevice "{self.dev_name}", method "set({variable}, {value:.6f})"')

        return executed

    def set(self, variable: str, value: float, attempts_max: int = 5, sync=False) -> bool:
        return self._execute(variable, value, attempts_max, sync)

    def get(self, variable: str, attempts_max: int = 5, sync=False) -> bool:
        return self._execute(variable, None, attempts_max, sync)

    @staticmethod
    def var_values_parser(msg: str = '') -> tuple[str, dict[str, float]]:
        """ General parser to be called when messages are received. """

        # msg = 'U_S2V>>0>>Current nval, -0.000080 nvar, Voltage nval,0.002420 nvar,'
        pattern = re.compile(r'[^,]+nval,[^,]+nvar')
        dev_name = msg.split('>>')[0]
        vars_vals = pattern.findall(msg.split('>>')[-1])
        dict_vals = {vars_vals[i].split(',')[0][:-5].strip(): float(vars_vals[i].split(',')[1][:-5])
                     for i in range(len(vars_vals))}

        return dev_name, dict_vals


if __name__ == '__main__':
    api_error.clear()

    my_dev = GeecsDevice('U_ESP_JetXYZ')
    # my_dev.get('')
    print(api_error)
