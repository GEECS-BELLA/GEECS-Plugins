from typing import Optional
import geecs_api.interface as gi
from geecs_api.interface.geecs_errors import api_error


class GeecsDevice:
    def __init__(self, name=''):
        self.dev_name: str = name
        self.dev_tcp: Optional[gi.TcpSubscriber] = None
        self.dev_udp: gi.UdpHandler = gi.UdpHandler()
        self.dev_ip: str = ''
        self.dev_port: int = 0

        if self.dev_name:
            self.dev_ip, self.dev_port = gi.GeecsDatabase.find_device(self.dev_name)
            if self.is_valid():
                print(f'Device found: {self.dev_ip}, {self.dev_port}')

                try:
                    self.dev_tcp = gi.TcpSubscriber(self.dev_name)
                    self.connect_var_listener()
                except Exception:
                    api_error.error('Failed creating TCP subscriber', 'GeecsDevice class, method "__init__"')
            else:
                print('Device not found')

    def __del__(self):
        try:
            self.dev_udp.__del__()
            if self.dev_tcp:
                self.dev_tcp.__del__()

        except Exception as ex:
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
        return

    def unregister_var_listener_handler(self):
        return

    def register_cmd_executed_handler(self):
        return

    def unregister_cmd_executed_handler(self):
        return

    def subscribe_var_values(self):
        return

    def unsubscribe_var_values(self):
        return

    def var_values_parser(self, msg: str = ''):
        """ General parser to be called when messages are received. """
        return


if __name__ == '__main__':
    api_error.clear()

    my_dev = GeecsDevice('U_ESP_JetXYZ')
    print(api_error)
