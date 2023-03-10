import socket
import struct
import threading
from geecs_api.interface.geecs_errors import *


class TcpSubscriber:
    def __init__(self):
        self.host = ''
        self.port = -1
        self.connected = False

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except Exception:
            self.sock = None

    def connect(self, ipv4=('', -1), err_in=ErrorAPI()):
        """ Connects to "host/IP" on port "port". """

        err = ErrorAPI()
        try:
            self.sock.connect(ipv4)
            self.host = ipv4[0]
            self.port = ipv4[1]
            self.connected = True

        except Exception:
            err = ErrorAPI('Failed to connect TCP client', 'TcpSubscriber class, method "connect"')
            self.host = ''
            self.port = -1
            self.connected = False

        return self.connected, err.merge_with_previous(err_in)

    def is_connected(self) -> bool:  # , timeout_sec=0.1):
        # try:
        #     ready = select.select([self.sock], [], [], timeout_sec)
        #     connected = ready[0]
        # except Exception:
        #     connected = False

        return self.connected

    def close(self):
        """ Closes the socket. """

        try:
            if self.sock:
                self.sock.close()
                self.sock = None
                self.host = ''
                self.port = -1
                self.connected = False
        except Exception:
            pass

    def async_msg_handler(self, message: str):
        try:
            cmd_tag, udp_msg, error = message

            if error.is_error:
                err_str = f'Error:\n\tDescription: {error.error_msg}\n\tSource: {error.error_src}'
            elif error.is_warning:
                err_str = f'Warning:\n\tDescription: {error.error_msg}\n\tSource: {error.error_src}'
            else:
                err_str = ''

            if err_str:
                print(err_str)
            else:
                print(f'Asynchronous UDP response to "{cmd_tag}":\n\t{udp_msg}')

        except Exception as ex:
            err = ErrorAPI(str(ex), 'UdpServer class, method "async_msg_handler"')
            print(err)

    def subscribe(self, var, err_in=ErrorAPI()):
        err = ErrorAPI()
        subscribed = False

        if self.connected:
            try:
                subscription_str = bytes(f'Wait>>{var}', 'ascii')
                subscription_len = len(subscription_str)
                size_pack = struct.pack('>i', subscription_len)
                self.sock.sendall(size_pack + subscription_str)
                subscribed = True

            except Exception:
                err = ErrorAPI(f'Failed to subscribe to variable "{var}"', 'TcpSubscriber class, method "subscribe"')
        else:
            err = ErrorAPI('Cannot subscribe, not connected', 'TcpSubscriber class, method "subscribe"', warning=True)

        return subscribed, err.merge_with_previous(err_in)
