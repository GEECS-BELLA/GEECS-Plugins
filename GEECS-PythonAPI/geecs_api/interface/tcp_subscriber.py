import socket
import select
import time
import struct
from geecs_api.interface.geecs_errors import *
from geecs_api.interface.event_handler import EventHandler


class TcpSubscriber:
    def __init__(self):
        self.host = ''
        self.port = 0

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except Exception:
            self.sock = None

    def connect(self, host=None, port=None, timeout_sec=None):
        """ Connects to "host/IP" on port "port". """

        if not port or not host:
            return False

        time_0 = time.time()
        while True:
            try:
                self.sock.connect((host, port))
                self.host = host
                self.port = port
                return True
            except ErrorAPI('TCP subscription failed!'):
                self.host = self.port = ''
                pass

            if timeout_sec:
                time_elapsed = time.time() - time_0
                if time_elapsed >= timeout_sec:
                    self.host = self.port = ''
                    return False

    def is_connected(self, timeout_sec=0.01):
        ready = select.select([self.sock], [], [], timeout_sec)
        return ready[0]

    def close(self):
        """ Closes the socket. """

        self.host = self.port = ''
        self.sock.close()

    def subscribe(self, var):
        if not self.is_connected():
            raise ErrorAPI('TCP subscriber not connected!')

        try:
            subscription_str = bytes('Wait>>' + str(var), 'ascii')
            subscription_len = len(subscription_str)
            size_pack = struct.pack('>i', subscription_len)
            self.sock.sendall(size_pack + subscription_str)

        except ErrorAPI('TCP subscriber failed to subscribe!'):
            pass
