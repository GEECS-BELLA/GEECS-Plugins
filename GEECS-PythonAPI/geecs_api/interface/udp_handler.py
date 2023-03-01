import socket
import select
from geecs_api.interface.geecs_errors import *


class UdpHandler:
    def __init__(self):
        """ Creates a UDP socket and binds it. """

        self.buffer_size = 1024

        try:
            # initialize socket out
            self.sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            # self.sock_out.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.sock_out.settimeout(15)

            # self-assign port and bind it
            self.sock_out.bind(('', 0))
            self.port_out = self.sock_out.getsockname()[1]
            self.port_in = self.port_out + 1
            self.bounded_out = True
        except WindowsError:
            self.sock_out = None
            self.port_out = self.port_in = 0
            self.bounded_out = False

        if self.bounded_out:
            try:
                # initialize socket in
                self.sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
                self.sock_in.settimeout(30)

                # Bind port
                self.sock_in.bind(('', self.port_in))
                self.bounded_in = True
            except WindowsError:
                self.bounded_in = False

        else:
            self.sock_in = None
            self.bounded_in = False

    def close_sock_out(self):
        """ Closes the socket. """

        self.sock_out.close()

    def close_sock_in(self):
        """ Closes the socket. """

        self.sock_in.close()

    def send(self, tcp=(None, None), msg=''):
        """ Send message. """

        try:
            self.sock_out.sendto(msg.encode('ascii'), tcp)
            return True
        except ErrorAPI('Failed to send UDP message!'):
            return False

    def receive(self, ready_timeout_sec=None):
        """ Listens for messages on port port_in. """

        try:
            ready = select.select([self.sock_in], [], [], ready_timeout_sec)
            if ready[0]:
                geecs_str = self.sock_in.recvfrom(self.buffer_size)
                geecs_ans = (geecs_str[0].decode('ascii')).split(">>")[-1]
                return geecs_ans == 'accepted'
            else:
                return False
        except ErrorAPI('Failed to read UDP message!'):
            return False

    def format(self, cmd=''):
        return cmd  # implement method
