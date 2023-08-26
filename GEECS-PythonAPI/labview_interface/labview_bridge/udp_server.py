import socket
import time


class UdpServer:
    def __init__(self, host=None, port=63673):
        """ Creates a UDP socket and binds it to "host" on port "port" (by default: 63673). """

        # initialize socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        if not host:
            host = socket.gethostbyname('localhost')

        # bind to the port
        try:
            self.sock.bind((host, port))
            self.bounded = True
        except WindowsError:
            self.bounded = False

        # ports
        self.local_port = port
        self.remote_port = 44782

    def close(self):
        """ Closes the socket. """

        self.sock.close()

    def broadcast(self, port=44782, msg='', timeout_sec=None):
        """ Broadcast message on port "port" every second until timeout (by default: 44782). """

        self.remote_port = port

        # noinspection PyBroadException
        try:
            time_0 = time.time()
            while True:
                time_elapsed = time.time() - time_0
                if time_elapsed >= timeout_sec:
                    break

                self.sock.sendto(msg.encode(), ('<broadcast>', port))
                time.sleep(1)
        except:  # KeyboardInterrupt:
            return False


if __name__ == "__main__":
    print('starting udp server')
    udp_server = UdpServer()

    if udp_server.bounded:
        tcp_port = 1234
        print('broadcasting on remote port ' + str(udp_server.remote_port))
        udp_server.broadcast(msg='<UDP>' + str(tcp_port), timeout_sec=30)

    print('closing udp server')
    udp_server.close()
