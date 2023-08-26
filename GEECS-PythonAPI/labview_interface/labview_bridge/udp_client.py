import socket
import select
import time


class UdpClient:
    def __init__(self, host=None, port=44782):
        """ Creates a UDP socket and binds it to "host" on port "port" (by default: 44782). """

        # initialize socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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

    def close(self):
        """ Closes the socket. """

        self.sock.close()

    def receive(self, timeout_sec=None):
        """ Listens for messages on port "self.local_port". """

        # set timeout
        self.sock.settimeout(timeout_sec)

        try:
            ready = select.select([self.sock], [], [], timeout_sec)
            if ready[0]:
                data = self.sock.recvfrom(548)
                return data[0].decode()
            else:
                return False
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    print('creating udp listener')
    udp_client = UdpClient()

    if udp_client.bounded:
        try:
            print('listening on local port ' + str(udp_client.local_port))
            time_0 = time.time()
            while True:
                time_elapsed = time.time() - time_0
                if time_elapsed >= 3:
                    break

                msg = udp_client.receive(timeout_sec=1)
                if isinstance(msg, bool):
                    msg = str(msg)
                elif msg is None:
                    break

                print(str(msg))
                time.sleep(0.25)

        except KeyboardInterrupt:
            pass

    print('closing udp listener')
    udp_client.close()
