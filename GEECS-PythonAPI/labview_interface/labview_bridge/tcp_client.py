import socket
import select
import time
import re


class TcpClient:
    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        else:
            self.sock = sock

    def connect(self, host=None, port=55449, timeout_sec=None):
        """ Connects to "host" on port "port" (by default: 55449). """

        if not host:
            host = socket.gethostbyname('localhost')

        time_0 = time.time()
        while True:
            # noinspection PyBroadException
            try:
                self.sock.connect((host, port))
                return True
            except:
                if timeout_sec:
                    time_elapsed = time.time() - time_0
                    if time_elapsed >= timeout_sec:
                        return False

    def is_connected(self, timeout_sec=0.01):
        ready = select.select([self.sock], [], [], timeout_sec)
        return ready[0]

    def close(self):
        """ Closes the socket. """

        self.sock.close()

    def echo(self, msg='', end_char='\r\n', timeout_sec=None, debug=False):
        # send message to server
        if debug:
            print('sending message using "send" method')

        sent = self.send(msg=msg, echo=True, end_char=end_char, timeout_sec=timeout_sec, debug=debug)

        if isinstance(sent, bool):
            if not sent:
                return 0
        elif not sent:
            return 0

        # receive echo from server
        if debug:
            print('fetching response using "receive" method')

        this_response = self.receive(end_char=end_char, timeout_sec=timeout_sec, debug=debug)

        this_echo = isinstance(this_response, str)
        if this_echo:
            this_echo = (this_response == msg)

        if debug:
            if this_echo:
                print('echo matched original message')
            else:
                print('echo failed')

        return this_response, this_echo

    def send_receive(self, msg='', end_char='\r\n', timeout_sec=None, debug=False):
        # send message to server
        if debug:
            print('sending message using "send" method')

        sent = self.send(msg=msg, echo=False, end_char=end_char, timeout_sec=timeout_sec, debug=debug)

        if isinstance(sent, bool):
            if not sent:
                return 0
        elif not sent:
            return 0

        # receive response from server
        if debug:
            print('fetching response using "receive" method')

        this_response = self.receive(end_char=end_char, timeout_sec=timeout_sec, debug=debug)

        if not isinstance(this_response, str):
            this_response = ''

        return this_response

    def send(self, msg='', echo=False, end_char='\r\n', timeout_sec=None, debug=False):
        if echo:
            msg = '=?' + msg

        if not re.findall(end_char, msg):
            msg = msg + end_char

        if debug:
            print('message to send: ' + msg)

        # noinspection PyBroadException
        try:
            self.sock.settimeout(timeout_sec)
            self.sock.sendall(msg.encode())

            if debug:
                print('message sent')

            if echo:
                return msg[2:]
            else:
                return msg

        except:
            return False

    def receive(self, end_char='\r\n', timeout_sec=None, debug=False):
        this_msg = ''

        # set timeout
        self.sock.settimeout(timeout_sec)

        if debug:
            print('waiting for message')

        # read message
        if debug:
            print('receiving message (unknown size)')

        chunk_size = 2**12
        time_0 = time.time()
        time_elapsed = 0

        while True:
            if timeout_sec:
                ready = select.select([self.sock], [], [], timeout_sec - time_elapsed)
            else:
                ready = select.select([self.sock], [], [], timeout_sec)

            if ready[0]:
                chunk = self.sock.recv(chunk_size)
                if not chunk:
                    break

                time_elapsed = time.time() - time_0

                this_msg += chunk.decode()
                line_breaks = [(m.start(), m.end()) for m in re.finditer(end_char, this_msg)]
                if line_breaks:
                    this_msg = this_msg[:line_breaks[0][0]]  # without the end character(s)
                    break
            else:
                break

        if debug:
            print('message received: ' + this_msg)

        return this_msg


if __name__ == "__main__":
    this_prefix = 8
    print('starting client')
    ns_client = TcpClient()

    client_connected = False
    if not ns_client.is_connected():
        client_connected = ns_client.connect(port=1234, timeout_sec=None)

    if client_connected:
        ns_client.echo(msg='Hello!', debug=True)
        print('closing client')
        ns_client.send(msg='close ns', debug=True)
        ns_client.close()
    else:
        print('could not connect to port')
