import socket
import select
import struct
import queue
import threading
import geecs_api.interface.message_handling as mh
from geecs_api.interface.geecs_errors import ErrorAPI
from geecs_api.interface.event_handler import EventHandler


class TcpSubscriber:
    def __init__(self, dev_name=''):
        self.device = dev_name

        # initialize publisher
        self.publisher = EventHandler(['TCP Message'])
        self.publisher.register('TCP Message', 'TCP subscriber', mh.async_msg_handler)

        # FIFO queue of messages
        self.queue_msgs = queue.Queue()

        # message notifier
        self.notifier = threading.Condition()

        # initialize socket
        self.unsubscribe_event = threading.Event()
        self.host = ''
        self.port = -1
        self.connected = False

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except Exception:
            self.sock = None

    def __del__(self):
        self.publisher.unregister('TCP Message', 'TCP subscriber')
        mh.flush_queue(self.queue_msgs)
        self.close_sock()

    def connect(self, ipv4: tuple[str, int] = ('', -1), err_in=ErrorAPI()) -> tuple[bool, ErrorAPI]:
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

    def close_sock(self):
        """ Closes the socket. """

        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        finally:
            self.sock = None
            self.host = ''
            self.port = -1
            self.connected = False

    def subscribe(self, var: str, err_in=ErrorAPI()) -> tuple[bool, ErrorAPI]:
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

    def unsubscribe(self):
        self.unsubscribe_event.set()

    def async_listener(self):
        """ Listens for and parses messages. """

        self.sock.settimeout(1.0)

        # read messages
        while True:
            err = ErrorAPI()
            ready = select.select([self.sock], [], [], 0.005)

            if ready[0]:
                try:
                    msg_len: int = struct.unpack('>i', self.sock.recv(4))[0]
                except socket.timeout:
                    continue
                except Exception:
                    err = ErrorAPI('Failed to read TCP header bytes', 'TcpSubscriber class, method "async_listener"')
                    continue

                this_msg = ''
                received = False
                while True:
                    try:
                        chunk = self.sock.recv(msg_len)
                        if chunk:
                            this_msg += chunk.decode('ascii')
                    except socket.timeout:
                        pass
                    except Exception:
                        err = ErrorAPI('Failed to read TCP message bytes',
                                       'TcpSubscriber class, method "async_listener"')
                        pass

                    received = len(this_msg) == msg_len
                    if received:
                        break
                    if self.unsubscribe_event.wait(0.):
                        return

                if received:
                    this_msg = this_msg[:line_breaks[0][0]]  # without the end character(s)

            if self.unsubscribe_event.wait(0.):
                return
