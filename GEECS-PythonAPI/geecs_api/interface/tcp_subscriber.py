import socket
import select
import struct
import queue
import threading
from datetime import datetime as dtime
import geecs_api.interface.message_handling as mh
from geecs_api.interface.geecs_errors import ErrorAPI, api_error
from geecs_api.interface.event_handler import EventHandler


class TcpSubscriber:
    def __init__(self, dev_name: str = '', reg_default_handler: bool = False):
        self.device = dev_name
        self.event_name = 'TCP Message'

        # initialize publisher
        self.publisher = EventHandler([self.event_name])
        if reg_default_handler:
            self.publisher.register(self.event_name, 'TCP subscriber', mh.async_msg_handler)

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
        try:
            self.publisher.unregister(self.event_name, 'TCP subscriber')
            mh.flush_queue(self.queue_msgs)
            self.close_sock()
        except Exception:
            pass

    def connect(self, ipv4: tuple[str, int] = ('', -1)) -> bool:
        """ Connects to "host/IP" on port "port". """

        try:
            self.sock.connect(ipv4)
            self.host = ipv4[0]
            self.port = ipv4[1]
            self.connected = True

        except Exception:
            api_error.error('Failed to connect TCP client', 'TcpSubscriber class, method "connect"')
            self.host = ''
            self.port = -1
            self.connected = False

        return self.connected

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

    def register_handler(self, subscriber: str = '', callback=None) -> bool:
        if callback is not None and subscriber.strip():
            self.publisher.register(self.event_name, subscriber, callback)
            return True
        else:
            return False

    def unregister_handler(self, subscriber: str = '') -> bool:
        if subscriber.strip():
            self.publisher.unregister(self.event_name, subscriber)
            return True
        else:
            return False

    def subscribe(self, var: str) -> bool:
        """ Subscribe to all variables listed in comma-separated string (e.g. 'varA,varB') """
        subscribed = False

        if self.connected:
            try:
                subscription_str = bytes(f'Wait>>{var}', 'ascii')
                subscription_len = len(subscription_str)
                size_pack = struct.pack('>i', subscription_len)
                self.sock.sendall(size_pack + subscription_str)

                listen_thread = threading.Thread(target=self.async_listener)
                listen_thread.start()
                subscribed = True

            except Exception:
                api_error.error(f'Failed to subscribe to variable "{var}"', 'TcpSubscriber class, method "subscribe"')
        else:
            api_error.warning('Cannot subscribe, not connected', 'TcpSubscriber class, method "subscribe"')

        return subscribed

    def unsubscribe(self):
        self.unsubscribe_event.set()

    def async_listener(self):
        """ Listens for and parses messages. """

        self.sock.settimeout(0.5)

        # read messages
        err = ErrorAPI()  # no error object
        while True:
            err.clear()
            try:
                ready = select.select([self.sock], [], [], 0.005)
            except Exception:
                break

            if ready[0]:
                try:
                    msg_len: int = struct.unpack('>i', self.sock.recv(4))[0]
                except socket.timeout:
                    continue
                except Exception:
                    err.error('Failed to read TCP header bytes', 'TcpSubscriber class, method "async_listener"')
                    continue

                this_msg = ''
                while True:
                    try:
                        chunk = self.sock.recv(msg_len)
                        if chunk:
                            this_msg += chunk.decode('ascii')
                    except socket.timeout:
                        pass
                    except Exception:
                        err.error('Failed to read TCP message bytes', 'TcpSubscriber class, method "async_listener"')
                        pass

                    received = len(this_msg) == msg_len
                    if received:
                        break
                    if self.unsubscribe_event.wait(0.):
                        self.unsubscribe_event.clear()
                        return

                if received:
                    stamp = dtime.now().__str__()
                    net_msg = mh.NetworkMessage(tag=self.device, stamp=stamp, msg=this_msg, err=err)
                    mh.broadcast_msg(net_msg, self.notifier, self.queue_msgs, self.publisher, self.event_name)
                    api_error.merge(err.error_msg, err.error_src, err.is_warning)

            if self.unsubscribe_event.wait(0.):
                self.unsubscribe_event.clear()
                return
