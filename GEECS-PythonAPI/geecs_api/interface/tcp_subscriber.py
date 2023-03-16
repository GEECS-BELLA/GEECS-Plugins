import socket
import select
import struct
from queue import Queue
from threading import Thread, Condition, Event
from typing import Optional
from datetime import datetime as dtime
import geecs_api.devices as gd
import geecs_api.interface.message_handling as mh
from geecs_api.interface.geecs_errors import ErrorAPI, api_error


class TcpSubscriber:
    def __init__(self):
        self.owner: Optional[gd.GeecsDevice] = None

        # FIFO queue of messages
        self.queue_msgs = Queue()

        # message notifier
        self.notifier = Condition()

        # initialize socket
        self.unsubscribe_event = Event()
        self.host = ''
        self.port = -1
        self.connected = False

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except Exception:
            self.sock = None

    def __del__(self):
        try:
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

    def is_connected(self) -> bool:
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

    def register_handler(self, subscriber: gd.GeecsDevice) -> bool:
        if subscriber.is_valid():
            self.owner = subscriber
            return True
        else:
            return False

    def unregister_handler(self):
        self.owner = None

    def subscribe(self, cmd: str) -> bool:
        """ Subscribe to all variables listed in comma-separated string (e.g. 'varA,varB') """
        subscribed = False

        if self.connected:
            try:
                subscription_str = bytes(f'Wait>>{cmd}', 'ascii')
                subscription_len = len(subscription_str)
                size_pack = struct.pack('>i', subscription_len)
                self.sock.sendall(size_pack + subscription_str)

                var_thread = Thread(target=self.async_listener)
                var_thread.start()
                subscribed = True

            except Exception:
                api_error.error(f'Failed to subscribe to variable(s) "{cmd}"',
                                'TcpSubscriber class, method "subscribe"')
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
            ready = False
            msg_len = 0

            try:
                ready = select.select([self.sock], [], [], 0.05)
            except socket.timeout:
                continue
            except Exception as ex:
                api_error.error(str(ex), 'TcpSubscriber class, method "async_listener"')
                return
            finally:
                if self.unsubscribe_event.wait(0.):
                    self.unsubscribe_event.clear()
                    api_error.merge(err.error_msg, err.error_src, err.is_warning)
                    return

            if ready[0]:
                try:
                    msg_len: int = struct.unpack('>i', self.sock.recv(4))[0]
                except socket.timeout:
                    continue
                except Exception:
                    api_error.error('Failed to read TCP header bytes', 'TcpSubscriber class, method "async_listener"')
                    return
                finally:
                    if self.unsubscribe_event.wait(0.):
                        self.unsubscribe_event.clear()
                        api_error.merge(err.error_msg, err.error_src, err.is_warning)
                        return

                if msg_len > 0:
                    this_msg = ''
                    while True:
                        try:
                            chunk = self.sock.recv(msg_len)
                            if chunk:
                                this_msg += chunk.decode('ascii')
                        except socket.timeout:
                            pass
                        except Exception:
                            api_error.error('Failed to read TCP message bytes',
                                            'TcpSubscriber class, method "async_listener"')
                            return

                        received = (len(this_msg) == msg_len)
                        if received:
                            break
                        if self.unsubscribe_event.wait(0.):
                            self.unsubscribe_event.clear()
                            api_error.merge(err.error_msg, err.error_src, err.is_warning)
                            return

                    if received:
                        stamp = dtime.now().__str__()
                        net_msg = mh.NetworkMessage(tag=self.owner.dev_name, stamp=stamp, msg=this_msg, err=err)
                        if self.owner:
                            try:
                                self.owner._handle_subscription(net_msg, self.notifier, self.queue_msgs)
                            except Exception:
                                api_error.error('Failed to handle TCP subscription',
                                                'TcpSubscriber class, method "async_listener"')
                                return

            if self.unsubscribe_event.wait(0.):
                self.unsubscribe_event.clear()
                api_error.merge(err.error_msg, err.error_src, err.is_warning)
                return
