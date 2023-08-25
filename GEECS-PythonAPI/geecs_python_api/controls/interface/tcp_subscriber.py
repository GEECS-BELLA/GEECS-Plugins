from __future__ import annotations
import socket
import select
import struct
from threading import Thread, Event
from typing import TYPE_CHECKING
from datetime import datetime as dtime
if TYPE_CHECKING:
    from geecs_python_api.controls.devices import GeecsDevice
import geecs_python_api.controls.interface.message_handling as mh
from geecs_python_api.controls.interface.geecs_errors import ErrorAPI, api_error


class TcpSubscriber:
    def __init__(self, owner: GeecsDevice):
        self.owner: GeecsDevice = owner
        self.subscribed = False

        # initialize socket
        self.unsubscribe_event = Event()
        self.host = ''
        self.port = -1
        self.connected = False

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except Exception:
            self.sock = None

    def close(self):
        try:
            self.unsubscribe()
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
            api_error.error(f'Failed to connect TCP client ({self.owner.get_name()})',
                            'TcpSubscriber class, method "connect"')
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

    def register_handler(self) -> bool:
        self.subscribed = self.owner.is_valid()
        return self.subscribed

    def unregister_handler(self):
        self.subscribed = False

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
                        if self.subscribed:
                            try:
                                net_msg = mh.NetworkMessage(tag=self.owner.get_name(),
                                                            stamp=stamp, msg=this_msg, err=err)
                                self.owner.handle_subscription(net_msg)
                            except Exception:
                                api_error.error('Failed to handle TCP subscription',
                                                'TcpSubscriber class, method "async_listener"')
                                return

            if self.unsubscribe_event.wait(0.):
                self.unsubscribe_event.clear()
                api_error.merge(err.error_msg, err.error_src, err.is_warning)
                return
