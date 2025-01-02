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
        self.sock = None
        self.unsubscribe_event = Event()
        self.host = ''
        self.port = -1
        self.connected = False

        # Optional: store a callback function
        self.message_callback = None
        self.api_shotnumber = 0

    def set_message_callback(self, callback):
        """ Set the callback function that will be called when a new message is received. """
        self.message_callback = callback

    def close(self):
        try:
            self.unsubscribe()
            self.close_sock()
        except Exception:
            pass

    def connect(self) -> bool:
        """ Connects to "host/IP" on port "port". """

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.owner.dev_ip, self.owner.dev_port))
            self.host = self.owner.dev_ip
            self.port = self.owner.dev_port
            self.connected = True
        
        except (TimeoutError, InterruptedError) as e:
            api_error.error(
                f'Error while connecting TCP client ({self.owner.get_name()}): {e}',
                'TcpSubscriber class, method "connect"'
            )
            self.sock = None
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
        if self.is_connected():
            self.close()

        subscribed = False
        self.connect()

        if self.connected:
            try:
                subscription_str = bytes(f'Wait>>{cmd}', 'ascii')
                subscription_len = len(subscription_str)
                size_pack = struct.pack('>i', subscription_len)
                self.sock.sendall(size_pack + subscription_str)

                self.unsubscribe_event.clear()
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
        """
        Listens for TCP messages asynchronously.
        Notifies registered subscribers using the EventHandler on each message received.
        """
        self.sock.settimeout(0.5)  # Set socket timeout to 0.5 seconds

        while True:
            try:
                # Wait for the socket to be ready (using select)
                ready = select.select([self.sock], [], [], 0.05)

                if ready[0]:  # If the socket is ready for reading
                    # Read the length of the incoming message (first 4 bytes)
                    msg_len = struct.unpack('>i', self.sock.recv(4))[0]

                    if msg_len > 0:  # If the message length is valid
                        this_msg = ''

                        # Read the actual message data
                        while len(this_msg) < msg_len:
                            chunk = self.sock.recv(msg_len - len(this_msg))
                            if chunk:
                                this_msg += chunk.decode('ascii')

                        # Notify event handler on receiving a new message (general update)
                        if self.message_callback:  # Invoke the callback if it is set
                            self.message_callback(this_msg)

                        # Handle the message if subscribed
                        if self.subscribed:
                            stamp = dtime.now().__str__()

                            # Properly initialize an empty error object
                            err = ErrorAPI()  # Initialize the error object properly

                            # Create a NetworkMessage object
                            net_msg = mh.NetworkMessage(tag=self.owner.get_name(),
                                                        stamp=stamp, msg=this_msg, err=err)

                            # Call handle_subscription with the proper NetworkMessage object
                            try:
                                self.owner.handle_subscription(net_msg)
                            except Exception as e:
                                api_error.error('Failed to handle TCP subscription',
                                                'TcpSubscriber class, method "async_listener"')
                                print(f"Exception in handle_subscription: {e}")

                # Check for unsubscribe event (to stop the listener)
                if self.unsubscribe_event.wait(0.):
                    self.unsubscribe_event.clear()
                    return  # Exit the listener loop when unsubscribed

            except socket.timeout:
                # Timeout simply continues the loop to check again
                continue

            except Exception as ex:
                api_error.error(str(ex), 'TcpSubscriber class, method "async_listener"')
                return