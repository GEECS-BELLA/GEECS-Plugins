from __future__ import annotations
import math
import socket
import time
import select
import inspect
from threading import Thread, Event
from datetime import datetime as dtime
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from geecs_api.devices import GeecsDevice
import geecs_api.interface.message_handling as mh
from geecs_api.interface.geecs_errors import api_error
from geecs_api.api_defs import ThreadInfo


class UdpHandler:
    def __init__(self, owner: GeecsDevice):
        """ Creates a UDP socket and binds it. """

        self.buffer_size: int = 1024
        self.sock_cmd = None
        self.bounded_cmd: bool = False
        self.mc_port: int = owner.mc_port

        try:
            # initialize socket out
            self.sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.sock_cmd.settimeout(15.0)

            # self-assign port and bind it
            self.sock_cmd.bind(('', 0))
            self.port_cmd = self.sock_cmd.getsockname()[1]
            self.port_exe = self.port_cmd + 1
            self.bounded_cmd = True

            # create UDP server for commands execution confirmation ("slow response")
            self.cmd_checker = UdpServer(owner=owner, port=self.port_exe)

        except Exception:
            self.port_cmd = self.port_exe = -1

    def cleanup(self):
        try:
            self.close_sock_cmd()
        except Exception:
            pass

    def close_sock_cmd(self):
        """ Closes the socket. """

        try:
            if self.sock_cmd:
                self.sock_cmd.close()
        except Exception:
            pass
        finally:
            self.sock_cmd = None
            self.port_cmd = -1
            self.bounded_cmd = False

    def send_cmd(self, ipv4: tuple[str, int] = ('', -1), msg: str = '') -> bool:
        """ Send message. """

        sent = False
        try:
            self.sock_cmd.sendto(msg.encode('ascii'), ipv4)
            sent = True
        except Exception:
            api_error.error('Failed to send UDP message', f'Class "UdpHandler", method "{inspect.stack()[0][3]}"')

        return sent

    def ack_cmd(self, sock: Optional[socket] = None, timeout: Optional[float] = 5.0) -> bool:
        """ Listen for command acknowledgement. """

        accepted = False
        if sock is None:
            sock = self.sock_cmd

        try:
            ready = select.select([sock], [], [], timeout)
            if ready[0]:
                geecs_str = sock.recv(self.buffer_size)
                geecs_ans = (geecs_str.decode('ascii')).split(">>")[-1]
                accepted = (geecs_ans == 'accepted') or (geecs_ans == 'ok')
            else:
                api_error.warning('Socket not ready to receive',
                                  f'Class "UdpHandler", method "{inspect.stack()[0][3]}"')

        except Exception:
            api_error.error('Failed to read UDP acknowledge message',
                            f'Class "UdpHandler", method "{inspect.stack()[0][3]}"')

        return accepted

    def send_scan_cmd(self, cmd: str) -> bool:
        accepted: bool = False
        sock_mc = None

        try:
            sock_mc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock_mc.settimeout(5.0)
            sock_mc.bind(('localhost', 0))

            sock_mc.sendto(cmd.encode('ascii'), ('localhost', self.mc_port + 2))
            accepted = self.ack_cmd(sock=sock_mc, timeout=5.0)

        except Exception as ex:
            api_error.error(str(ex), f'Class "UdpHandler", method "{inspect.stack()[0][3]}"')

        finally:
            try:
                if sock_mc:
                    sock_mc.close()
            except Exception:
                pass

        return accepted

    def register_handler(self) -> bool:
        self.cmd_checker.subscribed = self.cmd_checker.owner.is_valid()
        return self.cmd_checker.subscribed

    def unregister_handler(self):
        self.cmd_checker.subscribed = False


class UdpServer:
    def __init__(self, owner: GeecsDevice, port: int = -1):
        self.owner: GeecsDevice = owner
        self.subscribed = False

        # initialize socket
        self.buffer_size = 1024
        self.port = port
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.sock.settimeout(5.0)
            self.sock.bind(('', self.port))
            self.bounded = True
        except Exception:
            self.sock = None
            self.port = -1
            self.bounded = False

    def cleanup(self):
        try:
            self.close_sock_exe()
        except Exception:
            pass

    def close_sock_exe(self):
        """ Closes the socket. """

        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        finally:
            self.sock = None
            self.port = -1
            self.bounded = False

    def listen(self, cmd_tag: str, stop_event: Optional[Event] = None, timeout: Optional[float] = 1.0) -> str:
        """ Listens for command-executed messages. """

        # receive message
        geecs_ans = stamp = ''
        t0 = time.monotonic()
        if timeout is None:
            timeout = math.inf
            eff_timeout = 0.2
        else:
            eff_timeout = min(0.2, timeout / 10.0)  # to check on stop_event periodically

        while True:
            try:
                ready = select.select([self.sock], [], [], eff_timeout)
                if ready[0]:
                    geecs_str = self.sock.recvfrom(self.buffer_size)
                    geecs_ans = geecs_str[0].decode('ascii')
                    stamp = dtime.now().__str__()
                    break

            except socket.timeout:
                if (time.monotonic() - t0) >= timeout:
                    api_error.warning(f'Command timed out, tag: "{cmd_tag}"', 'UdpServer class, method "listen"')
                    return ''
                elif stop_event and stop_event.wait(0.):
                    stop_event.clear()
                    return ''
                else:
                    continue

            except Exception:
                api_error.error('Failed to read UDP message', 'UdpServer class, method "listen"')
                return ''

        # queue, notify & publish message
        try:
            net_msg = mh.NetworkMessage(tag=cmd_tag, stamp=stamp, msg=geecs_ans)
            if self.subscribed:
                try:
                    self.owner.handle_response(net_msg)
                except Exception:
                    api_error.error('Failed to handle UDP response', 'UdpServer class, method "listen"')

        except Exception:
            api_error.error('Failed to publish UDP message', 'UdpServer class, method "listen"')

        check_next_thread = Thread(target=self.owner.dequeue_command, args=())
        check_next_thread.start()

        return geecs_ans

    def wait_for_exe(self, cmd_tag: str, timeout: Optional[float] = 5.0, sync: bool = False) -> ThreadInfo:
        """ Waits for a UDP response (typ. command executed), in a separate thread by default. """

        exe_thread: Optional[Thread] = None
        stop_event: Optional[Event] = None

        try:
            if sync:
                self.listen(cmd_tag, timeout=timeout)
            else:
                exe_thread, stop_event = self.create_thread(cmd_tag, timeout)

        except Exception:
            api_error.error('Failed waiting for command execution', 'UdpServer class, method "wait_for_exe"')

        return exe_thread, stop_event

    def create_thread(self, cmd_tag: str, timeout: Optional[float] = 5.0) -> ThreadInfo:
        stop_event = Event()
        exe_thread = Thread(target=self.listen,
                            args=(cmd_tag, stop_event, timeout))
        return exe_thread, stop_event
