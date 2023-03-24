from __future__ import annotations

import math
import socket
import time
import select
from queue import Queue
from threading import Thread, Condition, Event, Lock
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

        self.buffer_size = 1024
        self.sock_cmd = None
        self.bounded_cmd = False

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
            api_error.error('Failed to send UDP message', 'UdpHandler class, method "send"')

        return sent

    def ack_cmd(self, timeout: Optional[float] = 5.0) -> bool:
        """ Listen for command acknowledgement. """

        accepted = False
        try:
            ready = select.select([self.sock_cmd], [], [], timeout)
            if ready[0]:
                geecs_str = self.sock_cmd.recv(self.buffer_size)
                geecs_ans = (geecs_str.decode('ascii')).split(">>")[-1]
                accepted = (geecs_ans == 'accepted')
            else:
                api_error.warning('Socket not ready to receive', 'UdpHandler class, method "ack_cmd"')

        except Exception:
            api_error.error('Failed to read UDP acknowledge message', 'UdpHandler class, method "ack_cmd"')

        return accepted

    def register_handler(self) -> bool:
        self.cmd_checker.subscribed = self.cmd_checker.owner.is_valid()
        return self.cmd_checker.subscribed

    def unregister_handler(self):
        self.cmd_checker.subscribed = False


class UdpServer:
    # Static
    threads_lists_access = Lock()
    all_threads: list[(Thread, Event)] = []  # need to be owned and managed by GeecsDevice

    @staticmethod
    def cleanup_threads():
        with UdpServer.threads_lists_access:
            for it in range(len(UdpServer.all_threads)):
                if not UdpServer.all_threads[-1 - it][0].is_alive():
                    UdpServer.all_threads.pop(-1 - it)

    @staticmethod
    def wait_for_all_devices(timeout: Optional[float] = None):
        UdpServer.cleanup_threads()
        any_alive = False

        with UdpServer.threads_lists_access:
            for thread in UdpServer.all_threads:
                thread[0].join(timeout)
                any_alive |= thread[0].is_alive()

        return not any_alive

    # Object
    def __init__(self, owner: GeecsDevice, port: int = -1):
        self.owner: GeecsDevice = owner
        self.subscribed = False
        self.own_threads: list[(Thread, Event)] = []  # need to be owned and managed by GeecsDevice

        # FIFO queue of messages
        self.queue_msgs = Queue()

        # message notifier
        self.notifier = Condition()

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
            self.stop_waiting_for_all_cmds()
            mh.flush_queue(self.queue_msgs)
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

    def _cleanup_threads(self):
        with UdpServer.threads_lists_access:
            for it in range(len(self.own_threads)):
                if not self.own_threads[-1 - it][0].is_alive():
                    self.own_threads.pop(-1 - it)

            for it in range(len(UdpServer.all_threads)):
                if not UdpServer.all_threads[-1 - it][0].is_alive():
                    UdpServer.all_threads.pop(-1 - it)

    def wait_for_all_cmds(self, timeout: Optional[float] = None) -> bool:
        self._cleanup_threads()
        any_alive = False

        with UdpServer.threads_lists_access.acquire:
            for thread in self.own_threads:
                thread[0].join(timeout)
                any_alive |= thread[0].is_alive()

        return not any_alive

    def stop_waiting_for_all_cmds(self):
        self._cleanup_threads()

        with UdpServer.threads_lists_access:
            for thread in self.own_threads:
                thread[1].set()

    def wait_for_cmd(self, thread: Thread, timeout: Optional[float] = None):
        with UdpServer.threads_lists_access:
            if self.owner.is_valid() and thread.is_alive():
                thread.join(timeout)

            alive = thread.is_alive()

        return not alive

    def stop_waiting_for_cmd(self, thread: Thread, stop: Event):
        if self.owner.is_valid() and thread.is_alive():
            stop.set()

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
                    self.owner.handle_response(net_msg, self.notifier, self.queue_msgs)
                except Exception:
                    api_error.error('Failed to handle UDP response', 'UdpServer class, method "listen"')

        except Exception:
            api_error.error('Failed to publish UDP message', 'UdpServer class, method "listen"')

        return geecs_ans

    def wait_for_exe(self, cmd_tag: str, timeout: Optional[float] = 5.0, sync: bool = False) -> ThreadInfo:
        """ Waits for a UDP response (typ. command executed), in a separate thread by default. """

        exe_thread: Optional[Thread] = None
        stop_event: Optional[Event] = None
        self._cleanup_threads()

        try:
            if sync:
                self.listen(cmd_tag, timeout=timeout)
            else:
                self.launch_thread(cmd_tag, timeout)

        except Exception:
            api_error.error('Failed waiting for command execution', 'UdpServer class, method "wait_for_exe"')

        return exe_thread, stop_event

    def launch_thread(self, cmd_tag: str, timeout: Optional[float] = 5.0):
        with UdpServer.threads_lists_access:
            stop_event = Event()
            exe_thread = Thread(target=self.listen,
                                args=(cmd_tag, stop_event, timeout))
            exe_thread.start()  # to be done GeecsDevice

            self.own_threads.append((exe_thread, stop_event))
            UdpServer.all_threads.append((exe_thread, stop_event))
