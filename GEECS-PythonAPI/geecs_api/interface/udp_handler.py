import socket
import select
from queue import Queue
from threading import Thread, Condition
from datetime import datetime as dtime
from typing import Optional
import geecs_api.devices as gd
import geecs_api.interface.message_handling as mh
from geecs_api.interface.geecs_errors import ErrorAPI, api_error


class UdpHandler:
    def __init__(self):
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
            self.cmd_checker = UdpServer(port=self.port_exe)

        except Exception:
            self.port_cmd = self.port_exe = -1

    def __del__(self):
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

    def ack_cmd(self, ready_timeout_sec: Optional[float] = 5.0) -> bool:
        """ Listen for command acknowledgement. """

        accepted = False
        try:
            ready = select.select([self.sock_cmd], [], [], ready_timeout_sec)
            if ready[0]:
                geecs_str = self.sock_cmd.recv(self.buffer_size)
                geecs_ans = (geecs_str.decode('ascii')).split(">>")[-1]
                accepted = (geecs_ans == 'accepted')
            else:
                api_error.warning('Socket not ready to receive', 'UdpHandler class, method "ack_cmd"')

        except Exception:
            api_error.error('Failed to read UDP acknowledge message', 'UdpHandler class, method "ack_cmd"')

        return accepted

    def register_handler(self, subscriber: gd.GeecsDevice) -> bool:
        if subscriber.is_valid():
            self.cmd_checker.owner = subscriber
            return True
        else:
            return False

    def unregister_handler(self):
        self.cmd_checker.owner = None


class UdpServer:
    def __init__(self, port: int = -1):
        self.owner: Optional[gd.GeecsDevice] = None
        self.threads: list[Thread] = []

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

    def __del__(self):
        try:
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
        for i_th in range(len(self.threads)):
            if not self.threads[-1-i_th].isAlive():
                self.threads.pop(-1-i_th)

    def wait_for_all(self, timeout: Optional[float] = None) -> bool:
        self._cleanup_threads()
        any_alive = False

        for thread in self.threads:
            thread.join(timeout)
            any_alive |= thread.is_alive()

        return not any_alive

    def wait_for_last(self, timeout: Optional[float] = None):
        self.threads[-1].join(timeout)
        alive = self.threads[-1].is_alive()
        return not alive

    def listen(self, cmd_tag: str, ready_timeout_sec: float = 5.0) -> str:
        """ Listens for messages. """

        # receive message
        geecs_ans = stamp = ''
        err = ErrorAPI()  # no error object

        try:
            ready = select.select([self.sock], [], [], ready_timeout_sec)
            if ready[0]:
                geecs_str = self.sock.recvfrom(self.buffer_size)
                geecs_ans = geecs_str[0].decode('ascii')
                stamp = dtime.now().__str__()
            else:
                err.warning('Socket not ready to receive', 'UdpServer class, method "listen"')

        except Exception:
            err.error('Failed to read UDP message', 'UdpServer class, method "listen"')

        # queue, notify & publish message
        try:
            net_msg = mh.NetworkMessage(tag=cmd_tag, stamp=stamp, msg=geecs_ans, err=err)
            if self.owner:
                try:
                    self.owner.handle_response(net_msg, self.notifier, self.queue_msgs)
                except Exception:
                    err.error('Failed to handle TCP subscription',
                              'TcpSubscriber class, method "async_listener"')
            api_error.merge(err.error_msg, err.error_src, err.is_warning)

        except Exception:
            api_error.error('Failed to publish UDP message', 'UdpServer class, method "listen"')

        return geecs_ans

    def wait_for_exe(self, cmd_tag: str, ready_timeout_sec: float = 120.0, sync: bool = False) -> Optional[Thread]:
        """ Waits for a UDP response (typ. command executed), in a separate thread by default. """

        exe_thread: Optional[Thread] = None
        self._cleanup_threads()

        try:
            if sync:
                geecs_ans, _ = self.listen(cmd_tag, ready_timeout_sec=ready_timeout_sec)
                print(f'Synchronous UDP response:\n\t{geecs_ans}')
            else:

                exe_thread = Thread(target=self.listen,
                                    args=(cmd_tag, ready_timeout_sec))
                exe_thread.start()
                self.threads.append(exe_thread)

        except Exception:
            api_error.error('Failed waiting for command execution', 'UdpServer class, method "wait_for_cmd"')

        return exe_thread
