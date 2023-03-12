import socket
import select
import queue
import threading
from datetime import datetime as dtime
from typing import Optional
import geecs_api.interface.message_handling as mh
from geecs_api.interface.geecs_errors import ErrorAPI, api_error
from geecs_api.interface.event_handler import EventHandler


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
        self.close_sock_cmd()

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
                api_error.warning('Socket not ready to receive', 'UdpHandler class, method "receive"')

        except Exception:
            api_error.error('Failed to read UDP message', 'UdpHandler class, method "receive"')

        return accepted

    def register_handler(self, subscriber: str = '', callback=None) -> bool:
        if callback is not None and subscriber.strip():
            self.cmd_checker.publisher.register(self.cmd_checker.event_name, subscriber, callback)
            return True
        else:
            return False

    def unregister_handler(self, subscriber: str = '') -> bool:
        if subscriber.strip():
            self.cmd_checker.publisher.unregister(self.cmd_checker.event_name, subscriber)
            return True
        else:
            return False

    @staticmethod  # placeholder, will be moved to system class, 2 levels up
    def send_preset(preset=''):
        MCAST_GRP = '234.5.6.8'
        MCAST_PORT = 58432
        MULTICAST_TTL = 4

        sock = None

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
            sock.sendto(f'preset>>{preset}>>{socket.gethostbyname(socket.gethostname())}'.encode(),
                        (MCAST_GRP, MCAST_PORT))

        except Exception:
            api_error.error(f'Failed to send preset "{preset}"', 'UdpHandler class, method "send_preset"')

        finally:
            try:
                sock.close()
            except Exception:
                pass


class UdpServer:
    def __init__(self, port: int = -1):
        # initialize publisher
        self.event_name = 'UDP Message'
        self.publisher = EventHandler([self.event_name])
        self.publisher.register(self.event_name, 'UDP handler', mh.async_msg_handler)

        # FIFO queue of messages
        self.queue_msgs = queue.Queue()

        # message notifier
        self.notifier = threading.Condition()

        # initialize socket
        self.buffer_size = 1024
        self.port = port
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.sock.settimeout(120.0)
            self.sock.bind(('', self.port))
            self.bounded = True
        except Exception:
            self.sock = None
            self.port = -1
            self.bounded = False

    def __del__(self):
        self.publisher.unregister(self.event_name, 'UDP handler')
        mh.flush_queue(self.queue_msgs)
        self.close_sock_exe()

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

    def listen(self, cmd_tag: str, ready_timeout_sec=15.0) -> str:
        """ Listens for messages. """

        # receive message
        geecs_ans = stamp = ''
        err = ErrorAPI()  # no error object

        try:
            ready = select.select([self.sock], [], [], ready_timeout_sec)
            if ready[0]:
                geecs_str = self.sock.recvfrom(self.buffer_size)
                geecs_ans = (geecs_str[0].decode('ascii')).split(">>")[-2]
                stamp = dtime.now().__str__()
            else:
                err.warning('Socket not ready to receive', 'UdpServer class, method "listen"')

        except Exception:
            err.error('Failed to read UDP message', 'UdpServer class, method "listen"')

        # queue, notify & publish message
        try:
            net_msg = mh.NetworkMessage(tag=cmd_tag, stamp=stamp, msg=geecs_ans, err=err)
            mh.broadcast_msg(net_msg, self.notifier, self.queue_msgs, self.publisher, self.event_name)
            api_error.merge(err.error_msg, err.error_src, err.is_warning)

        except Exception:
            api_error.error('Failed to publish UDP message', 'UdpServer class, method "listen"')

        return geecs_ans

    def wait_for_exe(self, cmd_tag: str, timeout_sec: Optional[float] = None, sync=False):
        """ Waits for a UDP response (typ. command executed), in a separate thread by default. """

        try:
            if sync:
                geecs_ans, err = self.listen(cmd_tag, ready_timeout_sec=timeout_sec)
                print(f'Synchronous UDP response:\n\t{geecs_ans}')
            else:
                listen_thread = threading.Thread(target=self.listen,
                                                 args=(cmd_tag, timeout_sec))
                listen_thread.start()

        except Exception:
            api_error.error('Failed waiting for command execution', 'UdpServer class, method "wait_for_cmd"')
