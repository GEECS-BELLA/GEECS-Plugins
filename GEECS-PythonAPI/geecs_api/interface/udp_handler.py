import socket
import select
import queue
import threading
from typing import Optional
from geecs_api.interface.geecs_errors import *
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
                self.sock_cmd = None
                self.port_cmd = -1
                self.bounded_cmd = False
        except Exception:
            pass

    def send_cmd(self, ipv4=('', -1), msg='', err_in=ErrorAPI()):
        """ Send message. """

        err = ErrorAPI()
        sent = False
        try:
            self.sock_cmd.sendto(msg.encode('ascii'), ipv4)
            sent = True
        except Exception:
            err = ErrorAPI('Failed to send UDP message', 'UdpHandler class, method "send"')

        return sent, err.merge_with_previous(err_in)

    def ack_cmd(self, ready_timeout_sec=5.0, err_in=ErrorAPI()):
        """ Listen for command acknowledgement. """

        err = ErrorAPI()
        accepted = False

        try:
            ready = select.select([self.sock_cmd], [], [], ready_timeout_sec)
            if ready[0]:
                geecs_str = self.sock_cmd.recvfrom(self.buffer_size)
                geecs_ans = (geecs_str[0].decode('ascii')).split(">>")[-1]
                accepted = (geecs_ans == 'accepted')
            else:
                err = ErrorAPI('Socket not ready to receive', 'UdpHandler class, method "receive"', warning=True)

        except ErrorAPI('Failed to read UDP message!'):
            err = ErrorAPI('Failed to read UDP message', 'UdpHandler class, method "receive"')

        return accepted, err.merge_with_previous(err_in)

    @staticmethod
    def send_preset(preset='', err_in=ErrorAPI()):
        MCAST_GRP = '234.5.6.8'
        MCAST_PORT = 58432
        MULTICAST_TTL = 4

        err = ErrorAPI()
        sock = None

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
            sock.sendto(f'preset>>{preset}>>{socket.gethostbyname(socket.gethostname())}'.encode(),
                        (MCAST_GRP, MCAST_PORT))

        except Exception:
            err = ErrorAPI(f'Failed to send preset "{preset}"', 'UdpHandler class, method "send_preset"')

        finally:
            try:
                sock.close()
            except Exception:
                pass

        return err.merge_with_previous(err_in)


class UdpServer:
    def __init__(self, port=-1):
        # initialize publisher
        self.publisher = EventHandler(['UDP Message'])
        self.publisher.register('UDP Message', 'UDP handler', self.async_msg_handler)

        # FIFO queue of messages
        self.queue_msgs = queue.Queue()

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
        self.publisher.unregister_all()
        self.flush_queue()
        self.close_sock_exe()

    def close_sock_exe(self):
        """ Closes the socket. """

        try:
            if self.sock:
                self.sock.close()
                self.sock = None
                self.port = -1
                self.bounded = False
        except Exception:
            pass

    def next_msg(self, *args, **kwargs):
        err = ErrorAPI()

        try:
            tag, msg, err = self.queue_msgs.get(*args, **kwargs)
            self.queue_msgs.task_done()
        except queue.Empty:
            msg = ''
        except Exception:
            msg = ''
            err = ErrorAPI('Failed to dequeue UDP message', 'UdpServer class, method "next_msg"')

        return msg, err

    def flush_queue(self):
        flushed_msgs = list()

        try:
            for q_element in range(self.queue_msgs.qsize()):
                this_element = self.queue_msgs.get()
                self.queue_msgs.task_done()
                flushed_msgs.append(this_element)
        except Exception:
            pass
        finally:
            self.queue_msgs = queue.Queue()

        return flushed_msgs

    def async_msg_handler(self, message):  # message = tuple of string and ErrorAPI
        try:
            cmd_tag, udp_msg, error = message
            self.next_msg(False)  # tmp (to be handled by device) Queue.get() call to dequeue message

            if error.is_error:
                err_str = f'Error:\n\tDescription: {error.error_msg}\n\tSource: {error.error_src}'
            elif error.is_warning:
                err_str = f'Warning:\n\tDescription: {error.error_msg}\n\tSource: {error.error_src}'
            else:
                err_str = ''

            if err_str:
                print(err_str)
            else:
                print(f'Asynchronous UDP response to "{cmd_tag}":\n\t{udp_msg}')

        except Exception as ex:
            err = ErrorAPI(str(ex), 'UdpServer class, method "async_msg_handler"')
            print(err)

    def listen(self, cmd_tag: str, ready_timeout_sec=15.0) -> tuple[str, ErrorAPI]:
        """ Listens for messages. """

        err = ErrorAPI()
        geecs_ans = ''

        # receive message
        try:
            ready = select.select([self.sock], [], [], ready_timeout_sec)
            if ready[0]:
                geecs_str = self.sock.recvfrom(self.buffer_size)
                geecs_ans = (geecs_str[0].decode('ascii')).split(">>")[-2]
            else:
                err = ErrorAPI('Socket not ready to receive', 'UdpServer class, method "listen"', warning=True)

        except Exception:
            err = ErrorAPI('Failed to read UDP message', 'UdpServer class, method "listen"')

        # queue & publish message
        self.queue_msgs.put((cmd_tag, geecs_ans, err))
        self.publisher.publish('UDP Message', (cmd_tag, geecs_ans, err))

        return geecs_ans, err

    def wait_for_exe(self, cmd_tag: str, timeout_sec: Optional[float] = None, sync=False, err_in=ErrorAPI()):
        """ Waits for a UDP response (typ. command executed), in a separate thread by default. """

        err = ErrorAPI()

        try:
            if sync:
                geecs_ans, err = self.listen(cmd_tag, ready_timeout_sec=timeout_sec)
                print(f'Synchronous UDP response:\n\t{geecs_ans}')
            else:
                listen_thread = threading.Thread(target=self.listen,
                                                 args=(cmd_tag, timeout_sec))
                listen_thread.start()

        except Exception:
            err = ErrorAPI('Failed waiting for command execution', 'UdpServer class, method "wait_for_cmd"')

        return err.merge_with_previous(err_in)
