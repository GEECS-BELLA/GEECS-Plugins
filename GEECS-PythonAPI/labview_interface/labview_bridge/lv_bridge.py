from labview_interface.labview_bridge.local_ip import local_ip
from labview_interface.labview_bridge.udp_client import UdpClient
from labview_interface.labview_bridge.tcp_server import TcpServer
from labview_interface.labview_bridge.tcp_client import TcpClient
import socket
import threading
import queue
import time
import re


class Bridge:
    def __init__(self):
        self.is_connected = False
        # self.sys_list = []

        self.remote_address = None
        self.remote_tcp_port = None
        self.local_tcp_port = None

        self.tcp_server = None
        self.tcp_client = None

        # user-defined handler should have the form: labview_handler(self, message)
        self.labview_handler = None
        self.app_id = 'LV_APP'

    def close(self):
        """ Closes the socket. """

        if self.tcp_server:
            self.tcp_server.flush_queue()
            self.tcp_server.close()

        if self.tcp_client:
            self.tcp_client.close()

        self.is_connected = False

    def handle_msgs(self, message=''):
        self.tcp_server.next_msg(False)

        if self.tcp_server is not None:
            if message.find('<CLOSE>') >= 0:
                lv_bridge.close()

            elif message.startswith('<ERROR>'):
                disconnect()
                raise ErrorLabVIEW(eval(message.replace('<ERROR>', '')))

            elif self.labview_handler is not None:
                self.labview_handler(message)
            else:
                print(message)


class ErrorLabVIEW(Exception):
    def __init__(self, sys_error=None):
        super().__init__(f'\n\nError: {sys_error[1]}\n\nCall chain: {sys_error[2]}\n\nDescription: {sys_error[0]}')


def listen_udp(host='localhost', port=44782, addresses=queue.Queue(), timeout_sec=None, debug=False):
    """ Listen on default UDP port for possible matching network stream. """

    if debug:
        print(f'UDP client on "{host}", port {port}')
    udp_local = UdpClient(host=host, port=port)

    if udp_local.bounded:
        time_0 = time.time()
        while True:
            if timeout_sec:
                time_elapsed = time.time() - time_0
                if time_elapsed >= timeout_sec:
                    udp_local.close()
                    addresses.put((None, None))
                    return False

            udp_msg_local = udp_local.receive(timeout_sec=1.0)

            # read message
            if isinstance(udp_msg_local, str):
                udp_msg = udp_msg_local
            else:
                udp_msg = None

            if udp_msg:
                pattern = re.compile(r'<TCP>IP-[0-9.]+-Port-[0-9]+-ID-' + lv_bridge.app_id)
                udp_match = pattern.search(udp_msg)

                if udp_match is not None:
                    if debug:
                        print('UDP broadcast received: ' + udp_msg)

                    pattern = re.compile(r'IP-[0-9.]+')
                    remote_address = pattern.search(udp_match[0])
                    remote_address = remote_address[0]
                    remote_address = remote_address[3:]

                    pattern = re.compile('Port-[0-9]+')
                    remote_port = pattern.search(udp_match[0])
                    remote_port = remote_port[0]
                    remote_tcp_port = int(remote_port[5:])

                    addresses.put((remote_address, remote_tcp_port))

                    udp_local.close()
                    return True

            time.sleep(0.25)
    else:
        udp_local.close()
        addresses.put((None, None))
        return False


def connect(timeout_sec=None, debug=False, mode='local'):
    """ Connects to LabVIEW. mode = 'local' or 'network' """

    if not lv_bridge.is_connected:
        # listen for UDP broadcast and collect remote IP and TCP port
        if timeout_sec:
            udp_timeout_sec = timeout_sec
        else:
            udp_timeout_sec = 5.0

        udp_threads = []
        udp_found = False
        udp_queue = queue.Queue()

        if mode.lower() == 'network':
            local_IP = local_ip()
        else:
            local_IP = 'localhost'
        if debug:
            print('local IP: ' + local_IP)

        while not udp_found:
            for udp_port_offset in range(5):
                udp_process = \
                    threading.Thread(target=listen_udp,
                                     args=[local_IP, 44782+udp_port_offset, udp_queue, udp_timeout_sec, debug])
                udp_process.start()
                udp_threads.append(udp_process)

            for udp_process in udp_threads:
                udp_process.join()

            this_address = (None, None)
            for q_element in range(udp_queue.qsize()):
                this_address = udp_queue.get()
                udp_queue.task_done()
                if this_address[0] is not None:
                    udp_found = True
                    break

            if (not udp_found) & (timeout_sec is not None):
                lv_bridge.close()
                raise ConnectionRefusedError
            elif udp_found:
                if socket.gethostbyname('localhost') == this_address[0]:
                    lv_bridge.remote_address = 'localhost'
                else:
                    lv_bridge.remote_address = this_address[0]
                lv_bridge.remote_tcp_port = this_address[1]

        # close any current TCP server
        if lv_bridge.tcp_server:
            lv_bridge.tcp_server.close()

        # create TCP server on self-assigned port
        lv_bridge.tcp_server = TcpServer(host=local_IP)
        lv_bridge.local_tcp_port = lv_bridge.tcp_server.port[1]

        # register for new TCP messages (originating from LabVIEW, not a response)
        lv_bridge.tcp_server.publisher.register('TCP Message', 'bridge', lv_bridge.handle_msgs)

        # start listener in separate thread
        end_char = '\r\n'
        thread_timeout_sec = None
        multi_thread = False
        publisher = None
        client_tread = threading.Thread(target=lv_bridge.tcp_server.accept,
                                        args=(end_char, thread_timeout_sec, multi_thread, publisher, debug))
        client_tread.start()

        # close any current TCP client
        if lv_bridge.tcp_client:
            lv_bridge.tcp_client.close()

        lv_bridge.tcp_client = TcpClient()

        # connect TCP client
        if not lv_bridge.tcp_client.connect(host=lv_bridge.remote_address,
                                            port=lv_bridge.remote_tcp_port, timeout_sec=timeout_sec):
            lv_bridge.close()
            raise ConnectionRefusedError

        # handshake with LabVIEW TCP server
        handshake_msg = '<TCP>IP-' + socket.gethostbyname(local_IP) + '-Port-' + str(lv_bridge.local_tcp_port)

        (this_response, this_echo) = \
            lv_bridge.tcp_client.echo(msg=handshake_msg, timeout_sec=timeout_sec, debug=debug)

        # if echoed, request system dictionary
        if this_echo:
            lv_bridge.is_connected = True
        else:
            lv_bridge.close()
            raise ConnectionRefusedError


def disconnect():
    """ Close connection to LabVIEW counterpart. """
    if lv_bridge.is_connected:
        lv_bridge.tcp_client.send(msg='<CLOSE>', timeout_sec=1.0)

    lv_bridge.close()


# e.g. bridge_com('box', 'set_do', [value, time], sync=sync)
def bridge_com(class_name=None, method_name=None, list_args=None, sync=True, timeout_sec=None, debug=False):
    """ Send command to, and receive response from LabVIEW. """
    if not lv_bridge.is_connected:
        connect(2.0, False, 'network')

    tag = time.time_ns()
    system_request = repr(tag) + ', ' + repr(sync) + f', {class_name}, {method_name}, {list_args}'

    if sync:
        try:
            system_response = \
                lv_bridge.tcp_client.send_receive(msg=system_request, timeout_sec=timeout_sec, debug=debug)
        except:
            disconnect()
            raise

        if system_response.startswith("'''<ERROR>"):
            error_message = system_response.replace('<ERROR>', '')
            lv_error = eval(eval(error_message))
            disconnect()
            raise ErrorLabVIEW(lv_error)
        elif not system_response:
            return None
        else:
            # noinspection PyBroadException
            try:
                system_response = eval(system_response)
                if not isinstance(system_response, tuple):
                    return None

                if int(system_response[0]) != tag:
                    # Try receiving again
                    system_response = \
                        lv_bridge.tcp_client.receive(timeout_sec=timeout_sec, debug=debug)

                    if system_response.startswith("'''<ERROR>"):
                        error_message = system_response.replace('<ERROR>', '')
                        lv_error = eval(eval(error_message))
                        disconnect()
                        raise ErrorLabVIEW(lv_error)
                    elif not system_response:
                        return None
                    else:
                        try:
                            system_response = eval(system_response)
                            if not isinstance(system_response, tuple):
                                return None

                            if int(system_response[0]) != tag:
                                return None
                            else:
                                return system_response[1]
                        except:
                            return None
                else:
                    return system_response[1]
            except:
                return None

    else:
        try:
            lv_bridge.tcp_client.send(msg=system_request, timeout_sec=timeout_sec, debug=debug)
        except:
            disconnect()
            raise

        return None


# instantiate Bridge class when module is called
lv_bridge = Bridge()


if __name__ == "__main__":
    print('connecting...')
    connect(2.0, True, 'network')

    if lv_bridge.is_connected:
        print('connected')
    else:
        print('connection failed')

    status = lv_bridge.is_connected
    t0 = time.time()
    while time.time()-t0 < 5:
        if lv_bridge.is_connected != status:
            status = lv_bridge.is_connected

            if status:
                print('reconnected')
            else:
                print('connection closed')

            time.sleep(0.5)

    if lv_bridge.is_connected:
        disconnect()
        print('disconnected')
