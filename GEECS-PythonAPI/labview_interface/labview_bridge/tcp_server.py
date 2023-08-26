from interface_api.labview_bridge.event import Publisher
import socket
import select
import threading
import queue
import time
import re


class TcpServer:
    def __init__(self, host=None, port=0, connections=1):
        """ Creates a IPv4 socket and binds it to "host" on port "port" (by default: self-assignment).
        Finally, it starts listening. """

        # initialize publisher
        self.publisher = Publisher(['TCP Message'])

        # FIFO queue of messages
        self.queue_msgs = queue.Queue()

        # initialize socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        if not host:
            host = socket.gethostbyname('localhost')

        # bind to the port
        self.sock.bind((host, port))

        # listens for connections
        self.sock.listen(connections)

        # register port number (port=0 => self-assignment)
        self.port = self.sock.getsockname()

    # def next_msg(self, read_timeout=1):
    def next_msg(self, *args, **kwargs):
        try:
            # this_msg = self.queue_msgs.get(timeout=read_timeout)
            this_msg = self.queue_msgs.get(*args, **kwargs)
            self.queue_msgs.task_done()
        except queue.Empty:
            this_msg = ''

        return this_msg

    def flush_queue(self):
        flushed_msgs = list()

        for q_element in range(self.queue_msgs.qsize()):
            this_element = self.queue_msgs.get()
            self.queue_msgs.task_done()
            flushed_msgs.append(this_element)

        self.queue_msgs = queue.Queue()
        return flushed_msgs

    def close(self):
        """ Closes the socket. """

        self.sock.close()

    def accept(self, end_char='\r\n', timeout_sec=None, multi_thread=True, handler=None, debug=False):
        """ Calls socket.accept() and runs a listener, in a separate thread by default. """

        # set timeout (= non-blocking)
        self.sock.settimeout(timeout_sec)

        # listens for connections
        if debug:
            print('started listening on port: ' + str(self.sock.getsockname()[1]) + ' of IP: ' +
                  self.sock.getsockname()[0])

        # noinspection PyBroadException
        try:
            # wait for a connection
            client, client_address = self.sock.accept()
            if debug:
                print('connection by: ' + str(client_address))

            # set timeout (= non-blocking socket)
            client.settimeout(timeout_sec)

            if multi_thread:
                if debug:
                    print('creating new thread')
                client_tread = threading.Thread(target=self.listen,
                                                args=(client, client_address, end_char, timeout_sec, handler, debug))
                if debug:
                    print('starting listener in new thread')
                client_tread.start()
            else:
                if debug:
                    print('starting listener in parent thread')
                self.listen(client, client_address,
                            end_char=end_char, timeout_sec=timeout_sec, handler=handler, debug=debug)

            return client_address

        except Exception:
            return False

    def listen(self, client, client_address, end_char='\r\n', timeout_sec=60, handler=None, debug=False):
        """ Listens for client messages and adds it to the object's queue.
        It echoes the message back if the message starts by '=?'. """

        connection_broken = False
        this_buffer = ''
        while True:
            if debug:
                print('--------------------\nwaiting for message from ' + str(client_address))

            # noinspection PyBroadException
            try:
                # pass buffer and check for stored messages
                line_breaks = [(m.start(), m.end()) for m in re.finditer(end_char, this_buffer)]

                if line_breaks:
                    this_msg = this_buffer[:line_breaks[0][0]]
                    this_buffer = this_buffer[line_breaks[0][1]:]
                else:
                    this_msg = str(this_buffer)

                    # read message and look for termination character
                    if debug:
                        print('receiving message (unknown size)')

                    chunk_size = 2**12
                    time_0 = time.time()
                    time_elapsed = 0
                    while True:
                        if timeout_sec:
                            ready = select.select([client], [], [], timeout_sec - time_elapsed)
                        else:
                            ready = select.select([client], [], [], timeout_sec)

                        if ready[0]:
                            chunk = client.recv(chunk_size)
                            if not chunk:
                                connection_broken = True
                                break

                            time_elapsed = time.time() - time_0

                            this_msg += chunk.decode()
                            line_breaks = [(m.start(), m.end()) for m in re.finditer(end_char, this_msg)]
                            if line_breaks:
                                this_buffer = this_msg[line_breaks[0][1]:]
                                this_msg = this_msg[:line_breaks[0][0]]
                                break
                        else:
                            break
            except Exception:
                connection_broken = True
                break

            # check for echo request
            if this_msg[0:2] == '=?':
                this_msg = this_msg[2:]

                this_reply = str(this_msg)
                if not re.findall(end_char, this_msg):
                    this_reply = this_msg + end_char

                if debug:
                    print('sending reply: ' + this_reply)

                client.sendall(this_reply.encode())
                if debug:
                    print('reply sent')

            if debug:
                print('client ' + str(client_address) + ' received: ' + repr(this_msg))

            # check for connection closure
            if this_msg.lower() == '<CLOSE>':
                connection_broken = True

            # skip publishing and queueing if connection is broken
            if connection_broken:
                break

            # publish message
            if handler is not None:
                if debug:
                    print('publishing "' + this_msg + '" on custom event publisher')
                handler.publish_all(this_msg)

            if debug:
                print('adding "' + this_msg + '" to the TCP server queue')
            self.queue_msgs.put(this_msg)

            if debug:
                print('publishing "' + this_msg + '" TCP Message')
            self.publisher.publish('TCP Message', this_msg)

        if connection_broken:
            if debug:
                print('closing client connection')
            client.close()

            if debug:
                print('publishing event "TCP Message" to close connection')
            self.publisher.publish('TCP Message', '<CLOSE>')


if __name__ == "__main__":
    print('starting server')
    ns_server = TcpServer(port=1234)
    print('server address: ' + str(ns_server.port))
    server_error = False

    try:
        print('waiting for connection')
        new_listener = ns_server.accept(timeout_sec=None, multi_thread=False, debug=True)

        if isinstance(new_listener, bool):
            if not new_listener:
                print('--------------------')
                print('server error')
        else:
            msg_index = 1
            while True:
                new_msg = ns_server.next_msg(read_timeout=2)
                if not new_msg:
                    break

                print('--------------------')
                print('message ' + str(msg_index) + ' from client: ' + new_msg)
                msg_index += 1
                if new_msg.lower() == 'close ns':
                    print('connection closed by client')
                    break

    except KeyboardInterrupt:
        print('execution interrupted')

    print('--------------------')
    print('unread messages: ' + str(ns_server.flush_queue()))
    print('closing server')
    ns_server.close()
