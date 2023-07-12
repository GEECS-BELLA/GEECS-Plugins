#!/usr/bin/env python3
"""ZeroMQ receiver for numpy array data"""

import zmq
import numpy as np
from binascii import crc_hqx
import logging
logging.basicConfig(level=logging.INFO)


def recv_array(socket, flags=0, copy=True, track=False):
    """Receive a numpy array with metadata

    Args:
        socket (zmq.Socket): ZeroMQ socket
        flags (int, optional): ZeroMQ socket flags. Defaults to 0.
        copy (bool, optional): determines if the data being received is copied
            or instead the received array buffer is built on the received data
            buffer. Defaults to True.
        track (bool):  Should the message be tracked for notification that ZMQ
            has finished with it? Ignored if copy=True. Defaults to False.

    Returns:
        np.ndarray: received numpy array
    """
    md = socket.recv_json(flags=flags)
    msg = socket.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    A = np.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])


def main():
    """Main process for testing receives

    Process is a PUB/SUB model that first registers itself using a
    Zero MQ request/reply connection, and then prints out each numpy
    array that it receives
    """
    LOG = logging.getLogger("main")
    try:
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.connect("tcp://localhost:5555")
        subscriber.setsockopt(zmq.SUBSCRIBE, b"")

        # https://zguide.zeromq.org/docs/chapter2/#Node-Coordination
        # Second, synchronize with publisher
        syncclient = context.socket(zmq.REQ)
        syncclient.connect("tcp://localhost:5556")

        # send a synchronization request
        syncclient.send(b'')

        # wait for synchronization reply
        syncclient.recv()

        while True:
            # Copy received data in to local variables
            array = recv_array(subscriber)
            LOG.info("%s", array)

    except KeyboardInterrupt:
        context.term()
        print("Exiting...")


if __name__ == "__main__":
    main()
