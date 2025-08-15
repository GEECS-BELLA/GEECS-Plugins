#!/usr/bin/env python3
"""ZeroMQ server to send numpy array data

Designed to take in a queue of numpy arrays and sends
each array individually. Currently uses TCP"""

import zmq
import json
from time import sleep
import numpy as np
from binascii import crc_hqx
import multiprocessing as mp
import logging
logging.basicConfig(level=logging.INFO)


def producer(emplaceQueue):
    """Test random array producer for sends

    Args:
        emplaceQueue (mp.Queue): queue to place data in to
    """
    LOG = logging.getLogger("producer")
    while True:
        LOG.info("Producing data...")
        array = (np.random.rand(11, 11) * 256).astype("uint8")
        emplaceQueue.put(array)
        sleep(1)


def send_array(socket, A, flags=0, copy=True, track=False):
    """Send a numpy array with metadata

    Args:
        socket (zmq.Socket): ZeroMQ socket
        A (np.ndarray): numpy array to be sent
        flags (int, optional): ZeroMQ socket flags. Defaults to 0.
        copy (bool, optional): determines if the data being sent is copied
            or instead sent directly from the array's data buffer. Defaults to True.
        track (bool):  Should the message be tracked for notification that ZMQ
            has finished with it? Ignored if copy=True. Defaults to False.

    Returns:
        int: result of zmq.Socket.send
    """
    md = dict(
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)


def main(waitQueue):
    """Main process for testing sends

    Process is a PUB/SUB model that waits for connections before sending
    data. Once a connection has been made using a Zero MQ request/reply
    connection, the process begins to send any data that was loaded in
    the Queue

    Args:
        waitQueue (mp.Queue): Queue used for passing data between processes
    """
    SUBSCRIBERS_EXPECTED = 1

    LOG = logging.getLogger("main")
    try:
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:5555")

        # Socket to receive signals
        syncservice = context.socket(zmq.REP)
        syncservice.bind("tcp://*:5556")

        # Get synchronization from subscribers
        subscribers = 0
        while subscribers < SUBSCRIBERS_EXPECTED:
            # wait for synchronization request
            msg = syncservice.recv()
            # send synchronization reply
            syncservice.send(b'')
            subscribers += 1
            LOG.info(f"+1 subscriber ({subscribers}/{SUBSCRIBERS_EXPECTED})")

        # Flush queue
        while not waitQueue.empty():
            waitQueue.get()

        while True:
            LOG.info("Awaiting data...")
            array = waitQueue.get()
            send_array(socket, array)

    except KeyboardInterrupt:
        context.term()
        print("Exiting...")


if __name__ == "__main__":
    # Create queue with a max size of 10
    queue = mp.Queue(10)
    p0 = mp.Process(target=main, args=(queue,))
    p1 = mp.Process(target=producer, args=(queue,))
    p0.start()
    p1.run()
    p0.kill()
