"""The Master Control stream over a real socket: every client, the same frames."""

import socket
import time

import pytest

from geecs_interlocks.monitor import InterlockStatus, Snapshot
from geecs_interlocks.status_server import StatusServer
from geecs_interlocks.wire import parse_status, read_frame


@pytest.fixture
def server():
    current = {"snap": Snapshot((InterlockStatus("a", "", "", "ok"),))}
    srv = StatusServer(
        lambda: current["snap"], "127.0.0.1", 0, interval=0.02, wire_format="summary"
    )
    srv.start()
    srv.current = current
    yield srv
    srv.stop()


def connect(srv):
    return socket.create_connection(srv.address, timeout=2)


def test_every_client_reads_the_verdict(server):
    with connect(server) as one, connect(server) as two:
        assert read_frame(one) == read_frame(two) == "SAFE | OK: a"
        deadline = time.time() + 2
        while len(server.clients()) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert len(server.clients()) == 2


def test_frames_follow_the_snapshot(server):
    with connect(server) as sock:
        read_frame(sock)
        server.current["snap"] = Snapshot(
            (InterlockStatus("a", "", "", "tripped", "P=5"),)
        )
        deadline = time.time() + 2
        while time.time() < deadline:
            if not parse_status(read_frame(sock)).safe:
                break
        else:
            pytest.fail("the trip never reached the stream")
    assert server.current_message() == "WARNING! TRIPPED | TRIPPED: a - P=5"


def test_a_departed_client_is_dropped(server):
    sock = connect(server)
    read_frame(sock)
    sock.close()
    deadline = time.time() + 3
    while server.clients() and time.time() < deadline:
        time.sleep(0.02)
    assert server.clients() == []
    assert server.frames_sent > 0


def test_a_taken_port_raises(server):
    other = StatusServer(lambda: Snapshot(), "127.0.0.1", server.address[1])
    with pytest.raises(OSError):
        other.start()


def test_the_default_stream_is_legacy():
    srv = StatusServer(
        lambda: Snapshot((InterlockStatus("a", "", "", "tripped", "P=5"),)),
        "127.0.0.1",
        0,
        interval=0.02,
    )
    srv.start()
    try:
        with connect(srv) as sock:
            assert read_frame(sock) == "WARNING! a - P=5"
    finally:
        srv.stop()
