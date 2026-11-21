"""The Master Control stream over a real socket: every client, the same frames."""

import dataclasses
import socket
import time

import pytest

from geecs_interlocks.monitor import InterlockStatus, Snapshot
from geecs_interlocks.status_server import StatusServer
from geecs_interlocks.wire import parse_status, read_frame


def fresh(snapshot):
    """*snapshot* as a running monitor publishes it: taken just now."""
    return dataclasses.replace(snapshot, taken=time.time())


@pytest.fixture
def server():
    current = {"snap": Snapshot((InterlockStatus("a", "", "", "ok"),))}
    srv = StatusServer(
        lambda: fresh(current["snap"]),
        "127.0.0.1",
        0,
        interval=0.02,
        wire_format="summary",
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


def test_the_address_after_stop_is_the_requested_one():
    srv = StatusServer(lambda: Snapshot(), "127.0.0.1", 0)
    srv.start()
    srv.stop()
    assert srv.address == ("127.0.0.1", 0)


def test_a_taken_port_raises(server):
    other = StatusServer(lambda: Snapshot(), "127.0.0.1", server.address[1])
    with pytest.raises(OSError):
        other.start()


def test_the_default_stream_is_summary():
    srv = StatusServer(
        lambda: fresh(Snapshot((InterlockStatus("a", "", "", "tripped", "P=5"),))),
        "127.0.0.1",
        0,
        interval=0.02,
    )
    srv.start()
    try:
        with connect(srv) as sock:
            assert read_frame(sock) == "WARNING! TRIPPED | TRIPPED: a - P=5"
    finally:
        srv.stop()


@pytest.mark.parametrize("wire_format", ["summary", "legacy"])
def test_a_stalled_evaluation_trips_the_stream(wire_format):
    """A last verdict of SAFE must not be re-sent once the loop stops replacing it."""
    safe = Snapshot((InterlockStatus("a", "", "", "ok"),), taken=time.time() - 5)
    assert safe.safe
    srv = StatusServer(
        lambda: safe, "127.0.0.1", 0, interval=0.02, wire_format=wire_format
    )
    srv.start()
    try:
        with connect(srv) as sock:
            message = read_frame(sock)
    finally:
        srv.stop()
    assert "WARNING" in message and "SAFE" not in message.upper().replace("UNSAFE", "")
    assert "evaluation stopped 5 s ago" in message
    assert not parse_status(message).safe
