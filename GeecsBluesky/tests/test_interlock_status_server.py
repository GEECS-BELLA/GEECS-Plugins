"""The interlock status stream: the retired server's wire format, over a loopback socket."""

from __future__ import annotations

import socket
import struct

import pytest

from geecs_bluesky.interlocks.install import InterlockRegistry, InterlockStatus
from geecs_bluesky.interlocks.status_server import (
    NO_MONITORS,
    InterlockStatusServer,
    frame,
    render_status,
    start_status_server,
)


def test_the_retired_servers_message_format() -> None:
    message = render_status(
        [
            InterlockStatus("cam1", unsafe=False),
            InterlockStatus("vacuum", unsafe=True, diagnostic="Pressure=0.001"),
            InterlockStatus("gauge", unsafe=True),
        ]
    )
    assert message == (
        "SAFE: cam1 | WARNING! vacuum UNSAFE - Pressure=0.001 | WARNING: gauge UNSAFE"
    )


def test_no_interlocks_is_a_warning() -> None:
    assert render_status([]) == NO_MONITORS == "WARNING! No monitors registered"


def test_a_pipe_never_reaches_the_wire() -> None:
    message = render_status([InterlockStatus("a|b", True, "x | y")])
    assert message == "WARNING! a/b UNSAFE - x / y"


def test_frame_is_a_big_endian_length_then_utf8() -> None:
    assert frame("SAFE: µ") == struct.pack(">I", 8) + "SAFE: µ".encode()  # µ is 2 bytes


def _read_frame(sock: socket.socket) -> str:
    def exactly(n: int) -> bytes:
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            assert chunk, "server closed the stream"
            data += chunk
        return data

    (length,) = struct.unpack(">I", exactly(4))
    return exactly(length).decode("utf-8")


def test_clients_read_the_live_snapshot() -> None:
    statuses = [InterlockStatus("vacuum", unsafe=False)]
    server = InterlockStatusServer(
        lambda: list(statuses), "127.0.0.1", 0, interval=0.02
    )
    server.start()
    try:
        with (
            socket.create_connection(server.address, timeout=2) as a,
            socket.create_connection(server.address, timeout=2) as b,
        ):
            assert _read_frame(a) == "SAFE: vacuum"
            assert _read_frame(b) == "SAFE: vacuum"
            statuses[0] = InterlockStatus("vacuum", True, "Pressure=0.001")
            for _ in range(50):  # within a few intervals
                if _read_frame(a).startswith("WARNING!"):
                    break
            else:
                pytest.fail("the trip never reached the stream")
    finally:
        server.stop()


def test_a_client_that_hangs_up_does_not_stop_the_server() -> None:
    server = InterlockStatusServer(list, "127.0.0.1", 0, interval=0.01)
    server.start()
    try:
        socket.create_connection(server.address, timeout=2).close()
        with socket.create_connection(server.address, timeout=2) as again:
            assert _read_frame(again) == NO_MONITORS
    finally:
        server.stop()


def _config(tmp_path, body: str):
    path = tmp_path / "config.ini"
    path.write_text(body)
    return path


def test_no_port_configured_means_no_server(tmp_path) -> None:
    assert start_status_server(None, config_path=_config(tmp_path, "[Paths]\n")) is None


def test_a_bad_port_means_no_server(tmp_path, caplog) -> None:
    path = _config(tmp_path, "[interlocks]\nstatus_port = five\n")
    assert start_status_server(None, config_path=path) is None
    assert "not a port" in caplog.text


def test_a_taken_port_means_no_server(tmp_path, caplog) -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        path = _config(
            tmp_path, f"[interlocks]\nstatus_host = 127.0.0.1\nstatus_port = {port}\n"
        )
        assert start_status_server(None, config_path=path) is None
    assert "cannot listen" in caplog.text


def test_configured_server_reports_the_registry(tmp_path) -> None:
    registry = InterlockRegistry(unarmed={"vacuum": "not connected: U_Gauge"})
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    path = _config(
        tmp_path, f"[interlocks]\nstatus_host = 127.0.0.1\nstatus_port = {port}\n"
    )
    server = start_status_server(registry, config_path=path)
    assert server is not None
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            assert _read_frame(client) == (
                "WARNING! vacuum UNSAFE - not armed: not connected: U_Gauge"
            )
    finally:
        server.stop()
