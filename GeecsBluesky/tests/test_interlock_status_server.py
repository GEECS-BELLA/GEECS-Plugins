"""The interlock status stream: the retired server's wire format, over a loopback socket."""

from __future__ import annotations

import socket
import struct

import pytest

from geecs_bluesky.interlocks.install import InterlockRegistry, InterlockStatus
from geecs_bluesky.interlocks.status_server import (
    InterlockStatusServer,
    frame,
    render_status,
    start_status_server,
)

OK = InterlockStatus("cam1", unsafe=False)
TRIP = InterlockStatus("vacuum", unsafe=True, diagnostic="Pressure=0.001")
TRIP_BARE = InterlockStatus("gauge", unsafe=True)
UNARMED = InterlockStatus(
    "stamp", unsafe=True, diagnostic="not armed: not connected: U_Gauge", armed=False
)


def test_all_safe_leads_with_safe() -> None:
    assert render_status([OK, InterlockStatus("vacuum", False)]) == (
        "SAFE | OK: cam1 | OK: vacuum"
    )


def test_a_trip_leads_with_the_warning() -> None:
    assert render_status([TRIP, OK, TRIP_BARE, UNARMED]) == (
        "WARNING! TRIPPED | TRIPPED: vacuum - Pressure=0.001 | OK: cam1"
        " | TRIPPED: gauge | NOT ARMED: stamp - not connected: U_Gauge"
    )


def test_no_interlocks_is_a_warning() -> None:
    assert render_status([]) == "WARNING! TRIPPED | NO INTERLOCKS"


def test_a_pipe_never_reaches_the_wire() -> None:
    assert render_status([InterlockStatus("a|b", True, "x | y")]) == (
        "WARNING! TRIPPED | TRIPPED: a/b - x / y"
    )


# Every name and reason here carries the keyword in some case.
_STATES = [
    InterlockStatus("laser Safety", False),
    InterlockStatus("SAFE gate", True, "SafeMode=1 (limit: <1)"),
    InterlockStatus(
        "shutter", True, "not armed: unknown plan 'safe_close'", armed=False
    ),
    InterlockStatus("unsafe-named", True),
    InterlockStatus("plain", False),
]


@pytest.mark.parametrize("mask", range(1, 2 ** len(_STATES)))
def test_the_keyword_appears_only_in_an_all_clear_frame(mask: int) -> None:
    """The one rule, over every non-empty mix: 'safe' (any case) iff all safe, and only as the lead."""
    statuses = [s for i, s in enumerate(_STATES) if mask >> i & 1]
    message = render_status(statuses)
    all_safe = not any(s.unsafe for s in statuses)
    assert ("safe" in message.lower()) is all_safe
    if all_safe:
        assert message.startswith("SAFE | ")
        assert "safe" not in message[len("SAFE") :].lower()


def test_a_keyword_in_a_name_stays_readable() -> None:
    assert render_status([InterlockStatus("laser Safety", False)]) == (
        "SAFE | OK: laser Sa-fety"
    )


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
            assert _read_frame(a) == "SAFE | OK: vacuum"
            assert _read_frame(b) == "SAFE | OK: vacuum"
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
            assert _read_frame(again) == "WARNING! TRIPPED | NO INTERLOCKS"
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
                "WARNING! TRIPPED | NOT ARMED: vacuum - not connected: U_Gauge"
            )
    finally:
        server.stop()
