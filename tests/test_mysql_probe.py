"""Pin ``scripts/mysql_probe.py`` — the handshake-completing DB reachability probe (#790).

MySQL's ``max_connect_errors`` counts connects that drop without completing
the handshake; a bare TCP probe of 3306 is exactly that and blocked the whole
VPN pool (error 1129, 2026-09-04). These tests stand up a stub MySQL server
that speaks just enough of the wire protocol to record what the client sent,
and assert the probe completes the handshake (a HandshakeResponse packet
arrives before the client closes) and classifies the server's answer.
"""

from __future__ import annotations

import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "mysql_probe.py"
LIB = REPO_ROOT / "scripts" / "lib" / "net_probes.sh"

pytest.importorskip(
    "mysql.connector",
    reason="the probe needs mysql-connector-python (root env has it via GEECS-Core)",
)


def _packet(seq: int, payload: bytes) -> bytes:
    return struct.pack("<I", len(payload))[:3] + bytes([seq]) + payload


def _greeting() -> bytes:
    """A protocol-10 HandshakeV10 with mysql_native_password, no SSL."""
    cap = 0x0001 | 0x0008 | 0x0200 | 0x8000 | 0x00080000 | 0x00200000
    return (
        bytes([10])
        + b"8.0.0-stub\0"
        + struct.pack("<I", 7)
        + b"12345678"
        + b"\0"
        + struct.pack("<H", cap & 0xFFFF)
        + bytes([0x21])
        + struct.pack("<H", 0x0002)
        + struct.pack("<H", cap >> 16)
        + bytes([21])
        + b"\0" * 10
        + b"901234567890\0"
        + b"mysql_native_password\0"
    )


def _err(errno: int, msg: str) -> bytes:
    return b"\xff" + struct.pack("<H", errno) + b"#HY000" + msg.encode()


class StubMySQL:
    """Accept one client: send a greeting, record its response, answer with ERR ``errno``."""

    def __init__(self, errno: int) -> None:
        self.errno = errno
        self.received = b""
        self.closed_cleanly = False
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self._sock.settimeout(10)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        conn, _ = self._sock.accept()
        conn.settimeout(5)
        with conn:
            conn.sendall(_packet(0, _greeting()))
            try:
                header = conn.recv(4)
                if len(header) == 4:
                    length = int.from_bytes(header[:3], "little")
                    body = b""
                    while len(body) < length:
                        chunk = conn.recv(length - len(body))
                        if not chunk:
                            break
                        body += chunk
                    self.received = header + body
                    conn.sendall(_packet(2, _err(self.errno, "stub says no")))
                    # A clean client close reads as EOF; an abort as a reset.
                    try:
                        self.closed_cleanly = conn.recv(64) in (
                            b"",
                            b"\x01\x00\x00\x00\x01",
                        )
                    except OSError:
                        self.closed_cleanly = False
            except OSError:
                pass
        self._sock.close()

    def join(self) -> None:
        self._thread.join(timeout=10)


def _run(port: int, timeout: str = "2") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--timeout",
            timeout,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_completes_the_handshake_and_reads_access_denied_as_reachable() -> None:
    server = StubMySQL(errno=1045)
    result = _run(server.port)
    server.join()
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("ok auth-refused")
    # The pin: a HandshakeResponse (sequence id 1, non-empty payload) reached
    # the server before the client went away — not a connect-and-drop.
    assert len(server.received) > 4, "client sent nothing after the greeting"
    assert server.received[3] == 1, "expected sequence id 1 (HandshakeResponse41)"
    assert b"geecs-lab-status-probe" in server.received or len(server.received) > 36


def test_host_block_1129_is_its_own_verdict() -> None:
    server = StubMySQL(errno=1129)
    result = _run(server.port)
    server.join()
    assert result.returncode == 3, result.stdout + result.stderr
    assert result.stdout.startswith("blocked")
    assert "FLUSH HOSTS" in result.stdout


def test_nothing_listening_is_down() -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # now guaranteed closed
    result = _run(port)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.startswith("down")


def test_no_bare_tcp_probe_of_the_db_port_anywhere() -> None:
    """The /dev/tcp probe lives once, in the shared lib, and refuses 3306."""
    for script in ("lab_status.sh", "fleet_status.sh"):
        text = (REPO_ROOT / "scripts" / script).read_text()
        assert "<>/dev/tcp" not in text, f"{script} carries its own /dev/tcp probe"
    lib = LIB.read_text()
    assert "<>/dev/tcp" in lib
    assert "3306" in lib and "mysql_probe.py" in lib


@pytest.mark.skipif(sys.platform == "win32", reason="bash library")
def test_shared_port_open_refuses_the_mysql_port() -> None:
    result = subprocess.run(
        ["bash", "-c", f'source "{LIB}"; TCP_TIMEOUT=1 port_open 127.0.0.1 3306'],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 2
    assert "mysql_probe" in result.stderr
