"""Pin ``scripts/mysql_probe.py`` — the handshake-completing DB reachability probe (#790).

MySQL's ``max_connect_errors`` counts connects that drop without completing
the handshake; a bare TCP probe of 3306 is exactly that and blocked the whole
VPN pool (error 1129, 2026-09-04). These tests stand up a stub MySQL server
that speaks just enough of the wire protocol to record what the client sent,
and assert the probe completes the handshake (a HandshakeResponse packet
arrives before the client closes) and classifies the server's answer.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import struct
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "mysql_probe.py"
LIB = REPO_ROOT / "scripts" / "lib" / "net_probes.sh"

pytest.importorskip(
    "mysql.connector",
    reason="the probe needs mysql-connector-python (root env has it via GEECS-Core)",
)

_spec = importlib.util.spec_from_file_location("mysql_probe", PROBE)
assert _spec and _spec.loader
mysql_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mysql_probe)


def _packet(seq: int, payload: bytes) -> bytes:
    return struct.pack("<I", len(payload))[:3] + bytes([seq]) + payload


def _greeting(auth_plugin: bytes = b"mysql_native_password") -> bytes:
    """A protocol-10 HandshakeV10 advertising ``auth_plugin``, no SSL."""
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
        + auth_plugin
        + b"\0"
    )


def _err(errno: int, msg: str) -> bytes:
    return b"\xff" + struct.pack("<H", errno) + b"#HY000" + msg.encode()


class StubMySQL:
    """Accept one client: send a greeting, record its response, answer with ERR ``errno``."""

    def __init__(
        self, errno: int, auth_plugin: bytes = b"mysql_native_password"
    ) -> None:
        self.errno = errno
        self.auth_plugin = auth_plugin
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
            conn.sendall(_packet(0, _greeting(self.auth_plugin)))
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


def _run(port: int, home: Path, timeout: str = "2") -> subprocess.CompletedProcess[str]:
    # An empty HOME means no ~/.config/geecs_python_api/config.ini, so the
    # credential lookup fails fast and the --host/--port fallback (the stub)
    # is what gets probed — never a real lab DB named in this machine's INI.
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home))
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
        env=env,
    )


def test_completes_the_handshake_and_reads_access_denied_as_reachable(
    tmp_path: Path,
) -> None:
    server = StubMySQL(errno=1045)
    result = _run(server.port, tmp_path)
    server.join()
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith(f"ok 127.0.0.1:{server.port} auth-refused")
    # The pin: a HandshakeResponse (sequence id 1, non-empty payload) reached
    # the server before the client went away — not a connect-and-drop.
    assert len(server.received) > 4, "client sent nothing after the greeting"
    assert server.received[3] == 1, "expected sequence id 1 (HandshakeResponse41)"
    assert b"geecs-lab-status-probe" in server.received
    assert "probe user" in result.stdout


def test_unsupported_auth_plugin_is_never_ok(tmp_path: Path) -> None:
    """A greeting the pure connector cannot answer (errno -1, no sqlstate).

    The connector raises NotSupportedError *before* sending a
    HandshakeResponse — a connect-and-drop from the server's side, the exact
    pattern #790 exists to eliminate. It must not read as a server answer.
    """
    server = StubMySQL(errno=1045, auth_plugin=b"authentication_ldap_sasl_client")
    result = _run(server.port, tmp_path)
    server.join()
    assert result.returncode == 5, result.stdout + result.stderr
    assert result.stdout.startswith("unverified ")
    assert not result.stdout.startswith("ok")
    assert server.received == b"", "the client sent something after all?"


def test_host_block_1129_is_its_own_verdict(tmp_path: Path) -> None:
    server = StubMySQL(errno=1129)
    result = _run(server.port, tmp_path)
    server.join()
    assert result.returncode == 3, result.stdout + result.stderr
    assert result.stdout.startswith("blocked")
    assert "FLUSH HOSTS" in result.stdout


def test_nothing_listening_is_down(tmp_path: Path) -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # now guaranteed closed
    result = _run(port, tmp_path)
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


class _Err(Exception):
    """Shape of ``mysql.connector.Error``: ``errno`` (never None, -1 default), ``sqlstate``, ``msg``."""

    def __init__(
        self, errno: int = -1, sqlstate: str | None = None, msg: str = ""
    ) -> None:
        super().__init__(msg)
        self.errno = errno
        self.sqlstate = sqlstate
        self.msg = msg


@pytest.mark.parametrize(
    ("exc", "code", "word"),
    [
        (_Err(1045, "28000", "Access denied"), 0, "ok"),
        # A 3xxx server error (MySQL 8) is still a server ERR packet: sqlstate present.
        (_Err(3159, "HY000", "insecure transport prohibited"), 0, "ok"),
        (_Err(1129, "HY000", "Host is blocked"), 3, "blocked"),
        (_Err(2003, None, "Can't connect"), 1, "down"),
        (_Err(2013, None, "Lost connection"), 1, "down"),
        # Connector-raised: no sqlstate, errno -1 — never a server answer.
        (_Err(-1, None, "Authentication plugin 'x' is not supported"), 5, "unverified"),
    ],
)
def test_only_server_sent_errors_prove_a_completed_handshake(
    exc: _Err, code: int, word: str
) -> None:
    got_code, line = mysql_probe.classify_error(exc, "h:1", "probe user")
    assert got_code == code
    assert line.split(" ", 1)[0] == word


def _fake_geecs_db(monkeypatch: pytest.MonkeyPatch, find) -> None:
    """Route ``from geecs_core.db.geecs_db import _find_credentials`` to ``find``."""
    pkg = types.ModuleType("geecs_core")
    db = types.ModuleType("geecs_core.db")
    mod = types.ModuleType("geecs_core.db.geecs_db")
    mod._find_credentials = find
    for name, m in (
        ("geecs_core", pkg),
        ("geecs_core.db", db),
        ("geecs_core.db.geecs_db", mod),
    ):
        monkeypatch.setitem(sys.modules, name, m)


def test_stalled_credential_lookup_falls_back_to_the_probe_user_in_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configurations.INI lives on the data share: a stalled SMB read must not become rc 137 upstream."""
    release = threading.Event()

    def stalled() -> dict:
        release.wait(30)
        return {}

    _fake_geecs_db(monkeypatch, stalled)
    try:
        kwargs, who = mysql_probe._credentials("fallback-host", 3307, creds_timeout=0.3)
    finally:
        release.set()
    assert who == "probe user"
    assert kwargs["host"] == "fallback-host" and kwargs["port"] == 3307
    assert kwargs["user"] == mysql_probe.PROBE_USER


def test_ini_target_wins_over_the_callers_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """[Database] ipaddress/port is the server the real clients use (P2, #794 review)."""
    _fake_geecs_db(
        monkeypatch,
        lambda: {
            "host": "db-from-ini",
            "port": 3316,
            "database": "geecs",
            "user": "u",
            "password": "p",
        },
    )
    kwargs, who = mysql_probe._credentials("tiled-host", 3306, creds_timeout=1)
    assert who == "geecs credentials"
    assert (kwargs["host"], kwargs["port"], kwargs["user"]) == (
        "db-from-ini",
        3316,
        "u",
    )


def test_failed_credential_lookup_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> dict:
        raise FileNotFoundError("no Configurations.INI")

    _fake_geecs_db(monkeypatch, boom)
    kwargs, who = mysql_probe._credentials("h", 1, creds_timeout=1)
    assert who == "probe user" and kwargs["host"] == "h"
