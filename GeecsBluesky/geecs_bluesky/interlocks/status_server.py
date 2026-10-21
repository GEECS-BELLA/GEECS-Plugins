"""The interlock status stream: the TCP feed the LabVIEW side reads.

The retired ``geecs_python_api.controls.interlocks`` server's framing (a
4-byte big-endian length, then UTF-8; every :data:`SEND_INTERVAL_S` to
each client; parts joined with ``" | "``), with the message led by the
overall verdict, because the master-control VI decides by looking for the
keyword ``SAFE``::

    SAFE | OK: <name> | OK: <name>                          every interlock safe
    WARNING! TRIPPED | TRIPPED: <name> - <reason> | OK: <name>
    WARNING! TRIPPED | NOT ARMED: <name> - <why>            could not be armed
    WARNING! TRIPPED | NO INTERLOCKS                        none registered

**The one rule** (pinned by the tests over every mix of states): the word
"safe", in any case, appears only as the leading ``SAFE`` of an all-clear
frame — so any keyword search, however the VI does it, is right.  The
retired server's ``SAFE: <name>`` / ``<name> UNSAFE`` parts broke that
(``UNSAFE`` contains ``SAFE``; a mixed frame still carried ``SAFE:``),
which is why the parts changed.  Inside a part a "safe" in a name or
reason is sent as ``sa-fe``, and a ``|`` as ``/``.

It reads the registry's cached verdicts, never a PV, for the worker
environment's lifetime; ``config.ini [interlocks] status_host /
status_port`` say where (no port, no server).
"""

from __future__ import annotations

import logging
import re
import socket
import struct
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from geecs_bluesky.interlocks.install import InterlockRegistry, InterlockStatus

logger = logging.getLogger(__name__)

#: Seconds between two frames to one client (the retired server's cadence).
SEND_INTERVAL_S = 0.1
#: A client that stops reading for this long is dropped (``sendall`` would
#: otherwise block its thread for good once the socket buffer fills).
SEND_TIMEOUT_S = 5.0
#: How often the accept loop wakes to notice :meth:`InterlockStatusServer.stop`.
ACCEPT_POLL_S = 0.5
#: The bind address when ``status_host`` is not set: every interface.
DEFAULT_HOST = "0.0.0.0"

#: The verdict that leads every frame.
ALL_SAFE = "SAFE"
TRIPPED = "WARNING! TRIPPED"
NO_INTERLOCKS = "NO INTERLOCKS"

_SAFE_WORD = re.compile("safe", re.IGNORECASE)
_NOT_ARMED_PREFIX = "not armed: "


def _clean(text: str) -> str:
    """*text* safe to put in a part: no separator, no keyword ("Safety" → "Sa-fety")."""
    text = text.replace("|", "/")
    return _SAFE_WORD.sub(lambda m: f"{m.group(0)[:2]}-{m.group(0)[2:]}", text)


def _part(status: InterlockStatus) -> str:
    name = _clean(status.name)
    if not status.unsafe:
        return f"OK: {name}"
    if not status.armed:
        why = status.diagnostic.removeprefix(_NOT_ARMED_PREFIX)
        return f"NOT ARMED: {name} - {_clean(why)}"
    if status.diagnostic:
        return f"TRIPPED: {name} - {_clean(status.diagnostic)}"
    return f"TRIPPED: {name}"


def render_status(statuses: Sequence[InterlockStatus]) -> str:
    """The message one frame carries: the overall verdict, then one part per interlock."""
    if not statuses:
        return f"{TRIPPED} | {NO_INTERLOCKS}"
    verdict = TRIPPED if any(s.unsafe for s in statuses) else ALL_SAFE
    return " | ".join([verdict, *(_part(s) for s in statuses)])


def frame(message: str) -> bytes:
    """*message* with its 4-byte big-endian length prefix."""
    payload = message.encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


class InterlockStatusServer:
    """Serve :func:`render_status` of *snapshot* to every connected client.

    Parameters
    ----------
    snapshot :
        Returns the current statuses (the registry's ``snapshot``).
    host, port :
        Where to listen; port ``0`` picks a free one (tests).
    interval :
        Seconds between frames to one client.
    """

    def __init__(
        self,
        snapshot: Callable[[], Sequence[InterlockStatus]],
        host: str = DEFAULT_HOST,
        port: int = 0,
        *,
        interval: float = SEND_INTERVAL_S,
    ) -> None:
        self._snapshot = snapshot
        self._host = host
        self._port = port
        self._interval = interval
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        """The bound ``(host, port)``; the requested one before :meth:`start`."""
        if self._socket is not None:
            host, port = self._socket.getsockname()[:2]
            return host, port
        return self._host, self._port

    def start(self) -> None:
        """Bind and start accepting (``OSError`` if the port is taken)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self._host, self._port))
            sock.listen()
        except OSError:
            sock.close()
            raise
        sock.settimeout(ACCEPT_POLL_S)
        self._socket = sock
        self._thread = threading.Thread(
            target=self._accept_loop, name="interlock-status", daemon=True
        )
        self._thread.start()
        logger.info("interlock status stream on %s:%d", *self.address)

    def stop(self) -> None:
        """Stop accepting and close every client (each within one interval)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(ACCEPT_POLL_S * 4)
        if self._socket is not None:
            self._socket.close()

    def _accept_loop(self) -> None:
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._socket.accept()
            except TimeoutError:
                continue
            except OSError:
                if not self._stop.is_set():
                    logger.exception("interlock status stream: accept failed")
                return
            threading.Thread(
                target=self._serve,
                args=(conn, addr),
                name=f"interlock-status-{addr[0]}",
                daemon=True,
            ).start()

    def _serve(self, conn: socket.socket, addr: tuple) -> None:
        logger.info("interlock status stream: client %s connected", addr[0])
        conn.settimeout(SEND_TIMEOUT_S)
        try:
            with conn:
                while not self._stop.is_set():
                    conn.sendall(frame(render_status(self._snapshot())))
                    self._stop.wait(self._interval)
        except OSError:
            pass  # the client went away (or stopped reading): drop it
        logger.info("interlock status stream: client %s disconnected", addr[0])


def start_status_server(
    registry: InterlockRegistry | None, *, config_path: Path | None = None
) -> InterlockStatusServer | None:
    """Start the stream from ``config.ini [interlocks]``; ``None`` when not configured.

    Never raises: a malformed port or a taken one is logged at ERROR and
    the worker runs without the stream.  With no registry (no
    ``interlocks.yaml``) the stream still runs and says ``NO INTERLOCKS``
    under a ``WARNING! TRIPPED`` verdict — never ``SAFE``.
    """
    from geecs_core.configs_repo import read_config_entry

    raw_port = read_config_entry("interlocks", "status_port", config_path)
    if raw_port is None:
        logger.info(
            "interlock status stream off (no [interlocks] status_port in config.ini)"
        )
        return None
    host = read_config_entry("interlocks", "status_host", config_path) or DEFAULT_HOST
    if not raw_port.isdigit():
        logger.error(
            "interlock status stream off: [interlocks] status_port=%r is not a port",
            raw_port,
        )
        return None
    port = int(raw_port)
    server = InterlockStatusServer(
        registry.snapshot if registry is not None else list, host, port
    )
    failure: OSError | None = None
    try:
        server.start()
    except OSError as exc:
        failure = exc
    if failure is not None:
        logger.error(
            "interlock status stream off: cannot listen on %s:%d (%s)",
            host,
            port,
            failure,
        )
        return None
    return server


__all__ = [
    "ALL_SAFE",
    "NO_INTERLOCKS",
    "TRIPPED",
    "SEND_INTERVAL_S",
    "InterlockStatusServer",
    "frame",
    "render_status",
    "start_status_server",
]
