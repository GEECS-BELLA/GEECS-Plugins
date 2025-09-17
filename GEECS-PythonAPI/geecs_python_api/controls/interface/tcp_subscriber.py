"""TCP subscription client for GEECS devices with logging-based error handling."""

from __future__ import annotations

import logging
import select
import socket
import struct
from datetime import datetime as dtime
from threading import Event, Thread
from typing import Callable, Optional, TYPE_CHECKING

import geecs_python_api.controls.interface.message_handling as mh

if TYPE_CHECKING:
    from geecs_python_api.controls.devices import GeecsDevice

logger = logging.getLogger(__name__)


class TcpSubscriber:
    """Manage a TCP connection to a device, spawn a listener, and dispatch messages."""

    def __init__(self, owner: "GeecsDevice"):
        """Initialize with owning device and default connection state."""
        self.owner: "GeecsDevice" = owner
        self.subscribed: bool = False

        self.sock: Optional[socket.socket] = None
        self.unsubscribe_event = Event()
        self.host: str = ""
        self.port: int = -1
        self.connected: bool = False

        self.message_callback: Optional[Callable[[str], None]] = None
        self.api_shotnumber: int = 0

        self._listener: Optional[Thread] = None

    def set_message_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Set a callable invoked for every raw message received."""
        self.message_callback = callback

    def close(self) -> None:
        """Unsubscribe and close the socket."""
        try:
            self.unsubscribe()
            self.close_sock()
        except Exception:
            logger.debug("ignored error during close()", exc_info=True)

    def connect(self) -> bool:
        """Open a TCP connection to the owner device."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.owner.dev_ip, self.owner.dev_port))
            self.host = self.owner.dev_ip
            self.port = self.owner.dev_port
            self.connected = True
            logger.info("connected to %s:%s", self.host, self.port)
        except ConnectionRefusedError:
            logger.error(
                "connection refused to %s:%s",
                self.owner.dev_ip,
                self.owner.dev_port,
                exc_info=True,
            )
            self._reset_conn_state()
        except (TimeoutError, InterruptedError):
            logger.error(
                "error connecting TCP client for %s",
                getattr(self.owner, "get_name", lambda: "?")(),
                exc_info=True,
            )
            self._reset_conn_state()
        except Exception:
            logger.exception("unexpected error creating TCP connection")
            self._reset_conn_state()
        return self.connected

    def is_connected(self) -> bool:
        """Return True if socket is connected."""
        return self.connected

    def close_sock(self) -> None:
        """Close the underlying socket and reset connection state."""
        try:
            if self.sock:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    # Shutdown may fail if already closed or never connected; ignore.
                    pass
                self.sock.close()
        except Exception:
            logger.debug("ignored error while closing socket", exc_info=True)
        finally:
            self._reset_conn_state()

    def _reset_conn_state(self) -> None:
        """Reset connection-related attributes."""
        self.sock = None
        self.host = ""
        self.port = -1
        self.connected = False

    def register_handler(self) -> bool:
        """Mark subscriber active if owner is valid."""
        self.subscribed = bool(getattr(self.owner, "is_valid", lambda: False)())
        return self.subscribed

    def unregister_handler(self) -> None:
        """Mark subscriber as inactive."""
        self.subscribed = False

    def subscribe(self, cmd: str) -> bool:
        """Send a subscription command and start the async listener."""
        if self.is_connected():
            self.close()

        if not self.connect():
            logger.warning('cannot subscribe "%s": not connected', cmd)
            return False

        try:
            subscription_str = f"Wait>>{cmd}".encode("ascii")
            size_pack = struct.pack(">i", len(subscription_str))
            assert self.sock is not None  # for type checkers
            self.sock.sendall(size_pack + subscription_str)

            self.unsubscribe_event.clear()
            self._listener = Thread(
                target=self.async_listener,
                name=f"TcpSubscriber[{self.owner.get_name()}]",
                daemon=True,
            )
            self._listener.start()
            logger.info('subscribed with command "%s"', cmd)
            return True
        except Exception:
            logger.exception('failed to subscribe to variable(s) "%s"', cmd)
            self.close_sock()
            return False

    def unsubscribe(self) -> None:
        """Signal the async listener to stop."""
        self.unsubscribe_event.set()

    def async_listener(self) -> None:
        """Continuously read framed messages (4-byte big-endian length + ASCII payload) and dispatch."""
        if self.sock is None:
            logger.debug("async_listener started without a socket")
            return

        try:
            self.sock.settimeout(0.5)
        except Exception:
            logger.debug("failed to set socket timeout", exc_info=True)

        while True:
            try:
                # Break promptly when asked to unsubscribe.
                if self.unsubscribe_event.is_set():
                    self.unsubscribe_event.clear()
                    logger.info("unsubscribe received; stopping listener")
                    return

                # Wait for readability.
                rlist, _, _ = select.select([self.sock], [], [], 0.05)
                if not rlist:
                    continue

                # Read message length.
                header = self._recvn(4)
                if header is None:
                    logger.debug("socket closed while reading header")
                    return
                msg_len = struct.unpack(">i", header)[0]
                if msg_len <= 0:
                    logger.debug("received non-positive msg length: %s", msg_len)
                    continue

                # Read message bytes.
                payload = self._recvn(msg_len)
                if payload is None:
                    logger.debug("socket closed while reading payload")
                    return
                this_msg = payload.decode("ascii", errors="replace")

                # Fire optional raw-message callback.
                if self.message_callback is not None:
                    try:
                        self.message_callback(this_msg)
                    except Exception:
                        logger.exception("message_callback raised")

                # Forward to owner if registered.
                if self.subscribed:
                    stamp = dtime.now().isoformat(timespec="milliseconds")
                    net_msg = mh.NetworkMessage(
                        tag=self.owner.get_name(), stamp=stamp, msg=this_msg, err=None
                    )
                    try:
                        self.owner.handle_subscription(net_msg)
                    except Exception:
                        logger.exception("owner.handle_subscription raised")

            except socket.timeout:
                continue
            except (OSError, ConnectionResetError):
                logger.exception("socket error in async_listener")
                return
            except Exception:
                logger.exception("unexpected error in async_listener")
                return

    def _recvn(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes or return None on EOF/error."""
        assert self.sock is not None
        chunks: list[bytes] = []
        remaining = n
        try:
            while remaining > 0:
                chunk = self.sock.recv(remaining)
                if not chunk:
                    return None
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)
        except Exception:
            logger.debug("recvn failed", exc_info=True)
            return None
