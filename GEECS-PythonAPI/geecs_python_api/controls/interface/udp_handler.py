"""UDP command, send, ack utilities and a lightweight UDP server for GEECS devices (logging-based, no ErrorAPI)."""

from __future__ import annotations

import logging
import select
import socket
import time
from datetime import datetime as dtime
from threading import Event, Thread
from typing import Optional, TYPE_CHECKING

import geecs_python_api.controls.interface.message_handling as mh
from geecs_python_api.controls.api_defs import ThreadInfo

if TYPE_CHECKING:
    from geecs_python_api.controls.devices import GeecsDevice

logger = logging.getLogger(__name__)


class UdpHandler:
    """Create a UDP client socket bound to an ephemeral port and manage command/ack flow."""

    def __init__(self, owner: "GeecsDevice"):
        """Initialize sockets and a companion UdpServer for slow-command acknowledgments."""
        self.buffer_size: int = 1024
        self.sock_cmd: Optional[socket.socket] = None
        self.bounded_cmd: bool = False
        self.mc_port: int = owner.mc_port
        self.port_cmd: int = -1
        self.port_exe: int = -1
        self.cmd_checker: Optional[UdpServer] = None  # <-- don’t bind yet

        try:
            self.sock_cmd = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
            )
            self.sock_cmd.settimeout(15.0)
            self.sock_cmd.bind(("", 0))
            self.port_cmd = self.sock_cmd.getsockname()[1]
            self.port_exe = self.port_cmd + 1
            self.bounded_cmd = True
            self.cmd_checker = UdpServer(owner=owner, port=self.port_exe)
            logger.info(
                "UDP cmd socket bound on port %s; exe port %s",
                self.port_cmd,
                self.port_exe,
            )
        except Exception:
            logger.exception("failed to initialize UdpHandler sockets")
            self.port_cmd = self.port_exe = -1

    def close(self) -> None:
        """Close underlying sockets."""
        try:
            self.close_sock_cmd()
        except Exception:
            logger.debug("ignored error during UdpHandler.close()", exc_info=True)

    def close_sock_cmd(self) -> None:
        """Close the command socket and reset state."""
        try:
            if self.sock_cmd:
                self.sock_cmd.close()
        except Exception:
            logger.debug("ignored error while closing UDP cmd socket", exc_info=True)
        finally:
            self.sock_cmd = None
            self.port_cmd = -1
            self.bounded_cmd = False

    def send_cmd(self, ipv4: tuple[str, int] = ("", -1), msg: str = "") -> bool:
        """Send an ASCII command datagram to (ip, port)."""
        if not self.sock_cmd:
            logger.error("send_cmd called with no socket initialized")
            return False
        try:
            self.sock_cmd.sendto(msg.encode("ascii"), ipv4)
            logger.debug('sent UDP cmd "%s" to %s:%s', msg, ipv4[0], ipv4[1])
            return True
        except Exception:
            logger.exception("failed to send UDP message")
            return False

    def ack_cmd(
        self, sock: Optional[socket.socket] = None, timeout: Optional[float] = 5.0
    ) -> bool:
        """Wait for an ack ('accepted' or 'ok') on `sock` (defaults to cmd socket) within timeout."""
        accepted = False
        sock = sock or self.sock_cmd
        if sock is None:
            logger.error("ack_cmd called with no socket")
            return False

        try:
            ready = select.select([sock], [], [], timeout)
            if ready[0]:
                geecs_str = sock.recv(self.buffer_size)
                geecs_ans = geecs_str.decode("ascii").split(">>")[-1]
                accepted = (geecs_ans == "accepted") or (geecs_ans == "ok")
                logger.debug("ack_cmd received %r -> accepted=%s", geecs_ans, accepted)
            else:
                logger.warning(
                    "socket not ready to receive ack within %.2fs",
                    0 if timeout is None else timeout,
                )
        except Exception:
            logger.exception("failed to read UDP acknowledge message")
        return accepted

    def send_scan_cmd(self, cmd: str, client_ip: str = "localhost") -> bool:
        """Send a scan command to MC port+2 and await an acknowledgement."""
        accepted: bool = False
        sock_mc: Optional[socket.socket] = None
        try:
            sock_mc = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
            )
            sock_mc.settimeout(5.0)
            sock_mc.bind(("", 0))
            sock_mc.sendto(cmd.encode("ascii"), (client_ip, self.mc_port + 2))
            logger.debug(
                'sent scan cmd "%s" to %s:%s', cmd, client_ip, self.mc_port + 2
            )
            accepted = self.ack_cmd(sock=sock_mc, timeout=5.0)
        except Exception:
            logger.exception("failed to send/ack scan command")
        finally:
            try:
                if sock_mc:
                    sock_mc.close()
            except Exception:
                logger.debug("ignored error while closing scan socket", exc_info=True)
        return accepted

    def register_handler(self) -> bool:
        """Enable server callbacks if owner is valid."""
        if self.cmd_checker is None:
            return False
        self.cmd_checker.subscribed = bool(self.cmd_checker.owner.is_valid())
        return self.cmd_checker.subscribed

    def unregister_handler(self) -> None:
        """Disable server callbacks."""
        if self.cmd_checker is not None:
            self.cmd_checker.subscribed = False


class UdpServer:
    """Minimal UDP server to receive execution confirmations and dispatch to the device."""

    def __init__(self, owner: "GeecsDevice", port: int = -1):
        """Bind a UDP socket to `port` (must be provided by UdpHandler)."""
        self.owner: "GeecsDevice" = owner
        self.subscribed: bool = False
        self.buffer_size: int = 1024
        self.port: int = port
        self.sock: Optional[socket.socket] = None
        self.bounded: bool = False

        try:
            self.sock = socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP
            )
            self.sock.settimeout(5.0)
            self.sock.bind(("", self.port))
            self.bounded = True
            logger.info("UDP server bound on port %s", self.port)
        except Exception:
            logger.exception("failed to bind UDP server")
            self.sock = None
            self.port = -1
            self.bounded = False

    def close(self) -> None:
        """Close the UDP server socket."""
        try:
            self.close_sock_exe()
        except Exception:
            logger.debug("ignored error during UdpServer.close()", exc_info=True)

    def close_sock_exe(self) -> None:
        """Close the execution socket and reset state."""
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            logger.debug("ignored error while closing UDP server socket", exc_info=True)
        finally:
            self.sock = None
            self.port = -1
            self.bounded = False

    def listen(
        self,
        cmd_tag: str,
        stop_event: Optional[Event] = None,
        timeout: Optional[float] = 1.0,
    ) -> str:
        """Block until a message arrives or timeout/stop_event triggers; dispatch to owner if subscribed."""
        if not self.sock:
            logger.error("listen called with no socket bound")
            return ""

        geecs_ans = ""
        stamp = ""
        t0 = time.monotonic()
        eff_timeout = 0.2 if (timeout is None) else min(0.2, max(0.0, timeout / 10.0))

        while True:
            try:
                rlist, _, _ = select.select([self.sock], [], [], eff_timeout)
                if rlist:
                    data, _addr = self.sock.recvfrom(self.buffer_size)
                    geecs_ans = data.decode("ascii", errors="replace")
                    stamp = dtime.now().isoformat(timespec="milliseconds")
                    break
            except socket.timeout:
                # python's select + recv timeout pattern uses select timeout; we emulate total timeout here.
                pass
            except Exception:
                logger.exception("failed to read UDP message")
                return ""

            # total-timeout / stop checks
            if timeout is not None and (time.monotonic() - t0) >= timeout:
                logger.warning('command timed out for tag "%s"', cmd_tag)
                return ""
            if stop_event is not None and stop_event.is_set():
                stop_event.clear()
                logger.debug("listen aborted by stop_event for tag %s", cmd_tag)
                return ""

        # Dispatch/publish
        try:
            net_msg = mh.NetworkMessage(
                tag=cmd_tag, stamp=stamp, msg=geecs_ans, err=None
            )
            if self.subscribed:
                try:
                    self.owner.handle_response(net_msg)
                except Exception:
                    logger.exception("owner.handle_response raised")
        except Exception:
            logger.exception("failed to construct/publish UDP NetworkMessage")

        # Trigger device dequeue in background
        try:
            Thread(
                target=self.owner.dequeue_command,
                name=f"UdpServer-dequeue[{cmd_tag}]",
                daemon=True,
            ).start()
        except Exception:
            logger.exception("failed to start dequeue_command thread")

        return geecs_ans

    def wait_for_exe(
        self, cmd_tag: str, timeout: Optional[float] = 5.0, sync: bool = False
    ) -> ThreadInfo:
        """Wait for an execution message synchronously or return a (thread, stop_event) for async wait."""
        exe_thread: Optional[Thread] = None
        stop_event: Optional[Event] = None
        try:
            if sync:
                self.listen(cmd_tag, timeout=timeout)
            else:
                exe_thread, stop_event = self.create_thread(cmd_tag, timeout)
        except Exception:
            logger.exception("failed waiting for command execution")
        return exe_thread, stop_event

    def create_thread(self, cmd_tag: str, timeout: Optional[float] = 5.0) -> ThreadInfo:
        """Create a listening thread and its stop_event for the given tag/timeout."""
        stop_event = Event()
        exe_thread = Thread(
            target=self.listen, args=(cmd_tag, stop_event, timeout), daemon=True
        )
        return exe_thread, stop_event
