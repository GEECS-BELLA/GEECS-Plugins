"""What the two interlock windows share: the stream reader and the state colours."""

from __future__ import annotations

import contextlib
import socket
import threading
import time

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget

from geecs_bluesky.interlocks.wire import (
    ALL_SAFE,
    STATE_NO_SIGNAL,
    STATE_NOT_ARMED,
    STATE_OK,
    STATE_TRIPPED,
    TRIPPED,
    read_frame,
)

#: Seconds a connect attempt, or one frame, may take before the reader gives up.
SOCKET_TIMEOUT_S = 2.0
#: Seconds between reconnect attempts.
RECONNECT_S = 2.0

#: State → (background, text).  The only colours either window sets.
STATE_COLOURS: dict[str, tuple[QColor, QColor]] = {
    STATE_OK: (QColor(0, 128, 0), QColor(255, 255, 255)),
    ALL_SAFE: (QColor(0, 128, 0), QColor(255, 255, 255)),
    STATE_TRIPPED: (QColor(192, 0, 0), QColor(255, 255, 255)),
    TRIPPED: (QColor(192, 0, 0), QColor(255, 255, 255)),
    STATE_NOT_ARMED: (QColor(255, 255, 0), QColor(0, 0, 0)),
    STATE_NO_SIGNAL: (QColor(0, 0, 0), QColor(255, 255, 0)),
}


def paint_state(widget: QWidget, state: str) -> None:
    """Fill *widget* (and its labels) with *state*'s colours."""
    background, text = STATE_COLOURS.get(state, STATE_COLOURS[STATE_TRIPPED])
    palette = widget.palette()
    palette.setColor(QPalette.ColorRole.Window, background)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    widget.setPalette(palette)
    widget.setAutoFillBackground(True)


def state_brushes(state: str) -> tuple[QBrush, QBrush]:
    """``(background, foreground)`` brushes for a table cell showing *state*."""
    background, text = STATE_COLOURS.get(state, STATE_COLOURS[STATE_TRIPPED])
    return QBrush(background), QBrush(text)


def scaled_font(factor: float, *, bold: bool = True) -> QFont:
    """The application font, *factor* times larger."""
    font = QFont(QApplication.font())
    font.setPointSizeF(font.pointSizeF() * factor)
    font.setBold(bold)
    return font


def hms(seconds: float) -> str:
    """``HH:MM:SS`` for a non-negative duration."""
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def clock(epoch: float | None = None) -> str:
    """Local ``HH:MM:SS`` of *epoch* (now by default)."""
    return time.strftime("%H:%M:%S", time.localtime(epoch))


def config_address(port_key: str) -> tuple[str, int | None]:
    """``(host, port)`` from ``config.ini [interlocks]``, for the window's defaults.

    The worker's bind-everything host reads as ``localhost``; no port, ``None``.
    """
    try:
        from geecs_core.configs_repo import read_config_entry
    except ImportError:
        return "localhost", None
    host = read_config_entry("interlocks", "status_host") or "localhost"
    if host == "0.0.0.0":
        host = "localhost"
    raw = read_config_entry("interlocks", port_key)
    return host, int(raw) if raw and raw.isdigit() else None


class VerdictPanel(QFrame):
    """The big verdict: a state-coloured sunken panel."""

    def __init__(self, factor: float = 2.6) -> None:
        super().__init__()
        self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self.setLineWidth(2)
        layout = QVBoxLayout(self)
        self.kind = QLabel()
        self.kind.setFont(scaled_font(factor))
        self.kind.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail = QLabel()
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setWordWrap(True)
        layout.addWidget(self.kind)
        layout.addWidget(self.detail)

    def show_verdict(self, kind: str, detail: str) -> None:
        """Show *kind* in its colours with *detail* under it."""
        self.kind.setText(kind)
        self.detail.setText(detail)
        paint_state(self, kind)


class StreamReader(QThread):
    """Read frames from ``host:port`` on a thread, reconnecting until stopped.

    Signals
    -------
    connected(str)
        The ``host:port`` just connected to.
    message(str)
        One frame's message.
    disconnected(str)
        Why the connection ended (or the attempt failed).
    """

    connected = pyqtSignal(str)
    message = pyqtSignal(str)
    disconnected = pyqtSignal(str)

    def __init__(self, host: str, port: int, *, auto_reconnect: bool = True) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.auto_reconnect = auto_reconnect
        self._stopping = threading.Event()
        self._sock: socket.socket | None = None

    def run(self) -> None:
        """Connect, read until the stream ends, and (if asked) try again."""
        while not self._stopping.is_set():
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=SOCKET_TIMEOUT_S
                )
            except OSError as exc:
                self.disconnected.emit(f"cannot connect: {exc}")
            else:
                self._read(sock)
            if not self.auto_reconnect:
                return
            self._stopping.wait(RECONNECT_S)

    def _read(self, sock: socket.socket) -> None:
        self._sock = sock
        self.connected.emit(f"{self.host}:{self.port}")
        try:
            with sock:
                while not self._stopping.is_set():
                    self.message.emit(read_frame(sock))
        except TimeoutError:
            reason = f"no frame for {SOCKET_TIMEOUT_S:g} s"
        except (OSError, ConnectionError) as exc:
            reason = str(exc) or type(exc).__name__
        else:
            reason = "disconnected"
        finally:
            self._sock = None
        if self._stopping.is_set():
            reason = "disconnected"
        self.disconnected.emit(reason)

    def stop(self) -> None:
        """End the connection and the thread (blocks until it has ended)."""
        self._stopping.set()
        sock = self._sock
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        self.wait(int((SOCKET_TIMEOUT_S + 1) * 1000))


__all__ = [
    "STATE_COLOURS",
    "StreamReader",
    "VerdictPanel",
    "clock",
    "config_address",
    "hms",
    "paint_state",
    "scaled_font",
    "state_brushes",
]
