"""The interlock client: the status stream LabVIEW reads, in a window.

``geecs-interlock-client [--host H] [--port P] [--compact]`` connects to
the worker's status stream (``config.ini [interlocks] status_port`` by
default) and shows the verdict, one row per interlock, the verdict's
history and the raw frame.  No frame for :data:`NO_SIGNAL_S` reads as
**NO SIGNAL** — never as the last verdict seen, which may have been SAFE.
The compact view is a small always-on-top strip with the same verdict.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from dataclasses import dataclass

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QGuiApplication
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from geecs_bluesky.interlocks.gui.common import (
    StreamReader,
    VerdictPanel,
    clock,
    config_address,
    hms,
    state_brushes,
)
from geecs_bluesky.interlocks.wire import (
    ALL_SAFE,
    STATE_NO_SIGNAL,
    STATE_OK,
    TRIPPED,
    ParsedStatus,
    parse_status,
)

#: Seconds without a frame before the client shows NO SIGNAL.
NO_SIGNAL_S = 1.0
#: How often the window re-checks the frame age, milliseconds.
TICK_MS = 250
#: Seconds between two beeps while not safe and not silenced.
BEEP_EVERY_S = 1.0


@dataclass(frozen=True)
class Verdict:
    """What the verdict panel shows."""

    #: ``SAFE``, ``WARNING! TRIPPED`` or ``NO SIGNAL``.
    kind: str
    #: The line under it.
    detail: str
    #: The non-OK interlocks, ``"<STATE> <name>"`` each.
    lines: tuple[str, ...] = ()
    #: The non-OK interlocks' names.
    names: tuple[str, ...] = ()


def judge(parsed: ParsedStatus | None, frame_age_s: float | None) -> Verdict:
    """The verdict to show for the latest frame and how old it is.

    ``frame_age_s`` is ``None`` when no frame has arrived on this connection.
    """
    if parsed is None or frame_age_s is None or frame_age_s > NO_SIGNAL_S:
        if frame_age_s is None:
            return Verdict(STATE_NO_SIGNAL, "Not receiving frames. Treat as not safe.")
        return Verdict(
            STATE_NO_SIGNAL, f"No frame for {frame_age_s:.1f} s. Treat as not safe."
        )
    if parsed.safe:
        n = len(parsed.parts)
        return Verdict(ALL_SAFE, f"{n} of {n} interlocks OK")
    if parsed.no_interlocks:
        return Verdict(TRIPPED, "No interlocks registered on the worker.")
    bad = [p for p in parsed.parts if p.state != STATE_OK]
    lines = tuple(f"{p.state} {p.name}" for p in bad)
    names = tuple(p.name for p in bad)
    return Verdict(TRIPPED, ", ".join(names) or "unrecognised frame", lines, names)


class CompactWindow(QWidget):
    """The always-on-top strip: the verdict and the interlocks that are not OK."""

    full_window = pyqtSignal()

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Interlocks")
        layout = QVBoxLayout(self)
        self.panel = VerdictPanel(factor=1.8)
        self.lines = QLabel()
        button = QPushButton("Full window")
        button.clicked.connect(self.full_window.emit)
        layout.addWidget(self.panel)
        layout.addWidget(self.lines)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignRight)

    def show_verdict(self, verdict: Verdict, detail: str) -> None:
        """Mirror the main window's verdict."""
        self.panel.show_verdict(verdict.kind, detail)
        self.lines.setText("\n".join(verdict.lines))
        self.lines.setVisible(bool(verdict.lines))

    def closeEvent(self, event: QCloseEvent) -> None:
        """Closing the strip brings the full window back (never no window at all)."""
        self.full_window.emit()
        event.accept()


class ClientWindow(QMainWindow):
    """The full client window (see the module docstring)."""

    def __init__(self, host: str = "localhost", port: int | None = None) -> None:
        super().__init__()
        self.setWindowTitle("GEECS Interlock Client")
        self.reader: StreamReader | None = None
        self.parsed: ParsedStatus | None = None
        self.last_frame_at: float | None = None
        self.connected_at: float | None = None
        self.frame_times: deque[float] = deque(maxlen=50)
        self.verdict: Verdict | None = None
        self.not_safe_since: float | None = None
        self.silenced = False
        self.last_beep = 0.0
        self.history: list[tuple[str, str, str]] = []
        self.compact = CompactWindow()
        self.compact.full_window.connect(self.show_full)
        self._build(host, port)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)
        self.tick()

    # -- layout -------------------------------------------------------------

    def _build(self, host: str, port: int | None) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self._connection_box(host, port))
        self.panel = VerdictPanel()
        self.silence = QPushButton("Silence")
        self.silence.clicked.connect(self.silence_alarm)
        self.panel.layout().addWidget(
            self.silence, alignment=Qt.AlignmentFlag.AlignRight
        )
        layout.addWidget(self.panel)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["State", "Interlock", "Reason"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.table, stretch=2)
        layout.addWidget(self._history_box(), stretch=1)
        raw_box = QGroupBox("Raw frame")
        raw_layout = QVBoxLayout(raw_box)
        self.raw = QPlainTextEdit(readOnly=True)
        self.raw.setMaximumHeight(70)
        raw_layout.addWidget(self.raw)
        layout.addWidget(raw_box)
        layout.addLayout(self._options_row())
        self.setCentralWidget(central)
        self.status_text = QLabel()
        self.uptime = QLabel()
        self.statusBar().addWidget(self.status_text, 1)
        self.statusBar().addPermanentWidget(self.uptime)
        self.resize(760, 820)

    def _connection_box(self, host: str, port: int | None) -> QGroupBox:
        box = QGroupBox("Connection")
        rows = QVBoxLayout(box)
        row = QHBoxLayout()
        self.host = QLineEdit(host)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(port or 1)
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        host_label = QLabel("&Host:")
        host_label.setBuddy(self.host)
        port_label = QLabel("&Port:")
        port_label.setBuddy(self.port)
        for widget in (
            host_label,
            self.host,
            port_label,
            self.port,
            self.connect_button,
        ):
            row.addWidget(widget)
        row.setStretch(1, 1)
        rows.addLayout(row)
        self.connection_text = QLabel()
        rows.addWidget(self.connection_text)
        return box

    def _history_box(self) -> QGroupBox:
        box = QGroupBox("Verdict changes")
        layout = QVBoxLayout(box)
        self.history_view = QPlainTextEdit(readOnly=True)
        export = QPushButton("Export CSV...")
        export.clicked.connect(self.export_history)
        layout.addWidget(self.history_view)
        layout.addWidget(export, alignment=Qt.AlignmentFlag.AlignRight)
        return box

    def _options_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.sound = QCheckBox("Sound on trip")
        self.sound.setChecked(True)
        self.on_top = QCheckBox("Always on top")
        self.on_top.toggled.connect(self.set_on_top)
        self.reconnect = QCheckBox("Auto-reconnect")
        self.reconnect.setChecked(True)
        self.reconnect.toggled.connect(self.set_auto_reconnect)
        compact = QPushButton("Compact view")
        compact.clicked.connect(self.show_compact)
        for widget in (self.sound, self.on_top, self.reconnect):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(compact)
        return row

    # -- connection ---------------------------------------------------------

    def toggle_connection(self) -> None:
        """Connect to the host and port in the fields, or disconnect."""
        if self.reader is not None:
            self.disconnect_stream()
        else:
            self.connect_stream()

    def connect_stream(self) -> None:
        """Start reading the stream at the fields' address."""
        self.reader = StreamReader(
            self.host.text().strip(),
            self.port.value(),
            auto_reconnect=self.reconnect.isChecked(),
        )
        self.reader.connected.connect(self.on_connected)
        self.reader.message.connect(self.on_message)
        self.reader.disconnected.connect(self.on_disconnected)
        reader = self.reader
        reader.finished.connect(lambda: self.on_reader_finished(reader))
        reader.start()
        self.connect_button.setText("Disconnect")
        self.host.setEnabled(False)
        self.port.setEnabled(False)
        self.status_text.setText(f"Connecting to {self.reader.host}:{self.reader.port}")

    def disconnect_stream(self) -> None:
        """Stop reading (the verdict turns NO SIGNAL)."""
        reader = self.reader
        if reader is not None:
            reader.stop()
            self.on_reader_finished(reader)

    def on_reader_finished(self, reader: StreamReader) -> None:
        """*reader* has ended (asked to, or no auto-reconnect)."""
        if reader is not self.reader:
            return  # an older reader, already replaced
        self.reader = None
        self.connected_at = self.last_frame_at = None
        self.connect_button.setText("Connect")
        self.host.setEnabled(True)
        self.port.setEnabled(True)

    def on_connected(self, address: str) -> None:
        """A connection is up; frames should follow within one interval."""
        self.connected_at = time.monotonic()
        self.last_frame_at = None
        self.status_text.setText(f"Connected to {address}")
        self.log("CONNECTED", address)

    def on_disconnected(self, reason: str) -> None:
        """The connection ended or could not be made."""
        if self.connected_at is not None:
            self.log("LOST", reason)
        self.connected_at = self.last_frame_at = None
        self.status_text.setText(f"Not connected: {reason}")

    def set_auto_reconnect(self, on: bool) -> None:
        """Whether the reader retries after a lost connection."""
        if self.reader is not None:
            self.reader.auto_reconnect = on

    # -- frames -------------------------------------------------------------

    def on_message(self, message: str) -> None:
        """One status frame arrived."""
        now = time.monotonic()
        self.last_frame_at = now
        self.frame_times.append(now)
        self.raw.setPlainText(message)
        self.parsed = parse_status(message)
        self.fill_table(self.parsed)
        self.tick()

    def fill_table(self, parsed: ParsedStatus) -> None:
        """One row per interlock in the frame, the state cell in its colour."""
        self.table.setRowCount(len(parsed.parts))
        for row, part in enumerate(parsed.parts):
            state = QTableWidgetItem(part.state)
            background, foreground = state_brushes(part.state)
            state.setBackground(background)
            state.setForeground(foreground)
            font = state.font()
            font.setBold(True)
            state.setFont(font)
            self.table.setItem(row, 0, state)
            self.table.setItem(row, 1, QTableWidgetItem(part.name))
            self.table.setItem(row, 2, QTableWidgetItem(part.reason))
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)

    def frame_age(self) -> float | None:
        """Seconds since the latest frame; ``None`` before the first one."""
        if self.last_frame_at is None:
            return None
        return time.monotonic() - self.last_frame_at

    def tick(self) -> None:
        """Re-judge the verdict from the frame age; beep while not safe."""
        now = time.monotonic()
        age = self.frame_age()
        verdict = judge(self.parsed, age)
        if self.verdict is None or verdict.kind != self.verdict.kind:
            self.on_verdict_change(verdict)
        self.verdict = verdict
        detail = verdict.detail
        if verdict.kind != ALL_SAFE and self.not_safe_since is not None:
            detail = f"{detail}. Not safe for {hms(now - self.not_safe_since)}."
        self.panel.show_verdict(verdict.kind, detail)
        self.compact.show_verdict(verdict, detail)
        self.table.setEnabled(verdict.kind != STATE_NO_SIGNAL)
        self.silence.setEnabled(verdict.kind != ALL_SAFE and not self.silenced)
        self.setWindowTitle(f"{verdict.kind} - GEECS Interlock Client")
        self.connection_text.setText(self._rate_text(age))
        if self.connected_at is not None:
            self.uptime.setText(f"Up {hms(now - self.connected_at)}")
        else:
            self.uptime.setText("")
        self._beep(verdict, now)

    def _rate_text(self, age: float | None) -> str:
        if self.connected_at is None:
            return "Not connected."
        if age is None:
            return "Connected, waiting for the first frame."
        recent = [t for t in self.frame_times if t > time.monotonic() - 2.0]
        return (
            f"Connected. Last frame {age:.1f} s ago, {len(recent) / 2.0:.1f} frames/s."
            f" No frame for {NO_SIGNAL_S:g} s = {STATE_NO_SIGNAL}."
        )

    def on_verdict_change(self, verdict: Verdict) -> None:
        """A new verdict: log it, re-arm the alarm, restart the not-safe clock."""
        self.silenced = False
        if verdict.kind == ALL_SAFE:
            self.not_safe_since = None
        elif self.not_safe_since is None:
            self.not_safe_since = time.monotonic()
        self.log(verdict.kind, ", ".join(verdict.names))

    def _beep(self, verdict: Verdict, now: float) -> None:
        if verdict.kind == ALL_SAFE or self.silenced or not self.sound.isChecked():
            return
        if now - self.last_beep >= BEEP_EVERY_S:
            self.last_beep = now
            QApplication.beep()

    def silence_alarm(self) -> None:
        """Stop beeping until the verdict changes."""
        self.silenced = True
        self.silence.setEnabled(False)

    # -- history, options, views --------------------------------------------

    def log(self, what: str, detail: str) -> None:
        """Add a line to the verdict changes."""
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.history.append((stamp, what, detail))
        self.history_view.appendPlainText(f"{clock()} {what:<16} {detail}".rstrip())

    def export_history(self) -> None:
        """Save the verdict changes as CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export verdict changes", "interlock_history.csv", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", "verdict", "detail"])
            writer.writerows(self.history)

    def set_on_top(self, on: bool) -> None:
        """Keep the full window above the others."""
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        self.show()

    def show_compact(self) -> None:
        """Swap the full window for the strip."""
        self.compact.show()
        self.hide()

    def show_full(self) -> None:
        """Swap the strip for the full window."""
        self.show()
        self.compact.hide()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the reader and close the strip with the window."""
        self.timer.stop()
        self.disconnect_stream()
        self.compact.close()
        event.accept()


def main(argv: list[str] | None = None) -> int:
    """``geecs-interlock-client``: open the window, connected when a port is known."""
    default_host, default_port = config_address("status_port")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--compact", action="store_true", help="start as the strip")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    QGuiApplication.setApplicationDisplayName("GEECS Interlock Client")
    window = ClientWindow(args.host, args.port)
    if args.compact:
        window.show_compact()
    else:
        window.show()
    if args.port is not None:
        window.connect_stream()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
