"""The interlock server window: the worker's detail stream, in a window.

``geecs-interlock-server-gui [--host H] [--port P]`` connects to the
worker's detail stream (``config.ini [interlocks] detail_port`` by
default) and shows what the worker's interlock stack is doing: the
overall verdict, the status stream LabVIEW reads (its address, clients and
the exact frame going out), the RunEngine's state, and one row per
interlock with its condition, live value, action plan and settle time.
Selecting a row shows its diagnostic and the last minute of its live
value against its limit.  The event log is the window's own record of
the state changes it saw.  Read-only: it never writes to the worker.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from typing import Any

from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QGuiApplication,
    QKeySequence,
    QPainter,
    QPaintEvent,
    QPen,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
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
    DETAIL_VERSION,
    STATE_NO_SIGNAL,
    STATE_NOT_ARMED,
    STATE_OK,
    STATE_TRIPPED,
    TRIPPED,
)

#: Seconds without a detail frame before the window shows NO SIGNAL.
NO_SIGNAL_S = 2.0
#: Seconds of live values the trend keeps.
TREND_S = 60.0
#: How often the window re-checks the frame age, milliseconds.
TICK_MS = 250

COLUMNS = ["State", "Name", "Condition", "Live", "on_trip", "Settle"]


def _number(value: float | None) -> str:
    return "---" if value is None else f"{value:.4g}"


class TrendPlot(QWidget):
    """The selected interlock's live value over the last minute, and its limit."""

    def __init__(self) -> None:
        super().__init__()
        self.points: list[tuple[float, float]] = []
        self.limit: float | None = None
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_series(
        self, points: list[tuple[float, float]], limit: float | None
    ) -> None:
        """Plot *points* (``(time, value)``) and a dashed line at *limit*."""
        self.points = points
        self.limit = limit
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Draw on the palette's base colour; the limit in red, dashed."""
        painter = QPainter(self)
        palette = self.palette()
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, palette.base())
        painter.setPen(palette.mid().color())
        painter.drawRect(rect)
        values = [v for _, v in self.points]
        if self.limit is not None:
            values.append(self.limit)
        if not self.points:
            painter.setPen(palette.text().color())
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "no live value")
            return
        low, high = min(values), max(values)
        pad = (high - low) * 0.1 or abs(high) * 0.1 or 1.0
        low, high = low - pad, high + pad
        end = self.points[-1][0]
        margin = 4

        def at(t: float, v: float) -> QPointF:
            x = (
                rect.left()
                + margin
                + (rect.width() - 2 * margin) * (1 - (end - t) / TREND_S)
            )
            y = (
                rect.bottom()
                - margin
                - (rect.height() - 2 * margin) * (v - low) / (high - low)
            )
            return QPointF(x, y)

        if self.limit is not None:
            pen = QPen(QColor(192, 0, 0))
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(at(end - TREND_S, self.limit), at(end, self.limit))
        painter.setPen(QPen(palette.text().color(), 1.5))
        painter.drawPolyline([at(t, v) for t, v in self.points])
        painter.drawText(
            rect.adjusted(margin, margin, -margin, -margin),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            f"{high:.4g}",
        )
        painter.drawText(
            rect.adjusted(margin, margin, -margin, -margin),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
            f"{low:.4g}",
        )


def _form_value() -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


class ServerWindow(QMainWindow):
    """The interlock server window (see the module docstring)."""

    def __init__(self, host: str = "localhost", port: int | None = None) -> None:
        super().__init__()
        self.reader: StreamReader | None = None
        self.document: dict[str, Any] | None = None
        self.last_frame_at: float | None = None
        self.previous: dict[str, str] = {}
        self.previous_run_state: str | None = None
        self.trends: dict[str, deque[tuple[float, float]]] = {}
        self.selected: str | None = None
        self._build(host, port)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)
        self.tick()

    # -- layout -------------------------------------------------------------

    def _build(self, host: str, port: int | None) -> None:
        self.setWindowTitle("GEECS Interlock Server")
        self._menus()
        self.addToolBar(self._toolbar(host, port))
        central = QWidget()
        body = QHBoxLayout(central)
        body.addWidget(self._left_column())
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._table())
        splitter.addWidget(self._bottom())
        splitter.setSizes([300, 400])
        body.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)
        self.worker_text = QLabel()
        self.count_text = QLabel()
        self.stream_text = QLabel()
        self.statusBar().addWidget(self.worker_text, 1)
        self.statusBar().addPermanentWidget(self.count_text)
        self.statusBar().addPermanentWidget(self.stream_text)
        self.resize(1280, 820)

    def _menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        stream_menu = self.menuBar().addMenu("&Stream")
        self.connect_action = QAction("&Connect", self)
        self.connect_action.triggered.connect(self.toggle_connection)
        stream_menu.addAction(self.connect_action)
        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(self.about)
        help_menu.addAction(about)

    def _toolbar(self, host: str, port: int | None) -> QToolBar:
        bar = QToolBar("Connection")
        bar.setMovable(False)
        self.host = QLineEdit(host)
        self.host.setMinimumWidth(220)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(port or 1)
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        bar.addWidget(QLabel(" Worker: "))
        bar.addWidget(self.host)
        bar.addWidget(QLabel(" Detail port: "))
        bar.addWidget(self.port)
        bar.addWidget(self.connect_button)
        return bar

    def _left_column(self) -> QWidget:
        column = QWidget()
        column.setFixedWidth(300)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        self.panel = VerdictPanel(factor=1.9)
        layout.addWidget(self.panel)
        stream = QGroupBox("Status stream (LabVIEW)")
        form = QFormLayout(stream)
        self.stream_address = _form_value()
        self.stream_interval = _form_value()
        self.stream_frames = _form_value()
        self.clients = QListWidget()
        self.clients.setMaximumHeight(90)
        form.addRow("Address:", self.stream_address)
        form.addRow("Interval:", self.stream_interval)
        form.addRow("Frames sent:", self.stream_frames)
        form.addRow("Clients:", self.clients)
        layout.addWidget(stream)
        engine = QGroupBox("Run engine")
        form = QFormLayout(engine)
        self.re_state = _form_value()
        self.re_suspended_by = _form_value()
        self.re_on_trip = _form_value()
        self.re_resume = _form_value()
        form.addRow("State:", self.re_state)
        form.addRow("Suspended by:", self.re_suspended_by)
        form.addRow("on_trip:", self.re_on_trip)
        form.addRow("Resume after:", self.re_resume)
        layout.addWidget(engine)
        layout.addStretch(1)
        return column

    def _table(self) -> QTableWidget:
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.table.itemSelectionChanged.connect(self.on_select)
        return self.table

    def _bottom(self) -> QWidget:
        bottom = QWidget()
        layout = QHBoxLayout(bottom)
        layout.setContentsMargins(0, 0, 0, 0)
        self.detail_box = QGroupBox("Selected: (none)")
        form = QFormLayout(self.detail_box)
        self.sel_state = _form_value()
        self.sel_description = _form_value()
        self.sel_diagnostic = _form_value()
        self.sel_plan = _form_value()
        self.trend_label = QLabel()
        self.trend = TrendPlot()
        copy = QPushButton("Copy diagnostic")
        copy.clicked.connect(self.copy_diagnostic)
        form.addRow("State:", self.sel_state)
        form.addRow("Description:", self.sel_description)
        form.addRow("Diagnostic:", self.sel_diagnostic)
        form.addRow("Plans:", self.sel_plan)
        form.addRow(self.trend_label)
        form.addRow(self.trend)
        form.addRow(copy)
        layout.addWidget(self.detail_box, stretch=1)
        right = QVBoxLayout()
        frame_box = QGroupBox("Outgoing frame")
        frame_layout = QVBoxLayout(frame_box)
        self.outgoing = QPlainTextEdit(readOnly=True)
        self.outgoing.setMaximumHeight(110)
        frame_layout.addWidget(self.outgoing)
        log_box = QGroupBox("Event log")
        log_layout = QVBoxLayout(log_box)
        self.events = QPlainTextEdit(readOnly=True)
        self.events.setMaximumBlockCount(2000)
        log_layout.addWidget(self.events)
        right.addWidget(frame_box)
        right.addWidget(log_box, stretch=1)
        layout.addLayout(right, stretch=1)
        return bottom

    # -- connection ---------------------------------------------------------

    def toggle_connection(self) -> None:
        """Connect to the worker in the toolbar, or disconnect."""
        if self.reader is not None:
            self.disconnect_stream()
        else:
            self.connect_stream()

    def connect_stream(self) -> None:
        """Start reading the detail stream."""
        reader = StreamReader(self.host.text().strip(), self.port.value())
        reader.connected.connect(lambda address: self.log("CONNECTED", address))
        reader.message.connect(self.on_message)
        reader.disconnected.connect(self.on_disconnected)
        self.reader = reader
        reader.start()
        self.connect_button.setText("Disconnect")
        self.connect_action.setText("&Disconnect")
        self.host.setEnabled(False)
        self.port.setEnabled(False)

    def disconnect_stream(self) -> None:
        """Stop reading (the verdict turns NO SIGNAL)."""
        reader, self.reader = self.reader, None
        if reader is not None:
            reader.stop()
        self.last_frame_at = None
        self.connect_button.setText("Connect")
        self.connect_action.setText("&Connect")
        self.host.setEnabled(True)
        self.port.setEnabled(True)

    def on_disconnected(self, reason: str) -> None:
        """The connection ended or could not be made (the reader retries)."""
        if self.last_frame_at is not None:
            self.log("LOST", reason)
        self.last_frame_at = None
        self.stream_text.setText(f"Detail stream: {reason}")

    # -- documents ----------------------------------------------------------

    def on_message(self, message: str) -> None:
        """One detail document arrived."""
        try:
            document = json.loads(message)
        except json.JSONDecodeError:
            self.stream_text.setText(
                "Detail stream: not JSON (is this the status port?)"
            )
            return
        if document.get("v") != DETAIL_VERSION:
            self.stream_text.setText(
                f"Detail stream: version {document.get('v')!r} unknown"
            )
            return
        self.last_frame_at = time.monotonic()
        self.document = document
        self.record(document)
        self.show_document(document)

    def record(self, document: dict[str, Any]) -> None:
        """Log state changes and extend the trends."""
        sent_at = document["sent_at"]
        for entry in document["interlocks"]:
            name, state = entry["name"], entry["state"]
            if self.previous.get(name) != state:
                self.log(state, f"{name} {entry['diagnostic']}".rstrip())
            self.previous[name] = state
            trend = self.trends.setdefault(name, deque())
            if entry["live"] is not None:
                trend.append((sent_at, entry["live"]))
            while trend and trend[0][0] < sent_at - TREND_S:
                trend.popleft()
        run_state = document["run_engine"]["state"]
        if run_state != self.previous_run_state:
            self.log("RUN ENGINE", str(run_state))
        self.previous_run_state = run_state

    def show_document(self, document: dict[str, Any]) -> None:
        """Fill every widget from *document*."""
        entries = document["interlocks"]
        self.fill_table(entries, document["sent_at"])
        self.fill_stream(document)
        self.fill_engine(document, entries)
        self.outgoing.setPlainText(document["frame"])
        armed = sum(e["state"] != STATE_NOT_ARMED for e in entries)
        self.count_text.setText(f"{len(entries)} interlocks, {armed} armed")
        self.worker_text.setText(
            f"Worker {document['host']}, environment open since {clock(document['opened_at'])}"
        )
        self.show_selected()

    def fill_table(self, entries: list[dict[str, Any]], sent_at: float) -> None:
        """One row per interlock; the selection follows the name."""
        self.table.blockSignals(True)
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            settle = entry["settle_s"]
            cells = [
                entry["state"],
                entry["name"],
                entry["condition"],
                _number(entry["live"]),
                entry["on_trip"] or "---",
                "---" if settle is None else f"{settle:g} s",
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 0:
                    background, foreground = state_brushes(entry["state"])
                    item.setBackground(background)
                    item.setForeground(foreground)
                if column == 3:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.table.setItem(row, column, item)
            if entry["name"] == self.selected:
                self.table.selectRow(row)
        if self.selected is None and entries:
            self.selected = entries[0]["name"]
            self.table.selectRow(0)
        self.table.resizeColumnToContents(0)
        self.table.blockSignals(False)

    def fill_stream(self, document: dict[str, Any]) -> None:
        """The status stream LabVIEW reads: where, how often, to whom."""
        stream = document["stream"]
        self.stream_address.setText(
            stream["address"] or "off (no [interlocks] status_port)"
        )
        interval = stream["interval_s"]
        self.stream_interval.setText(
            "---" if interval is None else f"{interval * 1000:g} ms"
        )
        self.stream_frames.setText(str(stream["frames_sent"]))
        self.clients.clear()
        for client in stream["clients"]:
            since = document["sent_at"] - client["since"]
            self.clients.addItem(f"{client['address']}  {hms(since)}")

    def fill_engine(
        self, document: dict[str, Any], entries: list[dict[str, Any]]
    ) -> None:
        """The RunEngine's state, and which interlocks hold it."""
        state = document["run_engine"]["state"]
        self.re_state.setText("---" if state is None else str(state).upper())
        tripped = [e for e in entries if e["state"] == STATE_TRIPPED]
        holding = tripped if state == "paused" else []
        self.re_suspended_by.setText(", ".join(e["name"] for e in holding) or "---")
        self.re_on_trip.setText(
            ", ".join(e["on_trip"] for e in holding if e["on_trip"]) or "---"
        )
        settles = [e["settle_s"] for e in holding if e["settle_s"] is not None]
        self.re_resume.setText(f"safe for {max(settles):g} s" if settles else "---")

    def on_select(self) -> None:
        """Follow the table's selection."""
        rows = self.table.selectionModel().selectedRows()
        if rows:
            item = self.table.item(rows[0].row(), 1)
            if item is not None:
                self.selected = item.text()
        self.show_selected()

    def show_selected(self) -> None:
        """Fill the selected interlock's box."""
        entry = self._selected_entry()
        if entry is None or self.document is None:
            self.detail_box.setTitle("Selected: (none)")
            self.trend.set_series([], None)
            return
        self.detail_box.setTitle(f"Selected: {entry['name']}")
        state = entry["state"]
        if entry["since"] is not None and state != STATE_OK:
            state = f"{state} for {hms(self.document['sent_at'] - entry['since'])}"
        self.sel_state.setText(state)
        self.sel_description.setText(entry["description"] or "---")
        self.sel_diagnostic.setText(entry["diagnostic"] or "---")
        self.sel_plan.setText(
            f"on_trip {entry['on_trip'] or '---'}, "
            f"before_resume {entry['before_resume'] or '---'}"
        )
        label = entry["live_label"] or "live value"
        limit = entry["limit"]
        self.trend_label.setText(
            f"{label}, last {TREND_S:g} s"
            + ("" if limit is None else f" (dashed: limit {limit:g})")
        )
        self.trend.set_series(list(self.trends.get(entry["name"], ())), limit)

    def _selected_entry(self) -> dict[str, Any] | None:
        if self.document is None:
            return None
        for entry in self.document["interlocks"]:
            if entry["name"] == self.selected:
                return entry
        return None

    # -- verdict ------------------------------------------------------------

    def tick(self) -> None:
        """Re-judge the verdict from the latest document and its age."""
        age = (
            None
            if self.last_frame_at is None
            else time.monotonic() - self.last_frame_at
        )
        stale = age is None or age > NO_SIGNAL_S or self.document is None
        if stale:
            detail = (
                "Not receiving the detail stream."
                if age is None
                else (f"No detail frame for {age:.1f} s.")
            )
            self.panel.show_verdict(STATE_NO_SIGNAL, detail)
            self.setWindowTitle(f"{STATE_NO_SIGNAL} - GEECS Interlock Server")
        else:
            assert self.document is not None
            entries = self.document["interlocks"]
            safe = self.document["frame"].startswith(f"{ALL_SAFE} | ")
            counts = {
                s: sum(e["state"] == s for e in entries)
                for s in (STATE_TRIPPED, STATE_NOT_ARMED, STATE_OK)
            }
            detail = (
                f"{counts[STATE_TRIPPED]} tripped, {counts[STATE_NOT_ARMED]} not armed, "
                f"{counts[STATE_OK]} OK"
            )
            kind = ALL_SAFE if safe else TRIPPED
            self.panel.show_verdict(kind, detail)
            self.setWindowTitle(f"{kind} - GEECS Interlock Server")
            self.stream_text.setText(f"Detail stream: frame {age:.1f} s ago")
        for widget in (self.table, self.detail_box, self.clients):
            widget.setEnabled(not stale)

    # -- actions ------------------------------------------------------------

    def log(self, what: str, detail: str) -> None:
        """Add a line to the event log."""
        self.events.appendPlainText(f"{clock()} {what:<10} {detail}".rstrip())

    def copy_diagnostic(self) -> None:
        """Copy the selected interlock's name and diagnostic."""
        entry = self._selected_entry()
        if entry is not None:
            QGuiApplication.clipboard().setText(
                f"{entry['name']}: {entry['diagnostic']}"
            )

    def about(self) -> None:
        """What this window reads."""
        QMessageBox.about(self, "GEECS Interlock Server", __doc__ or "")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the reader with the window."""
        self.timer.stop()
        self.disconnect_stream()
        event.accept()


def main(argv: list[str] | None = None) -> int:
    """``geecs-interlock-server-gui``: open the window, connected when a port is known."""
    default_host, default_port = config_address("detail_port")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    QGuiApplication.setApplicationDisplayName("GEECS Interlock Server")
    window = ServerWindow(args.host, args.port)
    window.show()
    if args.port is not None:
        window.connect_stream()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
