"""The Monitor tab: the verdict strip (M1), connections (M2), actions (M3), the table (M4), the log (M5).

It reads the engine's snapshot on a timer and only draws: every verdict
comes from the monitor, so the window, the stream and the web client can
never disagree. The table sorts worst first and keeps each group's
expansion and the selection across redraws.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from geecs_interlocks.engine import Engine
from geecs_interlocks.gui.log_pane import LogPane
from geecs_interlocks.gui.theme import SAFE_COLOURS, STATE_COLOURS
from geecs_interlocks.logs import LogBuffer
from geecs_interlocks.monitor import STATES, InterlockStatus, Snapshot

#: Milliseconds between two redraws.
REFRESH_MS = 200
COLUMNS = ("State", "Interlock", "Value", "Reason")


def number_text(value: float | None, seconds: bool = False) -> str:
    """A value for the table: four significant digits (an age in seconds), a dash for none."""
    if value is None:
        return "—"
    return (
        f"{value:.1f} s"
        if seconds and value < 10
        else f"{value:.0f} s"
        if seconds
        else f"{value:.4g}"
    )


def _is_age(condition_line: str) -> bool:
    """Whether a check's value is an age (stale and external checks)."""
    return (
        " connected" in condition_line
        or "updating every" in condition_line
        or " on UDP " in condition_line
    )


def verdict_text(snapshot: Snapshot) -> tuple[str, str]:
    """The strip's headline and its detail line."""
    armed = [s for s in snapshot.statuses if s.armed]
    if not snapshot.statuses:
        return "No interlocks", "no interlocks file — reported unsafe"
    if not armed:
        return "Nothing armed", "every interlock is off — reported unsafe"
    if snapshot.safe:
        return "Safe", f"all {len(armed)} armed interlocks OK"
    bad = [s for s in armed if s.unsafe]
    since = min(s.since for s in bad)
    tripped = sum(s.state == "tripped" for s in bad)
    headline = "Tripped" if tripped else "Not ready"
    when = time.strftime("%H:%M:%S", time.localtime(since))
    return headline, f"{len(bad)} of {len(armed)} armed not OK · since {when}"


def _rank(status: InterlockStatus) -> int:
    return STATES.index(status.state)


class MonitorTab(QWidget):
    """The Monitor tab over *engine*.

    Parameters
    ----------
    engine :
        The running engine.
    buffer :
        The log records the pane shows.
    notify :
        Shows one line to the operator (the window's status bar).
    web_url :
        The web client's address (``None``: not served; the button says so).
    """

    always_on_top = pyqtSignal(bool)

    def __init__(
        self,
        engine: Engine,
        buffer: LogBuffer,
        notify: Callable[[str], None],
        web_url: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self._notify = notify
        self._web_url = web_url
        self._drawn: Snapshot | None = None
        layout = QVBoxLayout(self)
        layout.addLayout(self._build_top())
        layout.addLayout(self._build_actions())
        layout.addWidget(self._build_split(buffer), 1)

        self.arm.clicked.connect(self.arm_selected)
        self.reload.clicked.connect(self.reload_file)
        self.web.clicked.connect(self.open_web)
        self.on_top.toggled.connect(self.always_on_top.emit)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(REFRESH_MS)
        self.refresh()

    def _build_top(self) -> QHBoxLayout:
        """M1, the verdict strip, and M2, the connections."""
        self.verdict = QFrame()
        self.verdict.setObjectName("verdict")
        self.headline = QLabel()
        self.headline.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.detail = QLabel()
        strip = QHBoxLayout(self.verdict)
        strip.setContentsMargins(12, 6, 12, 6)
        strip.addWidget(self.headline)
        strip.addSpacing(10)
        strip.addWidget(self.detail, 1)
        self.connections = QLabel()
        self.connections.setObjectName("footnote")
        top = QHBoxLayout()
        top.addWidget(self.verdict, 1)
        top.addWidget(self.connections)
        return top

    def _build_actions(self) -> QHBoxLayout:
        """M3: Arm settings, Reload file, Open web client, Always on top."""
        self.arm = QPushButton("Arm settings")
        self.arm.setToolTip(
            "Record the current settings of the selected interlock (or of every "
            "interlock holding settings) as the values to hold"
        )
        self.reload = QPushButton("Reload file")
        self.web = QPushButton("Open web client")
        self.web.setEnabled(self._web_url is not None)
        if self._web_url is None:
            self.web.setToolTip("The web client is not served by this server yet")
        self.on_top = QCheckBox("Always on top")
        actions = QHBoxLayout()
        for button in (self.arm, self.reload, self.web):
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(self.on_top)
        return actions

    def _build_split(self, buffer: LogBuffer) -> QSplitter:
        """M4, the table, over M5, the log, behind a splitter."""
        self.table = QTreeWidget()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHeaderLabels(COLUMNS)
        self.table.setRootIsDecorated(True)
        self.table.setUniformRowHeights(True)
        self.table.setColumnWidth(0, 90)
        self.table.setColumnWidth(1, 360)
        self.table.setColumnWidth(2, 90)

        # M5
        self.log = LogPane(buffer)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.log)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        return split

    # -- drawing --------------------------------------------------------------

    def refresh(self) -> None:
        """Redraw from the engine's latest snapshot (skipped when it has not changed)."""
        snapshot = self.engine.snapshot()
        self._draw_connections()
        if snapshot is self._drawn:
            return
        self._drawn = snapshot
        self._draw_verdict(snapshot)
        self._draw_table(snapshot)

    def _draw_verdict(self, snapshot: Snapshot) -> None:
        headline, detail = verdict_text(snapshot)
        unsafe = [s for s in snapshot.statuses if s.armed and s.unsafe]
        if snapshot.safe:
            bg, fg = SAFE_COLOURS
        elif unsafe:
            bg, fg = STATE_COLOURS[min(unsafe, key=_rank).state]
        else:
            bg, fg = STATE_COLOURS["tripped"]
        self.verdict.setStyleSheet(f"QFrame#verdict {{ background: {bg}; }}")
        self.headline.setStyleSheet(f"font-size: 18px; font-weight: 600; color: {fg};")
        self.detail.setStyleSheet(f"color: {fg};")
        self.headline.setText(headline)
        self.detail.setText(detail)

    def _draw_connections(self) -> None:
        stream = self.engine.stream
        host, port = stream.address
        clients = len(stream.clients())
        parts = [
            f"Master Control: {clients} client{'s' if clients != 1 else ''}",
            f"{host}:{port} · {stream.wire_format}",
        ]
        now = time.monotonic()
        for udp in self.engine.signals.ports():
            reading = self.engine.signals.signal(udp)
            age = (
                "no packet yet"
                if reading.received is None
                else f"last packet {now - reading.received:.0f} s ago"
            )
            parts.append(f"UDP {udp} · {age}")
        self.connections.setText("\n".join(parts))

    def _draw_table(self, snapshot: Snapshot) -> None:
        expanded = {
            self.table.topLevelItem(i).text(1)
            for i in range(self.table.topLevelItemCount())
            if self.table.topLevelItem(i).isExpanded()
        }
        selected = self.selected_name()
        bar = self.table.verticalScrollBar()
        scroll = bar.value() if bar is not None else 0
        self.table.clear()
        for status in sorted(snapshot.statuses, key=_rank):
            item = self._row(status)
            self.table.addTopLevelItem(item)
            if status.name in expanded:
                item.setExpanded(True)
            if status.name == selected:
                self.table.setCurrentItem(item)
        if bar is not None:
            bar.setValue(scroll)

    def _row(self, status: InterlockStatus) -> QTreeWidgetItem:
        bg, fg = STATE_COLOURS[status.state]
        grouped = len(status.checks) > 1
        value = (
            f"{len(status.checks)} checks"
            if grouped
            else number_text(status.value, _is_age(status.condition))
        )
        name = status.name + ("  (holds settings)" if status.armable else "")
        item = QTreeWidgetItem([status.state, name, value, status.diagnostic])
        item.setData(1, Qt.ItemDataRole.UserRole, status.name)
        item.setBackground(0, QBrush(QColor(bg)))
        item.setForeground(0, QBrush(QColor(fg)))
        item.setTextAlignment(0, Qt.AlignmentFlag.AlignCenter)
        item.setToolTip(1, status.description or status.condition)
        item.setToolTip(3, status.diagnostic)
        if not status.armed:
            for column in range(len(COLUMNS)):
                item.setForeground(column, QBrush(QColor(STATE_COLOURS["off"][1])))
        if grouped:
            for check in status.checks:
                reason = check.diagnostic if check.unsafe else "ok"
                child = QTreeWidgetItem(
                    [
                        "",
                        check.condition,
                        number_text(check.value, _is_age(check.condition)),
                        reason,
                    ]
                )
                if check.unsafe:
                    child.setForeground(3, QBrush(QColor(STATE_COLOURS["tripped"][1])))
                item.addChild(child)
        return item

    # -- actions --------------------------------------------------------------

    def selected_name(self) -> str | None:
        """The interlock selected in the table (a check selects its interlock)."""
        item = self.table.currentItem()
        if item is None:
            return None
        if item.parent() is not None:
            item = item.parent()
        return item.data(1, Qt.ItemDataRole.UserRole)

    def arm_selected(self) -> None:
        """Arm the selected interlock's settings, or every interlock's if none holding settings is selected."""
        name = self.selected_name()
        statuses = {s.name: s for s in self.engine.snapshot().statuses}
        target = name if name in statuses and statuses[name].armable else None
        try:
            armed = self.engine.arm(target)
        except ValueError as exc:
            self._notify(f"Not armed: {exc}")
            return
        self._notify(f"Armed: {', '.join(armed)}")

    def reload_file(self) -> None:
        """Re-read the interlocks file (the stream stays connected)."""
        self._notify(self.engine.reload())
        self.refresh()

    def open_web(self) -> None:
        """Open the web client in the default browser."""
        if self._web_url is None:
            return
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl(self._web_url))


__all__ = ["COLUMNS", "REFRESH_MS", "MonitorTab", "number_text", "verdict_text"]
