"""M5: the server log, as the terminal prints it, under the monitor's table.

Records arrive from :class:`~geecs_interlocks.logs.LogBuffer` on whatever
thread logged them and cross to the window's thread through a queued Qt
signal. The level and text filters, Clear and the end of a Pause redraw
from the buffer, so nothing logged while paused or filtered out is lost.
"""

from __future__ import annotations

import html
import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from geecs_interlocks.gui.theme import LOG_INFO, LOG_UDP, LOG_WARNING
from geecs_interlocks.logs import BUFFER_LINES, FORMAT, LogBuffer

#: The level filter's choices: (label, minimum level).
LEVELS = (
    ("Info and up", logging.INFO),
    ("Warnings only", logging.WARNING),
    ("Everything", logging.DEBUG),
)


def _colour(record: logging.LogRecord) -> str:
    if record.levelno >= logging.WARNING:
        return LOG_WARNING
    if record.name.endswith(".external"):
        return LOG_UDP
    return LOG_INFO


class LogPane(QWidget):
    """The log pane: filters, Pause, Copy, Clear, and the lines."""

    _arrived = pyqtSignal(object)

    def __init__(self, buffer: LogBuffer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buffer = buffer
        self._formatter = logging.Formatter(FORMAT)
        self._paused = False

        title = QLabel("Server log")
        title.setStyleSheet("font-weight: 600;")
        self.level = QComboBox()
        for label, _ in LEVELS:
            self.level.addItem(label)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter text")
        self.filter.setClearButtonEnabled(True)
        self.pause = QPushButton("Pause")
        self.pause.setCheckable(True)
        copy = QPushButton("Copy")
        clear = QPushButton("Clear")

        bar = QHBoxLayout()
        bar.addWidget(title)
        bar.addWidget(self.level)
        bar.addWidget(self.filter, 1)
        bar.addWidget(self.pause)
        bar.addWidget(copy)
        bar.addWidget(clear)

        self.text = QPlainTextEdit()
        self.text.setObjectName("log")
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(BUFFER_LINES)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.text)

        self.level.currentIndexChanged.connect(self.redraw)
        self.filter.textChanged.connect(self.redraw)
        self.pause.toggled.connect(self._toggle_pause)
        copy.clicked.connect(self.copy)
        clear.clicked.connect(self.clear)
        self._arrived.connect(self._append)
        buffer.subscribe(self._arrived.emit)
        self.redraw()

    def close_pane(self) -> None:
        """Stop listening to the buffer (the window is closing)."""
        self._buffer.unsubscribe(self._arrived.emit)

    def _shown(self, record: logging.LogRecord) -> bool:
        if record.levelno < LEVELS[self.level.currentIndex()][1]:
            return False
        needle = self.filter.text().strip().lower()
        return not needle or needle in record.getMessage().lower()

    def _html(self, record: logging.LogRecord) -> str:
        line = html.escape(self._formatter.format(record))
        return f'<span style="color:{_colour(record)}; white-space:pre">{line}</span>'

    def _append(self, record: logging.LogRecord) -> None:
        if self._paused or not self._shown(record):
            return
        self.text.appendHtml(self._html(record))

    def redraw(self) -> None:
        """Redraw every kept record that passes the filters."""
        self.text.clear()
        for record in self._buffer.records():
            if self._shown(record):
                self.text.appendHtml(self._html(record))  # one block per line
        bar = self.text.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())

    def _toggle_pause(self, paused: bool) -> None:
        self._paused = paused
        self.pause.setText("Resume" if paused else "Pause")
        if not paused:
            self.redraw()

    def copy(self) -> None:
        """Copy the lines shown to the clipboard."""
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.text.toPlainText())

    def clear(self) -> None:
        """Forget every line kept (the log file keeps them)."""
        self._buffer.clear()
        self.redraw()


__all__ = ["LEVELS", "LogPane"]
