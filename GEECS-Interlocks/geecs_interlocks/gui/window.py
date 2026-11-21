"""The server window: the Monitor and Builder tabs, the title, the status bar, Always on top.

It runs in the server's own process on the main thread; the engine's
threads keep evaluating and serving Master Control whatever the window
does, and closing the window stops the server.
"""

from __future__ import annotations

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMainWindow, QTabWidget

from geecs_interlocks.catalog import Catalog
from geecs_interlocks.engine import Engine
from geecs_interlocks.gui.builder_tab import BuilderTab
from geecs_interlocks.gui.monitor_tab import MonitorTab
from geecs_interlocks.logs import LogBuffer
from geecs_interlocks.sources import PreviewFactory

#: Where the window remembers its size and Always on top (per machine).
SETTINGS = ("GEECS", "geecs-interlocks")
#: Milliseconds a status-bar line stays.
NOTE_MS = 8000


class ServerWindow(QMainWindow):
    """The interlock server's window over *engine*."""

    def __init__(
        self,
        engine: Engine,
        buffer: LogBuffer,
        catalog: Catalog,
        make_preview: PreviewFactory | None = None,
        *,
        experiment: str | None = None,
        web_url: str | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.experiment = experiment
        self.settings = QSettings(*SETTINGS)
        self.tabs = QTabWidget()
        self.monitor = MonitorTab(engine, buffer, self.notify, web_url)
        self.builder = BuilderTab(engine, catalog, make_preview, self.notify)
        self.tabs.addTab(self.monitor, "Monitor")
        self.tabs.addTab(self.builder, "Builder")
        self.setCentralWidget(self.tabs)
        self.builder.changed.connect(self.update_title)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.monitor.always_on_top.connect(self.set_on_top)
        self.resize(1100, 760)
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        on_top = self.settings.value("on_top", False, type=bool)
        self.monitor.on_top.setChecked(bool(on_top))
        self.update_title()

    def notify(self, message: str) -> None:
        """Show *message* in the status bar for a while."""
        bar = self.statusBar()
        if bar is not None:
            bar.showMessage(message, NOTE_MS)

    def update_title(self) -> None:
        """``GEECS interlocks — <experiment> · <configuration>[ · unsaved changes]``."""
        parts = [p for p in (self.experiment,) if p]
        path = self.engine.path
        parts.append(path.stem if path is not None else "no configuration")
        if self.builder.draft.dirty:
            parts.append("unsaved changes")
        self.setWindowTitle("GEECS interlocks — " + " · ".join(parts))

    def set_on_top(self, on_top: bool) -> None:
        """Keep the window above every other (remembered per machine)."""
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on_top)
        self.settings.setValue("on_top", on_top)
        self.show()  # a flag change hides the window until shown again

    def _tab_changed(self, index: int) -> None:
        if self.tabs.widget(index) is self.builder:
            self.builder.sync_from_engine()
        self.update_title()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Remember the geometry and release the panes' listeners."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.monitor.timer.stop()
        self.builder.values_timer.stop()
        self.monitor.log.close_pane()
        self.builder.close_tab()
        super().closeEvent(event)


__all__ = ["ServerWindow"]
