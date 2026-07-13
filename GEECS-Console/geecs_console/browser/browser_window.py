"""The GEECS Scan Browser main window — screen-map regions B1-B6.

A quick-look Tiled client: pick a day (B1), pick a scan (B2), see what
happened — identity strip (B3), column plot (B4), pinned+plotted table
(B5), and the "Moved during scan" telemetry-drift rail (B6).

Architecture rules (mirroring the console package):

- The window depends on the :class:`~geecs_console.browser.catalog.ScanCatalog`
  protocol, never on ``tiled``.  Every catalog call runs off the GUI
  thread through :class:`~geecs_console.browser._background.BrowserWorker`
  (daemon thread + queued signal) — the GUI thread never blocks on Tiled.
- ``pyqtgraph`` is imported lazily (function-level) with
  ``PYQTGRAPH_QT_LIB=PySide6`` set first.
- Column interpretation (pinning, prettification, telemetry selection,
  scan-variable detection) goes through ``geecs_data_utils.tiled_schema``
  — the one module that knows the event schema.
- Open scan folder is **strictly read-only** (repo scan-folder invariant):
  resolution + ``is_dir()`` only, never creating anything on the scans
  path.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QDate, QStringListModel, Qt, QUrl, Slot
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDateEdit,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from geecs_data_utils import tiled_drift as drift_mod
from geecs_data_utils import tiled_schema as schema_map
from geecs_data_utils.tiled_catalog import (
    CatalogStatus,
    RunDetail,
    RunSummary,
    ScanCatalog,
    StubCatalog,
)

from geecs_console.browser._background import BrowserWorker
from geecs_console.services import ops_paths
from geecs_console.services.settings import ConsoleSettings

logger = logging.getLogger(__name__)

#: Screen-map dark palette (scan-browser-screen-map.html — shared console
#: family look, dark variant).
_C_BG = "#12161a"
_C_PANEL = "#1a2026"
_C_PANEL2 = "#20272e"
_C_LINE = "#2c353d"
_C_INK = "#dde4ea"
_C_DIM = "#8a95a0"
_C_FAINT = "#5c6670"
_C_ACCENT = "#4cc2b4"
_C_ACCENT_DIM = "#2b6f68"
_C_GOOD = "#3fae6e"
_C_BAD = "#cf5347"
_C_WARN = "#d9a13f"

_MONO = '"SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", monospace'

#: Series colors, mockup-first (accent, amber, then distinguishable extras).
_SERIES_COLORS = (
    "#4cc2b4",
    "#d9a13f",
    "#7fb2e5",
    "#cf5347",
    "#8fd08a",
    "#c79bd4",
    "#e0995e",
    "#9aa7e8",
)

#: The B4 X-axis entry meaning "row sequence" (``scan_event_index``).
_X_SHOT_LABEL = "shot #"

#: Cap on B5 rows rendered (quick-look, not a data browser).
_TABLE_MAX_ROWS = 2000

_BROWSER_QSS = f"""
QMainWindow, QWidget#b_root {{
    background-color: {_C_BG};
}}
QWidget {{
    color: {_C_INK};
    font-size: 12px;
}}
QLabel {{ background: transparent; }}
QLabel#b1_connection_chip, QLabel#b3_pills, QLabel#b5_footer,
QLabel#b6_summary {{
    color: {_C_DIM};
    font-family: {_MONO};
    font-size: 11px;
}}
QLabel#b2_day_label, QLabel#b4_x_label, QLabel#b4_y_label {{
    color: {_C_FAINT};
    font-family: {_MONO};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel#b3_title {{
    font-size: 15px;
    font-weight: 600;
}}
QLabel#b6_note {{
    color: {_C_FAINT};
    font-size: 11px;
}}
QLineEdit, QComboBox, QDateEdit {{
    background-color: {_C_PANEL2};
    border: 1px solid {_C_LINE};
    border-radius: 5px;
    padding: 3px 8px;
    color: {_C_INK};
    selection-background-color: {_C_ACCENT_DIM};
    selection-color: {_C_INK};
}}
QLineEdit:focus, QComboBox:focus, QDateEdit:focus {{
    border-color: {_C_ACCENT_DIM};
}}
QComboBox::drop-down, QDateEdit::drop-down {{ border: none; width: 16px; }}
QComboBox QAbstractItemView {{
    background-color: {_C_PANEL2};
    border: 1px solid {_C_LINE};
    selection-background-color: {_C_ACCENT_DIM};
    color: {_C_INK};
    outline: none;
}}
QPushButton {{
    background: transparent;
    border: 1px solid {_C_ACCENT_DIM};
    border-radius: 5px;
    padding: 3px 10px;
    color: {_C_ACCENT};
}}
QPushButton:hover {{ background-color: {_C_PANEL2}; }}
QPushButton:pressed {{ background-color: {_C_LINE}; }}
QPushButton:disabled {{ border-color: {_C_LINE}; color: {_C_FAINT}; }}
QListWidget {{
    background-color: {_C_PANEL};
    border: 1px solid {_C_LINE};
    border-radius: 5px;
    outline: none;
}}
QListWidget::item {{
    padding: 4px 8px;
    border-bottom: 1px solid {_C_PANEL2};
    color: {_C_INK};
}}
QListWidget::item:selected {{
    background-color: {_C_PANEL2};
    border-left: 2px solid {_C_ACCENT};
    color: {_C_INK};
}}
QListWidget::item:hover {{ background-color: {_C_PANEL2}; }}
QListWidget#b6_drift_list::item {{ font-family: {_MONO}; font-size: 11px; }}
QListWidget#b4_series_list::item {{ font-family: {_MONO}; font-size: 11px; }}
QTableWidget {{
    background-color: {_C_PANEL};
    alternate-background-color: {_C_PANEL2};
    border: 1px solid {_C_LINE};
    border-radius: 5px;
    gridline-color: {_C_PANEL2};
    font-family: {_MONO};
    font-size: 11px;
    color: {_C_DIM};
}}
QTableWidget QHeaderView::section {{
    background-color: {_C_PANEL2};
    color: {_C_FAINT};
    border: none;
    border-bottom: 1px solid {_C_LINE};
    padding: 3px 8px;
    font-family: {_MONO};
    font-size: 10px;
    font-weight: 600;
}}
QTableWidget QTableCornerButton::section {{
    background-color: {_C_PANEL2};
    border: none;
}}
QStatusBar {{
    background-color: {_C_PANEL};
    border-top: 1px solid {_C_LINE};
    color: {_C_DIM};
    font-family: {_MONO};
    font-size: 11px;
}}
QSplitter::handle {{ background-color: {_C_LINE}; }}
QScrollBar:vertical {{ background: {_C_PANEL}; width: 9px; }}
QScrollBar::handle:vertical {{
    background: {_C_LINE}; border-radius: 4px; min-height: 20px;
}}
QScrollBar:horizontal {{ background: {_C_PANEL}; height: 9px; }}
QScrollBar::handle:horizontal {{
    background: {_C_LINE}; border-radius: 4px; min-width: 20px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
"""


def _pg() -> Any:
    """Import pyqtgraph lazily, pinned to the PySide6 binding.

    Returns
    -------
    module
        The ``pyqtgraph`` module (``PYQTGRAPH_QT_LIB=PySide6`` set before
        the first import so it never binds a stray PyQt install).
    """
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    import pyqtgraph

    return pyqtgraph


def resolve_scan_folder(detail: RunDetail, day: date) -> Optional[Path]:
    """Resolve a run's scan folder for B3's Open button — strictly read-only.

    Prefers the run's own ``scan_folder`` start-doc path; falls back to
    building the daily ``scans/ScanNNN`` path for the **selected** date via
    :func:`geecs_console.services.ops_paths.todays_scan_folder` (pure path
    construction).  Only an *existing* directory is returned — nothing on
    the scans path is ever created (repo scan-folder invariant; pinned by a
    tree-untouched test).

    Parameters
    ----------
    detail : RunDetail
        The selected run (start doc + summary).
    day : datetime.date
        The B1-selected date the run was listed under.

    Returns
    -------
    Path or None
        The existing scan folder, or ``None`` when it is absent or
        unresolvable.
    """
    metadata_folder = detail.start_doc.get("scan_folder")
    if metadata_folder:
        candidate = Path(str(metadata_folder))
        return candidate if candidate.is_dir() else None
    scan_number = detail.summary.scan_number
    if scan_number is None:
        return None
    daily = ops_paths.todays_scan_folder(detail.summary.experiment, today=day)
    if daily is None:
        return None
    candidate = daily / f"Scan{scan_number:03d}"
    return candidate if candidate.is_dir() else None


def _fmt_time_of_day(epoch: float) -> str:
    """Format an epoch-seconds value as local ``HH:MM`` ("" for 0/invalid)."""
    if not epoch or not math.isfinite(epoch):
        return ""
    try:
        return datetime.fromtimestamp(epoch).strftime("%H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


def _fmt_timestamp_cell(epoch: float) -> str:
    """Format an ``acq_timestamp`` cell as ``HH:MM:SS.mmm`` (raw on failure)."""
    try:
        stamp = float(epoch)
        if math.isfinite(stamp) and stamp > 0:
            moment = datetime.fromtimestamp(stamp)
            return moment.strftime("%H:%M:%S.") + f"{moment.microsecond // 1000:03d}"
    except (TypeError, ValueError, OverflowError, OSError):
        pass
    return str(epoch)


def _fmt_value(value: Any) -> str:
    """Format one B5 table cell (general numeric format, str fallback)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.6g}"


class ScanBrowserWindow(QMainWindow):
    """The scan-browser main window (screen-map regions B1-B6).

    Parameters
    ----------
    catalog : ScanCatalog, optional
        The run source; default is the offline :class:`StubCatalog`.
        ``__main__`` injects the real ``TiledScanCatalog``.
    settings : ConsoleSettings, optional
        Persistent GUI state (shared with the console: the last-experiment
        memory).  Tests inject a tmp-backed instance.
    experiment : str, optional
        Experiment to open with; empty restores the remembered one.
    today : datetime.date, optional
        The initial B1 date (tests pin it); defaults to today.
    scan_folder_resolver : callable, optional
        ``(RunDetail, date) -> Path | None`` for B3's Open scan folder;
        defaults to :func:`resolve_scan_folder`.  Tests inject fakes.
    parent : QWidget, optional
        Qt parent.
    """

    def __init__(
        self,
        *,
        catalog: Optional[ScanCatalog] = None,
        settings: Optional[ConsoleSettings] = None,
        experiment: str = "",
        today: Optional[date] = None,
        scan_folder_resolver: Optional[
            Callable[[RunDetail, date], Optional[Path]]
        ] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.catalog: ScanCatalog = catalog if catalog is not None else StubCatalog()
        self._settings = settings if settings is not None else ConsoleSettings()
        self._scan_folder_resolver = (
            scan_folder_resolver
            if scan_folder_resolver is not None
            else resolve_scan_folder
        )

        self._detail: Optional[RunDetail] = None
        self._summaries: list[RunSummary] = []
        #: plotted Y columns in add order → their pyqtgraph items.
        self._series: dict[str, list[Any]] = {}
        self._series_stats: dict[str, str] = {}
        self._list_generation = 0
        self._detail_generation = 0
        self._closing = False

        # Workers (window-owned QObjects; daemon threads emit on them, the
        # queued connections below deliver on the GUI thread).
        self._probe_worker = BrowserWorker(self)
        self._list_worker = BrowserWorker(self)
        self._detail_worker = BrowserWorker(self)
        self._probe_worker.result_ready.connect(
            self._on_probe_result, Qt.ConnectionType.QueuedConnection
        )
        self._list_worker.result_ready.connect(
            self._on_list_result, Qt.ConnectionType.QueuedConnection
        )
        self._detail_worker.result_ready.connect(
            self._on_detail_result, Qt.ConnectionType.QueuedConnection
        )

        self.setWindowTitle("GEECS Scan Browser")
        self.resize(1180, 760)
        self._build_ui(today if today is not None else date.today())
        self.setStyleSheet(_BROWSER_QSS)

        initial = experiment or self._settings.last_experiment
        if initial:
            self.b1_experiment_combo.setEditText(initial)
        self._start_probe()
        self.reload_runs()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self, initial_day: date) -> None:
        """Assemble regions B1-B6 (code-built; object names ``b1_``-``b6_``)."""
        root = QWidget(self)
        root.setObjectName("b_root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 8, 10, 6)
        outer.setSpacing(6)
        outer.addWidget(self._build_b1(initial_day))

        body = QSplitter(Qt.Orientation.Horizontal, root)
        body.setObjectName("b_body")
        body.addWidget(self._build_b2())
        body.addWidget(self._build_main_column())
        body.addWidget(self._build_b6())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setStretchFactor(2, 0)
        body.setSizes([250, 660, 252])
        outer.addWidget(body, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("")

    def _build_b1(self, initial_day: date) -> QWidget:
        """Build the B1 session bar: experiment, date, chip, filter."""
        bar = QWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(10)

        self.b1_experiment_combo = QComboBox(bar)
        self.b1_experiment_combo.setObjectName("b1_experiment_combo")
        self.b1_experiment_combo.setEditable(True)
        self.b1_experiment_combo.setMinimumWidth(150)
        self.b1_experiment_combo.lineEdit().setPlaceholderText("experiment")
        self.b1_experiment_combo.lineEdit().editingFinished.connect(
            self._on_experiment_committed
        )
        layout.addWidget(self.b1_experiment_combo)

        self.b1_date_edit = QDateEdit(bar)
        self.b1_date_edit.setObjectName("b1_date_edit")
        self.b1_date_edit.setCalendarPopup(True)
        self.b1_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.b1_date_edit.setDate(
            QDate(initial_day.year, initial_day.month, initial_day.day)
        )
        self.b1_date_edit.dateChanged.connect(lambda _d: self.reload_runs())
        layout.addWidget(self.b1_date_edit)

        self.b1_reload_button = QPushButton("Reload", bar)
        self.b1_reload_button.setObjectName("b1_reload_button")
        self.b1_reload_button.clicked.connect(self.reload_runs)
        layout.addWidget(self.b1_reload_button)

        self.b1_connection_chip = QLabel("tiled: probing…", bar)
        self.b1_connection_chip.setObjectName("b1_connection_chip")
        layout.addWidget(self.b1_connection_chip)

        layout.addStretch(1)

        self.b1_filter_edit = QLineEdit(bar)
        self.b1_filter_edit.setObjectName("b1_filter_edit")
        self.b1_filter_edit.setPlaceholderText(
            "filter scans — mode, save set, description…"
        )
        self.b1_filter_edit.setMinimumWidth(230)
        self.b1_filter_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.b1_filter_edit)
        return bar

    def _build_b2(self) -> QWidget:
        """Build the B2 rail: the day's run list (metadata-only rows)."""
        rail = QWidget(self)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.b2_day_label = QLabel("", rail)
        self.b2_day_label.setObjectName("b2_day_label")
        layout.addWidget(self.b2_day_label)

        self.b2_run_list = QListWidget(rail)
        self.b2_run_list.setObjectName("b2_run_list")
        self.b2_run_list.currentItemChanged.connect(self._on_run_selected)
        layout.addWidget(self.b2_run_list, 1)
        return rail

    def _build_main_column(self) -> QWidget:
        """Build the center column: B3 identity, B4 plot, B5 table."""
        column = QWidget(self)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(6)
        layout.addWidget(self._build_b3())

        split = QSplitter(Qt.Orientation.Vertical, column)
        split.addWidget(self._build_b4())
        split.addWidget(self._build_b5())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)
        return column

    def _build_b3(self) -> QWidget:
        """Build the B3 run identity strip (title, pills, actions)."""
        strip = QWidget(self)
        layout = QVBoxLayout(strip)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.b3_title = QLabel("No scan selected", strip)
        self.b3_title.setObjectName("b3_title")
        layout.addWidget(self.b3_title)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.b3_pills = QLabel("", strip)
        self.b3_pills.setObjectName("b3_pills")
        row.addWidget(self.b3_pills, 1)

        self.b3_copy_uid_button = QPushButton("Copy uid", strip)
        self.b3_copy_uid_button.setObjectName("b3_copy_uid_button")
        self.b3_copy_uid_button.setEnabled(False)
        self.b3_copy_uid_button.clicked.connect(self._on_copy_uid)
        row.addWidget(self.b3_copy_uid_button)

        self.b3_open_folder_button = QPushButton("Open scan folder", strip)
        self.b3_open_folder_button.setObjectName("b3_open_folder_button")
        self.b3_open_folder_button.setEnabled(False)
        self.b3_open_folder_button.clicked.connect(self._on_open_scan_folder)
        row.addWidget(self.b3_open_folder_button)
        layout.addLayout(row)
        return strip

    def _build_b4(self) -> QWidget:
        """Build the B4 plot region: X/Y pickers, series list, pyqtgraph plot."""
        region = QWidget(self)
        layout = QVBoxLayout(region)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        pickers = QHBoxLayout()
        pickers.setSpacing(8)
        x_label = QLabel("X", region)
        x_label.setObjectName("b4_x_label")
        pickers.addWidget(x_label)

        self.b4_x_combo = QComboBox(region)
        self.b4_x_combo.setObjectName("b4_x_combo")
        self.b4_x_combo.setMinimumWidth(170)
        self.b4_x_combo.addItem(_X_SHOT_LABEL)
        self.b4_x_combo.currentTextChanged.connect(self._on_x_changed)
        pickers.addWidget(self.b4_x_combo)

        y_label = QLabel("Y", region)
        y_label.setObjectName("b4_y_label")
        pickers.addWidget(y_label)

        self.b4_y_edit = QLineEdit(region)
        self.b4_y_edit.setObjectName("b4_y_edit")
        self.b4_y_edit.setPlaceholderText("+ add — search columns…")
        self.b4_y_edit.returnPressed.connect(self._on_y_committed)
        # The completer must stay referenced (PySide6 ownership hazard).
        self._y_completer = QCompleter([], self)
        self._y_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._y_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._y_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.b4_y_edit.setCompleter(self._y_completer)
        pickers.addWidget(self.b4_y_edit, 1)

        self.b4_remove_series_button = QPushButton("Remove", region)
        self.b4_remove_series_button.setObjectName("b4_remove_series_button")
        self.b4_remove_series_button.clicked.connect(self._on_remove_series)
        pickers.addWidget(self.b4_remove_series_button)
        layout.addLayout(pickers)

        self.b4_series_list = QListWidget(region)
        self.b4_series_list.setObjectName("b4_series_list")
        self.b4_series_list.setMaximumHeight(58)
        layout.addWidget(self.b4_series_list)

        pg = _pg()
        self.b4_plot = pg.PlotWidget(region, background=_C_PANEL2)
        self.b4_plot.setObjectName("b4_plot")
        plot_item = self.b4_plot.getPlotItem()
        plot_item.showGrid(x=False, y=True, alpha=0.25)
        for axis_name in ("left", "bottom"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen(_C_LINE))
            axis.setTextPen(pg.mkPen(_C_FAINT))
        self._plot_legend = plot_item.addLegend(labelTextColor=_C_DIM)
        layout.addWidget(self.b4_plot, 1)
        return region

    def _build_b5(self) -> QWidget:
        """Build the B5 table region: pinned+plotted columns, CSV export."""
        region = QWidget(self)
        layout = QVBoxLayout(region)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.b5_table = QTableWidget(region)
        self.b5_table.setObjectName("b5_table")
        self.b5_table.setAlternatingRowColors(True)
        self.b5_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.b5_table.verticalHeader().setVisible(False)
        layout.addWidget(self.b5_table, 1)

        footer = QHBoxLayout()
        self.b5_footer = QLabel("", region)
        self.b5_footer.setObjectName("b5_footer")
        footer.addWidget(self.b5_footer, 1)
        self.b5_export_button = QPushButton("Export CSV", region)
        self.b5_export_button.setObjectName("b5_export_button")
        self.b5_export_button.setEnabled(False)
        self.b5_export_button.clicked.connect(self._on_export_csv)
        footer.addWidget(self.b5_export_button)
        layout.addLayout(footer)
        return region

    def _build_b6(self) -> QWidget:
        """Build the B6 rail: the "Moved during scan" telemetry-drift list."""
        rail = QWidget(self)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        title = QLabel("Moved during scan", rail)
        title.setObjectName("b6_title")
        layout.addWidget(title)

        note = QLabel(
            "Telemetry columns whose value changed beyond tolerance between "
            "first and last shot — drift you didn't command.",
            rail,
        )
        note.setObjectName("b6_note")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.b6_drift_list = QListWidget(rail)
        self.b6_drift_list.setObjectName("b6_drift_list")
        self.b6_drift_list.itemClicked.connect(self._on_drift_clicked)
        layout.addWidget(self.b6_drift_list, 1)

        self.b6_summary = QLabel("", rail)
        self.b6_summary.setObjectName("b6_summary")
        self.b6_summary.setWordWrap(True)
        layout.addWidget(self.b6_summary)
        return rail

    # ------------------------------------------------------------------
    # B1: session (probe, experiment, date, filter)
    # ------------------------------------------------------------------

    def _start_probe(self) -> None:
        """Kick the connection-chip probe on a daemon thread."""
        catalog = self.catalog
        self._probe_worker.run_async(catalog.probe, "browser-probe")

    @Slot(object)
    def _on_probe_result(self, outcome: object) -> None:
        """Apply the probe outcome to the B1 chip (GUI thread, queued)."""
        status, error = outcome  # type: ignore[misc]
        if error is not None or not isinstance(status, CatalogStatus):
            self.b1_connection_chip.setText("tiled: probe failed")
            return
        dot = _C_GOOD if status.ok else _C_BAD
        self.b1_connection_chip.setText(status.label)
        self.b1_connection_chip.setStyleSheet(
            f"color: {dot};" if status.ok else f"color: {_C_DIM};"
        )

    def selected_day(self) -> date:
        """Return the B1-selected date.

        Returns
        -------
        datetime.date
            The date-picker value.
        """
        qdate = self.b1_date_edit.date()
        return date(qdate.year(), qdate.month(), qdate.day())

    def current_experiment(self) -> str:
        """Return the B1 experiment text (trimmed).

        Returns
        -------
        str
            The experiment combo's current text.
        """
        return self.b1_experiment_combo.currentText().strip()

    def _on_experiment_committed(self) -> None:
        """Persist the committed experiment and reload the day's runs."""
        experiment = self.current_experiment()
        if experiment:
            self._settings.last_experiment = experiment
        self.reload_runs()

    def reload_runs(self) -> None:
        """Reload B2 for the current experiment/date (off the GUI thread)."""
        self._list_generation += 1
        generation = self._list_generation
        experiment = self.current_experiment()
        day = self.selected_day()
        self.b2_day_label.setText(day.strftime("%a %Y-%m-%d").upper())
        catalog = self.catalog
        self.statusBar().showMessage("Loading runs…")
        self._list_worker.run_async(
            lambda: (generation, catalog.list_runs(experiment, day)),
            "browser-list-runs",
        )
        self._start_probe()

    @Slot(object)
    def _on_list_result(self, outcome: object) -> None:
        """Populate B2 from a finished day query (GUI thread, queued)."""
        result, error = outcome  # type: ignore[misc]
        if error is not None:
            self.statusBar().showMessage(f"Run listing failed: {error}")
            return
        generation, summaries = result  # type: ignore[misc]
        if generation != self._list_generation:
            return  # a newer reload superseded this one
        self._summaries = list(summaries)
        self.b2_run_list.clear()
        for summary in self._summaries:
            item = QListWidgetItem(self._run_item_text(summary))
            item.setData(Qt.ItemDataRole.UserRole, summary.uid)
            if summary.exit_status and summary.exit_status != "success":
                item.setForeground(Qt.GlobalColor.red)
            self.b2_run_list.addItem(item)
        self._apply_filter(self.b1_filter_edit.text())
        self.statusBar().showMessage(
            f"{len(self._summaries)} run(s) on {self.selected_day().isoformat()}"
        )

    @staticmethod
    def _run_item_text(summary: RunSummary) -> str:
        """Compose one B2 row's text (dot, number, time, mode)."""
        dot = "●" if summary.exit_status == "success" else "○"
        number = (
            f"Scan {summary.scan_number:03d}"
            if summary.scan_number is not None
            else summary.uid[:8]
        )
        time_text = _fmt_time_of_day(summary.start_time)
        shots = f" · {summary.shots} shots" if summary.shots else ""
        return f"{dot} {number}   {time_text}   {summary.mode}{shots}"

    def _apply_filter(self, text: str) -> None:
        """Hide B2 rows whose metadata haystack misses *text* (B1 filter)."""
        needle = text.strip().lower()
        for index in range(self.b2_run_list.count()):
            item = self.b2_run_list.item(index)
            uid = item.data(Qt.ItemDataRole.UserRole)
            summary = next((s for s in self._summaries if s.uid == uid), None)
            visible = not needle or (
                summary is not None and needle in summary.filter_text()
            )
            item.setHidden(not visible)

    # ------------------------------------------------------------------
    # B2 → B3: selection loads a run
    # ------------------------------------------------------------------

    def _on_run_selected(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        """Load the newly selected run's detail off the GUI thread."""
        if current is None:
            return
        uid = str(current.data(Qt.ItemDataRole.UserRole))
        self._detail_generation += 1
        generation = self._detail_generation
        catalog = self.catalog
        self.statusBar().showMessage(f"Loading {uid[:8]}…")
        self._detail_worker.run_async(
            lambda: (generation, catalog.load_run(uid)),
            "browser-load-run",
        )

    @Slot(object)
    def _on_detail_result(self, outcome: object) -> None:
        """Populate B3-B6 from a loaded run (GUI thread, queued)."""
        result, error = outcome  # type: ignore[misc]
        if error is not None:
            self.statusBar().showMessage(f"Run load failed: {error}")
            return
        generation, detail = result  # type: ignore[misc]
        if generation != self._detail_generation or self._closing:
            return
        self._detail = detail
        self._clear_series()
        self._populate_b3(detail)
        self._populate_b4_pickers(detail)
        self._populate_b6(detail)
        self._refresh_table()
        self.statusBar().showMessage("")

    def _populate_b3(self, detail: RunDetail) -> None:
        """Fill the identity strip from the run's start/stop metadata."""
        summary = detail.summary
        start = detail.start_doc
        stop = detail.stop_doc
        number = (
            f"Scan {summary.scan_number:03d}"
            if summary.scan_number is not None
            else summary.uid[:8]
        )
        mode = summary.mode.lower()
        acquisition = str(start.get("acquisition_mode") or "")
        duration = ""
        if stop.get("time") and start.get("time"):
            duration = f" · {stop['time'] - start['time']:.0f} s"
        shots = f" · {summary.shots} shots" if summary.shots else ""
        self.b3_title.setText(f"{number}  —  {mode}{shots} · {acquisition}{duration}")

        pills = [summary.exit_status or "no stop doc"]
        if summary.save_sets:
            pills.append("save sets: " + ", ".join(summary.save_sets))
        pills.append(f"uid {summary.uid[:8]}…")
        self.b3_pills.setText("   |   ".join(pills))
        self.b3_copy_uid_button.setEnabled(True)
        self.b3_open_folder_button.setEnabled(True)

    def _on_copy_uid(self) -> None:
        """Copy the selected run's full uid to the clipboard."""
        if self._detail is None:
            return
        QGuiApplication.clipboard().setText(self._detail.summary.uid)
        self.statusBar().showMessage("uid copied", 3000)

    def _on_open_scan_folder(self) -> None:
        """Open the run's scan folder — read-only resolution, never creates."""
        if self._detail is None:
            return
        folder = self._scan_folder_resolver(self._detail, self.selected_day())
        if folder is None:
            self.statusBar().showMessage(
                "Scan folder not found (not created — scan folders are scanner-owned)",
                6000,
            )
            return
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ------------------------------------------------------------------
    # B4: plot
    # ------------------------------------------------------------------

    def _dataframe(self) -> Any:
        """Return the selected run's DataFrame (``None`` when absent)."""
        if self._detail is None:
            return None
        return self._detail.data

    def _populate_b4_pickers(self, detail: RunDetail) -> None:
        """Rebuild the X combo and the Y completer for the loaded run."""
        df = detail.data
        columns = [] if df is None else [str(c) for c in df.columns]
        data_cols = schema_map.data_columns(columns)

        self.b4_x_combo.blockSignals(True)
        self.b4_x_combo.clear()
        self.b4_x_combo.addItem(_X_SHOT_LABEL)
        for column in schema_map.scan_variable_columns(columns, detail.start_doc):
            self.b4_x_combo.addItem(column)
        for column in data_cols:
            if self.b4_x_combo.findText(column) < 0:
                self.b4_x_combo.addItem(column)
        # Stepped scans default to the scan variable on X (the s-file plot).
        if schema_map.is_stepped_scan(detail.start_doc) and self.b4_x_combo.count() > 1:
            self.b4_x_combo.setCurrentIndex(1)
        else:
            self.b4_x_combo.setCurrentIndex(0)
        self.b4_x_combo.blockSignals(False)

        self._y_model = QStringListModel(data_cols, self)
        self._y_completer.setModel(self._y_model)

    def _numeric_values(self, column: str) -> Optional[list[float]]:
        """Return *column* as floats, or ``None`` when it is not numeric.

        Non-numeric columns exist by contract (dtype-tolerant telemetry) —
        the caller shows a status message instead of crashing.
        """
        df = self._dataframe()
        if df is None or column not in df.columns:
            return None
        import pandas as pd

        series = pd.to_numeric(df[column], errors="coerce")
        values = [float(v) for v in series.tolist()]
        if not any(math.isfinite(v) for v in values):
            return None
        return values

    def _x_values(self) -> Optional[list[float]]:
        """Return the current X axis values (shot sequence or a column)."""
        df = self._dataframe()
        if df is None:
            return None
        choice = self.b4_x_combo.currentText()
        if choice == _X_SHOT_LABEL:
            if schema_map.SHOT_INDEX_COLUMN in df.columns:
                return self._numeric_values(schema_map.SHOT_INDEX_COLUMN)
            return [float(i + 1) for i in range(len(df))]
        return self._numeric_values(choice)

    def _x_is_scan_variable(self) -> bool:
        """Return whether X is a scan-variable column of a stepped scan."""
        if self._detail is None:
            return False
        choice = self.b4_x_combo.currentText()
        columns = (
            []
            if self._detail.data is None
            else [str(c) for c in self._detail.data.columns]
        )
        return choice in schema_map.scan_variable_columns(
            columns, self._detail.start_doc
        ) and schema_map.is_stepped_scan(self._detail.start_doc)

    def _on_y_committed(self) -> None:
        """Add the typed/completed Y column as a plot series."""
        column = self.b4_y_edit.text().strip()
        if column:
            if self.add_series(column):
                self.b4_y_edit.clear()

    def add_series(self, column: str) -> bool:
        """Add *column* to the B4 plot (and thereby the B5 table).

        Parameters
        ----------
        column : str
            An event-stream column name.

        Returns
        -------
        bool
            True when the series was added; False for unknown/non-numeric
            columns or duplicates (a status message explains).
        """
        df = self._dataframe()
        if df is None:
            self.statusBar().showMessage("No run loaded", 4000)
            return False
        if column in self._series:
            self.statusBar().showMessage(f"{column} is already plotted", 4000)
            return False
        if column not in df.columns:
            self.statusBar().showMessage(f"Unknown column: {column}", 5000)
            return False
        values = self._numeric_values(column)
        if values is None:
            self.statusBar().showMessage(
                f"{column} is not numeric — cannot plot (telemetry may be "
                "string-typed)",
                6000,
            )
            return False
        self._series[column] = []
        self._draw_series(column, values)
        self._refresh_series_list()
        self._refresh_table()
        return True

    def remove_series(self, column: str) -> None:
        """Remove *column* from the plot and table.

        Parameters
        ----------
        column : str
            A currently plotted column.
        """
        items = self._series.pop(column, [])
        self._series_stats.pop(column, None)
        plot_item = self.b4_plot.getPlotItem()
        for item in items:
            plot_item.removeItem(item)
        self._refresh_series_list()
        self._refresh_table()

    def _on_remove_series(self) -> None:
        """Remove the series selected in the B4 series list."""
        item = self.b4_series_list.currentItem()
        if item is not None:
            self.remove_series(str(item.data(Qt.ItemDataRole.UserRole)))

    def plotted_columns(self) -> list[str]:
        """Return the plotted Y columns in add order.

        Returns
        -------
        list of str
            The active series' column names.
        """
        return list(self._series)

    def _series_color(self, column: str) -> str:
        """Return the palette color assigned to *column*'s series slot."""
        index = list(self._series).index(column) if column in self._series else 0
        return _SERIES_COLORS[index % len(_SERIES_COLORS)]

    def _draw_series(self, column: str, values: list[float]) -> None:
        """Render one series (raw points, or per-step mean ± σ error bars)."""
        pg = _pg()
        x_values = self._x_values()
        if x_values is None:
            self.statusBar().showMessage(
                f"X axis {self.b4_x_combo.currentText()!r} is not numeric",
                6000,
            )
            return
        color = self._series_color(column)
        finite = [v for v in values if math.isfinite(v)]
        mean = sum(finite) / len(finite) if finite else float("nan")
        sigma = (
            math.sqrt(sum((v - mean) ** 2 for v in finite) / len(finite))
            if finite
            else float("nan")
        )
        stats = f"{mean:.4g} ± {sigma:.2g}"
        self._series_stats[column] = stats
        label = f"{self._display(column)}  ({stats})"
        items: list[Any] = []
        plot_item = self.b4_plot.getPlotItem()

        if self._x_is_scan_variable():
            steps = self._per_step_stats(x_values, values)
            if steps:
                import numpy as np

                xs = [s[0] for s in steps]
                ys = [s[1] for s in steps]
                errors = [s[2] for s in steps]
                error_item = pg.ErrorBarItem(
                    x=np.array(xs),
                    y=np.array(ys),
                    height=np.array([2 * e for e in errors]),
                    pen=pg.mkPen(color),
                )
                plot_item.addItem(error_item)
                items.append(error_item)
                curve = plot_item.plot(
                    xs,
                    ys,
                    pen=pg.mkPen(color, width=1.5),
                    symbol="o",
                    symbolBrush=color,
                    symbolPen=None,
                    symbolSize=7,
                    name=label,
                )
                items.append(curve)
        else:
            curve = plot_item.plot(
                x_values,
                values,
                pen=pg.mkPen(color, width=1.2),
                symbol="o",
                symbolBrush=color,
                symbolPen=None,
                symbolSize=6,
                name=label,
            )
            items.append(curve)
        self._series[column] = items

    def _per_step_stats(
        self, x_values: list[float], y_values: list[float]
    ) -> list[tuple[float, float, float]]:
        """Aggregate (x, y) by ``bin_number``: per-step x/y mean and y σ."""
        df = self._dataframe()
        if df is None or schema_map.BIN_COLUMN not in df.columns:
            return []
        bins = df[schema_map.BIN_COLUMN].tolist()
        grouped: dict[Any, list[tuple[float, float]]] = {}
        for bin_id, x, y in zip(bins, x_values, y_values):
            if math.isfinite(x) and math.isfinite(y):
                grouped.setdefault(bin_id, []).append((x, y))
        steps: list[tuple[float, float, float]] = []
        for bin_id in sorted(grouped):
            pairs = grouped[bin_id]
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            mean_y = sum(ys) / len(ys)
            sigma_y = math.sqrt(sum((v - mean_y) ** 2 for v in ys) / len(ys))
            steps.append((sum(xs) / len(xs), mean_y, sigma_y))
        return steps

    def _on_x_changed(self, _text: str) -> None:
        """Replot every series against the newly selected X axis."""
        self._redraw_all_series()

    def _redraw_all_series(self) -> None:
        """Clear and redraw all active series (X change, reload)."""
        columns = list(self._series)
        plot_item = self.b4_plot.getPlotItem()
        for column in columns:
            for item in self._series[column]:
                plot_item.removeItem(item)
            self._series[column] = []
        for column in columns:
            values = self._numeric_values(column)
            if values is not None:
                self._draw_series(column, values)
        self._refresh_series_list()

    def _clear_series(self) -> None:
        """Drop every series (new run selected)."""
        plot_item = self.b4_plot.getPlotItem()
        for items in self._series.values():
            for item in items:
                plot_item.removeItem(item)
        self._series.clear()
        self._series_stats.clear()
        self._refresh_series_list()

    def _display(self, column: str) -> str:
        """Prettify *column* through the run's ``geecs_scalar_headers`` map."""
        headers = None
        if self._detail is not None:
            headers = self._detail.start_doc.get("geecs_scalar_headers")
        return schema_map.display_name(column, headers)

    def _refresh_series_list(self) -> None:
        """Rebuild the B4 series list (name + mean ± σ, series color)."""
        self.b4_series_list.clear()
        for column in self._series:
            stats = self._series_stats.get(column, "")
            item = QListWidgetItem(f"{self._display(column)}   {stats}")
            item.setData(Qt.ItemDataRole.UserRole, column)
            item.setForeground(QBrush(QColor(self._series_color(column))))
            self.b4_series_list.addItem(item)

    # ------------------------------------------------------------------
    # B5: table + CSV export
    # ------------------------------------------------------------------

    def visible_columns(self) -> list[str]:
        """Return the B5 column set: pinned (seq/time) + plotted.

        Returns
        -------
        list of str
            Column names in display order (pinned first).
        """
        df = self._dataframe()
        if df is None or self._detail is None:
            return []
        columns = [str(c) for c in df.columns]
        pinned = schema_map.pinned_columns(columns, self._detail.start_doc)
        return pinned + [c for c in self.plotted_columns() if c not in pinned]

    def _refresh_table(self) -> None:
        """Rebuild the B5 table over the visible column selection."""
        df = self._dataframe()
        visible = self.visible_columns()
        self.b5_table.clear()
        if df is None or not visible:
            self.b5_table.setRowCount(0)
            self.b5_table.setColumnCount(0)
            self.b5_footer.setText("")
            self.b5_export_button.setEnabled(False)
            return
        rows = min(len(df), _TABLE_MAX_ROWS)
        self.b5_table.setColumnCount(len(visible))
        self.b5_table.setRowCount(rows)
        self.b5_table.setHorizontalHeaderLabels([self._display(c) for c in visible])
        for row in range(rows):
            for col, column in enumerate(visible):
                raw = df[column].iloc[row]
                if schema_map.is_acq_timestamp_column(column):
                    text = _fmt_timestamp_cell(raw)
                else:
                    text = _fmt_value(raw)
                self.b5_table.setItem(row, col, QTableWidgetItem(text))
        total_columns = len(df.columns)
        self.b5_footer.setText(
            f"{len(df)} shots × {total_columns} columns · showing pinned + plotted"
        )
        self.b5_export_button.setEnabled(True)

    def export_csv(self, path: Path) -> bool:
        """Write the visible B5 selection to *path* as CSV.

        Parameters
        ----------
        path : Path
            Destination file (parent must exist; nothing is created on the
            scans tree — this writes wherever the user pointed the dialog).

        Returns
        -------
        bool
            True on success.
        """
        df = self._dataframe()
        visible = self.visible_columns()
        if df is None or not visible:
            return False
        try:
            df[visible].to_csv(path, index=False)
        except OSError as exc:
            self.statusBar().showMessage(f"CSV export failed: {exc}", 6000)
            return False
        self.statusBar().showMessage(f"Exported {path}", 5000)
        return True

    def _on_export_csv(self) -> None:
        """Ask for a destination and export the visible selection."""
        if self._detail is None:
            return
        number = self._detail.summary.scan_number
        suggested = f"scan_{number:03d}.csv" if number is not None else "scan.csv"
        filename, _filter = QFileDialog.getSaveFileName(
            self, "Export CSV", suggested, "CSV files (*.csv)"
        )
        if filename:
            self.export_csv(Path(filename))

    # ------------------------------------------------------------------
    # B6: drift rail
    # ------------------------------------------------------------------

    def _populate_b6(self, detail: RunDetail) -> None:
        """Run the drift analysis over numeric telemetry columns."""
        self.b6_drift_list.clear()
        df = detail.data
        if df is None:
            self.b6_summary.setText("no event data")
            return
        columns = [str(c) for c in df.columns]
        candidates = schema_map.telemetry_columns(columns)
        samples = {c: df[c].tolist() for c in candidates}
        report = drift_mod.compute_drift(samples)
        for entry in report.drifting:
            delta_text = drift_mod.format_delta(entry)
            item = QListWidgetItem(f"{self._display(entry.column)}   {delta_text}")
            item.setData(Qt.ItemDataRole.UserRole, entry.column)
            color = _C_WARN if entry.delta > 0 else _C_ACCENT
            item.setForeground(QBrush(QColor(color)))
            self.b6_drift_list.addItem(item)
        self.b6_summary.setText(
            f"{report.steady} of {report.evaluated} columns steady · "
            f"tolerance: {drift_mod.DEFAULT_THRESHOLD:g}σ of in-scan spread · "
            "click → add to plot"
        )

    def _on_drift_clicked(self, item: QListWidgetItem) -> None:
        """Add the clicked drifting column to the B4 plot."""
        column = str(item.data(Qt.ItemDataRole.UserRole))
        self.add_series(column)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def closeEvent(self, event: Any) -> None:  # noqa: N802 — Qt naming
        """Disconnect the workers so late daemon results go nowhere."""
        self._closing = True
        for worker in (self._probe_worker, self._list_worker, self._detail_worker):
            try:
                worker.result_ready.disconnect()
            except RuntimeError:
                pass  # nothing connected / already gone
        super().closeEvent(event)
