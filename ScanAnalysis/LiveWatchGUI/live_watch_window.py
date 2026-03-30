"""Main window for the LiveWatch GUI.

Builds the UI programmatically and connects widgets to the
:class:`~LiveWatchGUI.worker.LiveWatchWorker` background thread.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QDate
from PyQt5.QtGui import QColor, QTextCharFormat
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QComboBox,
    QDateEdit,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QTextEdit,
    QLabel,
    QLineEdit,
    QSplitter,
    QSizePolicy,
)

from .log_handler import QtLogHandler
from .worker import LiveWatchConfig, LiveWatchWorker

logger = logging.getLogger(__name__)

# Mapping from log-level name to display colour
_LEVEL_COLOURS = {
    "DEBUG": QColor(128, 128, 128),     # grey
    "INFO": QColor(0, 0, 0),            # black
    "WARNING": QColor(200, 150, 0),     # amber
    "ERROR": QColor(200, 0, 0),         # red
    "CRITICAL": QColor(200, 0, 0),      # red
}

# Ordered log levels for the combo box
_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def _try_list_experiments() -> list[str]:
    """Return available experiment config names, or an empty list on failure."""
    try:
        from scan_analysis.config.config_loader import list_available_configs

        configs = list_available_configs()
        # Filter to experiment-level configs (those under experiments/ dir)
        names = []
        for name, paths in configs.items():
            for p in paths:
                if "experiments" in p.parts or "experiment" in p.parts:
                    names.append(name)
                    break
            else:
                # Include all configs if directory structure is flat
                if not names:
                    names.append(name)
        return sorted(set(names), key=str.lower) if names else sorted(configs.keys(), key=str.lower)
    except Exception as exc:
        logger.warning("Could not list experiment configs: %s", exc)
        return []


def _try_get_default_paths() -> tuple[Optional[Path], Optional[Path]]:
    """Return (scan_analysis_config_dir, image_analysis_config_dir) defaults."""
    scan_dir: Optional[Path] = None
    image_dir: Optional[Path] = None
    try:
        from geecs_data_utils import ScanPaths

        scan_dir = ScanPaths.paths_config.scan_analysis_configs_path
    except Exception:
        pass
    try:
        from geecs_data_utils import ScanPaths

        image_dir = ScanPaths.paths_config.image_analysis_configs_path
    except Exception:
        pass
    return scan_dir, image_dir


class LiveWatchWindow(QMainWindow):
    """Main application window for LiveWatch analysis configuration and control.

    The window is divided into four sections:

    1. **Configuration** — experiment, date, scan number, GDoc toggle
    2. **Runtime Options** — max items, dry run, rerun flags
    3. **Control** — start/stop button and status indicator
    4. **Log Output** — filterable, scrollable log display
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("LiveWatch Analysis")
        self.setMinimumSize(560, 620)
        self.resize(620, 720)

        self._worker: Optional[LiveWatchWorker] = None
        self._log_handler: Optional[QtLogHandler] = None

        # Build UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Use a splitter so the user can resize the log area
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # Top container (config + options + control)
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        top_layout.addWidget(self._build_config_group())
        top_layout.addWidget(self._build_options_group())
        top_layout.addWidget(self._build_control_group())
        top_layout.addStretch()

        splitter.addWidget(top_widget)

        # Bottom container (log)
        splitter.addWidget(self._build_log_group())

        # Give the log area more initial space
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # Populate experiment dropdown
        self._populate_experiments()

        # Pre-fill config paths
        self._populate_default_paths()

        # Install log handler
        self._install_log_handler()

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------

    def _build_config_group(self) -> QGroupBox:
        """Build the Configuration group box."""
        group = QGroupBox("Configuration")
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Experiment
        self.combo_experiment = QComboBox()
        self.combo_experiment.setEditable(True)
        self.combo_experiment.setToolTip(
            "Select the analyzer configuration group.\n"
            "Configs are loaded from the scan analysis config directory."
        )
        layout.addRow("Experiment:", self.combo_experiment)

        # Date
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        today = date.today()
        self.date_edit.setDate(QDate(today.year, today.month, today.day))
        self.date_edit.setToolTip(
            "Analysis date. Determines which day folder to watch for new scans."
        )
        layout.addRow("Date:", self.date_edit)

        # Starting scan number
        self.spin_start_scan = QSpinBox()
        self.spin_start_scan.setRange(0, 99999)
        self.spin_start_scan.setValue(0)
        self.spin_start_scan.setToolTip(
            "Minimum scan number to process.\n"
            "Scans below this number are skipped."
        )
        layout.addRow("Start Scan #:", self.spin_start_scan)

        # GDoc upload
        self.check_gdoc = QCheckBox("Enable GDoc Upload")
        self.check_gdoc.setToolTip(
            "Upload analysis results to the experiment Google Doc scan log.\n"
            "Requires logmaker_4_googledocs to be installed and configured."
        )
        layout.addRow("", self.check_gdoc)

        # Advanced: config paths
        self.line_scan_config = QLineEdit()
        self.line_scan_config.setPlaceholderText("(auto-detected)")
        self.line_scan_config.setToolTip(
            "Path to scan analysis config directory.\n"
            "Leave blank to use the default from ScanPaths configuration."
        )
        layout.addRow("Scan Config Dir:", self.line_scan_config)

        self.line_image_config = QLineEdit()
        self.line_image_config.setPlaceholderText("(auto-detected)")
        self.line_image_config.setToolTip(
            "Path to image analysis config directory.\n"
            "Leave blank to use the default from ScanPaths configuration."
        )
        layout.addRow("Image Config Dir:", self.line_image_config)

        group.setLayout(layout)
        return group

    def _build_options_group(self) -> QGroupBox:
        """Build the Runtime Options group box."""
        group = QGroupBox("Runtime Options")
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Max items per cycle
        self.spin_max_items = QSpinBox()
        self.spin_max_items.setRange(1, 50)
        self.spin_max_items.setValue(1)
        self.spin_max_items.setToolTip(
            "Maximum number of analyzer tasks to run per processing cycle.\n"
            "Higher values process more tasks per loop but may delay\n"
            "responsiveness to new scans."
        )
        layout.addRow("Max Items / Cycle:", self.spin_max_items)

        # Checkboxes row
        checks_layout = QHBoxLayout()

        self.check_dry_run = QCheckBox("Dry Run")
        self.check_dry_run.setToolTip(
            "Update status files but skip actual analyzer execution.\n"
            "Useful for testing configuration without running analysis."
        )
        checks_layout.addWidget(self.check_dry_run)

        self.check_rerun_completed = QCheckBox("Rerun Completed")
        self.check_rerun_completed.setToolTip(
            "Reset completed analyzers to queued so they run again.\n"
            "Applied once per discovered scan."
        )
        checks_layout.addWidget(self.check_rerun_completed)

        self.check_rerun_failed = QCheckBox("Rerun Failed")
        self.check_rerun_failed.setToolTip(
            "Requeue analyzers that previously failed,\n"
            "giving them another chance to run."
        )
        checks_layout.addWidget(self.check_rerun_failed)

        checks_layout.addStretch()
        layout.addRow("", checks_layout)

        group.setLayout(layout)
        return group

    def _build_control_group(self) -> QGroupBox:
        """Build the Control group box with start/stop and status."""
        group = QGroupBox("Control")
        layout = QHBoxLayout()

        self.btn_start_stop = QPushButton("▶  Start")
        self.btn_start_stop.setMinimumHeight(36)
        self.btn_start_stop.setToolTip(
            "Start or stop the LiveWatch file watcher and analysis loop."
        )
        self.btn_start_stop.clicked.connect(self._on_start_stop)
        layout.addWidget(self.btn_start_stop)

        layout.addSpacing(20)

        self.label_status = QLabel("● Idle")
        self.label_status.setToolTip("Current status of the LiveWatch runner.")
        self.label_status.setStyleSheet("font-weight: bold; color: grey;")
        layout.addWidget(self.label_status)

        layout.addStretch()

        group.setLayout(layout)
        return group

    def _build_log_group(self) -> QGroupBox:
        """Build the Log Output group box."""
        group = QGroupBox("Log Output")
        layout = QVBoxLayout()

        # Top row: level selector + clear button
        top_row = QHBoxLayout()

        top_row.addWidget(QLabel("Level:"))
        self.combo_log_level = QComboBox()
        self.combo_log_level.addItems(_LOG_LEVELS)
        self.combo_log_level.setCurrentText("INFO")
        self.combo_log_level.setToolTip(
            "Minimum log level to display.\n"
            "Messages below this level are hidden."
        )
        self.combo_log_level.currentTextChanged.connect(self._on_log_level_changed)
        top_row.addWidget(self.combo_log_level)

        top_row.addStretch()

        self.btn_clear_log = QPushButton("Clear")
        self.btn_clear_log.setToolTip("Clear all log messages from the display.")
        self.btn_clear_log.clicked.connect(self._on_clear_log)
        top_row.addWidget(self.btn_clear_log)

        layout.addLayout(top_row)

        # Log text area
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setLineWrapMode(QTextEdit.NoWrap)
        self.text_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.text_log)

        group.setLayout(layout)
        return group

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _populate_experiments(self) -> None:
        """Fill the experiment combo box from available configs."""
        experiments = _try_list_experiments()
        self.combo_experiment.clear()
        if experiments:
            self.combo_experiment.addItems(experiments)
        else:
            self.combo_experiment.addItem("Undulator")
            logger.info("No experiment configs found; added default 'Undulator'.")

    def _populate_default_paths(self) -> None:
        """Pre-fill config path fields from ScanPaths defaults."""
        scan_dir, image_dir = _try_get_default_paths()
        if scan_dir:
            self.line_scan_config.setText(str(scan_dir))
        if image_dir:
            self.line_image_config.setText(str(image_dir))

    def _install_log_handler(self) -> None:
        """Attach a :class:`QtLogHandler` to the relevant loggers."""
        self._log_handler = QtLogHandler(level=logging.DEBUG)
        self._log_handler.emitter.log_record.connect(self._on_log_record)

        for name in (
            "scan_analysis.live_task_runner",
            "scan_analysis.task_queue",
            "LiveWatchGUI",
        ):
            logging.getLogger(name).addHandler(self._log_handler)

    # ------------------------------------------------------------------
    # Build config from widgets
    # ------------------------------------------------------------------

    def _build_config(self) -> LiveWatchConfig:
        """Read current widget values and return a :class:`LiveWatchConfig`."""
        qdate = self.date_edit.date()
        experiment = self.combo_experiment.currentText().strip()

        scan_config_text = self.line_scan_config.text().strip()
        image_config_text = self.line_image_config.text().strip()

        return LiveWatchConfig(
            analyzer_group=experiment,
            year=qdate.year(),
            month=qdate.month(),
            day=qdate.day(),
            start_scan_number=self.spin_start_scan.value(),
            experiment=experiment,
            gdoc_enabled=self.check_gdoc.isChecked(),
            config_dir=Path(scan_config_text) if scan_config_text else None,
            image_config_dir=Path(image_config_text) if image_config_text else None,
            max_items=self.spin_max_items.value(),
            dry_run=self.check_dry_run.isChecked(),
            rerun_completed=self.check_rerun_completed.isChecked(),
            rerun_failed=self.check_rerun_failed.isChecked(),
        )

    # ------------------------------------------------------------------
    # Slot: Start / Stop
    # ------------------------------------------------------------------

    def _on_start_stop(self) -> None:
        """Handle the Start/Stop button click."""
        if self._worker is not None and self._worker.isRunning():
            self._stop_worker()
        else:
            self._start_worker()

    def _start_worker(self) -> None:
        """Create and start the background worker thread."""
        config = self._build_config()
        logger.info(
            "Starting LiveWatch: experiment=%s date=%04d-%02d-%02d scan>=%d",
            config.analyzer_group,
            config.year,
            config.month,
            config.day,
            config.start_scan_number,
        )

        self._worker = LiveWatchWorker(config)
        self._worker.status_changed.connect(self._on_status_changed)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

        self._set_running_ui(True)

    def _stop_worker(self) -> None:
        """Request the worker to stop gracefully."""
        if self._worker is not None:
            logger.info("Stopping LiveWatch...")
            self._worker.request_stop()
            # Don't block the GUI; the finished signal will clean up

    def _on_worker_finished(self) -> None:
        """Clean up after the worker thread exits."""
        self._set_running_ui(False)
        self._worker = None

    # ------------------------------------------------------------------
    # Slot: Status updates
    # ------------------------------------------------------------------

    def _on_status_changed(self, status: str) -> None:
        """Update the status label when the worker state changes."""
        if status == "running":
            self.label_status.setText("● Running")
            self.label_status.setStyleSheet("font-weight: bold; color: green;")
        elif status == "stopped":
            self.label_status.setText("● Stopped")
            self.label_status.setStyleSheet("font-weight: bold; color: grey;")
        elif status == "error":
            self.label_status.setText("● Error")
            self.label_status.setStyleSheet("font-weight: bold; color: red;")

    def _on_error(self, message: str) -> None:
        """Display an error from the worker."""
        logger.error("Worker error: %s", message)

    # ------------------------------------------------------------------
    # Slot: Log display
    # ------------------------------------------------------------------

    def _on_log_record(self, message: str, level_name: str) -> None:
        """Append a log record to the text area if it passes the level filter."""
        current_level = logging.getLevelName(self.combo_log_level.currentText())
        record_level = logging.getLevelName(level_name)
        if record_level < current_level:
            return

        colour = _LEVEL_COLOURS.get(level_name, QColor(0, 0, 0))
        fmt = QTextCharFormat()
        fmt.setForeground(colour)

        cursor = self.text_log.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(message + "\n", fmt)

        # Auto-scroll to bottom
        scrollbar = self.text_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_log_level_changed(self, level_text: str) -> None:
        """Update the log handler level when the combo changes."""
        # The handler always captures everything; filtering is done in
        # _on_log_record so changing the combo is instant (no re-query).
        pass

    def _on_clear_log(self) -> None:
        """Clear the log display."""
        self.text_log.clear()

    # ------------------------------------------------------------------
    # UI state management
    # ------------------------------------------------------------------

    def _set_running_ui(self, running: bool) -> None:
        """Enable/disable widgets based on whether the worker is running."""
        self.btn_start_stop.setText("■  Stop" if running else "▶  Start")

        # Disable config fields while running to prevent mid-run changes
        self.combo_experiment.setEnabled(not running)
        self.date_edit.setEnabled(not running)
        self.spin_start_scan.setEnabled(not running)
        self.check_gdoc.setEnabled(not running)
        self.line_scan_config.setEnabled(not running)
        self.line_image_config.setEnabled(not running)

        # Runtime options can be changed while running in the future,
        # but for now disable them too for safety
        self.spin_max_items.setEnabled(not running)
        self.check_dry_run.setEnabled(not running)
        self.check_rerun_completed.setEnabled(not running)
        self.check_rerun_failed.setEnabled(not running)

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Ensure the worker is stopped before the window closes."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(5000)  # wait up to 5 seconds
        # Remove log handler to avoid dangling references
        if self._log_handler is not None:
            for name in (
                "scan_analysis.live_task_runner",
                "scan_analysis.task_queue",
                "LiveWatchGUI",
            ):
                logging.getLogger(name).removeHandler(self._log_handler)
        super().closeEvent(event)
