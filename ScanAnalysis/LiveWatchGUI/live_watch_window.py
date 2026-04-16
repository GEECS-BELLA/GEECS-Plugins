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
    QFileDialog,
)

from geecs_data_utils.doc_id_lookup import DocIDLookup, EXPERIMENT_FILE_IDS
from .log_handler import QtLogHandler
from .worker import LiveWatchConfig, LiveWatchWorker

logger = logging.getLogger(__name__)

# Mapping from log-level name to display colour
_LEVEL_COLOURS = {
    "DEBUG": QColor(128, 128, 128),  # grey
    "INFO": QColor(0, 0, 0),  # black
    "WARNING": QColor(200, 150, 0),  # amber
    "ERROR": QColor(200, 0, 0),  # red
    "CRITICAL": QColor(200, 0, 0),  # red
}

# Ordered log levels for the combo box
_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def _try_list_experiments(config_dir: Optional[Path] = None) -> list[str]:
    """Return available experiment config names, or an empty list on failure.

    Parameters
    ----------
    config_dir : Path, optional
        Explicit scan analysis config directory to search.  When *None*,
        falls back to the globally configured base directory.
    """
    try:
        from scan_analysis.config.config_loader import list_available_configs

        configs = list_available_configs(config_dir=config_dir)
        # Filter to experiment-level configs (those under experiments/ dir)
        names = []
        for name, paths in configs.items():
            for p in paths:
                if "experiments" in p.parts or "experiment" in p.parts:
                    names.append(name)
                    break
        return sorted(set(names), key=str.lower)
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
        self._doc_id_lookup: Optional[DocIDLookup] = None

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

        # Pre-fill config paths first (so analyzer groups can use the config directory)
        self._populate_default_paths()

        # Then populate experiment dropdown using the now-available config directory
        self._populate_experiments()

        # Install log handler
        self._install_log_handler()

        # Connect facility and date changes to auto-populate Document ID
        self.combo_facility.currentTextChanged.connect(self._on_facility_changed)
        self.date_edit.dateChanged.connect(self._on_date_changed)

        # Trigger initial Document ID population for default facility and date
        # This happens synchronously after all UI is initialized
        default_facility = self.combo_facility.currentText()
        if default_facility:
            self._on_facility_changed(default_facility)

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------

    def _build_config_group(self) -> QGroupBox:
        """Build the Configuration group box."""
        group = QGroupBox("Configuration")
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Experiment (for Google Docs and Z Drive Navigation)
        self.combo_facility = QComboBox()
        self.combo_facility.addItems(["Undulator", "Thomson"])
        self.combo_facility.setToolTip(
            "Select the experiment for Google Docs integration.\n"
            "This corresponds to the Data Storage Drive Location.\n"
            "This determines which experiment INI file is used for Document ID resolution."
        )
        self.combo_facility.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addRow("Experiment (for Google Docs):", self.combo_facility)

        # Analyzer Group (combo + refresh button)
        analyzer_group_row = QHBoxLayout()
        self.combo_experiment = QComboBox()
        self.combo_experiment.setEditable(True)
        self.combo_experiment.setToolTip(
            "Select the analyzer configuration group.\n"
            "Configs are loaded from the scan analysis config directory.\n"
            "The config group determines which diagnostics are included in the LiveWatch Analysis."
        )
        self.combo_experiment.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        analyzer_group_row.addWidget(self.combo_experiment)

        self.btn_refresh_experiments = QPushButton("⟳")
        self.btn_refresh_experiments.setFixedWidth(32)
        self.btn_refresh_experiments.setToolTip(
            "Reload the analyzer group list from the scan analysis config directory."
        )
        self.btn_refresh_experiments.clicked.connect(self._on_refresh_experiments)
        analyzer_group_row.addWidget(self.btn_refresh_experiments)

        layout.addRow("Analyzer Group:", analyzer_group_row)

        # Date
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        today = date.today()
        self.date_edit.setDate(QDate(today.year, today.month, today.day))
        self.date_edit.setToolTip(
            "Select the date that will be analyzed.\n"
            "Today's date should autopopulate, and previous dates can be selected if desired."
        )
        layout.addRow("Date:", self.date_edit)

        # Starting scan number
        self.spin_start_scan = QSpinBox()
        self.spin_start_scan.setRange(0, 99999)
        self.spin_start_scan.setValue(0)
        self.spin_start_scan.setToolTip(
            "Scan number to begin LiveWatch analysis at.\n"
            "Defaults to 0 for analysis of all scans on selected date. \n"
            "Changing this may be useful when using options such as 'Rerun Completed' or 'Rerun Failed' are activated."
        )
        layout.addRow("Start Scan #:", self.spin_start_scan)

        # GDoc upload
        self.check_gdoc = QCheckBox("Enable GDoc Upload")
        self.check_gdoc.setToolTip(
            "Toggle whether or not analysis results are automatically uploaded to the scan log.\n"
            "Requires logmaker_4_googledocs to be installed and configured."
        )
        layout.addRow("", self.check_gdoc)

        # Document ID (for GDoc upload)
        self.line_document_id = QLineEdit()
        self.line_document_id.setPlaceholderText(
            "(auto-detected from experiment config)"
        )
        self.line_document_id.setToolTip(
            "Google Doc ID for the experiment scan log, visible for debugging.\n"
            "Leave blank to auto-detect from the experiment INI file for today's scan.\n"
            "Enter document ID manually if autofill doesn't work."
        )
        layout.addRow("Document ID:", self.line_document_id)

        # Advanced: config paths with browse buttons
        scan_config_row = QHBoxLayout()
        self.line_scan_config = QLineEdit()
        self.line_scan_config.setPlaceholderText("(auto-detected)")
        self.line_scan_config.setToolTip(
            "Path to scan analysis config directory.\n"
            "Leave blank to use the default from ScanPaths configuration.\n"
            "If you change this click ⟳ to reload the Analyzer Group list."
        )
        scan_config_row.addWidget(self.line_scan_config)

        self.btn_browse_scan_config = QPushButton("Browse…")
        self.btn_browse_scan_config.setToolTip(
            "Browse for scan analysis config directory."
        )
        self.btn_browse_scan_config.clicked.connect(self._on_browse_scan_config)
        scan_config_row.addWidget(self.btn_browse_scan_config)

        layout.addRow("Scan Config Dir:", scan_config_row)

        image_config_row = QHBoxLayout()
        self.line_image_config = QLineEdit()
        self.line_image_config.setPlaceholderText("(auto-detected)")
        self.line_image_config.setToolTip(
            "Path to image analysis config directory.\n"
            "Leave blank to use the default from ScanPaths configuration."
        )
        image_config_row.addWidget(self.line_image_config)

        self.btn_browse_image_config = QPushButton("Browse…")
        self.btn_browse_image_config.setToolTip(
            "Browse for image analysis config directory."
        )
        self.btn_browse_image_config.clicked.connect(self._on_browse_image_config)
        image_config_row.addWidget(self.btn_browse_image_config)

        layout.addRow("Image Config Dir:", image_config_row)

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
            "Minimum log level to display.\nMessages below this level are hidden."
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

    def _populate_experiments(self, config_dir: Optional[Path] = None) -> None:
        """Fill the experiment combo box from available configs.

        Parameters
        ----------
        config_dir : Path, optional
            Explicit scan analysis config directory.  When *None*, uses the
            path currently entered in the Scan Config Dir field (if any),
            falling back to the globally configured default.
        """
        if config_dir is None:
            text = self.line_scan_config.text().strip()
            if text:
                config_dir = Path(text)

        experiments = _try_list_experiments(config_dir=config_dir)
        previous = self.combo_experiment.currentText()
        self.combo_experiment.clear()
        if experiments:
            self.combo_experiment.addItems(experiments)
            # Restore previous selection if still available
            idx = self.combo_experiment.findText(previous)
            if idx >= 0:
                self.combo_experiment.setCurrentIndex(idx)
            logger.info("Loaded %d experiment config(s).", len(experiments))
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

    def _on_facility_changed(self, facility: str) -> None:
        """Load experiment-specific .tsv file when facility changes."""
        if not facility.strip():
            self.line_document_id.clear()
            self._doc_id_lookup = None
            return

        # Check if facility has a Google Drive file ID for date-aware lookup
        if facility in EXPERIMENT_FILE_IDS:
            file_id = EXPERIMENT_FILE_IDS[facility]
            self._doc_id_lookup = DocIDLookup(facility, file_id)

            # Try to load the .tsv file
            if not self._doc_id_lookup.refresh():
                logger.warning(
                    "Could not load doc_index.tsv for facility '%s'", facility
                )
                # Fall back to resolve_document_id()
                self._try_resolve_document_id_fallback(facility)
                self._doc_id_lookup = None
                return

            # Now that we have the lookup loaded, update Document ID for current date
            self._update_document_id_from_date()
        else:
            # Fall back to resolve_document_id() for facilities without .tsv
            self._try_resolve_document_id_fallback(facility)
            self._doc_id_lookup = None

    def _on_date_changed(self) -> None:
        """Refresh Document ID when date selection changes."""
        if self._doc_id_lookup is not None:
            # Use date-aware lookup from .tsv
            self._update_document_id_from_date()
        else:
            # Fall back to facility-based lookup
            facility = self.combo_facility.currentText().strip()
            if facility:
                self._try_resolve_document_id_fallback(facility)

    def _update_document_id_from_date(self) -> None:
        """Look up Document ID from current date using loaded .tsv mapping."""
        if self._doc_id_lookup is None:
            return

        qdate = self.date_edit.date()
        doc_id = self._doc_id_lookup.get_document_id(
            qdate.year(), qdate.month(), qdate.day()
        )

        if doc_id:
            self.line_document_id.setText(doc_id)
            logger.info(
                "Resolved Document ID from .tsv for date %04d-%02d-%02d: %s",
                qdate.year(),
                qdate.month(),
                qdate.day(),
                doc_id,
            )
        else:
            self.line_document_id.clear()
            logger.debug(
                "No Document ID found in .tsv for date %04d-%02d-%02d",
                qdate.year(),
                qdate.month(),
                qdate.day(),
            )

    def _try_resolve_document_id_fallback(self, facility: str) -> None:
        """Fall back to resolve_document_id() for facilities without .tsv files."""
        try:
            from scan_analysis.gdoc_upload import resolve_document_id

            doc_id = resolve_document_id(facility)
            if doc_id:
                self.line_document_id.setText(doc_id)
                logger.info(
                    "Resolved Document ID for facility '%s' (fallback): %s",
                    facility,
                    doc_id,
                )
            else:
                self.line_document_id.clear()
                logger.debug(
                    "No Document ID found for facility '%s' (fallback)", facility
                )
        except Exception as exc:
            logger.debug(
                "Could not resolve Document ID for facility '%s': %s", facility, exc
            )
            self.line_document_id.clear()

    def _trigger_initial_document_id_population(self) -> None:
        """Trigger initial Document ID population after UI is fully initialized."""
        default_facility = self.combo_facility.currentText()
        if default_facility:
            self._on_facility_changed(default_facility)

    # ------------------------------------------------------------------
    # Build config from widgets
    # ------------------------------------------------------------------

    def _build_config(self) -> LiveWatchConfig:
        """Read current widget values and return a :class:`LiveWatchConfig`."""
        qdate = self.date_edit.date()
        facility = self.combo_facility.currentText().strip()
        analyzer_group = self.combo_experiment.currentText().strip()

        scan_config_text = self.line_scan_config.text().strip()
        image_config_text = self.line_image_config.text().strip()
        document_id_text = self.line_document_id.text().strip()

        return LiveWatchConfig(
            analyzer_group=analyzer_group,
            year=qdate.year(),
            month=qdate.month(),
            day=qdate.day(),
            start_scan_number=self.spin_start_scan.value(),
            experiment=facility,
            gdoc_enabled=self.check_gdoc.isChecked(),
            document_id=document_id_text if document_id_text else None,
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
        """Request the worker to stop gracefully.

        Disconnects the log-handler signal *before* requesting stop to
        prevent cross-thread signal deadlocks during QThread teardown.
        The signal is reconnected in :meth:`_on_worker_finished`.
        """
        if self._worker is not None:
            # Prevent double-click while stopping
            self.btn_start_stop.setEnabled(False)

            # Disconnect log signal to avoid cross-thread deadlock during
            # the shutdown sequence (observer.stop / observer.join may
            # trigger log messages from the worker thread).
            if self._log_handler is not None:
                try:
                    self._log_handler.emitter.log_record.disconnect(self._on_log_record)
                except TypeError:
                    pass  # already disconnected

            logger.info("Stopping LiveWatch...")
            self._worker.request_stop()
            # Don't block the GUI; the finished signal will clean up

    def _on_worker_finished(self) -> None:
        """Clean up after the worker thread exits.

        Waits briefly for the QThread to fully terminate, then
        reconnects the log-handler signal and re-enables the UI.
        """
        if self._worker is not None:
            # Ensure the QThread is fully dead before releasing the reference
            self._worker.wait(3000)
        self._worker = None

        # Reconnect log handler for the next run
        if self._log_handler is not None:
            try:
                self._log_handler.emitter.log_record.connect(self._on_log_record)
            except TypeError:
                pass  # already connected

        self._set_running_ui(False)

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
    # Slot: Browse / Refresh
    # ------------------------------------------------------------------

    def _on_browse_scan_config(self) -> None:
        """Open a directory picker for the scan analysis config directory."""
        current = self.line_scan_config.text().strip()
        start_dir = current if current else ""
        chosen = QFileDialog.getExistingDirectory(
            self, "Select Scan Analysis Config Directory", start_dir
        )
        if chosen:
            self.line_scan_config.setText(chosen)
            # Auto-refresh experiments when the scan config dir changes
            self._populate_experiments(config_dir=Path(chosen))

    def _on_browse_image_config(self) -> None:
        """Open a directory picker for the image analysis config directory."""
        current = self.line_image_config.text().strip()
        start_dir = current if current else ""
        chosen = QFileDialog.getExistingDirectory(
            self, "Select Image Analysis Config Directory", start_dir
        )
        if chosen:
            self.line_image_config.setText(chosen)

    def _on_refresh_experiments(self) -> None:
        """Reload the experiment list from the current scan config directory."""
        self._populate_experiments()

    # ------------------------------------------------------------------
    # UI state management
    # ------------------------------------------------------------------

    def _set_running_ui(self, running: bool) -> None:
        """Enable/disable widgets based on whether the worker is running."""
        self.btn_start_stop.setText("■  Stop" if running else "▶  Start")
        self.btn_start_stop.setEnabled(True)  # re-enable after _stop_worker disabled it

        # Disable config fields while running to prevent mid-run changes
        self.combo_experiment.setEnabled(not running)
        self.btn_refresh_experiments.setEnabled(not running)
        self.date_edit.setEnabled(not running)
        self.spin_start_scan.setEnabled(not running)
        self.check_gdoc.setEnabled(not running)
        self.line_scan_config.setEnabled(not running)
        self.btn_browse_scan_config.setEnabled(not running)
        self.line_image_config.setEnabled(not running)
        self.btn_browse_image_config.setEnabled(not running)

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
        """Ensure the worker is stopped before the window closes.

        Closes the log handler *first* to prevent cross-thread signal
        emission during the shutdown wait, then stops the worker.
        """
        # Close the log handler first to stop cross-thread signal emission
        if self._log_handler is not None:
            self._log_handler.close()
            try:
                self._log_handler.emitter.log_record.disconnect(self._on_log_record)
            except TypeError:
                pass
            for name in (
                "scan_analysis.live_task_runner",
                "scan_analysis.task_queue",
                "LiveWatchGUI",
            ):
                logging.getLogger(name).removeHandler(self._log_handler)

        # Now safe to stop the worker — no log signals can deadlock
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(5000)  # wait up to 5 seconds
        super().closeEvent(event)
