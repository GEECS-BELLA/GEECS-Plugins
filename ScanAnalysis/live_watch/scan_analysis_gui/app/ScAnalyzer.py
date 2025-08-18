"""
ScAnalyzer GUI Logic.

Authors:
Kyle Jensen (kjensen11, kjensen@lbl.gov)

This module implements the main PyQt5 window and worker used to run scan analysis
in the background while keeping the UI responsive. It wires together:
- The UI generated from Qt Designer (`Ui_MainWindow`)
- Analyzer mappings per experiment (`map_Undulator`, `map_Thomson`)
- The ScanWatch backend that discovers and processes scans
"""

# =============================================================================
# %% imports
from typing import Optional, Callable, List
from importlib.metadata import version

import sys
import time
import traceback
from datetime import date
import configparser
import importlib
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QLineEdit,
    QDialog,
    QInputDialog,
    QMessageBox,
)
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from live_watch.scan_analysis_gui.app.gui.ScAnalyzer_ui import Ui_MainWindow
from live_watch.scan_analysis_gui.app.AnalysisActivator import (
    AnalysisDialog,
    ActivatorTuple,
)
from live_watch.scan_watch import ScanWatch
from live_watch.scan_analysis_gui.utils.exceptions import exception_hook

# TODO should have a smarter way to import these, perhaps outside of this gui code
from scan_analysis.mapping.map_Undulator import undulator_analyzers
from scan_analysis.mapping.map_Thomson import thomson_analyzers

EXPERIMENT_TO_MAPPING = {
    "Undulator": undulator_analyzers,
    "Thomson": thomson_analyzers,
}
# =============================================================================
# %% global variables
DEBUG_MODE = False
if DEBUG_MODE:
    import pdb

CURRENT_VERSION = "v" + version("scananalysis")

# Type aliases for readability
RunningCheck = Callable[[], bool]
AnalysisFunction = Callable[[pyqtSignal, RunningCheck], None]
# =============================================================================
# %% classes


class ScAnalyzerWindow(QMainWindow):
    """
    Main application window for the Scan Analyzer GUI.

    This class hosts the user interface and coordinates background scan analysis:
    selecting the experiment and analyzers, choosing scan ranges, handling Google
    Doc scanlog uploads, and streaming progress/error messages to the log view.

    The actual long‑running analysis runs in a separate `QThread` via `Worker`,
    so the UI remains responsive.

    Attributes
    ----------
    overwrite_processed_scans : bool
        If True, previously processed scans will be reprocessed.
    ignore_list : list[int] | None
        Scan numbers to skip.
    ui : Ui_MainWindow
        The compiled Qt UI instance.
    worker : Worker | None
        Background worker instance (lives in a `QThread`).
    worker_thread : QThread | None
        Thread that executes the worker.
    documentID : str | None
        Google Doc ID to use for scanlog uploads (optional).
    experiment_name : str | None
        Current experiment name as read from the GEECS config.
    analyzer_items : list[scan_analysis.base.ScanAnalyzerInfo]
        Active analyzer configurations for the selected experiment.
    """

    def __init__(self) -> None:
        """Initialize and wire the main window, widgets, and signals."""
        super().__init__()

        # define attribute defaults
        self.overwrite_processed_scans: bool = False
        self.ignore_list: list[int] = None

        # create instance of Ui_MainWindow, setup elements from .ui file
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # set title of gui window
        self.setWindowTitle(f"GEECS ScAnalyzer - {CURRENT_VERSION}")

        # set up buttons
        self.setup_overwrite_checkbox()
        self.setup_start_button()
        self.setup_stop_button()
        self.setup_analysis_activator_button()

        # set up text edits
        self.setup_date_inputs()
        self.setup_scan_inputs()

        # set up gui log to display output
        self.setup_log_display()

        # initialize worker information
        self.worker: Optional[Worker] = None
        self.worker_thread: Optional[QThread] = None

        # set up checkbox to toggle uploading images to the scanlog
        self.ui.checkBoxScanlog.toggled.connect(self.toggle_scanlog_upload)
        self.toggle_scanlog_upload()

        # set up line edit to specify document id
        self.documentID: Optional[str] = None
        self.ui.lineDocumentID.editingFinished.connect(self.updateDocumentID)

        # set up line edit to show experiment name and button to change the experiment name
        self.experiment_name: Optional[str] = None
        self.load_experiment_name_from_config()
        self.ui.buttonReset.clicked.connect(self.write_experiment_name_to_config)

        # set up analyzer list
        self.analyzer_items = EXPERIMENT_TO_MAPPING.get(self.experiment_name, []).copy()

    def start_analysis(self) -> None:
        """
        Prepare and launch the background analysis.

        This method:
        - Verifies a previous analysis isn't running
        - Builds the ignore list from UI fields
        - Enables/disables relevant controls
        - Spawns a `Worker` and moves it to a `QThread`
        """
        try:
            # thread worker status check
            if self.worker_thread and self.worker_thread.isRunning():
                raise AnalysisRunningError("Analysis is already running!")

            # set ignore list
            self.set_ignore_list()

            # enable select buttons while analysis is running
            to_enable = [self.ui.buttonStop]
            for field in to_enable:
                field.setEnabled(True)

            # disable input fields and select buttons while analysis is running
            to_disable = self.findChildren(QLineEdit) + [
                self.ui.checkBoxOverwrite,
                self.ui.buttonStart,
            ]
            for field in to_disable:
                field.setEnabled(False)

            # initialize worker and thread
            self.initialize_worker()

        except Exception as e:
            self.log_error_message(str(e))
            self.end_analysis()

    def run_analysis(
        self,
        progress_callback: pyqtSignal,
        error_callback: pyqtSignal,
        is_running: RunningCheck,
        wait_time: float = 0.5,
    ) -> None:
        """
        Execute scan analysis on the worker thread.

        The worker constructs a `ScanWatch` instance and processes queued scans
        until `is_running()` returns False. Progress and errors are signaled
        back to the UI thread via `progress_callback` and `error_callback`.

        Parameters
        ----------
        progress_callback : pyqtSignal
            Signal to emit progress messages (str).
        error_callback : pyqtSignal
            Signal to emit error messages (str).
        is_running : Callable[[], bool]
            Function checked periodically to decide whether to continue.
        wait_time : float, optional
            Polling delay for the watcher loop, by default 0.5 seconds.
        """
        try:
            analysis_date = date(
                year=int(self.ui.inputYear.text()),
                month=int(self.ui.inputMonth.text()),
                day=int(self.ui.inputDay.text()),
            )
            if (
                self.ui.checkBoxScanlog.isChecked()
                and analysis_date != date.today()
                and self.documentID is None
            ):
                raise DateCheckError(
                    "Cannot perform analysis on previous date without DocumentID"
                )

            # initialize scan watcher
            scan_watcher = ScanWatch(
                self.experiment_name,
                int(self.ui.inputYear.text()),
                int(self.ui.inputMonth.text()),
                int(self.ui.inputDay.text()),
                analyzer_list=self.analyzer_items,
                ignore_list=self.ignore_list,
                overwrite_previous=self.overwrite_processed_scans,
                upload_to_scanlog=self.ui.checkBoxScanlog.isChecked(),
                documentID=self.documentID,
            )

            # start analysis
            progress_callback.emit("Start ScanWatch.")
            scan_watcher.start(watch_folder_not_exist="raise")

            # let watcher idle while periodically checking for queue items
            processed_scans = scan_watcher.processed_list.copy()
            while is_running():
                scan_watcher.process_queue()

                # check for progress, emit to gui
                processed_scan_diff = list(
                    set(scan_watcher.processed_list) - set(processed_scans)
                )
                if processed_scan_diff:
                    progress_callback.emit(
                        f"Finished analyzing Scan {processed_scan_diff[0]}."
                    )
                    processed_scans = scan_watcher.processed_list.copy()

                time.sleep(wait_time)

            # terminate analysis
            progress_callback.emit("Terminating analysis.")

        except Exception as e:
            error_callback.emit(f"Analysis failed: {type(e)}: {str(e)}")
            error_callback.emit(traceback.format_exc())

    def end_analysis(self) -> None:
        """
        Terminate analysis and restore UI state.

        Cleans up the worker and thread (if present), disables the Stop button,
        re‑enables inputs, and logs status.
        """
        self.log_info_message("Terminating analysis.")

        # clean up worker thread
        if self.worker:
            self.log_info_message("Cleaning up worker.")
            self.cleanup_worker()
        if self.worker_thread:
            self.log_info_message("Cleaning up worker thread.")
            self.cleanup_thread()

        # disable select fields
        to_disable = [self.ui.buttonStop]
        for field in to_disable:
            field.setEnabled(False)

        # enable select fields
        to_enable = self.findChildren(QLineEdit) + [
            self.ui.checkBoxOverwrite,
            self.ui.buttonStart,
        ]
        for input_field in to_enable:
            input_field.setEnabled(True)

    def load_experiment_name_from_config(self):
        """
        Load the current experiment name from the GEECS config.

        Notes
        -----
        - Populates `self.experiment_name`
        - Updates the experiment name line edit
        - Falls back to a placeholder if config cannot be read
        """
        try:
            module = importlib.import_module("geecs_python_api.controls.interface")
            load_config = getattr(module, "load_config")
            config = load_config()
            self.experiment_name = config["Experiment"]["expt"]
            self.log_info_message(
                f"Loaded experiment name '{self.experiment_name}' from config.ini"
            )
        except (TypeError, NameError) as e:
            self.log_error_message(f"{type(e)}: Could not locate configuration file")
        except KeyError as e:
            self.log_error_message(
                f"{type(e)}: No field `Experiment/expt` in config.ini"
            )
        finally:
            display_text = (
                "-- Not Found --"
                if self.experiment_name is None
                else self.experiment_name
            )
            self.ui.lineExperimentName.setText(display_text)

    def write_experiment_name_to_config(self):
        """
        Prompt the user for a new experiment name and write it to the GEECS config.

        Notes
        -----
        - Disabled while an analysis is running
        - Requires GUI restart to take effect
        """
        if self.ui.buttonStop.isEnabled() is True:
            return  # Should not reset the experiment name if a scan is currently running

        text, ok = QInputDialog.getText(
            self, "Change Experiment Name", "Change Experiment Name:", text=""
        )
        if ok:
            config = configparser.ConfigParser()
            config_file_path = Path(
                "~/.config/geecs_python_api/config.ini"
            ).expanduser()
            config.read(config_file_path)
            config.set("Experiment", "expt", text)
            with open(config_file_path, "w") as file:
                config.write(file)
            QMessageBox.information(
                self,
                "Change Experiment Name",
                f"Wrote new experiment name '{text}', must restart window to reload.",
            )
            self.close()

    def initialize_worker(self) -> None:
        """
        Create and start the analysis worker and its thread.

        The worker runs `self.run_analysis` on a new `QThread`, with signals
        connected for progress, errors, and completion.
        """
        # initialize worker object, worker thread
        self.worker = Worker(self.run_analysis)
        self.worker_thread = QThread()

        # move worker to thread
        self.worker.moveToThread(self.worker_thread)

        # connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.end_analysis)
        self.worker.progress.connect(self.handle_worker_progress)
        self.worker.error.connect(self.handle_worker_error)

        # start thread
        self.worker_thread.start()

    def cleanup_thread(self) -> None:
        """
        Stop and dispose the worker thread.

        Ensures the thread is quit and waited on before clearing the reference.
        """
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None

    def cleanup_worker(self) -> None:
        """
        Stop and dispose the worker.

        Calls `stop()` and schedules deletion, then also cleans up the thread.
        """
        if self.worker:
            self.worker.stop()
            self.worker.deleteLater()
            self.worker = None
        self.cleanup_thread()

    def event_overwrite_checkbox_clicked(self, checked) -> None:
        """
        Handle the Overwrite checkbox toggle.

        Parameters
        ----------
        checked : bool
            Whether the checkbox is checked.
        """
        self.overwrite_processed_scans = checked
        self.log_info_message(
            f"Overwrite processed scans status: {self.overwrite_processed_scans}"
        )

    def event_start_button_clicked(self) -> None:
        """Handle Start button click: disable Start, enable Stop, and initiate analysis."""
        # enable/disable gui buttons
        self.ui.buttonStart.setEnabled(False)
        self.ui.buttonStop.setEnabled(True)

        # start analysis
        self.start_analysis()

    def event_stop_button_clicked(self) -> None:
        """Handle Stop button click: end analysis and restore button states."""
        # end analysis
        self.end_analysis()

        # check if buttons are enabled/disabled
        self.ui.buttonStart.setEnabled(
            True
        ) if not self.ui.buttonStart.isEnabled() else None
        self.ui.buttonStop.setEnabled(False) if self.ui.buttonStop.isEnabled() else None

    def event_analysis_activator_button_clicked(self) -> None:
        """
        Open the Analysis Activator dialog.

        Allows users to enable/disable analyzer entries per device for the current run.
        """
        # open dialog
        self.open_analysis_dialog()

    def setup_start_button(self) -> None:
        """Wire the Start button to begin analysis."""
        self.ui.buttonStart.setEnabled(True)
        self.ui.buttonStart.clicked.connect(self.event_start_button_clicked)

    def setup_stop_button(self) -> None:
        """Wire the Stop button to terminate analysis."""
        self.ui.buttonStop.setEnabled(False)
        self.ui.buttonStop.clicked.connect(self.event_stop_button_clicked)

    def setup_overwrite_checkbox(self) -> None:
        """
        Initialize the Overwrite checkbox and connect its toggle handler.

        Notes
        -----
        Uses `event_overwrite_checkbox_clicked` to update state and log the change.
        """
        self.ui.checkBoxOverwrite.setCheckState(self.overwrite_processed_scans)
        self.ui.checkBoxOverwrite.toggled.connect(self.event_overwrite_checkbox_clicked)

    def setup_analysis_activator_button(self) -> None:
        """Wire the Analysis Activator button to open the analyzer selection dialog."""
        self.ui.buttonAnalysisActivator.setEnabled(True)
        self.ui.buttonAnalysisActivator.clicked.connect(
            self.event_analysis_activator_button_clicked
        )

    def setup_date_inputs(self) -> None:
        """Populate the date inputs with today’s year/month/day."""
        today = date.today()
        self.ui.inputYear.setText(str(today.year))
        self.ui.inputMonth.setText(str(today.month))
        self.ui.inputDay.setText(str(today.day))

    def setup_scan_inputs(self) -> None:
        """Initialize scan range inputs (start at 1, empty ignore list)."""
        self.ui.inputStartScan.setText(str(1))
        self.ui.inputStartScan.setEnabled(True)
        self.ui.inputIgnore.setText("")

    def setup_log_display(self) -> None:
        """
        Make the log display read‑only and suitable for streaming messages.

        Notes
        -----
        The internal `write_to_log_display` helper appends text and auto‑scrolls.
        """
        self.ui.logDisplay.setReadOnly(True)

    def toggle_scanlog_upload(self):
        """
        Enable/disable the DocumentID line edit based on the Scanlog checkbox.

        Notes
        -----
        When checked, a DocumentID is required to upload to a non‑current date.
        """
        self.ui.lineDocumentID.setEnabled(self.ui.checkBoxScanlog.isChecked())

    def updateDocumentID(self):
        """
        Update `documentID` from the DocumentID line edit if a non‑empty string is present.

        Sets it back to `None` if the field is empty.
        """
        if line_contents := self.ui.lineDocumentID.text().strip():
            self.documentID = line_contents
        else:
            self.documentID = None

    def set_ignore_list(self) -> None:
        """
        Build the list of scan numbers to ignore from UI inputs.

        The resulting `ignore_list` is the union of:
        - All scans < start scan (1..start-1)
        - Any comma-separated values in the "ignore" line edit

        Raises
        ------
        ValueError
            If either field contains non-integer values
        UnexpectedError
            For any other unexpected parsing errors
        """
        try:
            # Get start scan value
            start_scan_text = self.ui.inputStartScan.text().strip()
            if not start_scan_text:
                raise ValueError("Start scan value cannot be empty.")

            # ignore scans before start scan
            start_scan = max(1, int(start_scan_text))
            ignore_list: List[int] = list(range(1, int(start_scan)))

            # append any from 'ignore' text box
            ignore_text = self.ui.inputIgnore.text().strip()
            if ignore_text:
                try:
                    additional_ignores = [
                        num
                        for num in [
                            int(text.strip())
                            for text in ignore_text.split(",")
                            if text.strip()
                        ]
                        if num > 0
                    ]
                    ignore_list.extend(additional_ignores)

                except ValueError:
                    raise ValueError(
                        "Invalid ignore list format. Please enter integers separated by commas."
                    )

            # remove duplicates and sort
            self.ignore_list = sorted(set(ignore_list)) if ignore_list else None

        except ValueError:
            raise
        except Exception as e:
            raise UnexpectedError(
                original_exception=e,
                custom_message="Unexpected error in 'set_ignore_list'.",
            )

    def handle_worker_error(self, message: str) -> None:
        """Append an error message from the worker to the GUI log."""
        self.log_warning_message(message)

    def handle_worker_progress(self, message: str) -> None:
        """
        Handle progress messages from the worker.

        Parameters
        ----------
        message : str
            Human‑readable status string
        """
        self.log_info_message(message)

    def log_info_message(self, text: str) -> None:
        """Write an info‑level message to the GUI log."""
        self.write_to_log_display(f"INFO : {text}")

    def log_warning_message(self, text: str) -> None:
        """Write a warning‑level message to the GUI log."""
        self.write_to_log_display(f"WARNING : {text}")

    def log_error_message(self, text: str) -> None:
        """Write an error‑level message to the GUI log."""
        self.write_to_log_display(f"{text}")

    def write_to_log_display(self, text: str) -> None:
        """
        Append text to the GUI log and auto‑scroll.

        Parameters
        ----------
        text : str
            Line to append
        """
        # write to log display
        self.ui.logDisplay.append(text)

        # auto-scroll display to newest text
        self.ui.logDisplay.verticalScrollBar().setValue(
            self.ui.logDisplay.verticalScrollBar().maximum()
        )

    def open_analysis_dialog(self) -> None:
        """
        Open the analyzer activation dialog and apply user selections.

        The dialog shows one row per analyzer configuration (class + device).
        After acceptance, the corresponding `is_active` fields are updated in `analyzer_items`.
        """
        # get list of analyses
        device_default = ActivatorTuple._field_defaults.get("device")
        analysis_list = [
            ActivatorTuple(
                analyzer=item.scan_analyzer_class.__name__,
                device=item.device_name or device_default,
                is_active=item.is_active,
            )
            for item in self.analyzer_items
        ]

        # open dialog
        dialog = AnalysisDialog(analysis_list, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            dialog_output = dialog.get_analysis_states()

        # update current states to analyzer_items
        for ind, analyzer in enumerate(self.analyzer_items):
            analysis_name = analyzer.scan_analyzer_class.__name__
            device_name = analyzer.device_name or device_default
            for item in dialog_output:
                if (
                    item.analyzer == analysis_name
                    and item.device == device_name
                    and item.is_active != analyzer.is_active
                ):
                    self.analyzer_items[ind] = analyzer._replace(
                        is_active=item.is_active
                    )

    def closeEvent(self, event) -> None:
        """
        Ensure the worker thread is terminated on window close.

        Parameters
        ----------
        event : QCloseEvent
            The Qt window close event
        """
        self.write_to_log_display("closeEvent: triggered")
        if self.worker_thread and self.worker_thread.isRunning():
            self.write_to_log_display("closeEvent: closing thread")
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()


class Worker(QObject):
    """
    Background worker that runs the analysis function on a `QThread`.

    Signals
    -------
    error : pyqtSignal(str)
        Emitted when the analysis raises an exception or error.
    progress : pyqtSignal(str)
        Emitted for human‑readable progress updates.
    finished : pyqtSignal()
        Emitted when the analysis completes or stops.

    Notes
    -----
    `analysis_func` is expected to take `(progress_signal, error_signal, is_running_callable)`
    and cooperatively check `is_running()` to know when to exit.
    """

    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, analysis_func: AnalysisFunction) -> None:
        """
        Construct worker.

        Parameters
        ----------
        analysis_func : Callable
            Function invoked on the worker thread to perform analysis
        """
        super().__init__()
        self.analysis_func = analysis_func
        self._is_running = True

    def stop(self) -> None:
        """
        Request the worker to stop at the next check point.

        Notes
        -----
        This sets an internal flag read by `is_running()`; the analysis function
        should check this periodically and return to finish gracefully.
        """
        self.progress.emit("Worker.stop() called.")
        self._is_running = False

    def is_running(self) -> bool:
        """
        Return whether the worker should continue running.

        Returns
        -------
        bool
            True if still running; False if stop was requested
        """
        return self._is_running

    @pyqtSlot()
    def run(self) -> None:
        """
        Entry point executed on the worker thread.

        Calls the provided analysis function, forwarding the `progress` and `error`
        signals plus the `is_running()` method. Emits `finished` on completion
        whether successful or with errors.
        """
        try:
            self._is_running = True
            while self.is_running():
                set_pdb_trace()

                # self.analysis_func(self.progress, lambda: self._is_running)
                self.analysis_func(self.progress, self.error, self.is_running)
                break  # exit after one iteration

            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()


# =============================================================================
# %% error handling


class CustomError(Exception):
    """
    Base class for custom errors with consistent formatting.

    Parameters
    ----------
    custom_message : str, optional
        Message to display before the error type
    """

    def __init__(self, custom_message: str = None) -> None:
        super().__init__(custom_message)
        self.custom_message = custom_message

    def __str__(self) -> str:
        """
        Return a formatted message including the class name.

        Returns
        -------
        str
        """
        return (
            f"ERROR: {self.custom_message if self.custom_message else ''}\n"
            f"Type: {self.get_class_name()}"
        )

    @classmethod
    def get_class_name(cls) -> str:
        """Return the name of the error class."""
        return cls.__name__


class AnalysisRunningError(CustomError):
    """Raised when a start is attempted while analysis is already running."""

    pass


class DateCheckError(Exception):
    """Raised when a past-date analysis requires a DocumentID but none is provided."""

    def __init__(self, message="Date is not valid without a given documentID"):
        self.message = message
        super().__init__(self.message)


class UnexpectedError(CustomError):
    """
    Wrapper to report unexpected exceptions with the original error type and message.

    Parameters
    ----------
    original_exception : Exception, optional
        The underlying exception
    custom_message : str, optional
        Contextual message
    """

    def __init__(
        self, original_exception: Exception = None, custom_message: str = None
    ) -> None:
        super().__init__(custom_message=custom_message)
        self.original_exception = original_exception

    def __str__(self) -> str:
        """
        Return a formatted message including type and details.

        Returns
        -------
        str
        """
        error_type = (
            "Unknown"
            if self.original_exception is None
            else self.get_original_exception_name()
        )
        error_details = (
            "" if self.original_exception is None else str(self.original_exception)
        )

        return (
            f"ERROR: {self.custom_message if self.custom_message else ''}\n"
            f"Type: {error_type}\n"
            f"Details: {error_details}"
        )

    def get_original_exception_name(self):
        """Return the class name of the original exception."""
        return type(self.original_exception).__name__


# =============================================================================
# %% functions


def set_pdb_trace():
    """
    Drop into pdb if DEBUG_MODE is true.

    Notes
    -----
    Helpful for stepping into the worker loop without modifying code paths.
    """
    if DEBUG_MODE:
        pdb.set_trace()


# =============================================================================
# %% routine

# =============================================================================
# %% execute
if __name__ == "__main__":
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)
    window = ScAnalyzerWindow()
    window.show()

    sys.exit(app.exec_())
