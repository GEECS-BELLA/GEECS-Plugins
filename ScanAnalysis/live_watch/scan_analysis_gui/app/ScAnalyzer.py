"""
ScAnalyzer GUI Logic

Authors:
Kyle Jensen (kjensen11, kjensen@lbl.gov)
"""
# =============================================================================
# %% imports
from typing import Optional, Callable, List
from importlib.metadata import version

import sys
import time
from datetime import date

from PyQt5.QtWidgets import QMainWindow, QApplication, QLineEdit
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from live_watch.scan_analysis_gui.app.gui.ScAnalyzer_ui import Ui_MainWindow
from live_watch.scan_watch import ScanWatch
from live_watch.scan_analysis_gui.utils.exceptions import exception_hook
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

    def __init__(self) -> None:
        super().__init__()

        # create instance of Ui_MainWindow, setup elements from .ui file
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # set title of gui window
        self.setWindowTitle(f"GEECS ScAnalyzer - {CURRENT_VERSION}")

        # set up buttons
        self.setup_overwrite_button()
        self.setup_start_button()
        self.setup_stop_button()

        # set up text edits
        self.setup_date_inputs()
        self.setup_scan_inputs()
        self.ignore_list = None

        # set up gui log to display output
        self.setup_log_display()

        # initialize worker information
        self.worker: Optional[Worker] = None
        self.worker_thread: Optional[QThread] = None

    def start_analysis(self) -> None:
        '''
        Prepare to run analysis. 
            - Handle GUI objects.
            - Set ignore list.
            - Initialize worker and worker thread.
        '''
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
            to_disable = self.findChildren(QLineEdit) + [self.ui.buttonOverwrite, self.ui.buttonStart]
            for field in to_disable:
                field.setEnabled(False)

            # initialize worker and thread
            self.initialize_worker()

        except Exception as e:
            self.log_error_message(str(e))
            self.end_analysis()

    def run_analysis(self,
                     progress_callback: pyqtSignal,
                     error_callback: pyqtSignal,
                     is_running: RunningCheck,
                     wait_time: float = 0.5) -> None:
        '''
        Run Scan Analysis by initializing and running ScanWatcher.
        Passed to Worker and run on worker thread.

        :param progress_callback: Signal to communicate from worker thread.
        :type progress_callback: Callable[[str], None]
        :param is_running: Function that returns a boolean to indicate whether the worker thread is or should be running.
        :type is_running: Callable[[], bool]
        :param wait_time: Sleep time between queue checks, defaults to 0.5
        :type wait_time: float, optional
        :return: No return.
        :rtype: None
        '''
        try:
            # initialize scan watcher
            scan_watcher = ScanWatch('Undulator',
                                     int(self.ui.inputYear.text()),
                                     int(self.ui.inputMonth.text()),
                                     int(self.ui.inputDay.text()),
                                     ignore_list=self.ignore_list,
                                     overwrite_previous=True)

            # start analysis
            progress_callback.emit("Start ScanWatch.")
            scan_watcher.start(watch_folder_not_exist='raise')

            # let watcher idle while periodically checking for queue items
            processed_scans = scan_watcher.processed_list.copy()
            while is_running():
                scan_watcher.process_queue()

                # check for progress, emit to gui
                processed_scan_diff = list(set(scan_watcher.processed_list) - set(processed_scans))
                if processed_scan_diff:
                    progress_callback.emit(f"Finished analyzing Scan {processed_scan_diff[0]}.")
                    processed_scans = scan_watcher.processed_list.copy()

                time.sleep(wait_time)

            # terminate analysis
            progress_callback.emit("Terminating analysis.")

        except Exception as e:
            error_callback.emit(f"Analysis failed: {str(e)}")

    def end_analysis(self) -> None:
        '''
        Termination of analysis. Clean up worker and worker thread. Handle GUI object status.

        :return: None
        :rtype: None
        '''
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
        to_enable = self.findChildren(QLineEdit) + [self.ui.buttonOverwrite, self.ui.buttonStart]
        for input_field in to_enable:
            input_field.setEnabled(True)

    def initialize_worker(self) -> None:
        '''
        Initialize Worker and worker thread. Move Worker to worker thread and connect signals. Start worker thread.
        :return: No return.
        :rtype: None
        '''
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
        '''
        Clean up worker thread.
        :return: No return.
        :rtype: None
        '''
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None

    def cleanup_worker(self) -> None:
        '''
        Clean up worker.
        :return: No return.
        :rtype: None

        '''
        if self.worker:
            self.worker.stop()
            self.worker.deleteLater()
            self.worker = None
        self.cleanup_thread()

    def event_overwrite_button_clicked(self) -> None:
        '''
        Actions performed when Overwrite button is clicked.

        Returns
        -------
        None
        '''
        self.ui.inputStartScan.setEnabled(self.ui.buttonOverwrite.isChecked())

    def event_start_button_clicked(self) -> None:
        '''
        Actions performed when Start button is clicked.
        :return: No return.
        :rtype: None

        '''
        # enable/disable gui buttons
        self.ui.buttonStart.setEnabled(False)
        self.ui.buttonStop.setEnabled(True)

        # start analysis
        self.start_analysis()

    def event_stop_button_clicked(self) -> None:
        '''
        Actions performed when Stop button is clicked.
        :return: No return.
        :rtype: None

        '''
        # end analysis
        self.end_analysis()

        # check if buttons are enabled/disabled
        self.ui.buttonStart.setEnabled(True) if not self.ui.buttonStart.isEnabled() else None
        self.ui.buttonStop.setEnabled(False) if self.ui.buttonStop.isEnabled() else None

    def setup_start_button(self) -> None:
        self.ui.buttonStart.setEnabled(True)
        self.ui.buttonStart.clicked.connect(self.event_start_button_clicked)

    def setup_stop_button(self) -> None:
        self.ui.buttonStop.setEnabled(False)
        self.ui.buttonStop.clicked.connect(self.event_stop_button_clicked)

    def setup_overwrite_button(self) -> None:
        '''
        Setup for Overwrite Processed List button.

        Returns
        -------
        None
        '''
        self.ui.buttonOverwrite.setCheckable(True)
        self.ui.buttonOverwrite.clicked.connect(self.event_overwrite_button_clicked)
        self.ui.buttonOverwrite.setStyleSheet("""
                                              QPushButton:checked {
                                                  background-color: #d0d0d0;
                                                  border: 1px solid #808080;
                                                  color: #404040;
                                              }
                                              QPushButton:disabled {
                                                  background-color: #d0d0d0;
                                                  border: 1px solid #808080;
                                                  color: #404040;
                                              }
                                              QPushButton {
                                                  background-color: #ffffff;
                                                  border: 1px solid #404040;
                                                   color: black;
                                               }
                                              """)

    def setup_date_inputs(self) -> None:
        # set default date
        today = date.today()
        self.ui.inputYear.setText(str(today.year))
        self.ui.inputMonth.setText(str(today.month))
        self.ui.inputDay.setText(str(today.day))

    def setup_scan_inputs(self) -> None:
        self.ui.inputStartScan.setText(str(1))
        self.ui.inputStartScan.setEnabled(False)
        self.ui.inputIgnore.setText('')

    def setup_log_display(self) -> None:
        self.ui.logDisplay.setReadOnly(True)

    def set_ignore_list(self) -> None:
        """
        Sets the ignore_list attribute with scan numbers to be ignored.
        
        Combines scan numbers from two sources:
        1. Range from 0 to start scan number (exclusive)
        2. Additional scan numbers from ignore input field (comma-separated)
        
        The final list is converted to a unique set of integers.
        
        Raises:
            ValueError: If input values cannot be converted to integers
            AttributeError: If UI elements are not properly initialized
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
                        num for num in [
                        int(text.strip()) 
                        for text in ignore_text.split(',') 
                        if text.strip()
                        ] if num > 0
                    ]
                    ignore_list.extend(additional_ignores)

                except ValueError:
                    raise ValueError("Invalid ignore list format. Please enter integers separated by commas.")

            # remove duplicates and sort
            self.ignore_list = sorted(set(ignore_list)) if ignore_list else None

        except ValueError:
            raise
        except Exception as e:
            raise UnexpectedError(original_exception=e,
                                  custom_message="Unexpected error in 'set_ignore_list'.")

    def handle_worker_error(self, message: str) -> None:
        # display error in GUI log
        self.log_warning_message(message)

    def handle_worker_progress(self, message: str) -> None:
        self.log_info_message(message)

    def log_info_message(self, text: str) -> None:
        self.write_to_log_display(f"INFO : {text}")

    def log_warning_message(self, text: str) -> None:
        self.write_to_log_display(f"WARNING : {text}")

    def log_error_message(self, text: str) -> None:
        self.write_to_log_display(f"{text}")

    def write_to_log_display(self, text: str) -> None:
        '''
        Pass text to display window on GUI.

        Note: It was noted that the GUI may slow down if lots of text is logged.
        It might be good to terminate old text (only store so much on gui window).
        All text could be logged in an external text file for reference or debugging.

        :param text: String to print on display.
        :type text: str
        :return: No return.
        :rtype: None

        '''
        # write to log display
        self.ui.logDisplay.append(text)

        # auto-scroll display to newest text
        self.ui.logDisplay.verticalScrollBar().setValue(
            self.ui.logDisplay.verticalScrollBar().maximum())

    def closeEvent(self, event) -> None:
        '''
        Save termination in the event of GUI window closure.
        Triggers automatically and terminates any existing worker threads.
        
        Args:
            event (QCloseEvent): Window close event from PyQt
        
        Returns:
            None
        '''
        self.write_to_log_display('closeEvent: triggered')
        if self.worker_thread and self.worker_thread.isRunning():
            self.write_to_log_display('closeEvent: closing thread')
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()

class Worker(QObject):
    """Worker class for handling threaded operations.
    
    Manages the execution of analysis functions in a separate thread,
    providing progress updates and error handling.
    """
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, analysis_func: AnalysisFunction) -> None:
        """Initialize the Worker with an analysis function.

        Args:
            analysis_func: Function to be executed in the worker thread.
                         Should accept progress callback and running check functions.
        """
        super().__init__()
        self.analysis_func = analysis_func
        self._is_running = True

    def stop(self) -> None:
        """Stop the worker's execution.
        
        Sets the running flag to False, which will terminate the analysis
        at the next check point.
        """
        self.progress.emit("Worker.stop() called.")
        self._is_running = False

    def is_running(self) -> bool:
        """Check if the worker is currently running.

        Returns:
            bool: True if the worker is running, False otherwise.
        """
        return self._is_running

    @pyqtSlot()
    def run(self) -> None:
        """Execute the analysis function in the worker thread.
        
        Runs the analysis function while handling exceptions and
        managing the running state. Emits signals for progress
        updates, errors, and completion.

        Signals:
            progress: Emits status messages as strings
            error: Emits error messages if exceptions occur
            finished: Emits when the analysis is complete
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
    def __init__(self, custom_message: str = None) -> None:
        """
        Initialize custom error with optional message.
        
        Args:
            custom_message (str, optional): Custom message to prepend
        """
        super().__init__(custom_message)
        self.custom_message = custom_message
        
    def __str__(self) -> str:
        """
        Returns formatted error message.
        
        Returns:
            str: Formatted error message including exception type and details
        """
        return (
            f"ERROR: {self.custom_message if self.custom_message else ''}\n"
            f"Type: {self.get_class_name()}"
        )

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

class AnalysisRunningError(CustomError):
    """Error raised when analysis is already running."""
    pass

class UnexpectedError(CustomError):
    """Custom error handler for unexpected exceptions."""
    
    def __init__(self, original_exception: Exception = None, custom_message: str = None) -> None:
        super().__init__(custom_message=custom_message)
        self.original_exception = original_exception

    def __str__(self) -> str:
        """
        Returns formatted error message.
        
        Returns:
            str: Formatted error message including exception type and details
        """
        error_type = "Unknown" if self.original_exception is None else self.get_original_exception_name()
        error_details = "" if self.original_exception is None else str(self.original_exception)

        return (
            f"ERROR: {self.custom_message if self.custom_message else ''}\n"
            f"Type: {error_type}\n"
            f"Details: {error_details}"
        )

    def get_original_exception_name(self):
        return type(self.original_exception).__name__
# =============================================================================
# %% functions

def set_pdb_trace():
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
