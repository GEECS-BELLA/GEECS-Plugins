"""
ScAnalyzer GUI Logic

Authors:
Kyle Jensen (kjensen11, kjensen@lbl.gov)
"""
# =============================================================================
# %% imports
from typing import Optional
import sys
import time
import traceback
from datetime import date

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from ScAnalyzer_ui import Ui_MainWindow
# =============================================================================
# %% global variables

CURRENT_VERSION = "v0.0"
# =============================================================================
# %% class

class ScAnalyzerWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # create instance of Ui_MainWindow, setup elements from .ui file
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # set title of gui window
        self.setWindowTitle(f"GEECS ScAnalyzer - {CURRENT_VERSION}")

        # set up gui log to display output
        self.ui.logDisplay.setReadOnly(True)

        # set initial button status
        self.ui.buttonStart.setEnabled(True)
        self.ui.buttonStop.setEnabled(False)

        # set the default values for line edits
        self.set_default_inputs()

        # initialize worker information
        self.worker: Optional[Worker] = None
        self.worker_thread: Optional[QThread] = None

        # connect buttons to functions
        self.ui.buttonStart.clicked.connect(self.start_analysis)
        self.ui.buttonStop.clicked.connect(self.end_analysis)

    def start_analysis(self) -> None:
        # thread worker status check
        if self.worker_thread and self.worker_thread.isRunning():
            self.ui.logDisplay.append("Analysis is already running!")
            return

        # disable start button while analysis is running
        self.ui.buttonStart.setEnabled(False)
        self.ui.buttonStop.setEnabled(True)

        # initialize worker and thread
        self.initialize_worker()

    def run_analysis(self, progress_callback, is_running) -> None:
        try:
            date_str = f"{self.ui.inputMonth.text()}-{self.ui.inputDay.text()}-{self.ui.inputYear.text()}"
            progress_callback.emit(f"Starting fake analysis for {date_str}...")

            for i in range(100):
                if not is_running():
                    progress_callback.emit("Analysis stopped by user.")
                    return
                progress_callback.emit(f"Analyzing fake scan {i}...")
                time.sleep(0.5)
            progress_callback.emit('Analysis complete.')

        except Exception as e:
            progress_callback.emit(f"Analysis failed: {str(e)}")

    def end_analysis(self):
        print("End analysis called")
        self.ui.buttonStart.setEnabled(True)
        self.ui.buttonStop.setEnabled(False)

        if self.worker:
            print("Stopping worker")
            self.cleanup_worker()
        if self.worker_thread:
            self.cleanup_thread()

    def initialize_worker(self):
        
        # initialize worker object, worker thread
        self.worker = Worker(self.run_analysis)
        self.worker_thread = QThread()

        # move worker to thread
        self.worker.moveToThread(self.worker_thread)

        # connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.end_analysis)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.handle_error)

        # start thread
        self.worker_thread.start()

    def cleanup_thread(self):
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None

    def cleanup_worker(self):
        if self.worker:
            self.worker.stop()
            self.worker.deleteLater()
            self.worker = None
        self.cleanup_thread()

    def update_progress(self, progress_string):
        self.ui.logDisplay.append(progress_string)
        self.ui.logDisplay.verticalScrollBar().setValue(
            self.ui.logDisplay.verticalScrollBar().maximum()
            )

    def handle_error(self, error_message):
        self.ui.logDisplay.append(f"Error: {error_message}")

    def set_default_inputs(self) -> None:

        # set default scan date
        today = date.today()
        self.ui.inputYear.setText(str(today.year))
        self.ui.inputMonth.setText(str(today.month))
        self.ui.inputDay.setText(str(today.day))

    def closeEvent(self, event):
        # properly close thread if window is closed
        # triggers automatically
        self.update_progress('closeEvent: triggered')
        if self.worker_thread and self.worker_thread.isRunning():
            self.update_progress('closeEvent: closing thread')
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, analysis_func):
        super().__init__()
        self.analysis_func = analysis_func
        self._is_running = True

    def stop(self):
        print("Worker.stop() called.")
        self._is_running = False

    def is_running(self):
        return self._is_running

    @pyqtSlot()
    def run(self):
        try:
            self._is_running = True
            while self._is_running:
                self.analysis_func(self.progress, lambda: self._is_running)
                break # exit after one iteration
            if not self._is_running:
                self.progress.emit('Worker stopped by user.')
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")
        finally:
            self.finished.emit()



# =============================================================================
# %% routine

def exception_hook(exctype, value, tb):
    """
    Global wrapper to print out tracebacks of python errors during the 
    execution of a PyQT window.

    :param exctype: Exception Type
    :param value: Value of the exception
    :param tb: Traceback
    """
    print("An error occurred:")
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)
    sys.exit(1)

# =============================================================================
# %% execute
if __name__ == "__main__":

    sys.excepthook = exception_hook
    app = QApplication(sys.argv)
    window = ScAnalyzerWindow()
    window.show()

    sys.exit(app.exec_())
