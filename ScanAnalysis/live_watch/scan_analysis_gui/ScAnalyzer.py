"""
ScAnalyzer GUI Logic

Authors:
Kyle Jensen (kjensen11, kjensen@lbl.gov)
"""
# =============================================================================
# %% imports
import sys
import time
import traceback
from datetime import date

from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QThread, pyqtSignal

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

        # set the default values for line edits
        self.set_default_inputs()

        # initialize worker information
        self.worker = None

        # connect buttons to functions
        self.ui.buttonStart.clicked.connect(self.start_analysis)

    def start_analysis(self) -> None:

        # check for existing worker thread
        if self.worker is not None:
            self.stop_worker()

        # create new worker for this analysis
        self.worker = Worker(self.run_analysis)

        # connect signals
        self.worker.progress.connect(self.log_progress)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.end_analysis)

        # disable start button
        self.ui.buttonStart.setEnabled(False)

        # start analysis
        self.worker.start()

    def run_analysis(self) -> None:

        date_str = f"{self.ui.inputMonth.text()}-{self.ui.inputDay.text()}-{self.ui.inputYear.text()}"

        self.log(f"Starting fake analysis for {date_str}...", level='info')
        time.sleep(0.5)
        self.log('Analyzing fake scan 1...', level='info')
        time.sleep(0.5)
        self.log('Analyzing fake scan 2...', level='info')
        time.sleep(0.5)
        self.log('Analyzing fake scan 3...', level='info')
        time.sleep(0.5)
        self.log('Done', level='info')

    def end_analysis(self):
        self.ui.logDisplay.append("Analysis completed.")
        self.ui.buttonStart.setEnabled(True)

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait() # wait for thread to finish

    def cleanup_worker(self):
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def set_default_inputs(self) -> None:

        # set default scan date
        today = date.today()
        self.ui.inputYear.setText(str(today.year))
        self.ui.inputMonth.setText(str(today.month))
        self.ui.inputDay.setText(str(today.day))

    def log_progress(self, message: str) -> None:
        self.ui.logDisplay.append(message)

    def handle_error(self, error_message):
        self.ui.logDisplay.append(f"ERROR: {error_message}")

class Worker(QThread):
    progress = pyqtSignal(str) # signal to emit strings
    error = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, analysis_func, *args, **kwargs):
        super().__init__()

        # store analysis functions and its arguments
        self.analysis_func = analysis_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            # call the analysis function with stored arguments
            self.analysis_func(*self.args, **self.kwargs)

        except:
            pass

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

    """Launches the ScAnalyzer GUI"""
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)

    window = ScAnalyzerWindow()
    window.show()

    sys.exit(app.exec_())
