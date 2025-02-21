
import sys
from PyQt5.QtWidgets import QApplication

from live_watch.scan_analysis_gui.utils.exceptions import exception_hook
from live_watch.scan_analysis_gui.app.ScAnalyzer import ScAnalyzerWindow

if __name__ == '__main__':
    """Launches the Scan Analysis GUI"""
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)

    application = ScAnalyzerWindow()
    application.show()
    application.raise_()
    application.activateWindow()

    sys.exit(app.exec_())
