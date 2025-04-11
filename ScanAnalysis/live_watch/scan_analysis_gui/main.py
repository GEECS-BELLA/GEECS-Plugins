
import sys
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, 'C:\\GEECS\\Developers Version\\source\\GEECS-Plugins\\ScanAnalysis')

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
