import sys
from PyQt5.QtWidgets import QApplication
from geecs_scanner.utils import exception_hook
from geecs_scanner.app.GEECSScanner import GEECSScannerWindow


if __name__ == '__main__':
    """Launches the GEECS Scanner GUI"""
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)

    application = GEECSScannerWindow()
    application.show()
    application.raise_()
    application.activateWindow()

    sys.exit(app.exec_())
