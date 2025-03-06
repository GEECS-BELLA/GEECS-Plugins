import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from geecs_scanner.utils import exception_hook
from geecs_scanner.app import GEECSScannerWindow
from geecs_scanner.app.gui import resources_rc


if __name__ == '__main__':
    """Launches the GEECS Scanner GUI"""
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)

    app.setWindowIcon(QIcon(":/application_icon.ico"))

    application = GEECSScannerWindow()
    application.show()
    application.raise_()
    application.activateWindow()

    sys.exit(app.exec_())
