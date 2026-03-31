"""Entry point: python -m config_manager  or  config-manager (poetry script)."""

import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> None:
    """Launch the Config Manager GUI."""
    # HiDPI support (helps on Retina / high-DPI displays)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("GEECS Config Manager")
    app.setOrganizationName("BELLA")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
