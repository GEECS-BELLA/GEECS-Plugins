"""GEECS-Console entry point: QApplication + MainWindow."""

import sys


def main() -> int:
    """Launch the console.

    Returns
    -------
    int
        The Qt event-loop exit code.
    """
    from PySide6.QtWidgets import QApplication

    from geecs_console.app.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
