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
    from geecs_console.services.health import GatewayTiledDbHealth

    app = QApplication(sys.argv)
    # Inject the real health probe in production; tests/offline fall back to the
    # all-unknown StubHealth default.  The probe is background-polled (never on
    # the GUI thread) and lazily imports aioca / httpx / GeecsDb inside poll().
    window = MainWindow(health=GatewayTiledDbHealth())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
