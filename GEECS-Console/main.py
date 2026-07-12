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
    from geecs_console.services.device_panel import GatewayDevicePanel
    from geecs_console.services.health import GatewayTiledDbHealth

    app = QApplication(sys.argv)
    # Inject the real seams in production; tests/offline fall back to the
    # stub defaults.  The health probe is background-polled (never on the GUI
    # thread) and lazily imports aioca / httpx / GeecsDb inside poll().  The
    # device panel hosts its CA monitor on one persistent daemon-thread event
    # loop and dispatches blocking :SP puts off the GUI thread.
    window = MainWindow(
        health=GatewayTiledDbHealth(),
        device_panel=GatewayDevicePanel(),
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
