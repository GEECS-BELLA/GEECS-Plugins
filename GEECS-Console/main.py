"""GEECS-Console entry point: QApplication + MainWindow."""

import argparse
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

#: Console log directory — a *user config* dir, so creating it (parents
#: included) is fine; the scan-folder invariant applies only to
#: ``scans/ScanNNN/`` trees, which this program never creates.
_LOG_DIR = Path("~/.config/geecs_console/logs").expanduser()
_LOG_FILE = _LOG_DIR / "console.log"
_LOG_MAX_BYTES = 2_000_000
_LOG_BACKUP_COUNT = 3


def configure_logging(level_name: str = "INFO") -> None:
    """Set up root logging: stderr plus a rotating file under the config dir.

    Parameters
    ----------
    level_name : str, optional
        Root logging level name (``--log-level``); default INFO.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    # The Tiled health probe issues an httpx GET every poll; at INFO httpx
    # logs one line per request (every 5 s, forever).  Quiet it to WARNING.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:  # an unwritable config dir must not kill the GUI
        root.warning("File logging unavailable (%s): %s", _LOG_FILE, exc)


def main(argv: Optional[list[str]] = None) -> int:
    """Launch the console.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).  Unrecognized
        arguments pass through to Qt.

    Returns
    -------
    int
        The Qt event-loop exit code.
    """
    parser = argparse.ArgumentParser(description="GEECS operator console")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Root logging level (default: INFO)",
    )
    args, qt_args = parser.parse_known_args(argv)
    configure_logging(args.log_level)

    from PySide6.QtWidgets import QApplication

    from geecs_console.app.main_window import MainWindow
    from geecs_console.services.device_panel import GatewayDevicePanel
    from geecs_console.services.health import GatewayTiledDbHealth
    from geecs_console.services.optimization import warm_up_optimization_stack

    app = QApplication([sys.argv[0], *qt_args])
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
    # With the `optimization` extra installed, pre-import the torch/botorch/
    # xopt stack on a daemon thread so the first optimize submission doesn't
    # freeze for its tens-of-seconds cold import (no-op without the extra;
    # started after the window is up so nothing races Qt initialization).
    warm_up_optimization_stack()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
