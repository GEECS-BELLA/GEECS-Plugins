"""
Entry point for the GEECS Scanner GUI.

Run from a terminal (e.g., ``python main.py``). Optional CLI flags allow you
to tweak logging behavior without editing code.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

from geecs_scanner.utils import exception_hook
from geecs_scanner.logging_setup import ensure_logging


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including logging controls.
    """
    p = argparse.ArgumentParser(description="Launch the GEECS Scanner GUI")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console log level (file log still rotates & captures all levels per config).",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for global logs (overrides $GEECS_LOG_DIR).",
    )
    p.add_argument(
        "--no-console",
        action="store_true",
        help="Disable console logging (file logging remains enabled).",
    )
    p.add_argument(
        "--max-bytes",
        type=int,
        default=10_000_000,
        help="Max bytes before rotating the global log file.",
    )
    p.add_argument(
        "--backup-count",
        type=int,
        default=5,
        help="Number of rotated log files to keep.",
    )
    return p.parse_args()


def _wrap_excepthook():
    """
    Route uncaught exceptions to logging, then delegate to the existing hook.

    Notes
    -----
    This preserves your current behavior while ensuring crashes are recorded
    in the central log.
    """

    def _hook(exc_type, exc, tb):
        logging.getLogger("geecs_scanner").exception(
            "Uncaught exception", exc_info=(exc_type, exc, tb)
        )
        exception_hook(exc_type, exc, tb)

    sys.excepthook = _hook


def main() -> int:
    """
    Launch the GEECS Scanner GUI.

    This function initializes centralized logging, installs an exception hook,
    and starts the Qt application loop.

    Returns
    -------
    int
        Qt application exit code.
    """
    args = _parse_args()

    # Initialize logging early
    ensure_logging(
        log_dir=str(args.log_dir) if args.log_dir else None,
        level=getattr(logging, args.log_level),
        console=not args.no_console,
        max_bytes=args.max_bytes,
        backup_count=args.backup_count,
    )

    _wrap_excepthook()

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(":/application_icon.ico"))

    # Import after logging is ready
    from geecs_scanner.app import GEECSScannerWindow

    application = GEECSScannerWindow()
    application.show()
    application.raise_()
    application.activateWindow()

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
