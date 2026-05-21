"""Entry point for the LiveWatch GUI.

Run from a terminal::

    python -m LiveWatchGUI.main

Or directly::

    python LiveWatchGUI/main.py

Optional CLI flags allow you to set the initial log level.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including logging controls.
    """
    p = argparse.ArgumentParser(description="Launch the LiveWatch Analysis GUI")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Initial console log level (default: INFO).",
    )
    return p.parse_args()


def main() -> int:
    """Launch the LiveWatch GUI application.

    Returns
    -------
    int
        Qt application exit code.
    """
    args = _parse_args()

    # Configure root logging so messages reach the console as well as the GUI
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Import Qt after logging is set up
    from PyQt5.QtWidgets import QApplication

    from LiveWatchGUI.live_watch_window import LiveWatchWindow

    app = QApplication(sys.argv)
    app.setApplicationName("LiveWatch Analysis")

    window = LiveWatchWindow()
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
