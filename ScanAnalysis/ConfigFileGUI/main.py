"""Entry point for the Config File Editor GUI.

Run from a terminal::

    python -m ConfigFileGUI.main

Or directly::

    python ConfigFileGUI/main.py

Optional CLI flags allow you to set the initial log level and specify
the directory containing device configuration YAML files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including logging controls and config directory.
    """
    p = argparse.ArgumentParser(description="Launch the Config File Editor GUI")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Initial console log level (default: INFO).",
    )
    p.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help=(
            "Path to the image_analysis_configs directory containing "
            "device configuration YAML files."
        ),
    )
    return p.parse_args()


def main() -> int:
    """Launch the Config File Editor GUI application.

    Returns
    -------
    int
        Qt application exit code.
    """
    args = _parse_args()

    # Configure root logging so messages reach the console
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve config directory if provided
    config_dir: Optional[Path] = None
    if args.config_dir is not None:
        config_dir = Path(args.config_dir).resolve()

    # Import Qt after logging is set up
    from PyQt5.QtWidgets import QApplication

    from ConfigFileGUI.config_editor_window import ConfigEditorWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Config File Editor")

    window = ConfigEditorWindow(config_dir=config_dir)
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
