"""Entry point for the Scan Config File Editor GUI.

Run from a terminal::

    python -m ConfigFileGUI.main

Or directly::

    python ConfigFileGUI/main.py

Optional CLI flags set the log level and the initial
``scan_analysis_configs/`` root directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Launch the Scan Config Editor GUI")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Initial console log level (default: INFO).",
    )
    p.add_argument(
        "--scan-config-dir",
        type=str,
        default=None,
        help=(
            "Path to the scan_analysis_configs/ root directory containing "
            "the analyzers/ and groups/ subtrees."
        ),
    )
    return p.parse_args()


def main() -> int:
    """Launch the Scan Config Editor GUI application.

    Returns
    -------
    int
        Qt application exit code.
    """
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    scan_config_dir: Optional[Path] = None
    if args.scan_config_dir is not None:
        scan_config_dir = Path(args.scan_config_dir).resolve()

    # Import Qt after logging is set up
    from PyQt5.QtWidgets import QApplication

    from ConfigFileGUI.config_editor_window import ConfigEditorWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Scan Config Editor")

    window = ConfigEditorWindow(scan_config_dir=scan_config_dir)
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
