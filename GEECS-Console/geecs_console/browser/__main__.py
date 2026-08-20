"""GEECS Scan Browser entry point.

Run as ``geecs-scan-browser`` (console script) or
``python -m geecs_console.browser``.

Applies the console family stylesheet application-wide (the same
``app/style.qss`` the console loads — the loading approach is duplicated
here rather than imported because the browser must not import from
``app/main_window.py``), then the browser window layers its dark
screen-map palette on top.  The real :class:`TiledScanCatalog` is
(from ``geecs_data_utils``, built via ``TiledScanCatalog.from_config()``)
is injected; it degrades gracefully offline (the window itself defaults
to the stub for tests).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _load_console_stylesheet() -> str:
    """Read the console QSS, resolving the ui-directory asset token.

    The same behavior as the console's ``load_stylesheet`` (kept separate
    — the browser never imports ``app/main_window.py``): read
    ``geecs_console/app/style.qss`` and substitute ``@UI_DIR@`` with the
    packaged ``ui/`` directory for the combo-arrow asset.

    Returns
    -------
    str
        The resolved stylesheet text ("" when the asset is missing).
    """
    import geecs_console.app as app_pkg

    app_dir = Path(app_pkg.__file__).parent
    qss_path = app_dir / "style.qss"
    try:
        qss = qss_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("console stylesheet unavailable (%s): %s", qss_path, exc)
        return ""
    return qss.replace("@UI_DIR@", (app_dir / "ui").as_posix())


def main(argv: Optional[list[str]] = None) -> int:
    """Launch the scan browser.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (defaults to ``sys.argv[1:]``).
        Unrecognized arguments pass through to Qt.

    Returns
    -------
    int
        The Qt event-loop exit code.
    """
    parser = argparse.ArgumentParser(description="GEECS scan browser (Tiled)")
    parser.add_argument(
        "--experiment",
        default="",
        help="Experiment to open with (default: last remembered)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Root logging level (default: INFO)",
    )
    args, qt_args = parser.parse_known_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # The connection probe issues an httpx GET per reload; quiet its INFO line.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from PySide6.QtWidgets import QApplication

    from geecs_data_utils.tiled_catalog import TiledScanCatalog

    from geecs_console.browser.browser_window import ScanBrowserWindow

    app = QApplication([sys.argv[0], *qt_args])
    app.setStyleSheet(_load_console_stylesheet())
    window = ScanBrowserWindow(
        catalog=TiledScanCatalog.from_config(),
        experiment=args.experiment,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
