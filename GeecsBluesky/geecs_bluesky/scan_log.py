"""Per-scan ``scan.log`` file handling, shared by the bridge and the session.

Every legacy scan folder carries a ``scan.log``; the Bluesky stack matches
that with a scoped ``logging.FileHandler`` attached for the duration of one
scan.  Extracted verbatim from ``BlueskyScanner._scan_log`` (Gate-2 finding:
headless ``GeecsSession.run()`` scans had no scan.log because the helper was
bridge-internal) so both front doors share one implementation — the session
must not import the bridge.

The handler captures the whole scan story, not just this package: during
optimization scans the evaluator (``geecs_scanner.optimization``) and its
analyzers (``scan_analysis``, ``image_analysis``) do the per-bin work, and
their file-mapping / objective lines belong in ``scan.log`` too.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

#: Logger namespaces captured into every scan.log (see module docstring).
CAPTURE_LOGGER_NAMES = (
    "geecs_bluesky",
    "geecs_scanner.optimization",
    "scan_analysis",
    "image_analysis",
)


class ScanLogContextFilter(logging.Filter):
    """Add scan id context to records written to one scan log."""

    def __init__(self, scan_id: str) -> None:
        super().__init__()
        self._scan_id = scan_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Stamp *record* with the scan id; never drops records.

        Parameters
        ----------
        record : logging.LogRecord
            The record about to be written to the scan log.

        Returns
        -------
        bool
            Always ``True``.
        """
        record.scan_id = self._scan_id
        return True


@contextmanager
def scan_log(scan_number: int | None, scan_folder: str | None):
    """Attach a per-scan ``scan.log`` file handler for the enclosed block.

    A no-op when the scan number/folder are unknown (nothing was claimed —
    e.g. ``save_data=False`` or the NetApp is unreachable) or the folder
    does not exist.  On exit the handler is removed and closed and every
    captured logger's level is restored, even on abort.

    Parameters
    ----------
    scan_number : int or None
        The claimed day-scoped scan number.
    scan_folder : str or None
        The claimed ``scans/ScanNNN`` folder path.

    Yields
    ------
    None
        Run the scan inside the ``with`` block.
    """
    if scan_number is None or scan_folder is None:
        yield
        return

    folder = Path(scan_folder)
    if not folder.is_dir():
        logger.warning("Scan folder %s does not exist; skipping scan.log", folder)
        yield
        return

    scan_id = f"Scan{scan_number:03d}"
    handler = logging.FileHandler(folder / "scan.log", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s "
            "[%(threadName)s] scan=%(scan_id)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(ScanLogContextFilter(scan_id))

    capture_loggers = [logging.getLogger(name) for name in CAPTURE_LOGGER_NAMES]
    old_levels = [lg.level for lg in capture_loggers]
    for lg in capture_loggers:
        if lg.level == logging.NOTSET or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
        lg.addHandler(handler)
    try:
        logger.info("scan %s: starting (dir=%s)", scan_id, scan_folder)
        yield
        logger.info("scan %s: finished", scan_id)
    finally:
        for lg, old_level in zip(capture_loggers, old_levels):
            lg.removeHandler(handler)
            lg.setLevel(old_level)
        handler.close()
