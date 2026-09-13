"""Per-scan ``scan.log``: a root-logger file handler for the duration of one run.

Every legacy scan folder carries a ``scan.log``; the Bluesky stack matches
that with a scoped ``logging.FileHandler`` attached to the **root logger**
from a run's start document to its stop document
(:class:`geecs_bluesky.callbacks.ScanLogCallback`), so ``scan.log`` records
the same story the worker's journal shows — ``bluesky`` RunEngine state
changes, ``ophyd_async`` connect failures, ``geecs_data_utils`` folder and
export lines — not just this package's namespaces.  Once the engine runs
inside a queueserver worker there is no operator-attached terminal, only a
machine-global journal, so the per-scan file must stay the complete record.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Third-party transport chatter kept out of scan.log below WARNING (live
#: finding, 2026-08-20 Scan001: Tiled's per-request httpx lines and MySQL
#: auth-plugin loads added ~15 lines of non-scan-story noise per scan).
#: Their WARNING+ records still land — only INFO chatter is dropped, and
#: only from the scan.log capture, never from the terminal.
QUIET_LOGGER_PREFIXES = ("httpx", "mysql.connector")


class _QuietNoisyLoggers(logging.Filter):
    """Drop sub-WARNING records from the known-noisy transport namespaces."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep *record* unless it is INFO-level transport chatter.

        Parameters
        ----------
        record : logging.LogRecord
            The record about to be captured.

        Returns
        -------
        bool
            ``False`` only for sub-WARNING records from
            :data:`QUIET_LOGGER_PREFIXES` namespaces.
        """
        if record.levelno >= logging.WARNING:
            return True
        return not any(
            record.name == prefix or record.name.startswith(prefix + ".")
            for prefix in QUIET_LOGGER_PREFIXES
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


class ScanLogFile:
    """One scan's ``scan.log``: :meth:`open` at the start document, :meth:`close` at the stop.

    Attaches to the root logger and lowers its level to INFO for the run
    (restored on close) so records from NOTSET-level loggers reach the
    handler; loggers with an explicit higher level keep it — terminal
    parity, not extra verbosity.  A missing folder means no file (a
    warning), never a created one: the scan folder is the claim's.
    """

    def __init__(self) -> None:
        self._handler: logging.Handler | None = None
        self._old_root_level: int = logging.NOTSET
        self.path: Path | None = None

    @property
    def is_open(self) -> bool:
        """Whether a handler is attached."""
        return self._handler is not None

    def open(self, scan_number: int, scan_folder: str | Path) -> Path | None:
        """Attach the file handler for ``Scan{scan_number:03d}`` inside *scan_folder*.

        Returns the file path, or ``None`` when the folder does not exist.
        A second ``open`` while one is attached closes the first.
        """
        if self.is_open:
            self.close()
        folder = Path(scan_folder)
        if not folder.is_dir():
            logger.warning("Scan folder %s does not exist; skipping scan.log", folder)
            return None
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
        handler.addFilter(_QuietNoisyLoggers())
        root = logging.getLogger()
        self._old_root_level = root.level
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
        root.addHandler(handler)
        self._handler = handler
        self.path = folder / "scan.log"
        logger.info("scan %s: starting (dir=%s)", scan_id, folder)
        return self.path

    def close(self, note: str = "finished") -> None:
        """Detach and close the handler, restoring the root logger's level."""
        handler = self._handler
        if handler is None:
            return
        logger.info("scan log: %s", note)
        root = logging.getLogger()
        root.removeHandler(handler)
        root.setLevel(self._old_root_level)
        handler.close()
        self._handler = None
        self.path = None


@contextmanager
def plan_report_sink(logger_name: str, *, stream: Any = None) -> Iterator[None]:
    """Make one logger's INFO records visible for the duration of a *non-scan* plan.

    A scan gets its narrative in ``scan.log``, because :class:`ScanLogFile`
    attaches a file handler and lifts the root level to INFO while the run is
    open.  A plan that opens **no run** gets neither, and the root logger sits
    above INFO by default — so every ``logger.info`` a queue plan emits is
    discarded, with nothing in the journal, nothing on the console stream, and
    no return value a queueserver client can retrieve.

    That is fine for a plan whose product is a side effect on the machine
    (``mv``).  It is not fine for one whose product is *a report for a human*:
    the shot-offset calibration measured ten shots on hardware and its table
    vanished entirely (GEECS-Plugins#861, found on the first real run).

    This attaches a stdout handler scoped to *logger_name* and lifts just that
    logger to INFO, restoring both afterwards — so the plan's own narrative
    reaches the worker's stdout (and from there the journal) without raising
    the log level of anything else.

    Parameters
    ----------
    logger_name :
        The logger to surface, e.g. ``"geecs_bluesky.plans.calibration"``.
    stream :
        Where to write; defaults to ``sys.stdout`` (what the service unit
        captures).  Injectable for tests.

    Yields
    ------
    None
    """
    target = logging.getLogger(logger_name)
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s - %(message)s"))
    previous_level = target.level
    if target.level == logging.NOTSET or target.level > logging.INFO:
        target.setLevel(logging.INFO)
    target.addHandler(handler)
    try:
        yield
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)
        handler.flush()
        handler.close()


__all__ = [
    "QUIET_LOGGER_PREFIXES",
    "ScanLogContextFilter",
    "ScanLogFile",
    "plan_report_sink",
]
