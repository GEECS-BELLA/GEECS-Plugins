"""Per-scan ``scan.log`` file handling, shared by the bridge and the session.

Every legacy scan folder carries a ``scan.log``; the Bluesky stack matches
that with a scoped ``logging.FileHandler`` attached for the duration of one
scan.  Extracted verbatim from ``BlueskyScanner._scan_log`` (Gate-2 finding:
headless ``GeecsSession.run()`` scans had no scan.log because the helper was
bridge-internal) so both front doors share one implementation — the session
must not import the bridge.

The handler attaches to the **root logger**, so ``scan.log`` records the
same story the process's terminal shows — ``bluesky`` RunEngine state
changes, ``ophyd_async`` connect failures, ``geecs_data_utils`` folder and
export lines — not just this package's namespaces.  The per-scan file must
stay the complete record: once the engine runs inside a queueserver worker
there is no operator-attached terminal, only a machine-global journal.

Two mechanisms cover the window before the scan folder exists (the file
cannot be created earlier, and the most diagnostic lines — submission,
device connects, telemetry drops — happen there):

- :func:`begin_pre_scan_capture` starts buffering root-logger records at
  submission time (the bridge's ``reinitialize`` calls it; headless callers
  may call it themselves).
- :func:`scan_log` flushes that buffer into the file the moment it attaches,
  and discards it on the no-claim paths (nothing was saved, so the buffered
  lines have no per-scan home — they remain on the terminal/journal).
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

#: Maximum buffered pre-claim records (older records drop first).  Sized far
#: above a real submission window (tens of lines); the cap only bounds a
#: pathological reinitialize-then-never-start session.
PRE_SCAN_BUFFER_CAPACITY = 2000

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
        return not record.name.startswith(QUIET_LOGGER_PREFIXES)


class PreScanLogBuffer(logging.Handler):
    """Buffer root-logger records emitted before the scan folder exists.

    A plain bounded buffer: :meth:`emit` appends, and :meth:`flush_into`
    replays every held record through a target handler (whose own filters —
    the scan-id stamp — apply on replay).  Never auto-flushes.
    """

    def __init__(self, capacity: int = PRE_SCAN_BUFFER_CAPACITY) -> None:
        super().__init__(level=logging.INFO)
        self.addFilter(_QuietNoisyLoggers())
        self._records: deque[logging.LogRecord] = deque(maxlen=capacity)
        #: Root-logger level before the capture lowered it (restored by
        #: discard/take — record *creation* is gated by the root level, so
        #: the capture window must run the root at INFO to see the story).
        self.original_root_level: int = logging.NOTSET

    def emit(self, record: logging.LogRecord) -> None:
        """Hold *record* for a later :meth:`flush_into`.

        Parameters
        ----------
        record : logging.LogRecord
            The record to buffer.
        """
        self._records.append(record)

    def flush_into(self, handler: logging.Handler) -> None:
        """Replay every buffered record through *handler*, then clear.

        Parameters
        ----------
        handler : logging.Handler
            The destination handler (the scan.log file handler).
        """
        for record in self._records:
            handler.handle(record)
        self._records.clear()


#: The at-most-one pending pre-scan buffer (one scan at a time per process).
_pending_buffer: PreScanLogBuffer | None = None


def begin_pre_scan_capture() -> None:
    """Start buffering root-logger records for the next scan's ``scan.log``.

    Called at submission time (bridge ``reinitialize`` / headless callers).
    Any previous pending buffer is discarded first — a superseded submission
    must not leak its lines into the next scan's file.
    """
    global _pending_buffer
    discard_pre_scan_capture()
    root = logging.getLogger()
    buffer = PreScanLogBuffer()
    buffer.original_root_level = root.level
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    _pending_buffer = buffer
    root.addHandler(buffer)


def discard_pre_scan_capture() -> None:
    """Drop any pending pre-scan buffer without writing it anywhere."""
    _detach_pending_buffer()


def _take_pending_buffer() -> PreScanLogBuffer | None:
    """Detach and return the pending buffer (None when nothing is pending)."""
    return _detach_pending_buffer()


def _detach_pending_buffer() -> PreScanLogBuffer | None:
    """Remove the pending buffer from the root logger, restoring its level."""
    global _pending_buffer
    buffer = _pending_buffer
    if buffer is not None:
        root = logging.getLogger()
        root.removeHandler(buffer)
        root.setLevel(buffer.original_root_level)
        _pending_buffer = None
    return buffer


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


def log_claimed_scan_failure(
    scan_number: int | None,
    scan_folder: str | None,
    *,
    label: str = "Scan",
    aborted: bool = False,
) -> None:
    """Log that a claimed scan folder was left behind by a failure or abort.

    The folder is never deleted (scan-folder lifecycle invariant: once a
    ``scans/ScanNNN/`` folder exists it must not be removed or recreated),
    so the claimed-but-not-completed state is surfaced instead of being
    silent.  A genuine failure is an ERROR; an operator-requested abort
    (``aborted=True``) is an intentional outcome and gets one calm WARNING
    instead.  A no-op when nothing was claimed.

    Parameters
    ----------
    scan_number : int or None
        The claimed day-scoped scan number.
    scan_folder : str or None
        The claimed ``scans/ScanNNN`` folder path.
    label : str
        Message prefix naming the scan kind (e.g. ``"Optimization scan"``).
    aborted : bool
        ``True`` when the scan ended because the operator requested an
        abort (WARNING, calm wording) rather than failing (ERROR).
    """
    if scan_number is None and scan_folder is None:
        return
    if aborted:
        logger.warning(
            "%s %s aborted by operator; folder %s kept (never deleted) — "
            "partial data may be present",
            label,
            scan_number,
            scan_folder,
        )
        return
    logger.error(
        "%s %s failed or aborted after its folder was claimed at %s; "
        "the folder is left in place (never deleted) and may be missing "
        "ScanInfo or data",
        label,
        scan_number,
        scan_folder,
    )


@contextmanager
def scan_log(scan_number: int | None, scan_folder: str | None):
    """Attach a per-scan ``scan.log`` file handler for the enclosed block.

    A no-op when the scan number/folder are unknown (nothing was claimed —
    e.g. ``save_data=False`` or the NetApp is unreachable) or the folder
    does not exist; any pending pre-scan buffer is discarded on those paths.
    On attach, buffered pre-claim records flush into the file first.  On
    exit the handler is removed and closed and the root logger's level is
    restored, even on abort.

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
        discard_pre_scan_capture()
        yield
        return

    folder = Path(scan_folder)
    if not folder.is_dir():
        discard_pre_scan_capture()
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
    handler.addFilter(_QuietNoisyLoggers())

    # Take the pending buffer FIRST — it restores the pre-capture root level,
    # so the save/restore below brackets the true original.
    buffer = _take_pending_buffer()

    # Root-logger attach: scan.log records what the terminal shows.  The
    # root level is lowered to INFO for the scan (and restored) so records
    # from NOTSET-level loggers reach the handler; loggers with an explicit
    # higher level keep it — terminal parity, not extra verbosity.
    root = logging.getLogger()
    old_root_level = root.level
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)

    # Pre-claim records (submission, connects, telemetry drops) replay into
    # the file first — they are chronologically earlier than everything the
    # live handler will see, and the handler's filter stamps them.
    if buffer is not None:
        buffer.flush_into(handler)

    root.addHandler(handler)
    try:
        logger.info("scan %s: starting (dir=%s)", scan_id, scan_folder)
        yield
        logger.info("scan %s: finished", scan_id)
    finally:
        root.removeHandler(handler)
        root.setLevel(old_root_level)
        handler.close()
