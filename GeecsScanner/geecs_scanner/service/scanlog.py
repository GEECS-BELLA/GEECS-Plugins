"""The scan.log tail: read what the worker appended to the run's ``scan.log``.

The worker writes ``<scan_folder>/scan.log`` for the span of a run
(``geecs_bluesky.scan_log.ScanLogFile``, attached at the start document);
the start document names the folder (``scan_folder``, claimed worker-side),
so this module needs no date arithmetic and no ``ScanPaths``.  It **reads
only**: a missing folder or file is reported, never created — the
cross-package rule that only the scanner side brings scan folders into
existence, and this process is a client, not the worker.

The scanner runs on the worker host, where the folder path resolves; from
another host it may not, and the answer says so (``available=False``).
"""

from __future__ import annotations

from pathlib import Path

from geecs_scanner.service.models import ScanLogOut

#: Bytes read per call — a bound on one SSE round, not a file limit.
CHUNK_BYTES = 64 * 1024


def read_scan_log(
    folder: str, offset: int = 0, *, limit: int = CHUNK_BYTES
) -> ScanLogOut:
    """Return the complete lines of ``<folder>/scan.log`` from *offset* on.

    Parameters
    ----------
    folder : str
        The run's folder as the start document spells it.
    offset : int
        Where the previous call stopped (``ScanLogOut.offset``); ``0``
        replays from the top.  An offset past the end (the file was
        replaced) restarts at ``0``.
    limit : int
        Most bytes read in one call; a longer file arrives over several.

    Returns
    -------
    ScanLogOut
        ``lines`` are whole lines only — a partial last line stays unread
        until its newline lands, and ``offset`` points at it.
    """
    path = Path(folder) / "scan.log"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return ScanLogOut(
            available=False,
            folder=folder,
            offset=offset,
            detail=f"no scan.log readable at {path} ({exc.__class__.__name__})",
        )
    if offset > size:
        offset = 0
    with path.open("rb") as fh:
        fh.seek(offset)
        raw = fh.read(limit)
    text = raw.decode("utf-8", errors="replace")
    end = text.rfind("\n")
    if end < 0:
        return ScanLogOut(
            available=True, folder=folder, offset=offset, more=len(raw) >= limit
        )
    complete = text[: end + 1]
    lines = complete.splitlines()
    consumed = len(complete.encode("utf-8", errors="replace"))
    return ScanLogOut(
        available=True,
        folder=folder,
        offset=offset + consumed,
        lines=lines,
        more=len(raw) >= limit,
    )
