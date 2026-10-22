"""One atomic file write for the package: temp file beside the target, ``os.replace``.

Used by the configs resolver (``shot_offsets.yaml``, presets) and by the
ScanInfo callback (the scan's ini, which the logbook and the portal parse
while a scan runs). A reader sees the old file or the new one, never a
truncated one.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def replace_text(path: Path, text: str, *, fsync: bool = True) -> None:
    """Write *text* to *path* through a temp file beside it and ``os.replace``.

    A failure part-way leaves the previous file intact and no temporary
    behind, and the destination's mode is kept (0644 for a new file) so the
    file stays readable to whoever reviews it, to git run as anyone else,
    and to another host's reader.

    Parameters
    ----------
    path : Path
        The destination; its directory must exist (never created here).
    text : str
        The whole new content.
    fsync : bool
        Flush to disk before the rename — durability for a document that
        cost shots (a calibration); a file rewritten moments later (the
        ScanInfo ini) can skip it.

    Raises
    ------
    OSError
        From the write or the replace (a full or disconnected share; on SMB
        a rename over a file another client holds open without delete
        sharing).
    """
    # The destination's mode, or the default a normal umask would give:
    # NamedTemporaryFile creates 0600 and os.replace keeps the temp
    # inode's mode, so without this one write makes the file unreadable
    # to everyone but the writer.
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = 0o644
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.stem}-",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            if fsync:
                # Durability before the rename: without it a crash can leave
                # a zero-length file where a calibration used to be.
                os.fsync(handle.fileno())
        try:
            os.chmod(temporary, mode)
        except OSError:
            # The configs repo and the scan folders usually live on the data
            # share, and CIFS mounts reject chmod unless mounted with unix
            # extensions. A mode we could not set is cosmetic; losing the
            # document is not, so never fail the write for it.
            logger.warning(
                "could not set the mode of %s (the share may not support "
                "it) — the file is written, but check it is readable by "
                "whoever has to read it",
                path,
                exc_info=True,
            )
        os.replace(temporary, path)
        temporary = None
    finally:
        # Covers the write and the chmod as well as the replace: a full
        # or disconnected share fails in `handle.write`, and a leftover
        # dot-file is somebody's next puzzle.
        if temporary is not None:
            temporary.unlink(missing_ok=True)
