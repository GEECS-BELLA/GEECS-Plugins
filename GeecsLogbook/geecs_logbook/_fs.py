"""Filesystem primitives the attachment store and the mirror share.

Two writers (the request path and the periodic mirror sync) and two
destinations (local disk and the share) all need the same two things:
an atomic replace that never leaves a torn file, and a way to claim a
filename that two concurrent pastes cannot both win.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

#: Read once: ``os.umask`` can only be queried by setting it.
UMASK = os.umask(0)
os.umask(UMASK)


def replace_with(path: Path, data: bytes) -> None:
    """Write ``data`` to a uniquely named sibling, then rename over ``path``.

    A reader on another machine sees the old file or the new one, never
    half of each. The temp name is unique per call, so two writers on the
    same target cannot clobber each other's half-written file. ``mkstemp``
    creates 0600; the file gets the mode a plain write would have had,
    because a mirror is for people to read. On an SMB share the rename
    needs DELETE permission on the target (``scan_analysis.task_queue``
    learned this the hard way); the mirror is written by the service
    account, which has it.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        os.fchmod(fd, 0o666 & ~UMASK)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def claim_name(folder: Path, filename: str) -> Path:
    """Reserve ``filename`` in ``folder``, numbering it if it is taken.

    Claimed on disk with ``O_EXCL`` rather than by looking first: every
    clipboard paste is ``image.png``, and two pastes in flight at once must
    not both decide the name is free. The caller replaces the empty
    placeholder with the real bytes, or unlinks it on failure.
    """
    stem, ext = os.path.splitext(filename)
    n = 1
    while True:
        candidate = folder / (filename if n == 1 else f"{stem}-{n}{ext}")
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            n += 1
            continue
        os.close(fd)
        return candidate


def write_claimed(folder: Path, filename: str, data: bytes) -> Path:
    """Claim a free name in ``folder`` and land ``data`` there atomically.

    Returns the path actually used. On a failed write the claimed name is
    released, so a retry gets the original name rather than ``-2``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    target = claim_name(folder, filename)
    try:
        replace_with(target, data)
    except OSError:
        with contextlib.suppress(OSError):
            target.unlink()
        raise
    return target
