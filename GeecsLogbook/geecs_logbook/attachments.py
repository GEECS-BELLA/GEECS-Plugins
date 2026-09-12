"""The attachment store: uploaded bytes, on the host, beside the database.

Text goes to the store first and the share second so that a flaky share
never loses a sentence. Bytes follow the same rule for the same reason: a
screenshot pasted while the mount is wedged lands here, is served from
here, and reaches the share when the mirror gets to it. Everything
irreplaceable — the database and this directory — is then in one place,
which is the whole backup story.

Layout::

    <root>/
      <entry_id>/
        image.png
        image-2.png       a second paste with the same name
        jet-trace.pdf

An entry's files are in one directory named by its id, and a body refers
to them by the relative link ``attachments/<entry_id>/<filename>`` — the
same link the mirror writes beside the markdown, so the durable copy
resolves offline and the web view rewrites it at render time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from geecs_logbook._fs import write_claimed

logger = logging.getLogger(__name__)


class AttachmentStore:
    """Bytes for entries, under one directory the service owns.

    Parameters
    ----------
    root : Path
        The directory. Created if absent — it lives beside the database in
        the service's state directory, never on the data share.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, entry_id: str, filename: str, data: bytes) -> str:
        """Store ``data`` for an entry and return the filename actually used.

        ``image.png`` twice becomes ``image.png`` and ``image-2.png``; the
        caller records the returned name, not the one it asked for.
        """
        target = write_claimed(self.root / entry_id, filename, data)
        logger.info(
            "attachment stored: %s/%s (%d bytes)", entry_id, target.name, len(data)
        )
        return target.name

    def path(self, entry_id: str, filename: str) -> Optional[Path]:
        """Return the file for a served link, or ``None`` if it is not one.

        Contained twice over: the entry directory must sit directly inside
        the root, and the file directly inside the entry directory. The
        first check is the one that matters — without it ``..`` as an
        entry id resolves to the state directory and the database beside
        this store is served as an attachment.
        """
        root = self.root.resolve()
        folder = (root / entry_id).resolve()
        if folder.parent != root:
            return None
        target = (folder / filename).resolve()
        if target.parent != folder or not target.is_file():
            return None
        return target

    def files(self, entry_id: str) -> list[Path]:
        """Every stored file of an entry, by name."""
        folder = self.root / entry_id
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.iterdir() if p.is_file())
