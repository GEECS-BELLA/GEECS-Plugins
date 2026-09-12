"""The markdown mirror: the copy of the notes that outlives the software.

:mod:`~geecs_logbook.store` is what the page reads and writes. This module
writes the same entries a second time, as markdown files beside the data on
the share::

    {day}/
      scans/            <- the scanner's; never written here
      analysis/
      logbook/          <- this module's, a SIBLING of scans/
        day.md                          the day intro
        Scan005/
          1154-sbarber-e7f2a1.md        one file per entry
          attachments/e7f2a1/
            jet-trace.png
        after-Scan003/                  interscan entries
          1210-agonsalves-b91c04.md

Why a sibling of ``scans/``
---------------------------
Human commentary never mixes into the raw-data tree, the day intro has a
home, and — the reason that matters most — nothing here ever traverses
``scans/ScanNNN/``. The repository's scan-folder invariant (root
``CLAUDE.md``) is satisfied by construction rather than by care.

One creation the invariant does forbid is guarded explicitly: the **day
folder** itself. The mirror lands only once the scanner has made the day;
an intro written at 08:00 before the first scan stays in the store and is
mirrored when the folder appears. See :func:`write_entry`.

Links are relative
------------------
A body that says ``![trace](attachments/e7f2a1/jet-trace.png)`` resolves
from the file on disk in any markdown viewer, offline, in 2031. The web
view rewrites that path to its serving route at render time. Writing a
serving URL into the file instead would leave the durable copy full of dead
links to a service that no longer exists — which would quietly gut the
whole reason for the mirror.

Write order
-----------
The store is written first, the mirror second. A slow or unreachable share
must never lose the sentence someone just typed; ``mirrored_at`` stays null
until the file lands and :func:`sync` retries what is owed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Union

from geecs_data_utils import ScanPaths
from geecs_schemas.log_entry import LogEntry

from geecs_logbook.store import NotesStore

logger = logging.getLogger(__name__)

#: The logbook directory's name, beside ``scans/`` and ``analysis/``.
LOGBOOK_DIR = "logbook"

#: The attachments directory's name inside an entry's directory.
ATTACHMENTS_DIR = "attachments"


class MirrorUnavailable(OSError):
    """The share cannot take the write right now.

    Raised, never swallowed at this layer: the store has already saved the
    words, so the caller's only job is to leave ``mirrored_at`` null and let
    :func:`sync` try again. Subclasses ``OSError`` so a caller that already
    handles share trouble handles this too.
    """


# ------------------------------------------------------------------ paths


def logbook_root(
    day: Union[date, str],
    experiment: str,
    base_directory: Optional[Union[Path, str]] = None,
) -> Path:
    """Return the day's ``logbook/`` directory. Only a path; nothing is made.

    Parameters
    ----------
    day : date or str
        The run day, as a date or ``YYYY-MM-DD``.
    experiment : str
        The experiment whose share this is.
    base_directory : Path or str, optional
        Override the configured data-share root. Used by tests.
    """
    when = date.fromisoformat(day) if isinstance(day, str) else day
    tag = ScanPaths.get_scan_tag(
        when.year, when.month, when.day, number=0, experiment=experiment
    )
    try:
        scans = ScanPaths.get_daily_scan_folder(tag=tag, base_directory=base_directory)
    except Exception as exc:  # noqa: BLE001 — no share config, unmounted drive …
        # A host with no data-share configuration is the same situation as
        # a share that is down: the words are safe in the store, the file
        # is owed. Surfacing it as a 500 after the row was written would
        # tell the writer their entry failed when it did not.
        raise MirrorUnavailable(f"cannot resolve the data share: {exc}") from exc
    return scans.parent / LOGBOOK_DIR


def entry_dir(entry: LogEntry, root: Path) -> Path:
    """Return the directory an entry's file and attachments live in.

    A day-level entry sits at the root; a scan's entries in ``ScanNNN/``;
    an interscan entry in ``after-ScanNNN/``. Nothing here is created.
    """
    if entry.after is not None:
        return root / f"after-Scan{entry.after:03d}"
    if entry.scan is None:
        return root
    return root / f"Scan{entry.scan:03d}"


def entry_path(entry: LogEntry, root: Path) -> Path:
    """Return the markdown file for an entry. Only a path; nothing is made.

    The name is stable across edits — it carries the *creation* time, the
    author and the id, none of which change — so an edit overwrites the
    same file rather than leaving a trail.
    """
    stamp = entry.created_at.astimezone(timezone.utc).strftime("%H%M")
    author = "".join(c for c in entry.author.lower() if c.isalnum()) or "anon"
    return entry_dir(entry, root) / f"{stamp}-{author}-{entry.entry_id[:6]}.md"


def attachment_link(entry: LogEntry, filename: str) -> str:
    """Return the relative link a body uses to reference an attachment.

    Relative to the entry's own directory, so it resolves from the file on
    disk with no server involved. The web view rewrites it when rendering.
    """
    return f"{ATTACHMENTS_DIR}/{entry.entry_id}/{filename}"


# ---------------------------------------------------------------- render


def render(entry: LogEntry) -> str:
    """Render an entry as markdown with a front-matter envelope.

    The front matter carries what the body cannot — who, when, which scan,
    the template it started from, the version — as plain ``key: value``
    lines a reader can parse back without a YAML library. The body follows
    verbatim: it is the author's text and this module does not touch it.
    """
    lines = ["---"]
    lines.append(f"entry_id: {entry.entry_id}")
    lines.append(f"day: {entry.day}")
    if entry.after is not None:
        lines.append(f"after: {entry.after}")
    elif entry.scan is not None:
        lines.append(f"scan: {entry.scan}")
    lines.append(f"author: {entry.author}")
    lines.append(f"kind: {entry.kind}")
    lines.append(f"status: {entry.status}")
    lines.append(f"template: {entry.template}")
    lines.append(f"created_at: {entry.created_at.isoformat()}")
    if entry.edited_at:
        lines.append(f"edited_at: {entry.edited_at.isoformat()}")
    lines.append(f"version: {entry.version}")
    lines.append(f"schema_version: {entry.schema_version}")
    if entry.payload is not None:
        lines.append(
            "payload: "
            + json.dumps(entry.payload.model_dump(mode="json"), sort_keys=True)
        )
    if entry.attachments:
        lines.append("attachments:")
        for a in entry.attachments:
            lines.append(f"  - {attachment_link(entry, a.filename)}")
    lines.append("---")
    lines.append("")
    body = entry.body_md.rstrip("\n")
    return "\n".join(lines) + "\n" + body + "\n"


# ----------------------------------------------------------------- write


def _day_folder_exists(root: Path) -> bool:
    """Whether the scanner has made the day this logbook belongs to."""
    return root.parent.is_dir()


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and rename, so a torn write never lands.

    A reader on another machine either sees the old file or the new one,
    never half of each — the property a mirror on a shared drive needs.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_entry(entry: LogEntry, root: Path) -> Path:
    """Write an entry's markdown to the share and return the path.

    Raises
    ------
    MirrorUnavailable
        When the day folder does not exist yet — the scanner makes days,
        never this module — or when the share refuses the write. The
        caller leaves ``mirrored_at`` null and :func:`sync` retries.

    Notes
    -----
    ``mkdir(parents=True)`` is used here and is safe *because of* the day
    folder guard above it: with the day present, the deepest thing it can
    create is ``logbook/ScanNNN/``, and it never touches ``scans/``.
    """
    if not _day_folder_exists(root):
        raise MirrorUnavailable(
            f"day folder not present yet, not creating it: {root.parent}"
        )
    path = entry_path(entry, root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, render(entry))
    except OSError as exc:
        raise MirrorUnavailable(f"cannot write {path}: {exc}") from exc
    logger.info("mirrored %s -> %s", entry.entry_id, path)
    return path


def write_attachment(entry: LogEntry, filename: str, data: bytes, root: Path) -> str:
    """Store an uploaded file beside the entry and return its relative link.

    Same guard and same reasoning as :func:`write_entry`. The bytes live
    only here; the store keeps the manifest.
    """
    if not _day_folder_exists(root):
        raise MirrorUnavailable(
            f"day folder not present yet, not creating it: {root.parent}"
        )
    target = entry_dir(entry, root) / ATTACHMENTS_DIR / entry.entry_id / filename
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
    except OSError as exc:
        raise MirrorUnavailable(f"cannot write {target}: {exc}") from exc
    return attachment_link(entry, filename)


def remove_entry(entry: LogEntry, root: Path) -> bool:
    """Remove an entry's markdown. Attachments are left for a human.

    Returns whether a file was removed. A missing file is not an error:
    the entry may never have been mirrored.
    """
    path = entry_path(entry, root)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise MirrorUnavailable(f"cannot remove {path}: {exc}") from exc
    return True


# ------------------------------------------------------------------ sync


def sync(
    store: NotesStore,
    experiment: str,
    base_directory: Optional[Union[Path, str]] = None,
    limit: int = 100,
) -> tuple[int, int]:
    """Mirror everything the store owes, and return ``(written, deferred)``.

    The reconciliation half of writing the store first. Each entry is tried
    on its own; one day's absent folder or one refused write defers that
    entry and moves on, so a single bad share path never blocks the rest.
    A tombstoned entry's owed operation is the removal of its file, and it
    counts as written once the file is gone.
    """
    written = deferred = 0
    for entry in store.unmirrored(limit=limit):
        root = logbook_root(entry.day, experiment, base_directory)
        try:
            if entry.is_deleted:
                remove_entry(entry, root)
            else:
                write_entry(entry, root)
        except MirrorUnavailable as exc:
            logger.info("deferring %s: %s", entry.entry_id, exc)
            deferred += 1
            continue
        store.mark_mirrored(entry.entry_id, datetime.now(timezone.utc))
        written += 1
    return written, deferred
