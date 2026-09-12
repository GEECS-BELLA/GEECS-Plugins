"""The markdown mirror: the copy of the notes that outlives the software.

:mod:`~geecs_logbook.store` is what the page reads and writes. This module
writes the same entries a second time, as markdown files on the share, in
a tree the logbook **owns**::

    {experiment}/
      Y2026/09-Sep/26_0911/scans/      <- the scanner's; never written here
      logbook/                          <- this module's
        Y2026/
          09-Sep/
            26_0911/
              0834-sbarber-1c2d3e.md          a day-level entry (either book)
              Scan005/
                1154-sbarber-e7f2a1.md        one file per entry
                attachments/e7f2a1/
                  jet-trace.png
              after-Scan003/                  interscan entries
                1210-toperator-b91c04.md

Why its own tree, not ``logbook/`` inside each day folder
--------------------------------------------------------
The first cut put ``logbook/`` beside ``scans/`` inside the day, and
refused to create the day folder because the scanner makes days. That
stranded every entry written on a day with no scans — the operations
book's staple. A tree of the logbook's own has the same date shape (so a
person browsing by day still finds it), is always writable, backs up and
syncs as one folder, and — the reason that matters most — never enters
the data tree at all. The repository's scan-folder invariant (root
``CLAUDE.md``) is satisfied by construction: no path this module makes
has ``scans`` in it, and :func:`_assert_own_tree` pins that.

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
The store (and, for bytes, :mod:`~geecs_logbook.attachments`) is written
first, the mirror second. A slow or unreachable share must never lose the
sentence someone just typed or the screenshot they pasted; ``mirrored_at``
stays null until the files land and :func:`sync` retries what is owed.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Union

from geecs_data_utils import ScanPaths
from geecs_schemas.log_entry import LogEntry

from geecs_logbook._fs import replace_with
from geecs_logbook.attachments import AttachmentStore
from geecs_logbook.store import NotesStore

logger = logging.getLogger(__name__)

#: The logbook tree's name, a sibling of the ``Y{YYYY}`` year folders.
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
    """Return the day's directory in the logbook tree. Only a path; nothing is made.

    Derived from where the scanner would put that day's ``scans/`` so the
    two trees share one date layout and one configured share root, then
    re-rooted under ``{experiment}/logbook/``.

    Parameters
    ----------
    day : date or str
        The run day, as a date or ``YYYY-MM-DD``.
    experiment : str
        The experiment whose share this is.
    base_directory : Path or str, optional
        Override the configured data-share root. Used by tests.

    Raises
    ------
    MirrorUnavailable
        When the share root cannot be resolved at all (no configuration,
        unmounted drive). A host with no share is the same situation as a
        share that is down: the words are safe in the store, the file is
        owed.
    """
    when = date.fromisoformat(day) if isinstance(day, str) else day
    tag = ScanPaths.get_scan_tag(
        when.year, when.month, when.day, number=0, experiment=experiment
    )
    try:
        scans = ScanPaths.get_daily_scan_folder(tag=tag, base_directory=base_directory)
    except Exception as exc:  # noqa: BLE001 — no share config, unmounted drive …
        raise MirrorUnavailable(f"cannot resolve the data share: {exc}") from exc
    day_dir = scans.parent
    month_dir = day_dir.parent
    year_dir = month_dir.parent
    root = year_dir.parent / LOGBOOK_DIR / year_dir.name / month_dir.name / day_dir.name
    _assert_own_tree(root)
    return root


def _assert_own_tree(path: Path) -> None:
    """Refuse any path that enters the data tree. Pins the invariant."""
    if "scans" in path.parts or LOGBOOK_DIR not in path.parts:
        raise RuntimeError(f"mirror path is not in the logbook tree: {path}")


def entry_dir(entry: LogEntry, root: Path) -> Path:
    """Return the directory an entry's file and attachments live in.

    A day-level entry sits at the day's root; a scan's entries in
    ``ScanNNN/``; an interscan entry in ``after-ScanNNN/``.
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
    same file rather than leaving a trail. The time is the host's local
    time (the unit sets ``TZ`` from ``site.env``), the same clock the
    scan folders and the day itself are named by, so a listing sorts the
    way the day ran.
    """
    stamp = entry.created_at.astimezone().strftime("%H%M")
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

    The front matter carries what the body cannot — who, when, which book
    and scan, the tags, the version — as plain ``key: value`` lines a
    reader can parse back without a YAML library. The body follows
    verbatim: it is the author's text and this module does not touch it.
    """
    lines = ["---"]
    lines.append(f"entry_id: {entry.entry_id}")
    lines.append(f"day: {entry.day}")
    lines.append(f"book: {entry.book}")
    if entry.after is not None:
        lines.append(f"after: {entry.after}")
    elif entry.scan is not None:
        lines.append(f"scan: {entry.scan}")
    lines.append(f"author: {entry.author}")
    lines.append(f"kind: {entry.kind}")
    lines.append(f"status: {entry.status}")
    lines.append(f"template: {entry.template}")
    if entry.tags:
        lines.append("tags: " + ", ".join(entry.tags))
    lines.append(f"created_at: {entry.created_at.isoformat()}")
    if entry.edited_at:
        lines.append(f"edited_at: {entry.edited_at.isoformat()}")
    if entry.edited_by:
        lines.append(f"edited_by: {entry.edited_by}")
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


def write_entry(entry: LogEntry, root: Path) -> Path:
    """Write an entry's markdown to the share and return the path.

    ``mkdir(parents=True)`` is used freely: the deepest thing it can create
    is a day inside ``{experiment}/logbook/``, and :func:`_assert_own_tree`
    has already refused any path that touches the data tree.

    Raises
    ------
    MirrorUnavailable
        When the share refuses the write. The caller leaves ``mirrored_at``
        null and :func:`sync` retries.
    """
    _assert_own_tree(root)
    path = entry_path(entry, root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        replace_with(path, render(entry).encode("utf-8"))
    except OSError as exc:
        raise MirrorUnavailable(f"cannot write {path}: {exc}") from exc
    logger.info("mirrored %s -> %s", entry.entry_id, path)
    return path


def mirror_attachments(entry: LogEntry, root: Path, source: AttachmentStore) -> int:
    """Copy the entry's stored files beside its markdown; return how many landed.

    A file already on the share at the same size is left alone, so the
    periodic sync costs one ``stat`` per attachment rather than a re-copy.

    Raises
    ------
    MirrorUnavailable
        When the share refuses a write.
    """
    files = source.files(entry.entry_id)
    if not files:
        return 0
    _assert_own_tree(root)
    folder = entry_dir(entry, root) / ATTACHMENTS_DIR / entry.entry_id
    copied = 0
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = folder / src.name
            size = src.stat().st_size
            if dst.is_file() and dst.stat().st_size == size:
                continue
            replace_with(dst, src.read_bytes())
            copied += 1
    except OSError as exc:
        raise MirrorUnavailable(f"cannot write {folder}: {exc}") from exc
    return copied


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

#: One writer at a time to the share, per process: the request that just
#: saved and the periodic sync both mirror entries, and without this a
#: sync holding a stale read could write it over a newer file the request
#: had already mirrored and marked.
WRITE_LOCK = threading.Lock()


def mirror_one(
    store: NotesStore,
    entry_id: str,
    experiment: str,
    base_directory: Optional[Union[Path, str]] = None,
    attachments: Optional[AttachmentStore] = None,
) -> None:
    """Mirror the *current* state of one entry and mark that version done.

    Reads the entry afresh under the lock, so what is written is what is
    marked, and nothing older can land afterwards. With ``attachments``,
    the entry's stored files are copied beside the markdown too.

    Raises
    ------
    MirrorUnavailable
        When the share refuses; the caller records the deferral.
    """
    with WRITE_LOCK:
        entry = store.get(entry_id, include_deleted=True)
        if entry is None:
            return
        root = logbook_root(entry.day, experiment, base_directory)
        if entry.is_deleted:
            remove_entry(entry, root)
        else:
            write_entry(entry, root)
            if attachments is not None:
                mirror_attachments(entry, root, attachments)
        store.mark_mirrored(
            entry.entry_id, datetime.now(timezone.utc), version=entry.version
        )


def sync(
    store: NotesStore,
    experiment: str,
    base_directory: Optional[Union[Path, str]] = None,
    attachments: Optional[AttachmentStore] = None,
    limit: int = 100,
) -> tuple[int, int]:
    """Mirror everything the store owes, and return ``(written, deferred)``.

    The reconciliation half of writing the store first. Each entry is tried
    on its own; one refused write defers that entry and moves on, so a
    single bad path never blocks the rest. A tombstoned entry's owed
    operation is the removal of its file, and it counts as written once
    the file is gone.
    """
    written = deferred = 0
    for owed in store.unmirrored(limit=limit):
        try:
            mirror_one(store, owed.entry_id, experiment, base_directory, attachments)
        except MirrorUnavailable as exc:
            logger.info("deferring %s: %s", owed.entry_id, exc)
            store.mark_deferred(owed.entry_id)
            deferred += 1
            continue
        written += 1
    return written, deferred
