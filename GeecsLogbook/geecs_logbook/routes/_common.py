"""What every route module shares: the mounted context and small helpers.

The router is assembled from modules (day, entries, attachments — month
next) that each take a :class:`Context` and register their routes on an
``APIRouter``. The context carries the things ``create_log_router`` was
given, the stores it built from them, and the two operations more than
one module needs: reading a day off the share and mirroring an entry.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union

from fastapi import HTTPException, Request
from fastapi.templating import Jinja2Templates
from geecs_schemas.log_entry import LogEntry

from geecs_logbook import mirror
from geecs_logbook.attachments import AttachmentStore
from geecs_logbook.models import DaySummary
from geecs_logbook.scan_reader import read_day
from geecs_logbook.store import NotesStore

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

#: How many dates the rail offers as quick links, centred on the shown day.
RAIL_SPAN = 15

#: Mirror reconciliation runs opportunistically on page views, at most this
#: often. Cheap when nothing is owed (an indexed query); bounded when the
#: share is down so a broken mount does not add a timeout to every view.
SYNC_INTERVAL_S = 60.0


@dataclass(frozen=True)
class RenderedEntry:
    """An entry plus its HTML, for a template."""

    entry: LogEntry
    html: str


@dataclass
class Context:
    """Everything the route modules need, built once by ``create_log_router``.

    Attributes
    ----------
    experiment : str
        The experiment whose share is read and mirrored to.
    base_directory : Path or str, optional
        Share-root override, for tests.
    store : NotesStore, optional
        The commentary store. ``None`` means a read-only logbook: no entry
        routes, no attachment routes.
    attachments : AttachmentStore, optional
        Uploaded bytes, beside the database. Present exactly when ``store`` is.
    templates : Jinja2Templates
        The page templates, with the logbook's filters installed.
    """

    experiment: str
    base_directory: Optional[Union[Path, str]]
    store: Optional[NotesStore]
    attachments: Optional[AttachmentStore]
    templates: Jinja2Templates
    _last_sync: dict = field(default_factory=lambda: {"at": 0.0})

    @property
    def writable(self) -> bool:
        """Whether this logbook takes entries."""
        return self.store is not None

    def load_day(self, day: date) -> DaySummary:
        """Read one day off the share, turning share trouble into an honest 503."""
        try:
            return read_day(day, self.experiment, base_directory=self.base_directory)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — surface, don't 500 blankly
            logger.exception("reading %s failed", day)
            raise HTTPException(
                status_code=503, detail=f"data share unavailable: {exc}"
            ) from exc

    def maybe_sync(self) -> None:
        """Pay the store's mirror debt, at most every ``SYNC_INTERVAL_S``."""
        if self.store is None:
            return
        now = time.monotonic()
        if now - self._last_sync["at"] < SYNC_INTERVAL_S:
            return
        self._last_sync["at"] = now
        try:
            written, deferred = mirror.sync(
                self.store, self.experiment, self.base_directory, self.attachments
            )
            if written or deferred:
                logger.info("mirror sync: %d written, %d deferred", written, deferred)
        except Exception:  # noqa: BLE001 — a sync must never take down a view
            logger.exception("mirror sync failed")

    def mirror(self, entry_id: str) -> None:
        """Try to land one entry on the share; defer quietly if it cannot."""
        assert self.store is not None
        try:
            mirror.mirror_one(
                self.store,
                entry_id,
                self.experiment,
                self.base_directory,
                self.attachments,
            )
        except mirror.MirrorUnavailable as exc:
            logger.info("mirror deferred for %s: %s", entry_id, exc)
            self.store.mark_deferred(entry_id)


# --------------------------------------------------------------- helpers


def parse_day(raw: str) -> date:
    """Parse a ``YYYY-MM-DD`` path segment, or 400."""
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"day must be YYYY-MM-DD, got {raw!r}"
        ) from exc


def rail_days(centre: date) -> list[date]:
    """Return the dates offered as quick links in the rail, newest first.

    Centred on the shown date rather than trailing it, so stepping forward
    is as easy as stepping back; the date picker covers anything outside
    this window.
    """
    half = RAIL_SPAN // 2
    return [centre + timedelta(days=half - offset) for offset in range(RAIL_SPAN)]


def initials(name: str) -> str:
    """Two letters for an avatar: first and last word, or the first two."""
    parts = [p for p in str(name).replace(".", " ").split() if p]
    if not parts:
        return "??"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def attachment_base(request: Request) -> str:
    """The serving prefix a body's relative ``attachments/…`` links map onto.

    Built from the route's own URL so it is right under any mount prefix
    or reverse-proxy ``root_path``. The relative link already carries the
    entry id and filename, so the base is the route's root.
    """
    full = request.url_for("_attachment", entry_id="X", filename="Y").path
    return full[: -len("/X/Y")]


def api_base(request: Request) -> str:
    """``/log/api`` under whatever prefix the host mounted us at."""
    probe = request.url_for("_entries_json", day="0000-00-00").path
    return probe[: -len("/day/0000-00-00/entries")]
