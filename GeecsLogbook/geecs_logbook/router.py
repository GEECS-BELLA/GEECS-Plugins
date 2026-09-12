"""The logbook's HTTP surface, mounted by GEECS-DataPortal at ``/log``.

Following the config-editor precedent (`scan_analysis.config_editor`), this
module exposes a factory returning an :class:`~fastapi.APIRouter` rather
than an app, so the portal owns the process, the port and the unit.

Routes
------
``GET /log/``
    Redirect to today.
``GET /log/day/{day}``
    The day document: every scan folder present for that date, with the
    entries people wrote about them.
``GET /log/api/day/{day}``
    The derived half as JSON.
``GET /log/api/day/{day}/entries``
    The commentary half as JSON.

With a notes store configured, the write verbs — the only ones this
package has, and the reason it is a charter exception in the portal:

``POST /log/api/entries``, ``PATCH /log/api/entries/{id}``,
``POST /log/api/entries/{id}/status``, ``DELETE /log/api/entries/{id}``,
``POST /log/api/entries/{id}/attachments``

Every write goes to the store first and the share second — see
:mod:`geecs_logbook.mirror` for why that order. Writes touch ``logbook/``
only, a sibling of ``scans/``; nothing here can create a scan folder.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional, Union

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from geecs_schemas.log_entry import Attachment, LogEntry
from pydantic import BaseModel, Field

from geecs_logbook import mirror
from geecs_logbook.models import DaySummary
from geecs_logbook.render import render_markdown
from geecs_logbook.scan_reader import read_day
from geecs_logbook.store import ConflictError, NotesStore

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_TEMPLATES = _HERE / "templates"
_STATIC = _HERE / "static"

#: How many dates the rail offers as quick links, centred on the shown day.
_RAIL_SPAN = 15

#: Mirror reconciliation runs opportunistically on page views, at most this
#: often. Cheap when nothing is owed (an indexed query); bounded when the
#: share is down so a broken mount does not add a timeout to every view.
_SYNC_INTERVAL_S = 60.0

#: Attachment limits. The cap is rejected with a clear 413, never a 500.
_MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
_ATTACHMENT_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}


# ------------------------------------------------------------- payloads


class EntryCreate(BaseModel):
    """What a client sends to add an entry."""

    day: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    scan: Optional[int] = Field(
        None, ge=0, description="Scan number; 0 = the day intro."
    )
    after: Optional[int] = Field(
        None, ge=0, description="Interscan: the scan it follows."
    )
    author: str = Field(min_length=1, max_length=120)
    body_md: str = Field(max_length=200_000)
    template: str = Field("blank", max_length=64)
    kind: Literal["note", "agent_analysis", "agent_draft"] = "note"
    status: Literal["kept", "draft"] = "kept"
    payload: Optional[dict] = None


class EntryUpdate(BaseModel):
    """What a client sends to edit an entry's text."""

    body_md: str = Field(max_length=200_000)
    author: str = Field(min_length=1, max_length=120)
    expected_version: int = Field(ge=1)


class StatusUpdate(BaseModel):
    """Keep or un-keep an entry — a human act on unchanged text."""

    status: Literal["kept", "draft"]


@dataclass(frozen=True)
class RenderedEntry:
    """An entry plus its HTML, for the template."""

    entry: LogEntry
    html: str


# --------------------------------------------------------------- helpers


def _parse_day(raw: str) -> date:
    """Parse a ``YYYY-MM-DD`` path segment, or 400."""
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"day must be YYYY-MM-DD, got {raw!r}"
        ) from exc


def _rail_days(centre: date) -> list[date]:
    """Return the dates offered as quick links in the rail, newest first.

    Centred on the shown date rather than trailing it, so stepping forward
    is as easy as stepping back; the date picker covers anything outside
    this window.
    """
    half = _RAIL_SPAN // 2
    return [centre + timedelta(days=half - offset) for offset in range(_RAIL_SPAN)]


def _initials(name: str) -> str:
    """Two letters for an avatar: first and last word, or the first two."""
    parts = [p for p in str(name).replace(".", " ").split() if p]
    if not parts:
        return "??"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _scope(entry: LogEntry) -> str:
    """The URL segment naming an entry's directory under ``logbook/``."""
    if entry.after is not None:
        return f"after-Scan{entry.after:03d}"
    if entry.scan == 0 or entry.scan is None:
        return "day"
    return f"Scan{entry.scan:03d}"


# ---------------------------------------------------------------- router


def create_log_router(
    experiment: str,
    base_directory: Optional[Union[Path, str]] = None,
    notes_db: Optional[Union[Path, str]] = None,
) -> APIRouter:
    """Build the logbook router.

    Parameters
    ----------
    experiment : str
        The experiment whose share to read, e.g. ``"Undulator"``. Supplied
        by the host application; this package carries no default, since a
        facility value belongs in the site profile rather than in code.
    base_directory : Path or str, optional
        Override the configured data-share root. Used by tests.
    notes_db : Path or str, optional
        The SQLite file for commentary. Without it the logbook is the
        read-only day view of phase 01 — no entries, no write routes.

    Returns
    -------
    APIRouter
        Mount it with ``app.include_router(router, prefix="/log")``.
    """
    router = APIRouter()
    templates = Jinja2Templates(directory=str(_TEMPLATES))
    templates.env.filters["initials"] = _initials
    store: Optional[NotesStore] = NotesStore(notes_db) if notes_db else None
    last_sync = {"at": 0.0}

    # ---------------------------------------------------------- reading

    def _load(day: date) -> DaySummary:
        """Read one day, turning share trouble into an honest 503."""
        try:
            return read_day(day, experiment, base_directory=base_directory)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — surface, don't 500 blankly
            logger.exception("reading %s failed", day)
            raise HTTPException(
                status_code=503, detail=f"data share unavailable: {exc}"
            ) from exc

    def _maybe_sync() -> None:
        """Pay the store's mirror debt, at most every _SYNC_INTERVAL_S."""
        if store is None:
            return
        now = time.monotonic()
        if now - last_sync["at"] < _SYNC_INTERVAL_S:
            return
        last_sync["at"] = now
        try:
            written, deferred = mirror.sync(store, experiment, base_directory)
            if written or deferred:
                logger.info("mirror sync: %d written, %d deferred", written, deferred)
        except Exception:  # noqa: BLE001 — a sync must never take down a view
            logger.exception("mirror sync failed")

    def _attachment_base(request: Request, entry: LogEntry) -> str:
        """The serving prefix an entry's relative attachment links map onto.

        Built from the route's own URL so it is right under any mount
        prefix or reverse-proxy ``root_path``.
        """
        full = request.url_for(
            "_attachment",
            day=entry.day,
            scope=_scope(entry),
            entry_id="X",
            filename="Y",
        ).path
        return full[: -len("/X/Y")]

    def _grouped(request: Request, day: str) -> dict[str, list[RenderedEntry]]:
        """Entries for a day, rendered and keyed by anchor."""
        if store is None:
            return {}
        out: dict[str, list[RenderedEntry]] = {}
        for entry in store.for_day(day):
            html = render_markdown(
                entry.body_md, attachment_base=_attachment_base(request, entry)
            )
            out.setdefault(entry.anchor, []).append(RenderedEntry(entry, html))
        return out

    # ------------------------------------------------------------ pages

    @router.get("/", response_class=RedirectResponse)
    def _today() -> RedirectResponse:
        """Redirect to today's log."""
        return RedirectResponse(url=f"day/{date.today().isoformat()}")

    @router.get("/static/{name}")
    def _static(name: str) -> FileResponse:
        """Serve the page's own assets.

        A plain route rather than ``router.mount(StaticFiles(...))``:
        ``Mount`` is a ``BaseRoute``, not a ``Route``, and
        ``APIRouter.include_router`` drops it silently on the FastAPI
        versions this package's floor allows — the stylesheet would 404
        and ``url_for`` would raise, with CI green because the lock pins a
        newer release. ``scan_analysis.config_editor`` solved it this way
        first; this follows it.
        """
        target = (_STATIC / name).resolve()
        if target.parent != _STATIC.resolve() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"no such asset: {name}")
        return FileResponse(target)

    @router.get("/day/{day}", response_class=HTMLResponse)
    def _day_page(request: Request, day: str) -> HTMLResponse:
        """Render the day document."""
        when = _parse_day(day)
        summary = _load(when)
        _maybe_sync()
        grouped = _grouped(request, day)
        return templates.TemplateResponse(
            request=request,
            name="day.html",
            context={
                "summary": summary,
                "experiment": experiment,
                "rail_days": _rail_days(when),
                "today": date.today(),
                "prev_day": when - timedelta(days=1),
                "next_day": when + timedelta(days=1),
                "entries": grouped,
                "entry_count": sum(len(v) for v in grouped.values()),
                "writable": store is not None,
                # /log/api under whatever prefix the host mounted us at
                "api_base": request.url_for("_entries_json", day=day).path[
                    : -len(f"/day/{day}/entries")
                ],
            },
        )

    @router.get("/attachments/{day}/{scope}/{entry_id}/{filename}")
    def _attachment(day: str, scope: str, entry_id: str, filename: str) -> FileResponse:
        """Serve an uploaded file from ``logbook/`` on the share."""
        _parse_day(day)
        root = mirror.logbook_root(day, experiment, base_directory)
        base = (root if scope == "day" else root / scope).resolve()
        target = (base / mirror.ATTACHMENTS_DIR / entry_id / filename).resolve()
        if (
            not str(target).startswith(str(base / mirror.ATTACHMENTS_DIR))
            or not target.is_file()
        ):
            raise HTTPException(status_code=404, detail="no such attachment")
        return FileResponse(target)

    # -------------------------------------------------------------- api

    @router.get("/api/day/{day}")
    def _day_json(day: str) -> DaySummary:
        """Return one day's derived half as JSON."""
        return _load(_parse_day(day))

    @router.get("/api/day/{day}/entries")
    def _entries_json(day: str) -> list[LogEntry]:
        """Return one day's commentary as JSON."""
        _parse_day(day)
        return store.for_day(day) if store else []

    if store is None:
        return router

    # --------------------------------------------------------- writes
    # Everything below exists only with a store. Store first, share second.

    def _mirror(entry: LogEntry) -> None:
        """Try to land an entry on the share; defer quietly if it cannot."""
        try:
            mirror.write_entry(
                entry, mirror.logbook_root(entry.day, experiment, base_directory)
            )
        except mirror.MirrorUnavailable as exc:
            logger.info("mirror deferred for %s: %s", entry.entry_id, exc)
            return
        store.mark_mirrored(entry.entry_id, datetime.now(timezone.utc))

    @router.post("/api/entries", status_code=201)
    def _create(body: EntryCreate) -> LogEntry:
        """Add an entry. The words are safe the instant this returns."""
        try:
            entry = store.create(**body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _mirror(entry)
        return store.get(entry.entry_id) or entry

    @router.patch("/api/entries/{entry_id}")
    def _update(entry_id: str, body: EntryUpdate) -> LogEntry:
        """Edit an entry's text, refusing to overwrite someone else's save."""
        try:
            entry = store.update(
                entry_id,
                body_md=body.body_md,
                author=body.author,
                expected_version=body.expected_version,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="no such entry") from exc
        except ConflictError as exc:
            # 409 with what is there now, so the client can show what it
            # lost to rather than only that it lost.
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(exc),
                    "current": exc.current.model_dump(mode="json"),
                },
            ) from exc
        _mirror(entry)
        return store.get(entry_id) or entry

    @router.post("/api/entries/{entry_id}/status")
    def _set_status(entry_id: str, body: StatusUpdate) -> LogEntry:
        """Keep a draft. A human act; an agent has no route to promote itself."""
        try:
            entry = store.set_status(entry_id, body.status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="no such entry") from exc
        _mirror(entry)
        return store.get(entry_id) or entry

    @router.delete("/api/entries/{entry_id}", status_code=204)
    def _delete(entry_id: str) -> Response:
        """Remove an entry and its markdown. Attachments are left for a human."""
        entry = store.get(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="no such entry")
        store.delete(entry_id)
        try:
            mirror.remove_entry(
                entry, mirror.logbook_root(entry.day, experiment, base_directory)
            )
        except mirror.MirrorUnavailable as exc:
            logger.warning("could not remove mirror of %s: %s", entry_id, exc)
        return Response(status_code=204)

    @router.post("/api/entries/{entry_id}/attachments", status_code=201)
    async def _upload(entry_id: str, file: UploadFile) -> dict:
        """Store an uploaded file beside the entry and return its link.

        Unlike text, bytes live only on the share, so this one write does
        need the share up — a 503 says so plainly rather than pretending.
        """
        entry = store.get(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="no such entry")
        ext = _ATTACHMENT_TYPES.get(file.content_type or "")
        if ext is None:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported type {file.content_type!r}; "
                f"accepted: {', '.join(sorted(_ATTACHMENT_TYPES))}",
            )
        data = await file.read(_MAX_ATTACHMENT_BYTES + 1)
        if len(data) > _MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"attachment over the {_MAX_ATTACHMENT_BYTES // (1024 * 1024)} MiB cap",
            )
        if not data:
            raise HTTPException(status_code=422, detail="empty upload")

        stem = (
            "".join(
                c
                for c in Path(file.filename or "upload").stem
                if c.isalnum() or c in "-_"
            )
            or "upload"
        )
        filename = f"{stem}{ext}"
        root = mirror.logbook_root(entry.day, experiment, base_directory)
        try:
            link = mirror.write_attachment(entry, filename, data, root)
        except mirror.MirrorUnavailable as exc:
            raise HTTPException(
                status_code=503, detail=f"the share is not taking uploads: {exc}"
            ) from exc

        attachment = Attachment(
            id=entry.entry_id,
            filename=filename,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(data),
            uploaded_at=datetime.now(timezone.utc),
        )
        updated = store.add_attachment(entry_id, attachment)
        _mirror(updated)
        return {"attachment": attachment.model_dump(mode="json"), "link": link}

    return router
