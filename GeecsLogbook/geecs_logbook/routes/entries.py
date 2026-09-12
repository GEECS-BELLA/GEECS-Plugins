"""The entry write verbs — the reason the logbook is a charter exception.

Registered only when a store exists. Every write goes to the store first
and the share second; see :mod:`geecs_logbook.mirror` for why that order.

``POST   /log/api/entries``                     create (201)
``PATCH  /log/api/entries/{id}``                edit the text (409 on conflict)
``POST   /log/api/entries/{id}/status``         keep or un-keep
``DELETE /log/api/entries/{id}``                tombstone (204)
``GET    /log/api/entries/{id}``                one entry
``GET    /log/api/entries/{id}/history``        its earlier states
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Response
from geecs_schemas.log_entry import Book, EntryKind, EntryStatus, LogEntry
from pydantic import BaseModel, Field

from fastapi import Request

from geecs_logbook.render import render_markdown
from geecs_logbook.routes._common import Context, attachment_base
from geecs_logbook.store import ConflictError


class EntryCreate(BaseModel):
    """What a client sends to add an entry.

    Tags are not a field: they are read out of ``body_md`` at save.
    """

    day: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    book: Book = "scans"
    scan: Optional[int] = Field(
        None, ge=1, description="Scan number; omit both for a day-level entry."
    )
    after: Optional[int] = Field(
        None, ge=0, description="Interscan: the scan it follows."
    )
    author: str = Field(min_length=1, max_length=120)
    body_md: str = Field(max_length=200_000)
    template: str = Field("blank", max_length=64)
    kind: EntryKind = "note"
    status: EntryStatus = "kept"
    payload: Optional[dict] = None


class EntryUpdate(BaseModel):
    """What a client sends to edit an entry's text.

    ``editor`` is who is making the edit; the entry's ``author`` is not
    theirs to change.
    """

    body_md: str = Field(max_length=200_000)
    editor: str = Field(min_length=1, max_length=120)
    expected_version: int = Field(ge=1)


class StatusUpdate(BaseModel):
    """Keep or un-keep an entry — a human act on unchanged text."""

    status: EntryStatus


class PreviewRequest(BaseModel):
    """A body to render as the page would, before it is saved."""

    body_md: str = Field(max_length=200_000)
    entry_id: Optional[str] = Field(
        None,
        description="Unused for now; attachment links resolve the same way for every entry.",
    )


class HistoryItem(BaseModel):
    """One earlier state, as served."""

    version: int
    reason: str
    recorded_at: str
    entry: LogEntry


def register(router: APIRouter, ctx: Context) -> None:
    """Add the entry routes to ``router``. Requires a store."""
    store = ctx.store
    assert store is not None

    @router.post("/api/preview")
    def _preview(request: Request, body: PreviewRequest) -> dict:
        """Render a body as the page will, for the composer's preview."""
        return {
            "html": render_markdown(
                body.body_md, attachment_base=attachment_base(request)
            )
        }

    @router.post("/api/entries", status_code=201)
    def _create(body: EntryCreate) -> LogEntry:
        """Add an entry. The words are safe the instant this returns."""
        try:
            entry = store.create(**body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ctx.mirror(entry.entry_id)
        return store.get(entry.entry_id) or entry

    @router.get("/api/entries/{entry_id}")
    def _one(entry_id: str) -> LogEntry:
        """Return one entry."""
        entry = store.get(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="no such entry")
        return entry

    @router.get("/api/entries/{entry_id}/history")
    def _history(entry_id: str) -> list[HistoryItem]:
        """Return an entry's earlier states, oldest first.

        Available for a deleted entry too — that is when it is wanted.
        """
        if store.get(entry_id, include_deleted=True) is None:
            raise HTTPException(status_code=404, detail="no such entry")
        return [
            HistoryItem(
                version=h.version,
                reason=h.reason,
                recorded_at=h.recorded_at.isoformat(),
                entry=h.entry,
            )
            for h in store.history(entry_id)
        ]

    @router.patch("/api/entries/{entry_id}")
    def _update(entry_id: str, body: EntryUpdate) -> LogEntry:
        """Edit an entry's text, refusing to overwrite someone else's save."""
        try:
            entry = store.update(
                entry_id,
                body_md=body.body_md,
                editor=body.editor,
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
        ctx.mirror(entry_id)
        return store.get(entry_id) or entry

    @router.post("/api/entries/{entry_id}/status")
    def _set_status(entry_id: str, body: StatusUpdate) -> LogEntry:
        """Keep a draft. A human act; an agent has no route to promote itself."""
        try:
            entry = store.set_status(entry_id, body.status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="no such entry") from exc
        ctx.mirror(entry_id)
        return store.get(entry_id) or entry

    @router.delete("/api/entries/{entry_id}", status_code=204)
    def _delete(entry_id: str) -> Response:
        """Tombstone an entry and remove its markdown.

        Attachments are left in place, on the host and on the share; the
        history keeps the entry as it was. If the share refuses the removal
        the tombstone stays owed and the sync retries it.
        """
        if store.get(entry_id) is None:
            raise HTTPException(status_code=404, detail="no such entry")
        store.delete(entry_id)
        ctx.mirror(entry_id)
        return Response(status_code=204)
