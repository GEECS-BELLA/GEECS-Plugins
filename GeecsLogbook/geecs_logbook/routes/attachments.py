"""Attachments: uploaded bytes, stored on the host and served from there.

Registered only when a store exists. The share is not in the save path —
:mod:`geecs_logbook.attachments` holds the bytes beside the database, and
the mirror copies them beside the markdown when it gets to the entry.

``POST /log/api/entries/{id}/attachments``     upload (201; 413/415/422)
``GET  /log/attachments/{entry_id}/{filename}`` serve
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from geecs_schemas.log_entry import Attachment

from geecs_logbook.mirror import attachment_link
from geecs_logbook.routes._common import Context

#: The cap is rejected with a clear 413, never a 500.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
ATTACHMENT_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}


def register(router: APIRouter, ctx: Context) -> None:
    """Add the attachment routes to ``router``. Requires a store."""
    store, blobs = ctx.store, ctx.attachments
    assert store is not None and blobs is not None

    @router.get("/attachments/{entry_id}/{filename}")
    def _attachment(entry_id: str, filename: str) -> FileResponse:
        """Serve an uploaded file from the host's attachment store."""
        target = blobs.path(entry_id, filename)
        if target is None:
            raise HTTPException(status_code=404, detail="no such attachment")
        return FileResponse(target)

    @router.post("/api/entries/{entry_id}/attachments", status_code=201)
    def _upload(entry_id: str, file: UploadFile) -> dict:
        """Store an uploaded file for the entry and return its link.

        A plain ``def`` on purpose: the disk write runs in the threadpool
        like every other route rather than on the event loop.
        """
        entry = store.get(entry_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="no such entry")
        ext = ATTACHMENT_TYPES.get(file.content_type or "")
        if ext is None:
            raise HTTPException(
                status_code=415,
                detail=f"unsupported type {file.content_type!r}; "
                f"accepted: {', '.join(sorted(ATTACHMENT_TYPES))}",
            )
        data = file.file.read(MAX_ATTACHMENT_BYTES + 1)
        if len(data) > MAX_ATTACHMENT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"attachment over the {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MiB cap",
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
        # Every clipboard paste arrives as image.png; the store claims a
        # free name on disk and says which one it got.
        try:
            filename = blobs.save(entry_id, f"{stem}{ext}", data)
        except OSError as exc:
            raise HTTPException(
                status_code=503, detail=f"cannot store the upload: {exc}"
            ) from exc

        attachment = Attachment(
            id=uuid.uuid4().hex[:12],
            filename=filename,
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(data),
            uploaded_at=datetime.now(timezone.utc),
        )
        try:
            store.add_attachment(entry_id, attachment)
        except KeyError as exc:  # deleted between the check and the write
            raise HTTPException(status_code=404, detail="no such entry") from exc
        ctx.mirror(entry_id)
        return {
            "attachment": attachment.model_dump(mode="json"),
            "link": attachment_link(entry, filename),
        }
