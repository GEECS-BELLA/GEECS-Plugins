"""The logbook's HTTP surface, mounted by GEECS-DataPortal at ``/log``.

Following the config-editor precedent (`scan_analysis.config_editor`), this
module exposes a factory returning an :class:`~fastapi.APIRouter` rather
than an app, so the portal owns the process, the port and the unit.

The routes live in :mod:`geecs_logbook.routes`, one module per concern:

- ``routes.day`` — the scans book: the day document and its JSON peers.
  Always registered.
- ``routes.entries`` — the write verbs, the reason the logbook is a
  charter exception in the portal. Only with a store.
- ``routes.attachments`` — uploads, stored on the host and served from
  there. Only with a store.

Every write goes to the store first and the share second — see
:mod:`geecs_logbook.mirror` for why that order. The mirror writes a tree
of its own under ``{experiment}/logbook/``; nothing here can create a scan
folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter
from fastapi.templating import Jinja2Templates

from geecs_logbook.attachments import AttachmentStore
from geecs_logbook.routes import attachments, day, entries
from geecs_logbook.routes._common import TEMPLATES_DIR, Context, initials
from geecs_logbook.store import NotesStore

#: The attachment directory's name, beside the database file.
ATTACHMENTS_DIRNAME = "attachments"


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
        The SQLite file for commentary. Uploaded bytes go to an
        ``attachments/`` directory beside it. Without it the logbook is
        the read-only day view — no entries, no write routes.

    Returns
    -------
    APIRouter
        Mount it with ``app.include_router(router, prefix="/log")``.
    """
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["initials"] = initials

    store: Optional[NotesStore] = None
    blobs: Optional[AttachmentStore] = None
    if notes_db:
        store = NotesStore(notes_db)
        blobs = AttachmentStore(Path(notes_db).parent / ATTACHMENTS_DIRNAME)

    ctx = Context(
        experiment=experiment,
        base_directory=base_directory,
        store=store,
        attachments=blobs,
        templates=templates,
    )
    router = APIRouter()
    day.register(router, ctx)
    if store is not None:
        entries.register(router, ctx)
        attachments.register(router, ctx)
    return router
