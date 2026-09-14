"""The logbook process: its app factory.

The logbook is its own service since 0.10.0 — its own port, unit and
state directory — after riding inside the Data Portal's process as a
router at ``/log``. The web-scanner brief's argument settled it: a write
path holding irreplaceable data should not live inside a process the
fleet kills on purpose (the portal's ``MemoryMax=`` is *meant* to fire).

The web glue every GEECS surface shares — the ``X-Forwarded-Prefix``
middleware, the ``/theme`` mount, the templates factory that puts ``root``
in every context — is imported from ``geecs_web_theme.web`` (the ``web``
extra), never copied here. What is the logbook's own: the stores built
from the arguments, the four route modules, the named ``/static`` mount
and ``/health``.

The routes live in :mod:`geecs_logbook.routes`, one module per concern:

- ``routes.day`` — the scans book: the day document and its JSON peers.
  Always registered.
- ``routes.month`` — the ops book: a month of day-level entries, read
  from the store alone. Always registered (empty without a store).
- ``routes.entries`` — the write verbs. Only with a store.
- ``routes.attachments`` — uploads, stored on the host and served from
  there. Only with a store.

Every write goes to the store first and the share second — see
:mod:`geecs_logbook.mirror` for why that order. The mirror writes a tree
of its own under ``{experiment}/logbook/``; nothing here can create a scan
folder.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter, FastAPI
from fastapi.staticfiles import StaticFiles
from geecs_web_theme.web import ForwardedPrefixMiddleware, make_templates, mount_theme

from geecs_logbook.attachments import AttachmentStore
from geecs_logbook.mirror import ATTACHMENTS_DIR
from geecs_logbook.models import KIT_STATE
from geecs_logbook.routes import attachments, day, entries, month
from geecs_logbook.routes._common import STATIC_DIR, TEMPLATES_DIR, Context, initials
from geecs_logbook.seed_templates import SeedTemplates
from geecs_logbook.store import NotesStore


def _version() -> str:
    try:
        return version("geecs-logbook")
    except PackageNotFoundError:  # pragma: no cover - a source tree without install
        return "0.0.0"


__version__ = _version()


def create_app(
    experiment: str,
    *,
    base_directory: Optional[Union[Path, str]] = None,
    notes_db: Optional[Union[Path, str]] = None,
    templates_dir: Optional[Union[Path, str]] = None,
    root_path: str = "",
) -> FastAPI:
    """Build the logbook application.

    Parameters
    ----------
    experiment : str
        The experiment whose share to read, e.g. ``"Undulator"``. This
        package carries no default: a facility value belongs in the site
        profile, not in code.
    base_directory : Path or str, optional
        Override the configured data-share root. Used by tests.
    notes_db : Path or str, optional
        The SQLite file for commentary. Uploaded bytes go to an
        ``attachments/`` directory beside it. Without it the logbook is
        the read-only day view — no entries, no write routes.
    templates_dir : Path or str, optional
        A directory of ``*.md`` seed templates — the type buttons on every
        composer (:mod:`geecs_logbook.seed_templates`). On a host this is
        ``logbook_templates/`` at the top of the configs checkout. Without
        it composers are plain.
    root_path : str, optional
        The URL prefix the service is mounted under behind a reverse proxy
        (``/log`` at the fleet's front door). A proxy-sent
        ``X-Forwarded-Prefix`` header overrides it per request.

    Returns
    -------
    FastAPI
        The configured application, serving at its root: ``/day/…``,
        ``/month/…``, ``/api/…``, ``/static/…``, ``/theme/…``, ``/health``.
    """
    app = FastAPI(
        title="GEECS Logbook",
        version=__version__,
        root_path=root_path,
        docs_url=None,
        redoc_url=None,
    )
    app.add_middleware(ForwardedPrefixMiddleware)
    mount_theme(app)
    # The page's own assets. A named mount, addressed as {{ root }}/static/…
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="logbook_static")

    templates = make_templates(
        TEMPLATES_DIR,
        # A global, not a per-response key: every template that renders a
        # scan status needs it, and threading it through each context is
        # how one of them ends up rendering an uncoloured chip.
        globals={"kit_state": KIT_STATE},
        filters={"initials": initials},
    )

    store: Optional[NotesStore] = None
    blobs: Optional[AttachmentStore] = None
    if notes_db:
        store = NotesStore(notes_db)
        # The same name as inside the mirror tree, so the relative link
        # ``attachments/<id>/<file>`` is true on the host and on the share.
        blobs = AttachmentStore(Path(notes_db).parent / ATTACHMENTS_DIR)

    ctx = Context(
        experiment=experiment,
        base_directory=base_directory,
        store=store,
        attachments=blobs,
        templates=templates,
        seeds=SeedTemplates(Path(templates_dir) if templates_dir else None),
    )
    router = APIRouter()
    day.register(router, ctx)
    month.register(router, ctx)
    if store is not None:
        entries.register(router, ctx)
        attachments.register(router, ctx)
    app.include_router(router)

    @app.get("/health")
    def health() -> dict:
        """Liveness + version + whether entries are taken (the fleet-map probe).

        Deliberately cheap: it never touches the share, so a hung mount
        cannot make the service look down.
        """
        return {
            "ok": True,
            "version": __version__,
            "experiment": experiment,
            "writable": store is not None,
        }

    return app
