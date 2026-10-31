"""The Plot tab: GEECSplotter's XY page (and its image strip), as an app.

GeecsPlotter folded into the portal (2026-10-28; the package was deleted
once everything had moved, and its rules are the portal's ``CLAUDE.md``):
its page moved here nearly unchanged and the portal mounts it at
``/plot`` (:func:`create_app` with ``mount="/plot"``), so the page keeps
its own form, its named settings sets and its routes, and the portal's tab
bar links it to the scan's other tabs. Unmounted (``mount=""``, as the
tests build it) the same page carries its own prefix middleware and no
portal tab bar; nothing serves it that way.

The web glue every GEECS surface shares is imported from
``geecs_web_theme.web``, never copied. What is the plotter's own: the
routes, the ``/static`` mount (the page script and stylesheet; the
vendored Plotly bundle is the portal's) and ``/health``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter, FastAPI, Request
from geecs_data_utils.folder_catalog import FolderScanCatalog
from geecs_web_theme.web import (
    ForwardedPrefixMiddleware,
    RevalidatingStaticFiles,
    make_templates,
    mount_theme,
    root_of,
)

from geecs_portal.plotter.routes import STATIC_DIR, TEMPLATES_DIR, Context, api, pages
from geecs_portal.plotter.routes import images as image_routes
from geecs_portal.plotter.settings import resolve_store


def _version() -> str:
    try:
        return version("geecs-data-portal")
    except PackageNotFoundError:  # pragma: no cover - a source tree without install
        return "0.0.0"


__version__ = _version()


def create_app(
    experiment: str = "",
    *,
    base_directory: Optional[Union[Path, str]] = None,
    settings_dir: Optional[Union[Path, str]] = None,
    root_path: str = "",
    mount: str = "",
    logbook_url: str = "",
) -> FastAPI:
    """Build the plotter application.

    Parameters
    ----------
    experiment : str, optional
        The experiment folder to read; empty means ``config.ini
        [Experiment] expt``. No default in code: a facility value.
    base_directory : Path or str, optional
        Override the configured data root. Used by tests.
    settings_dir : Path or str, optional
        Where named settings sets are saved; default
        ``{geecs_data}/DataPortal/settings`` from ``config.ini``, else
        ``<main checkout>/user data/DataPortal/settings``
        (:func:`geecs_portal.plotter.settings.resolve_store`).
    root_path : str, optional
        The URL prefix behind a prefix-preserving reverse proxy.
    mount : str, optional
        Where the portal mounts this app (``"/plot"``). Then the portal's
        middleware has already set the request's prefix, the page draws
        the portal's tab bar, and ``portal_root`` (the prefix without the
        mount) reaches the templates; empty is the standalone plotter.
    logbook_url : str, optional
        The portal's ``--logbook-url``. Absolute, the plot's toolbar offers
        "send to the scan's log entry", which posts to the portal's own
        ``/api/run/{uid}/logbook`` (server to server, as the scan page's).

    Returns
    -------
    FastAPI
        The app: ``/``, ``/api/…``, ``/static/…``, ``/theme/…``, ``/health``.
    """
    app = FastAPI(title="GEECS Plotter", version=__version__, root_path=root_path)
    if not mount:
        # Mounted, the portal's own middleware has set the prefix; a second
        # pass would replace it with the bare header and drop the mount.
        app.add_middleware(ForwardedPrefixMiddleware)

    def portal_context(request: Request) -> dict:
        root = root_of(request)
        if mount and root.endswith(mount):
            return {"portal_root": root[: -len(mount)], "in_portal": True}
        return {"portal_root": root, "in_portal": False}

    mount_theme(app)
    app.mount(
        "/static",
        RevalidatingStaticFiles(directory=str(STATIC_DIR)),
        name="plotter_static",
    )
    ctx = Context(
        experiment=experiment,
        templates=make_templates(TEMPLATES_DIR, context_processors=[portal_context]),
        catalog=FolderScanCatalog(Path(base_directory) if base_directory else None),
        store=resolve_store(Path(settings_dir) if settings_dir else None),
        logbook_send=logbook_url.startswith(("http://", "https://")),
    )
    router = APIRouter()
    pages.register(router, ctx)
    api.register(router, ctx)
    image_routes.register(router, ctx)
    app.include_router(router)

    @app.get("/health")
    def health() -> dict:
        """Liveness and version; never touches the share."""
        return {"ok": True, "version": __version__}

    return app
