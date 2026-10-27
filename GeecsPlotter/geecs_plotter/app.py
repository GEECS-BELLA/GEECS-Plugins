"""The plotter process: its app factory.

The web glue every GEECS surface shares — the ``X-Forwarded-Prefix``
middleware, the ``/theme`` mount, the templates factory that puts ``root``
in every context — is imported from ``geecs_web_theme.web``, never copied.
What is the plotter's own: the routes, the ``/static`` mount (the vendored
Plotly bundle and the page script) and ``/health``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Union

from fastapi import APIRouter, FastAPI
from geecs_data_utils.folder_catalog import FolderScanCatalog
from geecs_web_theme.web import (
    ForwardedPrefixMiddleware,
    RevalidatingStaticFiles,
    make_templates,
    mount_theme,
)

from geecs_plotter.routes import STATIC_DIR, TEMPLATES_DIR, Context, api, pages
from geecs_plotter.settings import resolve_store


def _version() -> str:
    try:
        return version("geecs-plotter")
    except PackageNotFoundError:  # pragma: no cover - a source tree without install
        return "0.0.0"


__version__ = _version()


def create_app(
    experiment: str = "",
    *,
    base_directory: Optional[Union[Path, str]] = None,
    settings_dir: Optional[Union[Path, str]] = None,
    root_path: str = "",
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
        ``{geecs_data}/GeecsPlotter/settings`` from ``config.ini``, else
        ``<repo>/user data/GeecsPlotter/settings``
        (:func:`geecs_plotter.settings.resolve_store`).
    root_path : str, optional
        The URL prefix behind a prefix-preserving reverse proxy.

    Returns
    -------
    FastAPI
        The app: ``/``, ``/api/…``, ``/static/…``, ``/theme/…``, ``/health``.
    """
    app = FastAPI(title="GEECS Plotter", version=__version__, root_path=root_path)
    app.add_middleware(ForwardedPrefixMiddleware)
    mount_theme(app)
    app.mount(
        "/static",
        RevalidatingStaticFiles(directory=str(STATIC_DIR)),
        name="plotter_static",
    )
    ctx = Context(
        experiment=experiment,
        templates=make_templates(TEMPLATES_DIR),
        catalog=FolderScanCatalog(Path(base_directory) if base_directory else None),
        store=resolve_store(Path(settings_dir) if settings_dir else None),
    )
    router = APIRouter()
    pages.register(router, ctx)
    api.register(router, ctx)
    app.include_router(router)

    @app.get("/health")
    def health() -> dict:
        """Liveness and version; never touches the share."""
        return {"ok": True, "version": __version__}

    return app
