"""The GEECS Data Portal FastAPI application.

The web view layer over :class:`geecs_data_utils.tiled_catalog.ScanCatalog`
(the Qt console's scan browser was the first, until its deletion in
2026-09): server-rendered pages for
day → scan → metadata/plots navigation, reachable from any browser on the
lab network with nothing to install.

Architecture rules (see this package's ``CLAUDE.md``):

- **Read-only by doctrine** — no write verbs; nothing on the scans path
  is ever created (repo scan-folder invariant).
- **The ScanCatalog seam** — :func:`create_app` takes any implementation
  of the protocol; tests inject fakes, ``__main__`` injects
  ``TiledScanCatalog.from_config()``.  This module never imports
  ``tiled``.
- **Column semantics live in ``geecs_data_utils.tiled_schema``** — the
  pick list is :func:`~geecs_data_utils.tiled_schema.plottable_columns`
  and coercion is :func:`~geecs_data_utils.tiled_schema.numeric_series`,
  one module so no front end reinterprets a column.
- **No build chain** — server-rendered Jinja2 templates, plots rendered
  server-side to PNG via the matplotlib object API (thread-safe: no
  pyplot global state on FastAPI's threadpool); no npm, no CDN.
"""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from geecs_data_utils.tiled_catalog import ScanCatalog

from geecs_portal import analysis_runs
from geecs_portal.cache import ShotDataCache
from geecs_portal.routes import browse, images, logbook, pages, plot_api, runs
from geecs_portal.routes.common import _root
from geecs_portal.routes.config_editor import mount_config_editor
from geecs_portal.state import PortalState

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

#: Requests carrying a proxy mount prefix — the Grafana/JupyterHub
#: convention every reverse proxy speaks.
_FORWARDED_PREFIX_HEADER = b"x-forwarded-prefix"


#: A valid mount prefix: non-empty ``/segment`` parts of RFC-3986-ish
#: path characters — no ``//``, no query/fragment/quote characters, no
#:  whitespace or backslashes.
_PREFIX_RE = re.compile(r"(?:/[A-Za-z0-9._~%@+-]+)+")


def _clean_prefix(raw: str) -> str:
    """Normalize a mount prefix: ``/portal/`` → ``/portal``; bad → ``""``.

    Accepts only what :data:`_PREFIX_RE` matches — anything else is
    treated as no prefix rather than propagated into every link on the
    page.  A bare ``/`` means "mounted at root", i.e. no prefix.
    """
    prefix = raw.strip().rstrip("/")
    if not prefix:
        return ""
    if _PREFIX_RE.fullmatch(prefix) is None:
        return ""
    return prefix


class _ForwardedPrefixMiddleware:
    """Adopt the proxy's ``X-Forwarded-Prefix`` as the ASGI root_path.

    Behind ``proxy /portal → portal:8200`` the app itself never sees
    the mount point; the proxy names it in this header.  Setting
    ``scope["root_path"]`` makes every link, form, redirect, and JS
    fetch (all built through the one ``root`` context value) carry the
    prefix, so the portal works at root and under any mount point —
    including OSPREY panel tabs.  The header, when present, wins over a
    static ``--root-path`` (the proxy is authoritative for where it
    mounted us); a client faking it only rewrites its own page's links.

    The path is re-prefixed too (the ASGI-canonical shape: ``path``
    includes ``root_path``).  Starlette's router strips ``root_path``
    from the FRONT of ``path`` wherever it happens to match, so the
    proxy-stripped path alone would double-strip under a mount named
    like a route head (``/run``, ``/api``, …), 404ing that whole route
    family — and its trailing-slash redirects build the Location from
    ``path``, which would drop the prefix.  Re-prefixing makes the
    strip exact and the redirects complete.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            for name, value in scope.get("headers", []):
                if name == _FORWARDED_PREFIX_HEADER:
                    prefix = _clean_prefix(value.decode("latin-1"))
                    if prefix:
                        scope["root_path"] = prefix
                        scope["path"] = prefix + scope["path"]
                    break
        await self.app(scope, receive, send)


def create_app(
    catalog: ScanCatalog,
    *,
    default_experiment: str = "",
    processing_config_dir: Optional[Path] = None,
    analysis_factory: Optional[analysis_runs.AnalyzerFactory] = None,
    config_editor: bool = False,
    logbook_url: str = "",
) -> FastAPI:
    """Build the portal application over an injected catalog.

    Parameters
    ----------
    catalog : ScanCatalog
        The catalog implementation (real Tiled client in production,
        fakes in tests).
    default_experiment : str, optional
        Experiment preselected when a request names none.
    processing_config_dir : Path, optional
        Root of the scan-analysis configs tree for the Images tab's
        ephemeral-processing selector. ``None`` (the default) turns
        the feature OFF — the portal never falls back to the global
        config resolution (two competing resolution paths exist, so
        the portal names its tree explicitly). The selector also hides
        itself when
        ImageAnalysis (the ``analysis`` extra) is not installed.
    analysis_factory : callable, optional
        ``(analyzer_id, config_dir) -> ScanAnalyzer`` for the analysis
        runs (``/api/run/{uid}/analysis``). ``None``
        (the default) uses the real ScanAnalysis factory, which then
        also needs the ``analysis`` extra; tests inject a fake. The
        feature shares ``processing_config_dir`` — the unified
        diagnostic tree — and is OFF with it.

    Returns
    -------
    FastAPI
        The configured application.
    logbook_url : str, optional
        The logbook's base URL — its own service since GeecsLogbook
        0.10.0 (port 8400; the portal no longer mounts it). The run page
        links each scan to its card there. An absolute URL is used as
        given; a path (``/log``, the front door's route) is same-origin
        and carries this app's own mount prefix. Empty (the default)
        shows no link. The link is built for runs of
        ``default_experiment`` alone — the logbook serves one
        experiment's share and scan numbers restart per experiment.
    config_editor : bool, default False
        Mount the analysis config editor (``scan_analysis.config_editor``)
        at ``/configs`` over the same ``processing_config_dir`` tree, with a
        live preview of the document under edit on the scan page's current
        shot. A **write verb** (it saves YAML into the configs tree) and
        therefore explicit opt-in — ``--config-editor`` on the CLI; nothing
        without ``processing_config_dir`` and the ``analysis`` extra.
    """
    # The analysis-run worker outlives requests: built
    # before the app so the lifespan can refuse new runs and log any
    # in-flight one at shutdown (a running job cannot be interrupted —
    # the interpreter joins the worker at exit; see DEPLOYMENT.md).
    runner = analysis_runs.AnalysisRunner()

    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            yield
        finally:
            runner.shutdown()

    app = FastAPI(
        title="GEECS Data Portal", docs_url=None, redoc_url=None, lifespan=_lifespan
    )
    app.add_middleware(_ForwardedPrefixMiddleware)
    # Every template gets `root`: the mount prefix each root-absolute
    # href/action/src/fetch must carry (empty when served at root).
    templates = Jinja2Templates(
        directory=str(_TEMPLATES_DIR),
        context_processors=[lambda request: {"root": _root(request)}],
    )
    # The one committed JS asset: the version-pinned vendored Plotly
    # bundle (doctrine amendment 2026-08-30 — still no npm, no CDN).
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # The shared palette every GEECS web surface draws from. Mounted here
    # because the portal is the host: the config editor is a router inside
    # this app, so one mount serves both and a viewer's choice follows
    # them between pages.
    from geecs_web_theme import static_dir as _theme_dir

    app.mount("/theme", StaticFiles(directory=str(_theme_dir())), name="theme")
    # Per-app pixel cache: completed runs' shot data kept in memory so
    # within-scan navigation never re-reads the share (owner doctrine,
    # 2026-08-29 — lazy stays the rule ACROSS scans only).
    data_cache = ShotDataCache()
    logbook_base = logbook_url.rstrip("/")
    portal = PortalState(
        catalog=catalog,
        templates=templates,
        runner=runner,
        data_cache=data_cache,
        factory=analysis_factory or analysis_runs.scan_analysis_factory,
        default_experiment=default_experiment,
        processing_config_dir=processing_config_dir,
        analysis_factory=analysis_factory,
        logbook_base=logbook_base,
        logbook_send_base=(
            logbook_base if logbook_base.startswith(("http://", "https://")) else ""
        ),
    )
    app.state.portal = portal
    for family in (plot_api, runs, logbook, browse, pages, images):
        app.include_router(family.router)

    if processing_config_dir is not None and not portal.processing_names():
        # The flag is explicit operator intent — a typo'd path, a tree
        # without analyzers/, or a missing 'analysis' extra must not
        # no-op silently into a hidden selector.
        logger.warning(
            "processing_config_dir %s yielded no diagnostics (missing/"
            "unlistable tree, or the 'analysis' extra is not installed) "
            "— the processing selector is disabled",
            processing_config_dir,
        )

    if config_editor and processing_config_dir is not None:
        mount_config_editor(app, portal)

    return app
