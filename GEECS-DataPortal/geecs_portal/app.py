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
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI
from geecs_web_theme.web import (
    ForwardedPrefixMiddleware,
    RevalidatingStaticFiles,
    make_templates,
    mount_theme,
)

from geecs_data_utils.tiled_catalog import ScanCatalog

from geecs_portal import analysis_runs
from geecs_portal.cache import ShotDataCache
from geecs_portal.batch import WorkerLink
from geecs_portal.day_scans import DayScans
from geecs_portal.routes import (
    analysis_configs,
    batch,
    browse,
    images,
    logbook,
    pages,
    plot_api,
    runs,
)
from geecs_portal.state import PortalState

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"

#: A run's stop-document ``exit_status`` as the kit's status word: a
#: Bluesky run ends ``success``, ``abort`` (amber, as the logbook shows an
#: aborted scan) or ``fail``. No stop document is ``unknown``, never
#: "running": a run whose worker died has none either, and a week-old
#: crash must not pulse on the day list (the logbook's rule: status is
#: reported, not inferred).
_RUN_STATES = {"success": "ok", "abort": "degraded", "fail": "failed"}
#: The chip's text where it is not the stored ``exit_status`` itself: an
#: aborted run reads "aborted" on every surface (the scanner, the logbook).
_RUN_WORDS = {"abort": "aborted"}


def run_state(exit_status: Optional[str]) -> str:
    """The kit status word (``data-state``) for a run's ``exit_status``."""
    return _RUN_STATES.get(exit_status or "", "unknown")


def run_word(exit_status: Optional[str]) -> str:
    """The word a run's chip shows for its ``exit_status`` (display only; the value is unchanged)."""
    return _RUN_WORDS.get(exit_status or "", exit_status or "unknown")


def create_app(
    catalog: ScanCatalog,
    *,
    default_experiment: str = "",
    processing_config_dir: Optional[Path] = None,
    analysis_factory: Optional[analysis_runs.AnalyzerFactory] = None,
    logbook_url: str = "",
    logbook_browser_url: str = "",
    logbook_client: Optional[httpx.Client] = None,
    plotter: Optional[FastAPI] = None,
    worker: Optional[WorkerLink] = None,
    day_scans: Optional[DayScans] = None,
    configurator_url: str = "",
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
        the analysis stack (the ``analysis`` extra) is not installed.
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
        given; a path is same-origin and carries this app's own mount
        prefix (it links but cannot send: behind a front door use an
        absolute address here and ``logbook_browser_url`` for the links). Empty (the default)
        shows no link. The link is built for runs of
        ``default_experiment`` alone — the logbook serves one
        experiment's share and scan numbers restart per experiment.
    logbook_browser_url : str, optional
        The logbook as the browser reaches it, when that differs from
        ``logbook_url`` (behind a front door: the browser's
        ``https://<name>/<experiment>/logbook``, while ``logbook_url``
        stays an address this process dials). Every link uses it; the
        plot send still posts to ``logbook_url``. Empty: links use
        ``logbook_url``. Absolute only (``ValueError`` otherwise): a path
        would take this app's prefix, which is where it never is.
    logbook_client : httpx.Client, optional
        The HTTP client the plot send talks to the logbook with. ``None``
        (the default) makes one per send; tests inject a client over an
        in-process logbook so the whole conversation runs without a
        socket.
    plotter : FastAPI, optional
        GEECSplotter's page (:func:`geecs_portal.plotter.app.create_app`
        with ``mount="/plot"``), mounted at ``/plot`` as the Plot tab.
        ``None`` (the default, and the tests') mounts nothing, and the
        scan page then has no Plot tab. Built by the caller because it
        resolves its settings folder from ``config.ini`` when made.
    worker : WorkerLink, optional
        The analysis worker (``scan-analysis-worker``, its own service) the
        day page's "Analyze this day" panel reaches through
        ``/api/batch/...``. ``None`` (the default) shows no panel.
    configurator_url : str, optional
        The configurator as the browser reaches it (a site value). The
        Analysis tab's **edit** opens a recipe, ``.himg`` job or group in
        the embedded editor (GeecsConfigurator's ``static/embed.js``), with
        this portal's preview of the unsaved recipe on the page's scan; the
        save is the configurator's (the operator as git author, its tier
        and dry run, which is this portal's ``/api/analysis/validate``).
        The configurator lists this portal's origin in its CORS origins.
        Empty (the default): no edit button for the configurator.
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
    # The shared glue (geecs_web_theme.web): X-Forwarded-Prefix becomes the
    # ASGI root_path, and every template gets `root`, the prefix each
    # root-absolute href/action/src/fetch must carry (empty at root).
    app.add_middleware(ForwardedPrefixMiddleware)
    templates = make_templates(
        _TEMPLATES_DIR, filters={"run_state": run_state, "run_word": run_word}
    )
    # The portal's own stylesheet.
    app.mount(
        "/static", RevalidatingStaticFiles(directory=str(_STATIC_DIR)), name="static"
    )
    # The shared palette and kit, and the vendored Plotly bundle beside them
    # (the theme's, one copy for every surface: still no npm, no CDN); the
    # run page's embedded configurator editor loads kit.js and form.js here.
    mount_theme(app)
    # Per-app pixel cache: completed runs' shot data kept in memory so
    # within-scan navigation never re-reads the share (owner doctrine,
    # 2026-08-29 — lazy stays the rule ACROSS scans only).
    data_cache = ShotDataCache()
    if logbook_browser_url and not logbook_browser_url.startswith(
        ("http://", "https://")
    ):
        # A path would be taken as same-origin and get this app's own prefix:
        # a logbook on its own path behind the front door is never there.
        raise ValueError(
            f"logbook_browser_url must be absolute (http(s)://...): {logbook_browser_url!r}"
        )
    logbook_send_url = logbook_url.rstrip("/")
    logbook_base = (logbook_browser_url or logbook_url).rstrip("/")
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
            logbook_send_url
            if logbook_send_url.startswith(("http://", "https://"))
            else ""
        ),
        logbook_client=logbook_client,
        plot_page=plotter is not None,
        worker=worker,
        day_scans=day_scans or DayScans(),
        configurator_url=configurator_url.rstrip("/"),
    )
    app.state.portal = portal
    for family in (
        plot_api,
        runs,
        logbook,
        browse,
        pages,
        images,
        batch,
        analysis_configs,
    ):
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

    if plotter is not None:
        app.mount("/plot", plotter, name="plot")

    return app
