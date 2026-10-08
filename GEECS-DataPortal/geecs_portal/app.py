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
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

from geecs_data_utils import tiled_schema as schema_map
from geecs_data_utils.tiled_catalog import ScanCatalog, fmt_time_of_day, metadata_rows

from geecs_portal import analysis, analysis_runs, figures, logbook_send, resources
from geecs_portal.cache import ShotDataCache
from geecs_portal.routes import images, plot_api
from geecs_portal.routes.common import (
    _LISTING_HEADERS,
    _acq_timestamp,
    _image_folder,
    _jump_target,
    _parse_day,
    _parse_iso_day,
    _portal_version,
    _resolved_folder,
    _root,
    _run_day,
    _scan_label,
    _sticky_query,
    _summary_json,
)
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


class PlotToLogbook(BaseModel):
    """One rendered plot on its way to the scan logbook.

    The image arrives as a data URL rather than as multipart form data:
    the page already holds one (``Plotly.toImage`` returns it) and form
    parsing would pull in ``python-multipart``, which this package does
    not otherwise need. Base64 costs a third in size, which a plot PNG
    can afford.
    """

    #: The logbook requires a name on every entry and invents none.
    #: ``\S`` because a blank one survives ``min_length`` and is then the
    #: logbook's 422 — our malformed request arriving as the peer's fault.
    author: str = Field(min_length=1, max_length=120, pattern=r"\S")
    #: ``data:image/png;base64,…`` — the only form accepted.
    image: str = Field(max_length=12_000_000)
    #: Alt text: what the plot shows.
    caption: str = Field("", max_length=300)
    #: The portal URL that made it — the page state IS the analysis.
    source_url: str = Field("", max_length=4_000)
    #: Append to this entry when the page already made one for the scan.
    #: Anchored to the logbook's id alphabet (``uuid4().hex[:12]``): the
    #: value becomes a path segment in the URLs we build, and a ``/`` or a
    #: ``..`` in it would address some other route under that base.
    entry: str = Field("", max_length=64, pattern=r"^[0-9a-f]*$")


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

    @app.get("/health")
    def health() -> dict:
        """Liveness + catalog probe (the fleet-map health check) + version."""
        status = portal.catalog.probe()
        return {
            "ok": status.ok,
            "catalog": status.label,
            "version": _portal_version(),
        }

    app.include_router(plot_api.router)

    @app.get("/api/run/{uid}/analysis")
    def run_analysis_list(uid: str, day: str = "") -> JSONResponse:
        """The scan's analyzers: applicability, job record, files on disk.

        ``applicable`` = the diagnostic's data device (``scan.device``,
        else its name) has a data folder in this scan. Every loadable
        diagnostic is listed regardless, so a device-less one is still
        reachable; the tab collapses the inapplicable ones.
        """
        portal.analysis_available()
        detail, folder, analysis_folder, tag = portal.analysis_context(uid, day)
        devices = set(resources.image_devices(folder))
        try:
            present = {p.name for p in folder.iterdir() if p.is_dir()}
        except OSError:
            present = set()
        jobs = portal.runner.jobs_for(uid)
        analyzers = []
        for name, info in portal.processing_infos().items():
            job = jobs.get(name)
            # What the tab shows: everything on disk under the output dir
            # (summaries + per-bin visuals, classified server-side) plus
            # any label the finished job returned that is not a file —
            # described (servable / inline / kind / bin) so the page
            # never guesses from a path's shape.
            files = analysis_runs.list_artifacts(analysis_folder, info.output_name)
            shown = list(files)
            if job is not None and job.state == analysis_runs.DONE:
                shown += [a for a in job.artifacts if a not in files]
            analyzers.append(
                {
                    "id": name,
                    "device": info.device,
                    "applicable": info.device in devices or info.device in present,
                    "output_dir": info.output_name,
                    "destructive": info.destructive,
                    "job": job.to_json() if job is not None else None,
                    "files": files,
                    "artifacts": analysis_runs.describe_artifacts(
                        analysis_folder, shown, known_files=set(files)
                    ),
                }
            )
        running = portal.runner.running_for(uid)
        return JSONResponse(
            {
                "analyzers": analyzers,
                "running": running.analyzer_id if running is not None else None,
                # The number the start endpoint's confirm check compares
                # against (parsed from the resolved folder, like the run's
                # tag) — the page asks for this one, never a guess.
                "scan_number": tag.number,
            },
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/api/run/{uid}/analysis", status_code=202)
    def run_analysis_start(
        uid: str, analyzer: str, day: str = "", confirm: str = ""
    ) -> JSONResponse:
        """Start one analyzer on this scan (202 + the fresh job record).

        Ladder: feature off / extra missing 404 · unknown or unloadable
        diagnostic 404 · folder unresolvable 404 · a destructive kind
        without ``confirm=<this scan's number>`` 400 (the tab asks for
        it; the gate is here so no client runs a delete unasked) · a job
        already running for this scan 409 (its record in the body).
        Build + run + cleanup happen on the worker thread, so everything
        past this point — config errors included — lands in the record
        as ``failed``.
        """
        portal.analysis_available()
        info = portal.processing_infos().get(analyzer)
        if info is None:
            raise HTTPException(status_code=404, detail=f"no diagnostic: {analyzer!r}")
        _, _, analysis_folder, tag = portal.analysis_context(uid, day)
        if info.destructive and confirm.strip() != str(tag.number):
            raise HTTPException(
                status_code=400,
                detail=f"{analyzer!r} deletes data: confirm with this scan's "
                "number (confirm=<scan number>) to run it",
            )
        config_dir = Path(portal.processing_config_dir)

        def run(progress: analysis_runs.ProgressSink) -> Optional[list]:
            # The opt-in reaches the factory only past the confirm check
            # above — the one place a destructive kind gets built here.
            return analysis_runs.run_scan_analyzer(
                portal.factory,
                analyzer,
                config_dir,
                tag,
                progress=progress,
                allow_destructive=info.destructive,
            )

        try:
            job = portal.runner.start(uid, analyzer, run, relative_to=analysis_folder)
        except analysis_runs.RunInProgress as exc:
            return JSONResponse(
                {
                    "detail": "a run is in progress for this scan",
                    "job": exc.job.to_json(),
                },
                status_code=409,
            )
        except RuntimeError as exc:  # shutting down
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return JSONResponse(job.to_json(), status_code=202)

    @app.post("/api/run/{uid}/logbook", status_code=201)
    def run_logbook_send(
        uid: str, payload: PlotToLogbook, day: str = ""
    ) -> JSONResponse:
        """Put one rendered plot into this scan's entry in the logbook.

        The portal's **third** write verb (after the analysis runs and the
        config editor) and, like them, an explicit act: nothing is sent
        that a person did not click.  It writes to the logbook's own
        service through that service's public API, and touches neither the
        scans tree nor any portal state.

        Ladder: no absolute ``--logbook-url`` / not this experiment / no
        scan number or day → 404 (there is no entry for it to join) · a
        malformed image → 400 · the logbook unreachable → 503 · the
        logbook refusing → its own status for the verdicts that are about
        this payload (409/413/415), else 502.
        """
        detail = portal.load_run(uid)
        # _run_day, not _resolved_folder: the folder is not wanted, and
        # resolving one stats the SMB share.  A logbook write touches the
        # scans mount not at all.
        run_day = _run_day(detail, day)
        if not portal.logbook_sendable(detail, run_day):
            raise HTTPException(
                status_code=404, detail="no logbook entry this scan could join"
            )
        scan = detail.summary.scan_number
        try:
            png = logbook_send.decode_png_data_url(payload.image)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            result = logbook_send.send_plot(
                base_url=portal.logbook_send_base,
                day=run_day.isoformat(),
                scan=scan,
                author=payload.author.strip(),
                png=png,
                caption=payload.caption,
                source_url=payload.source_url,
                entry_id=payload.entry or None,
                filename=f"scan{scan:03d}-plot.png",
            )
        except ValueError as exc:  # the image itself: too big, or empty
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except logbook_send.LogbookUnreachable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except logbook_send.LogbookRefused as exc:
            # 409/413/415 are verdicts on what we sent and mean the same
            # to our caller; anything else is the peer's own trouble, and
            # a peer's 500 must not read as ours.
            status = exc.status if exc.status in (409, 413, 415) else 502
            raise HTTPException(status_code=status, detail=exc.detail) from exc
        return JSONResponse(
            {
                "entry_id": result.entry_id,
                "appended": result.appended,
                "url": f"{portal.logbook_send_base}/entry/{result.entry_id}",
            },
            status_code=201,
        )

    @app.get("/run/{uid}/artifact")
    def run_artifact(uid: str, path: str, day: str = "") -> FileResponse:
        """Serve one file a run produced, from the scan's analysis folder only.

        Containment is the contract (:func:`analysis_runs.contained_artifact`):
        anything that resolves outside the scan's own analysis folder is
        a 404, same as a missing file. Same feature gate as the run
        endpoints. Raster images render inline; every other type is a
        download (``attachment`` + ``nosniff``) — the share is writable
        by many hands, and a planted HTML/SVG must never execute in the
        portal's (or, behind the OSPREY proxy, OSPREY's) origin.
        ``no-cache``: re-runs overwrite by name.
        """
        portal.analysis_available()
        _, _, analysis_folder, _ = portal.analysis_context(uid, day)
        file = analysis_runs.contained_artifact(analysis_folder, path)
        if file is None:
            raise HTTPException(status_code=404, detail="no such artifact")
        headers = {"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"}
        inline = file.suffix.lower() in analysis_runs.INLINE_IMAGE_SUFFIXES
        return FileResponse(
            file,
            headers=headers,
            content_disposition_type="inline" if inline else "attachment",
            filename=file.name,
        )

    # ------------------------- browsing JSON API -------------------------
    # The page-shaped reads (day list, run overview, device probe, the
    # day jump) as JSON — for scripts and agents, so everything a
    # browser shows is readable without scraping HTML.  Same helpers as
    # the templates, so the two surfaces cannot drift.

    @app.get("/api/day/{day}")
    def api_day(
        request: Request, day: str, experiment: str = "", filter: str = ""
    ) -> JSONResponse:
        """The day page as JSON: its runs (newest first), filtered."""
        selected = _parse_iso_day(day)
        exp = experiment or portal.default_experiment
        runs, error = portal.list_day(exp, selected, filter)
        if error:
            raise HTTPException(status_code=503, detail=f"catalog unavailable: {error}")
        payload = {
            "day": selected.isoformat(),
            "prev_day": (selected - timedelta(days=1)).isoformat(),
            "next_day": (selected + timedelta(days=1)).isoformat(),
            "experiment": exp,
            "filter": filter,
            "runs": [_summary_json(run) for run in runs],
            "page": f"{_root(request)}/day/{selected.isoformat()}",
        }
        return JSONResponse(payload, headers=_LISTING_HEADERS)

    @app.get("/api/run/jump/{day}")
    def api_run_jump(
        request: Request, day: str, prefer: int = 0, experiment: str = ""
    ) -> JSONResponse:
        """The day steppers' target as JSON (see :func:`run_jump`)."""
        selected = _parse_iso_day(day)
        exp = experiment or portal.default_experiment
        runs, error = portal.list_day(exp, selected, "")
        if error:
            raise HTTPException(status_code=503, detail=f"catalog unavailable: {error}")
        target = _jump_target(runs, prefer)
        page = (
            f"{_root(request)}/run/{target.uid}"
            if target
            else f"{_root(request)}/day/{selected.isoformat()}"
        )
        payload = {
            "day": selected.isoformat(),
            "experiment": exp,
            "prefer": prefer,
            "uid": target.uid if target else None,
            "scan_number": target.scan_number if target else None,
            "matched": bool(target and prefer and target.scan_number == prefer),
            "runs": len(runs),
            "page": page,
        }
        return JSONResponse(payload, headers=_LISTING_HEADERS)

    @app.get("/api/run/{uid}")
    def api_run(
        request: Request, uid: str, day: str = "", experiment: str = ""
    ) -> JSONResponse:
        """One run as JSON: the rail, the Overview tab and the device list.

        ``metadata`` is the Overview table verbatim (``metadata_rows``);
        ``devices`` are the image device folders (their tier is the
        separate ``/device`` probe — one listing per device); the
        neighbours and ``day_runs`` are the scan steppers and dropdown;
        ``analysis_enabled`` says whether the page offers the Analysis
        tab (same gate as the template).
        """
        detail = portal.load_run(uid)
        run_day, folder = _resolved_folder(detail, day)
        exp = experiment or portal.default_experiment
        prev_uid, next_uid, day_runs = portal.neighbours(uid, exp, run_day)
        payload = {
            "uid": uid,
            "run_day": run_day.isoformat() if run_day else None,
            "experiment": exp,
            "summary": _summary_json(detail.summary),
            "metadata": [[field, value] for field, value in metadata_rows(detail)],
            "start_doc": analysis.jsonable_document(detail.start_doc or {}),
            "stop_doc": analysis.jsonable_document(detail.stop_doc or {}),
            "event_rows": None if detail.data is None else len(detail.data),
            "scan_folder": str(folder) if folder else None,
            "devices": resources.image_devices(folder) if folder else [],
            "neighbours": {"prev_uid": prev_uid, "next_uid": next_uid},
            "day_runs": [_summary_json(run) for run in day_runs],
            "prev_day": (run_day - timedelta(days=1)).isoformat() if run_day else None,
            "next_day": (run_day + timedelta(days=1)).isoformat() if run_day else None,
            "processing_options": portal.processing_names(),
            "analysis_enabled": portal.analysis_enabled_for(folder),
            "config_editor": portal.config_editor_enabled,
            "logbook": portal.logbook_url(request, detail, run_day) or None,
            "logbook_send": portal.logbook_sendable(detail, run_day),
            "page": f"{_root(request)}/run/{uid}",
            "portal_version": _portal_version(),
        }
        return JSONResponse(payload, headers=_LISTING_HEADERS)

    @app.get("/api/run/{uid}/device")
    def api_device(uid: str, device: str = "", day: str = "") -> JSONResponse:
        """One device's gallery tier — what clicking its name decides.

        Same ``device_kind`` probe as the page (one directory listing,
        no pixel reads), so the JSON and the Images tab can never
        disagree about whether a device renders.
        """
        detail = portal.load_run(uid)
        if not device:
            raise HTTPException(status_code=400, detail="device is required")
        run_day, folder = _resolved_folder(detail, day)
        if folder is None:
            raise HTTPException(status_code=404, detail="scan folder not resolvable")
        probe = resources.device_kind(folder, device)
        if probe.kind == "missing":
            raise HTTPException(status_code=404, detail=f"unknown device {device!r}")
        renderable = probe.kind in ("stack", "native")
        payload = {
            "uid": uid,
            "run_day": run_day.isoformat() if run_day else None,
            "device": device,
            "kind": probe.kind,
            "renderable": renderable,
            "path": str(probe.path) if probe.path else None,
            "ext": probe.ext,
            "event_rows": None if detail.data is None else len(detail.data),
            "planned_shots": detail.summary.shots,
            "processing_options": portal.processing_names() if renderable else [],
        }
        return JSONResponse(payload, headers=_LISTING_HEADERS)

    @app.get("/", response_class=RedirectResponse)
    def index(request: Request) -> str:
        """Redirect to today's day view."""
        return f"{_root(request)}/day/{date.today().isoformat()}"

    @app.get("/go", response_class=RedirectResponse)
    def go(
        request: Request, day: str = "", experiment: str = "", filter: str = ""
    ) -> str:
        """The day/experiment picker form's target: redirect to the day view."""
        selected = _parse_day(day)
        query = _sticky_query({"experiment": experiment, "filter": filter})
        return (
            f"{_root(request)}/day/{selected.isoformat()}{'?' + query if query else ''}"
        )

    @app.get("/day/{day}", response_class=HTMLResponse)
    def day_view(
        request: Request, day: str, experiment: str = "", filter: str = ""
    ) -> HTMLResponse:
        """The run list for one day (newest first, as the catalog lists)."""
        selected = _parse_iso_day(day)
        exp = experiment or portal.default_experiment
        runs, error = portal.list_day(exp, selected, filter)
        if error:
            error = f"catalog error: {error}"
        day_state = {"experiment": exp, "filter": filter}
        return portal.templates.TemplateResponse(
            request,
            "day.html",
            {
                "day": selected,
                "prev_day": (selected - timedelta(days=1)).isoformat(),
                "next_day": (selected + timedelta(days=1)).isoformat(),
                "experiment": exp,
                "filter": filter,
                "rows": [(run, fmt_time_of_day(run.start_time)) for run in runs],
                "scan_label": _scan_label,
                "error": error,
                "qs": lambda **kw: _sticky_query(day_state, **kw),
            },
        )

    @app.get("/run/jump/{day}", response_class=RedirectResponse)
    def run_jump(request: Request, day: str, prefer: int = 0) -> str:
        """Day-step from the scan page without losing the analysis.

        Redirects to the target day's run with scan number *prefer*
        (else its newest run), carrying every other query param through
        verbatim — the rail's day steppers point here so filters,
        columns, and tab survive the hop.  A day with no runs falls
        back to the day page.
        """
        selected = _parse_iso_day(day)
        carried = [
            (key, value)
            for key, value in request.query_params.multi_items()
            if key != "prefer"
        ]
        experiment = (
            request.query_params.get("experiment", "") or portal.default_experiment
        )
        runs, _ = portal.list_day(experiment, selected, "")  # failure → the day page
        carried = [(k, v) for (k, v) in carried if k != "day"]
        carried.append(("day", selected.isoformat()))
        query = urlencode(carried, doseq=True)
        if not runs:
            day_query = _sticky_query(
                {
                    "experiment": experiment,
                    "filter": request.query_params.get("filter", ""),
                }
            )
            return (
                f"{_root(request)}/day/{selected.isoformat()}"
                f"{'?' + day_query if day_query else ''}"
            )
        target = _jump_target(runs, prefer)
        assert target is not None  # runs is non-empty here
        return f"{_root(request)}/run/{target.uid}?{query}"

    @app.get("/run/{uid}", response_class=HTMLResponse)
    def run_view(
        request: Request,
        uid: str,
        day: str = "",
        experiment: str = "",
        y: list[str] = Query(default=[]),
        x: str = "",
        device: str = "",
        shot: int = 1,
        filter: str = "",
        tab: str = "",
        filters: str = "",
        bincfg: str = "",
        view: str = "",
        display: str = "",
        processing: str = "",
        gridcfg: str = "",
        gridbin: str = "",
        imagebin: str = "",
    ) -> HTMLResponse:
        """One run: the rail + tabs (Overview / Plot / Images).

        ``tab``/``filters``/``bincfg``/``view``/``y``/``x`` are the
        analysis-tab state, carried in the URL (statelessness doctrine:
        a link IS the analysis) and consumed by the page's JS — the
        server only threads them through the sticky query so steppers
        keep the whole setup.
        """
        detail = portal.load_run(uid)
        run_day, folder = _resolved_folder(detail, day)
        devices = resources.image_devices(folder) if folder else []
        sel_device = device if device in devices else ""
        if sel_device:
            # Reuse the listing just computed — no second directory scan.
            probe = resources.device_kind(folder, sel_device, devices=devices)
            kind, kind_path = probe.kind, probe.path
            # Pixels or x-vs-y: the stack says which, and the gallery
            # renders a line for the array kinds (an (n, 2) lineout
            # drawn as pixels is a two-pixel-wide strip).
            content_kind = resources.stack_content(probe)
        else:
            kind, kind_path = "", None
            content_kind = "image"
        n_rows = None if detail.data is None else len(detail.data)
        shot = max(1, min(shot, n_rows) if n_rows else shot)
        analysis_enabled = portal.analysis_enabled_for(folder)
        if (
            kind == "native"
            and folder is not None
            and n_rows
            and detail.summary.exit_status
            and detail.data is not None
            and schema_map.device_acq_timestamp_column(
                [str(c) for c in detail.data.columns], sel_device
            )
            is not None
        ):
            # Background-warm the whole diagnostic (timestamp-joined shots
            # only — ordinal resolutions are never cached), so stepping
            # through shots serves from memory.
            warm_key = (uid, sel_device)
            warm_folder, warm_device, warm_detail = folder, sel_device, detail

            def _warm_one(s: int) -> None:
                acq_s, present = _acq_timestamp(warm_detail, warm_device, s)
                if acq_s is None:
                    return  # device missed the shot (or no column)
                resources.load_shot_image(
                    warm_folder,
                    warm_device,
                    s,
                    acq_timestamp=acq_s,
                    data_cache=portal.data_cache,
                    cache_key=warm_key,
                )

            portal.data_cache.warm_native(
                warm_key, _warm_one, list(range(1, min(n_rows, 2000) + 1))
            )
        prev_uid, next_uid, day_runs = portal.neighbours(
            uid, experiment or portal.default_experiment, run_day
        )
        state = {
            "day": day,
            "experiment": experiment or portal.default_experiment,
            # The analysis-tab state (URL-carried; the page JS owns it):
            "tab": tab,
            "y": [c for c in y if c],
            "x": x,
            "view": view,
            "filters": filters,
            "bincfg": bincfg,
            "display": display,
            "processing": processing,
            "gridcfg": gridcfg,
            "gridbin": gridbin,
            "imagebin": imagebin,
            "device": sel_device,
            "shot": shot if sel_device else "",
            "filter": filter,  # the day list's filter, carried for the back link
        }
        return portal.templates.TemplateResponse(
            request,
            "run.html",
            {
                "uid": uid,
                "day": day,
                "run_day": run_day.isoformat() if run_day else "",
                "experiment": experiment or portal.default_experiment,
                "summary": detail.summary,
                "rows": metadata_rows(detail),
                "start_time_of_day": fmt_time_of_day(detail.summary.start_time),
                "prev_uid": prev_uid,
                "next_uid": next_uid,
                "day_runs": day_runs,
                "scan_number": detail.summary.scan_number or 0,
                "logbook_url": portal.logbook_url(request, detail, run_day),
                "logbook_send": portal.logbook_sendable(detail, run_day),
                "prev_day": (
                    (run_day - timedelta(days=1)).isoformat() if run_day else ""
                ),
                "next_day": (
                    (run_day + timedelta(days=1)).isoformat() if run_day else ""
                ),
                "tab": (
                    tab
                    if tab in ("overview", "plot", "grid", "images")
                    or (tab == "analysis" and analysis_enabled)
                    else "plot"
                ),
                "analysis_enabled": analysis_enabled,
                "config_editor": portal.config_editor_enabled and analysis_enabled,
                "devices": devices,
                "sel_device": sel_device,
                "kind": kind,
                "kind_path": str(kind_path) if kind_path else "",
                "content_kind": content_kind,
                "is_trace": content_kind != "image",
                "shot": shot,
                "has_next_shot": n_rows is None or shot < n_rows,
                "total_shots": detail.summary.shots,
                "processing": processing,
                "processing_options": (
                    portal.processing_names()
                    if sel_device and content_kind == "image"
                    else []
                ),
                "display": display,
                "portal_version": _portal_version(),
                # The rail's chips and the display popup must stay in
                # step with the server-authored figures — one palette,
                # one marker default, both injected.
                "trace_colors": list(figures.TRACE_COLORS),
                "msize_default": figures.MARKER_SIZE_DEFAULT,
                "qs": lambda **kw: _sticky_query(state, **kw),
            },
        )

    app.include_router(images.router)

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

    # ---- the analysis config editor (deferred to its own arc) ----
    # Mounted at /configs over the processing tree. The store writes only
    # into that tree (never the scans tree); the live preview renders the
    # UNSAVED document on the current shot through the same write-free
    # ephemeral seam the Images tab uses, so dialling in an ROI is a
    # type-and-look loop without a save per iteration.
    def _preview_scan(uid: str, device: str, day: str):
        """The run + its scan folder for a preview, with the editor's error ladder."""
        try:
            detail = portal.load_run(uid)
            folder, _ = _image_folder(detail, day, device)
        except HTTPException as exc:
            kind = LookupError if exc.status_code == 404 else ValueError
            raise kind(str(exc.detail)) from exc
        return detail, folder

    def _line_trace(diag, detail, folder, device: str, shot: int):
        """One shot's trace as a scan run reads it: ``(Nx2 array, auxiliary)``.

        The shot resolves through the run path's own source rules
        (:func:`scan_analysis.core_source.prepare_source` — file tail,
        ``data_format``, the stack-only rule for ``pva_stack``) over
        ``device``, the recipe's own data folder; the reference is
        read with the document's loading, auxiliary columns handed over the
        way ``analyze_image_file`` hands them to line analyzers. A stitched
        input (``sibling_folders``) is the source's own joined trace, as the
        run reads it, and carries no auxiliary columns (the legacy stitcher
        passed none either).
        """
        from dataclasses import replace

        import pandas as pd
        from image_analysis.data_1d_utils import read_1d_data
        from scan_analysis.core_recipe import scan_recipe
        from scan_analysis.core_source import prepare_source

        if detail.data is not None and shot > len(detail.data):
            raise LookupError("shot beyond the run's recorded events")
        # The run joins by the diagnostic's device, not the folder.
        acq, column_present = _acq_timestamp(detail, diag.device, shot)
        if column_present and acq is None:
            raise LookupError("device missed this shot (no timestamp)")
        # The shot's own event row: the mapper finds the device's
        # acq_timestamp AND valid companions in it by normalized name, so a
        # row the run would skip (valid False) is skipped here too.
        if detail.data is not None:
            rows = detail.data.iloc[[shot - 1]].copy()
        else:
            rows = pd.DataFrame(index=[0])
        rows["Shotnumber"] = shot
        source = prepare_source(replace(scan_recipe(diag), folder=device), folder, rows)
        reference = source.references.get(shot)
        if reference is None:
            raise LookupError(f"no {device} file for shot {shot}")
        if source.siblings:
            return source.load(shot), None
        trace = read_1d_data(reference, diag.line_loading)
        aux = (
            {
                "_aux_columns": {
                    name: np.asarray(values, dtype=float)
                    for name, values in trace.auxiliary_column_data.items()
                }
            }
            if trace.auxiliary_column_data
            else None
        )
        return trace.data, aux

    def _camera_frame(detail, folder, uid: str, device: str, shot: int):
        """One shot's pixel array through the Images tab's own source ladder."""
        if detail.data is not None and shot > len(detail.data):
            raise LookupError("shot beyond the run's recorded events")
        acq, column_present = _acq_timestamp(detail, device, shot)
        if column_present and acq is None:
            raise LookupError("device missed this shot (no timestamp)")
        complete = bool(detail.summary.exit_status)
        try:
            resolved = resources.load_shot_array(
                folder,
                device,
                shot,
                acq_timestamp=acq,
                data_cache=portal.data_cache if complete else None,
                cache_key=(uid, device) if complete else None,
            )
        except HTTPException as exc:
            kind = LookupError if exc.status_code == 404 else ValueError
            raise kind(str(exc.detail)) from exc
        if resolved.array is None:
            raise LookupError(resolved.reason or resolved.kind)
        return resolved.array

    def _line_preview(diag, uid: str, device: str, day: str, shot: int) -> bytes:
        """A LINE document's preview: the shot's trace drawn as the run draws it."""
        ephemeral = portal.ephemeral_module()
        detail, folder = _preview_scan(uid, device, day)
        data, aux = _line_trace(diag, detail, folder, device, shot)
        figures = ephemeral.render_document_as_run(
            diag, [data], scan_folder=folder, auxiliary_data=aux
        )
        if not figures:
            raise ValueError("this analyzer draws no figure for a single trace")
        return resources.figure_png(figures[0], tight=True)

    #: the summary preview reads this many shots at most: a handful, never a
    #: scan (an image run is gigabytes; the host is shared)
    _SUMMARY_SHOTS_MAX = 8

    def _config_editor_summary_preview(
        document: dict, params: dict, index: int
    ) -> bytes:
        """The document's ``index``-th summary over the scan's first shots.

        ``params.shots`` (default 4, at most 8) shots are read from shot 1
        through the Images tab's own source ladder (frames) or the run's
        trace reader (lines); shots the device missed are skipped. The
        summary is drawn by ScanAnalysis' ``core_preview.preview_summary``
        — the sink's own summary call — with one panel per shot at its shot
        number under the run's noscan label, so an image grid shows one
        panel per shot, a waterfall one row per shot, and the ``average``
        kind the shots' average: the kind's layout on real frames (a run's
        grid panels are per-bin averages).
        """
        from geecs_analysis.recipe import is_line
        from geecs_schemas.analysis import load_analysis_document

        ephemeral = portal.ephemeral_module()
        uid = str(params.get("uid") or "")
        day = str(params.get("day") or "")
        raw_shots = params.get("shots")
        try:
            shots = 4 if raw_shots in (None, "") else int(raw_shots)
        except (TypeError, ValueError) as exc:
            raise ValueError("shots must be an integer") from exc
        if not uid:
            raise LookupError("summary preview needs a scan")
        shots = max(1, min(_SUMMARY_SHOTS_MAX, shots))
        diag = load_analysis_document(document)
        # the document's own data folder, never a host-picked device: a
        # preview of one camera's frames under another's recipe is a lie
        device = diag.data_folder
        detail, folder = _preview_scan(uid, device, day)
        arrays, positions, missing = [], [], []
        for shot in range(1, shots + 1):
            try:
                if is_line(diag):
                    array, _aux = _line_trace(diag, detail, folder, device, shot)
                else:
                    array = _camera_frame(detail, folder, uid, device, shot)
            except LookupError as exc:
                missing.append(f"shot {shot}: {exc}")
                continue
            arrays.append(array)
            positions.append(float(shot))
        if not arrays:
            raise LookupError(
                f"none of shots 1-{shots} has a {device} frame: " + "; ".join(missing)
            )
        from scan_analysis.core_products import NOSCAN_POSITION_LABEL

        fig = ephemeral.render_summary_as_run(
            diag, arrays, positions, NOSCAN_POSITION_LABEL, index, scan_folder=folder
        )
        return resources.figure_png(fig, tight=True)

    def _config_editor_preview(document: dict, params: dict) -> bytes:
        """The editor's preview: the shot drawn as a run of this document draws it.

        Through ``render_document_as_run`` — the analysis sink's own
        per-frame call with the document's figure block — and cropped
        tight like the sink's PNGs, so the pane shows the product file the
        run would write, not a portal rendering of it.
        """
        ephemeral = portal.ephemeral_module()
        from geecs_analysis.recipe import is_line
        from geecs_schemas.analysis import load_analysis_document

        uid = str(params.get("uid") or "")
        day = str(params.get("day") or "")
        try:
            shot = int(params.get("shot") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("shot must be an integer") from exc
        if not uid or shot < 1:
            raise LookupError("preview needs a scan and a shot (>= 1)")
        diag = load_analysis_document(document)
        # the document's own data folder (see the summary preview)
        device = diag.data_folder
        if is_line(diag):
            return _line_preview(diag, uid, device, day, shot)
        detail, folder = _preview_scan(uid, device, day)
        array = _camera_frame(detail, folder, uid, device, shot)
        # the recipe's frame inputs load from ITS device folder under this scan,
        # exactly as the run loads them (a background image under {scan_dir})
        figures = ephemeral.render_document_as_run(diag, [array], scan_folder=folder)
        if not figures:
            raise ValueError("this analyzer draws no figure for a single frame")
        return resources.figure_png(figures[0], tight=True)

    if config_editor and portal.processing_config_dir is not None:
        try:
            from scan_analysis.config_editor import create_editor_router
            from scan_analysis.config_store import ConfigStore
        except ImportError as exc:  # the analysis extra without the editor extra
            logger.warning("config editor requested but not installed: %s", exc)
        else:
            app.include_router(
                create_editor_router(
                    ConfigStore(Path(portal.processing_config_dir)),
                    preview=_config_editor_preview,
                    summary_preview=_config_editor_summary_preview,
                    summary_shots_max=_SUMMARY_SHOTS_MAX,
                    theme_url="/theme",
                ),
                prefix="/configs",
            )
            portal.config_editor_enabled = True

    return app
