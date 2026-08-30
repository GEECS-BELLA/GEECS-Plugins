"""The GEECS Data Portal FastAPI application.

A second view layer over :class:`geecs_data_utils.tiled_catalog.ScanCatalog`
(the console scan browser is the first): server-rendered pages for
day → scan → metadata/plots navigation, reachable from any browser on the
lab network with nothing to install.

Architecture rules (see this package's ``CLAUDE.md`` and
``Planning/data_portal/01_data_portal_scope.md``):

- **Read-only by doctrine** — no write verbs; nothing on the scans path
  is ever created (repo scan-folder invariant).
- **The ScanCatalog seam** — :func:`create_app` takes any implementation
  of the protocol; tests inject fakes, ``__main__`` injects
  ``TiledScanCatalog.from_config()``.  This module never imports
  ``tiled``.
- **Column semantics live in ``geecs_data_utils.tiled_schema``** — the
  pick list is :func:`~geecs_data_utils.tiled_schema.plottable_columns`
  and coercion is :func:`~geecs_data_utils.tiled_schema.numeric_series`
  (the console's B4 is owed a rewire onto the same helpers).
- **No build chain** — server-rendered Jinja2 templates, plots rendered
  server-side to PNG via the matplotlib object API (thread-safe: no
  pyplot global state on FastAPI's threadpool); no npm, no CDN.
"""

from __future__ import annotations

import io
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from matplotlib.figure import Figure
from starlette.requests import Request

from geecs_data_utils import tiled_schema as schema_map
from geecs_data_utils.tiled_catalog import (
    ScanCatalog,
    metadata_rows,
    resolve_scan_folder,
)

from geecs_portal import resources

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

#: Cap on rows fed to a plot (quick-look, not a data browser).
_PLOT_MAX_ROWS = 100_000


def _fmt_hhmm(epoch: float) -> str:
    """Format epoch seconds as local ``HH:MM`` ("" for 0/invalid)."""
    if not epoch:
        return ""
    try:
        return datetime.fromtimestamp(epoch).strftime("%H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


def _parse_day(day: str) -> date:
    """Parse an ISO day query param, falling back to today."""
    try:
        return date.fromisoformat(day) if day else date.today()
    except ValueError:
        return date.today()


def _run_day(detail, day: str) -> Optional[date]:
    """The day used to re-base a run's scan folder — the run's OWN day.

    The start document's time is authoritative: trusting the caller's
    ``day`` (or defaulting to today) would let a bookmarked link resolve
    a *different* scan's same-numbered folder, since GEECS scan numbers
    restart daily.  A run with no usable start time therefore resolves
    only through an explicit ``day`` param — never today's folder.
    """
    start_time = detail.summary.start_time or 0.0
    if start_time > 0:
        try:
            return datetime.fromtimestamp(start_time).date()
        except (OverflowError, OSError, ValueError):
            pass
    if day:
        try:
            return date.fromisoformat(day)
        except ValueError:
            return None
    return None


def _sticky_query(state: dict, **overrides) -> str:
    """One query string carrying the page's sticky params.

    Template links/forms build their hrefs through this (empty values
    dropped) so navigating one control never silently resets another —
    the plot selection survives shot stepping, the day filter survives
    run round-trips.  The one deliberate exception is the day page's
    "clear" link, whose whole job is dropping the filter.
    """
    merged = {**state, **overrides}
    return urlencode({k: v for k, v in merged.items() if v not in ("", None)})


def _acq_timestamp(detail, device: str, shot: int) -> tuple[Optional[float], bool]:
    """The event row's ``acq_timestamp`` for *device* at 1-based *shot*.

    Column matching goes through
    :func:`geecs_data_utils.tiled_schema.device_acq_timestamp_column`
    (schema-safe normalization — never re-derived here).

    Returns
    -------
    tuple of (float or None, bool)
        ``(value, column_present)``.  No column → ``(None, False)`` and
        the resource layer may fall back to ordinal file order; column
        present but the row invalid (NaN / non-positive: the device
        missed this shot) → ``(None, True)`` — the caller must refuse
        rather than serve a neighbouring shot's image.
    """
    import math

    frame = detail.data
    if frame is None or shot < 1 or shot > len(frame):
        return (None, False)
    column = schema_map.device_acq_timestamp_column(
        [str(c) for c in frame.columns], device
    )
    if column is None:
        return (None, False)
    try:
        value = float(frame[column].iloc[shot - 1])
    except (TypeError, ValueError):
        return (None, True)
    if not math.isfinite(value) or value <= 0:
        return (None, True)
    return (value, True)


def _default_x(detail, columns: list[str]) -> str:
    """The console-parity default X: the scan variable on stepped scans."""
    if not schema_map.is_stepped_scan(detail.start_doc):
        return ""
    scan_vars = schema_map.scan_variable_columns(columns, detail.start_doc)
    return scan_vars[0] if scan_vars else ""


def create_app(catalog: ScanCatalog, *, default_experiment: str = "") -> FastAPI:
    """Build the portal application over an injected catalog.

    Parameters
    ----------
    catalog : ScanCatalog
        The catalog implementation (real Tiled client in production,
        fakes in tests).
    default_experiment : str, optional
        Experiment preselected when a request names none.

    Returns
    -------
    FastAPI
        The configured application.
    """
    app = FastAPI(title="GEECS Data Portal", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

    def _load_run(uid: str):
        """Load one run, mapping failures to honest HTTP status codes.

        ``KeyError`` is the fakes' and the Tiled client's unknown-uid
        signal → 404.  Anything else (connection errors, unconfigured
        URI) means the catalog itself is unavailable → 503, so an outage
        never reads as "run not found" for runs that exist.
        """
        try:
            return catalog.load_run(uid)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"run not found: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 — surface, don't 500
            logger.warning("catalog load_run failed: %s", exc)
            raise HTTPException(
                status_code=503, detail=f"catalog unavailable: {exc}"
            ) from exc

    def _png_headers(detail) -> dict:
        """Caching headers for the immutable-per-URL PNG endpoints.

        A completed run (stop doc present) never changes, so its plot
        and shot images are cacheable indefinitely; a still-running run
        must revalidate.
        """
        if detail.summary.exit_status:
            return {"Cache-Control": "public, max-age=31536000, immutable"}
        return {"Cache-Control": "no-cache"}

    @app.get("/health")
    def health() -> dict:
        """Liveness + catalog probe (the fleet-map health check)."""
        status = catalog.probe()
        return {"ok": status.ok, "catalog": status.label}

    @app.get("/", response_class=RedirectResponse)
    def index() -> str:
        """Redirect to today's day view."""
        return f"/day/{date.today().isoformat()}"

    @app.get("/go", response_class=RedirectResponse)
    def go(day: str = "", experiment: str = "", filter: str = "") -> str:
        """The day/experiment picker form's target: redirect to the day view."""
        selected = _parse_day(day)
        query = _sticky_query({"experiment": experiment, "filter": filter})
        return f"/day/{selected.isoformat()}{'?' + query if query else ''}"

    @app.get("/day/{day}", response_class=HTMLResponse)
    def day_view(
        request: Request, day: str, experiment: str = "", filter: str = ""
    ) -> HTMLResponse:
        """The run list for one day (newest first, as the catalog lists)."""
        try:
            selected = date.fromisoformat(day)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="bad date") from exc
        exp = experiment or default_experiment
        try:
            runs = catalog.list_runs(exp, selected)
            error = ""
        except Exception as exc:  # noqa: BLE001 — surface, don't 500
            logger.warning("day listing failed: %s", exc)
            runs, error = [], f"catalog error: {exc}"
        needle = filter.strip().lower()
        if needle:
            runs = [run for run in runs if needle in run.filter_text()]
        day_state = {"experiment": exp, "filter": filter}
        return templates.TemplateResponse(
            request,
            "day.html",
            {
                "day": selected,
                "prev_day": (selected - timedelta(days=1)).isoformat(),
                "next_day": (selected + timedelta(days=1)).isoformat(),
                "experiment": exp,
                "filter": filter,
                "rows": [(run, _fmt_hhmm(run.start_time)) for run in runs],
                "error": error,
                "qs": lambda **kw: _sticky_query(day_state, **kw),
            },
        )

    @app.get("/run/{uid}", response_class=HTMLResponse)
    def run_view(
        request: Request,
        uid: str,
        day: str = "",
        experiment: str = "",
        y: str = "",
        x: str = "",
        device: str = "",
        shot: int = 1,
        filter: str = "",
    ) -> HTMLResponse:
        """One run: metadata, plottable-column picker + plot, image gallery."""
        detail = _load_run(uid)
        columns = (
            [] if detail.data is None else schema_map.plottable_columns(detail.data)
        )
        selected = y if y in columns else ""
        x_column = x if x in columns else ""
        if selected and not x:
            x_column = _default_x(detail, columns)
        run_day = _run_day(detail, day)
        folder = resolve_scan_folder(detail, run_day) if run_day else None
        devices = resources.image_devices(folder) if folder else []
        sel_device = device if device in devices else ""
        kind, kind_path = (
            resources.device_kind(folder, sel_device) if sel_device else ("", None)
        )
        n_rows = None if detail.data is None else len(detail.data)
        shot = max(1, min(shot, n_rows) if n_rows else shot)
        state = {
            "day": day,
            "experiment": experiment or default_experiment,
            "y": selected,
            "x": x_column,
            "device": sel_device,
            "shot": shot if sel_device else "",
            "filter": filter,  # the day list's filter, carried for the back link
        }
        return templates.TemplateResponse(
            request,
            "run.html",
            {
                "uid": uid,
                "day": day,
                "experiment": experiment or default_experiment,
                "rows": metadata_rows(detail),
                "columns": columns,
                "selected": selected,
                "x_column": x_column,
                "devices": devices,
                "sel_device": sel_device,
                "kind": kind,
                "kind_path": str(kind_path) if kind_path else "",
                "shot": shot,
                "has_next_shot": n_rows is None or shot < n_rows,
                "total_shots": detail.summary.shots,
                "qs": lambda **kw: _sticky_query(state, **kw),
            },
        )

    @app.get("/run/{uid}/image.png")
    def run_image(uid: str, device: str, shot: int = 1, day: str = "") -> Response:
        """One device shot rendered for display (stack or native file)."""
        detail = _load_run(uid)
        run_day = _run_day(detail, day)
        folder = resolve_scan_folder(detail, run_day) if run_day else None
        if folder is None:
            raise HTTPException(status_code=404, detail="scan folder not resolvable")
        # A shot beyond the recorded event rows must refuse outright:
        # falling through to the ordinal join would serve an orphan
        # frame (pre/post-scan extras) labeled as a shot that never
        # happened — the never-serve-a-neighbour doctrine.
        if detail.data is not None and shot > len(detail.data):
            raise HTTPException(
                status_code=404, detail="shot beyond the run's recorded events"
            )
        acq, column_present = _acq_timestamp(detail, device, shot)
        if column_present and acq is None:
            raise HTTPException(
                status_code=404, detail="device missed this shot (no timestamp)"
            )
        result = resources.load_shot_image(folder, device, shot, acq_timestamp=acq)
        if result.png is None:
            raise HTTPException(status_code=404, detail=result.reason or result.kind)
        headers = (
            _png_headers(detail) if result.cacheable else {"Cache-Control": "no-cache"}
        )
        return Response(content=result.png, media_type="image/png", headers=headers)

    @app.get("/run/{uid}/plot.png")
    def run_plot(uid: str, y: str, x: str = "") -> Response:
        """Server-rendered scalar plot: *y* column vs *x* (default row index).

        Uses the matplotlib object API (``Figure``, never pyplot) — no
        global figure registry, safe on FastAPI's threadpool.
        """
        detail = _load_run(uid)
        if detail.data is None:
            raise HTTPException(status_code=404, detail="run has no event rows")
        frame = detail.data.head(_PLOT_MAX_ROWS)
        y_series = schema_map.numeric_series(frame, y)
        if y_series is None:
            raise HTTPException(status_code=404, detail=f"no plottable column {y!r}")
        x_series = None
        if x:
            x_series = schema_map.numeric_series(frame, x)
            if x_series is None:
                raise HTTPException(
                    status_code=404, detail=f"no plottable column {x!r}"
                )
        fig = Figure(figsize=(7.5, 4.0), dpi=110)
        ax = fig.subplots()
        if x_series is not None:
            ax.plot(x_series, y_series, ".", markersize=4)
            ax.set_xlabel(x, parse_math=False)
        else:
            ax.plot(y_series.to_numpy(), ".", markersize=4)
            ax.set_xlabel("row")
        ax.set_ylabel(y, parse_math=False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers=_png_headers(detail),
        )

    return app
