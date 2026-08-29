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
- **No build chain** — server-rendered Jinja2 templates, plots rendered
  server-side to PNG with matplotlib (Agg); no npm, no CDN.
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before pyplot: headless server, no display

import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from geecs_data_utils.tiled_catalog import ScanCatalog, metadata_rows

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"

#: Cap on rows fed to a plot (quick-look, not a data browser).
_PLOT_MAX_ROWS = 100_000


def _fmt_hhmm(epoch: float) -> str:
    """Format epoch seconds as local ``HH:MM`` ("" for 0/invalid)."""
    from datetime import datetime

    if not epoch:
        return ""
    try:
        return datetime.fromtimestamp(epoch).strftime("%H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


def _numeric_columns(detail) -> list[str]:
    """Column names of the run's numeric event columns, frame order."""
    if detail.data is None:
        return []
    frame = detail.data
    return [
        str(name)
        for name in frame.columns
        if frame[name].dtype.kind in "fiu"  # float / int / unsigned
    ]


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

    @app.get("/health")
    def health() -> dict:
        """Liveness + catalog probe (the fleet-map health check)."""
        status = catalog.probe()
        return {"ok": status.ok, "catalog": status.label}

    @app.get("/", response_class=RedirectResponse)
    def index() -> str:
        """Redirect to today's day view."""
        return f"/day/{date.today().isoformat()}"

    @app.get("/day/{day}", response_class=HTMLResponse)
    def day_view(request: Request, day: str, experiment: str = "") -> HTMLResponse:
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
        return templates.TemplateResponse(
            request,
            "day.html",
            {
                "day": selected,
                "prev_day": (selected - timedelta(days=1)).isoformat(),
                "next_day": (selected + timedelta(days=1)).isoformat(),
                "experiment": exp,
                "rows": [(run, _fmt_hhmm(run.start_time)) for run in runs],
                "error": error,
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
    ) -> HTMLResponse:
        """One run: metadata rows, plottable-column links, selected plot."""
        try:
            detail = catalog.load_run(uid)
        except Exception as exc:  # noqa: BLE001 — surface, don't 500
            raise HTTPException(
                status_code=404, detail=f"run not found: {exc}"
            ) from exc
        columns = _numeric_columns(detail)
        return templates.TemplateResponse(
            request,
            "run.html",
            {
                "uid": uid,
                "day": day,
                "experiment": experiment or default_experiment,
                "rows": metadata_rows(detail),
                "columns": columns,
                "selected": y if y in columns else "",
                "x_column": x if x in columns else "",
            },
        )

    @app.get("/run/{uid}/plot.png")
    def run_plot(uid: str, y: str, x: str = "") -> Response:
        """Server-rendered scalar plot: *y* column vs *x* (default row index)."""
        try:
            detail = catalog.load_run(uid)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=404, detail=f"run not found: {exc}"
            ) from exc
        if detail.data is None:
            raise HTTPException(status_code=404, detail="run has no event rows")
        frame = detail.data.head(_PLOT_MAX_ROWS)
        numeric = set(_numeric_columns(detail))
        if y not in numeric:
            raise HTTPException(status_code=404, detail=f"no numeric column {y!r}")
        if x and x not in numeric:
            raise HTTPException(status_code=404, detail=f"no numeric column {x!r}")
        fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=110)
        try:
            if x:
                ax.plot(frame[x], frame[y], ".", markersize=4)
                ax.set_xlabel(x)
            else:
                ax.plot(frame[y].to_numpy(), ".", markersize=4)
                ax.set_xlabel("row")
            ax.set_ylabel(y)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png")
        finally:
            plt.close(fig)
        return Response(content=buffer.getvalue(), media_type="image/png")

    return app
