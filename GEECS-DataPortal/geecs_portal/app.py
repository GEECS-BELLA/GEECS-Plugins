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
  and coercion is :func:`~geecs_data_utils.tiled_schema.numeric_series`,
  shared with the console's B4 so the two front-ends cannot drift.
- **No build chain** — server-rendered Jinja2 templates, plots rendered
  server-side to PNG via the matplotlib object API (thread-safe: no
  pyplot global state on FastAPI's threadpool); no npm, no CDN.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.templating import Jinja2Templates
from matplotlib.figure import Figure
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

from geecs_data_utils import tiled_schema as schema_map
from geecs_data_utils.data.binning import bin_frame, compute_bin_key
from geecs_data_utils.data.row_filters import filter_mask
from geecs_data_utils.io.images import average_frames
from geecs_data_utils.scan_frame import PROVENANCE_RUN, scan_frame
from geecs_data_utils.tiled_catalog import (
    ScanCatalog,
    fmt_time_of_day,
    metadata_rows,
    resolve_scan_folder,
)

from geecs_portal import analysis, figures, resources
from geecs_portal.cache import ShotDataCache

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"


def _portal_version() -> str:
    """The installed package version — the /api cache-bust key.

    Completed-run /api responses are served immutable, but their SHAPE
    changes with portal releases; the page keys every /api fetch by
    this version so browser caches roll over on upgrade.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("geecs-data-portal")
    except PackageNotFoundError:  # source tree without install metadata
        return "dev"


#: Cap on rows fed to a plot (quick-look, not a data browser).
_PLOT_MAX_ROWS = 100_000

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


def _root(request: Request) -> str:
    """The request's URL prefix (``""`` at root) — prepend to every path."""
    return request.scope.get("root_path", "").rstrip("/")


#: Multi-Y ceiling on the Plot tab (mockup ruling: up to 4).
_MAX_Y_COLUMNS = 4


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
    kept = {k: v for k, v in merged.items() if v not in ("", None, [], ())}
    return urlencode(kept, doseq=True)


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
    # Per-app pixel cache: completed runs' shot data kept in memory so
    # within-scan navigation never re-reads the share (owner doctrine,
    # 2026-08-29 — lazy stays the rule ACROSS scans only).
    data_cache = ShotDataCache()

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

    def _union(detail, day: str):
        """The union frame + the run's resolved day (ISO or None).

        One-liner over :func:`geecs_data_utils.scan_frame.scan_frame`;
        the s-file is re-read per request (one small text file — the
        catalog detail behind it is already cached for completed runs).
        """
        run_day = _run_day(detail, day)
        folder = resolve_scan_folder(detail, run_day) if run_day else None
        pf = scan_frame(detail, folder)
        return pf, (run_day.isoformat() if run_day else None)

    def _masked(pf, filters_raw: str):
        """Parse the filters param and mask the union frame (400 on bad)."""
        try:
            filters = analysis.parse_filters(filters_raw)
            mask = filter_mask(pf.frame, filters)
        except (analysis.BadParam, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return filters, mask

    def _y_columns(cols: list[str]) -> list[str]:
        requested = list(dict.fromkeys(c for c in cols if c))
        if len(requested) > _MAX_Y_COLUMNS:
            raise HTTPException(
                status_code=400,
                detail=f"at most {_MAX_Y_COLUMNS} y columns",
            )
        return requested

    def _display(display_raw: str) -> dict:
        """Parse the display param (400 on bad)."""
        try:
            return analysis.parse_display(display_raw)
        except analysis.BadParam as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _pretty_names(detail, pf, columns: list[str]) -> dict:
        """Figure titles/legend names, by the columns endpoint's rule.

        Run-provenance columns prettify; s-file names are already human.
        """
        scalar_headers = (detail.start_doc or {}).get("geecs_scalar_headers")
        return {
            column: (
                schema_map.display_name(column, scalar_headers)
                if pf.provenance.get(column, PROVENANCE_RUN) == PROVENANCE_RUN
                else column
            )
            for column in columns
        }

    @app.get("/health")
    def health() -> dict:
        """Liveness + catalog probe (the fleet-map health check)."""
        status = catalog.probe()
        return {"ok": status.ok, "catalog": status.label}

    # ------------------------- analysis JSON API -------------------------
    # One-liners over the data-utils primitives (03 design doc): every
    # response is reproducible in a notebook by the snippet it carries.

    @app.get("/api/run/{uid}/columns")
    def api_columns(uid: str, day: str = "") -> JSONResponse:
        """The union pick list: every plottable column with provenance."""
        detail = _load_run(uid)
        pf, _ = _union(detail, day)
        scalar_headers = (detail.start_doc or {}).get("geecs_scalar_headers")
        columns = [
            {
                "name": column,
                "provenance": pf.provenance.get(column, PROVENANCE_RUN),
                "pretty": (
                    schema_map.display_name(column, scalar_headers)
                    if pf.provenance.get(column, PROVENANCE_RUN) == PROVENANCE_RUN
                    else column
                ),
                # Drives the picker's off-by-default "timestamps" toggle
                # (ts_ event-recording times ONLY — acq_timestamp picks
                # stay always visible as legitimate X choices; the frame
                # endpoint's `kinds` map is the datetime-rendering
                # verdict and is deliberately broader).
                "timestamp": schema_map.is_key_timestamp_column(column),
            }
            for column in schema_map.plottable_columns(pf.frame)
        ]
        payload = {
            "columns": columns,
            "default_x": _default_x(detail, [c["name"] for c in columns]),
            "total": len(pf.frame),
        }
        return JSONResponse(payload, headers=_png_headers(detail))

    @app.get("/api/run/{uid}/frame")
    def api_frame(
        uid: str,
        cols: list[str] = Query(default=[]),
        x: str = "",
        filters: str = "",
        display: str = "",
        day: str = "",
    ) -> JSONResponse:
        """Per-shot series + the ready figure for the selection."""
        detail = _load_run(uid)
        pf, run_day = _union(detail, day)
        flt, mask = _masked(pf, filters)
        disp = _display(display)
        requested = _y_columns(cols)
        series = {}
        kinds = {}
        for column in dict.fromkeys([*requested, *([x] if x else [])]):
            # Coerce on the FULL frame: a filter that empties the frame
            # must not turn a valid column into a 404.
            full = schema_map.numeric_series(pf.frame, column)
            if full is None:
                raise HTTPException(
                    status_code=404, detail=f"no plottable column {column!r}"
                )
            epoch = schema_map.timestamp_epoch(column)
            if epoch:
                # Timestamps plot as real local datetimes, never raw
                # seconds (ts_ = Unix epoch; acq_timestamp = LabVIEW).
                series[column] = analysis.jsonable_datetimes(full[mask], epoch)
                kinds[column] = "datetime"
            else:
                series[column] = analysis.jsonable_values(full[mask])
        # The shot-axis rule (scan_event_index, NA-coalesced from the
        # s-file's Shotnumber) lives in figures.shot_axis_for_frame —
        # ONE implementation, shared with the notebook snippet's path.
        shot_values = analysis.jsonable_values(
            figures.shot_axis_for_frame(pf.frame)[mask]
        )
        # An unservable x already 404'd in the coercion loop above;
        # empty means "no X picked" — the shot axis, and the snippet
        # omits the x argument.
        x_name = x or None
        payload = {
            "series": series,
            "kinds": kinds,
            "shot": shot_values,
            "pass": int(mask.sum()),
            "total": len(pf.frame),
            "code": analysis.frame_code(
                uid,
                run_day,
                requested,
                flt,
                {column: schema_map.timestamp_epoch(column) for column in kinds},
                x=x_name,
                display=disp,
            ),
        }
        if requested:
            payload["figure"] = figures.shots_figure(
                series,
                requested,
                x=x_name,
                shot=shot_values,
                kinds=kinds,
                pretty=_pretty_names(
                    detail, pf, [*requested, *([x_name] if x_name else [])]
                ),
                display=disp,
            ).to_plotly_json()
        return JSONResponse(payload, headers=_png_headers(detail))

    @app.get("/api/run/{uid}/binned")
    def api_binned(
        uid: str,
        cols: list[str] = Query(default=[]),
        x: str = "",
        filters: str = "",
        bincfg: str = "",
        display: str = "",
        day: str = "",
    ) -> JSONResponse:
        """Per-bin centers + error bands + the ready binned figure.

        Bins GROUP the data (``bincfg.bin_col``); the selected ``x``
        PLACES it — each bin plots at the per-bin mean of the X column
        (the owner's ruling: real scan-parameter positions now, x error
        bars maybe later).  No ``x`` keeps the bin labels as the axis.
        """
        detail = _load_run(uid)
        pf, run_day = _union(detail, day)
        flt, mask = _masked(pf, filters)
        disp = _display(display)
        requested = _y_columns(cols)
        if x and schema_map.numeric_series(pf.frame, x) is None:
            raise HTTPException(status_code=404, detail=f"no plottable column {x!r}")
        try:
            cfg = analysis.parse_bincfg(bincfg)
        except analysis.BadParam as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        for column in requested:
            if schema_map.numeric_series(pf.frame, column) is None:
                raise HTTPException(
                    status_code=404, detail=f"no plottable column {column!r}"
                )
        cfg = dataclasses.replace(cfg, value_cols=tuple(requested))
        try:
            result = bin_frame(pf.frame[mask], cfg)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"no bin column: {exc}"
            ) from exc
        except ValueError as exc:  # e.g. degenerate percentile bounds
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TypeError as exc:
            # A coercible-string column (dtype-tolerant telemetry) plots
            # in per-shot view but bin_frame aggregates the RAW dtype —
            # refuse honestly, never 500 (package doctrine).
            raise HTTPException(
                status_code=400,
                detail=f"binned view needs numeric columns: {exc}",
            ) from exc
        x_centers = None
        if x:
            # Same primitive, mean-aggregated — the snippet mirrors this
            # exactly.  The x call's dropna/min_count runs over x ALONE,
            # so its surviving bins can differ from the y call's:
            # reindex onto the y bins, or points silently plot at the
            # wrong bin's x (a missing x center degrades to a null →
            # Plotly skips that point instead of mis-placing it).
            try:
                x_result = bin_frame(
                    pf.frame[mask],
                    dataclasses.replace(cfg, value_cols=(x,), agg="mean"),
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"binned view needs a numeric x: {exc}",
                ) from exc
            if (x, "center") in x_result.frame.columns:
                x_centers = analysis.jsonable_values(
                    x_result.frame[(x, "center")].reindex(result.frame.index)
                )
        bin_labels = analysis.jsonable_labels(result.frame.index)
        binned_series = {
            column: {
                sub: analysis.jsonable_values(result.frame[(column, sub)])
                for sub in ("center", "err_low", "err_high")
            }
            for column in requested
            if (column, "center") in result.frame.columns
        }
        x_name = x or None
        pretty = _pretty_names(detail, pf, [*requested, *([x_name] if x_name else [])])
        payload = {
            "bins": bin_labels,
            "counts": [int(count) for count in result.counts],
            "series": binned_series,
            "pass": int(mask.sum()),
            "total": len(pf.frame),
            "code": analysis.binned_code(
                uid, run_day, requested, flt, cfg, x=x_name, display=disp
            ),
        }
        if x_centers is not None:
            payload["x_centers"] = x_centers
        if requested:
            payload["figure"] = figures.binned_figure(
                bin_labels,
                binned_series,
                requested,
                bin_col=cfg.bin_col,
                x_values=x_centers,
                x_label=pretty.get(x_name) if x_name else None,
                pretty=pretty,
                display=disp,
            ).to_plotly_json()
        return JSONResponse(payload, headers=_png_headers(detail))

    @app.get("/api/run/{uid}/filter-count")
    def api_filter_count(uid: str, filters: str = "", day: str = "") -> dict:
        """Live pass count for the filters popup: ``{pass, total}``."""
        detail = _load_run(uid)
        pf, _ = _union(detail, day)
        _, mask = _masked(pf, filters)
        return {"pass": int(mask.sum()), "total": len(pf.frame)}

    def _bin_groups(pf, mask, bincfg_raw: str):
        """Per-bin shot membership over the filtered union frame.

        Same primitives and grouping semantics as ``/binned``
        (``compute_bin_key`` + ``groupby(dropna=False, observed=True,
        sort)``), so the two endpoints' bin orders agree — and the same
        code path serves both bin-images endpoints, so a ``bin`` INDEX
        is stable between the JSON listing and the PNG renders
        regardless of how labels serialize.

        Returns
        -------
        tuple of (BinningConfig, list of (label, list of int))
            The parsed config and, per bin in group order, the label
            and the sorted 1-based shot numbers (rows with no shot
            identity are dropped — no shot number, no image).
        """
        try:
            cfg = analysis.parse_bincfg(bincfg_raw)
        except analysis.BadParam as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        frame = pf.frame[mask]
        try:
            labels, _ = compute_bin_key(frame, cfg)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"no bin column: {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        shots = figures.shot_axis_for_frame(frame)
        groups = [
            (label, sorted({int(s) for s in group.dropna()}))
            for label, group in shots.groupby(labels, dropna=False, observed=True)
        ]
        return cfg, groups

    def _image_folder(detail, day: str, device: str):
        """Resolve + validate the (folder, device) pair for image endpoints."""
        run_day = _run_day(detail, day)
        folder = resolve_scan_folder(detail, run_day) if run_day else None
        if folder is None:
            raise HTTPException(status_code=404, detail="scan folder not resolvable")
        if device not in resources.image_devices(folder):
            raise HTTPException(status_code=404, detail=f"unknown device {device!r}")
        return folder, (run_day.isoformat() if run_day else None)

    @app.get("/api/run/{uid}/bin-images")
    def api_bin_images(
        uid: str, device: str = "", filters: str = "", bincfg: str = "", day: str = ""
    ) -> JSONResponse:
        """Per-bin membership for the Images tab's averaged grid.

        The JSON carries the numbers (bins, counts, member shots — all
        notebook-reproducible via the snippet); the pixels are served
        by ``/run/{uid}/bin-image.png?bin=<index>`` per bin, so the
        grid lazy-loads exactly like the per-shot gallery.
        """
        detail = _load_run(uid)
        if not device:
            raise HTTPException(status_code=400, detail="device is required")
        folder, run_day = _image_folder(detail, day, device)
        pf = scan_frame(detail, folder)
        flt, mask = _masked(pf, filters)
        cfg, groups = _bin_groups(pf, mask, bincfg)
        bin_labels = analysis.jsonable_labels([label for label, _ in groups])
        payload = {
            "device": device,
            "bin_col": cfg.bin_col,
            "bins": [
                {"bin": label, "count": len(shots), "shots": shots}
                for label, (_, shots) in zip(bin_labels, groups)
            ],
            "pass": int(mask.sum()),
            "total": len(pf.frame),
            "code": analysis.bin_images_code(uid, run_day, device, flt, cfg),
        }
        return JSONResponse(payload, headers=_png_headers(detail))

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
                "rows": [(run, fmt_time_of_day(run.start_time)) for run in runs],
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
        try:
            selected = date.fromisoformat(day)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="bad date") from exc
        carried = [
            (key, value)
            for key, value in request.query_params.multi_items()
            if key != "prefer"
        ]
        experiment = request.query_params.get("experiment", "") or default_experiment
        try:
            runs = catalog.list_runs(experiment, selected)
        except Exception as exc:  # noqa: BLE001 — degrade to the day page
            logger.warning("jump listing failed: %s", exc)
            runs = []
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
        target = next(
            (run for run in runs if prefer and run.scan_number == prefer),
            runs[0],  # newest (catalog order)
        )
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
    ) -> HTMLResponse:
        """One run: the rail + tabs (Overview / Plot / Images).

        ``tab``/``filters``/``bincfg``/``view``/``y``/``x`` are the
        analysis-tab state, carried in the URL (statelessness doctrine:
        a link IS the analysis) and consumed by the page's JS — the
        server only threads them through the sticky query so steppers
        keep the whole setup.
        """
        detail = _load_run(uid)
        run_day = _run_day(detail, day)
        folder = resolve_scan_folder(detail, run_day) if run_day else None
        devices = resources.image_devices(folder) if folder else []
        sel_device = device if device in devices else ""
        if sel_device:
            # Reuse the listing just computed — no second directory scan.
            probe = resources.device_kind(folder, sel_device, devices=devices)
            kind, kind_path = probe.kind, probe.path
        else:
            kind, kind_path = "", None
        n_rows = None if detail.data is None else len(detail.data)
        shot = max(1, min(shot, n_rows) if n_rows else shot)
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
                    data_cache=data_cache,
                    cache_key=warm_key,
                )

            data_cache.warm_native(
                warm_key, _warm_one, list(range(1, min(n_rows, 2000) + 1))
            )
        # The day's listing (newest first) feeds both the rail's scan
        # dropdown and the stepper neighbours.  A listing failure just
        # hides them — never sinks the page.
        prev_uid = next_uid = ""
        day_runs: list = []
        if run_day is not None:
            try:
                day_runs = list(
                    catalog.list_runs(experiment or default_experiment, run_day)
                )
                day_uids = [run.uid for run in day_runs]
                position = day_uids.index(uid)
                next_uid = day_uids[position - 1] if position > 0 else ""
                prev_uid = (
                    day_uids[position + 1] if position + 1 < len(day_uids) else ""
                )
            except Exception as exc:  # noqa: BLE001 — stepper is optional
                logger.warning("neighbour listing failed: %s", exc)
                day_runs = []
        state = {
            "day": day,
            "experiment": experiment or default_experiment,
            # The analysis-tab state (URL-carried; the page JS owns it):
            "tab": tab,
            "y": [c for c in y if c],
            "x": x,
            "view": view,
            "filters": filters,
            "bincfg": bincfg,
            "display": display,
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
                "run_day": run_day.isoformat() if run_day else "",
                "experiment": experiment or default_experiment,
                "summary": detail.summary,
                "rows": metadata_rows(detail),
                "start_time_of_day": fmt_time_of_day(detail.summary.start_time),
                "prev_uid": prev_uid,
                "next_uid": next_uid,
                "day_runs": day_runs,
                "scan_number": detail.summary.scan_number or 0,
                "prev_day": (
                    (run_day - timedelta(days=1)).isoformat() if run_day else ""
                ),
                "next_day": (
                    (run_day + timedelta(days=1)).isoformat() if run_day else ""
                ),
                "tab": tab if tab in ("overview", "plot", "images") else "plot",
                "devices": devices,
                "sel_device": sel_device,
                "kind": kind,
                "kind_path": str(kind_path) if kind_path else "",
                "shot": shot,
                "has_next_shot": n_rows is None or shot < n_rows,
                "total_shots": detail.summary.shots,
                "portal_version": _portal_version(),
                # The rail's chips and the display popup must stay in
                # step with the server-authored figures — one palette,
                # one marker default, both injected.
                "trace_colors": list(figures.TRACE_COLORS),
                "msize_default": figures.MARKER_SIZE_DEFAULT,
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
        complete = bool(detail.summary.exit_status)
        result = resources.load_shot_image(
            folder,
            device,
            shot,
            acq_timestamp=acq,
            data_cache=data_cache if complete else None,
            cache_key=(uid, device) if complete else None,
        )
        if result.png is None:
            raise HTTPException(status_code=404, detail=result.reason or result.kind)
        headers = (
            _png_headers(detail) if result.cacheable else {"Cache-Control": "no-cache"}
        )
        return Response(content=result.png, media_type="image/png", headers=headers)

    @app.get("/run/{uid}/bin-image.png")
    def run_bin_image(
        uid: str,
        device: str,
        bin_index: int = Query(default=0, alias="bin"),
        filters: str = "",
        bincfg: str = "",
        day: str = "",
    ) -> Response:
        """One bin's ``nanmean``-averaged device image, display-rendered.

        ``bin`` is the bin's INDEX in ``/api/.../bin-images`` order
        (same ``_bin_groups`` call, so the two always agree). Member
        shots that resolve to pixels are averaged (``average_frames``)
        and windowed once; shots the device missed (no timestamp) or
        that fail to load are skipped — the JSON's ``count`` is the
        membership, the pixels are what actually loaded.
        """
        detail = _load_run(uid)
        folder, _ = _image_folder(detail, day, device)
        pf = scan_frame(detail, folder)
        _, mask = _masked(pf, filters)
        _, groups = _bin_groups(pf, mask, bincfg)
        if not 0 <= bin_index < len(groups):
            raise HTTPException(
                status_code=404, detail=f"bin index {bin_index} of {len(groups)}"
            )
        _, shots = groups[bin_index]
        complete = bool(detail.summary.exit_status)
        n_rows = None if detail.data is None else len(detail.data)
        arrays: list = []
        cacheable = True
        for shot in shots:
            # Same refusals as the per-shot endpoint: never fall through
            # to an ordinal join beyond the recorded events (orphan
            # frames), never average a neighbour's image in.
            if n_rows is not None and shot > n_rows:
                continue
            acq, column_present = _acq_timestamp(detail, device, shot)
            if column_present and acq is None:
                continue  # device missed this shot
            resolved = resources.load_shot_array(
                folder,
                device,
                shot,
                acq_timestamp=acq,
                data_cache=data_cache if complete else None,
                cache_key=(uid, device) if complete else None,
            )
            if resolved.array is None:
                if resolved.kind in ("vendor", "unrenderable"):
                    # Device-level refusal — identical for every shot.
                    raise HTTPException(
                        status_code=404, detail=resolved.reason or resolved.kind
                    )
                continue
            cacheable = cacheable and resolved.cacheable
            arrays.append(resolved.array)
        averaged = average_frames(arrays, label=f"{device} bin {bin_index}")
        if averaged is None:
            raise HTTPException(
                status_code=404, detail="no renderable frames in this bin"
            )
        headers = _png_headers(detail) if cacheable else {"Cache-Control": "no-cache"}
        return Response(
            content=resources.to_display_png(averaged),
            media_type="image/png",
            headers=headers,
        )

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
