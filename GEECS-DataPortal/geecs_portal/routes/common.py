"""Request helpers the portal's routers share (no app state)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional
from urllib.parse import urlencode

from fastapi import HTTPException
from starlette.requests import Request

from geecs_data_utils import tiled_schema as schema_map
from geecs_data_utils.data.binning import compute_bin_key
from geecs_data_utils.data.row_filters import filter_mask
from geecs_data_utils.scan_frame import scan_frame
from geecs_data_utils.tiled_catalog import (
    RunSummary,
    fmt_time_of_day,
    resolve_scan_folder,
)

from geecs_portal import analysis, figures, resources


def _portal_version() -> str:
    """The installed package version — the /api cache-bust key.

    The page keys every /api fetch and bin-image card by this version.
    Union-frame responses are ``no-cache`` now (``_UNION_HEADERS``), so
    for them the key is only insurance against intermediaries; it still
    matters for a payload-shape change behind any cached response.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("geecs-data-portal")
    except PackageNotFoundError:  # source tree without install metadata
        return "dev"


#: Headers for every response computed over the union frame (columns,
#: frame, binned, bin-images, bin-image.png). The event table is frozen
#: once the run stops, but the s-file half of the union is NOT: ScanAnalysis
#: appends its columns to ``analysis/sN.txt`` after the scan — hours later
#: when it is re-run by hand — so a completed run's column list can grow.
#: An immutable response would pin a browser to the pre-analysis shape
#: for a year (the 7-vs-30-columns incident, 2026-09-01). No validator
#: is emitted, so ``no-cache`` means a re-fetch, not a 304 — an ETag
#: over the s-file's stat is the follow-up if revisits feel slow.
_UNION_HEADERS = {"Cache-Control": "no-cache"}

#: Headers for the JSON browsing surface (day listing, run detail, device
#: probe, jump). A day gains scans while it is being taken, a running
#: run gains its stop document, and a device folder fills in as the run
#: writes — always re-fetch, never pin.
_LISTING_HEADERS = {"Cache-Control": "no-cache"}


def _root(request: Request) -> str:
    """The request's URL prefix (``""`` at root) — prepend to every path."""
    return request.scope.get("root_path", "").rstrip("/")


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
        missed this shot; or the device's ``valid`` companion false: its
        frame belongs to a different physical shot, the row a scan run
        maps no file for) → ``(None, True)`` — the caller must refuse
        rather than serve a neighbouring shot's image.  The ``valid``
        column is matched by
        :func:`geecs_data_utils.tiled_schema.device_valid_column`.
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
    valid = schema_map.device_valid_column([str(c) for c in frame.columns], device)
    if valid is not None:
        try:
            if not bool(frame[valid].iloc[shot - 1]):
                return (None, True)
        except (TypeError, ValueError):  # pd.NA: unknown, not false
            pass
    try:
        value = float(frame[column].iloc[shot - 1])
    except (TypeError, ValueError):
        return (None, True)
    if not math.isfinite(value) or value <= 0:
        return (None, True)
    return (value, True)


def _scan_label(summary: RunSummary) -> str:
    """The listing's scan cell: ``Scan 002``, else the uid's first 8 chars."""
    if summary.scan_number is not None:
        return f"Scan {summary.scan_number:03d}"
    return summary.uid[:8]


def _summary_json(summary: RunSummary) -> dict:
    """One run's listing row, as the day page's table shows it.

    The JSON twin of a ``day.html`` row (scan cell, HH:MM, mode,
    description, shots, status) plus the fields the page keeps in
    attributes or omits (uid, epoch start, experiment, save sets).
    """
    started = None
    if summary.start_time > 0:
        started = analysis.jsonable_datetimes([summary.start_time], "unix")[0]
    return {
        "uid": summary.uid,
        "scan": _scan_label(summary),
        "scan_number": summary.scan_number,
        "start_time": summary.start_time,
        "started": started,
        "time": fmt_time_of_day(summary.start_time),
        "mode": summary.mode,
        "description": summary.description,
        "shots": summary.shots,
        "exit_status": summary.exit_status,
        "running": not summary.exit_status,
        "experiment": summary.experiment,
        "save_sets": list(summary.save_sets),
    }


def _parse_iso_day(day: str) -> date:
    """An ISO day path segment, or the routes' shared 404."""
    try:
        return date.fromisoformat(day)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="bad date") from exc


def _jump_target(runs: list[RunSummary], prefer: int) -> Optional[RunSummary]:
    """The day steppers' rule — the run numbered ``prefer``, else the newest.

    ONE implementation for the HTML redirect and the JSON twin, so the
    two cannot drift (``None`` only for a day with no runs).
    """
    return next(
        (run for run in runs if prefer and run.scan_number == prefer),
        runs[0] if runs else None,
    )


def _png_headers(detail) -> dict:
    """Caching headers for responses over the event table ALONE.

    A completed run (stop doc present) never changes, so its plot
    and per-shot images are cacheable indefinitely; a still-running
    run must revalidate. Anything touching the union frame (and so
    the mutable s-file) uses ``_UNION_HEADERS`` instead.
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


def _display(display_raw: str) -> dict:
    """Parse the display param (400 on bad)."""
    try:
        return analysis.parse_display(display_raw)
    except analysis.BadParam as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _render_opts(disp: dict) -> dict:
    """The image-rendering slice of the display state (value-degrade)."""
    return {
        "cmap": disp.get("cmap"),
        "plo": disp.get("plo"),
        "phi": disp.get("phi"),
    }


def _rendered(disp: dict) -> bool:
    """``display.mode == "rendered"`` — the analyzer-figure view."""
    return disp.get("mode") == "rendered"


def _figure_kwargs(render: dict) -> dict:
    """The display slice as the ephemeral renderers' kwargs (value-degrade).

    The colormap defaults to ``gray`` — the pixel view's palette when
    ``cmap`` is absent or unknown — so the rendered checkbox changes
    only what it claims to (the base renderer's own default is
    plasma).
    """
    return {
        "cmap": resources.safe_cmap(render.get("cmap")) or "gray",
        "window": resources.window_percentiles(render.get("plo"), render.get("phi")),
    }


def _resolved_folder(detail, day: str):
    """``(run_day, folder)``: the run's own day and its existing scan folder."""
    run_day = _run_day(detail, day)
    folder = resolve_scan_folder(detail, run_day) if run_day else None
    return run_day, folder


def _bin_groups(pf, mask, bincfg_raw: str):
    """Per-bin shot membership over the filtered union frame.

    Same primitives and grouping semantics as ``/binned``
    (``compute_bin_key`` + ``groupby(dropna=False, observed=True,
    sort)``, ``min_count`` applied to per-bin ROW counts exactly as
    ``bin_frame`` does), so the two tabs' bins agree under one
    shared ``bincfg`` — and the same code path serves both
    bin-images endpoints, so a ``bin`` INDEX is stable between the
    JSON listing and the PNG renders regardless of how labels
    serialize.

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
        raise HTTPException(status_code=404, detail=f"no bin column: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    shots = figures.shot_axis_for_frame(frame)
    groups = [
        (label, sorted({int(s) for s in group.dropna()}))
        for label, group in shots.groupby(labels, dropna=False, observed=True)
        # min_count mirrors bin_frame (row counts, not shot counts):
        # the binset popup's "min shots / bin" must govern this grid
        # exactly as it governs the Plot tab's binned view.
        if cfg.min_count <= 1 or len(group) >= cfg.min_count
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
