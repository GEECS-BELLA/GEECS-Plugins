"""The Plot and Grid tabs' JSON API over the union frame (+ bin membership)."""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from geecs_data_utils import tiled_schema as schema_map
from geecs_data_utils.data.binning import bin_frame
from geecs_data_utils.scan_frame import PROVENANCE_RUN, scan_frame
from geecs_data_utils.scan_grid import grid_axes, grid_scan

from geecs_portal import analysis, figures
from geecs_portal.routes.common import (
    _UNION_HEADERS,
    _bin_groups,
    _display,
    _image_folder,
    _masked,
    _union,
)
from geecs_portal.state import PortalState, get_state

router = APIRouter()

#: Multi-Y ceiling on the Plot tab (mockup ruling: up to 4).
_MAX_Y_COLUMNS = 4


def _default_x(detail, columns: list[str]) -> str:
    """The console-parity default X: the scan variable on stepped scans."""
    if not schema_map.is_stepped_scan(detail.start_doc):
        return ""
    scan_vars = schema_map.scan_variable_columns(columns, detail.start_doc)
    return scan_vars[0] if scan_vars else ""


def _y_columns(cols: list[str]) -> list[str]:
    requested = list(dict.fromkeys(c for c in cols if c))
    if len(requested) > _MAX_Y_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"at most {_MAX_Y_COLUMNS} y columns",
        )
    return requested


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


# ------------------------- analysis JSON API -------------------------
# One-liners over the data-utils primitives: every
# response is reproducible in a notebook by the snippet it carries.


@router.get("/api/run/{uid}/columns")
def api_columns(
    uid: str, day: str = "", portal: PortalState = Depends(get_state)
) -> JSONResponse:
    """The union pick list: every plottable column with provenance."""
    detail = portal.load_run(uid)
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
        "grid_axes": grid_axes(list(pf.frame.columns), detail.start_doc),
        "total": len(pf.frame),
    }
    return JSONResponse(payload, headers=_UNION_HEADERS)


@router.get("/api/run/{uid}/grid")
def api_grid(
    uid: str,
    gridcfg: str = "",
    filters: str = "",
    day: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """Two-axis geometry, filtered scalar statistics and paired figures."""
    detail = portal.load_run(uid)
    pf, run_day = _union(detail, day)
    try:
        cfg = analysis.parse_gridcfg(gridcfg)
        flt = analysis.parse_filters(filters)
        result = grid_scan(pf.frame, detail.start_doc, cfg, flt)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no grid column: {exc}") from exc
    except (ValueError, TypeError, IndexError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pretty = _pretty_names(detail, pf, [result.config.x, result.config.y, cfg.value])
    payload = {
        "cells": result.cells.to_dict("records"),
        "config": result.config.model_dump(),
        "kind": result.kind,
        "x_values": result.x_values,
        "y_values": result.y_values,
        "visits": result.visits,
        "pass": result.passing,
        "total": result.total,
        "bin_column": result.bin_column,
        "notes": result.notes,
        "error_label": figures.grid_error_label(result),
        "pretty": pretty,
        "figures": {
            name: figures.page_figure(fig)
            for name, fig in figures.grid_figures(
                result, pretty=pretty, palette=figures.THEMED_PALETTE
            ).items()
        },
        "code": analysis.grid_code(uid, run_day, result.config, flt, pretty),
    }
    return JSONResponse(analysis.jsonable_document(payload), headers=_UNION_HEADERS)


@router.get("/api/run/{uid}/frame")
def api_frame(
    uid: str,
    cols: list[str] = Query(default=[]),
    x: str = "",
    filters: str = "",
    display: str = "",
    day: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """Per-shot series + the ready figure for the selection."""
    detail = portal.load_run(uid)
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
    # s-file's Shotnumber) lives in tiled_schema.shot_axis_for_frame —
    # ONE implementation, shared with the notebook snippet's path.
    shot_values = analysis.jsonable_values(figures.shot_axis_for_frame(pf.frame)[mask])
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
        payload["figure"] = figures.page_figure(
            figures.shots_figure(
                series,
                requested,
                palette=figures.THEMED_PALETTE,
                x=x_name,
                shot=shot_values,
                kinds=kinds,
                pretty=_pretty_names(
                    detail, pf, [*requested, *([x_name] if x_name else [])]
                ),
                display=disp,
            )
        )
    return JSONResponse(payload, headers=_UNION_HEADERS)


@router.get("/api/run/{uid}/binned")
def api_binned(
    uid: str,
    cols: list[str] = Query(default=[]),
    x: str = "",
    filters: str = "",
    bincfg: str = "",
    display: str = "",
    day: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """Per-bin centers + error bands + the ready binned figure.

    Bins GROUP the data (``bincfg.bin_col``); the selected ``x``
    PLACES it — each bin plots at the per-bin mean of the X column
    (the owner's ruling: real scan-parameter positions now, x error
    bars maybe later).  No ``x`` keeps the bin labels as the axis.
    """
    detail = portal.load_run(uid)
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
        raise HTTPException(status_code=404, detail=f"no bin column: {exc}") from exc
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
        payload["figure"] = figures.page_figure(
            figures.binned_figure(
                bin_labels,
                binned_series,
                requested,
                palette=figures.THEMED_PALETTE,
                bin_col=cfg.bin_col,
                x_values=x_centers,
                x_label=pretty.get(x_name) if x_name else None,
                pretty=pretty,
                display=disp,
            )
        )
    return JSONResponse(payload, headers=_UNION_HEADERS)


@router.get("/api/run/{uid}/filter-count")
def api_filter_count(
    uid: str, filters: str = "", day: str = "", portal: PortalState = Depends(get_state)
) -> JSONResponse:
    """Live pass count for the filters popup: ``{pass, total}``."""
    detail = portal.load_run(uid)
    pf, _ = _union(detail, day)
    _, mask = _masked(pf, filters)
    payload = {"pass": int(mask.sum()), "total": len(pf.frame)}
    return JSONResponse(payload, headers=_UNION_HEADERS)


@router.get("/api/run/{uid}/bin-images")
def api_bin_images(
    uid: str,
    device: str = "",
    filters: str = "",
    bincfg: str = "",
    day: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """Per-bin membership for the Images tab's averaged grid.

    The JSON carries the numbers (bins, counts, member shots — all
    notebook-reproducible via the snippet); the pixels are served
    by ``/run/{uid}/bin-image.png?bin=<index>`` per bin, so the
    grid lazy-loads exactly like the per-shot gallery.
    """
    detail = portal.load_run(uid)
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
    return JSONResponse(payload, headers=_UNION_HEADERS)
