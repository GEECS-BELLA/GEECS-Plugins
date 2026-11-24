"""The scan's plot JSON API over the union frame: columns and binned."""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from geecs_data_utils import tiled_schema as schema_map
from geecs_data_utils.scan_frame import PROVENANCE_RUN
from geecs_data_utils.scan_grid import grid_axes

from geecs_portal import analysis, figures
from geecs_portal.routes.common import (
    _UNION_HEADERS,
    _display,
    _masked,
    _union,
)
from geecs_portal.state import PortalState, get_state

router = APIRouter()

#: Multi-Y ceiling on /binned (mockup ruling: up to 4).
_MAX_Y_COLUMNS = 4


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
        "default_x": analysis.default_x(detail.start_doc, [c["name"] for c in columns]),
        "grid_axes": grid_axes(list(pf.frame.columns), detail.start_doc),
        "total": len(pf.frame),
    }
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
    try:
        result, binned_series, x_centers = analysis.binned_series(
            pf.frame[mask], cfg, requested, x or None
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no bin column: {exc}") from exc
    except TypeError as exc:
        # A coercible-string column (dtype-tolerant telemetry) plots
        # in per-shot view but bin_frame aggregates the RAW dtype —
        # refuse honestly, never 500 (package doctrine).
        raise HTTPException(
            status_code=400,
            detail=f"binned view needs numeric columns: {exc}",
        ) from exc
    except ValueError as exc:  # e.g. degenerate percentile bounds
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    cfg = dataclasses.replace(cfg, value_cols=tuple(requested))
    bin_labels = analysis.jsonable_labels(result.frame.index)
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
