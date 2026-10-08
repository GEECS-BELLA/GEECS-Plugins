"""Rendered resources: per-shot and per-bin images, traces, the scalar PNG."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from matplotlib.figure import Figure

from geecs_data_utils import tiled_schema as schema_map
from geecs_data_utils.io.images import average_frames
from geecs_data_utils.scan_frame import scan_frame

from geecs_portal import figures, resources
from geecs_portal.routes.common import (
    _UNION_HEADERS,
    _acq_timestamp,
    _bin_groups,
    _display,
    _image_folder,
    _masked,
    _png_headers,
    _render_opts,
    _rendered,
)
from geecs_portal.state import PortalState, get_state

router = APIRouter()

#: Cap on rows fed to a plot (quick-look, not a data browser).
_PLOT_MAX_ROWS = 100_000


@router.get("/api/run/{uid}/trace")
def api_run_trace(
    uid: str,
    device: str,
    shot: int = 1,
    day: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """One shot of an ARRAY capture stack as a server-authored figure.

    The line twin of ``/run/{uid}/image.png``.  A camera shot is
    served as a rendered PNG because a 2048² frame as JSON is
    absurd; a scope trace is a few thousand numbers and wants the
    hover readout, so it travels as figure JSON like every other
    plot here.  Same refusals as the image endpoint: a shot beyond
    the recorded events, and a device that missed the shot, both
    404 rather than serving a neighbour's trace.
    """
    detail = portal.load_run(uid)
    folder, _ = _image_folder(detail, day, device)
    if detail.data is not None and shot > len(detail.data):
        raise HTTPException(
            status_code=404, detail="shot beyond the run's recorded events"
        )
    acq, column_present = _acq_timestamp(detail, device, shot)
    if column_present and acq is None:
        raise HTTPException(
            status_code=404, detail="device missed this shot (no timestamp)"
        )
    resolved = resources.load_shot_trace(folder, device, shot, acq_timestamp=acq)
    if resolved.result is None:
        raise HTTPException(status_code=404, detail=resolved.reason or resolved.kind)
    trace = resolved.result
    x_title = trace.x_label or ""
    if x_title and trace.x_units:
        x_title = f"{x_title} ({trace.x_units})"
    figure = figures.trace_figure(
        trace.data[:, 0],
        trace.data[:, 1],
        name=device,
        x_title=x_title,
        # The stack names the VARIABLE it captured but not its units
        # (those ride in the analyzer config), so the axis is titled
        # with the quantity and claims no unit it was not told.
        y_title=trace.y_label or "",
        palette=figures.THEMED_PALETTE,
    )
    # A running scan's stack grows, so a trace is as mutable as the
    # union frame — the same no-cache headers every other /api
    # response here carries.
    return JSONResponse(
        {
            "figure": figures.page_figure(figure),
            "content": resolved.content,
            "points": int(trace.data.shape[0]),
        },
        headers=_UNION_HEADERS,
    )


@router.get("/run/{uid}/image.png")
def run_image(
    uid: str,
    device: str,
    shot: int = 1,
    day: str = "",
    processing: str = "",
    display: str = "",
    portal: PortalState = Depends(get_state),
) -> Response:
    """One device shot rendered for display (stack or native file).

    ``processing`` names a diagnostic to run ephemerally on the
    loaded pixels first (its ``processed_image`` renders instead of
    the raw frame) — the write-free seam; raw serving is untouched
    when the param is absent. ``display`` carries the image
    cosmetics (``cmap`` + ``plo``/``phi`` window — types 400,
    values degrade, per the display doctrine).
    """
    detail = portal.load_run(uid)
    disp = _display(display)
    render = _render_opts(disp)
    folder, _ = _image_folder(detail, day, device)
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
    if processing:
        resolved = resources.load_shot_array(
            folder,
            device,
            shot,
            acq_timestamp=acq,
            data_cache=portal.data_cache if complete else None,
            cache_key=(uid, device) if complete else None,
        )
        if resolved.array is None:
            raise HTTPException(
                status_code=404, detail=resolved.reason or resolved.kind
            )
        if _rendered(disp):
            # The analyzer's own figure (overlays + axes + colorbar)
            # instead of the windowed processed pixels.
            png = portal.render_processing_figure(resolved.array, processing, render)
        else:
            (processed,) = portal.apply_processing([resolved.array], processing)
            try:
                png = resources.to_display_png(processed, **render)
            except Exception as exc:
                raise HTTPException(
                    status_code=404, detail=f"render failed: {exc}"
                ) from exc
        # A processed response is a function of (pixels, diagnostic
        # YAML, ImageAnalysis version); the URL keys only the first,
        # and the configs tree is local and MUTABLE — iterating on
        # it is the selector's purpose. Never immutable-cache what
        # a config edit must be able to change.
        return Response(
            content=png,
            media_type="image/png",
            headers={"Cache-Control": "no-cache"},
        )
    result = resources.load_shot_image(
        folder,
        device,
        shot,
        acq_timestamp=acq,
        data_cache=portal.data_cache if complete else None,
        cache_key=(uid, device) if complete else None,
        **render,
    )
    if result.png is None:
        raise HTTPException(status_code=404, detail=result.reason or result.kind)
    headers = (
        _png_headers(detail) if result.cacheable else {"Cache-Control": "no-cache"}
    )
    return Response(content=result.png, media_type="image/png", headers=headers)


@router.get("/run/{uid}/bin-image.png")
def run_bin_image(  # noqa: C901
    uid: str,
    device: str,
    bin_index: int = Query(default=0, alias="bin"),
    filters: str = "",
    bincfg: str = "",
    day: str = "",
    processing: str = "",
    display: str = "",
    portal: PortalState = Depends(get_state),
) -> Response:
    """One bin's ``nanmean``-averaged device image, display-rendered.

    ``bin`` is the bin's INDEX in ``/api/.../bin-images`` order
    (same ``_bin_groups`` call, so the two always agree). Member
    shots that resolve to pixels are averaged (``average_frames``)
    and windowed once; shots the device missed (no timestamp) or
    that fail to load are skipped — the JSON's ``count`` is the
    membership, the pixels are what actually loaded. With
    ``processing``, each member is ephemeral-processed FIRST and
    the processed images average (process-then-average — the
    correct order for nonlinear pipeline steps like thresholding).
    """
    detail = portal.load_run(uid)
    disp = _display(display)
    render = _render_opts(disp)
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
            data_cache=portal.data_cache if complete else None,
            cache_key=(uid, device) if complete else None,
        )
        if resolved.array is None:
            if resolved.kind in ("vendor", "unrenderable"):
                # Device-level refusal — identical for every shot.
                raise HTTPException(
                    status_code=404, detail=resolved.reason or resolved.kind
                )
            continue
        arrays.append(resolved.array)
    if processing and arrays:
        arrays = portal.apply_processing(arrays, processing)
    averaged = average_frames(arrays, label=f"{device} bin {bin_index}")
    if averaged is None:
        raise HTTPException(status_code=404, detail="no renderable frames in this bin")
    if processing and _rendered(disp):
        # An average of several results is not one result: the base
        # renderer (axes + colorbar), per-shot overlays dropped.
        png = portal.render_frame_figure(averaged, render)
    else:
        try:
            png = resources.to_display_png(averaged, **render)
        except Exception as exc:
            raise HTTPException(
                status_code=404, detail=f"render failed: {exc}"
            ) from exc
    # Never immutable: bin membership comes off the union frame
    # (mutable s-file), and the diagnostic YAML behind ``processing``
    # is likewise a mutable input the URL does not key (see run_image).
    return Response(content=png, media_type="image/png", headers=_UNION_HEADERS)


@router.get("/run/{uid}/plot.png")
def run_plot(
    uid: str, y: str, x: str = "", portal: PortalState = Depends(get_state)
) -> Response:
    """Server-rendered scalar plot: *y* column vs *x* (default row index).

    Uses the matplotlib object API (``Figure``, never pyplot) — no
    global figure registry, safe on FastAPI's threadpool.
    """
    detail = portal.load_run(uid)
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
            raise HTTPException(status_code=404, detail=f"no plottable column {x!r}")
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
    return Response(
        content=resources.figure_png(fig),
        media_type="image/png",
        headers=_png_headers(detail),
    )
