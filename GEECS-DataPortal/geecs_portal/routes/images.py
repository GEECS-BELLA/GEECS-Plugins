"""Rendered resources: the per-shot image and the scalar PNG."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from matplotlib.figure import Figure

from geecs_data_utils import tiled_schema as schema_map
from geecs_data_utils.io import shot_images
from geecs_data_utils.io.display_png import to_display_png

from geecs_portal import resources
from geecs_portal.routes.common import (
    _acq_timestamp,
    _display,
    _file_shot,
    _image_folder,
    _png_headers,
    _render_opts,
    _rendered,
)
from geecs_portal.state import PortalState, get_state

router = APIRouter()

#: Cap on rows fed to a plot (quick-look, not a data browser).
_PLOT_MAX_ROWS = 100_000


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
    acq, column_present = _acq_timestamp(detail, device, shot, folder)
    if column_present and acq is None:
        raise HTTPException(
            status_code=404, detail="device missed this shot (no timestamp)"
        )
    complete = bool(detail.summary.exit_status)
    if processing:
        resolved = shot_images.load_shot_array(
            folder,
            device,
            shot,
            acq_timestamp=acq,
            data_cache=portal.data_cache if complete else None,
            cache_key=(uid, device) if complete else None,
            file_shot=_file_shot(detail, shot, folder),
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
                png = to_display_png(processed, **render)
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
    result = shot_images.load_shot_image(
        folder,
        device,
        shot,
        acq_timestamp=acq,
        data_cache=portal.data_cache if complete else None,
        cache_key=(uid, device) if complete else None,
        file_shot=_file_shot(detail, shot, folder),
        **render,
    )
    if result.png is None:
        raise HTTPException(status_code=404, detail=result.reason or result.kind)
    headers = (
        _png_headers(detail) if result.cacheable else {"Cache-Control": "no-cache"}
    )
    return Response(content=result.png, media_type="image/png", headers=headers)


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
    frame = detail.scalars().head(_PLOT_MAX_ROWS)
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
