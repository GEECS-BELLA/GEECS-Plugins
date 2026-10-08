"""The analysis config editor's mount and its live preview over a scan."""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from geecs_portal import resources
from geecs_portal.routes.common import _acq_timestamp, _image_folder
from geecs_portal.state import PortalState

logger = logging.getLogger(__name__)


# ---- the analysis config editor (deferred to its own arc) ----
# Mounted at /configs over the processing tree. The store writes only
# into that tree (never the scans tree); the live preview renders the
# UNSAVED document on the current shot through the same write-free
# ephemeral seam the Images tab uses, so dialling in an ROI is a
# type-and-look loop without a save per iteration.
def _preview_scan(portal: PortalState, uid: str, device: str, day: str):
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


def _camera_frame(
    portal: PortalState, detail, folder, uid: str, device: str, shot: int
):
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


def _line_preview(
    portal: PortalState, diag, uid: str, device: str, day: str, shot: int
) -> bytes:
    """A LINE document's preview: the shot's trace drawn as the run draws it."""
    ephemeral = portal.ephemeral_module()
    detail, folder = _preview_scan(portal, uid, device, day)
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
    portal: PortalState, document: dict, params: dict, index: int
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
    detail, folder = _preview_scan(portal, uid, device, day)
    arrays, positions, missing = [], [], []
    for shot in range(1, shots + 1):
        try:
            if is_line(diag):
                array, _aux = _line_trace(diag, detail, folder, device, shot)
            else:
                array = _camera_frame(portal, detail, folder, uid, device, shot)
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


def _config_editor_preview(portal: PortalState, document: dict, params: dict) -> bytes:
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
        return _line_preview(portal, diag, uid, device, day, shot)
    detail, folder = _preview_scan(portal, uid, device, day)
    array = _camera_frame(portal, detail, folder, uid, device, shot)
    # the recipe's frame inputs load from ITS device folder under this scan,
    # exactly as the run loads them (a background image under {scan_dir})
    figures = ephemeral.render_document_as_run(diag, [array], scan_folder=folder)
    if not figures:
        raise ValueError("this analyzer draws no figure for a single frame")
    return resources.figure_png(figures[0], tight=True)


def mount_config_editor(app: FastAPI, portal: PortalState) -> None:
    """Mount ``scan_analysis.config_editor`` at ``/configs`` over the processing tree."""
    try:
        from scan_analysis.config_editor import create_editor_router
        from scan_analysis.config_store import ConfigStore
    except ImportError as exc:  # the analysis extra without the editor extra
        logger.warning("config editor requested but not installed: %s", exc)
    else:
        app.include_router(
            create_editor_router(
                ConfigStore(Path(portal.processing_config_dir)),
                preview=functools.partial(_config_editor_preview, portal),
                summary_preview=functools.partial(
                    _config_editor_summary_preview, portal
                ),
                summary_shots_max=_SUMMARY_SHOTS_MAX,
                theme_url="/theme",
            ),
            prefix="/configs",
        )
        portal.config_editor_enabled = True
