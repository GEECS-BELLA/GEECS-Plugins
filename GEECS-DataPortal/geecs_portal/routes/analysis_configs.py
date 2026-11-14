"""The analysis documents as their consumer checks them: schema, validate and preview.

The configurator owns the analysis documents (recipes, ``.himg`` jobs,
groups) and their saves; this portal runs them, so it answers for them.
Its dry run is ``POST /api/analysis/validate/{kind}`` (a draft, never
written: the schema, the binding to GEECS-Analysis's registry, a stem the
loader would refuse, a group's references) and its form is
``GET /api/analysis/schema/{kind}`` (a recipe's ``steps`` and ``measure``
bound to the registry). The run page's embedded editor previews an unsaved
recipe on this scan through ``POST /api/analysis/preview`` and
``/preview/summary``: the run's own draw, read-only, never a write.

Every route is the processing feature's: off without ``--processing-configs``
or the ``analysis`` extra (404). The kinds are GEECS-Schemas' registry
names, the configurator's.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import JSONResponse, Response

from geecs_portal import resources
from geecs_portal.routes.common import _acq_timestamp, _file_shot, _image_folder
from geecs_portal.state import PortalState, get_state

logger = logging.getLogger(__name__)
router = APIRouter()

#: A draft as JSON: validated by its kind's model, never trusted here.
Json = dict[str, Any]  # Any: the draft before validation


def _kinds() -> dict[str, Any]:  # Any: the kind's storage row (a FileStorage)
    """The analysis kinds this portal checks: every file kind GEECS-Schemas keeps in the analysis root.

    Imported here, not at module load: GEECS-Schemas comes with the
    ``analysis`` extra, which the route's gate has already checked.
    """
    from geecs_schemas import KIND_STORAGE, FileStorage

    return {k: s for k, s in KIND_STORAGE.items() if isinstance(s, FileStorage) and s.root == "analysis"}  # fmt: skip


def _kind(kind: str) -> str:
    kinds = _kinds()
    if kind not in kinds:
        raise HTTPException(404, f"no analysis kind {kind!r} (one of {', '.join(kinds)})")  # fmt: skip
    return kind


def _model(kind: str) -> Any:  # Any: the kind's pydantic model class
    from geecs_schemas import SCHEMA_REGISTRY

    return SCHEMA_REGISTRY[kind]


def _root(portal: PortalState) -> Path:
    """The configs tree, once the feature's gate passed (configured, installed)."""
    portal.ephemeral_module()
    return Path(portal.processing_config_dir)


def _doc_path(doc_id: str) -> PurePosixPath:
    """The document's id below its kind's folder (the configurator's id); never a path out of it."""
    path = PurePosixPath(doc_id)
    if (
        path.is_absolute()
        or not path.parts
        or any(p in ("", ".", "..") for p in path.parts)
    ):
        raise HTTPException(422, f"id {doc_id!r} is not a path below the kind's folder")
    return path


def _stem_errors(root: Path, doc_id: str | None) -> list[Json]:
    """A stem the loader would refuse: another document of the analyzers tree already has it.

    Documents are run by stem (``discover_diagnostics``), across every
    facility folder, so a second file with the same stem makes every run
    by name fail; a tree already holding one is refused with the loader's
    own message.
    """
    from geecs_data_utils.analysis_configs import discover_diagnostics
    from geecs_schemas.analysis.document import ANALYZERS_FOLDER

    if doc_id is None:
        return []
    path = _doc_path(doc_id)
    try:
        index = discover_diagnostics(root)
    except FileNotFoundError:
        return []
    except ValueError as exc:
        return [{"loc": [], "msg": str(exc)}]
    here = (root / ANALYZERS_FOLDER / path).resolve()
    held = index.get(path.stem)
    if held is not None and held.resolve() != here:
        where = held.relative_to(root / ANALYZERS_FOLDER).as_posix()
        return [{"loc": [], "msg": f"the stem {path.stem!r} is already {where}: diagnostic stems are unique across the whole analyzers tree"}]  # fmt: skip
    return []


def _ref_errors(root: Path, group: Any) -> list[Json]:  # Any: an AnalysisGroup
    """Why a run could not load the group: the loader's own resolution (``resolve_group``), never a copy of it.

    It refuses a reference no diagnostic answers to, a reference given
    twice, and a referenced diagnostic that does not load; the error is
    placed on the reference it names, else on the list.
    """
    from scan_analysis.config.analysis_group_loader import discover_analyzers, resolve_group  # fmt: skip

    try:
        resolve_group(group, discover_analyzers(root))
    except FileNotFoundError:  # no analyzers/ folder: every reference is unknown
        return [{"loc": ["analyzers", i, "ref"], "msg": f"unknown diagnostic {ref.ref!r}"} for i, ref in enumerate(group.analyzers)]  # fmt: skip
    except (ValueError, KeyError, OSError) as exc:
        msg = str(exc).split(". Known IDs:")[0]
        about = msg.removeprefix(
            f"Group '{group.name}'"
        )  # the group's own name is quoted too
        named = [i for i, ref in enumerate(group.analyzers) if f"'{ref.ref}'" in about]
        return [{"loc": ["analyzers", named[-1], "ref"] if named else ["analyzers"], "msg": msg}]  # fmt: skip
    return []


def validate_document(root: Path, kind: str, document: Json, doc_id: str | None = None) -> Json:  # fmt: skip
    """The consumer's check of a draft: ``{"ok", "errors": [{"loc": [...], "msg"}]}``; nothing is written."""
    from geecs_schemas.analysis import AnalysisGroup, AnalysisRecipe
    from geecs_schemas.analysis.document import ANALYZERS_FOLDER
    from pydantic import ValidationError

    try:
        model = _model(kind).model_validate(document)
    except ValidationError as exc:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return {"ok": False, "errors": errors}
    errors = []
    if isinstance(model, AnalysisRecipe):
        from geecs_analysis.recipe import binding_errors

        errors += binding_errors(model)
    if isinstance(model, AnalysisGroup):
        errors += _ref_errors(root, model)
    if _kinds()[kind].folder == ANALYZERS_FOLDER:  # run by stem: unique across the tree
        errors += _stem_errors(root, doc_id)
    return {"ok": not errors, "errors": errors}


@router.get("/api/analysis/blank/{kind}")
def analysis_blank(
    kind: str,
    name: str,
    input: str = "",
    device: str = "",
    fields: str = "",
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """A new document of ``kind`` that already validates: the starting draft an editor opens, never written here.

    A recipe is ``geecs_analysis.recipe.blank_recipe``: the ``input`` kind
    (its frames), no steps, the registry's default measure for those frames,
    ``output_name`` the name; ``fields`` (a JSON object) is what the input
    kind requires beyond its kind, e.g. a trace's ``{"loading": {"data_type":
    "tsv"}}``. A group is an empty list of analyzers. A ``.himg`` job
    has no blank: its kind is the decision (one of them deletes files).
    """
    _root(portal)
    _kind(kind)
    if kind == "analysis_group":
        from pydantic import ValidationError

        document: Json = {"name": name, "analyzers": []}
        try:
            _model(kind).model_validate(document)
        except ValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        return JSONResponse({"document": document})
    if kind != "analysis_recipe":
        raise HTTPException(404, f"no blank {kind}: its kind is the operator's decision (one of them deletes files)")  # fmt: skip
    from geecs_analysis.recipe import RecipeError, blank_recipe

    try:
        input_fields = json.loads(fields) if fields else {}
    except ValueError as exc:
        raise HTTPException(422, f"fields is not JSON: {exc}") from exc
    if not isinstance(input_fields, dict):
        raise HTTPException(422, "fields is a JSON object (the input's fields beyond its kind)")  # fmt: skip
    try:
        return JSONResponse({"document": blank_recipe(input, name=name, device=device or None, input_fields=input_fields)})  # fmt: skip
    except RecipeError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/api/analysis/schema/{kind}")
def analysis_schema(kind: str, portal: PortalState = Depends(get_state)) -> JSONResponse:  # fmt: skip
    """The kind's JSON Schema: a recipe's ``steps`` and ``measure`` bound to the analysis core's registry."""
    _root(portal)
    if _kind(kind) == "analysis_recipe":
        from geecs_analysis.recipe import recipe_schema

        return JSONResponse(recipe_schema())
    return JSONResponse(_model(kind).model_json_schema())


@router.post("/api/analysis/validate/{kind}")
def analysis_validate(
    kind: str,
    document: Json = Body(embed=True),
    id: str | None = Body(None, embed=True),
    portal: PortalState = Depends(get_state),
) -> JSONResponse:
    """Validate a draft as this portal's runs would load it; ``id`` (below the kind's folder) adds the stem rule."""
    root = _root(portal)
    return JSONResponse(validate_document(root, _kind(kind), document, id))


# ---- the previews: an unsaved recipe drawn on this scan, as a run draws it.
# Through the same write-free ephemeral seam the Images tab uses, so dialling
# in an ROI is a type-and-look loop without a save per iteration.


def _preview_scan(portal: PortalState, uid: str, device: str, day: str):
    """The run + its scan folder for a preview, with the preview's error ladder."""
    try:
        detail = portal.load_run(uid)
        folder, _ = _image_folder(detail, day, device)
    except HTTPException as exc:
        kind = LookupError if exc.status_code == 404 else ValueError
        raise kind(str(exc.detail)) from exc
    return detail, folder


def _line_trace(diag, detail, folder, device: str, shot: int):
    """One shot's trace as a scan run reads it (an ``N x 2`` array, or wider).

    The shot resolves through the run path's own source rules
    (:func:`scan_analysis.core_source.prepare_source` — file tail,
    ``data_format``, the stack-only rule for ``pva_stack``) over
    ``device``, the recipe's own data folder; the reference is read with
    the document's loading. A stitched input (``sibling_folders``) is the
    source's own joined trace, as the run reads it, and a recipe whose
    measure reads auxiliary channels (``frog_spectral_phase``'s weights)
    gets the source's trace too, the channels appended as columns the way
    the run appends them.
    """
    from dataclasses import replace

    import pandas as pd
    from geecs_data_utils.io.array1d import Data1DConfig, read_1d_data
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
    if source.siblings or source.channels:
        # The core's own trace: joined siblings, or the auxiliary channels
        # the measure reads appended as columns (x, y, channels…).
        return source.load(shot)
    # The schema's Data1DLoading mirrors the reader's Data1DConfig field for
    # field (pinned by the data-utils mirror test): a dump/validate converts.
    trace = read_1d_data(
        reference,
        Data1DConfig.model_validate(diag.line_loading.model_dump(mode="json")),
    )
    return trace.data


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
            file_shot=_file_shot(detail, shot, folder),
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
    data = _line_trace(diag, detail, folder, device, shot)
    figures = ephemeral.render_document_as_run(diag, [data], scan_folder=folder)
    if not figures:
        raise ValueError("this analyzer draws no figure for a single trace")
    return resources.figure_png(figures[0], tight=True)


#: the summary preview reads this many shots at most: a handful, never a
#: scan (an image run is gigabytes; the host is shared)
SUMMARY_SHOTS_MAX = 8


def _drawable(document: dict):
    """Validate a document that draws frames; a ``.himg`` job draws none (400)."""
    from geecs_schemas.analysis import HimgJob, load_analysis_document

    diag = load_analysis_document(document)
    if isinstance(diag, HimgJob):
        raise ValueError(f"a .himg job ({diag.kind}) draws no figure")
    return diag


def summary_preview(portal: PortalState, document: dict, params: dict, index: int) -> bytes:  # fmt: skip
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
    shots = max(1, min(SUMMARY_SHOTS_MAX, shots))
    diag = _drawable(document)
    # the document's own data folder, never a host-picked device: a
    # preview of one camera's frames under another's recipe is a lie
    device = diag.data_folder
    detail, folder = _preview_scan(portal, uid, device, day)
    arrays, positions, missing = [], [], []
    for shot in range(1, shots + 1):
        try:
            if is_line(diag):
                array = _line_trace(diag, detail, folder, device, shot)
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


def frame_preview(portal: PortalState, document: dict, params: dict) -> bytes:
    """One shot drawn as a run of this document draws it.

    Through ``render_document_as_run`` — the analysis sink's own
    per-frame call with the document's figure block — and cropped
    tight like the sink's PNGs, so the pane shows the product file the
    run would write, not a portal rendering of it.
    """
    ephemeral = portal.ephemeral_module()
    from geecs_analysis.recipe import is_line

    uid = str(params.get("uid") or "")
    day = str(params.get("day") or "")
    try:
        shot = int(params.get("shot") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("shot must be an integer") from exc
    if not uid or shot < 1:
        raise LookupError("preview needs a scan and a shot (>= 1)")
    diag = _drawable(document)
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


def _render(portal: PortalState, body: Json, draw: Callable[[Json, Json], bytes]) -> Response:  # fmt: skip
    """Validate the body's recipe as a run would load it, draw it, map the errors to statuses."""
    document, params = body.get("document"), body.get("params") or {}
    if not isinstance(document, dict) or not isinstance(params, dict):
        raise HTTPException(422, "body needs document + params mappings")
    report = validate_document(_root(portal), "analysis_recipe", document)
    if not report["ok"]:
        return JSONResponse({"detail": "invalid document", "errors": report["errors"]}, status_code=422)  # fmt: skip
    try:
        png = draw(document, params)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # a render failure is the document's, never a 500
        raise HTTPException(400, f"preview failed: {exc}") from exc
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/api/analysis/preview")
def analysis_preview(body: Json = Body(...), portal: PortalState = Depends(get_state)) -> Response:  # fmt: skip
    """``{document, params: {uid, day, shot}}`` → the unsaved recipe's frame on that shot, as a PNG."""
    return _render(portal, body, lambda d, p: frame_preview(portal, d, p))


@router.post("/api/analysis/preview/summary")
def analysis_summary_preview(body: Json = Body(...), portal: PortalState = Depends(get_state)) -> Response:  # fmt: skip
    """``{document, params: {uid, day, shots}, index}`` → its ``index``-th summary over the scan's first shots (at most 8)."""
    index = body.get("index", 0)
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise HTTPException(422, "body.index must be a non-negative integer")
    return _render(portal, body, lambda d, p: summary_preview(portal, d, p, index))
