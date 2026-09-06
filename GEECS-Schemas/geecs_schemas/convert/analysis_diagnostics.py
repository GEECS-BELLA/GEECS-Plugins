"""One-shot converter: pre-0.19.0 analysis diagnostics (v1) → :class:`AnalysisDiagnostic` v2.

The analysis-config corpus in GEECS-Plugins-configs is the whole universe of
diagnostics, so the models validate the current shape only; this module is
the mechanical rewrite that brought the corpus to v2 (and can go the way of
the scan-variables converter once that regeneration has landed).

v1 shape (``image_analyzer`` class path, analyzer settings split between
``image.analysis`` and constructor ``kwargs``)::

    name: UC_TopView
    image_analyzer: image_analysis.analyzers.beam_analyzer.BeamAnalyzer
    image: {type: camera, analysis: {compute_slopes: true}, pipeline: {steps: [roi]}}
    scan: {priority: 10, renderer_kwargs: {cmap: plasma}}

v2 shape::

    schema_version: 2
    name: UC_TopView
    analyzer: {kind: beam, compute_slopes: true}
    image: {type: camera, pipeline: [roi]}
    scan: {priority: 10, renderer: {cmap: plasma}}

Mapping:

- ``image_analyzer`` class path → ``analyzer.kind`` via
  :data:`V1_CLASS_PATH_TO_KIND`; constructor ``kwargs`` and
  ``image.analysis`` merge into the spec's fields (the legacy nested
  ``analysis.magspec`` block is flattened).
- HASO's ``mask_top/bottom/left/right`` kwargs → ``mask: {top, ...}``.
- LineStitcher's ``name`` kwarg → ``output_label`` (it labels the stitched
  output folder next to the master device and is *not* the diagnostic
  name — dropping it would write stitched traces over the raw inputs).
- BCaveMagOpt's ``line_config_name`` kwarg is dropped (never accepted by
  the constructor).
- ``image.pipeline: {steps: [...]}`` → the bare list; ``image.data_format``
  → ``image.label``; the 1D background fields ``constant_value`` /
  ``background_file`` → ``constant_level`` / ``file_path``.
- ``scan.renderer_kwargs`` → ``scan.renderer``.
- Stale ``enabled`` flags inside processing sections (pre-#412 residue the
  old nested models ignored) are dropped; the pipeline list is the gate.
- A ``schema_version: 2`` stamp is added.

Groups are unchanged apart from the ``schema_version: 1`` stamp.

The result is validated, so a v1 file whose ignored ``image.analysis`` keys
never matched its analyzer (the old loader was lenient there) fails the
conversion loudly instead of losing them silently.

CLI (rewrite a ``scan_analysis_configs`` tree in place, canonical YAML)::

    python -m geecs_schemas.convert.analysis_diagnostics <tree> --dry-run
    python -m geecs_schemas.convert.analysis_diagnostics <tree> --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ValidationError

from geecs_schemas.analysis import ANALYZER_SPECS, AnalysisDiagnostic, AnalysisGroup
from geecs_schemas.convert._common import SchemaConversionError, load_legacy

__all__ = [
    "V1_CLASS_PATH_TO_KIND",
    "canonical_document",
    "convert_v1_diagnostic",
    "is_v1_diagnostic",
    "regenerate_tree",
]

#: v1 ``image_analyzer`` class paths → v2 ``analyzer.kind``.
V1_CLASS_PATH_TO_KIND: dict[str, str] = {
    "image_analysis.analyzers.standard_analyzer.StandardAnalyzer": "standard",
    "image_analysis.analyzers.standard_1d_analyzer.Standard1DAnalyzer": "trace",
    "image_analysis.analyzers.line_analyzer.LineAnalyzer": "line",
    "image_analysis.analyzers.beam_analyzer.BeamAnalyzer": "beam",
    "image_analysis.analyzers.magspec_manual_calib_analyzer.MagSpecManualCalibAnalyzer": "magspec",
    "image_analysis.analyzers.grenouille_analyzer.GrenouilleAnalyzer": "frog_retrieval",
    "image_analysis.analyzers.frog_spectral_phase_analyzer.FrogSpectralPhaseAnalyzer": "frog_spectral_phase",
    "image_analysis.analyzers.ict_1d_analyzer.ICT1DAnalyzer": "ict",
    "image_analysis.analyzers.line_stitcher.LineStitcher": "line_stitcher",
    "image_analysis.analyzers.HASO_himg_has_processor.HASOHimgHasProcessor": "haso",
    "image_analysis.analyzers.downramp_phase_analyzer.DownrampPhaseAnalyzer": "downramp_phase",
    "image_analysis.analyzers.Undulator.hi_res_mag_cam_analyzer.HiResMagCamAnalyzer": "hi_res_mag_cam",
    "image_analysis.analyzers.Undulator.BCaveMagSpecStitcher.BCaveMagSpecStitcherAnalyzer": "bcave_magspec_stitcher",
    "image_analysis.analyzers.Undulator.BCaveMagSpecStitcherOpt.BCaveMagOpt": "bcave_mag_opt",
    "image_analysis.analyzers.density_from_phase_analysis.PhaseDownrampProcessor": "phase_downramp",
}

_DROPPED_KWARGS: dict[str, frozenset[str]] = {
    "bcave_mag_opt": frozenset({"line_config_name"}),
}
_RENAMED_KWARGS: dict[str, dict[str, str]] = {
    "line_stitcher": {"name": "output_label"},
}
_LINE_BACKGROUND_RENAMES = {
    "constant_value": "constant_level",
    "background_file": "file_path",
}


def is_v1_diagnostic(data: Mapping[str, Any]) -> bool:
    """Whether a raw mapping is in the v1 layout (it names an ``image_analyzer``)."""
    return "image_analyzer" in data


def _analyzer(image_analyzer: Any, context: str) -> tuple[str, dict[str, Any]]:
    if isinstance(image_analyzer, str):
        class_path, kwargs = image_analyzer, {}
    elif isinstance(image_analyzer, Mapping):
        data = dict(image_analyzer)
        class_path = data.pop("class_path", None) or data.pop("class", None)
        kwargs = dict(data.pop("kwargs", None) or {})
        if class_path is None or data:
            raise SchemaConversionError(
                f"{context}: image_analyzer mapping must be {{class_path, kwargs}}; "
                f"got keys {sorted(image_analyzer)}"
            )
    else:
        raise SchemaConversionError(
            f"{context}: image_analyzer must be a class-path string or a mapping"
        )
    kind = V1_CLASS_PATH_TO_KIND.get(class_path)
    if kind is None:
        raise SchemaConversionError(
            f"{context}: unknown analyzer class {class_path!r}; the v2 kinds are "
            f"{sorted(ANALYZER_SPECS)}"
        )
    return kind, kwargs


def _image(image: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(image, Mapping):
        return image, {}
    image = dict(image)
    analysis = image.pop("analysis", None) or {}
    if isinstance(analysis, Mapping) and isinstance(analysis.get("magspec"), Mapping):
        analysis = dict(analysis["magspec"])
    if "data_format" in image:
        image["label"] = image.pop("data_format")
    pipeline = image.get("pipeline")
    if isinstance(pipeline, Mapping):
        image["pipeline"] = list(pipeline.get("steps") or [])
    # The pre-#412 processing sections carried an ``enabled`` flag the old
    # (extra-tolerant) nested models ignored; the pipeline list is the gate.
    for key, section in list(image.items()):
        if isinstance(section, Mapping) and "enabled" in section and key != "metadata":
            section = dict(section)
            section.pop("enabled")
            image[key] = section
    if image.get("type") == "line" and isinstance(image.get("background"), Mapping):
        background = dict(image["background"])
        for old, new in _LINE_BACKGROUND_RENAMES.items():
            if old in background:
                background[new] = background.pop(old)
        image["background"] = background
    return image, dict(analysis)


def _params(kind: str, kwargs: dict[str, Any], analysis: dict[str, Any]) -> dict:
    params = {**kwargs, **analysis}
    for dropped in _DROPPED_KWARGS.get(kind, ()):
        params.pop(dropped, None)
    for old, new in _RENAMED_KWARGS.get(kind, {}).items():
        if old in params:
            params[new] = params.pop(old)
    if kind == "haso":
        mask = {
            side: params.pop(f"mask_{side}")
            for side in ("top", "bottom", "left", "right")
            if f"mask_{side}" in params
        }
        if mask:
            params["mask"] = mask
    return params


def convert_v1_diagnostic(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Convert one v1 diagnostic (mapping or YAML path) to a validated v2 mapping.

    Returns the canonical v2 document (``model_dump`` of the validated
    :class:`AnalysisDiagnostic`, unset fields and default-``None`` fields
    omitted).  A document already in v2 layout is validated and returned
    canonically too.

    Raises
    ------
    SchemaConversionError
        If the analyzer class is unknown, the layout is malformed, or the
        merged parameters do not validate against the analyzer's spec.
    """
    data = (
        dict(load_legacy(source)) if not isinstance(source, Mapping) else dict(source)
    )
    context = (
        str(source) if not isinstance(source, Mapping) else data.get("name", "<dict>")
    )
    if is_v1_diagnostic(data):
        kind, kwargs = _analyzer(data.pop("image_analyzer"), context)
        image, analysis = _image(data.get("image"))
        if image is not None:
            data["image"] = image
        else:
            data.pop("image", None)
        data["analyzer"] = {"kind": kind, **_params(kind, kwargs, analysis)}
        scan = data.get("scan")
        if isinstance(scan, Mapping):
            scan = dict(scan)
            if "renderer_kwargs" in scan:
                scan["renderer"] = scan.pop("renderer_kwargs")
            data["scan"] = scan
        elif scan is None:
            data.pop("scan", None)
    data["schema_version"] = 2
    try:
        model = AnalysisDiagnostic.model_validate(data)
    except ValidationError as exc:
        raise SchemaConversionError(f"{context}: {exc}") from exc
    return canonical_document(model)


def convert_group(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate a group document and return it canonically (with its version stamp)."""
    data = (
        dict(load_legacy(source)) if not isinstance(source, Mapping) else dict(source)
    )
    try:
        model = AnalysisGroup.model_validate(data)
    except ValidationError as exc:
        raise SchemaConversionError(f"{source}: {exc}") from exc
    return canonical_document(model)


def canonical_document(model: BaseModel) -> dict[str, Any]:
    """The canonical on-disk form: set fields only, minus ``None`` values whose default is ``None``.

    ``schema_version`` is always written first so a reader sees the format
    before the content.
    """
    dumped = model.model_dump(mode="json", exclude_unset=True)
    pruned = _prune_default_nones(model, dumped)
    version = getattr(model, "schema_version", None)
    if version is not None:
        pruned = {
            "schema_version": version,
            **{k: v for k, v in pruned.items() if k != "schema_version"},
        }
    return pruned


def _prune_default_nones(model: BaseModel, dumped: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in dumped.items():
        field = type(model).model_fields.get(name)
        if value is None and field is not None and field.default is None:
            continue
        child = getattr(model, name, None)
        if isinstance(child, BaseModel) and isinstance(value, dict):
            value = _prune_default_nones(child, value)
        out[name] = value
    return out


def regenerate_tree(
    root: Path,
    *,
    write: bool = False,
    out: Optional[Path] = None,
    skip_namespaces: frozenset[str] = frozenset({"UNCLASSIFIED"}),
) -> list[str]:
    """Convert every diagnostic and group under a ``scan_analysis_configs`` tree.

    Returns a report (one line per file).  With ``write=False`` nothing is
    written; with ``write=True`` files are rewritten in place (or under
    ``out`` when given) in canonical v2 YAML.
    """
    import yaml

    root = Path(root)
    report: list[str] = []

    def emit(rel: Path, document: dict[str, Any]) -> None:
        if not write:
            return
        target = (out / rel) if out is not None else (root / rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
        )

    for path in sorted((root / "analyzers").glob("*/*.y*ml")):
        rel = path.relative_to(root)
        if path.parent.name in skip_namespaces:
            report.append(f"skip     {rel} (namespace {path.parent.name})")
            continue
        try:
            document = convert_v1_diagnostic(path)
        except SchemaConversionError as exc:
            report.append(f"FAILED   {rel}: {str(exc).splitlines()[0][:160]}")
            continue
        emit(rel, document)
        report.append(f"ok       {rel}")
    for path in sorted((root / "groups").glob("*/*.y*ml")):
        rel = path.relative_to(root)
        try:
            document = convert_group(path)
        except SchemaConversionError as exc:
            report.append(f"FAILED   {rel}: {str(exc).splitlines()[0][:160]}")
            continue
        emit(rel, document)
        report.append(f"ok       {rel}")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point: report or rewrite a configs tree."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tree", type=Path, help="the scan_analysis_configs root")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="report only")
    mode.add_argument("--write", action="store_true", help="rewrite files in place")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write under this root instead of in place",
    )
    args = parser.parse_args(argv)
    report = regenerate_tree(args.tree, write=args.write, out=args.out)
    print("\n".join(report))
    failed = [line for line in report if line.startswith("FAILED")]
    print(f"\n{len(report) - len(failed)} ok, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
