"""The diagnostic document: one YAML per device, all of its analysis in one place.

A diagnostic names the device whose files are analyzed, picks the analyzer
(``analyzer:``), says how the raw frames or traces are cleaned up first
(``image:``), and how the analyzer runs over a scan (``scan:``).  The file
stem is the diagnostic's ID — the name groups refer to and the task queue
tracks.

This is format version 2.  Version 1 (the pre-0.19.0 unified diagnostic,
no ``schema_version``, an ``image_analyzer`` class path, analyzer settings
split between ``image.analysis`` and constructor ``kwargs``) is lifted
automatically at validation — see :meth:`AnalysisDiagnostic._lift_v1_layout`.
"""

from __future__ import annotations

from typing import Annotated, Any, Mapping, Optional, Union

from pydantic import Field, PrivateAttr, model_validator

from geecs_schemas._base import VersionedSchemaModel, stale_schema_version
from geecs_schemas.analysis.analyzers import ANALYZER_SPECS, AnalyzerSpec
from geecs_schemas.analysis.processing_1d import Line1DConfig
from geecs_schemas.analysis.processing_2d import CameraConfig
from geecs_schemas.analysis.scan_runtime import ScanRuntime

CURRENT_SCHEMA_VERSION = 2

ImageSection = Annotated[
    Union[CameraConfig, Line1DConfig],
    Field(discriminator="type"),
]

#: v1 ``image_analyzer`` class paths → v2 ``analyzer.kind``.  The class path
#: left the document in v2 (ImageAnalysis keeps kind → class in its own
#: registry, so refactoring a module no longer breaks configs); this table
#: is the lift's memory of where each kind used to live.
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

# v1 constructor kwargs that did not survive into the spec: BCaveMagOpt's
# ``line_config_name`` (never accepted by the constructor — the v1 form could
# not catch that).
_V1_DROPPED_KWARGS: dict[str, frozenset[str]] = {
    "bcave_mag_opt": frozenset({"line_config_name"}),
}

# v1 constructor kwargs renamed on the spec: LineStitcher's ``name`` labelled
# the stitched-output folder next to the master device; it is the spec's
# ``output_label`` (the deployed stitchers set it to something other than
# the diagnostic name, so it cannot be dropped in favour of ``output_name``).
_V1_RENAMED_KWARGS: dict[str, dict[str, str]] = {
    "line_stitcher": {"name": "output_label"},
}

_V1_LINE_BACKGROUND_RENAMES = {
    "constant_value": "constant_level",
    "background_file": "file_path",
}


def _lift_v1_analyzer(image_analyzer: Any) -> tuple[str, dict[str, Any]]:
    """Return ``(kind, kwargs)`` for a v1 ``image_analyzer`` value."""
    if isinstance(image_analyzer, str):
        class_path, kwargs = image_analyzer, {}
    elif isinstance(image_analyzer, Mapping):
        data = dict(image_analyzer)
        class_path = data.pop("class_path", None) or data.pop("class", None)
        kwargs = dict(data.pop("kwargs", None) or {})
        if class_path is None or data:
            raise ValueError(
                "v1 image_analyzer mapping must be {class_path, kwargs}; "
                f"got keys {sorted(image_analyzer)}"
            )
    else:
        raise ValueError(
            "v1 image_analyzer must be a class-path string or a mapping; "
            f"got {type(image_analyzer).__name__}"
        )
    kind = V1_CLASS_PATH_TO_KIND.get(class_path)
    if kind is None:
        raise ValueError(
            f"unknown v1 analyzer class {class_path!r}; the v2 kinds are "
            f"{sorted(ANALYZER_SPECS)}"
        )
    return kind, kwargs


def _lift_v1_image(image: Any) -> tuple[Any, dict[str, Any]]:
    """Return the v2 ``image:`` mapping and the v1 ``image.analysis`` dict."""
    if not isinstance(image, Mapping):
        return image, {}
    image = dict(image)
    analysis = image.pop("analysis", None) or {}
    if isinstance(analysis, Mapping) and isinstance(analysis.get("magspec"), Mapping):
        analysis = dict(analysis["magspec"])  # the legacy nested magspec form
    if "data_format" in image:
        # A v2 key already present (an override written in the new spelling)
        # wins over the lifted v1 key — same rule for every rename below.
        image.setdefault("label", image.pop("data_format"))
    pipeline = image.get("pipeline")
    if isinstance(pipeline, Mapping):
        image["pipeline"] = list(pipeline.get("steps") or [])
    if image.get("type") == "line" and isinstance(image.get("background"), Mapping):
        background = dict(image["background"])
        for old, new in _V1_LINE_BACKGROUND_RENAMES.items():
            if old in background:
                background.setdefault(new, background.pop(old))
        image["background"] = background
    return image, dict(analysis)


def _lift_v1_params(
    kind: str, kwargs: dict[str, Any], analysis: dict[str, Any]
) -> dict:
    """Merge v1 constructor kwargs and ``image.analysis`` into v2 spec fields."""
    params = {**kwargs, **analysis}
    for dropped in _V1_DROPPED_KWARGS.get(kind, ()):
        params.pop(dropped, None)
    for old, new in _V1_RENAMED_KWARGS.get(kind, {}).items():
        if old in params:
            params.setdefault(new, params.pop(old))
    if kind == "haso":
        mask = {
            side: params.pop(f"mask_{side}")
            for side in ("top", "bottom", "left", "right")
            if f"mask_{side}" in params
        }
        if mask:
            params["mask"] = mask
    return params


class AnalysisDiagnostic(VersionedSchemaModel):
    """One device's analysis: which analyzer, how frames are cleaned up, how it runs over a scan.

    The file's stem is the diagnostic ID that groups reference.  ``name``
    is the device whose folder holds the input files; ``output_name``
    (defaulting to ``name``) labels everything written — s-file columns,
    the output folder, figure files.
    """

    schema_version: int = Field(
        CURRENT_SCHEMA_VERSION,
        description=(
            "Format version of this config file. Leave at 2 — tools update "
            "this automatically when the file format changes."
        ),
    )
    name: str = Field(
        ...,
        min_length=1,
        description="The device whose data folder under scans/ScanNNN/ is analyzed.",
    )
    output_name: Optional[str] = Field(
        None,
        description=(
            "Label for everything this analyzer writes (s-file column prefix, "
            "output folder). Defaults to name; set it to run two analyzers "
            "over one device with distinct outputs."
        ),
    )
    metric_suffix: Optional[str] = Field(
        None,
        description="Suffix appended to every s-file column name; affects scalars only, never files or folders.",
    )
    description: Optional[str] = Field(
        None, description="Free-text note about this diagnostic."
    )
    analyzer: AnalyzerSpec = Field(
        ..., description="Which analyzer runs and its own parameters; chosen by kind."
    )
    image: Optional[ImageSection] = Field(
        None,
        description=(
            "How raw frames (type: camera) or traces (type: line) are cleaned "
            "up before analysis. Omit for analyzers that read their own file "
            "formats (kind haso, phase_downramp)."
        ),
    )
    scan: ScanRuntime = Field(
        default_factory=ScanRuntime,
        description="How the analyzer runs over a scan: order, per shot or per bin, saving, files.",
    )
    _source_id: Optional[str] = PrivateAttr(default=None)

    @property
    def source_id(self) -> Optional[str]:
        """Filename-derived diagnostic ID when loaded from disk, if known."""
        return self._source_id

    @property
    def effective_output_name(self) -> str:
        """``output_name`` if set, else ``name`` — the one output label."""
        return self.output_name if self.output_name is not None else self.name

    @property
    def image_kind(self) -> Optional[str]:
        """``"camera"``, ``"line"`` or ``None`` — read off the image section."""
        return None if self.image is None else self.image.type

    @model_validator(mode="before")
    @classmethod
    def _lift_v1_layout(cls, data: object) -> object:
        """Lift the v1 unified-diagnostic layout into the v2 shape.

        A v1 document is recognised by its ``image_analyzer`` key.  The
        class path becomes ``analyzer.kind`` (via
        :data:`V1_CLASS_PATH_TO_KIND`), constructor ``kwargs`` and
        ``image.analysis`` merge into the analyzer spec, ``image.data_format``
        becomes ``image.label``, ``pipeline: {steps: [...]}`` becomes the bare
        list, the 1D background fields take the camera spellings, and
        ``scan.renderer_kwargs`` becomes ``scan.renderer``.  A v1-layout
        document is always stamped 2 (its layout defines its version); a
        v2-layout document with a stale stamp is normalised up to 2 and one
        with a newer stamp is left alone.

        Parameters
        ----------
        data : object
            The raw input; non-mapping input passes through untouched.

        Returns
        -------
        object
            The (copied) mapping in v2 layout, or *data* unchanged.

        Raises
        ------
        ValueError
            If a v1 document names an analyzer class this version does not
            know, or mixes ``image_analyzer`` with a v2 ``analyzer`` block.
        """
        if not isinstance(data, dict):
            return data
        is_v1 = "image_analyzer" in data
        if not is_v1 and not stale_schema_version(data, CURRENT_SCHEMA_VERSION):
            return data
        if is_v1 and "analyzer" in data:
            raise ValueError(
                "a diagnostic cannot carry both the v1 'image_analyzer' and the "
                "v2 'analyzer' block"
            )
        lifted = dict(data)
        if is_v1:
            kind, kwargs = _lift_v1_analyzer(lifted.pop("image_analyzer"))
            image, analysis = _lift_v1_image(lifted.get("image"))
            if image is not None:
                lifted["image"] = image
            lifted["analyzer"] = {
                "kind": kind,
                **_lift_v1_params(kind, kwargs, analysis),
            }
            scan = lifted.get("scan")
            if isinstance(scan, Mapping):
                scan = dict(scan)
                if "renderer_kwargs" in scan:
                    scan.setdefault("renderer", scan.pop("renderer_kwargs"))
                lifted["scan"] = scan
            elif scan is None:
                lifted.pop("scan", None)
        lifted["schema_version"] = CURRENT_SCHEMA_VERSION
        return lifted

    @model_validator(mode="after")
    def _sections_agree(self) -> "AnalysisDiagnostic":
        """Check the analyzer's image kind against the image section, and the renderer options against it."""
        wanted = type(self.analyzer).image_kind
        have = self.image_kind
        if wanted != have:
            if wanted is None:
                raise ValueError(
                    f"analyzer kind {self.analyzer.kind!r} takes no image section, "
                    f"but image.type is {have!r}"
                )
            raise ValueError(
                f"analyzer kind {self.analyzer.kind!r} needs image.type "
                f"{wanted!r}, but the document has {have!r}"
            )
        if self.analyzer.kind == "line_stitcher":
            label = self.analyzer.output_label or self.effective_output_name
            master = self.scan.device or self.name
            if label == master:
                raise ValueError(
                    f"line_stitcher output label {label!r} equals the master "
                    "device's data folder; the stitched traces would overwrite "
                    "the raw input files. Set analyzer.output_label (or "
                    "output_name) to a different name."
                )
        renderer = self.scan.renderer
        if have == "camera":
            wrong = renderer.fields_set_for(renderer.LINE_ONLY)
            if wrong:
                raise ValueError(
                    f"scan.renderer fields {wrong} apply to line diagnostics only"
                )
        if have == "line":
            wrong = renderer.fields_set_for(renderer.CAMERA_ONLY)
            if wrong:
                raise ValueError(
                    f"scan.renderer fields {wrong} apply to camera diagnostics only"
                )
        return self


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "AnalysisDiagnostic",
    "ImageSection",
    "V1_CLASS_PATH_TO_KIND",
]
