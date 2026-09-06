"""The diagnostic document: one YAML per device, all of its analysis in one place.

A diagnostic names the device whose files are analyzed, picks the analyzer
(``analyzer:``), says how the raw frames or traces are cleaned up first
(``image:``), and how the analyzer runs over a scan (``scan:``).  The file
stem is the diagnostic's ID — the name groups refer to and the task queue
tracks.

This is format version 2.  Version 1 (the pre-0.19.0 unified diagnostic,
no ``schema_version``, an ``image_analyzer`` class path, analyzer settings
split between ``image.analysis`` and constructor ``kwargs``) is refused
here; the corpus was regenerated with
:mod:`geecs_schemas.convert.analysis_diagnostics`.
"""

from __future__ import annotations

from typing import Annotated, Optional, Union

from pydantic import Field, PrivateAttr, model_validator

from geecs_schemas._base import VersionedSchemaModel, stale_schema_version
from geecs_schemas.analysis.analyzers import AnalyzerSpec
from geecs_schemas.analysis.processing_1d import Line1DConfig
from geecs_schemas.analysis.processing_2d import CameraConfig
from geecs_schemas.analysis.scan_runtime import ScanRuntime

CURRENT_SCHEMA_VERSION = 2

ImageSection = Annotated[
    Union[CameraConfig, Line1DConfig],
    Field(discriminator="type"),
]

#: The pre-0.19.0 layout is recognised by these keys; such a document is
#: refused with a pointer to the one-shot converter
#: (:mod:`geecs_schemas.convert.analysis_diagnostics`) — the corpus is
#: regenerated in v2, not lifted at run time.
_V1_MARKERS = ("image_analyzer",)


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
    def _refuse_v1_layout(cls, data: object) -> object:
        """Refuse the pre-0.19.0 layout with a pointer to the converter.

        A v1 document (``image_analyzer`` class path, analyzer settings in
        ``image.analysis`` / constructor ``kwargs``) or an explicit
        ``schema_version: 1`` is not lifted here: the analysis-config corpus
        is regenerated in v2 with
        ``python -m geecs_schemas.convert.analysis_diagnostics``.

        Parameters
        ----------
        data : object
            The raw input; non-mapping input passes through untouched.

        Returns
        -------
        object
            *data* unchanged.

        Raises
        ------
        ValueError
            If the document is in the v1 layout.
        """
        if not isinstance(data, dict):
            return data
        if any(marker in data for marker in _V1_MARKERS) or stale_schema_version(
            data, CURRENT_SCHEMA_VERSION
        ):
            raise ValueError(
                "this is a pre-v2 analysis diagnostic (image_analyzer / "
                "schema_version 1); regenerate it with "
                "`python -m geecs_schemas.convert.analysis_diagnostics <tree> --write`"
            )
        return data

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
]
