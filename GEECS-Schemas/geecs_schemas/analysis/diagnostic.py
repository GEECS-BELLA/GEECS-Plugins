"""The diagnostic document: one YAML per device, all of its analysis in one place.

A diagnostic names the device whose files are analyzed, picks the analyzer
(``analyzer:``), says how the raw frames or traces are cleaned up first
(``image:``), and how the analyzer runs over a scan (``scan:``).  The file
stem is the diagnostic's ID — the name groups refer to and the task queue
tracks.

This is format version 2.  Version 1 (the pre-0.19.0 unified diagnostic,
no ``schema_version``, an ``image_analyzer`` class path, analyzer settings
split between ``image.analysis`` and constructor ``kwargs``) is refused
here; the corpus was regenerated in v2 once (0.19.0) and is authored
v2-only since.
"""

from __future__ import annotations

from typing import Annotated, Optional, Union

from pydantic import Field, PrivateAttr, model_validator

from geecs_schemas._base import VersionedSchemaModel, stale_schema_version
from geecs_schemas.analysis.analyzers import AnalyzerSpec
from geecs_schemas.analysis.processing_1d import Data1DLoading, Line1DConfig
from geecs_schemas.analysis.processing_2d import CameraConfig
from geecs_schemas.analysis.scan_runtime import ScanRuntime

CURRENT_SCHEMA_VERSION = 2

ImageSection = Annotated[
    Union[CameraConfig, Line1DConfig],
    Field(discriminator="type"),
]

#: The pre-0.19.0 layout is recognised by these keys; such a document is
#: refused — the corpus is v2 only, nothing is lifted at run time.
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

    # The format-neutral names a consumer reads without asking which document
    # it holds; the v3 recipe (``recipe.py``) carries the same properties.

    @property
    def device(self) -> str:
        """The device whose files are analyzed (``name``); the recipe's ``device``."""
        return self.name

    @property
    def input_kind(self) -> Optional[str]:
        """``"camera"``, ``"line"`` or ``None``; the recipe's ``input.kind``."""
        return self.image_kind

    @property
    def data_folder(self) -> str:
        """The subfolder under scans/ScanNNN/ holding the files: ``scan.device`` or the device."""
        return self.scan.device or self.name

    @property
    def line_loading(self) -> Optional[Data1DLoading]:
        """How one trace file is read, for a line diagnostic; ``None`` otherwise."""
        return self.image.data_loading if isinstance(self.image, Line1DConfig) else None

    @model_validator(mode="before")
    @classmethod
    def _refuse_v1_layout(cls, data: object) -> object:
        """Refuse the pre-0.19.0 layout.

        A v1 document (``image_analyzer`` class path, analyzer settings in
        ``image.analysis`` / constructor ``kwargs``) or an explicit
        ``schema_version: 1`` is not lifted: the analysis-config corpus was
        regenerated in v2 once (GEECS-Schemas 0.19.0), so a v1 file is a
        stray to rewrite by hand or restore from the configs repo.

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
                "schema_version 1); the corpus is v2 only — rewrite it as "
                "`analyzer: {kind: ...}` + `image:` + `scan:` or restore it "
                "from the configs repo"
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
            self._check_line_source()
        return self

    def _check_line_source(self) -> None:
        """Where a LINE diagnostic reads its traces from must be said once, consistently.

        A camera diagnostic needs one switch — ``scan.data_format`` —
        because the loader recognises a capture-stack ``ShotRef`` on
        sight.  A line diagnostic needs two that agree, because its
        loader dispatches on the configured ``data_loading.data_type``
        instead, and the two failure modes are both silent: a stack
        handed to a file reader, or a per-shot path handed to the stack
        reader, fails once per shot and yields an empty analysis rather
        than an error anyone sees.  So the pairing is checked here, where
        both sections are in hand — together with the one analyzer that
        cannot read a stack at all whatever the pair says.
        """
        from geecs_schemas.analysis.processing_1d import (
            Data1DType,
            LineBackgroundMethod,
        )

        stack_source = self.image.data_loading.data_type == Data1DType.PVA_STACK
        stack_format = self.scan.data_format == "device_hdf5"
        if stack_source != stack_format:
            raise ValueError(
                "a line diagnostic reading the per-device capture stack needs "
                "BOTH image.data_loading.data_type: pva_stack and "
                "scan.data_format: device_hdf5 — this document has "
                f"data_type {self.image.data_loading.data_type.value!r} with "
                f"data_format {self.scan.data_format!r}, which reads nothing"
            )
        if stack_format and self.analyzer.kind == "line_stitcher":
            # scan.data_format's own rule ("only for analyzers that do not
            # derive output names from the shot file path") names this
            # analyzer exactly: the stitcher finds its sibling devices by
            # rewriting the master's per-shot path and writes its output
            # beside it, and a stack frame has no such path.
            raise ValueError(
                "analyzer kind 'line_stitcher' cannot read the per-device "
                "capture stack: it finds its sibling traces by rewriting the "
                "master device's per-shot file path, and writes its output "
                "beside that file — a stack frame has neither"
            )
        background = self.image.background
        if (
            stack_source
            and background is not None
            and background.method == LineBackgroundMethod.FROM_FILE
        ):
            raise ValueError(
                "background.method 'from_file' cannot read a pva_stack: a "
                "capture stack holds every shot of a scan, so reading one "
                "needs a frame index and background.file_path has nowhere to "
                "put one. Use a per-shot background file, or 'constant'."
            )


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "AnalysisDiagnostic",
    "ImageSection",
]
