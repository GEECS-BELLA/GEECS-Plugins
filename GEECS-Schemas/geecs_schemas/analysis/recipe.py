"""The analysis recipe: one device's analysis as the analysis core runs it (format v3).

A recipe says where the frames come from (``input:``), how each frame is
processed (``steps:``, an ordered, repeatable list), what is measured on it
(``measure:``), how it is drawn (``figure:``, the per-frame draw), which
scan-level figures summarise the scan (``summaries:``, a list of frozen
kinds), and how the run behaves (``scan:``).  The file stem is the
recipe's ID, the name groups refer to and the task queue tracks.

Two vocabularies meet here.  The *frame* of the document (naming, input,
runtime, figure, summaries) is schema vocabulary and is fully typed in
this module.  The *numerical* vocabulary (which steps and measures exist
and what parameters they take) is the analysis core's registry
(``geecs_analysis.steps`` / ``geecs_analysis.measures``, one file per
builtin) and is not duplicated here: a :class:`StepRef` / :class:`MeasureRef`
carries the registered name plus its parameters as they were written, and
the core binds them to the registry when it compiles the recipe, refusing
unknown names, unknown parameters and steps that do not support the
input's dimensionality.  A schema-only consumer (a listing, MCP, this
package's corpus walk) therefore validates a recipe's frame without the
core; the editor lists the step vocabulary from the core's registry.

This is format version 3, the shape the analysis core reads natively.  The
version 2 document (:class:`~geecs_schemas.analysis.diagnostic.AnalysisDiagnostic`)
is still read for the analyzer kinds the core has not ported; a v2
recipe the core serves converts to this shape once through
``geecs_analysis.compat.convert``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.analysis.processing_1d import Data1DLoading, Data1DType

CURRENT_RECIPE_VERSION = 3


def declared_schema_version(data: Mapping[str, object]) -> Optional[int]:
    """Return the ``schema_version`` a raw document declares, or ``None``.

    A quoted digit counts the same as the int (pydantic coerces it later);
    anything else (absent, unparseable) is ``None`` and the document is
    read as the default version of whichever model receives it.
    """
    version = data.get("schema_version")
    if isinstance(version, str) and version.isdigit():
        version = int(version)
    if isinstance(version, int) and not isinstance(version, bool):
        return version
    return None


# --------------------------------------------------------------- vocabulary


class StepRef(SchemaModel):
    """One processing step by its registered name, parameters as written.

    Every key other than ``step`` is a parameter of that step and is
    validated by the analysis core's registry when the recipe compiles
    (unknown parameters are refused there, exactly as unknown keys are
    refused here for typed sections).
    """

    model_config = ConfigDict(extra="allow")

    step: str = Field(
        ...,
        min_length=1,
        description=(
            "Registered step name ('roi', 'median', 'background_constant', "
            "...). The remaining keys are that step's parameters."
        ),
    )

    def parameters(self) -> Dict[str, Any]:
        """The step's parameters as written (everything but ``step``)."""
        return dict(self.model_extra or {})


class MeasureRef(SchemaModel):
    """The measure by its registered kind, parameters as written.

    ``none`` (the default) processes frames without measuring anything.
    """

    model_config = ConfigDict(extra="allow")

    kind: str = Field(
        "none",
        min_length=1,
        description=(
            "Registered measure kind ('beam', 'line', 'none'). The remaining "
            "keys are that measure's parameters."
        ),
    )

    def parameters(self) -> Dict[str, Any]:
        """The measure's parameters as written (everything but ``kind``)."""
        return dict(self.model_extra or {})


# -------------------------------------------------------------------- input


class _InputBase(SchemaModel):
    #: Declared here so ``kind`` stays the first key of every input; each
    #: input narrows it to its literal.
    kind: str = Field(..., description="Input kind.")
    folder: Optional[str] = Field(
        None,
        description=(
            "Data subfolder under scans/ScanNNN/ when it differs from the "
            "device (stitched or post-processed outputs in a sibling folder)."
        ),
    )
    file_tail: Optional[str] = Field(
        None,
        description=(
            "Filename suffix that identifies this device's files ('.png', "
            "'.txt', '_postprocessed.tsv'); unset uses the kind's default."
        ),
    )
    format: Optional[Literal["per_shot_files", "device_hdf5"]] = Field(
        None,
        description=(
            "'device_hdf5' reads the per-device frame stack the PVA gateway's "
            "file plugin writes; a camera falls back to per-shot files when no "
            "stack can be mapped, a 'pva_stack' trace does not."
        ),
    )


class CameraInput(_InputBase):
    """Frames are camera images (2D)."""

    ndim: ClassVar[int] = 2

    kind: Literal["camera"] = Field("camera", description="Image input.")


class LineInput(_InputBase):
    """Frames are traces (1D): how one file is read and scaled."""

    ndim: ClassVar[int] = 1

    kind: Literal["line"] = Field("line", description="Trace input.")
    loading: Data1DLoading = Field(..., description="How to read one trace file.")
    x_scale: float = Field(
        1.0, description="Multiply the file's x values by this before processing."
    )
    y_scale: float = Field(
        1.0, description="Multiply the file's y values by this before processing."
    )
    x_unit: str = Field("", description="Unit of the scaled x axis (axis labels).")
    y_unit: str = Field("", description="Unit of the scaled y values (axis labels).")
    label: str = Field("", description="What the trace is (the y-axis label).")
    storage_dtype: Literal["float32", "float64"] = Field(
        "float32",
        description=(
            "Precision of the stored processed trace; statistics are taken "
            "from the stored values, so this rounds them too."
        ),
    )

    @model_validator(mode="after")
    def _stack_source_agrees(self) -> "LineInput":
        """A capture-stack trace needs both switches, or the reader reads nothing."""
        stack_source = self.loading.data_type == Data1DType.PVA_STACK
        stack_format = self.format == "device_hdf5"
        if stack_source != stack_format:
            raise ValueError(
                "a line input reading the per-device capture stack needs BOTH "
                "loading.data_type: pva_stack and format: device_hdf5 — this "
                f"recipe has data_type {self.loading.data_type.value!r} with "
                f"format {self.format!r}, which reads nothing"
            )
        return self


RecipeInput = Annotated[Union[CameraInput, LineInput], Field(discriminator="kind")]


class FrameInput(SchemaModel):
    """A frame the source layer loads before the run and binds by name.

    Steps refer to it by the key it is stored under (``background_frame``'s
    ``source``).  The analysis core never opens the path; the scan host
    resolves ``{scan_dir}`` to the device's data directory and loads it.
    """

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "File to load ('{scan_dir}' stands for the device's data "
            "directory under the scan)."
        ),
    )
    fallback_level: Optional[float] = Field(
        None,
        description=(
            "When the file cannot be read, subtract this constant instead of "
            "the frame and warn; unset makes a failed read an error."
        ),
    )


# ------------------------------------------------------------------- figure


class FigureStyle(SchemaModel):
    """The per-frame draw: matplotlib keyword groups and overlay styles by id.

    Each group passes straight to the matplotlib call of that name and is
    validated by rendering (the editor's preview); the analysis core owns
    the frame's geometry, so image ``extent`` cannot be set.  ``colorbar.show``
    hides the colorbar; an overlay entry's ``hidden`` hides that overlay and
    its ``scale`` sets a projection's height as a fraction of the image.
    Every summary kind reuses this draw for its panels.
    """

    # dict[str, Any] is deliberate throughout: the values are matplotlib's
    # own vocabulary (strings, numbers, sequences, scale names), validated
    # by drawing, not a second schema of matplotlib.
    imshow: Dict[str, Any] = Field(
        default_factory=dict,
        description="Axes.imshow keywords (cmap, vmin, vmax, ...).",
    )
    pcolormesh: Dict[str, Any] = Field(
        default_factory=dict,
        description="Axes.pcolormesh keywords, used for nonuniform image axes.",
    )
    plot: Dict[str, Any] = Field(
        default_factory=dict, description="Axes.plot keywords for traces."
    )
    colorbar: Dict[str, Any] = Field(
        default_factory=dict,
        description="Figure.colorbar keywords (label, ...); show: false hides it.",
    )
    axes: Dict[str, Any] = Field(
        default_factory=dict, description="Axes.set keywords (xlabel, ylabel, title)."
    )
    fig: Dict[str, Any] = Field(
        default_factory=dict, description="Figure keywords (figsize, dpi)."
    )
    overlays: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Style per overlay id the measure emits (projection_x, "
            "projection_y, com): Axes.plot keywords, plus hidden and scale."
        ),
    )


# ---------------------------------------------------------------- summaries


class _SummaryBase(SchemaModel):
    #: Frame dimensionalities this kind can summarise; checked against the
    #: recipe's input (validation, never selection).
    frame_ndim: ClassVar[frozenset[int]]


class ImageGridSummary(_SummaryBase):
    """One panel per bin (the bin-averaged image), one shared colour scale."""

    frame_ndim: ClassVar[frozenset[int]] = frozenset({2})

    kind: Literal["image_grid"] = Field(
        "image_grid", description="One panel per bin, one shared colour scale."
    )
    columns: Optional[int] = Field(
        None, ge=1, description="Panels per row; unset squares the grid."
    )
    panel_size: Optional[Tuple[float, float]] = Field(
        None,
        description="Panel (width, height) in inches; unset uses the renderer default.",
    )


class WaterfallSummary(_SummaryBase):
    """Every bin's (or, on a noscan, every shot's) trace as one row of a heat map."""

    frame_ndim: ClassVar[frozenset[int]] = frozenset({1})

    kind: Literal["waterfall"] = Field(
        "waterfall", description="Every row a bin's (or shot's) trace."
    )
    sort_key: Optional[str] = Field(
        None,
        description=(
            "On a noscan, order rows by this s-file column instead of shot "
            "number ('Device:Var' or a substring); also skips bin averaging."
        ),
    )
    sort_sigma: float = Field(
        3.0,
        gt=0,
        description="Drop rows whose sort-key value lies outside mean ± this many standard deviations.",
    )
    sort_bounds: Optional[Tuple[float, float]] = Field(
        None,
        description="Explicit (low, high) bounds on the sort key; overrides the sigma cut.",
    )
    even_spacing: Optional[bool] = Field(
        None,
        description=(
            "Draw rows at equal height regardless of position spacing; unset "
            "means yes when sorting by a key, no otherwise."
        ),
    )
    scale: Literal["auto", "sequential", "diverging", "custom"] = Field(
        "auto",
        description=(
            "'sequential' runs 0 to max; 'diverging' is symmetric about zero; "
            "'auto' centres on zero when the data crosses it; 'custom' uses "
            "vmin/vmax as given."
        ),
    )
    cmap: Optional[str] = Field(None, description="Matplotlib colormap name.")
    vmin: Optional[float] = Field(None, description="Colour scale minimum.")
    vmax: Optional[float] = Field(None, description="Colour scale maximum.")


class AverageSummary(_SummaryBase):
    """The scan's averaged frame (a noscan or count scan), drawn with ``figure``."""

    frame_ndim: ClassVar[frozenset[int]] = frozenset({1, 2})

    kind: Literal["average"] = Field(
        "average", description="The scan's averaged frame."
    )


Summary = Annotated[
    Union[ImageGridSummary, WaterfallSummary, AverageSummary],
    Field(discriminator="kind"),
]

#: kind → option model, the frozen set of summary kinds.
SUMMARY_KINDS: Dict[str, type[_SummaryBase]] = {
    "image_grid": ImageGridSummary,
    "waterfall": WaterfallSummary,
    "average": AverageSummary,
}


# ------------------------------------------------------------------ runtime


class RecipeRuntime(SchemaModel):
    """How the recipe runs over a scan."""

    priority: int = Field(
        100,
        ge=0,
        description="Run order within a group: lower runs first. 100 is the background default.",
    )
    average_frames_first: bool = Field(
        False,
        description=(
            "Average each bin's frames before processing and measure once per "
            "bin, for metrics that are not linear in the image; default "
            "measures every frame."
        ),
    )
    save: bool = Field(
        True,
        description=(
            "Write per-shot / per-bin products and the summary figures into "
            "the analysis tree. S-file scalar columns are written regardless."
        ),
    )


# ----------------------------------------------------------------- document


class AnalysisRecipe(VersionedSchemaModel):
    """One device's analysis: input, ordered steps, a measure, the draw, the summaries.

    The file's stem is the recipe ID that groups reference.  ``device`` is
    the device whose folder holds the input files and whose name stems the
    product files; ``output_name`` (defaulting to ``device``) labels
    everything written — s-file columns, the output folder.
    """

    schema_version: int = Field(
        CURRENT_RECIPE_VERSION,
        description=(
            "Format version of this recipe. Leave at 3 — tools update this "
            "automatically when the file format changes."
        ),
    )
    device: str = Field(
        ...,
        min_length=1,
        description="The device whose data folder under scans/ScanNNN/ is analyzed.",
    )
    output_name: Optional[str] = Field(
        None,
        description=(
            "Label for everything this recipe writes (s-file column prefix, "
            "output folder). Defaults to device; set it to run two recipes "
            "over one device with distinct outputs."
        ),
    )
    scalar_suffix: Optional[str] = Field(
        None,
        description="Suffix appended to every s-file column name; scalars only, never files or folders.",
    )
    description: Optional[str] = Field(
        None, description="Free-text note about this recipe."
    )
    # Documentary free-form fields (location, calibration notes): nothing
    # reads them, so they stay an open mapping rather than a model.
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Free-form documentary fields; nothing reads them."
    )
    input: RecipeInput = Field(
        ..., description="Where the frames come from and how one is read."
    )
    inputs: Dict[str, FrameInput] = Field(
        default_factory=dict,
        description=(
            "Frames loaded before the run and bound by name for steps that "
            "take one (a background image)."
        ),
    )
    steps: List[StepRef] = Field(
        default_factory=list,
        description="Processing steps in order; any order, repeats allowed.",
    )
    measure: MeasureRef = Field(
        default_factory=MeasureRef,
        description="What is measured on each processed frame.",
    )
    scan: RecipeRuntime = Field(
        default_factory=RecipeRuntime, description="How the recipe runs over a scan."
    )
    figure: FigureStyle = Field(
        default_factory=FigureStyle,
        description="The per-frame draw, reused by every summary kind.",
    )
    summaries: List[Summary] = Field(
        default_factory=list,
        description=(
            "Scan-level figures, each a frozen kind with its own options; an "
            "empty list draws no summary."
        ),
    )
    _source_id: Optional[str] = PrivateAttr(default=None)

    @property
    def source_id(self) -> Optional[str]:
        """Filename-derived recipe ID when loaded from disk, if known."""
        return self._source_id

    @property
    def effective_output_name(self) -> str:
        """``output_name`` if set, else ``device`` — the one output label."""
        return self.output_name if self.output_name is not None else self.device

    @property
    def input_kind(self) -> str:
        """``"camera"`` or ``"line"``."""
        return self.input.kind

    @model_validator(mode="before")
    @classmethod
    def _refuse_other_versions(cls, data: object) -> object:
        """Refuse a document of another format; nothing is lifted here.

        A v2 diagnostic (``analyzer:`` + ``image:``) is a different document
        read by :class:`AnalysisDiagnostic`; ``load_analysis_document``
        dispatches on ``schema_version``.
        """
        if not isinstance(data, dict):
            return data
        version = declared_schema_version(data)
        if version is not None and version != CURRENT_RECIPE_VERSION:
            raise ValueError(
                f"schema_version {version} is not an analysis recipe (format "
                f"{CURRENT_RECIPE_VERSION}); a v2 diagnostic is read by "
                "AnalysisDiagnostic — use load_analysis_document to dispatch"
            )
        if version is None and ("analyzer" in data or "image" in data):
            raise ValueError(
                "this is a v2 analysis diagnostic (analyzer: + image:), not an "
                "analysis recipe; use load_analysis_document to dispatch"
            )
        return data

    @model_validator(mode="after")
    def _summaries_fit_the_input(self) -> "AnalysisRecipe":
        """Every summary kind must accept the input's dimensionality."""
        ndim = type(self.input).ndim
        for summary in self.summaries:
            if ndim not in summary.frame_ndim:
                raise ValueError(
                    f"summary kind {summary.kind!r} does not draw "
                    f"{self.input.kind} frames ({ndim}D)"
                )
        return self


__all__ = [
    "CURRENT_RECIPE_VERSION",
    "SUMMARY_KINDS",
    "AnalysisRecipe",
    "AverageSummary",
    "CameraInput",
    "FigureStyle",
    "FrameInput",
    "ImageGridSummary",
    "LineInput",
    "MeasureRef",
    "RecipeInput",
    "RecipeRuntime",
    "StepRef",
    "Summary",
    "WaterfallSummary",
    "declared_schema_version",
]
