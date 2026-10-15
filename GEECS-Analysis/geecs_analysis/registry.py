"""Builtin declarations without importing any numerical execution dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, ClassVar, Mapping, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from matplotlib.figure import Figure
    from geecs_analysis.measurement import Measurement


class SpecModel(BaseModel):
    """Immutable parameters; typos and nonfinite values are errors."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class StepSpec(SpecModel):
    """Base of processing declarations."""


class MeasureSpec(SpecModel):
    """Base of measurement declarations with numpy-free scalar discovery."""

    #: What each scalar the measure can emit means, keyed by its bare name:
    #: one plain sentence with its units. The recipe reference and the config
    #: editor show these (via ``recipe_schema``'s ``x-scalars``); a test pins
    #: that every key ``emitted_scalars`` can return has an entry.
    scalar_docs: ClassVar[Mapping[str, str]] = {}

    def emitted_scalars(self) -> frozenset[str]:
        """Return every key this measure can emit for the configured options."""
        raise NotImplementedError


SpecT = TypeVar("SpecT", bound=StepSpec)


@dataclass(frozen=True)
class StepDefinition:
    """A spec paired with its numerical function and supported dimensions."""

    spec: type[StepSpec]
    # Ordinary steps take (frame, spec); input-bound steps additionally take
    # one already-loaded Frame. The declaration below selects that signature.
    function: Callable[..., Frame]
    ndim: frozenset[int]
    input_field: str | None = None


_STEPS: dict[type[StepSpec], StepDefinition] = {}


def step(
    spec: type[SpecT], *, ndim: set[int], input_field: str | None = None
) -> Callable[[Callable[..., Frame]], Callable[..., Frame]]:
    """Register a builtin spec/function pair before the schema union is built."""
    if not ndim or not ndim <= {1, 2}:
        raise ValueError("Step dimensions must be a nonempty subset of {1, 2}")
    if input_field is not None and (
        input_field not in spec.model_fields
        or spec.model_fields[input_field].annotation is not str
    ):
        raise ValueError("An input field must name a string field on the step spec")

    def register(function: Callable[..., Frame]):
        if spec in _STEPS:
            raise ValueError(f"Step spec already registered: {spec.__name__}")
        _STEPS[spec] = StepDefinition(spec, function, frozenset(ndim), input_field)
        return function

    return register


def definitions() -> tuple[StepDefinition, ...]:
    """Return a stable snapshot of registered builtins in declaration order."""
    return tuple(_STEPS.values())


def definition(spec: StepSpec) -> StepDefinition:
    """Look up the function for a validated spec without string dispatch."""
    return _STEPS[type(spec)]


@dataclass(frozen=True)
class MeasureDefinition:
    """A measurement spec, function and supported dimensions.

    ``service`` names a host-supplied collaborator the measure calls (the
    FROG retrieval and the HASO WaveKit engine, external programs the core
    may not start); the host binds it in ``inputs`` under that key and the
    function then takes ``(frame, spec, service)``. ``sidecar`` names the
    per-shot table a scan host writes beside each shot from the
    measurement's 1D ``extras`` (``None``: none). ``shot_store`` names the
    per-scan store a scan host writes under the analysis tree from every
    single-shot measurement's frame and extras (``None``: none) — the
    ``haso`` measure's wavefront products, one HDF5 per scan.

    ``input_field`` names an optional string field of the spec holding a
    frame-binding key: a comparison frame of the measured kind (the
    ``haso`` measure's reference). The recipe declares it in ``inputs``
    like a step's background; the evaluator processes it through the same
    steps as the measured frame and hands it to the function as the last
    argument — ``None`` when the field is unset.

    ``channels`` names the line input's auxiliary channels the measure
    reads beside the trace (``frog_spectral_phase``'s ``weights``): columns
    the input's loading declares by name (``auxiliary_columns``), loaded
    with the trace, cut by the same sample selection, and handed to the
    function as a ``{name: Frame}`` mapping after every other argument —
    holding only the channels the input supplies.

    ``tables`` names the optional string fields of the spec that hold
    table-binding keys (the ``magspec`` measure's calibration surveys): the
    recipe declares each key under ``inputs`` as a ``table`` the scan host
    reads into a ``geecs_data_utils.tables.Table``, and the function
    receives them as keyword arguments named by the field, ``None`` for a
    field the spec leaves unset.

    ``shot_scalars`` names the optional string fields of the spec that hold
    a column name of the scan's recorded scalars (a teslameter's field read
    per shot): the host binds that column's value for each shot on its
    ``ShotMeta.scalars``, and the function reads it from the frame's shot —
    nothing is passed, so a shot that lacks the column is the measure's own
    fallback case.
    """

    spec: type[MeasureSpec]
    function: Callable[..., Measurement]
    ndim: frozenset[int]
    service: str | None = None
    sidecar: str | None = None
    shot_store: str | None = None
    input_field: str | None = None
    channels: tuple[str, ...] = ()
    tables: tuple[str, ...] = ()
    shot_scalars: tuple[str, ...] = ()


_MEASURES: dict[type[MeasureSpec], MeasureDefinition] = {}
MeasureT = TypeVar("MeasureT", bound=MeasureSpec)


def _optional_string_field(spec: type[MeasureSpec], name: str) -> bool:
    """Whether ``name`` is an ``Optional[str]`` field of the spec (a binding slot)."""
    return name in spec.model_fields and spec.model_fields[name].annotation in (
        Optional[str],
        str | None,
    )


def _check_measure_fields(
    spec: type[MeasureSpec],
    input_field: str | None,
    tables: tuple[str, ...],
    shot_scalars: tuple[str, ...],
) -> None:
    """The binding slots a measure declares are distinct optional string fields."""
    if input_field is not None and not _optional_string_field(spec, input_field):
        raise ValueError(
            "A measure's input field must name an optional string field on its spec"
        )
    for label, names in (("tables", tables), ("shot_scalars", shot_scalars)):
        if len(set(names)) != len(names) or not all(
            _optional_string_field(spec, name) for name in names
        ):
            raise ValueError(
                f"A measure's {label} must name distinct optional string fields "
                "on its spec"
            )
    if input_field is not None and input_field in tables:
        raise ValueError("A measure's input field cannot also be a table field")


def measure(
    spec: type[MeasureT],
    *,
    ndim: set[int],
    service: str | None = None,
    sidecar: str | None = None,
    shot_store: str | None = None,
    input_field: str | None = None,
    channels: tuple[str, ...] = (),
    tables: tuple[str, ...] = (),
    shot_scalars: tuple[str, ...] = (),
) -> Callable[[Callable[..., Measurement]], Callable[..., Measurement]]:
    """Register a builtin measure before constructing the spec union.

    A measure with a ``service`` receives the host's bound collaborator as a
    third argument; ``sidecar`` and ``shot_store`` name what a scan host
    persists per shot; ``input_field`` names the spec field that binds an
    optional comparison frame; ``channels`` the trace's auxiliary channels
    it reads; ``tables`` the spec fields that bind tables; ``shot_scalars``
    the spec fields naming recorded columns read per shot; see
    :class:`MeasureDefinition`.
    """
    if not ndim or not ndim <= {1, 2}:
        raise ValueError("Measure dimensions must be a nonempty subset of {1, 2}")
    for label, value in (
        ("service", service),
        ("sidecar", sidecar),
        ("shot_store", shot_store),
    ):
        if value is not None and not value:
            raise ValueError(f"A measure's {label} must be a nonempty name")
    tables = tuple(tables)
    shot_scalars = tuple(shot_scalars)
    _check_measure_fields(spec, input_field, tables, shot_scalars)
    channels = tuple(channels)
    if channels and (
        ndim != {1} or not all(channels) or len(set(channels)) != len(channels)
    ):
        raise ValueError("A measure's channels are distinct names on a trace measure")

    def register(function: Callable[..., Measurement]):
        if spec in _MEASURES:
            raise ValueError(f"Measure spec already registered: {spec.__name__}")
        _MEASURES[spec] = MeasureDefinition(
            spec,
            function,
            frozenset(ndim),
            service,
            sidecar,
            shot_store,
            input_field,
            channels,
            tables,
            shot_scalars,
        )
        return function

    return register


def measure_definitions() -> tuple[MeasureDefinition, ...]:
    """Return the builtin measurement declarations in registration order."""
    return tuple(_MEASURES.values())


def measure_definition(spec: MeasureSpec) -> MeasureDefinition:
    """Look up numerical execution for a validated measurement spec."""
    return _MEASURES[type(spec)]


@dataclass(frozen=True)
class SummaryOutput:
    """What a summary layout produced: its figure plus scan-level numbers.

    ``scalars`` are numbers computed across the whole scan (a fit's zero
    crossing), keyed by name; nonfinite values stay as they are, explained
    by ``notes``. A layout with no numbers returns a bare ``Figure``;
    :func:`summary_output` turns either shape into this one.
    """

    figure: Figure
    scalars: Mapping[str, float] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Own the numbers: a read-only copy of floats, notes as a tuple."""
        scalars = {str(k): float(v) for k, v in self.scalars.items()}
        object.__setattr__(self, "scalars", MappingProxyType(scalars))
        object.__setattr__(self, "notes", tuple(self.notes))


def summary_output(value: object) -> SummaryOutput:
    """Normalise a summary layout's return value to a :class:`SummaryOutput`.

    A layout returns either a matplotlib ``Figure`` (every picture-only
    kind) or a ``SummaryOutput``; consumers call this once instead of
    asking which.
    """
    if isinstance(value, SummaryOutput):
        return value
    return SummaryOutput(figure=value)


@dataclass(frozen=True)
class SummaryDefinition:
    """A summary kind: its option model, layout function, inputs and file marker.

    The layout returns a ``Figure`` or a :class:`SummaryOutput` (a figure
    plus scan-level scalars the sink writes as a JSON sidecar); consumers
    normalise through :func:`summary_output`. ``consumes`` names the scan product the layout draws: ``"panels"`` (one
    measurement per bin, or per shot on a noscan waterfall) or
    ``"average"`` (the scan's one averaged measurement). ``filename`` is the
    marker the sink appends to the device name; the portal's filename
    parser and MCP's display-file contract read these markers.
    """

    spec: type
    function: Callable[..., object]
    ndim: frozenset[int]
    consumes: str
    filename: str


_SUMMARIES: dict[type, SummaryDefinition] = {}


def summary(
    spec: type, *, consumes: str, filename: str
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Register a summary kind's layout function for its schema option model.

    ``spec`` is one of the frozen kinds in ``geecs_schemas.analysis.recipe``
    (its ``frame_ndim`` says which frames it draws); the registry is what
    the sink and the editor enumerate, so adding a kind is one file here
    plus its option model there.
    """
    ndim = getattr(spec, "frame_ndim", None)
    if not ndim or not frozenset(ndim) <= {1, 2}:
        raise ValueError("A summary spec must declare frame_ndim within {1, 2}")
    if consumes not in {"panels", "average"}:
        raise ValueError("A summary consumes 'panels' or 'average'")
    if not filename or "/" in filename or "\\" in filename:
        raise ValueError("A summary needs a single-component filename marker")

    def register(function: Callable[..., object]):
        if spec in _SUMMARIES:
            raise ValueError(f"Summary spec already registered: {spec.__name__}")
        _SUMMARIES[spec] = SummaryDefinition(
            spec, function, frozenset(ndim), consumes, filename
        )
        return function

    return register


def summary_definitions() -> tuple[SummaryDefinition, ...]:
    """Return the registered summary kinds in registration order."""
    return tuple(_SUMMARIES.values())


def summary_definition(spec: object) -> SummaryDefinition:
    """Look up the layout for a validated summary option model."""
    return _SUMMARIES[type(spec)]
