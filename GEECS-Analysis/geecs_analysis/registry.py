"""Builtin declarations without importing any numerical execution dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, TypeVar

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement


class SpecModel(BaseModel):
    """Immutable parameters; typos and nonfinite values are errors."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class StepSpec(SpecModel):
    """Base of processing declarations."""


class MeasureSpec(SpecModel):
    """Base of measurement declarations with numpy-free scalar discovery."""

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
    """

    spec: type[MeasureSpec]
    function: Callable[..., Measurement]
    ndim: frozenset[int]
    service: str | None = None
    sidecar: str | None = None
    shot_store: str | None = None


_MEASURES: dict[type[MeasureSpec], MeasureDefinition] = {}
MeasureT = TypeVar("MeasureT", bound=MeasureSpec)


def measure(
    spec: type[MeasureT],
    *,
    ndim: set[int],
    service: str | None = None,
    sidecar: str | None = None,
    shot_store: str | None = None,
) -> Callable[[Callable[..., Measurement]], Callable[..., Measurement]]:
    """Register a builtin measure before constructing the spec union.

    A measure with a ``service`` receives the host's bound collaborator as a
    third argument; ``sidecar`` and ``shot_store`` name what a scan host
    persists per shot; see :class:`MeasureDefinition`.
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

    def register(function: Callable[..., Measurement]):
        if spec in _MEASURES:
            raise ValueError(f"Measure spec already registered: {spec.__name__}")
        _MEASURES[spec] = MeasureDefinition(
            spec, function, frozenset(ndim), service, sidecar, shot_store
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
class SummaryDefinition:
    """A summary kind: its option model, layout function, inputs and file marker.

    ``consumes`` names the scan product the layout draws: ``"panels"`` (one
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
