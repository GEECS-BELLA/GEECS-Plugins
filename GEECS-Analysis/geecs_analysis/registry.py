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
    """A measurement spec, function and supported dimensions."""

    spec: type[MeasureSpec]
    function: Callable[[Frame, MeasureSpec], Measurement]
    ndim: frozenset[int]


_MEASURES: dict[type[MeasureSpec], MeasureDefinition] = {}
MeasureT = TypeVar("MeasureT", bound=MeasureSpec)


def measure(
    spec: type[MeasureT], *, ndim: set[int]
) -> Callable[
    [Callable[[Frame, MeasureT], Measurement]], Callable[[Frame, MeasureT], Measurement]
]:
    """Register a builtin measure before constructing the spec union."""
    if not ndim or not ndim <= {1, 2}:
        raise ValueError("Measure dimensions must be a nonempty subset of {1, 2}")

    def register(function: Callable[[Frame, MeasureT], Measurement]):
        if spec in _MEASURES:
            raise ValueError(f"Measure spec already registered: {spec.__name__}")
        _MEASURES[spec] = MeasureDefinition(spec, function, frozenset(ndim))
        return function

    return register


def measure_definitions() -> tuple[MeasureDefinition, ...]:
    """Return the builtin measurement declarations in registration order."""
    return tuple(_MEASURES.values())


def measure_definition(spec: MeasureSpec) -> MeasureDefinition:
    """Look up numerical execution for a validated measurement spec."""
    return _MEASURES[type(spec)]
