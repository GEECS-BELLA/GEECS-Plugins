"""Builtin declarations without importing any numerical execution dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, TypeVar

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class StepSpec(BaseModel):
    """Immutable processing parameters; typos and nonfinite values are errors."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


SpecT = TypeVar("SpecT", bound=StepSpec)


@dataclass(frozen=True)
class StepDefinition:
    """A spec paired with its numerical function and supported dimensions."""

    spec: type[StepSpec]
    function: Callable[[Frame, StepSpec], Frame]
    ndim: frozenset[int]


_STEPS: dict[type[StepSpec], StepDefinition] = {}


def step(
    spec: type[SpecT], *, ndim: set[int]
) -> Callable[[Callable[[Frame, SpecT], Frame]], Callable[[Frame, SpecT], Frame]]:
    """Register a builtin spec/function pair before the schema union is built."""
    if not ndim or not ndim <= {1, 2}:
        raise ValueError("Step dimensions must be a nonempty subset of {1, 2}")

    def register(function: Callable[[Frame, SpecT], Frame]):
        if spec in _STEPS:
            raise ValueError(f"Step spec already registered: {spec.__name__}")
        _STEPS[spec] = StepDefinition(spec, function, frozenset(ndim))
        return function

    return register


def definitions() -> tuple[StepDefinition, ...]:
    """Return a stable snapshot of registered builtins in declaration order."""
    return tuple(_STEPS.values())


def definition(spec: StepSpec) -> StepDefinition:
    """Look up the function for a validated spec without string dispatch."""
    return _STEPS[type(spec)]
