"""Numpy-free numerical recipe schemas assembled from builtin declarations."""

from typing import Annotated, Union

from pydantic import Field

from geecs_analysis import measures as _measures  # noqa: F401 -- registers measure specs
from geecs_analysis import steps as _builtins  # noqa: F401 -- registers step specs
from geecs_analysis.measures.none import NoneSpec
from geecs_analysis.registry import StepSpec, definitions, measure_definitions

Step = Annotated[
    Union[tuple(item.spec for item in definitions())], Field(discriminator="step")
]
Measure = Annotated[
    Union[tuple(item.spec for item in measure_definitions())],
    Field(discriminator="kind"),
]


class Pipeline(StepSpec):
    """Ordered, repeatable processing steps; an empty sequence is a no-op."""

    steps: tuple[Step, ...] = ()


class Analysis(Pipeline):
    """Ephemeral numerical recipe; sources, rendering and sinks are separate."""

    measure: Measure = Field(default_factory=NoneSpec)
