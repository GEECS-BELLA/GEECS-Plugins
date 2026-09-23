"""Numpy-free pipeline schemas assembled from the builtin declarations."""

from typing import Annotated, Union

from pydantic import Field

from geecs_analysis import steps as _builtins  # noqa: F401 -- registers builtin specs
from geecs_analysis.registry import StepSpec, definitions

Step = Annotated[
    Union[tuple(item.spec for item in definitions())], Field(discriminator="step")
]


class Pipeline(StepSpec):
    """Ordered, repeatable processing steps; an empty sequence is a no-op."""

    steps: tuple[Step, ...] = ()
