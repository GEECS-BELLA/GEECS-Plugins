"""Set samples strictly below the level to zero."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class ZeroBelowSpec(StepSpec):
    """Set samples strictly below the level to zero."""

    step: Literal["zero_below"] = "zero_below"
    level: float = Field(
        ..., description="Samples strictly below this level become zero."
    )


@step(ZeroBelowSpec, ndim={1, 2})
def zero_below(frame: Frame, spec: ZeroBelowSpec) -> Frame:
    """Set samples strictly below the level to zero."""
    import numpy as np

    return frame.replace(data=np.where(frame.data >= spec.level, frame.data, 0.0))
