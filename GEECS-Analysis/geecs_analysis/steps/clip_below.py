"""Floor samples at the specified level."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class ClipBelowSpec(StepSpec):
    """Floor samples at the specified level."""

    step: Literal["clip_below"] = "clip_below"
    level: float


@step(ClipBelowSpec, ndim={1, 2})
def clip_below(frame: Frame, spec: ClipBelowSpec) -> Frame:
    """Floor samples at the specified level."""
    import numpy as np

    return frame.replace(data=np.maximum(frame.data, spec.level))
