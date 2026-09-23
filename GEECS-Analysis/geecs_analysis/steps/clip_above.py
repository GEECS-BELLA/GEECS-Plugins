"""Cap samples at the specified level."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from geecs_analysis.registry import StepSpec, step

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame


class ClipAboveSpec(StepSpec):
    """Cap samples at the specified level."""

    step: Literal["clip_above"] = "clip_above"
    level: float


@step(ClipAboveSpec, ndim={1, 2})
def clip_above(frame: Frame, spec: ClipAboveSpec) -> Frame:
    """Cap samples at the specified level."""
    import numpy as np

    return frame.replace(data=np.minimum(frame.data, spec.level))
