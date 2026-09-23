"""Preprocessing-only measurement for standard images and traces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from geecs_analysis.registry import MeasureSpec, measure

if TYPE_CHECKING:
    from geecs_data_utils.frames import Frame
    from geecs_analysis.measurement import Measurement


class NoneSpec(MeasureSpec):
    """Keep the processed frame without computing scalars."""

    kind: Literal["none"] = "none"

    def emitted_scalars(self) -> frozenset[str]:
        """Preprocessing-only recipes emit no scalar measurements."""
        return frozenset()


@measure(NoneSpec, ndim={1, 2})
def none(frame: Frame, spec: NoneSpec) -> Measurement:
    """Return the processed frame unchanged."""
    from geecs_analysis.measurement import Measurement

    return Measurement(scalars={}, frame=frame)
