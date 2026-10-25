"""A device's arrays, read on request: the reply ``GET /api/devices/{name}/arrays`` sends.

Arrays (a camera's profiles and histogram, a scope's capture) are never
monitored (Appendix A rule 3): monitored, they would arrive at the
producer's full rate. The bridge reads a device's ``on_request`` roles
together, at most once per interval and shared by every caller
(:class:`~geecs_panels.feed.ArraysSource`), and sends each as a
min/max-decimated trace (:func:`geecs_panels.scope.decimate`), so a
one-sample spike survives the reduction to a screen's width.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from geecs_panels.contracts import _finite
from geecs_panels.scope import decimate

#: The most columns a trace is decimated to (two points each).
MAX_COLUMNS = 4000


class Trace(BaseModel):
    """One array as a page draws it: x and y, at most two points per screen column."""

    model_config = ConfigDict(extra="forbid")

    x: list[float] = Field(description="Sample index, or time for a scope trace (s).")
    y: list[Optional[float]] = Field(description="The values; null where not finite.")

    @field_validator("y", mode="before")
    @classmethod
    def _finite_y(cls, y: object) -> object:
        return [_finite(v) for v in y] if isinstance(y, list) else y


class ArraysReply(BaseModel):
    """``GET /api/devices/{name}/arrays``: the device's arrays, read on request and decimated."""

    model_config = ConfigDict(extra="forbid")

    device: str = Field(description="The device.")
    traces: dict[str, Trace] = Field(
        default_factory=dict,
        description=(
            "Role (or, for a scope, channel) → its trace. A camera's profiles are "
            "in the analysis ROI's output pixels, cut to their valid length."
        ),
    )
    error: Optional[str] = Field(
        None, description="Why the arrays could not be read (the page says so)."
    )


def trace(values: list[float], columns: int, x: Optional[list[float]] = None) -> Trace:
    """*values* against *x* (default: the index), min/max-decimated to *columns*."""
    xs = x if x is not None else [float(i) for i in range(len(values))]
    tx, ty = decimate(xs, values, columns)
    return Trace(x=tx, y=ty)
