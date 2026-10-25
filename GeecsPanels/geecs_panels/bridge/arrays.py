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

from collections.abc import Awaitable, Callable
from typing import Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from geecs_panels.contracts import _finite
from geecs_panels.scope import decimate

#: The most columns a trace is decimated to (two points each).
MAX_COLUMNS = 4000
#: A device's arrays are read at most this often, shared by every caller.
EVERY_S = 0.5

#: pvs -> their values (an array, or a scalar), read in one request; raises on failure.
ReadPvs = Callable[[list[str]], Awaitable[list[object]]]


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
    waiting: Optional[str] = Field(
        None,
        description=(
            "Why no traces are shown though the read worked: a scope whose timebase "
            "changed after its held capture waits for a capture at the new one."
        ),
    )


class ArraysReader(Protocol):
    """A device's arrays, read on request: one per device, made by its template's rules."""

    async def reply(self, columns: int) -> ArraysReply:
        """Read (or reuse a read within the interval) and answer, never raising for a failed read."""


def trace(values: list[float], columns: int, x: Optional[list[float]] = None) -> Trace:
    """*values* against *x* (default: the index), min/max-decimated to *columns*."""
    xs = x if x is not None else [float(i) for i in range(len(values))]
    tx, ty = decimate(xs, values, columns)
    return Trace(x=tx, y=ty)


def floats(value: object) -> list[float]:
    """An array read (or an on-request scalar) as floats."""
    if isinstance(value, (int, float)):
        return [float(value)]
    try:
        return [float(x) for x in value]  # type: ignore[union-attr]  # an array
    except TypeError:
        return []
