"""Owned measurement values and typed overlay instructions, without rendering."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping

from geecs_data_utils.frames import Frame


@dataclass(frozen=True)
class Projection:
    """A 1D projection along an image dimension, identified for renderer styling."""

    id: str
    axis: int
    frame: Frame

    def __post_init__(self) -> None:
        """Require a trace and a valid image dimension."""
        if self.frame.data.ndim != 1 or self.axis not in (0, 1):
            raise ValueError("Projection requires a 1D frame and axis 0 or 1")


@dataclass(frozen=True)
class Marker:
    """A point in the source frame's physical coordinate system."""

    id: str
    x: float
    y: float


Overlay = Projection | Marker


@dataclass(frozen=True)
class Measurement:
    """Bare scalar keys, processed frame, typed overlays and explicit validity notes.

    Nonfinite scalars are retained as the algorithms emit them. Notes expose
    those undefined results to consumers instead of turning them into zeros.
    Caller dictionaries and sequences are copied to immutable containers.
    """

    scalars: Mapping[str, float]
    frame: Frame
    overlays: tuple[Overlay, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Own the results and annotate invalid scalars without hiding them."""
        if not isinstance(self.frame, Frame):
            raise TypeError("Measurement requires a Frame")
        values = {}
        for key, value in self.scalars.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Scalar keys must be nonempty strings")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Scalar {key} must be real numerical data")
            values[key] = float(value)
        overlays = tuple(self.overlays)
        if any(not isinstance(o, (Projection, Marker)) for o in overlays):
            raise TypeError("Unsupported overlay type")
        if len({o.id for o in overlays}) != len(overlays):
            raise ValueError("Overlay ids must be unique within a measurement")
        notes = tuple(self.notes) + tuple(
            f"Nonfinite scalar: {key}"
            for key, value in values.items()
            if not isfinite(value)
        )
        object.__setattr__(self, "scalars", MappingProxyType(values))
        object.__setattr__(self, "overlays", overlays)
        object.__setattr__(self, "notes", notes)
