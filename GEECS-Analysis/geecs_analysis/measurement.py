"""Owned measurement values and typed overlay instructions, without rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    A measurement pickles (a worker process returns one to its parent), and
    the copy carries the same scalars, frame, overlays, notes and extras.

    ``extras`` are named auxiliary frames a measure produces beside its main
    frame (the FROG retrieval's temporal and spectral lineouts). They are
    neither drawn nor averaged; a scan host may persist them per shot as the
    measure's registered ``sidecar`` table.
    """

    scalars: Mapping[str, float]
    frame: Frame
    overlays: tuple[Overlay, ...] = ()
    notes: tuple[str, ...] = ()
    extras: Mapping[str, Frame] = field(default_factory=dict)

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
        extras = dict(self.extras)
        for key, value in extras.items():
            if not isinstance(key, str) or not key:
                raise ValueError("Extra keys must be nonempty strings")
            if not isinstance(value, Frame):
                raise TypeError(f"Extra {key} must be a Frame")
        object.__setattr__(self, "scalars", MappingProxyType(values))
        object.__setattr__(self, "overlays", overlays)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "extras", MappingProxyType(extras))

    def __getstate__(self) -> dict:
        """Pickle the owned values as plain containers (a proxy view cannot be)."""
        return {
            "scalars": dict(self.scalars),
            "frame": self.frame,
            "overlays": self.overlays,
            "notes": self.notes,
            "extras": dict(self.extras),
        }

    def __setstate__(self, state: dict) -> None:
        """Restore the read-only view without re-annotating the notes."""
        object.__setattr__(self, "scalars", MappingProxyType(dict(state["scalars"])))
        object.__setattr__(self, "frame", state["frame"])
        object.__setattr__(self, "overlays", tuple(state["overlays"]))
        object.__setattr__(self, "notes", tuple(state["notes"]))
        object.__setattr__(self, "extras", MappingProxyType(dict(state["extras"])))
