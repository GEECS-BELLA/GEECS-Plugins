"""Owned, coordinate-aware samples shared by image and trace consumers.

The order of axes is numpy dimension order: a trace has ``(x,)`` and an
image has ``(y, x)``. Coordinates describe the current samples, not an implied
offset into another array. This module performs no reading or writing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _owned_array(values: ArrayLike) -> NDArray[np.float64]:
    """Copy real numerical samples into a read-only float64 array."""
    source = np.asarray(values)
    if source.dtype.kind not in "biuf":
        raise TypeError("Samples must be real numerical values")
    array = np.array(source, dtype=np.float64, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class Axis:
    """Coordinates of one dimension, copied from the caller and read-only.

    Coordinates must be finite but may descend or be nonuniform. Axis order
    is preserved; constructing an axis never sorts or resamples its data.
    """

    values: NDArray[np.float64]
    unit: str = "px"
    label: str = ""

    def __post_init__(self) -> None:
        """Validate and own the coordinate vector."""
        values = _owned_array(self.values)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("An axis must be a nonempty one-dimensional vector")
        if not np.isfinite(values).all():
            raise ValueError("Axis coordinates must be finite")
        object.__setattr__(self, "values", values)

    def sliced(self, selection: slice) -> Axis:
        """Select coordinates without changing their units or labels."""
        return replace(self, values=self.values[selection])


@dataclass(frozen=True)
class ShotMeta:
    """Read-only shot identity; sources normalize timestamps to Unix seconds."""

    device: str
    shot_number: int | None = None
    acq_timestamp: float | None = None

    def __post_init__(self) -> None:
        """Refuse ambiguous or invalid shot provenance."""
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("Shot provenance requires a device name")
        if self.shot_number is not None and (
            type(self.shot_number) is not int or self.shot_number < 1
        ):
            raise ValueError("Shot number must be a positive integer")
        if self.acq_timestamp is not None and not isfinite(self.acq_timestamp):
            raise ValueError("Acquisition timestamp must be finite")


@dataclass(frozen=True, eq=False)
class Frame:
    """An owned 1D trace or 2D image with one coordinate axis per dimension.

    Processing samples are float64. Non-finite data is retained so a measure
    can explicitly handle invalid samples; coordinates must always be finite.
    Construction copies data, preventing live-source buffer reuse or caller
    mutation from changing an analysis in flight. Arrays are read-only to
    catch accidental in-place processing. ``replace`` creates the next frame.
    """

    data: NDArray[np.float64]
    axes: tuple[Axis, ...]
    shot: ShotMeta | None = None
    unit: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        """Validate the sample geometry and own the data."""
        data = _owned_array(self.data)
        axes = tuple(self.axes)
        if data.ndim not in (1, 2) or data.size == 0:
            raise ValueError("A frame must be a nonempty 1D trace or 2D image")
        if len(axes) != data.ndim or any(not isinstance(axis, Axis) for axis in axes):
            raise ValueError("A frame requires one Axis per data dimension")
        if tuple(axis.values.size for axis in axes) != data.shape:
            raise ValueError("Axis lengths must match the data shape")
        if self.shot is not None and not isinstance(self.shot, ShotMeta):
            raise TypeError("Shot provenance must be ShotMeta or None")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "axes", axes)

    @classmethod
    def from_array(
        cls,
        data: ArrayLike,
        *,
        axes: tuple[Axis, ...] | None = None,
        shot: ShotMeta | None = None,
        unit: str = "",
        label: str = "",
    ) -> Frame:
        """Construct a frame, defaulting to local pixel/sample-index coordinates."""
        array = np.asarray(data)
        if array.ndim not in (1, 2):
            raise ValueError("A frame must be a 1D trace or 2D image")
        if axes is None:
            labels = ("x",) if array.ndim == 1 else ("y", "x")
            axes = tuple(
                Axis(np.arange(size), label=name)
                for size, name in zip(array.shape, labels, strict=True)
            )
        return cls(array, axes, shot, unit, label)

    @classmethod
    def from_trace(
        cls,
        trace: ArrayLike,
        *,
        x_unit: str = "",
        y_unit: str = "",
        x_label: str = "x",
        y_label: str = "",
        shot: ShotMeta | None = None,
    ) -> Frame:
        """Adapt an Nx2 reader result without sorting or altering its coordinates."""
        array = np.asarray(trace)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("A trace reader result must have shape (N, 2)")
        return cls(
            array[:, 1], (Axis(array[:, 0], x_unit, x_label),), shot, y_unit, y_label
        )

    def as_trace(self) -> NDArray[np.float64]:
        """Return an independent Nx2 trace for compatibility with existing readers."""
        if self.data.ndim != 1:
            raise ValueError("Only a 1D frame can be represented as an Nx2 trace")
        return np.column_stack((self.axes[0].values, self.data))

    def replace(
        self, *, data: ArrayLike, axes: tuple[Axis, ...] | None = None
    ) -> Frame:
        """Return new owned samples, retaining axes and provenance unless supplied."""
        return replace(self, data=data, axes=self.axes if axes is None else axes)

    def crop(self, selections: tuple[slice, ...]) -> Frame:
        """Slice samples and coordinates together in numpy dimension order."""
        if len(selections) != self.data.ndim or any(
            not isinstance(selection, slice) for selection in selections
        ):
            raise ValueError("Crop requires one slice per dimension")
        return self.replace(
            data=self.data[selections],
            axes=tuple(
                axis.sliced(selection)
                for axis, selection in zip(self.axes, selections, strict=True)
            ),
        )
