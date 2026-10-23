"""Image pixels ↔ the analysis ROI's own pixels.

The image (MJPG1) and the ROI's bounds are both in **image** pixels — PROC1's
output, after any transform. STAT's results (centroid, cursor, profiles)
are in the ROI's **output** pixels: offset by ``MinX``, divided by ``BinX``,
and counted from the far edge when ``ReverseX`` is on. Output pixel ``i``
covers image pixels ``MinX + i·BinX … MinX + (i+1)·BinX − 1`` (mirrored when
reversed); a position maps to the centre of what it covers.

One module for both directions, so "set crosshair to centroid" (ROI →
image) and, later, "lineout through a clicked point" (image → ROI) cannot
disagree.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    """One axis of ROI1: its image-pixel bounds, binning and direction."""

    min: int
    size: int
    bin: int = 1
    reverse: bool = False


def to_image(axis: Axis, roi: float) -> float:
    """An ROI output position → image pixels."""
    b = max(axis.bin, 1)
    offset = roi * b + (b - 1) / 2
    if axis.reverse:
        return axis.min + axis.size - 1 - offset
    return axis.min + offset


def to_roi(axis: Axis, image: float) -> float:
    """An image position → the ROI's output pixels (the inverse of :func:`to_image`)."""
    b = max(axis.bin, 1)
    offset = (axis.min + axis.size - 1 - image) if axis.reverse else (image - axis.min)
    return (offset - (b - 1) / 2) / b
