"""Numpy-free figure styling; matplotlib validates its own keyword arguments."""

from typing import Any

from pydantic import Field

from geecs_analysis.registry import SpecModel


class FigureSpec(SpecModel):
    """Matplotlib keyword groups and overlay styles keyed by stable overlay id.

    ``hidden`` and ``scale`` (projection height as a fraction of the image) are
    overlay-only controls; other entries pass to Axes.plot. ``show`` is the
    colorbar visibility control. Geometry belongs to Frame, so image ``extent``
    cannot be overridden. Nonuniform image axes use pcolormesh, not imshow.
    """

    # Any is deliberate: matplotlib accepts strings, sequences, Normalize and
    # colormap objects, etc. Validation happens by rendering the preview.
    imshow: dict[str, Any] = Field(default_factory=dict)
    pcolormesh: dict[str, Any] = Field(default_factory=dict)
    plot: dict[str, Any] = Field(default_factory=dict)
    colorbar: dict[str, Any] = Field(default_factory=dict)
    axes: dict[str, Any] = Field(default_factory=dict)
    fig: dict[str, Any] = Field(default_factory=dict)
    overlays: dict[str, dict[str, Any]] = Field(default_factory=dict)
