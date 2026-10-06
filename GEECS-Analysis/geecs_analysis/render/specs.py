"""Numpy-free figure styling; matplotlib validates its own keyword arguments."""

from typing import Optional

from pydantic import ConfigDict, Field
from geecs_schemas.analysis.recipe import FigureStyle


class FigureSpec(FigureStyle):
    """The recipe's ``figure:`` block as the renderer consumes it: immutable.

    The field list is the schema's (``FigureStyle``): matplotlib keyword
    groups and overlay styles keyed by stable overlay id. ``hidden`` and
    ``scale`` (projection height as a fraction of the image) are overlay-only
    controls; other entries pass to Axes.plot. ``show`` is the colorbar
    visibility control. Geometry belongs to Frame, so image ``extent``
    cannot be overridden. Nonuniform image axes use pcolormesh, not imshow.

    ``fit_canvas`` is the renderer's own control, not a recipe field: whether
    ``single`` trims a fixed-aspect image's canvas to what was drawn. Unset,
    it follows ``fig``: fitted when no ``figsize`` is named, an explicit
    ``(width, height)`` drawn as given. A translation whose figsize is a side
    length rather than a chosen shape (the v2 renderer's ``figsize_inches``)
    sets it, so every consumer of the spec draws the same canvas.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    fit_canvas: Optional[bool] = Field(
        None,
        description=(
            "Trim a fixed-aspect image's canvas to the drawn content; "
            "unset: only when fig names no figsize."
        ),
    )
