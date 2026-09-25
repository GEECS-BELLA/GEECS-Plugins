"""Numpy-free figure styling; matplotlib validates its own keyword arguments."""

from pydantic import ConfigDict
from geecs_schemas.analysis.recipe import FigureStyle


class FigureSpec(FigureStyle):
    """The recipe's ``figure:`` block as the renderer consumes it: immutable.

    The field list is the schema's (``FigureStyle``): matplotlib keyword
    groups and overlay styles keyed by stable overlay id. ``hidden`` and
    ``scale`` (projection height as a fraction of the image) are overlay-only
    controls; other entries pass to Axes.plot. ``show`` is the colorbar
    visibility control. Geometry belongs to Frame, so image ``extent``
    cannot be overridden. Nonuniform image axes use pcolormesh, not imshow.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
