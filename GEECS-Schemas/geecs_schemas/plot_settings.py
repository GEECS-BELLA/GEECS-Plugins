"""PlotterSettings — a named settings set of the Data Portal's Plot and Images pages.

``<experiment>/plot_settings/<name>.yaml`` in the configs repository, read
by the portal's plotter (GEECSplotter's pages): every control of the two
pages, each field with its default, so a set saved by an older portal opens
with new controls at their defaults.

Unlike every other document here, these models **keep unknown fields**
instead of refusing them: a portal must load, and save again without
dropping, a set that a newer portal wrote with more fields (a rollback, or
two hosts running different versions against one configs repo). A change of
a field's meaning bumps ``schema_version``.

The choice fields (``tab``, ``image_zscale``, a fit name, a colour map ...)
are plain strings on purpose: a set naming a choice a newer portal added
still loads, and the portal checks the choices where the page posts them.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from geecs_schemas._base import SchemaModel, Tier, VersionedSchemaModel

#: The sets' folder in the experiment's configs folder.
PLOT_SETTINGS_FOLDER = "plot_settings"

#: Y, Y2 and Y3: the most Y lists a set holds.
MAX_Y_LISTS = 3

#: The highest polynomial order a fit takes.
MAX_FIT_ORDER = 10

#: The most image tiles across.
MAX_IMAGE_COLUMNS = 20

#: Cameras stacked on the Images page, the first included.
MAX_CAMERAS = 8

#: The most scan scalars the Images page lists.
MAX_SCALARS = 12

#: The bounds of the strip's max tile width, in CSS pixels.
TILE_MAX_RANGE = (40, 4000)

#: Every model of a set keeps the fields it does not know (the module docstring).
_KEEP_UNKNOWN = ConfigDict(extra="allow")

#: The highest scope channel a trace row may name (from 0).
MAX_CHANNEL = 15


def _one_channel_to_list(data: object, old: str, new: str) -> object:
    """A set saved with one channel (*old*: an int) read as the list *new* holding it.

    Trace rows drew one scope channel each until they overlaid several; a
    set saved then opens on the same channel. A set naming both keeps the
    list and drops the old key.
    """
    if isinstance(data, dict) and old in data:
        data = dict(data)
        one = data.pop(old)
        if new not in data and isinstance(one, int) and not isinstance(one, bool):
            data[new] = [one]
    return data


class ScanRef(SchemaModel):
    """One scan named by its day and its scan number."""

    model_config = ConfigDict(extra="allow", frozen=True)

    day: date = Field(description="The scan's day.")
    scan: int = Field(ge=1, description="The scan's number on that day.")


class ImageCamera(SchemaModel):
    """A camera stacked under the first on the Images page, and how its row looks.

    The first camera's choices are the set's own ``image_*`` fields; each
    further camera keeps the same choices here, under the same names without
    ``image_``. Its region of interest is ``image_rois[device]``, as the
    first camera's is.
    """

    model_config = _KEEP_UNKNOWN

    device: Optional[str] = Field(None, description="The camera's device folder.")
    device_filter: str = Field("", description="The camera picker's filter text.")
    cmap: str = Field("white_blue_yellow", description="The colour map's name.")
    zscale: str = Field(
        "shared", description="The z scale: per tile, shared by the row, or fixed."
    )
    zmin: Optional[float] = Field(None, description="The fixed z scale's black.")
    zmax: Optional[float] = Field(None, description="The fixed z scale's white.")
    hits: str = Field("none", description="The X-ray-hit filter applied to each shot.")
    centre: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Centre each shot on its peak above this threshold (1 or more: "
            "counts; below 1: a fraction of the peak); empty: not centred."
        ),
    )
    cbar: bool = Field(True, description="Draw colour bars.")
    bgsub: bool = Field(True, description="Subtract the set's background scan.")
    channels: list[int] = Field(
        default_factory=lambda: [0],
        min_length=1,
        max_length=MAX_CHANNEL + 1,
        description=(
            "The row's scope channels (from 0), overlaid on each trace tile; "
            "a device with one channel reads it whatever is named."
        ),
    )
    xmin: Optional[float] = Field(
        None, description="A trace row's x window, low side; empty: no limit."
    )
    xmax: Optional[float] = Field(
        None, description="A trace row's x window, high side; empty: no limit."
    )

    @model_validator(mode="before")
    @classmethod
    def _old_channel(cls, data: object) -> object:
        """An older set's one ``channel`` as ``channels``."""
        return _one_channel_to_list(data, "channel", "channels")


class PlotterSettings(VersionedSchemaModel):
    """Every control of the Plot and Images pages, each with its default.

    A set follows its viewer from scan to scan: the scan last opened under it
    (``day``, ``scan``) is where it reopens, and the axes, fits, filters and
    image choices apply to whichever scan is open.
    """

    model_config = _KEEP_UNKNOWN

    tier: ClassVar[Tier] = "operator"

    schema_version: Literal[1] = Field(1, description="Format version. Leave at 1.")

    @model_validator(mode="before")
    @classmethod
    def _old_channel(cls, data: object) -> object:
        """An older set's one ``image_channel`` as ``image_channels``."""
        return _one_channel_to_list(data, "image_channel", "image_channels")

    x: Optional[str] = Field(
        None, description="The X column; empty: the scan variable; '' the shot number."
    )
    y: Optional[str] = Field(
        None, description="Y1's column; empty: the first measurement."
    )
    y2: Optional[str] = Field(
        None, description="Y2's column, on its own axis; empty: not picked."
    )
    y3: Optional[str] = Field(
        None, description="Y3's column, on its own axis; empty: not picked."
    )
    y_count: int = Field(
        1,
        ge=1,
        le=MAX_Y_LISTS,
        description="How many of Y1, Y2, Y3 are shown; the others keep their picks.",
    )
    x_filter: str = Field("", description="The X list's filter text.")
    y_filter: str = Field("", description="Y1's list filter text.")
    y2_filter: str = Field("", description="Y2's list filter text.")
    y3_filter: str = Field("", description="Y3's list filter text.")
    x_alias: bool = Field(True, description="Show X's columns by their aliases.")
    y_alias: bool = Field(True, description="Show Y1's columns by their aliases.")
    y2_alias: bool = Field(True, description="Show Y2's columns by their aliases.")
    y3_alias: bool = Field(True, description="Show Y3's columns by their aliases.")
    x_auto: bool = Field(
        True,
        description="X follows the open scan's variable (GEECSplotter's autofill par1).",
    )
    y_fit: str = Field("none", description="Y1's fit ('none': no fit).")
    y2_fit: str = Field("none", description="Y2's fit ('none': no fit).")
    y3_fit: str = Field("none", description="Y3's fit ('none': no fit).")
    y_fit_order: int = Field(
        2, ge=0, le=MAX_FIT_ORDER, description="Y1's polynomial fit order."
    )
    y2_fit_order: int = Field(
        2, ge=0, le=MAX_FIT_ORDER, description="Y2's polynomial fit order."
    )
    y3_fit_order: int = Field(
        2, ge=0, le=MAX_FIT_ORDER, description="Y3's polynomial fit order."
    )
    day: Optional[date] = Field(None, description="The day of the scan last opened.")
    scan: Optional[int] = Field(None, description="The number of the scan last opened.")
    extra_scans: list[ScanRef] = Field(
        default_factory=list,
        description="Scans plotted beside the main one, from any day.",
    )
    scan_view: str = Field(
        "overlay",
        description="Several scans: one trace each ('overlay') or pooled ('merged').",
    )
    overlay_y: int = Field(
        1, ge=1, le=MAX_Y_LISTS, description="Which Y list the overlay draws (1-3)."
    )
    tab: str = Field(
        "xy", description="The Plot page's view: the XY plot or the 2-D map."
    )
    image_device: Optional[str] = Field(
        None, description="The first camera; empty: the scan's first."
    )
    image_device_filter: str = Field("", description="The camera picker's filter text.")
    image_strip: str = Field(
        "bins", description="One tile per bin, per shot, or one picked shot per bin."
    )
    image_pick: str = Field(
        "first", description="The picked shot of a bin: its first, or highest Y1."
    )
    image_hits: str = Field(
        "none", description="The X-ray-hit filter applied to each shot."
    )
    image_centre: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Centre each shot on its peak above this threshold (1 or more: "
            "counts; below 1: a fraction of the peak); empty: not centred."
        ),
    )
    image_layout: str = Field("grid", description="The strip as a grid or one row.")
    image_columns: int = Field(
        5, ge=1, le=MAX_IMAGE_COLUMNS, description="Tiles across."
    )
    image_cmap: str = Field("white_blue_yellow", description="The colour map's name.")
    image_zscale: str = Field(
        "shared", description="The z scale: per tile, shared by the row, or fixed."
    )
    image_zmin: Optional[float] = Field(None, description="The fixed z scale's black.")
    image_zmax: Optional[float] = Field(None, description="The fixed z scale's white.")
    image_cbar: bool = Field(True, description="Draw colour bars.")
    image_bg: Optional[ScanRef] = Field(
        None,
        description="The background scan, whose average comes off each shot; empty: none.",
    )
    image_bgsub: bool = Field(
        True, description="The first camera subtracts the background."
    )
    image_channels: list[int] = Field(
        default_factory=lambda: [0],
        min_length=1,
        max_length=MAX_CHANNEL + 1,
        description=(
            "The first row's scope channels (from 0), overlaid on each trace tile; "
            "a device with one channel reads it whatever is named."
        ),
    )
    image_xmin: Optional[float] = Field(
        None, description="The first trace row's x window, low side; empty: no limit."
    )
    image_xmax: Optional[float] = Field(
        None, description="The first trace row's x window, high side; empty: no limit."
    )
    image_rois: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Each camera's region of interest, [x0, y0, x1, y1] in pixels.",
    )
    image_more: list[ImageCamera] = Field(
        default_factory=list,
        max_length=MAX_CAMERAS - 1,
        description="The cameras stacked under the first, one row each.",
    )
    image_scalars: list[str] = Field(
        default_factory=list,
        max_length=MAX_SCALARS,
        description="The scan scalars summarized on the Images page, in order.",
    )
    image_tile_max: Optional[int] = Field(
        None,
        ge=TILE_MAX_RANGE[0],
        le=TILE_MAX_RANGE[1],
        description="The widest a tile is drawn, in CSS pixels; empty: no limit.",
    )
    image_scalar_stat: str = Field(
        "mean_std", description="The statistic the scan scalars share."
    )
    image_scalar_filter: str = Field("", description="The scalar picker's filter text.")
    merge_tolerance: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Merged scans: bins this close in X are one bin; empty: half the "
            "smallest step of any one scan."
        ),
    )
    xy_view: str = Field("shot", description="The XY plot per shot or binned.")
    shot_filters: str = Field("", description="The shot filters, as JSON; empty: none.")
    bin_config: str = Field(
        "", description="The bin settings, as JSON; empty: defaults."
    )
    display: str = Field(
        "", description="The plot cosmetics, as JSON; empty: defaults."
    )
    map_value: Optional[str] = Field(
        None, description="The 2-D map's scalar; empty: Y1."
    )
    map_x: Optional[str] = Field(
        None, description="The 2-D map's X; empty: the scan's first variable."
    )
    map_y: Optional[str] = Field(
        None, description="The 2-D map's Y; empty: the scan's second variable."
    )
    map_filter: str = Field("", description="The map scalar picker's filter text.")
    map_config: str = Field(
        "", description="The 2-D map's statistics, as JSON; empty: defaults."
    )
