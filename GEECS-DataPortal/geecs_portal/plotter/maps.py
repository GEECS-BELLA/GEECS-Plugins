"""The 2-D map's columns, its config, and the snippet that rebuilds it.

The map is the portal's old Grid tab on the Plot page: one scalar's average
and error in each cell of two scan variables, the numbers from
``geecs_data_utils.scan_grid.grid_scan``. The set names the three columns
(``map_value``, ``map_x``, ``map_y``; ``None`` is the default) apart from
the statistics (``map_config``), so a set that follows the viewer to a scan
with other variables maps that scan's own axes.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from geecs_data_utils.data.row_filters import RowFilters
from geecs_data_utils.scan_grid import GridConfig, grid_axes
from geecs_data_utils.tiled_catalog import RunDetail

from geecs_portal import analysis
from geecs_portal.plotter.settings import PlotterSettings

#: The column fields of ``GridConfig``: the set names them on their own.
_COLUMNS = ("value", "x", "y")


def columns(
    run: RunDetail,
    picks: list[str],
    settings: PlotterSettings,
    y_default: str,
) -> tuple[str, str, str]:
    """The map's scalar, X axis and Y axis for the open scan.

    A saved pick this scan has is used; otherwise the scalar is
    *y_default* (the XY plot's Y1) and the axes are the scan's variables
    (``grid_axes``: of two, the last one across and the first one up, as
    the Grid tab drew them). A scan read from its folder names one
    variable: it is X, and Y is ``""`` until the viewer picks one (the map
    then places each bin at its measured position).

    Parameters
    ----------
    run : RunDetail
        The open scan (its start document names the scan variables).
    picks : list[str]
        The scan's plottable columns.
    settings : PlotterSettings
        The set's saved picks.
    y_default : str
        The scalar when none is saved.

    Returns
    -------
    tuple[str, str, str]
        ``(value, x, y)``.
    """
    axes = [a for a in grid_axes(picks, run.start_doc) if a in picks]

    def chosen(saved: Optional[str], default: str) -> str:
        return saved if saved in picks else default

    x_default = axes[-1] if axes else ""
    y_axis_default = axes[0] if len(axes) >= 2 else ""
    return (
        chosen(settings.map_value, y_default),
        chosen(settings.map_x, x_default),
        chosen(settings.map_y, y_axis_default),
    )


def statistics(text: str) -> dict:
    """The set's map statistics, defaults filled in, without the columns.

    A bad saved config shows the defaults rather than no controls: the
    set's text was checked when it was saved, so this only meets a set
    written by hand or by a newer copy (the map itself then says what is
    wrong with it).
    """
    try:
        cfg = analysis.parse_gridcfg(text)
    except analysis.BadParam:
        cfg = GridConfig()
    return cfg.model_dump(exclude=set(_COLUMNS))


def grid_config(text: str, value: str, x: str, y: str) -> GridConfig:
    """``GridConfig`` from the statistics JSON *text* and the three columns.

    Raises
    ------
    analysis.BadParam
        *text* is not a valid config.
    """
    cfg = analysis.parse_gridcfg(text)
    return cfg.model_copy(update={"value": value, "x": x, "y": y})


def map_code(
    day: date,
    scan: int,
    cfg: GridConfig,
    filters: RowFilters,
    pretty: dict[str, str],
    experiment: str = "",
) -> str:
    """The notebook snippet that rebuilds the map's numbers and both figures.

    Parameters
    ----------
    day : datetime.date
        The scan's day.
    scan : int
        The scan number.
    cfg : GridConfig
        The map as drawn (columns and statistics).
    filters : RowFilters
        The shot filters; ``grid_scan`` applies them.
    pretty : dict[str, str]
        Column → the name the figures title it with.
    experiment : str, optional
        Named only when the app was started with one.

    Returns
    -------
    str
        Python source.
    """
    where = f", experiment={experiment!r}" if experiment else ""
    return (
        "from datetime import date\n"
        "from geecs_data_utils.data.row_filters import RowFilters\n"
        "from geecs_data_utils.scan_grid import GridConfig, grid_scan\n"
        "from geecs_portal.figures import grid_figures\n"
        "from geecs_portal.plotter import scans\n\n"
        f"run = scans.find_scan(date({day.year}, {day.month}, {day.day}), {scan}{where})\n"
        "frame = scans.load_frame(run)\n"
        f"cfg = GridConfig.model_validate_json({cfg.model_dump_json()!r})\n"
        f"filters = RowFilters.model_validate_json({filters.model_dump_json()!r})\n"
        "result = grid_scan(frame, run.start_doc, cfg, filters)\n"
        f"for fig in grid_figures(result, pretty={pretty!r}).values():\n"
        "    fig.show()\n"
    )
