"""Find one scan and read its s-file — calls into GEECS-Data-Utils, no numerics.

Read-only: a scan that is not on the share is ``None``, never created (the
repo's scan-folder invariant). No Tiled: the frame is the s-file alone.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from geecs_data_utils import tiled_schema
from geecs_data_utils.data.sfile import column_alias
from geecs_data_utils.folder_catalog import FolderScanCatalog, folder_uid
from geecs_data_utils.scan_frame import scan_frame
from geecs_data_utils.scan_paths import ScanPaths
from geecs_data_utils.tiled_catalog import RunDetail


def find_scan(
    day: date,
    number: int,
    *,
    experiment: str = "",
    base_path: Optional[Path] = None,
    catalog: Optional[FolderScanCatalog] = None,
) -> Optional[RunDetail]:
    """Look one scan up on the share, or ``None`` when it is not there.

    The folder catalog's lookup, so an unpadded ``Scan7`` folder opens too.

    Parameters
    ----------
    day : datetime.date
        The scan's day.
    number : int
        The scan number.
    experiment : str, optional
        The experiment folder; empty means ``config.ini [Experiment] expt``.
    base_path : Path, optional
        The data root; defaults to ``config.ini``. Tests pass a tmp tree.
    catalog : FolderScanCatalog, optional
        A long-lived catalog (the app's), which remembers finished scans.

    Returns
    -------
    RunDetail or None
        The scan's documents; its folder is :func:`scan_folder`.
    """
    config = ScanPaths.paths_config
    name = experiment or (getattr(config, "experiment", None) or "")
    if not name:
        return None
    catalog = catalog or FolderScanCatalog(base_path)
    try:
        return catalog.load_run(folder_uid(name, day, number))
    except (KeyError, OSError):
        return None


def scan_folder(run: RunDetail) -> Path:
    """The scan's ``scans/ScanNNN`` folder."""
    return Path(run.start_doc["scan_folder"])


def load_frame(run: RunDetail) -> pd.DataFrame:
    """The scan's per-shot table: its s-file, one row per shot."""
    return scan_frame(scan_folder=scan_folder(run)).frame


def columns(frame: pd.DataFrame) -> list[str]:
    """The columns offered as plot picks (the portal's rule, shared)."""
    return tiled_schema.plottable_columns(frame)


def default_x(run: RunDetail, picks: list[str]) -> str:
    """The default X, GEECSplotter's "autofill par1": the scan variable.

    Parameters
    ----------
    run : RunDetail
        The scan, from :func:`find_scan`.
    picks : list of str
        The offered columns; the scan variable must be one of them.

    Returns
    -------
    str
        The column, or ``""`` on a no-scan (the shot number is X then).
    """
    if not tiled_schema.is_stepped_scan(run.start_doc):
        return ""
    found = tiled_schema.scan_variable_columns(picks, run.start_doc)
    return found[0] if found else ""


def plot_series(frame: pd.DataFrame, names: list[str]) -> dict[str, pd.Series]:
    """Coerce the named columns to floats.

    Parameters
    ----------
    frame : pandas.DataFrame
        The scan's table from :func:`load_frame`.
    names : list of str
        The picked columns.

    Returns
    -------
    dict of str to pandas.Series
        Column → float values (NaN where a shot has none).

    Raises
    ------
    KeyError
        A name that is not a plottable column of the frame.
    """
    series = {}
    for name in dict.fromkeys(names):
        values = tiled_schema.numeric_series(frame, name)
        if values is None:
            raise KeyError(name)
        series[name] = values
    return series


def shot_axis(frame: pd.DataFrame) -> pd.Series:
    """The shot numbers, one per row (the portal's shot-identity rule)."""
    return tiled_schema.shot_axis_for_frame(frame)


def display_name(column: str, use_alias: bool) -> str:
    """The name shown for *column*: its LabVIEW alias when asked and it has one.

    Parameters
    ----------
    column : str
        The s-file column, e.g. ``"U_ESP Position.Axis 1 Alias:Jet_X (mm)"``.
    use_alias : bool
        GEECSplotter's "use aliases" toggle for this axis.

    Returns
    -------
    str
        ``"Jet_X (mm)"`` with the toggle on, else the column unchanged.
    """
    return (column_alias(column) or column) if use_alias else column
