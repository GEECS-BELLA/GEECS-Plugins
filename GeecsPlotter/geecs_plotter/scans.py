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
from geecs_data_utils.folder_catalog import folder_start_doc
from geecs_data_utils.scan_frame import scan_frame
from geecs_data_utils.scan_paths import daily_scan_folder


def scan_folder(
    day: date,
    number: int,
    *,
    experiment: str = "",
    base_path: Optional[Path] = None,
) -> Optional[Path]:
    """Return the existing ``scans/ScanNNN`` folder, or ``None``.

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

    Returns
    -------
    Path or None
        The folder when it exists on the share.
    """
    daily = daily_scan_folder(experiment, base_path=base_path, day=day)
    if daily is None:
        return None
    folder = daily / f"Scan{number:03d}"
    return folder if folder.is_dir() else None


def load_frame(folder: Path) -> pd.DataFrame:
    """The scan's per-shot table: its s-file, one row per shot."""
    return scan_frame(scan_folder=folder).frame


def columns(frame: pd.DataFrame) -> list[str]:
    """The columns offered as plot picks (the portal's rule, shared)."""
    return tiled_schema.plottable_columns(frame)


def default_x(folder: Path, day: date, number: int, picks: list[str]) -> str:
    """The default X, GEECSplotter's "autofill par1": the scan variable.

    Parameters
    ----------
    folder : Path
        The scan folder (its ScanInfo names the scan variable).
    day : datetime.date
        The scan's day.
    number : int
        The scan number.
    picks : list of str
        The offered columns; the scan variable must be one of them.

    Returns
    -------
    str
        The column, or ``""`` on a no-scan (the shot number is X then).
    """
    start_doc, _ = folder_start_doc(folder, "", day, number)
    if not tiled_schema.is_stepped_scan(start_doc):
        return ""
    found = tiled_schema.scan_variable_columns(picks, start_doc)
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
