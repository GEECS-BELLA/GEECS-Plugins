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
from geecs_data_utils.data.binning import summarize_columns
from geecs_data_utils.scan_conditions import read_condition_labels, scan_conditions
from geecs_data_utils.scan_frame import scan_frame
from geecs_data_utils.scan_info import analysis_folder_for
from geecs_data_utils.scan_paths import current_paths_config
from geecs_data_utils.tiled_catalog import RunDetail, RunSummary


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
    config = current_paths_config()
    name = experiment or (getattr(config, "experiment", None) or "")
    if not name:
        return None
    catalog = catalog or FolderScanCatalog(base_path)
    try:
        return catalog.load_run(folder_uid(name, day, number))
    except (KeyError, OSError):
        return None


def day_scans(
    day: date,
    *,
    experiment: str = "",
    catalog: FolderScanCatalog,
) -> list[RunSummary]:
    """The day's scans on the share, by number; empty for a day with none.

    Raises
    ------
    OSError
        The data root is unreachable.
    """
    config = current_paths_config()
    name = experiment or (getattr(config, "experiment", None) or "")
    if not name:
        return []
    runs = catalog.list_runs(name, day)
    return sorted(runs, key=lambda r: r.scan_number or 0)


def scan_folder(run: RunDetail) -> Path:
    """The scan's ``scans/ScanNNN`` folder."""
    return Path(run.start_doc["scan_folder"])


def load_frame(run: RunDetail) -> pd.DataFrame:
    """The scan's per-shot table: its s-file, one row per shot."""
    return scan_frame(scan_folder=scan_folder(run)).frame


def condition_labels(run: RunDetail, frame: pd.DataFrame) -> dict[str, list[str]]:
    """The labels of the condition columns *frame* has: ``{column: [label by number]}``.

    The shot filters' label menu: a condition column stores the number of
    the rule that matched (``geecs_data_utils.scan_conditions``).
    """
    labels = read_condition_labels(analysis_folder_for(scan_folder(run)))
    return {c: names for c, names in labels.items() if c in frame.columns}


def conditions(run: RunDetail, frame: pd.DataFrame) -> list[dict]:
    """The scan's condition columns: each one's label over the scan's shots.

    The labels are the scalar-columns jobs' own, read from the scan's
    analysis folder (``geecs_data_utils.scan_conditions``); a scan no
    condition job has run on has none. Read-only.
    """
    labels = read_condition_labels(analysis_folder_for(scan_folder(run)))
    return [
        {
            "column": c.column,
            "label": c.label,
            "mixed": c.mixed,
            "detail": c.detail,
            "counts": c.counts,
            "unknown": c.unknown,
        }
        for c in scan_conditions(frame, labels)
    ]


#: The Images page's scan-scalar statistics: stored name → (agg, err, label).
SCALAR_STATS = {
    "mean_std": ("mean", "std", "mean ± std"),
    "median_iqr": ("median", "iqr", "median [quartiles]"),
    "mean_stderr": ("mean", "stderr", "mean ± std. error"),
}


def _number(value: float) -> str:
    """4 significant figures, the way the page writes a scan scalar."""
    return f"{value:.4g}"


def scan_scalars(
    frame: pd.DataFrame,
    keep: Optional[frozenset[int]],
    columns: list[str],
    stat: str,
    names: Optional[dict[str, str]] = None,
) -> dict:
    """The picked columns over the shots the strip shows, one statistic for all.

    *keep* is the strip's kept rows (from 1, :func:`images.shot_rows`;
    ``None`` is every shot). The numbers are ``summarize_columns``' (the
    binning core's own statistics, the rows as one bin). Returns
    ``{stat, label, shots, scalars}``, each scalar ``{column, name, text,
    missing, center, err_low, err_high, count}``; a column the scan lacks
    is ``missing`` and keeps its place.
    """
    agg, err, label = SCALAR_STATS.get(stat, SCALAR_STATS["mean_std"])
    rows = frame if keep is None else frame.iloc[sorted(r - 1 for r in keep)]
    found = summarize_columns(rows, columns, agg=agg, err=err)
    shots = len(rows)
    out = []
    for column in columns:
        name = (names or {}).get(column) or display_name(column, True)
        summary = found.get(column)
        if summary is None:
            out.append(
                {
                    "column": column,
                    "name": name,
                    "missing": True,
                    "text": f"{name}: not in this scan",
                }
            )
            continue
        if summary.count == 0:
            text = f"{name}: no values"
        elif err == "iqr":
            low = summary.center - summary.err_low
            high = summary.center + summary.err_high
            text = (
                f"{name}: {_number(summary.center)} [{_number(low)} – {_number(high)}]"
            )
        else:
            text = f"{name}: {_number(summary.center)} ± {_number(summary.err_low)}"
        if 0 < summary.count < shots:
            text += f" (n={summary.count})"
        out.append(
            {
                "column": column,
                "name": name,
                "missing": False,
                "text": text,
                "center": summary.center,
                "err_low": summary.err_low,
                "err_high": summary.err_high,
                "count": summary.count,
            }
        )
    return {"stat": stat, "label": label, "shots": shots, "scalars": out}


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
