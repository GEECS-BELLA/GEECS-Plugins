"""Several scans on one XY plot: the merged frame and the overlay's groups.

The main scan and the set's extra scans (``PlotterSettings.extra_scans``)
are read into one frame by ``geecs_data_utils.data.scan_merge``: each row
tagged ``scan`` and ``day``, the union of columns, every shot keeping its
own number. Both views start there, so a shot filter can name ``scan``:

- **merged** (LabVIEW's "merge s files") plots that frame as if it were
  one scan; its bins are by a value, never by ``Bin #``, which restarts
  in every scan (:func:`merged_bins`).
- **overlay** splits it back into one group per scan (:func:`overlay_groups`,
  :func:`overlay_bins`), each its own trace and colour.

Read-only, like :mod:`geecs_portal.plotter.scans`.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from geecs_data_utils.data.binning import BinningConfig
from geecs_data_utils.data.scan_merge import (
    DAY_COLUMN,
    SCAN_COLUMN,
    ScanRef,
    merge_scan_frames,
)
from geecs_data_utils.folder_catalog import FolderScanCatalog

from geecs_portal import analysis
from geecs_portal.plotter import scans

#: The bin column that numbers bins within one scan (it restarts in each).
PER_SCAN_BIN = "Bin #"


def scan_label(day: date, number: int) -> str:
    """A scan's legend name: ``"10-26 S006"``."""
    return f"{day:%m-%d} S{number:03d}"


def merged_frame(
    refs: Sequence[ScanRef],
    *,
    experiment: str = "",
    base_path: Optional[Path] = None,
    catalog: Optional[FolderScanCatalog] = None,
) -> pd.DataFrame:
    """The scans' s-files as one frame (``merge_scan_frames``), in *refs*' order.

    Raises
    ------
    LookupError
        A scan that is not on the share.
    """
    frames = {}
    for day, number in refs:
        run = scans.find_scan(
            day, number, experiment=experiment, base_path=base_path, catalog=catalog
        )
        if run is None:
            raise LookupError(f"no Scan{number:03d} on {day.isoformat()}")
        frames[(day, number)] = scans.load_frame(run)
    return merge_scan_frames(frames)


def scan_parts(
    frame: pd.DataFrame, refs: Sequence[ScanRef]
) -> list[tuple[str, pd.DataFrame]]:
    """``(label, rows)`` per scan of *refs*, in order; a scan with no row kept is empty.

    One per scan whatever the filters keep, so a scan's colour (its place
    in *refs*) never shifts when another one's shots are all filtered out.
    """
    return [
        (
            scan_label(day, number),
            frame[
                (frame[SCAN_COLUMN] == number) & (frame[DAY_COLUMN] == day.isoformat())
            ],
        )
        for day, number in refs
    ]


def overlay_groups(
    frame: pd.DataFrame,
    kept: pd.DataFrame,
    refs: Sequence[ScanRef],
    names: Sequence[str],
) -> list[tuple[str, dict[str, pd.Series], pd.Series]]:
    """The overlay's per-shot groups: ``(label, series, shot numbers)`` per scan.

    As the single-scan plot: the columns are read off the whole merged
    *frame* (a column only some scans have is NaN on the others), then cut
    to each scan's kept rows.

    Raises
    ------
    KeyError
        A name no scan has as a plottable column.
    """
    whole = scans.plot_series(frame, list(names))
    return [
        (label, {n: v.loc[part.index] for n, v in whole.items()}, scans.shot_axis(part))
        for label, part in scan_parts(kept, refs)
    ]


def overlay_bins(
    kept: pd.DataFrame,
    refs: Sequence[ScanRef],
    cfg: BinningConfig,
    y: str,
    x: Optional[str],
) -> list[tuple[str, list, dict, Optional[list]]]:
    """The overlay's binned groups: each scan binned on its own.

    ``Bin #`` restarts in every scan, so bins are made within a scan only.
    Each is ``(label, bin labels, {y: centre and errors}, X positions)``,
    as :func:`geecs_portal.analysis.binned_series` gives them; a scan with
    no shot kept has no bins.

    Raises
    ------
    KeyError, TypeError, ValueError
        As ``binned_series``: a missing bin column, a column that does not
        aggregate.
    """
    groups: list[tuple[str, list, dict, Optional[list]]] = []
    for label, part in scan_parts(kept, refs):
        if part.empty:
            groups.append((label, [], {}, None))
            continue
        result, series, x_centers = analysis.binned_series(part, cfg, [y], x)
        groups.append(
            (label, analysis.jsonable_labels(result.frame.index), series, x_centers)
        )
    return groups


def merged_bins(cfg: BinningConfig, x: Optional[str]) -> BinningConfig:
    """The merged view's bins: by the X value where *cfg* says ``Bin #``.

    ``Bin #`` restarts in every scan, so bin 1 of two scans is two
    different settings; pooled shots are grouped by X instead (the bin
    width applies; none groups equal values). Any other bin column is
    kept.

    Raises
    ------
    ValueError
        ``Bin #`` with X the shot number: there is no value to bin by.
    """
    if cfg.bin_col != PER_SCAN_BIN:
        return cfg
    if not x:
        raise ValueError(
            "merged scans bin by the X value (Bin # restarts in every scan): "
            "pick an X, or a bin column under ⚙ bins"
        )
    return dataclasses.replace(cfg, bin_col=x)


def column_coverage(frames: dict[Any, pd.DataFrame]) -> dict[str, int]:
    """How many of *frames* offer each plottable column, union order."""
    counts: dict[str, int] = {}
    for frame in frames.values():
        for column in scans.columns(frame):
            counts[column] = counts.get(column, 0) + 1
    return counts
