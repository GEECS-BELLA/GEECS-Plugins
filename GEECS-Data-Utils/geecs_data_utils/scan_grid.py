"""Read-only two-axis scan geometry and statistics, independent of any UI.

Geometry is resolved from the unfiltered acquisition bins. Filters affect
statistics and membership only. Planned rectangular scans retain unvisited
positions; arbitrary trajectories retain measured points without interpolation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from geecs_data_utils.data.binning import BinningConfig, bin_frame
from geecs_data_utils.data.row_filters import RowFilters, filter_mask
from geecs_data_utils.tiled_schema import (
    numeric_series,
    scan_motors,
    scan_variable_columns,
    shot_axis_for_frame,
)

MAX_GRID_CELLS = 20_000


class GridConfig(BaseModel):
    """Independent center, error, coordinates and visit selection."""

    model_config = ConfigDict(extra="forbid", strict=True)

    x: str = ""
    y: str = ""
    value: str = ""
    average: Literal["mean", "median"] = "median"
    error: Literal["std", "stderr", "mad", "iqr", "percentile"] = "iqr"
    lower: float = Field(default=0.25, ge=0, le=1)
    upper: float = Field(default=0.75, ge=0, le=1)
    error_view: Literal["width", "low", "high"] = "width"
    xscale: Literal["linear", "log", "index"] = "linear"
    yscale: Literal["linear", "log", "index"] = "linear"
    min_count: int = Field(default=3, ge=1)
    visit: int = Field(default=1, ge=1)


@dataclass(frozen=True)
class GridResult:
    """Rows of cells, complete axis coordinates and source membership."""

    cells: pd.DataFrame
    config: GridConfig
    kind: str
    x_values: list[float]
    y_values: list[float]
    visits: int
    passing: int
    total: int
    bin_column: str
    notes: tuple[str, ...]


def grid_axes(columns: list[str], start: Mapping) -> list[str]:
    """Resolve one readback column per motor, preserving motor order.

    Descriptor hints win; the existing column semantics supply the fallback.
    The s-file's exact header is accepted when only that provider is present.
    """
    hints = start.get("hints") or {}
    dimensions = hints.get("dimensions") or []
    hinted = [c for d in dimensions for c in d[0] if c in columns]
    headers = start.get("geecs_scalar_headers") or {}
    axes = []
    for motor in scan_motors(start):
        candidates = scan_variable_columns(columns, {"motors": [motor]})
        preferred = [c for c in hinted if c in candidates]
        if preferred:
            axes.append(preferred[0])
        elif candidates:
            axes.append(motor if motor in candidates else candidates[0])
        else:
            matched = [
                v
                for k, v in headers.items()
                if (k == motor or k.startswith(motor + "-")) and v in columns
            ]
            if matched:
                axes.append(matched[0])
    return list(dict.fromkeys(axes))


def _points(spec: Mapping) -> list[float]:
    kind = spec.get("kind", "range")
    if kind == "list":
        values = spec["positions"]
        if len(values) > MAX_GRID_CELLS:
            raise ValueError("Too many grid positions.")
        points = np.asarray(values, dtype=float)
    else:
        n = int(spec["num"])
        if not 1 <= n <= MAX_GRID_CELLS:
            raise ValueError("Too many grid positions.")
        with np.errstate(over="ignore", invalid="ignore"):
            points = (
                np.logspace(spec["start_exp"], spec["stop_exp"], n)
                if kind == "log"
                else np.linspace(spec["start"], spec["stop"], n)
            )
    if points.ndim != 1 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Invalid planned grid coordinates.")
    return points.tolist()


def _planned(start: Mapping) -> tuple[list[list[float]], bool, list[bool]]:
    """Decode recorded two-axis grids; never import or execute a scan plan."""
    trajectory = (start.get("sweep") or {}).get("trajectory") or {}
    if trajectory.get("kind") == "axes" and trajectory.get("combine") == "product":
        specs = trajectory["axes"]
        if len(specs) != 2:
            return [], False, []
        return (
            [_points(a) for a in specs],
            bool(trajectory.get("snake")),
            [bool(a.get("relative")) for a in specs],
        )
    pattern = start.get("plan_pattern", "")
    relative = str(start.get("plan_name", "")).startswith("rel_")
    if pattern == "outer_list_product":
        args = start.get("plan_pattern_args", {}).get("args", [])
        if len(args) != 4:
            return [], False, []
        axes = [_points({"kind": "list", "positions": args[i]}) for i in (1, 3)]
        snake = start.get("plan_pattern_args", {}).get("snake_axes", "False")
        # Stock list_grid_scan serializes this as repr. For two axes the
        # only non-empty legal selection is the inner motor. Never eval it.
        if snake in (False, None, "False", "None", "[]", "()"):
            snake = False
        elif snake in (True, "True") or isinstance(snake, (list, tuple)):
            snake = bool(snake)
        elif isinstance(snake, str) and snake.startswith(("[", "(")):
            snake = True
        else:
            raise ValueError("Unrecognized grid snaking metadata.")
        return axes, snake, [relative, relative]
    if pattern == "outer_product":
        shape, extents = start.get("shape", []), start.get("extents", [])
        if len(shape) == len(extents) == 2:
            axes = [
                _points({"num": n, "start": limits[0], "stop": limits[1]})
                for n, limits in zip(shape, extents)
            ]
            snaking = start.get("snaking") or [False, False]
            return axes, bool(snaking[1]), [relative, relative]
    return [], False, []


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise KeyError(column)
    # All-missing selected values remain a legitimate empty map.
    if frame[column].isna().all():
        return pd.Series(np.nan, index=frame.index, dtype=float)
    series = numeric_series(frame, column)
    if series is None:
        raise ValueError(f"Column {column!r} is not numeric.")
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def grid_scan(
    frame: pd.DataFrame,
    start: Mapping,
    config: GridConfig,
    filters: RowFilters | None = None,
) -> GridResult:
    """Aggregate a two-axis scan without letting filters erase geometry.

    Parameters
    ----------
    frame : pandas.DataFrame
        The complete union shot frame, before filtering.
    start : Mapping
        Recorded run metadata; no device or catalog access is performed.
    config : GridConfig
        Axis/scalar selection, independent statistics and display choices.
    filters : RowFilters, optional
        The same row filters consumed by Plot and Images.

    Returns
    -------
    GridResult
        One row per acquisition bin/visit, including unacquired cells.
        Counts are finite samples of the selected scalar, not other columns.
    """
    if config.lower >= config.upper:
        raise ValueError("The lower percentile must be below the upper percentile.")
    axes = grid_axes(list(frame.columns), start)
    if len(scan_motors(start)) > 2:
        raise ValueError(
            "Select a two-axis scan; higher-dimensional slices are not yet supported."
        )
    x = config.x or (axes[-1] if len(axes) >= 2 else "")
    y = config.y or (axes[0] if len(axes) >= 2 else "")
    if not x or not y or x == y:
        raise ValueError("Choose two distinct numeric scan axes.")
    if not config.value:
        raise ValueError("Choose a scalar for the grid.")
    cfg = config.model_copy(
        update={
            "x": x,
            "y": y,
            **({"lower": 0.25, "upper": 0.75} if config.error == "iqr" else {}),
        }
    )
    work = pd.DataFrame(index=frame.index)
    for column in dict.fromkeys([x, y, cfg.value]):
        work[column] = _numeric(frame, column)
    # Use one provider, exactly as Images does; never reconcile namespaces.
    bins = pd.Series(np.nan, index=frame.index)
    bin_column = "bin_number"
    for column in ("bin_number", "Bin #"):
        if column in frame:
            candidate = pd.to_numeric(frame[column], errors="coerce")
            if candidate.notna().any():
                bins, bin_column = candidate, column
                break
    if bins.isna().all():
        raise ValueError("This scan has no acquisition-bin column.")
    if ((bins.dropna() < 1) | (bins.dropna() % 1 != 0)).any():
        raise ValueError("Acquisition bin numbers must be positive integers.")
    work["__bin__"] = bins
    shots = shot_axis_for_frame(frame)
    mask = filter_mask(frame, filters or RowFilters())
    observed = work.groupby("__bin__")[[x, y]].mean()
    notes = []
    if bins.isna().any():
        notes.append(
            f"{int(bins.isna().sum())} rows have no {bin_column} identity and cannot be placed."
        )
    planned = []
    if len(axes) == 2 and {x, y} == set(axes):
        positions, snake, relative = _planned(start)
        if positions:
            if len(positions[0]) * len(positions[1]) > MAX_GRID_CELLS:
                raise ValueError(f"Grid exceeds {MAX_GRID_CELLS} positions.")
            planned = [
                [a, b]
                for i, a in enumerate(positions[0])
                for b in (positions[1][::-1] if snake and i % 2 else positions[1])
            ]
            for index, is_relative in enumerate(relative):
                if not is_relative:
                    continue
                anchors = observed[axes[index]].dropna()
                anchors = anchors[
                    (anchors.index >= 1) & (anchors.index <= len(planned))
                ]
                if anchors.empty:
                    planned = []
                    break
                bin_id = int(anchors.index[0])
                offset = float(anchors.iloc[0]) - planned[bin_id - 1][index]
                for position in planned:
                    position[index] += offset
                notes.append(
                    "Relative coordinates anchored to the first available bin readback."
                )
    kind = "grid" if planned else "points"
    coordinates = {}
    if planned:
        xi, yi = axes.index(x), axes.index(y)
        coordinates = {i + 1: (p[xi], p[yi]) for i, p in enumerate(planned)}
        if any(b not in coordinates for b in observed.index):
            raise ValueError(
                "Acquisition bins exceed the recorded grid; refusing to misplace cells."
            )
        notes.append(
            "Cells use planned positions; measured readbacks are shown on selection."
        )
    else:
        coordinates = {
            int(b): (float(row[x]), float(row[y]))
            for b, row in observed.iterrows()
            if np.isfinite(row[x]) and np.isfinite(row[y])
        }
        notes.append(
            "Measured bin positions; no interpolation or inferred unacquired positions."
        )
    if len(coordinates) > MAX_GRID_CELLS:
        raise ValueError(f"Grid exceeds {MAX_GRID_CELLS} positions.")
    valid = mask & work[cfg.value].notna() & bins.notna()
    selected = work.loc[valid]
    bcfg = BinningConfig(
        bin_col="__bin__",
        value_cols=[cfg.value],
        agg=cfg.average,
        err=cfg.error,
        percentiles=(cfg.lower, cfg.upper),
        scale_to_sigma=True,
    )
    summary = bin_frame(selected, bcfg)
    quantiles = (
        selected.groupby("__bin__")[cfg.value].quantile([cfg.lower, cfg.upper])
        if not selected.empty
        else pd.Series(dtype=float)
    )
    acquired = bins.value_counts()
    accepted = bins[mask].value_counts()
    members = {
        int(b): [int(s) for s in group.dropna() if np.isfinite(s)]
        for b, group in shots[mask].groupby(bins[mask])
    }
    visits = Counter()
    rows = []
    for b, (px, py) in coordinates.items():
        visits[(px, py)] += 1
        n = int(summary.counts.get(b, 0))
        center = low = high = error = np.nan
        if n:
            values = summary.frame.loc[b, cfg.value]
            center, low, high = (
                float(values[k]) for k in ("center", "err_low", "err_high")
            )
            if n >= cfg.min_count:
                if cfg.error in ("iqr", "percentile"):
                    error = (
                        float(quantiles.loc[b, cfg.upper] - quantiles.loc[b, cfg.lower])
                        if cfg.error_view == "width"
                        else low
                        if cfg.error_view == "low"
                        else high
                    )
                else:
                    error = high
        count = int(acquired.get(b, 0))
        passing = int(accepted.get(b, 0))
        status = (
            "unacquired"
            if not count
            else "filtered"
            if not passing
            else "missing"
            if not n
            else "low_count"
            if n < cfg.min_count
            else "ok"
        )
        rows.append(
            dict(
                bin=b,
                x=px,
                y=py,
                visit=visits[(px, py)],
                center=center,
                err_low=low,
                err_high=high,
                error=error,
                count=n,
                acquired=count,
                passing=passing,
                shots=members.get(b, []),
                status=status,
                x_readback=float(observed.loc[b, x]) if b in observed.index else np.nan,
                y_readback=float(observed.loc[b, y]) if b in observed.index else np.nan,
            )
        )
    cells = pd.DataFrame(
        rows,
        columns=[
            "bin",
            "x",
            "y",
            "visit",
            "center",
            "err_low",
            "err_high",
            "error",
            "count",
            "acquired",
            "passing",
            "shots",
            "status",
            "x_readback",
            "y_readback",
        ],
    )
    for column, scale in ((x, cfg.xscale), (y, cfg.yscale)):
        if scale == "log" and any(v <= 0 for v in cells["x" if column == x else "y"]):
            raise ValueError("Logarithmic axes require strictly positive coordinates.")
    if kind == "points" and "index" in (cfg.xscale, cfg.yscale):
        raise ValueError("Equal-cell spacing requires a recorded rectangular grid.")
    return GridResult(
        cells=cells,
        config=cfg,
        kind=kind,
        x_values=sorted(set(cells.x)),
        y_values=sorted(set(cells.y)),
        visits=max(visits.values(), default=1),
        passing=int(mask.sum()),
        total=len(frame),
        bin_column=bin_column,
        notes=tuple(dict.fromkeys(notes)),
    )
