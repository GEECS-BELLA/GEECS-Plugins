"""Plan legacy scan summary products from core results, without rendering or I/O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from geecs_analysis.compat.v2 import V2Recipe
from geecs_analysis.compat.v2_average import average_results
from geecs_analysis.compat.v2_run import UnitResult
from geecs_analysis.measurement import Measurement

from scan_analysis.core_scan import group_shots


#: The position label a noscan's per-shot panels carry (the waterfall's y axis
#: and title); a preview over a few shots uses the same name.
NOSCAN_POSITION_LABEL = "Shot Number"


@dataclass(frozen=True)
class Product:
    """One averaged/single measurement with its legacy filename identifier."""

    identifier: int | str
    measurement: Measurement
    position: float | None = None


@dataclass(frozen=True)
class ProductPlan:
    """Single-file products and ordered summary panels, with explicit omissions.

    ``singles`` are the per-unit products (each bin, or the scan's average on
    a noscan); ``summary`` the ordered panels a scan-level summary kind draws
    (bin averages, or every shot on a noscan waterfall). Which kinds draw
    them is the recipe's ``summaries`` list, resolved by the sink.
    """

    singles: tuple[Product, ...] = ()
    summary: tuple[Product, ...] = ()
    position_label: str = ""
    notes: tuple[str, ...] = ()


def plan_products(
    recipe: V2Recipe,
    outcomes: Sequence[UnitResult],
    rows: pd.DataFrame,
    *,
    average_before_analysis: bool,
    noscan: bool,
    parameter_column: str | None = None,
    sort_requested: bool = False,
    sort_column: str | None = None,
    sort_bounds: tuple[float, float] | None = None,
    sort_sigma: float | None = 3.0,
) -> ProductPlan:
    """Choose v2 scan products while retaining the old grouping conventions.

    The legacy gate requires more than two successful execution units before
    producing figures; scalar updates are independent and must not use this
    gate. Noscan averages all successful units without weighting, and line
    summaries show individual units ordered by key or an optional sort column.
    A line sort request also bypasses scanned-bin rendering; ``sort_requested``
    is ``bool(waterfall_sort_key)`` and ``sort_column`` its resolved s-file
    column or ``None``, always passed together. Scanned summaries
    average per-shot results by bin, or reuse already-analyzed raw bin means;
    parameter positions use every scalar row in the bin, not only loaded shots.
    """
    successful = [o for o in outcomes if o.measurement is not None]
    if len({o.group.key for o in successful}) != len(successful):
        raise ValueError("Product inputs cannot repeat an execution-unit key")
    if len(successful) <= 2:
        return ProductPlan(
            notes=("Legacy summaries require more than two successful units",)
        )
    line = recipe.input_kind == "line"
    if noscan or (line and sort_requested):
        average = average_results(
            [o.measurement for o in successful], recipe, mode="noscan"
        )
        singles = (Product("average", average),) if average is not None else ()
        notes = (
            []
            if average is not None
            else ["Skipped average: incompatible result shapes"]
        )
        if not line:
            return ProductPlan(singles=singles, notes=tuple(notes))
        panels = []
        for outcome in sorted(successful, key=lambda o: o.group.key):
            position = float(outcome.group.key)
            if sort_column is not None:
                values = rows.loc[rows["Shotnumber"] == outcome.group.key, sort_column]
                position = float(values.iloc[0]) if not values.empty else float("nan")
            panels.append(Product(outcome.group.key, outcome.measurement, position))
        if sort_column is not None:
            count = len(panels)
            panels = [p for p in panels if np.isfinite(p.position)]
            if panels:
                values = np.array([p.position for p in panels])
                bounds = sort_bounds
                if bounds is None and sort_sigma is not None:
                    mean, std = values.mean(), values.std()
                    bounds = (mean - sort_sigma * std, mean + sort_sigma * std)
                if bounds is not None:
                    panels = [p for p in panels if bounds[0] <= p.position <= bounds[1]]
                panels.sort(key=lambda p: p.position)
            if len(panels) != count:
                notes.append(f"Waterfall sort excluded {count - len(panels)} units")
        return ProductPlan(
            singles,
            tuple(panels),
            sort_column or NOSCAN_POSITION_LABEL,
            tuple(notes),
        )

    if "Bin #" not in rows or parameter_column is None or parameter_column not in rows:
        return ProductPlan(
            notes=("Skipped scanned products: missing bin or parameter column",)
        )
    by_key = {o.group.key: o.measurement for o in successful}
    if average_before_analysis:
        measurements = by_key
    else:
        measurements = {
            group.key: average_results(
                [by_key[n] for n in group.shots if n in by_key], recipe, mode="bin"
            )
            for group in group_shots(rows, by_key, "per_bin")
        }
    panels = []
    notes = []
    for key, measurement in sorted(measurements.items()):
        if measurement is None:
            notes.append(f"Skipped bin {key}: incompatible result shapes")
            continue
        position = float(rows.loc[rows["Bin #"] == key, parameter_column].mean())
        panels.append(Product(key, measurement, position))
    return ProductPlan(
        tuple(panels),
        tuple(panels) if len(panels) > 1 else (),
        parameter_column,
        tuple(notes),
    )
