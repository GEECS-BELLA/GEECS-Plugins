"""Plan legacy scan summary products from core results, without rendering or I/O.

:class:`ProductCollector` folds the outcomes as a run streams them, so a
camera scan holds one running frame per product (the noscan average, or one
per bin) however many shots it has; :func:`plan_products` is the same
planning over a sequence of outcomes already in hand, built on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from geecs_analysis.compat.v2 import V2Recipe
from geecs_analysis.compat.v2_average import RunningAverage
from geecs_analysis.compat.v2_run import UnitResult
from geecs_analysis.measurement import Measurement

#: The position label a noscan's per-shot panels carry (the waterfall's y axis
#: and title); a preview over a few shots uses the same name.
NOSCAN_POSITION_LABEL = "Shot Number"

#: The legacy figure gate: more than this many successful execution units.
MIN_SUCCESSFUL_UNITS = 2


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


def bin_of_shot(rows: pd.DataFrame) -> dict[int, int]:
    """``{shot number: bin key}`` for every row with a bin, from the s-file rows.

    Which bin each shot's result folds into is known before any frame is
    loaded, so a scanned run accumulates per bin as it streams. Rows without
    a bin value form no bin, as ``core_scan.group_shots`` decides; an absent
    ``Bin #`` column means no bins at all.
    """
    if "Bin #" not in rows:
        return {}
    return {
        int(shot): int(value)
        for shot, value in zip(rows["Shotnumber"], rows["Bin #"], strict=True)
        if not pd.isna(value)
    }


class ProductCollector:
    """Fold successful outcomes into the scan's products as the run streams them.

    Built for a run before it starts: a noscan (or a line recipe with a
    waterfall sort request) keeps one running average over every unit, and
    a line recipe's units for the waterfall panels; a scanned per-shot run
    keeps one running average per bin (membership from the rows, known
    before any load); a scanned raw-bin run keeps each bin's own
    measurement. A camera frame is folded and dropped, so memory is the
    number of products, not the number of shots; line traces are retained,
    the waterfall needs every one. :meth:`plan` then chooses the products
    exactly as the old sequence planner did (see :func:`plan_products`).
    """

    def __init__(
        self,
        recipe: V2Recipe,
        rows: pd.DataFrame,
        *,
        average_before_analysis: bool,
        noscan: bool,
        sort_requested: bool = False,
    ) -> None:
        self.recipe = recipe
        self.line = recipe.input_kind == "line"
        self.average_before_analysis = average_before_analysis
        self.noscan = noscan
        self.sort_requested = sort_requested
        self.successful = 0
        self._keys: set[int] = set()
        self._whole: RunningAverage | None = None
        self._units: dict[int, Measurement] = {}
        self._bin_of: dict[int, int] = {}
        self._bins: dict[int, RunningAverage] = {}
        self._raw_bins: dict[int, Measurement] = {}
        if self.unbinned:
            self._whole = RunningAverage(recipe, mode="noscan")
        elif not average_before_analysis:
            self._bin_of = bin_of_shot(rows)

    @property
    def unbinned(self) -> bool:
        """Whether the products are the whole scan's (noscan, or a sorted waterfall)."""
        return self.noscan or (self.line and self.sort_requested)

    def add(self, outcome: UnitResult) -> None:
        """Fold one outcome; a failed one contributes nothing."""
        measurement = outcome.measurement
        if measurement is None:
            return
        key = outcome.group.key
        if key in self._keys:
            raise ValueError("Product inputs cannot repeat an execution-unit key")
        self._keys.add(key)
        self.successful += 1
        if self.unbinned:
            self._whole.add(measurement)
            if self.line:
                self._units[key] = measurement
        elif self.average_before_analysis:
            self._raw_bins[key] = measurement
        else:
            bin_key = self._bin_of.get(key)
            if bin_key is None:
                return
            running = self._bins.get(bin_key)
            if running is None:
                running = self._bins[bin_key] = RunningAverage(self.recipe, mode="bin")
            running.add(measurement)

    def plan(
        self,
        rows: pd.DataFrame,
        *,
        parameter_column: str | None = None,
        sort_column: str | None = None,
        sort_bounds: tuple[float, float] | None = None,
        sort_sigma: float | None = 3.0,
    ) -> ProductPlan:
        """Choose the products from everything folded so far.

        ``rows`` are the scalar rows as they stand now — a sort column that
        is one of this run's own outputs resolves against them, so they are
        read here rather than at construction. The legacy gate requires more
        than two successful execution units before producing figures.
        """
        if self.successful <= MIN_SUCCESSFUL_UNITS:
            return ProductPlan(
                notes=("Legacy summaries require more than two successful units",)
            )
        if self.unbinned:
            return self._plan_unbinned(rows, sort_column, sort_bounds, sort_sigma)
        return self._plan_binned(rows, parameter_column)

    def _plan_unbinned(
        self,
        rows: pd.DataFrame,
        sort_column: str | None,
        sort_bounds: tuple[float, float] | None,
        sort_sigma: float | None,
    ) -> ProductPlan:
        average = self._whole.result()
        singles = (Product("average", average),) if average is not None else ()
        notes = (
            []
            if average is not None
            else ["Skipped average: incompatible result shapes"]
        )
        if not self.line:
            return ProductPlan(singles=singles, notes=tuple(notes))
        panels = []
        for key in sorted(self._units):
            position = float(key)
            if sort_column is not None:
                values = rows.loc[rows["Shotnumber"] == key, sort_column]
                position = float(values.iloc[0]) if not values.empty else float("nan")
            panels.append(Product(key, self._units[key], position))
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

    def _plan_binned(
        self, rows: pd.DataFrame, parameter_column: str | None
    ) -> ProductPlan:
        if (
            "Bin #" not in rows
            or parameter_column is None
            or parameter_column not in rows
        ):
            return ProductPlan(
                notes=("Skipped scanned products: missing bin or parameter column",)
            )
        if self.average_before_analysis:
            measurements = dict(self._raw_bins)
        else:
            measurements = {
                key: running.result() for key, running in self._bins.items()
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
    The fold is :class:`ProductCollector`'s, one definition for both.
    """
    collector = ProductCollector(
        recipe,
        rows,
        average_before_analysis=average_before_analysis,
        noscan=noscan,
        sort_requested=sort_requested,
    )
    for outcome in outcomes:
        collector.add(outcome)
    return collector.plan(
        rows,
        parameter_column=parameter_column,
        sort_column=sort_column,
        sort_bounds=sort_bounds,
        sort_sigma=sort_sigma,
    )
