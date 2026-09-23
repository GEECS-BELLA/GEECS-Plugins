"""Prepare and stream explicit v2 scan analysis without writing scan products."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import pandas as pd
from geecs_analysis.compat.v2_run import ShotGroup, UnitResult, run_units
from geecs_schemas.analysis import AnalysisDiagnostic

from scan_analysis.core_inputs import PreparedRecipe, prepare_v2
from scan_analysis.core_source import V2ShotSource, prepare_source, source_directory


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or int(value) != value
        or (positive and value < 1)
    ):
        raise ValueError(
            f"{label} must contain {'positive ' if positive else ''}integers"
        )
    return int(value)


def group_shots(
    rows: pd.DataFrame,
    mapped_shots: Iterable[int],
    mode: Literal["per_shot", "per_bin"],
) -> tuple[ShotGroup, ...]:
    """Snapshot execution groups in scalar-row order with legacy bin membership.

    Per-shot groups contain mapped shots only. Per-bin groups retain every row
    in each bin with at least one mapped input, including members whose data
    cannot be loaded. Without a bin column, one group contains mapped shots
    only. Missing bin values do not form a bin, matching the legacy wrapper.
    Duplicate/invalid shot identities and fractional bin ids are refused before
    loading data instead of silently truncating or analyzing a shot twice.
    """
    if mode not in ("per_shot", "per_bin"):
        raise ValueError("Scan mode must be per_shot or per_bin")
    if "Shotnumber" not in rows:
        raise ValueError("Shot rows require a Shotnumber column")
    shots = tuple(_integer(n, "Shotnumber", positive=True) for n in rows["Shotnumber"])
    if len(set(shots)) != len(shots):
        raise ValueError("Shot rows cannot repeat a Shotnumber")
    mapped = set(mapped_shots)
    available = tuple(n for n in shots if n in mapped)
    if mode == "per_shot":
        return tuple(ShotGroup(n, (n,)) for n in available)
    if "Bin #" not in rows:
        return (ShotGroup(0, available),) if available else ()
    bins: dict[int, list[int]] = {}
    for shot, value in zip(shots, rows["Bin #"], strict=True):
        if pd.isna(value):
            continue
        key = _integer(value, "Bin #")
        bins.setdefault(key, []).append(shot)
    return tuple(
        ShotGroup(key, tuple(members))
        for key, members in bins.items()
        if any(n in mapped for n in members)
    )


@dataclass(frozen=True)
class PreparedScan:
    """One run's recipe, input references, grouping and scalar naming snapshot."""

    prepared: PreparedRecipe
    source: V2ShotSource
    groups: tuple[ShotGroup, ...]
    average_before_analysis: bool
    output_name: str
    metric_suffix: str

    def run(self) -> Iterator[UnitResult]:
        """Yield core measurements and explicit load/analysis failures lazily."""
        return run_units(
            self.prepared.recipe,
            self.groups,
            self.source.load,
            average_before_analysis=self.average_before_analysis,
            inputs=self.prepared.inputs,
        )

    def scalar_records(self, result: UnitResult) -> list[dict[str, float | int]]:
        """Project bare scalars onto every member row without mutating results.

        Failed outcomes yield no updates. Undefined numerical values are kept
        as emitted and remain annotated in the measurement. In per-bin mode,
        members with missing inputs receive the successful bin's scalars, as
        the old scan wrapper does. This function performs no s-file writes.
        """
        if result.measurement is None:
            return []
        prefix = f"{self.output_name}_" if self.output_name else ""
        values = {
            f"{prefix}{key}{self.metric_suffix}": value
            for key, value in result.measurement.scalars.items()
        }
        if not values:
            return []
        return [{"Shotnumber": number, **values} for number in result.group.shots]


def prepare_scan(
    document: AnalysisDiagnostic, scan_folder: Path, rows: pd.DataFrame
) -> PreparedScan:
    """Snapshot a supported recipe and completed scan before reading shot arrays.

    Compilation rejects unsupported processing before any input reads. File
    backgrounds are loaded once during preparation; native shot arrays are
    loaded only as ``run`` advances. Neither preparation nor execution changes
    the caller's document/rows, writes files, or creates missing scan folders.
    The host owns completion checks, logging failures, and product sinks.
    """
    snapshot = document.model_copy(deep=True)
    row_snapshot = rows.copy(deep=True)
    directory = source_directory(snapshot, scan_folder)
    prepared = prepare_v2(snapshot, data_dir=directory)
    source = prepare_source(snapshot, scan_folder, row_snapshot)
    return PreparedScan(
        prepared,
        source,
        group_shots(row_snapshot, source.references, snapshot.scan.mode),
        snapshot.scan.mode == "per_bin",
        snapshot.effective_output_name,
        snapshot.metric_suffix or "",
    )
