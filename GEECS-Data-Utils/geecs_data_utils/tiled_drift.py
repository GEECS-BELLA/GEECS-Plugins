"""Telemetry drift analysis over a Tiled-recorded run — pure functions.

The "moved during scan" heuristic: a numeric telemetry column drifted when
the change between its first and last valid samples exceeds
``threshold × σ`` of its in-scan spread.  σ ≈ 0 (a quiet setpoint that
stepped once mid-scan has near-zero jitter) is guarded with a *relative*
epsilon so the comparison never divides by nothing and a genuine step on a
dead-quiet channel is still flagged.

Everything here operates on plain sequences of floats (zero Qt, zero
pandas at import) so the module stays trivially unit-testable and
reusable from any consumer — GUI or batch analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

#: Default drift threshold: |last − first| must exceed this many sigmas.
DEFAULT_THRESHOLD = 3.0

#: Relative σ ≈ 0 floor (see module docstring): effective sigma is at least
#: this fraction of the column's |mean|.
RELATIVE_SIGMA_EPSILON = 1e-6

#: Absolute sigma floor for columns whose mean is itself ~0.
ABSOLUTE_SIGMA_EPSILON = 1e-12

#: Minimum number of finite samples for a column to be evaluated at all.
MIN_VALID_SAMPLES = 3


@dataclass(frozen=True)
class ColumnDrift:
    """One drifting column's report.

    Attributes
    ----------
    column : str
        The event-stream column name.
    first : float
        First finite sample in the run.
    last : float
        Last finite sample in the run.
    delta : float
        ``last − first`` (signed).
    sigma : float
        In-scan standard deviation (population, over finite samples).
    significance : float
        ``|delta| / effective_sigma`` — the sort key (largest first).
    percent : float or None
        ``delta`` as a percentage of ``|mean|``, or ``None`` when the mean
        is too close to zero for a percentage to make sense.
    """

    column: str
    first: float
    last: float
    delta: float
    sigma: float
    significance: float
    percent: Optional[float]


@dataclass(frozen=True)
class DriftReport:
    """The full drift-analysis result.

    Attributes
    ----------
    drifting : tuple of ColumnDrift
        Columns beyond tolerance, sorted by significance (largest first).
    evaluated : int
        Number of columns that had enough finite numeric samples to judge.
    steady : int
        ``evaluated − len(drifting)`` (the "N of M steady" line).
    """

    drifting: tuple[ColumnDrift, ...]
    evaluated: int

    @property
    def steady(self) -> int:
        """Number of evaluated columns within tolerance."""
        return self.evaluated - len(self.drifting)


def _finite(values: Sequence[float]) -> list[float]:
    """Return the finite float samples of *values*, in order.

    Non-numeric entries (strings from dtype-tolerant telemetry columns)
    and NaN/inf are dropped.
    """
    finite: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return finite


def analyze_column(
    column: str,
    values: Sequence[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> Optional[ColumnDrift]:
    """Judge one column; return its report when it drifted.

    Parameters
    ----------
    column : str
        The column name (carried into the report).
    values : sequence of float
        The column's samples in shot order.  NaN and non-numeric entries
        are ignored; fewer than :data:`MIN_VALID_SAMPLES` finite samples
        means the column cannot be judged (returns ``None``).
    threshold : float, optional
        Sigma multiple beyond which the column counts as drifting.

    Returns
    -------
    ColumnDrift or None
        The drift report when ``|last − first| > threshold × σ_eff``,
        else ``None`` (steady or unjudgeable).
    """
    finite = _finite(values)
    if len(finite) < MIN_VALID_SAMPLES:
        return None
    n = len(finite)
    mean = sum(finite) / n
    variance = sum((v - mean) ** 2 for v in finite) / n
    sigma = math.sqrt(variance)
    sigma_eff = max(
        sigma,
        RELATIVE_SIGMA_EPSILON * abs(mean),
        ABSOLUTE_SIGMA_EPSILON,
    )
    first, last = finite[0], finite[-1]
    delta = last - first
    if abs(delta) <= threshold * sigma_eff:
        return None
    percent: Optional[float] = None
    if abs(mean) > ABSOLUTE_SIGMA_EPSILON:
        percent = 100.0 * delta / abs(mean)
    return ColumnDrift(
        column=column,
        first=first,
        last=last,
        delta=delta,
        sigma=sigma,
        significance=abs(delta) / sigma_eff,
        percent=percent,
    )


def compute_drift(
    columns: Mapping[str, Sequence[float]],
    threshold: float = DEFAULT_THRESHOLD,
) -> DriftReport:
    """Judge every column and report the drifters, most significant first.

    Parameters
    ----------
    columns : mapping of str to sequence of float
        Column name → samples in shot order (typically the numeric
        telemetry columns; see ``schema_map.telemetry_columns``).
    threshold : float, optional
        Sigma multiple beyond which a column counts as drifting.

    Returns
    -------
    DriftReport
        Drifting columns sorted by significance, plus the evaluated count
        for the "N of M steady" line.  Columns without enough finite
        samples are excluded from both counts.
    """
    drifting: list[ColumnDrift] = []
    evaluated = 0
    for name, values in columns.items():
        finite_count = len(_finite(values))
        if finite_count < MIN_VALID_SAMPLES:
            continue
        evaluated += 1
        result = analyze_column(name, values, threshold)
        if result is not None:
            drifting.append(result)
    drifting.sort(key=lambda d: d.significance, reverse=True)
    return DriftReport(drifting=tuple(drifting), evaluated=evaluated)


def format_delta(drift: ColumnDrift) -> str:
    """Render one drifter's signed change for display.

    Parameters
    ----------
    drift : ColumnDrift
        The column's report.

    Returns
    -------
    str
        ``"+0.29 %"`` style when a percentage is meaningful, else the raw
        signed delta (``"−3.2"`` style, general format).
    """
    if drift.percent is not None:
        return f"{drift.percent:+.2g} %"
    return f"{drift.delta:+.3g}"
