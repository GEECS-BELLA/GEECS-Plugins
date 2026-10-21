"""The interlock verdict: one pure function over the latest readings.

No I/O and no Bluesky: :func:`evaluate` takes an interlock's condition, the
latest cached readings of what it watches and whether it is tripped now,
and answers unsafe-or-not with the diagnostic an operator reads.  The
suspender (:mod:`geecs_bluesky.interlocks.suspender`) owns the caching and
the RunEngine; this module owns the semantics, carried over from the
retired ``geecs_interlocks`` package (``ThresholdCheck``,
``AlignmentCheck``, staleness).

Fail-safe throughout: a device the gateway reports ``Disconnected``, a
reading it marks INVALID (a dead device's stale value), a missing reading
and a non-numeric one are all unsafe.

Hysteresis applies only while tripped: the limit (or tolerance) moves
toward the safe side by ``hysteresis``, so a reading hovering at the limit
does not resume and re-trip the scan on every update.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from geecs_schemas import (
    AlignmentCondition,
    InterlockCondition,
    StaleCondition,
    ThresholdCondition,
)

_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@dataclass(frozen=True)
class Sample:
    """The latest reading of one watched signal (``value is None``: none yet)."""

    value: Any = None
    invalid: bool = False


@dataclass(frozen=True)
class Inputs:
    """Everything one evaluation reads.

    Attributes
    ----------
    value, target :
        The watched variable and, for an alignment, its target.
    disconnected :
        Watched devices whose ``CONNECTED`` PV reads ``Disconnected``.
    stamp_age_s :
        Seconds since the device's ``acq_timestamp`` last advanced (a stale
        condition with ``max_age_s``); ``None`` when not watched.
    """

    value: Sample = Sample()
    target: Sample = Sample()
    disconnected: Sequence[str] = ()
    stamp_age_s: float | None = None


@dataclass(frozen=True)
class Verdict:
    """Unsafe or not, with the operator-facing reason (empty when safe)."""

    unsafe: bool
    diagnostic: str = ""


SAFE = Verdict(False)


def evaluate(
    condition: InterlockCondition, inputs: Inputs, *, tripped: bool
) -> Verdict:
    """Return the verdict of *condition* over *inputs*.

    Parameters
    ----------
    condition :
        The interlock's condition (``geecs_schemas.InterlockCondition``).
    inputs :
        The latest cached readings.
    tripped :
        Whether the interlock is tripped now (selects the hysteresis side).
    """
    if inputs.disconnected:
        return Verdict(True, f"{', '.join(inputs.disconnected)} disconnected")
    if isinstance(condition, ThresholdCondition):
        return _threshold(condition, inputs.value, tripped)
    if isinstance(condition, AlignmentCondition):
        return _alignment(condition, inputs.value, inputs.target, tripped)
    if isinstance(condition, StaleCondition):
        return _stale(condition, inputs.stamp_age_s)
    raise TypeError(f"unknown interlock condition {type(condition)!r}")


def _number(name: str, sample: Sample) -> float | Verdict:
    """The sample as a finite float, or the unsafe verdict saying why not."""
    if sample.invalid:
        return Verdict(True, f"{name} INVALID (device down)")
    if sample.value is None:
        return Verdict(True, f"{name}: no data")
    try:
        number = float(sample.value)
    except (TypeError, ValueError):
        return Verdict(True, f"{name}={sample.value!r} is not a number")
    if math.isnan(number):
        return Verdict(True, f"{name}: NaN")
    return number


def _threshold(condition: ThresholdCondition, sample: Sample, tripped: bool) -> Verdict:
    value = _number(condition.variable, sample)
    if isinstance(value, Verdict):
        return value
    limit = condition.limit
    if tripped:
        # Back past the limit by the hysteresis before it reads safe.
        if condition.operator in (">", ">="):
            limit -= condition.hysteresis
        else:
            limit += condition.hysteresis
    if _OPERATORS[condition.operator](value, limit):
        return Verdict(
            True,
            f"{condition.variable}={value:g} "
            f"(limit: {condition.operator}{condition.limit:g})",
        )
    return SAFE


def _alignment(
    condition: AlignmentCondition, sample: Sample, target: Sample, tripped: bool
) -> Verdict:
    value = _number(condition.variable, sample)
    if isinstance(value, Verdict):
        return value
    reference = _number(condition.target_variable, target)
    if isinstance(reference, Verdict):
        return reference
    tolerance = condition.tolerance - (condition.hysteresis if tripped else 0.0)
    distance = abs(value - reference)
    if distance >= tolerance:
        return Verdict(
            True,
            f"{condition.variable}={value:g} is {distance:g} from "
            f"{condition.target_variable}={reference:g} "
            f"(tolerance: {condition.tolerance:g})",
        )
    return SAFE


def _stale(condition: StaleCondition, age_s: float | None) -> Verdict:
    max_age = condition.max_age_s
    if max_age is not None and age_s is not None and age_s > max_age:
        return Verdict(
            True,
            f"{condition.device}: no new data for {age_s:.1f} s (max {max_age:g} s)",
        )
    return SAFE


__all__ = ["SAFE", "Inputs", "Sample", "Verdict", "evaluate"]
