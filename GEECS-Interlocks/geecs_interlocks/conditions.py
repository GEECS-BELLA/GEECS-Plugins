"""The interlock verdict: one pure function over the latest readings.

No I/O: :func:`evaluate` takes an interlock's condition, the latest readings
of what it watches and whether it is tripped now, and answers unsafe-or-not
with the diagnostic an operator reads. The monitor owns the clock, the
settle timer and the state; this module owns the semantics.

Fail-safe throughout: a disconnected device, a missing reading, a value
that is not a number and a NaN are all unsafe.

Hysteresis applies only while tripped: the limit (or tolerance) moves toward
the safe side by ``hysteresis``, so a reading hovering at the limit does not
flicker between OK and TRIPPED on every update.
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

#: The diagnostic of a reading that has not arrived yet (the monitor shows it as "waiting").
NO_DATA = "no data"


@dataclass(frozen=True)
class Inputs:
    """Everything one evaluation reads.

    Attributes
    ----------
    value, target :
        The watched variable's latest value and, for an alignment, its
        target's (``None``: nothing received yet).
    disconnected :
        Watched devices whose subscription is down now.
    age_s :
        Seconds since the watched device's last update (a stale condition);
        ``None`` when nothing has arrived yet.
    """

    value: Any = None
    target: Any = None
    disconnected: Sequence[str] = ()
    age_s: float | None = None


@dataclass(frozen=True)
class Verdict:
    """Unsafe or not, with the operator-facing reason (empty when safe).

    ``waiting`` marks an unsafe verdict that only says "nothing has arrived
    yet", so a window can label it apart from a real trip; the wire reports
    it unsafe all the same.
    """

    unsafe: bool
    diagnostic: str = ""
    waiting: bool = False


SAFE = Verdict(False)


def watched(condition: InterlockCondition) -> list[tuple[str, str | None]]:
    """The ``(device, variable)`` pairs *condition* reads; ``None`` is the device itself."""
    if isinstance(condition, ThresholdCondition):
        return [(condition.device, condition.variable)]
    if isinstance(condition, AlignmentCondition):
        target_device = condition.target_device or condition.device
        return [
            (condition.device, condition.variable),
            (target_device, condition.target_variable),
        ]
    return [(condition.device, None)]


def describe(condition: InterlockCondition) -> str:
    """One line saying what *condition* watches, for tables and the wire's readers."""
    if isinstance(condition, ThresholdCondition):
        return (
            f"{condition.device} {condition.variable} "
            f"{condition.operator} {condition.limit:g}"
        )
    if isinstance(condition, AlignmentCondition):
        target_device = condition.target_device or condition.device
        target = (
            condition.target_variable
            if target_device == condition.device
            else f"{target_device} {condition.target_variable}"
        )
        return (
            f"{condition.device} |{condition.variable} − {target}| "
            f"< {condition.tolerance:g}"
        )
    if condition.max_age_s is None:
        return f"{condition.device} connected"
    return f"{condition.device} updating every ≤ {condition.max_age_s:g} s"


def number(value: Any) -> float | None:
    """*value* as a finite float, or ``None`` when it is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def evaluate(
    condition: InterlockCondition, inputs: Inputs, *, tripped: bool
) -> Verdict:
    """Return the verdict of *condition* over *inputs*.

    Parameters
    ----------
    condition :
        The interlock's condition (``geecs_schemas.InterlockCondition``).
    inputs :
        The latest readings.
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
        return _stale(condition, inputs.age_s)
    raise TypeError(f"unknown interlock condition {type(condition)!r}")


def _checked(name: str, value: Any) -> float | Verdict:
    """*value* as a finite float, or the unsafe verdict saying why not."""
    if value is None:
        return Verdict(True, f"{name}: {NO_DATA}", waiting=True)
    result = number(value)
    if result is None:
        return Verdict(True, f"{name}={value!r} is not a number")
    return result


def _threshold(condition: ThresholdCondition, value: Any, tripped: bool) -> Verdict:
    reading = _checked(condition.variable, value)
    if isinstance(reading, Verdict):
        return reading
    limit = condition.limit
    if tripped:
        # Back past the limit by the hysteresis before it reads safe.
        if condition.operator in (">", ">="):
            limit -= condition.hysteresis
        else:
            limit += condition.hysteresis
    if _OPERATORS[condition.operator](reading, limit):
        return Verdict(
            True,
            f"{condition.variable}={reading:g} "
            f"(limit: {condition.operator}{condition.limit:g})",
        )
    return SAFE


def _alignment(
    condition: AlignmentCondition, value: Any, target: Any, tripped: bool
) -> Verdict:
    reading = _checked(condition.variable, value)
    if isinstance(reading, Verdict):
        return reading
    reference = _checked(condition.target_variable, target)
    if isinstance(reference, Verdict):
        return reference
    tolerance = condition.tolerance - (condition.hysteresis if tripped else 0.0)
    distance = abs(reading - reference)
    if distance >= tolerance:
        return Verdict(
            True,
            f"{condition.variable}={reading:g} is {distance:g} from "
            f"{condition.target_variable}={reference:g} "
            f"(tolerance: {condition.tolerance:g})",
        )
    return SAFE


def _stale(condition: StaleCondition, age_s: float | None) -> Verdict:
    if age_s is None:
        return Verdict(True, f"{condition.device}: {NO_DATA}", waiting=True)
    max_age = condition.max_age_s
    if max_age is not None and age_s > max_age:
        return Verdict(
            True,
            f"{condition.device}: no new data for {age_s:.1f} s (max {max_age:g} s)",
        )
    return SAFE


__all__ = [
    "NO_DATA",
    "SAFE",
    "Inputs",
    "Verdict",
    "describe",
    "evaluate",
    "number",
    "watched",
]
