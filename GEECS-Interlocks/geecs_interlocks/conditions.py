"""The interlock verdict: one pure function over the latest readings.

No I/O: :func:`evaluate` takes an interlock's condition, the latest readings
of what it watches and whether it is tripped now, and answers unsafe-or-not
with the diagnostic an operator reads. The monitor owns the clock, the
settle timer and the state; this module owns the semantics.

Fail-safe throughout: a disconnected device, a device silent past the
file's ``stale_after_s``, a missing reading, a value that is not a number,
a NaN, an external signal before its first packet and a configuration
check that is not armed are all unsafe.

Hysteresis applies only while tripped: the limit (or tolerance) moves toward
the safe side by ``hysteresis``, so a reading hovering at the limit does not
flicker between OK and TRIPPED on every update.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from geecs_interlocks.external import SignalReading

from geecs_schemas import (
    AlignmentCondition,
    ConfigChangeCondition,
    ExternalCondition,
    InterlockCondition,
    StaleCondition,
    ThresholdCondition,
)

_OPERATORS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
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
    stale :
        ``(device, seconds silent)`` of watched devices silent past the
        file's ``stale_after_s``.
    age_s :
        Seconds since the watched device's last update (a stale condition)
        or the signal's last packet (an external one); ``None`` when nothing
        has arrived yet.
    signal :
        The external signal's latest reading (an external condition).
    settings, baseline :
        A configuration-change condition's current values and the values
        recorded when it was armed (``None``: not armed).
    """

    value: Any = None
    target: Any = None
    disconnected: Sequence[str] = ()
    stale: Sequence[tuple[str, float]] = ()
    age_s: float | None = None
    signal: SignalReading | None = None
    settings: Mapping[str, Any] | None = None
    baseline: Mapping[str, Any] | None = None


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
    if isinstance(condition, ConfigChangeCondition):
        return [(condition.device, v) for v in condition.variables]
    if isinstance(condition, ExternalCondition):
        return []  # a UDP signal, not a GEECS device
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
    if isinstance(condition, ConfigChangeCondition):
        return f"{condition.device} {', '.join(condition.variables)} unchanged"
    if isinstance(condition, ExternalCondition):
        return f"{condition.source or 'external signal'} on UDP {condition.port}"
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
    if inputs.stale:
        return Verdict(
            True,
            "; ".join(f"{d}: no new data for {age:.1f} s" for d, age in inputs.stale),
        )
    if isinstance(condition, ThresholdCondition):
        return _threshold(condition, inputs.value, tripped)
    if isinstance(condition, AlignmentCondition):
        return _alignment(condition, inputs.value, inputs.target, tripped)
    if isinstance(condition, StaleCondition):
        return _stale(condition, inputs.age_s)
    if isinstance(condition, ExternalCondition):
        return _external(condition, inputs.signal, inputs.age_s)
    if isinstance(condition, ConfigChangeCondition):
        return _config_change(condition, inputs.settings or {}, inputs.baseline)
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


def _external(
    condition: ExternalCondition, signal: SignalReading | None, age_s: float | None
) -> Verdict:
    name = condition.source or f"UDP {condition.port}"
    if signal is None or signal.received is None:
        return Verdict(
            True, f"{name}: no packet yet on UDP port {condition.port}", waiting=True
        )
    max_age = condition.max_age_s
    if max_age is not None and age_s is not None and age_s > max_age:
        return Verdict(True, f"{name}: no packet for {age_s:.1f} s (max {max_age:g} s)")
    if signal.state is None:
        return Verdict(True, f"{name}: unrecognised packet {signal.raw!r}")
    if signal.state:
        return Verdict(True, f"{name}: alarm ({signal.raw!r})")
    return SAFE


def same_setting(a: Any, b: Any) -> bool:
    """Whether two readings of one setting are the same: as numbers when both are, else as text."""
    x, y = number(a), number(b)
    if x is not None and y is not None:
        return x == y
    return str(a) == str(b)


def _config_change(
    condition: ConfigChangeCondition,
    settings: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> Verdict:
    if baseline is None:
        return Verdict(True, f"{condition.device}: not armed (press Arm)")
    missing = [v for v in condition.variables if settings.get(v) is None]
    if missing:
        return Verdict(
            True, f"{condition.device} {', '.join(missing)}: {NO_DATA}", waiting=True
        )
    changed = [
        f"{v} changed from {baseline[v]} to {settings[v]}"
        for v in condition.variables
        if not same_setting(settings[v], baseline[v])
    ]
    if changed:
        return Verdict(True, f"{condition.device}: {'; '.join(changed)} (re-arm)")
    return SAFE


__all__ = [
    "NO_DATA",
    "SAFE",
    "Inputs",
    "Verdict",
    "describe",
    "evaluate",
    "number",
    "same_setting",
    "watched",
]
