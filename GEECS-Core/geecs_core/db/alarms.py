"""Alarm-limit policy for gateway readback PVs.

The GEECS database's existing ``min``/``max`` metadata is not alarm policy:
those fields are command/display metadata and are often meaningless for
read-only variables.  This module models the optional ``ca_alarm_limits`` table,
which is an explicit, curated overlay for value-based CA alarms.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AlarmSeverityName(str, Enum):
    """EPICS alarm severities supported by the curated limit overlay."""

    MINOR = "MINOR"
    MAJOR = "MAJOR"
    INVALID = "INVALID"


class AlarmLevel(str, Enum):
    """Named scalar alarm thresholds in EPICS limit-alarm order."""

    LOLO = "LOLO"
    LOW = "LOW"
    HIGH = "HIGH"
    HIHI = "HIHI"


class AlarmEvaluation(BaseModel):
    """Result of evaluating one value against an alarm limit policy."""

    model_config = ConfigDict(frozen=True)

    level: AlarmLevel
    severity: AlarmSeverityName


class AlarmLimits(BaseModel):
    """Curated scalar alarm thresholds for one gateway readback PV.

    ``None`` means that threshold is not configured.  Hysteresis, when set,
    delays clearing an already-active threshold until the value has moved back
    inside the limit by at least that amount.
    """

    model_config = ConfigDict(extra="forbid")

    lolo: float | None = None
    low: float | None = None
    high: float | None = None
    hihi: float | None = None
    lolo_severity: AlarmSeverityName = AlarmSeverityName.MAJOR
    low_severity: AlarmSeverityName = AlarmSeverityName.MINOR
    high_severity: AlarmSeverityName = AlarmSeverityName.MINOR
    hihi_severity: AlarmSeverityName = AlarmSeverityName.MAJOR
    hysteresis: float | None = Field(default=None, ge=0)
    description: str = ""

    @model_validator(mode="after")
    def _validate_limits(self) -> "AlarmLimits":
        """Require at least one threshold and enforce monotonic ordering."""
        ordered = [
            (AlarmLevel.LOLO, self.lolo),
            (AlarmLevel.LOW, self.low),
            (AlarmLevel.HIGH, self.high),
            (AlarmLevel.HIHI, self.hihi),
        ]
        configured = [(name, value) for name, value in ordered if value is not None]
        if not configured:
            raise ValueError("at least one alarm threshold must be configured")
        for (left_name, left), (right_name, right) in zip(configured, configured[1:]):
            if left >= right:
                raise ValueError(
                    f"alarm limits must increase from LOLO to HIHI; "
                    f"{left_name.value}={left} is not below {right_name.value}={right}"
                )
        return self

    def evaluate(
        self, value: float, previous: AlarmLevel | None = None
    ) -> AlarmEvaluation | None:
        """Evaluate *value* and return its active alarm, if any.

        Parameters
        ----------
        value : float
            Live scalar readback value.
        previous : AlarmLevel, optional
            Previously active threshold for hysteresis handling.

        Returns
        -------
        AlarmEvaluation or None
            Active threshold/severity, or ``None`` when the value is normal.
        """
        current = self._direct_evaluate(value)
        if self.hysteresis is None or self.hysteresis == 0 or previous is None:
            return current
        if current is not None and _more_extreme(current.level, previous):
            return current
        if self._inside_hysteresis_band(value, previous):
            return self._evaluation_for(previous)
        return current

    def _direct_evaluate(self, value: float) -> AlarmEvaluation | None:
        """Evaluate without considering hysteresis."""
        if self.hihi is not None and value >= self.hihi:
            return self._evaluation_for(AlarmLevel.HIHI)
        if self.high is not None and value >= self.high:
            return self._evaluation_for(AlarmLevel.HIGH)
        if self.lolo is not None and value <= self.lolo:
            return self._evaluation_for(AlarmLevel.LOLO)
        if self.low is not None and value <= self.low:
            return self._evaluation_for(AlarmLevel.LOW)
        return None

    def _inside_hysteresis_band(self, value: float, level: AlarmLevel) -> bool:
        """Return whether *level* should remain active under hysteresis."""
        if self.hysteresis is None:
            return False
        if level is AlarmLevel.HIHI and self.hihi is not None:
            return value >= self.hihi - self.hysteresis
        if level is AlarmLevel.HIGH and self.high is not None:
            return value >= self.high - self.hysteresis
        if level is AlarmLevel.LOLO and self.lolo is not None:
            return value <= self.lolo + self.hysteresis
        if level is AlarmLevel.LOW and self.low is not None:
            return value <= self.low + self.hysteresis
        return False

    def _evaluation_for(self, level: AlarmLevel) -> AlarmEvaluation:
        """Build the evaluation object for *level*."""
        severities = {
            AlarmLevel.LOLO: self.lolo_severity,
            AlarmLevel.LOW: self.low_severity,
            AlarmLevel.HIGH: self.high_severity,
            AlarmLevel.HIHI: self.hihi_severity,
        }
        return AlarmEvaluation(level=level, severity=severities[level])


def _more_extreme(current: AlarmLevel, previous: AlarmLevel) -> bool:
    """Return whether *current* is farther out of range than *previous*."""
    more_extreme_than = {
        AlarmLevel.HIGH: {AlarmLevel.HIHI},
        AlarmLevel.LOW: {AlarmLevel.LOLO},
    }
    return current in more_extreme_than.get(previous, set())
