"""InterlockSet — software interlocks that pause a scan while a condition is unsafe.

An interlock watches one or two device variables during a scan.  When its
condition turns unsafe the scan pauses (the trigger box stops firing), the
interlock's ``on_trip`` action plan runs once, and the scan resumes by
itself once the condition has been safe again for ``settle_s`` seconds
(after the optional ``before_resume`` plan).  You would edit this file to
protect a scan from a known hazard — a vacuum gauge rising, a camera
saturating, a beam drifting off its reference.

Every interlock also fails safe: a device the CA gateway reports as
disconnected, or a reading it marks INVALID, counts as unsafe.

Developer notes
---------------
Successor of the standalone ``geecs_interlocks`` package (branch
``refactor/softwareInterlock``, built on the deleted GEECS-PythonAPI).  The
condition kinds keep that package's semantics (``ThresholdCheck``,
``AlignmentCheck``, staleness); its ``CustomCheck`` (an arbitrary Python
predicate) is not carried over — code belongs in plans, not config files.
The worker (``geecs_bluesky.interlocks``) turns each entry into a Bluesky
suspender; action-plan names are checked against the experiment's library
when the worker installs them, not here.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel

#: The comparison operators a threshold condition accepts.
ThresholdOperator = Literal["<", "<=", ">", ">="]


class ThresholdCondition(SchemaModel):
    """Unsafe when a variable crosses a limit.

    For example ``operator: '>'`` with ``limit: 4094`` trips when a camera's
    ``MaxCounts`` saturates.
    """

    kind: Literal["threshold"] = Field(
        description="Condition type. 'threshold' compares one variable to a limit."
    )
    device: str = Field(description="Device to watch, e.g. 'U_HP_Vacuum'.")
    variable: str = Field(description="Variable on the device, e.g. 'Pressure'.")
    operator: ThresholdOperator = Field(
        description=(
            "The reading is UNSAFE when 'value <operator> limit' holds, e.g. "
            "'>' trips above the limit."
        )
    )
    limit: float = Field(description="The limit the reading is compared against.")
    hysteresis: float = Field(
        0.0,
        ge=0,
        description=(
            "How far back past the limit the reading must come before the "
            "interlock counts as safe again (0 = resume as soon as it is back)."
        ),
    )


class AlignmentCondition(SchemaModel):
    """Unsafe when a variable strays from a target variable by ``tolerance`` or more.

    For example a beam position compared against its reference setpoint.
    """

    kind: Literal["alignment"] = Field(
        description=(
            "Condition type. 'alignment' compares a variable with a target variable."
        )
    )
    device: str = Field(description="Device carrying the watched variable.")
    variable: str = Field(description="The watched variable, e.g. 'CentroidX'.")
    target_device: str | None = Field(
        None,
        description="Device carrying the target variable (default: the same device).",
    )
    target_variable: str = Field(description="The target variable, e.g. 'TargetX'.")
    tolerance: float = Field(
        gt=0,
        description=(
            "UNSAFE when |value - target| is this large or larger "
            "(in the variable's units)."
        ),
    )
    hysteresis: float = Field(
        0.0,
        ge=0,
        description=(
            "How much closer than the tolerance the value must come before the "
            "interlock counts as safe again."
        ),
    )

    @model_validator(mode="after")
    def _hysteresis_inside_tolerance(self) -> AlignmentCondition:
        """Refuse a hysteresis that would make the interlock never resume."""
        if self.hysteresis >= self.tolerance:
            raise ValueError(
                f"hysteresis ({self.hysteresis}) must be smaller than the "
                f"tolerance ({self.tolerance}), or the interlock could never resume"
            )
        return self


class StaleCondition(SchemaModel):
    """Unsafe when a device stops publishing.

    Always unsafe while the gateway reports the device disconnected.  With
    ``max_age_s`` it is also unsafe when the device's ``acq_timestamp`` has
    not advanced for that long — only meaningful for a free-running device
    that keeps publishing while the scan is paused.
    """

    kind: Literal["stale"] = Field(
        description="Condition type. 'stale' trips when a device stops publishing."
    )
    device: str = Field(description="Device to watch.")
    max_age_s: float | None = Field(
        None,
        gt=0,
        description=(
            "Seconds without a new acq_timestamp before the device counts as "
            "stale. Leave empty to trip only on a disconnect."
        ),
    )


InterlockCondition = Annotated[
    Union[ThresholdCondition, AlignmentCondition, StaleCondition],
    Field(discriminator="kind"),
]


class Interlock(SchemaModel):
    """One interlock: a condition, and what to do when it trips."""

    name: str = Field(
        min_length=1,
        description="Short name shown to operators and on the status stream.",
    )
    description: str = Field(
        "", description="Optional note on what this interlock protects."
    )
    enabled: bool = Field(
        True, description="Set false to keep the entry without arming it."
    )
    condition: InterlockCondition = Field(
        description="The condition that makes this interlock unsafe."
    )
    on_trip: str | None = Field(
        None,
        description=(
            "Action plan (from the experiment's action library) to run once "
            "when the interlock trips, e.g. 'close_shutters'."
        ),
    )
    before_resume: str | None = Field(
        None,
        description=(
            "Action plan to run once the condition is safe again, just before "
            "the scan resumes."
        ),
    )
    settle_s: float = Field(
        0.0,
        ge=0,
        description=(
            "Seconds to wait after the condition is safe again before resuming."
        ),
    )


class InterlockSet(VersionedSchemaModel):
    """The experiment's software interlocks, armed on the scan worker.

    Notes
    -----
    Stored as ``interlocks.yaml`` in the experiment's scanner configs.  The
    worker reads it when its environment opens; an edit takes effect at the
    next environment open.
    """

    interlocks: list[Interlock] = Field(
        default_factory=list, description="The interlocks, in display order."
    )

    @model_validator(mode="after")
    def _unique_names(self) -> InterlockSet:
        """Refuse two interlocks with the same name (the status stream keys on it)."""
        seen: set[str] = set()
        for interlock in self.interlocks:
            if interlock.name in seen:
                raise ValueError(f"duplicate interlock name {interlock.name!r}")
            seen.add(interlock.name)
        return self


__all__ = [
    "AlignmentCondition",
    "Interlock",
    "InterlockCondition",
    "InterlockSet",
    "StaleCondition",
    "ThresholdCondition",
    "ThresholdOperator",
]
