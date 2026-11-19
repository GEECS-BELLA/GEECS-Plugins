"""InterlockSet — the experiment's software interlocks, watched and reported.

An interlock watches one or two device variables. While its condition is
unsafe the interlock server reports it TRIPPED on its status stream, which
Master Control reads; once the condition has been safe again for
``settle_s`` seconds it reports OK. The server only reports — what a trip
stops is Master Control's decision. You would edit this file to watch a
known hazard: a vacuum gauge rising, a camera saturating, a beam drifting
off its reference.

Every interlock fails safe: a device that is disconnected, has not
answered yet, or sends a value that is not a number counts as unsafe.

Developer notes
---------------
The condition kinds keep the semantics of the original standalone
package's ``ThresholdCheck`` and ``AlignmentCheck``, plus a staleness check.
Its ``CustomCheck`` (an arbitrary Python predicate) is not carried over:
code does not belong in a config file. The reader is ``geecs_interlocks``
(GEECS-Interlocks).
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Union

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, Tier, VersionedSchemaModel

#: The experiment's interlocks, at the top of its configs folder.
INTERLOCKS_FILENAME = "interlocks.yaml"

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

    Always unsafe while the device's subscription is disconnected.  With
    ``max_age_s`` it is also unsafe when no new update has arrived from the
    device for that long — only meaningful for a free-running device that
    keeps publishing between shots.
    """

    kind: Literal["stale"] = Field(
        description="Condition type. 'stale' trips when a device stops publishing."
    )
    device: str = Field(description="Device to watch.")
    max_age_s: float | None = Field(
        None,
        gt=0,
        description=(
            "Seconds without a new update before the device counts as "
            "stale. Leave empty to trip only on a disconnect."
        ),
    )


InterlockCondition = Annotated[
    Union[ThresholdCondition, AlignmentCondition, StaleCondition],
    Field(discriminator="kind"),
]


class Interlock(SchemaModel):
    """One interlock: a name and the condition that makes it unsafe."""

    name: str = Field(
        min_length=1,
        description="Short name shown to operators and on the status stream.",
    )
    description: str = Field(
        "", description="Optional note on what this interlock protects."
    )
    enabled: bool = Field(
        True,
        description=(
            "Set false to keep the entry without arming it (shown as not "
            "armed, never counted in the verdict)."
        ),
    )
    condition: InterlockCondition = Field(
        description="The condition that makes this interlock unsafe."
    )
    settle_s: float = Field(
        0.0,
        ge=0,
        description=(
            "Seconds the condition must stay safe after a trip before the "
            "interlock reports OK again."
        ),
    )


class InterlockSet(VersionedSchemaModel):
    """The experiment's software interlocks, watched by the interlock server.

    Notes
    -----
    Stored as ``interlocks.yaml`` at the top of the experiment's scanner
    configs.  The interlock server reads it at start and on a reload from
    its window; an edit takes effect at the next reload.
    """

    tier: ClassVar[Tier] = "operator"

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
    "INTERLOCKS_FILENAME",
]
