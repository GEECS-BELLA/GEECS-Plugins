"""InterlockSet — the experiment's software interlocks, watched and reported.

An interlock holds one or more conditions and is unsafe while any of them
is (a logical OR), so one interlock can group several checks under one
name — a camera's saturation and its pointing, say. While it is unsafe the
interlock server reports it TRIPPED on its status stream, which Master
Control reads; once every condition has been safe again for ``settle_s``
seconds it reports OK. The server only reports — what a trip stops is
Master Control's decision. You would edit this file to watch a known
hazard: a vacuum gauge rising, a camera saturating, a beam drifting off its
reference, a motion controller signalling that it is moving.

Every interlock fails safe: a device that is disconnected, has not
answered yet, has gone silent for ``stale_after_s``, or sends a value that
is not a number counts as unsafe; so does an external signal before its
first packet, and a configuration-change condition before it is armed.

An experiment keeps any number of named interlock configurations, one file
each under ``interlocks/``; the server runs one at a time.

Developer notes
---------------
The condition kinds keep the semantics of the original standalone
package's ``ThresholdCheck``, ``AlignmentCheck``, ``ConfigurationChangeCheck``
and ``ExternalStatusCheck``, plus a staleness check; its ``MultiCheck`` is an
interlock's ``conditions`` list, and its multi-axis alignment one alignment
condition per axis. Its ``CustomCheck`` (an arbitrary Python predicate) is
not carried over: code does not belong in a config file. The reader is
``geecs_interlocks`` (GEECS-Interlocks).
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal, Union

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, Tier, VersionedSchemaModel

#: Where an experiment's interlock configurations live: ``interlocks/<name>.yaml``.
INTERLOCKS_FOLDER = "interlocks"

#: The comparison operators a threshold condition accepts.
ThresholdOperator = Literal["<", "<=", ">", ">=", "==", "!="]


class ThresholdCondition(SchemaModel):
    """Unsafe when a variable crosses a limit.

    For example ``operator: '>'`` with ``limit: 4094`` trips when a camera's
    ``MaxCounts`` saturates. ``==`` and ``!=`` are for status codes and
    other whole numbers: ``operator: '!='`` with ``limit: 0`` trips on any
    nonzero fault code.
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

    @model_validator(mode="after")
    def _no_hysteresis_on_equality(self) -> ThresholdCondition:
        """Refuse a hysteresis on '==' / '!=', which have no direction to come back in."""
        if self.operator in ("==", "!=") and self.hysteresis:
            raise ValueError(
                f"hysteresis has no meaning with operator {self.operator!r}: leave it 0"
            )
        return self


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


class ExternalCondition(SchemaModel):
    """Unsafe while an external program says so over UDP.

    For a program outside GEECS — a motion controller on another computer,
    say — that sends one short text packet to the interlock server whenever
    its state changes. The packet's first word decides (case does not
    matter): ``warning`` or ``unsafe`` makes the condition unsafe; ``safe``
    makes it safe unless a negation follows (``not``, ``no``, ``false``,
    ``off``, ``0``). Any other packet makes the signal unknown, which is
    unsafe until a recognised packet arrives; so is the time before the
    first packet.
    """

    kind: Literal["external"] = Field(
        description="Condition type. 'external' follows UDP status packets."
    )
    port: int = Field(
        ge=1,
        le=65535,
        description=(
            "UDP port the interlock server listens on for this signal. "
            "Conditions naming the same port share one signal."
        ),
    )
    source: str = Field(
        "",
        description="Optional name of the sending program, shown to operators.",
    )
    max_age_s: float | None = Field(
        None,
        gt=0,
        description=(
            "Seconds without a packet before the signal counts as lost. Leave "
            "empty when the sender transmits only on a change (silence is normal)."
        ),
    )


class ConfigChangeCondition(SchemaModel):
    """Unsafe when device settings change from the values they had when armed.

    The operator arms it from the interlock server's window once the
    settings are right; until then it is unsafe. Any later change to one of
    the variables trips it until it is re-armed.
    """

    kind: Literal["config_change"] = Field(
        description=(
            "Condition type. 'config_change' trips when settings move from "
            "their armed values."
        )
    )
    device: str = Field(description="Device carrying the settings.")
    variables: list[str] = Field(
        min_length=1, description="The settings to hold, e.g. ['Exposure', 'Gain']."
    )

    @model_validator(mode="after")
    def _unique_variables(self) -> ConfigChangeCondition:
        """Refuse a variable listed twice."""
        if len(set(self.variables)) != len(self.variables):
            raise ValueError(f"a variable is listed twice in {self.variables}")
        return self


InterlockCondition = Annotated[
    Union[
        ThresholdCondition,
        AlignmentCondition,
        StaleCondition,
        ExternalCondition,
        ConfigChangeCondition,
    ],
    Field(discriminator="kind"),
]


class Interlock(SchemaModel):
    """One interlock: a name and the conditions that make it unsafe (any one is enough)."""

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
    conditions: list[InterlockCondition] = Field(
        min_length=1,
        description=(
            "The conditions watched; the interlock is unsafe while any one of "
            "them is. List several to report them under one name."
        ),
    )
    settle_s: float = Field(
        0.0,
        ge=0,
        description=(
            "Seconds every condition must stay safe after a trip before the "
            "interlock reports OK again."
        ),
    )


class InterlockSet(VersionedSchemaModel):
    """The experiment's software interlocks, watched by the interlock server.

    Notes
    -----
    One named configuration, stored as ``interlocks/<name>.yaml`` in the
    experiment's scanner configs.  The interlock server reads it at start
    and on a reload or save from its window; an edit takes effect at the
    next reload.
    """

    tier: ClassVar[Tier] = "operator"

    interlocks: list[Interlock] = Field(
        default_factory=list, description="The interlocks, in display order."
    )
    stale_after_s: float | None = Field(
        5.0,
        gt=0,
        description=(
            "Seconds without an update from a device before every interlock "
            "reading it trips. Leave empty for devices that publish only on "
            "a change (a disconnect still trips)."
        ),
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
    "ConfigChangeCondition",
    "ExternalCondition",
    "Interlock",
    "InterlockCondition",
    "InterlockSet",
    "StaleCondition",
    "ThresholdCondition",
    "ThresholdOperator",
    "INTERLOCKS_FOLDER",
]
