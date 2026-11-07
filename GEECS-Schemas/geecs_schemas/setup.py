"""Setup — a named device state, applied in order: set, wait, check, apply another.

A setup is how an experiment says "put the machine in *low power*": the
devices and the values that make the state, in the order to apply them,
with pauses to settle and checks that a readback arrived. It is usually
captured from the live readbacks (``captured`` says by whom, when and from
which readbacks) and then edited by hand. One setup per file, at
``<experiment>/setups/<name>.yaml``, so a commit touches one setup.

Developer notes
---------------
A ``set`` is the device's own completion (``abs_set(wait=True)`` on the
namespace's settable, which confirms by readback wherever the device has
one); the setup adds no second confirmation layer. A ``check`` matches as
the confirm settable does: a number within ``tolerance``, a string by
equality (:func:`setup_value_matches`). An ``apply`` step inlines another
setup by name (:func:`flatten_setup`); unknown names and loops are refused
before anything is written.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, ClassVar, Literal, Union

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, Tier, VersionedSchemaModel
from geecs_schemas.action_plan import ActionValue, WaitStep

#: Where an experiment's setups live: ``setups/<name>.yaml``, one per setup.
SETUPS_FOLDER = "setups"

#: A setup's name is its file stem: a plain word, no path separators.
_SETUP_NAME = r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$"

#: Relative float slack in a tolerance comparison, as the devices' own check
#: has it: an exactly-on-tolerance reading otherwise loses to representation
#: error (``|1.20 - 1.15| == 0.050000000000000044``).
_ULP_SLACK = 4 * 2.220446049250313e-16


class SetupSetStep(SchemaModel):
    """One step that sets a device variable and waits for the device to finish."""

    do: Literal["set"] = Field(
        description="Step type. 'set' writes a value and waits for the device."
    )
    device: str = Field(min_length=1, description="The device, e.g. 'U_S1H'.")
    variable: str = Field(min_length=1, description="The variable, e.g. 'Current'.")
    value: ActionValue = Field(
        description="The value to write: a number, or a word such as 'on'."
    )


class SetupCheckStep(SchemaModel):
    """One step that reads a device variable and stops the setup on a mismatch.

    A number matches within ``tolerance``; a word matches exactly.
    """

    do: Literal["check"] = Field(
        description="Step type. 'check' reads a variable and stops on a mismatch."
    )
    device: str = Field(min_length=1, description="The device to read.")
    variable: str = Field(min_length=1, description="The variable to read.")
    expected: ActionValue = Field(
        description="The value the reading must match: a number or a word."
    )
    tolerance: float = Field(
        0.0,
        ge=0,
        description=(
            "How far a numeric reading may be from 'expected' and still match. "
            "Only for a numeric 'expected'; a word matches exactly."
        ),
    )

    @model_validator(mode="after")
    def _tolerance_is_numeric(self) -> SetupCheckStep:
        if isinstance(self.expected, str) and self.tolerance:
            raise ValueError(
                f"check {self.device}:{self.variable} expects the word "
                f"{self.expected!r}, which matches exactly: drop 'tolerance'"
            )
        return self


class SetupApplyStep(SchemaModel):
    """One step that applies another setup, inlined where it sits."""

    do: Literal["apply"] = Field(
        description="Step type. 'apply' runs another setup's steps here."
    )
    setup: str = Field(
        pattern=_SETUP_NAME, description="The other setup's name (its file stem)."
    )


SetupStep = Annotated[
    Union[SetupSetStep, WaitStep, SetupCheckStep, SetupApplyStep],
    Field(discriminator="do"),
]

#: A step that does something: what :func:`flatten_setup` returns.
ConcreteSetupStep = Union[SetupSetStep, WaitStep, SetupCheckStep]


class CapturedReadback(SchemaModel):
    """One readback a captured value was read from."""

    device: str = Field(min_length=1, description="The device read.")
    variable: str = Field(min_length=1, description="The variable read.")


class SetupCapture(SchemaModel):
    """Where a setup's values came from: who captured them, when, from which readbacks."""

    by: str = Field(min_length=1, description="The operator who captured it.")
    at: datetime = Field(description="When it was captured.")
    readbacks: list[CapturedReadback] = Field(
        default_factory=list, description="The readbacks the values were read from."
    )


class Setup(VersionedSchemaModel):
    """A named device state: the steps that put the machine in it, in order.

    Each step is one of: ``set`` a device variable, ``wait`` some seconds,
    ``check`` a readback, or ``apply`` another setup. Steps run strictly in
    order, top to bottom; a failed ``set`` or ``check`` stops the setup.
    """

    tier: ClassVar[Tier] = "operator"

    name: str = Field(
        pattern=_SETUP_NAME,
        description="The setup's name, also its file stem, e.g. 'low_power'.",
    )
    description: str = Field("", description="What this state is and when to use it.")
    steps: list[SetupStep] = Field(
        min_length=1, description="The steps to perform, in order from top to bottom."
    )
    captured: SetupCapture | None = Field(
        None, description="Where the values came from; absent for a hand-written setup."
    )

    @model_validator(mode="after")
    def _does_not_apply_itself(self) -> Setup:
        for step in self.steps:
            if isinstance(step, SetupApplyStep) and step.setup == self.name:
                raise SetupCycleError([self.name, self.name])
        return self


class SetupNotFoundError(ValueError):
    """An ``apply`` step (or a caller) named a setup that does not exist."""

    def __init__(self, name: str, known: list[str] | None = None) -> None:
        self.name = name
        self.known = sorted(known) if known else []
        super().__init__(f"setup {name!r} does not exist")


class SetupCycleError(ValueError):
    """Nested ``apply`` steps form a loop; *chain* is the names that close it."""

    def __init__(self, chain: list[str]) -> None:
        self.chain = list(chain)
        super().__init__("setups apply each other in a loop: " + " -> ".join(chain))


def flatten_setup(
    setup: Setup, *, setups: Mapping[str, Setup]
) -> list[tuple[ConcreteSetupStep, str | None]]:
    """Flatten *setup* into its concrete steps, inlining nested ``apply`` steps.

    It touches no device, so it is the fail-fast check for unknown names and
    loops before anything is written.

    Parameters
    ----------
    setup : Setup
        The setup to flatten.
    setups : Mapping[str, Setup]
        The experiment's setups by name, which ``apply`` steps may name.

    Returns
    -------
    list of (step, from_setup)
        The ``set`` / ``wait`` / ``check`` steps in execution order, each with
        the name of the nested setup it was inlined from (the innermost), or
        ``None`` for *setup*'s own steps.

    Raises
    ------
    SetupNotFoundError
        When an ``apply`` step names a setup missing from *setups*.
    SetupCycleError
        When nested ``apply`` steps form a loop.
    """
    flattened: list[tuple[ConcreteSetupStep, str | None]] = []

    def _walk(current: Setup, origin: str | None, chain: tuple[str, ...]) -> None:
        for step in current.steps:
            if not isinstance(step, SetupApplyStep):
                flattened.append((step, origin))
                continue
            if step.setup in chain:
                raise SetupCycleError([*chain, step.setup])
            nested = setups.get(step.setup)
            if nested is None:
                raise SetupNotFoundError(step.setup, list(setups))
            _walk(nested, step.setup, (*chain, step.setup))

    _walk(setup, None, (setup.name,))
    return flattened


def setup_value_matches(
    expected: ActionValue, actual: object, tolerance: float = 0.0
) -> bool:
    """Whether a reading *actual* matches a ``check``'s *expected*, as the confirm settable matches.

    A numeric *expected* matches a reading that is a number, or a string that
    parses as one, within *tolerance* (with a few ULPs of slack); a
    non-finite reading never matches. A word *expected* matches only a
    string reading equal to it: a numeric-looking word is still a word.

    Parameters
    ----------
    expected : str | float | int
        The step's ``expected``.
    actual : object
        The value read back from the device.
    tolerance : float
        The step's ``tolerance`` (numbers only).

    Returns
    -------
    bool
        True when the reading matches.
    """
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(actual, bool) or not isinstance(actual, (int, float, str)):
        return False
    try:
        reading = float(actual)
    except ValueError:
        return False
    target = float(expected)
    if not (math.isfinite(reading) and math.isfinite(target)):
        return False
    slack = _ULP_SLACK * max(abs(reading), abs(target))
    return abs(reading - target) <= tolerance + slack
