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
from pathlib import Path
from typing import Annotated, ClassVar, Literal, SupportsFloat, Union, cast

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from geecs_schemas._base import SchemaModel, Tier, VersionedSchemaModel
from geecs_schemas.action_plan import WaitStep

#: Where an experiment's setups live: ``setups/<name>.yaml``, one per setup.
SETUPS_FOLDER = "setups"

#: A setup's name is its file stem: a plain word, no path separators, and no
#: ``.yaml`` / ``.yml`` of its own (that would name ``x.yaml.yaml``).
_SETUP_NAME = r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$"

#: The suffixes a setup file may carry, the written one first.
_SETUP_SUFFIXES = (".yaml", ".yml")


def _no_yaml_suffix(name: str) -> str:
    if name.lower().endswith(_SETUP_SUFFIXES):
        raise ValueError(
            f"a setup's name is its file stem, without the suffix: {name!r}"
        )
    return name


#: A setup's name, as its file stem and as an ``apply`` step names it.
SetupName = Annotated[str, Field(pattern=_SETUP_NAME), AfterValidator(_no_yaml_suffix)]


def _refuse_bool(value: object) -> object:
    """Refuse a boolean: YAML reads an unquoted ``on`` / ``off`` / ``yes`` / ``no`` as one."""
    if isinstance(value, bool):
        raise ValueError(
            f"{value!r} is how YAML reads an unquoted on/off/yes/no/true/false: "
            "quote the word the device expects (value: 'on')"
        )
    return value


#: A value written to or expected from a device: a word, an integer or a
#: float, each kept as written (no coercion between them), never a boolean.
SetupValue = Annotated[
    Union[StrictStr, StrictInt, StrictFloat], BeforeValidator(_refuse_bool)
]

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
    value: SetupValue = Field(
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
    expected: SetupValue = Field(
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
    setup: SetupName = Field(description="The other setup's name (its file stem).")


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

    name: SetupName = Field(
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
    ValueError
        When a key of *setups* is not its setup's name.
    SetupNotFoundError
        When an ``apply`` step names a setup missing from *setups*.
    SetupCycleError
        When nested ``apply`` steps form a loop.
    """
    for key, value in setups.items():
        if key != value.name:
            raise ValueError(
                f"setup {value.name!r} is listed under another name, {key!r}"
            )
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
    expected: str | int | float, actual: object, tolerance: float = 0.0
) -> bool:
    """Whether a reading *actual* matches a ``check``'s *expected*, as the confirm settable matches.

    A numeric *expected* floats the reading (anything ``float()`` takes: a
    number, a boolean, a numeric string) and matches within *tolerance*,
    with a few ULPs of slack; a reading that does not float, or is not
    finite, never matches. A word *expected* matches only a string reading
    equal to it: a numeric-looking word is still a word.

    Parameters
    ----------
    expected : str | int | float
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
    try:
        reading = float(cast(SupportsFloat, actual))  # whatever float() takes
    except (TypeError, ValueError):
        return False
    target = float(expected)
    if not (math.isfinite(reading) and math.isfinite(target)):
        return False
    slack = _ULP_SLACK * max(abs(reading), abs(target))
    return abs(reading - target) <= tolerance + slack


def load_setups(experiment_dir: str | Path) -> dict[str, Setup]:
    """Read every setup of an experiment: ``<experiment_dir>/setups/<name>.yaml`` (or ``.yml``).

    The one reader every executor and editor shares. A missing ``setups``
    folder is no setups (it is never created here).

    Parameters
    ----------
    experiment_dir : str or Path
        The experiment's configs folder.

    Returns
    -------
    dict of str to Setup
        The setups by name, in file-name order.

    Raises
    ------
    ValueError
        When a file does not validate (``pydantic.ValidationError``), its
        ``name`` is not its file stem, or two files share a stem.
    """
    folder = Path(experiment_dir) / SETUPS_FOLDER
    if not folder.is_dir():
        return {}
    setups: dict[str, Setup] = {}
    paths: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.suffix not in _SETUP_SUFFIXES or not path.is_file():
            continue
        if path.stem in paths:
            raise ValueError(
                f"two files name the setup {path.stem!r}: {paths[path.stem].name} "
                f"and {path.name}; keep one"
            )
        setup = Setup.from_path(path)
        if setup.name != path.stem:
            raise ValueError(
                f"{path.name} holds the setup {setup.name!r}; a setup's name is its "
                "file stem, so rename the file or the setup"
            )
        paths[path.stem] = path
        setups[setup.name] = setup
    return setups
