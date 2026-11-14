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

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    ClassVar,
    Literal,
    SupportsFloat,
    Union,
    cast,
)

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

if TYPE_CHECKING:
    from geecs_schemas.epics import NativeDevices
    from geecs_schemas.epics.family import FamilyFile
    from geecs_schemas.epics.operations import OperatorPolicy
    from geecs_schemas.epics.panel_devices import PanelDevices

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


def setup_digest(steps: Sequence[tuple[ConcreteSetupStep, str | None]]) -> str:
    """The SHA-256 of a flattened setup: what it will do, nested setups included.

    Two readers that flatten the same files get the same digest, and any
    edit that changes a value, a step or its order (in the setup or in one it
    applies) changes it. An executor compares the digest it was handed with
    the one it computes before the first write, so what an operator
    confirmed is what runs.

    Parameters
    ----------
    steps : sequence of (step, from_setup)
        :func:`flatten_setup`'s answer.

    Returns
    -------
    str
        The hex digest.
    """
    canonical = [
        {"from": origin, **step.model_dump(mode="json")} for step, origin in steps
    ]
    text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


class SetupHazard(SchemaModel):
    """A ``set`` step that needs the hazard confirm, and why."""

    index: int = Field(description="The step's number in the flattened setup, from 1.")
    device: str = Field(description="The device written.")
    variable: str = Field(description="The variable written.")
    value: SetupValue = Field(description="The value written.")
    reasons: list[str] = Field(description="Why it needs the confirm, one line each.")


def _family_roles(
    device: str,
    families: Mapping[str, FamilyFile],
    native: NativeDevices | None,
    panels: PanelDevices | None,
    db_families: Mapping[str, str] | None,
) -> tuple[str | None, FamilyFile | None, dict[str, bool]]:
    """``(family, family file, role -> hazard)`` of *device*; ``(None, None, {})`` when unknown."""
    if panels is not None and device in panels.devices:
        panel = panels.devices[device]
        family = families.get(panel.family)
        roles = panel.resolve(family) if family is not None else []
        return panel.family, family, {r.name: r.confirm == "hazard" for r in roles}
    if native is not None and device in native.devices:
        name = native.devices[device].family_template
        family = families.get(name)
        roles = family.roles if family is not None else []
        return name, family, {r.name: r.confirm == "hazard" for r in roles}
    if db_families is not None and device in db_families:
        return db_families[device], None, {}
    return None, None, {}


def _role_candidates(variable: str, family: FamilyFile | None) -> set[str]:
    """Every role a step's *variable* may name on a device of *family*.

    The variable itself and the part after an axis (``m1.val``); a role
    whose name or record suffix it is (the worker names a supply's settable
    by its readback record); a bare axis's setpoint; and every role whose
    readback is one of those (the setpoint a readback's set is written at).
    Deliberately wide: asking about a role that is not the one written
    costs a question, while missing the hazardous one costs the confirm.
    """
    words = {variable}
    _, dot, rest = variable.partition(".")
    if dot and rest:
        words.add(rest)
    if family is None:
        return words
    matched = {r.name for r in family.roles if r.name in words or r.suffix in words}
    if family.per_axis and not dot:
        matched |= {r.name for r in family.roles if r.type == "setpoint" and r.readback}
    matched |= {r.name for r in family.roles if r.readback in matched}
    return words | matched


def setup_hazards(
    steps: Sequence[tuple[ConcreteSetupStep, str | None]],
    *,
    policy: OperatorPolicy,
    families: Mapping[str, FamilyFile],
    native: NativeDevices | None = None,
    panels: PanelDevices | None = None,
    db_families: Mapping[str, str] | None = None,
) -> list[SetupHazard]:
    """The ``set`` steps of a flattened setup that need the hazard confirm.

    The same two sources the device module's gate reads: a family role with
    ``confirm: hazard`` (as the device has it: ``panel_devices.yaml``'s
    ``hazard`` added), and the operator policy's ``hazards`` (by device, or
    by family: a native device's family template, a DB device's device
    type). It fails toward asking: a device whose family cannot be told
    (in neither device file, and the DB's types unavailable or silent on
    it) is a hazard whenever the policy names any of its candidate roles
    for any family.

    Parameters
    ----------
    steps : sequence of (step, from_setup)
        :func:`flatten_setup`'s answer.
    policy : OperatorPolicy
        The experiment's ``operator_policy.yaml`` (an absent file: an empty policy).
    families : Mapping[str, FamilyFile]
        The installed families (:func:`geecs_schemas.epics.family.load_families`).
    native, panels : NativeDevices, PanelDevices, optional
        The experiment's two device files.
    db_families : Mapping[str, str], optional
        DB device name to its device type; ``None`` when the DB did not answer.

    Returns
    -------
    list of SetupHazard
        In step order.
    """
    found: list[SetupHazard] = []
    for index, (step, _origin) in enumerate(steps, 1):
        if not isinstance(step, SetupSetStep):
            continue
        family_name, family, roles = _family_roles(
            step.device, families, native, panels, db_families
        )
        candidates = _role_candidates(step.variable, family)
        reasons = [
            f"the family role {role!r} asks for the hazard confirm"
            for role in sorted(candidates)
            if roles.get(role)
        ]
        for rule in policy.hazards:
            named = sorted(candidates & set(rule.roles))
            if not named:
                continue
            if step.device in rule.devices or (
                family_name is not None and family_name in rule.families
            ):
                reasons.append(
                    f"the operator policy's hazards name {step.device} {named[0]!r}"
                )
            elif family_name is None and rule.families:
                reasons.append(
                    f"the operator policy's hazards name {named[0]!r} for the families "
                    f"{sorted(rule.families)}, and the family of {step.device} is unknown"
                )
        if reasons:
            found.append(
                SetupHazard(
                    index=index,
                    device=step.device,
                    variable=step.variable,
                    value=step.value,
                    reasons=list(dict.fromkeys(reasons)),
                )
            )
    return found
