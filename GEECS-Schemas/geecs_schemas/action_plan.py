"""ActionPlan — named sequences of device actions run around and during scans.

An action plan is a checklist the scanner executes for you: set a device
variable, wait a few seconds, check that a readback came back as expected, or
run another named plan.  You would edit one to automate a routine procedure —
"close the shutters, insert the beam stops, verify they are in" — so it runs
the same way every time instead of being clicked through by hand.  Plans are
referenced by name from a :class:`~geecs_schemas.scan_request.ScanRequest`
(``actions: {setup: [...], per_step: [...], closeout: [...]}``).

Developer notes
---------------
This is the successor of the deleted legacy scanner's action library
(tag ``legacy-scanner-final``).  Step semantics are carried over
verbatim:

- ``set`` — legacy ``SetStep`` (``wait_for_execution`` default ``True``).
- ``wait`` — legacy ``WaitStep`` (``wait`` renamed to ``seconds``).
- ``check`` — legacy ``GetStep`` (``expected_value`` renamed to ``expected``);
  a mismatch aborts the plan, exactly like the legacy ActionManager.
- ``run`` — legacy ``ExecuteStep`` (``action_name`` renamed to ``plan``);
  runs another named plan, so plans compose and nest.

The legacy ``RunStep`` (execute an external Python script/class) is
**deliberately not carried into v1** — no config in the corpus used it, and
script execution belongs in Bluesky plans, not config files.

There is no converter from the legacy ``actions:`` dialect any more: the
corpus was regenerated once as ``ActionPlanLibrary`` documents (0.22.0) and
is authored v1-only since; a consumer meeting the legacy shape refuses it.

One step library, two executors: GeecsBluesky compiles a plan into
Bluesky plan stubs inside a scan, and GeecsCanopy's module runs one from a
screen button.  Both walk it with :func:`flatten_action_steps` (nested
``run`` steps inlined, in execution order) and judge a ``check`` with
:func:`values_match`, so they agree on the order and the comparison.  How
a ``set`` waits is each executor's: the module's gate always waits for
the put (``wait_for_execution`` is honoured in a scan only).
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from geecs_schemas._base import SchemaModel, VersionedSchemaModel

#: Where an experiment's library lives in its configs folder: ``action_library/actions.yaml``.
ACTION_LIBRARY_FOLDER, ACTION_LIBRARY_FILENAME = "action_library", "actions.yaml"

# Values written to / expected from a device variable travel the GEECS wire
# protocol, which accepts words ("on"), numbers, or enum strings — hence the
# str | float | int union rather than a single scalar type.
ActionValue = Union[str, float, int]


class SetStep(SchemaModel):
    """One step that sets a device variable to a value.

    Use this to command hardware as part of a plan — for example setting a
    PLC output to ``'on'`` or driving a magnet current to 0.

    Notes
    -----
    Mirrors the legacy ``SetStep``.  ``wait_for_execution`` keeps its legacy
    default of ``True`` (the plan blocks until the device confirms).
    """

    do: Literal["set"] = Field(
        description="Step type. 'set' writes a value to a device variable."
    )
    device: str = Field(description="Name of the device to command, e.g. 'U_148_PLC'.")
    variable: str = Field(
        description="Which variable on the device to set, e.g. 'DO.Ch9'."
    )
    value: ActionValue = Field(
        description=(
            "The value to write — a number, or a word the device understands "
            "such as 'on' or 'off'."
        )
    )
    wait_for_execution: bool = Field(
        True,
        description=(
            "Wait for the device to confirm the change before moving to the "
            "next step. Leave on unless you know the step is fire-and-forget."
        ),
    )


class WaitStep(SchemaModel):
    """One step that simply pauses for a number of seconds.

    Use this to give hardware time to settle — for example waiting 3 seconds
    after moving a shutter before checking that it arrived.
    """

    do: Literal["wait"] = Field(
        description="Step type. 'wait' pauses the plan for a fixed time."
    )
    seconds: float = Field(
        gt=0,
        description="How long to pause, in seconds (must be greater than 0).",
    )


class CheckStep(SchemaModel):
    """One step that reads a device variable and verifies its value.

    Use this to confirm hardware actually did what you asked — for example
    reading a limit-switch input and requiring it to be ``'on'``.  If the
    reading does not match, the plan stops with an error instead of carrying
    on blindly.

    Notes
    -----
    Mirrors the legacy ``GetStep`` (``expected_value`` → ``expected``).  The
    mismatch-aborts behaviour is inherited from the legacy ActionManager.
    """

    do: Literal["check"] = Field(
        description=(
            "Step type. 'check' reads a device variable and stops the plan "
            "with an error if the value is not what you expected."
        )
    )
    device: str = Field(
        description="Name of the device to read, e.g. 'U_GaiaSVEReader'."
    )
    variable: str = Field(
        description="Which variable on the device to read, e.g. 'InternalShutterA'."
    )
    expected: ActionValue = Field(
        description=(
            "The value the reading must match for the plan to continue — a "
            "number or a word such as 'on'."
        )
    )


class RunPlanStep(SchemaModel):
    """One step that runs another named action plan.

    Use this to build bigger procedures out of smaller ones — for example an
    ``experiment_CLOSEOUT`` plan that runs ``zero_steering_magnets`` and then
    ``close_shutters``.

    Notes
    -----
    Mirrors the legacy ``ExecuteStep`` (``action_name`` → ``plan``).  The
    referenced name must exist in the same
    :class:`ActionPlanLibrary`; the library validator enforces this.
    """

    do: Literal["run"] = Field(
        description="Step type. 'run' executes another named plan from the library."
    )
    plan: str = Field(
        description="Name of the plan to run, as listed in the action library."
    )


ActionStep = Annotated[
    Union[SetStep, WaitStep, CheckStep, RunPlanStep],
    Field(discriminator="do"),
]


class ActionPlan(VersionedSchemaModel):
    """An ordered checklist of steps the scanner runs for you.

    Each step is one of: ``set`` a device variable, ``wait`` some seconds,
    ``check`` a readback, or ``run`` another named plan.  Steps run strictly
    in order, top to bottom.  You would edit a plan to capture a routine
    procedure so it is executed identically every time.

    Notes
    -----
    Plans are stored by name in an :class:`ActionPlanLibrary` and referenced
    by that name from ``ScanRequest.actions`` (``setup`` / ``per_step`` /
    ``closeout`` slots).
    """

    steps: list[ActionStep] = Field(
        min_length=1,
        description="The steps to perform, in order from top to bottom.",
    )
    description: str = Field(
        "",
        description=(
            "Optional note to your future self about what this plan does and "
            "when to use it."
        ),
    )


class ActionPlanLibrary(VersionedSchemaModel):
    """The collection of all named action plans for an experiment.

    Plans in the library can be picked by name in a scan's setup / per-step /
    closeout slots, and can run each other by name.  You would edit this file
    to add a new automated procedure or rename an existing one.

    Notes
    -----
    Successor of the legacy ``ActionLibrary`` (``actions:`` mapping).  The
    validator checks every nested ``run`` reference so a renamed plan cannot
    leave a dangling pointer behind.
    """

    plans: dict[str, ActionPlan] = Field(
        description="All named plans, keyed by the name used to invoke them."
    )

    def flatten(self, name: str) -> list[tuple[ConcreteStep, str | None]]:
        """The plan *name*'s concrete steps in execution order (:func:`flatten_action_steps`).

        Raises
        ------
        ActionPlanNotFoundError
            When *name*, or a nested ``run`` target, is not in the library.
        ActionPlanCycleError
            When nested ``run`` steps form a loop.
        """
        plan = self.plans.get(name)
        if plan is None:
            raise ActionPlanNotFoundError(name, list(self.plans))
        return flatten_action_steps(plan, registry=self.plans)

    @model_validator(mode="before")
    @classmethod
    def _refuse_legacy_dialect(cls, data: object) -> object:
        """Refuse the legacy ``actions:`` dialect with the regeneration named.

        There is no converter any more (0.22.0): the corpus was regenerated
        once, so a file in the old shape is a stray to regenerate, and every
        consumer (the worker's resolver, the Console's store, a listing)
        gets the same message from ``model_validate``.

        Parameters
        ----------
        data : object
            The raw input; non-mapping input passes through untouched.

        Returns
        -------
        object
            *data* unchanged.

        Raises
        ------
        ValueError
            If the document is in the legacy dialect.
        """
        if (
            data == {}
        ):  # an empty document (a fresh experiment's placeholder) is the empty library
            return {"plans": {}}
        if isinstance(data, dict) and "actions" in data and "plans" not in data:
            raise ValueError(
                "this is an action library in the legacy 'actions:' dialect, "
                "which has no converter any more (GEECS-Schemas 0.22.0) — "
                "regenerate it as an ActionPlanLibrary document "
                "(schema_version: 1, plans: {...})."
            )
        return data

    def model_post_init(self, __context: object) -> None:
        """Verify that every nested ``run`` step references a plan that exists.

        Parameters
        ----------
        __context : object
            Pydantic-internal context (unused).

        Raises
        ------
        ValueError
            If a ``run`` step names a plan not present in this library.
        """
        for name, plan in self.plans.items():
            for step in plan.steps:
                if isinstance(step, RunPlanStep) and step.plan not in self.plans:
                    raise ValueError(
                        f"Plan {name!r} has a 'run' step referencing unknown "
                        f"plan {step.plan!r}. Known plans: {sorted(self.plans)}"
                    )


def load_action_library(path: str | Path) -> ActionPlanLibrary:
    """Read an experiment's library; a missing file, an empty one or ``{}`` is the empty library (a fresh experiment's placeholder).

    Raises
    ------
    ValueError
        When the file does not validate (``pydantic.ValidationError``), or
        is in the legacy dialect.
    """
    file = Path(path)
    return ActionPlanLibrary.from_path(file) if file.exists() else ActionPlanLibrary(plans={})  # fmt: skip


#: A step that does something: what :func:`flatten_action_steps` returns.
ConcreteStep = Union[SetStep, WaitStep, CheckStep]


class ActionPlanNotFoundError(ValueError):
    """A ``run`` step (or a caller) named a plan the library does not hold."""

    def __init__(self, plan_name: str, known_plans: list[str] | None = None) -> None:
        self.plan_name = plan_name
        self.known_plans = sorted(known_plans) if known_plans else []
        super().__init__(f"action plan {plan_name!r} is not in the library")


class ActionPlanCycleError(ValueError):
    """Nested ``run`` steps form a loop; *chain* is the names that close it."""

    def __init__(self, chain: list[str]) -> None:
        self.chain = list(chain)
        super().__init__("action plans run each other in a loop: " + " -> ".join(chain))


def flatten_action_steps(
    plan: ActionPlan,
    *,
    registry: Mapping[str, ActionPlan],
) -> list[tuple[ConcreteStep, str | None]]:
    """Flatten *plan* into its concrete steps, resolving nested ``run`` steps.

    The one walk of an action plan: nested plans are inlined where their
    ``run`` step sits, in the order every executor runs them.  It touches
    no device, so it is the fail-fast check for unknown nested names and
    cycles before anything is written.

    Parameters
    ----------
    plan : ActionPlan
        The validated plan to flatten.
    registry : Mapping[str, ActionPlan]
        Named plans that ``run`` steps may reference.

    Returns
    -------
    list of (step, from_plan)
        Concrete steps (``set`` / ``wait`` / ``check``) in execution order.
        ``from_plan`` is the name of the nested plan a step was inlined
        from (the innermost enclosing ``run`` target), or ``None`` for
        *plan*'s own steps.

    Raises
    ------
    ActionPlanNotFoundError
        When a ``run`` step names a plan missing from *registry*.
    ActionPlanCycleError
        When nested ``run`` steps form a loop.
    """
    flattened: list[tuple[ConcreteStep, str | None]] = []

    def _walk(current: ActionPlan, origin: str | None, stack: tuple[str, ...]) -> None:
        for step in current.steps:
            if isinstance(step, RunPlanStep):
                if step.plan in stack:
                    raise ActionPlanCycleError([*stack, step.plan])
                nested = registry.get(step.plan)
                if nested is None:
                    raise ActionPlanNotFoundError(step.plan, list(registry))
                _walk(nested, step.plan, (*stack, step.plan))
            else:
                flattened.append((step, origin))

    _walk(plan, None, ())
    return flattened


def values_match(expected: object, actual: object) -> bool:
    """Compare a ``check`` step's reading against its expected value, legacy-style.

    The *actual* is floated first when it is a string that parses, then
    plain ``==``, no tolerance.  Quirks included: a numeric expected
    matches a device reporting ``"25"``, but a *string* expected ``"25"``
    does **not** match a numeric-looking readback.

    Parameters
    ----------
    expected : object
        The ``expected`` value from the plan (``str | float | int``).
    actual : object
        The value read back from the device.

    Returns
    -------
    bool
        True when the values match under the legacy comparison rules.
    """
    if isinstance(actual, str):
        with contextlib.suppress(ValueError):
            actual = float(actual)
    return bool(actual == expected)
