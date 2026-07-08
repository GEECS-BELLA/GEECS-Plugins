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
This is the successor of the legacy action library
(``geecs_scanner.engine.models.actions``).  Step semantics are carried over
verbatim:

- ``set`` — legacy ``SetStep`` (``wait_for_execution`` default ``True``).
- ``wait`` — legacy ``WaitStep`` (``wait`` renamed to ``seconds``).
- ``check`` — legacy ``GetStep`` (``expected_value`` renamed to ``expected``);
  a mismatch aborts the plan, exactly like the legacy ActionManager.
- ``run`` — legacy ``ExecuteStep`` (``action_name`` renamed to ``plan``);
  runs another named plan, so plans compose and nest.

The legacy ``RunStep`` (execute an external Python script/class) is
**deliberately not carried into v1** — no config in the corpus uses it, and
script execution belongs in Bluesky plans, not config files.  The converter
raises loudly if it ever encounters one.

In the target architecture (vision doc §4.5) a plan compiles to Bluesky plan
stubs, inheriting abort/logging/event emission for free.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field

from geecs_schemas._base import SchemaModel, VersionedSchemaModel

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
