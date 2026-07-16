"""Typed exception hierarchy for GeecsBluesky (scan level).

Wire-protocol and database exceptions (``GeecsError`` base,
``GeecsConnectionError``, ``GeecsCommandError`` and subclasses,
``GeecsDeviceNotFoundError``) live in :mod:`geecs_ca_gateway.exceptions` — the
GEECS access layer — and are re-exported here so existing imports keep working.
Scan-level exceptions below subclass the same ``GeecsError`` base.
"""

from __future__ import annotations

from geecs_ca_gateway.exceptions import (
    GeecsCommandError,
    GeecsCommandFailedError,
    GeecsCommandRejectedError,
    GeecsConnectionError,
    GeecsDeviceNotFoundError,
    GeecsError,
)

__all__ = [
    "GeecsError",
    "GeecsConnectionError",
    "GeecsCommandError",
    "GeecsCommandRejectedError",
    "GeecsCommandFailedError",
    "GeecsDeviceNotFoundError",
    "GeecsTriggerTimeoutError",
    "GeecsQuiescenceTimeoutError",
    "GeecsMotorTimeoutError",
    "GeecsConfirmTimeoutError",
    "GeecsT0SyncError",
    "GeecsConfigurationError",
    "GeecsDeviceDownError",
    "GeecsStaleDevicesError",
    "GeecsUnservedVariablesError",
    "ActionCheckFailedError",
    "ActionPlanNotFoundError",
    "ActionPlanCycleError",
]


# ---------------------------------------------------------------------------
# Timeout errors — operation started but did not complete in time
# ---------------------------------------------------------------------------


class GeecsTriggerTimeoutError(GeecsError):
    """``acq_timestamp`` did not advance within the trigger timeout.

    Raised by :class:`~geecs_bluesky.devices.ca.triggerable.CaTriggerable`
    when no new shot arrives within ``_trigger_timeout`` seconds.  Typical
    causes: DG645 not firing, camera not acquiring, or trigger cable fault.
    """

    def __init__(self, device_name: str, timeout: float, message: str = "") -> None:
        self.device_name = device_name
        self.timeout = timeout
        super().__init__(message or f"{device_name}: no shot within {timeout:.1f}s")


class GeecsQuiescenceTimeoutError(GeecsError):
    """Free-running trigger did not stop within the timeout.

    Raised by :func:`~geecs_bluesky.plans.single_shot.geecs_confirm_quiescent`
    when device ``acq_timestamp`` values keep advancing after the shot
    controller was put in single-shot (``ARMED``) mode — so plan-owned
    single-shot firing cannot safely begin (a residual free-running shot would
    be mistaken for the plan's fired shot).  Typical cause: the ``ARMED`` state
    did not actually switch the trigger source to single-shot mode.
    """

    def __init__(self, timeout: float, message: str = "") -> None:
        self.timeout = timeout
        super().__init__(
            message
            or f"trigger still firing after {timeout:.1f}s; single-shot arm failed"
        )


class GeecsMotorTimeoutError(GeecsError):
    """Motor did not reach the target position within ``move_timeout``.

    Raised by :class:`~geecs_bluesky.devices.ca.motor.CaMotor` when the
    position polling loop expires.  Possible causes: stage stall, mechanical
    obstruction, wrong tolerance, or very long move.  Do not auto-retry —
    a stalled stage may need operator intervention.
    """

    def __init__(
        self,
        device_name: str,
        variable: str,
        target: float,
        current: float,
        timeout: float,
    ) -> None:
        self.device_name = device_name
        self.variable = variable
        self.target = target
        self.current = current
        self.timeout = timeout
        super().__init__(
            f"{device_name}/{variable}: position {current} did not reach "
            f"{target} within {timeout:.1f}s"
        )


class GeecsConfirmTimeoutError(GeecsError):
    """A confirming variable did not match the set target within ``timeout``.

    Raised by :class:`~geecs_bluesky.devices.ca.confirm.CaConfirmSettable` when
    the polling loop on the *confirming* variable (a different variable from
    the one written — the ``ScanVariable.confirm`` "topology C" case) expires.
    Possible causes: the confirming variable's tolerance is too tight for its
    real jitter, the two variables are not actually coupled the way the config
    assumes, or the device is stuck. Do not auto-retry — this is the same
    "needs operator attention" posture as :class:`GeecsMotorTimeoutError`.
    """

    def __init__(
        self,
        device_name: str,
        variable: str,
        confirm_variable: str,
        target: float | str,
        current: float | str,
        timeout: float,
    ) -> None:
        self.device_name = device_name
        self.variable = variable
        self.confirm_variable = confirm_variable
        self.target = target
        self.current = current
        self.timeout = timeout
        super().__init__(
            f"{device_name}/{variable}: confirming variable {confirm_variable} "
            f"reads {current!r}, did not match {target!r} within {timeout:.1f}s"
        )


class GeecsT0SyncError(GeecsError):
    """Coordinated t0 capture could not establish a common physical shot.

    Raised by :func:`~geecs_bluesky.plans.t0_sync.geecs_t0_sync` when device
    ``acq_timestamp`` values are spread wider than the acceptance window (the
    cached frames do not all come from the same physical trigger) or when a
    device has no cached ``acq_timestamp`` at all.  Never proceed unseeded —
    shot IDs from unsynchronized t0s are not comparable across devices.
    """

    def __init__(
        self,
        message: str,
        timestamps: dict[str, float | None] | None = None,
        window_s: float | None = None,
    ) -> None:
        self.timestamps = timestamps or {}
        self.window_s = window_s
        super().__init__(message)


# ---------------------------------------------------------------------------
# Configuration / setup errors
# ---------------------------------------------------------------------------


class GeecsConfigurationError(GeecsError):
    """Runtime configuration is incomplete or inconsistent."""


class GeecsDeviceDownError(GeecsError):
    """A device's gateway ``CONNECTED`` PV reports ``Disconnected``.

    ``CONNECTED`` is the authoritative liveness signal — CA-connect success
    never implies device liveness (PV_CONTRACT.md §1/§5; rationale in
    ``GeecsBluesky/CLAUDE.md``).  Raised, or carried inside the pre-claim
    operator dialog, by the pre-flight liveness check and by
    :func:`~geecs_bluesky.plans.single_shot.geecs_single_shot` when a
    no-frame device turns out to be disconnected mid-scan.  The message is
    operator-facing.
    """

    def __init__(self, message: str, device_name: str | None = None) -> None:
        self.device_name = device_name
        super().__init__(message)


class GeecsStaleDevicesError(GeecsError):
    """Free-run sync device(s) are CONNECTED but have no fresh frames.

    Carried inside the pre-claim operator dialog raised by the free-run
    staleness check: all-stale means the trigger is probably off / not
    free-running; a stale subset is a per-device acquisition problem.
    Genuinely *dead* devices are :class:`GeecsDeviceDownError` territory.
    The message is operator-facing.
    """


class GeecsUnservedVariablesError(GeecsError):
    """Save-set variable(s) the gateway does not serve as PVs.

    The gateway serves each enabled device's ``get='yes'`` variables plus its
    settable control surface (``GeecsCAGateway/DEPLOYMENT.md``); a save-set
    variable outside that set has no PV, so its detector signal can never
    connect (a 20 s ophyd ``NotConnectedError``, observed live 2026-07-15).
    Carried inside the pre-claim operator dialog raised by the
    unserved-variables pre-flight check.  The message is operator-facing.

    Parameters
    ----------
    message :
        The operator-facing dialog body.
    unserved :
        ``{device: [variables]}`` — the unserved variables, by device.
    """

    def __init__(
        self, message: str, unserved: dict[str, list[str]] | None = None
    ) -> None:
        self.unserved = dict(unserved or {})
        super().__init__(message)


# ---------------------------------------------------------------------------
# Action-plan errors — compiled ActionPlan execution (plans/action_compiler)
# ---------------------------------------------------------------------------


class ActionCheckFailedError(GeecsError):
    """A ``check`` step read a value that did not match what the plan expected.

    Raised by the compiled action plan (see
    :func:`~geecs_bluesky.plans.action_compiler.compile_action_plan`) when a
    ``check`` step's readback differs from its ``expected`` value.  The plan
    always stops here — a mismatch is never papered over.  The message is
    operator-facing.
    """

    def __init__(
        self,
        device: str,
        variable: str,
        expected: object,
        actual: object,
    ) -> None:
        self.device = device
        self.variable = variable
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Check failed: {device}:{variable} read back {actual!r} but the "
            f"plan expected {expected!r}. The action plan stopped here so the "
            "mismatch can be looked at before anything else runs."
        )


class ActionPlanNotFoundError(GeecsError):
    """A ``run`` step referenced a plan name that is not in the registry.

    Raised while a compiled action plan executes a nested ``run`` step whose
    ``plan`` name has no entry in the plan registry — typically a renamed or
    deleted plan that something still points at.
    """

    def __init__(self, plan_name: str, known_plans: list[str] | None = None) -> None:
        self.plan_name = plan_name
        self.known_plans = sorted(known_plans) if known_plans else []
        known = (
            f" Known plans: {', '.join(self.known_plans)}."
            if self.known_plans
            else " The registry is empty."
        )
        super().__init__(
            f"Action plan {plan_name!r} is not defined in the plan registry — "
            f"it may have been renamed or removed.{known}"
        )


class ActionPlanCycleError(GeecsError):
    """Nested ``run`` steps form a loop, so the plan would never finish.

    Raised before recursing into a nested plan whose name is already on the
    execution stack (for example plan A runs B, and B runs A again); the
    message shows the chain of plan names that closes the loop.
    """

    def __init__(self, chain: list[str]) -> None:
        self.chain = list(chain)
        super().__init__(
            "Action plans call each other in a loop and would never finish: "
            + " -> ".join(self.chain)
            + ". Remove one of the 'run' steps to break the cycle."
        )
