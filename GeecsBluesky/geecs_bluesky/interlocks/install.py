"""Arm the experiment's interlocks on the worker's RunEngine.

:func:`install_interlocks` is called once, by the startup profile, after
the namespace and ``RE`` exist.  Per enabled interlock it resolves the
watched signals in the namespace, resolves and checks its action plans
against the experiment's library (every ``set`` / ``check`` target looked
up, as ``run_action`` does before its first write), connects every device
involved, and installs a
:class:`~geecs_bluesky.interlocks.suspender.GeecsInterlockSuspender` on
the RunEngine's loop.

A bad entry never stops environment open (the posture of
``GeecsNamespace.add_pseudos``): it is logged at ERROR and kept in the
registry as **not armed**, so the status stream reports it unsafe with the
reason rather than leaving it silently absent.

The action plans are connected here because a suspender's plans do not
pass through the RunEngine's preprocessors: ``connect_on_demand`` never
sees their messages.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from geecs_schemas import (
    ActionPlan,
    AlignmentCondition,
    CheckStep,
    Interlock,
    InterlockSet,
    SetStep,
    StaleCondition,
)

from geecs_bluesky.actions.compiler import compile_action_plan
from geecs_bluesky.actions.steps import flatten_action_steps
from geecs_bluesky.devices.background import WARM_UP_TIMEOUT_S, warm_up_on
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.exceptions import GeecsError
from geecs_bluesky.log_markers import (
    INTERLOCK_CLEARED_LOG_PREFIX,
    INTERLOCK_TRIPPED_LOG_PREFIX,
)
from geecs_bluesky.interlocks.conditions import Verdict
from geecs_bluesky.interlocks.suspender import (
    GeecsInterlockSuspender,
    Watched,
    watched_devices,
)

logger = logging.getLogger(__name__)

# The trip / clear lines are a client-parsed contract (the progress
# stream's paused reason), so their spelling lives in log_markers.
TRIPPED_LOG_PREFIX = INTERLOCK_TRIPPED_LOG_PREFIX
CLEARED_LOG_PREFIX = INTERLOCK_CLEARED_LOG_PREFIX


@dataclass(frozen=True)
class InterlockStatus:
    """One interlock's state as the status stream reports it."""

    name: str
    unsafe: bool
    diagnostic: str = ""
    armed: bool = True


@dataclass
class InterlockRegistry:
    """The interlocks installed on one RunEngine, armed or not."""

    suspenders: list[GeecsInterlockSuspender] = field(default_factory=list)
    #: Interlock name → why it is not armed, in document order.
    unarmed: dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> list[InterlockStatus]:
        """Every interlock's current state, armed ones first, in document order.

        An interlock not yet evaluated, or not armed at all, reads unsafe:
        the status stream must never call something safe it does not know.
        """
        statuses = []
        for suspender in self.suspenders:
            verdict = suspender.verdict or Verdict(True, "not evaluated yet")
            statuses.append(
                InterlockStatus(suspender.name, verdict.unsafe, verdict.diagnostic)
            )
        for name, reason in self.unarmed.items():
            statuses.append(
                InterlockStatus(name, True, f"not armed: {reason}", armed=False)
            )
        return statuses

    def remove(self, run_engine: Any) -> None:
        """Remove every installed suspender from *run_engine*."""
        for suspender in self.suspenders:
            run_engine.remove_suspender(suspender)
        self.suspenders.clear()


def log_change(suspender: GeecsInterlockSuspender) -> None:
    """Log an interlock's verdict change (the suspender's ``on_change``)."""
    verdict = suspender.verdict
    if verdict is None:
        return
    if verdict.unsafe:
        logger.warning(
            "%s: %s: %s", TRIPPED_LOG_PREFIX, suspender.name, verdict.diagnostic
        )
    else:
        logger.warning("%s: %s", CLEARED_LOG_PREFIX, suspender.name)


def install_interlocks(
    run_engine: Any,
    interlock_set: InterlockSet,
    namespace: Any,
    resolver: Any | None,
    *,
    mock: bool = False,
    timeout: float = WARM_UP_TIMEOUT_S,
    on_change: Callable[[GeecsInterlockSuspender], None] = log_change,
) -> InterlockRegistry:
    """Install every enabled interlock of *interlock_set* on *run_engine*.

    Parameters
    ----------
    run_engine :
        The worker's RunEngine (its loop thread must be running).
    interlock_set :
        The experiment's ``interlocks.yaml``.
    namespace :
        The device namespace (signals, and the action plans' settables).
    resolver :
        The configs resolver (``resolve_action_plan`` /
        ``action_plan_registry``); ``None`` refuses entries naming a plan.
    mock :
        Connect the devices in mock mode (tests).
    timeout :
        Budget for connecting every involved device, seconds.
    on_change :
        Called on every verdict change (default: the trip / clear log line).

    Returns
    -------
    InterlockRegistry
        The armed suspenders and the entries that could not be armed.
    """
    registry = InterlockRegistry()
    prepared: list[tuple[Interlock, Watched, Any, Any, list[str]]] = []
    for interlock in interlock_set.interlocks:
        if not interlock.enabled:
            continue
        try:
            watched = _watch(interlock, namespace)
            pre_plan, pre_devices = _action(
                interlock.on_trip, interlock, resolver, namespace
            )
            post_plan, post_devices = _action(
                interlock.before_resume, interlock, resolver, namespace
            )
        except (KeyError, ValueError, GeecsError) as exc:
            # Unknown device/variable/plan, a bad target, nothing to watch:
            # a bad entry never stops environment open.
            _unarm(registry, interlock, f"{type(exc).__name__}: {exc}")
            continue
        devices = [*watched_devices(interlock), *pre_devices, *post_devices]
        prepared.append((interlock, watched, pre_plan, post_plan, devices))

    owners = {
        id(dev): dev
        for *_, devices in prepared
        for dev in (namespace[name] for name in devices)
    }
    failed = set(
        warm_up_on(
            run_engine,
            list(owners.values()),
            mock=mock,
            timeout=timeout,
            label="interlocks",
            retry_note="the interlocks on them are not armed until the "
            "environment reopens",
        )
    )

    for interlock, watched, pre_plan, post_plan, devices in prepared:
        down = [name for name in devices if _geecs_name(namespace, name) in failed]
        if down:
            _unarm(registry, interlock, f"not connected: {', '.join(down)}")
            continue
        suspender = GeecsInterlockSuspender(
            interlock,
            watched,
            pre_plan=pre_plan,
            post_plan=post_plan,
            on_change=on_change,
        )
        try:
            install_on_loop(run_engine, suspender)
        except Exception as exc:  # a failed install never stops environment open
            logger.exception("interlock %r: install failed", interlock.name)
            registry.unarmed[interlock.name] = f"install failed: {exc}"
            continue
        registry.suspenders.append(suspender)

    logger.info(
        "interlocks: %d armed, %d not armed",
        len(registry.suspenders),
        len(registry.unarmed),
    )
    return registry


def _unarm(registry: InterlockRegistry, interlock: Interlock, reason: str) -> None:
    logger.error("interlock %r NOT ARMED: %s", interlock.name, reason)
    registry.unarmed[interlock.name] = reason


def _geecs_name(namespace: Any, name: str) -> str:
    dev = namespace[name]
    return getattr(dev, "_geecs_device_name", name)


def _signal(namespace: Any, device: str, variable: str) -> Any:
    """The readback signal of ``device:variable`` (a settable's streamed readback)."""
    child = namespace.variable(device, variable)
    return child.readback_signal if isinstance(child, CaSettable) else child


def _watch(interlock: Interlock, namespace: Any) -> Watched:
    """The signals *interlock*'s condition reads, resolved in the namespace."""
    condition = interlock.condition
    value = target = stamp = None
    if isinstance(condition, StaleCondition):
        if condition.max_age_s is not None:
            stamp = _stamp(namespace, condition.device)
    else:
        value = _signal(namespace, condition.device, condition.variable)
        if isinstance(condition, AlignmentCondition):
            target = _signal(
                namespace,
                condition.target_device or condition.device,
                condition.target_variable,
            )
    connected = {}
    for device in watched_devices(interlock):
        signal = getattr(namespace[device], "connected_status", None)
        if signal is not None:
            connected[device] = signal
    if value is None and stamp is None and not connected:
        raise ValueError(f"{condition.device} has nothing to watch")
    return Watched(value=value, target=target, stamp=stamp, connected=connected)


def _stamp(namespace: Any, device: str) -> Any:
    """The device's ``acq_timestamp`` signal (a detector's own, or a served scalar)."""
    try:
        return _signal(namespace, device, "acq_timestamp")
    except KeyError:
        pass
    stamp = getattr(namespace[device], "acq_timestamp", None)
    if stamp is None:
        raise ValueError(f"{device} serves no acq_timestamp, so it cannot go stale")
    return stamp


def _action(
    name: str | None, interlock: Interlock, resolver: Any | None, namespace: Any
) -> tuple[Callable[[], Any] | None, list[str]]:
    """The plan factory for action plan *name* and the devices it touches.

    Every ``set`` / ``check`` target is looked up now, so a typo fails the
    entry at environment open instead of during a trip.
    """
    if name is None:
        return None, []
    if resolver is None:
        raise ValueError(f"action plan {name!r} needs a configs resolver")
    plan: ActionPlan = resolver.resolve_action_plan(name)
    library = resolver.action_plan_registry()
    devices: list[str] = []
    for step, _origin in flatten_action_steps(plan, registry=library):
        if isinstance(step, SetStep):
            namespace.get_settable(step.device, step.variable)
        elif isinstance(step, CheckStep):
            namespace.get_readable(step.device, step.variable)
        else:
            continue
        devices.append(step.device)

    def run_plan():
        logger.warning("interlock %s: running action plan %r", interlock.name, name)
        return (
            yield from compile_action_plan(plan, registry=library, settables=namespace)
        )

    return run_plan, list(dict.fromkeys(devices))


def install_on_loop(run_engine: Any, suspender: GeecsInterlockSuspender) -> None:
    """``RE.install_suspender`` on the RunEngine's loop.

    An ophyd-async subscription needs a running loop, and the CA monitors
    belong on the one the namespace's devices connected on.
    """

    async def install() -> None:
        run_engine.install_suspender(suspender)

    asyncio.run_coroutine_threadsafe(install(), run_engine._loop).result(10.0)


__all__ = [
    "CLEARED_LOG_PREFIX",
    "TRIPPED_LOG_PREFIX",
    "InterlockRegistry",
    "InterlockStatus",
    "install_interlocks",
    "install_on_loop",
    "log_change",
]
