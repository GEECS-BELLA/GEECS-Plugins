"""Strict single-shot acquisition as the stock ``take_reading`` hook.

``bps.one_shot`` (``bp.count``'s ``per_shot``) and ``bps.one_nd_step`` (the
scan plans' ``per_step``) take a ``take_reading`` callable whose default is
``bps.trigger_and_read``.  A GEECS camera acquires only when the trigger box
fires, so the one thing GEECS changes is **where the fire goes**: between
the triggers and the wait.  :func:`geecs_take_reading` is
``trigger_and_read`` with that one difference plus the bounded refire
(``Planning/native_bluesky/03_clean_room_rebuild.md`` §4.B, §11.5 — why
free-running edges are not an exact substitute).

Everything else is the stock plan: ``bp.list_scan(dets, motor, points,
per_step=geecs_per_step(shot_control))`` moves, checkpoints, rewinds and
records run metadata exactly as it does for any other detector.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable

import bluesky.plan_stubs as bps
from bluesky.preprocessors import contingency_wrapper, rewindable_wrapper
from bluesky.protocols import Triggerable
from bluesky.utils import FailedStatus, all_safe_rewind, separate_devices, short_uid
from geecs_schemas.trigger_profile import TriggerState
from ophyd_async.core import StandardDetector

from geecs_bluesky.devices.ca._pv import GATEWAY_DISCONNECTED
from geecs_bluesky.devices.detector import STRICT_TRIGGER_INFO
from geecs_bluesky.exceptions import GeecsDeviceDownError, GeecsTriggerTimeoutError

logger = logging.getLogger(__name__)


def _confirmed_down(devices: Sequence[Any], device_name: str):
    """Plan: ``True`` iff the named device's gateway ``CONNECTED`` PV reads down.

    Tells the two causes of a no-frame timeout apart: a dropped frame (the
    device is live, re-firing recovers) and a device whose TCP stream to
    the gateway died mid-scan (re-firing cannot help).  **Fail-open**: no
    matching device, no ``connected_status`` child, or a failed read all
    return ``False``; only the exact ``Disconnected`` choice string is a
    verdict, so a mock backend's ``""`` default reads live.
    """
    device = next(
        (
            obj
            for obj in devices
            if device_name
            in (getattr(obj, "_geecs_device_name", None), getattr(obj, "name", None))
        ),
        None,
    )
    signal = getattr(device, "connected_status", None)
    if signal is None:
        return False
    try:
        value = yield from bps.rd(signal)
    except Exception:
        logger.debug(
            "CONNECTED read failed for %s; assuming live (fail-open)",
            device_name,
            exc_info=True,
        )
        return False
    return value == GATEWAY_DISCONNECTED


def fire_and_await_shot(
    devices: Sequence[Any],
    fire: Callable,
    *,
    max_refires: int = 2,
):
    """Arm the waiters, fire one shot, and await the frames, with refire.

    The one GEECS line in the scan path (§4.B): the fire sits *between* the
    triggers and the wait, so every detector has baselined its stamp before
    the shot exists.  If a triggered device produces no frame for a fire
    (the group wait fails with :exc:`~bluesky.utils.FailedStatus` caused by
    its :exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError`), the
    shot is re-fired up to *max_refires* times before the failure
    propagates — the whole-event redo, so positional joins stay exact for
    every essential detector.  Strict semantics survive refire: a failed
    attempt records nothing, and the next attempt's ``trigger()``
    re-baselines, so every recorded row is one physical shot.  Refire, not
    a longer timeout, is the recovery because a missed pulse never yields a
    frame (live-verified; ``CHANGELOG.md`` 0.20.0).

    Refire is gated on gateway liveness: a frameless device whose
    ``CONNECTED`` PV reads Disconnected went down mid-scan, so
    :exc:`~geecs_bluesky.exceptions.GeecsDeviceDownError` is raised instead
    of burning refires; a live or unreadable status (fail-open) keeps the
    bounded-refire behaviour.

    Parameters
    ----------
    devices:
        The devices of the shot.  Triggerable ones are armed and awaited.
    fire:
        Plan-stub callable emitting exactly one trigger.
    max_refires:
        Extra fire attempts after the first fails.

    Yields
    ------
    Bluesky messages.
    """
    triggerables = [obj for obj in devices if isinstance(obj, Triggerable)]
    attempts = max_refires + 1
    for attempt in range(1, attempts + 1):
        # A fresh, unique group per attempt: statuses from a failed attempt
        # must never be waited on again (a fresh trigger supersedes them).
        grp = short_uid("single_shot")
        try:
            for obj in triggerables:
                yield from bps.trigger(obj, group=grp, wait=False)
            fire_t0 = time.monotonic()
            yield from fire()
            fire_done = time.monotonic()
            if triggerables:
                yield from bps.wait(group=grp)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "shot phases: fire %.1f ms, frame wait %.1f ms (%d triggerables)",
                    (fire_done - fire_t0) * 1e3,
                    (time.monotonic() - fire_done) * 1e3,
                    len(triggerables),
                )
        except FailedStatus as exc:
            # Only a detector's no-frame timeout is a dropped frame.  A failed
            # *fire* (the SINGLESHOT put refused or ambiguous) or any other
            # failed status is not, and re-firing on it could issue extra
            # physical shots — re-raise those untouched (Codex review of
            # #811).  No cancellation of abandoned statuses is needed: the
            # RunEngine stashes a late FailedStatus and throws it into the
            # plan at the next yield, but co-missing devices share the same
            # deadline so the stash is consumed right here at the wait; a
            # straggler that lands inside the next attempt is caught by this
            # same try and merely consumes one refire.
            cause = exc.__cause__
            if not isinstance(cause, GeecsTriggerTimeoutError):
                raise
            device_name = cause.device_name
            down = yield from _confirmed_down(devices, device_name)
            if down:
                raise GeecsDeviceDownError(
                    f"{device_name}: the gateway reports this device "
                    "DISCONNECTED — it went down mid-scan (not a frame "
                    "drop; re-firing cannot help). Check the GEECS device.",
                    device_name=device_name,
                ) from exc
            if attempt == attempts:
                raise
            logger.warning(
                "single-shot attempt %d of %d: no frame from %s — re-firing "
                "(known camera frame-drop intermittency, ~1%% observed)",
                attempt,
                attempts,
                device_name,
            )
        else:
            return


def geecs_take_reading(
    shot_control: Any,
    *,
    max_refires: int = 2,
    name: str = "primary",
) -> Callable[[Sequence[Any]], Any]:
    """Return a ``take_reading`` that fires the trigger box between trigger and wait.

    Parameters
    ----------
    shot_control :
        The :class:`~geecs_bluesky.devices.shot_control.ShotControl` device;
        the fire is ``bps.mv(shot_control, "SINGLESHOT")``.
    max_refires :
        Extra fire attempts after a no-frame wait (see
        :func:`fire_and_await_shot`).
    name :
        Event stream name.

    Returns
    -------
    callable
        ``take_reading(devices)`` — a plan yielding one event row.
    """
    fire = partial(bps.mv, shot_control, TriggerState.SINGLESHOT.value)

    def take_reading(devices: Sequence[Any]):
        devices = separate_devices(devices)
        rewindable = all_safe_rewind(devices)

        def inner():
            # A GEECS detector is edge-triggered only, so the implicit
            # prepare inside trigger() (INTERNAL) would be refused; prepare
            # it for one externally triggered event.  Idempotent per
            # detector: the providers are reused and the edge setup is a
            # no-op, so repeating it per shot costs a few soft-signal awaits.
            group = short_uid("prepare")
            detectors = [d for d in devices if isinstance(d, StandardDetector)]
            for det in detectors:
                yield from bps.prepare(
                    det, STRICT_TRIGGER_INFO, group=group, wait=False
                )
            if detectors:
                yield from bps.wait(group=group)
            yield from fire_and_await_shot(devices, fire, max_refires=max_refires)
            yield from bps.create(name)

            def read_plan():
                ret: dict[str, Any] = {}
                for obj in devices:
                    reading = yield from bps.read(obj)
                    if reading is not None:
                        ret.update(reading)
                return ret

            def standard_path():
                yield from bps.save()

            def exception_path(exp):
                yield from bps.drop()
                raise exp

            return (
                yield from contingency_wrapper(
                    read_plan(), except_plan=exception_path, else_plan=standard_path
                )
            )

        return (yield from rewindable_wrapper(inner(), rewindable))

    return take_reading


def geecs_per_shot(shot_control: Any, **kwargs: Any) -> Callable[..., Any]:
    """``bp.count(..., per_shot=geecs_per_shot(shot_control))``."""
    return partial(
        bps.one_shot, take_reading=geecs_take_reading(shot_control, **kwargs)
    )


def geecs_per_step(shot_control: Any, **kwargs: Any) -> Callable[..., Any]:
    """``bp.scan(..., per_step=geecs_per_step(shot_control))`` (any N-d scan plan)."""
    return partial(
        bps.one_nd_step, take_reading=geecs_take_reading(shot_control, **kwargs)
    )
