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

from collections.abc import Sequence
from functools import partial
from typing import Any, Callable

import bluesky.plan_stubs as bps
from bluesky.preprocessors import contingency_wrapper, rewindable_wrapper
from bluesky.utils import all_safe_rewind, separate_devices, short_uid
from geecs_schemas.trigger_profile import TriggerState
from ophyd_async.core import StandardDetector

from geecs_bluesky.devices.detector import STRICT_TRIGGER_INFO
from geecs_bluesky.plans.single_shot import fire_and_await_shot


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
        :func:`~geecs_bluesky.plans.single_shot.fire_and_await_shot`).
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
