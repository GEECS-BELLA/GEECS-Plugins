"""Strict single-shot acquisition as the stock ``take_reading`` hook.

``bps.one_shot`` (``bp.count``'s ``per_shot``) and ``bps.one_nd_step`` (the
scan plans' ``per_step``) take a ``take_reading`` callable whose default is
``bps.trigger_and_read``.  A GEECS camera acquires only when the trigger box
fires, so the one thing GEECS changes is **where the fire goes**: between
the triggers and the wait.  :func:`geecs_take_reading` is
``trigger_and_read`` with that one difference plus the bounded retake
(``Planning/native_bluesky/03_clean_room_rebuild.md`` §4.B, §11.5 — why
free-running edges are not an exact substitute; ``06_pva_file_plugin.md``
§2.1 — why a missed frame keeps its row and adds one).

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
from geecs_bluesky.devices.ca._view import ScalarsView
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


def _device_named(devices: Sequence[Any], device_name: str) -> Any | None:
    return next(
        (
            obj
            for obj in devices
            if device_name
            in (getattr(obj, "_geecs_device_name", None), getattr(obj, "name", None))
        ),
        None,
    )


def fire_and_await_shot(devices: Sequence[Any], fire: Callable):
    """Arm the waiters, fire one shot, await every device; return the ones that missed.

    The one GEECS line in the scan path (§4.B): the fire sits *between* the
    triggers and the wait, so every detector has baselined its stamp before
    the shot exists.  Each triggerable is awaited in its own group, so
    every device's outcome is known — a no-frame timeout on one device
    (:exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError`) does not
    hide the others'.  The devices that missed are returned; the caller
    records the row with their columns empty and decides on another shot
    (design §2.1).  A frameless device whose ``CONNECTED`` PV reads
    Disconnected went down mid-scan, so
    :exc:`~geecs_bluesky.exceptions.GeecsDeviceDownError` is raised instead —
    another shot cannot help.  Any other failed status (a refused
    SINGLESHOT put, an unexpected error) propagates untouched (Codex review
    of #811): re-firing on it could issue extra physical shots.

    The RunEngine delivers a failed status by *throwing* it into the plan
    at the next message, and a second failure landing meanwhile replaces
    the first, so the exceptions seen here are not a reliable census of
    the misses; the devices' own ``missed_shot`` flags are.  The waits are
    repeated until each group is consumed, so no stashed failure escapes to
    a later message.

    Parameters
    ----------
    devices:
        The devices of the shot.  Triggerable ones are armed and awaited.
    fire:
        Plan-stub callable emitting exactly one trigger.

    Yields
    ------
    Bluesky messages.

    Returns
    -------
    list
        The triggerables whose frame never arrived (empty for a complete shot).
    """
    triggerables = [obj for obj in devices if isinstance(obj, Triggerable)]
    groups = {
        obj: f"{short_uid('single_shot')}-{i}" for i, obj in enumerate(triggerables)
    }
    for obj, group in groups.items():
        yield from bps.trigger(obj, group=group, wait=False)
    fire_t0 = time.monotonic()
    yield from fire()
    fire_done = time.monotonic()
    for obj, group in groups.items():
        for _ in range(2 * len(groups) + 2):
            try:
                yield from bps.wait(group=group)
                break
            except FailedStatus as exc:
                if not isinstance(exc.__cause__, GeecsTriggerTimeoutError):
                    raise
        else:  # pragma: no cover - the RunEngine consumes a group per wait
            raise RuntimeError(f"trigger group {group!r} never settled")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "shot phases: fire %.1f ms, frame wait %.1f ms (%d triggerables)",
            (fire_done - fire_t0) * 1e3,
            (time.monotonic() - fire_done) * 1e3,
            len(triggerables),
        )
    missed = [obj for obj in triggerables if getattr(obj, "missed_shot", False)]
    for obj in missed:
        device_name = getattr(obj, "_geecs_device_name", None) or getattr(
            getattr(obj, "_owner", None), "_geecs_device_name", obj.name
        )
        down = yield from _confirmed_down(devices, device_name)
        if down:
            raise GeecsDeviceDownError(
                f"{device_name}: the gateway reports this device "
                "DISCONNECTED — it went down mid-scan (not a frame "
                "drop; another shot cannot help). Check the GEECS device.",
                device_name=device_name,
            )
    return missed


def geecs_take_reading(
    shot_control: Any,
    *,
    max_refires: int = 2,
    name: str = "primary",
) -> Callable[[Sequence[Any]], Any]:
    """Return a ``take_reading`` that fires the trigger box between trigger and wait.

    A missed frame does not void the row (design §2.1, Sam 2026-09-11): the
    row is saved with every scalar the shot produced — the missing device's
    columns empty (``NaN``, stamp included) — and **one more shot** is
    taken for the step, up to *max_refires* extra shots, until a complete
    row exists.  The partial row carries **no frames**: the RunEngine
    bundler requires one datum per external key per event, all of one
    width, so a frame from a device that delivered cannot be referenced by
    a row where another device has none.  Before the next shot every
    plugin-backed device is therefore rewound to the last frame a document
    referenced
    (:meth:`~geecs_bluesky.devices.detector.GeecsDetector.discard_uncollected`,
    through the stock ``wait_for`` stub) — the delivered device's
    uncollected frame and any late frame of the missed one alike.  If the
    quota is exhausted the step fails loudly, naming the devices; the
    partial rows already saved stay — their scalars are data.

    Parameters
    ----------
    shot_control :
        The :class:`~geecs_bluesky.devices.shot_control.ShotControl` device;
        the fire is ``bps.mv(shot_control, "SINGLESHOT")``.
    max_refires :
        Extra shots after an incomplete one (see :func:`fire_and_await_shot`).
    name :
        Event stream name.

    Returns
    -------
    callable
        ``take_reading(devices)`` — a plan yielding one complete event row
        (and any partial rows before it).
    """
    fire = partial(bps.mv, shot_control, TriggerState.SINGLESHOT.value)

    def take_reading(devices: Sequence[Any]):
        devices = separate_devices(devices)
        # A scalars view (``X.scalars``) beside the owner's own scanned child
        # (``X.current``): siblings to separate_devices, but the view already
        # reads the child's readback — keep one copy of the column.
        views = [d for d in devices if isinstance(d, ScalarsView)]
        devices = [d for d in devices if not any(v.covers(d) for v in views)]
        rewindable = all_safe_rewind(devices)

        def record_row():
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
            attempts = max_refires + 1
            for attempt in range(1, attempts + 1):
                missed = yield from fire_and_await_shot(devices, fire)
                if missed:
                    # The partial row carries no frames (one same-width datum
                    # per external key per event, or none): rewind every
                    # plugin before the row is read, so no datum is collected.
                    guards = [
                        d.discard_uncollected
                        for d in devices
                        if getattr(d, "plugin_backed", False)
                    ]
                    if guards:
                        yield from bps.wait_for(guards)
                ret = yield from record_row()
                if not missed:
                    return ret
                names = ", ".join(
                    getattr(d, "_geecs_device_name", None) or d.name for d in missed
                )
                if attempt == attempts:
                    raise GeecsTriggerTimeoutError(
                        names,
                        STRICT_TRIGGER_INFO.exposure_timeout,
                        f"no complete row after {attempts} shot(s): no frame from "
                        f"{names} (partial rows kept; known camera frame-drop "
                        "intermittency, ~1% observed — or a device that stopped "
                        "acquiring)",
                    )
                logger.warning(
                    "shot %d of %d incomplete: no frame from %s — row kept with "
                    "empty columns and no frames, taking another shot",
                    attempt,
                    attempts,
                    names,
                )
            return None  # pragma: no cover - the loop returns or raises

        return (yield from rewindable_wrapper(inner(), rewindable))

    return take_reading


class BinCounter:
    """The ``bin_number`` column: which step of the scan a row belongs to.

    GEECS analysis groups an s-file's rows by ``Bin #`` — one bin per scan
    step, every shot of that step in it — and the Tiled → s-file exporter
    reads it from a ``bin_number`` event column
    (``geecs_data_utils.tiled_export``).  The stock plans have no such
    notion (one row per step), so the GEECS ``per_step`` counts steps and
    reads this tiny ``Readable`` into every row.  Nothing ophyd about it: a
    plain Bluesky ``Readable`` — ``name``, ``parent``, ``read``,
    ``describe`` — with no connection to make.
    """

    name = "bin_number"
    parent = None

    def __init__(self) -> None:
        self.value = 0

    def read(self) -> dict[str, Any]:
        """The current bin number as a Bluesky reading."""
        return {self.name: {"value": self.value, "timestamp": time.time()}}

    def describe(self) -> dict[str, Any]:
        """The data key: a scalar integer sourced from the plan."""
        return {
            self.name: {"source": "plan:bin_number", "dtype": "integer", "shape": []}
        }


def geecs_per_shot(shot_control: Any, **kwargs: Any) -> Callable[..., Any]:
    """``bp.count(..., per_shot=geecs_per_shot(shot_control))``.

    A count is one bin: every row carries ``bin_number = 1``.
    """
    take_reading = geecs_take_reading(shot_control, **kwargs)
    bins = BinCounter()
    bins.value = 1

    def per_shot(detectors: Sequence[Any], take_reading_: Any = None):
        return (yield from take_reading([*detectors, bins]))

    per_shot.__name__ = per_shot.__qualname__ = "geecs_per_shot"
    return per_shot


def geecs_per_step(
    shot_control: Any, *, shots_per_step: int = 1, **kwargs: Any
) -> Callable[..., Any]:
    """``bp.scan(..., per_step=geecs_per_step(shot_control))`` (any N-d scan plan).

    ``bps.one_nd_step`` with two GEECS additions: *shots_per_step* shots at
    every position (the legacy "Shots per step", each a strict single shot
    with its own row) and the ``bin_number`` column counting the steps.
    Each row reads the detectors, the step's motors and the bin counter,
    exactly as the stock step reads detectors and motors.

    Parameters
    ----------
    shot_control :
        The trigger box device (see :func:`geecs_take_reading`).
    shots_per_step :
        Rows per position; at least 1.
    """
    if shots_per_step < 1:
        raise ValueError(f"shots_per_step must be >= 1, got {shots_per_step}")
    take_reading = geecs_take_reading(shot_control, **kwargs)
    bins = BinCounter()

    def per_step(
        detectors: Sequence[Any], step: Any, pos_cache: Any, take_reading_: Any = None
    ):
        motors = list(step.keys())
        yield from bps.move_per_step(step, pos_cache)
        bins.value += 1
        for _ in range(shots_per_step):
            yield from take_reading([*detectors, *motors, bins])

    per_step.__name__ = per_step.__qualname__ = "geecs_per_step"
    return per_step
