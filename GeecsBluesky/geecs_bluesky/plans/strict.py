"""The strict shot's shared helpers: the fire-and-await, the liveness, the bin.

The strict slot itself is :mod:`geecs_bluesky.plans.on_demand`.  Here:
:func:`fire_and_await_shot` (trigger every triggerable, fire between the
triggers and the wait, report the ones that missed — the shot-offset
calibration's shot), the confirmed-down read and the plugin's refusal
reasons the slot reports, :func:`name_failed_status`, and
:class:`BinCounter`, the ``bin_number`` column.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Callable

import bluesky.plan_stubs as bps
from bluesky.protocols import Triggerable
from bluesky.utils import FailedStatus, short_uid

from geecs_bluesky.devices.ca._view import geecs_device_name
from geecs_bluesky.devices.ca.liveness import read_disconnected
from geecs_bluesky.exceptions import (
    GeecsDeviceDownError,
    GeecsTriggerTimeoutError,
    failure_cause_text,
)

logger = logging.getLogger(__name__)


def _confirmed_down(devices: Sequence[Any], device_name: str):
    """Plan: ``True`` iff the named device's gateway ``CONNECTED`` PV reads down.

    Tells the two causes of a no-frame timeout apart: a dropped frame (the
    device is live, re-firing recovers) and a device whose TCP stream to
    the gateway died mid-scan (re-firing cannot help).  The read and its
    verdict rule are :func:`~geecs_bluesky.devices.ca.liveness.read_disconnected`
    — **fail-open**: no matching device, no ``connected_status`` child, or
    a failed read all return ``False``; only the exact ``Disconnected``
    choice string is a verdict, so a mock backend's ``""`` default reads
    live.
    """
    signal = getattr(_device_named(devices, device_name), "connected_status", None)
    if signal is None:
        return False
    down = yield from read_disconnected({device_name: signal})
    return bool(down)


def geecs_name(obj: Any) -> str:
    """The GEECS device name of a plan object (the one rule, :func:`geecs_device_name`)."""
    return geecs_device_name(obj)


def plugin_reasons(devices: Sequence[Any]):
    """Plan: each device's file-plugin reasons, joined; ``""`` without a plugin or a message.

    A plugin-backed camera that yields no frame may be delivering every
    frame to a plugin that refuses them — a stack that could not be
    opened, a frame of another shape than the open stack's — and the
    refusal reaches only the plugin's ``WriteMessage``.  Read once per miss through
    :meth:`~geecs_bluesky.devices.detector.GeecsDetector.plugin_reasons`
    (bounded per plugin), so the incomplete-shot warning and the step's
    failure carry the plugin's reason beside the device's name instead of
    blaming camera frame drops.  A device without the method — a scalars
    view, a scalar device — contributes ``""``.
    """
    reasons: list[str] = []
    for obj in devices:
        read = getattr(obj, "plugin_reasons", None)
        if read is None:
            reasons.append("")
            continue
        found: list[str] = []

        async def collect(read: Any = read, found: list[str] = found) -> None:
            found.extend(await read())

        yield from bps.wait_for([collect])
        reasons.append("; ".join(found))
    return reasons


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


def name_failed_status(plan: Any):
    """Plan wrapper: a ``FailedStatus`` escaping *plan* reads as its cause.

    The stock ``run_wrapper`` writes ``str(exc)`` into the stop document's
    ``reason`` as the exception passes it, and a ``FailedStatus``'s own text
    is the status repr — which for a *falsy* cause (a refused
    ``aioca.CANothing``) is not even ``errored: …`` but ``done``.  The
    exception's text is replaced with the
    shared rendering (:func:`~geecs_bluesky.exceptions.failure_cause_text`:
    the cause by ``str`` plus its notes) as it passes; type, cause and
    traceback are untouched, so every ``except FailedStatus`` upstream still
    matches.  Idempotent, so the GEECS hooks (inside ``run_wrapper``, for the
    stop document) and the bound plan (outside it, for the manager's report
    of a failure before or after the run) can both apply it.
    """
    try:
        return (yield from plan)
    except FailedStatus as exc:
        if exc.__cause__ is not None:  # a cause-less status keeps its own text
            exc.args = (failure_cause_text(exc),)
        raise


def _log_non_frame_failure(exc: FailedStatus) -> None:
    """ERROR-log a failed status that is *not* a missing frame, naming its cause.

    This line is the scan log's record of what failed at the fire: the
    ``FailedStatus`` propagates unwrapped, rendered by the shared
    :func:`~geecs_bluesky.exceptions.failure_cause_text`.
    """
    logger.error(
        "shot failed, but not from a missing frame — another shot cannot "
        "help, so the failure propagates: %s",
        failure_cause_text(exc),
    )


def fire_and_await_shot(devices: Sequence[Any], fire: Callable):  # noqa: C901
    """Arm the waiters, fire one shot, await every device; return the ones that missed.

    The one GEECS line in the scan path: the fire sits *between* the
    triggers and the wait, so every detector has baselined its stamp before
    the shot exists.  Each triggerable is awaited in its own group, so
    every device's outcome is known — a no-frame timeout on one device
    (:exc:`~geecs_bluesky.exceptions.GeecsTriggerTimeoutError`) does not
    hide the others'.  The devices that missed are returned; the caller
    records the row with their columns empty and decides on another shot.
    A frameless device whose ``CONNECTED`` PV reads
    Disconnected went down mid-scan, so
    :exc:`~geecs_bluesky.exceptions.GeecsDeviceDownError` is raised instead —
    another shot cannot help.  Any other failed status (a refused
    SINGLESHOT put, an unexpected error) propagates untouched: re-firing on
    it could issue extra physical shots.

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
    try:
        yield from fire()
    except FailedStatus as exc:
        # The fire is a waited ``mv``, so a refused SINGLESHOT put fails
        # right here, before any wait — name it, then propagate untouched.
        # A detector's own no-frame timeout can also be thrown in here (its
        # wait budget started at trigger; a stalled put outlives it): that
        # is not this line's fault class, so it propagates unlabelled.
        if not isinstance(exc.__cause__, GeecsTriggerTimeoutError):
            _log_non_frame_failure(exc)
        raise
    fire_done = time.monotonic()
    for group in groups.values():
        for _ in range(2 * len(groups) + 2):
            try:
                yield from bps.wait(group=group)
                break
            except FailedStatus as exc:
                if not isinstance(exc.__cause__, GeecsTriggerTimeoutError):
                    _log_non_frame_failure(exc)
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
        device_name = geecs_name(obj)
        down = yield from _confirmed_down(devices, device_name)
        if down:
            raise GeecsDeviceDownError(
                f"{device_name}: the gateway reports this device "
                "DISCONNECTED — it went down mid-scan (not a frame "
                "drop; another shot cannot help). Check the GEECS device.",
                device_name=device_name,
            )
    return missed


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
