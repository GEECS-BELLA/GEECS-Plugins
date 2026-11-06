"""The on-demand (strict) policy over the shot-table layout: one slot per fire.

A strict ``count`` / ``sweep`` writes ``primary`` as the **shot table** —
one event per fire, refires included, carrying the shot's facts and no
device data — and every per-shot device in its own ``<device>_stream``
(:class:`~geecs_bluesky.plans.streams.DeviceStreams`).  The start document
says so (``layout: "shot-table/1"``, :func:`start_keys`).  The slot's
steps are :class:`OnDemand`'s.

``complete`` is the plan's decision to stop refiring; which record is
which shot's is the join's (``geecs_data_utils.shot_table``).  The slot is
not rewindable: a resume replays nothing.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Sequence
from typing import Any, Callable

import bluesky.plan_stubs as bps
from bluesky.preprocessors import rewindable_wrapper
from bluesky.utils import FailedStatus, separate_devices
from geecs_data_utils.shot_table import (
    LAYOUT_KEY,
    SHOT_TABLE_LAYOUT,
    STREAM_META_KEY,
    SYNCHRONOUS,
)
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.devices.ca._view import ScalarsView, owner_of
from geecs_bluesky.devices.detector import DEFAULT_SHOT_TIMEOUT
from geecs_bluesky.devices.native import (
    NativeCamera,
    NativeTriggerSource,
    down_native_cameras,
)
from geecs_bluesky.devices.scan_mode import ANNOUNCE
from geecs_bluesky.plans.streams import DeviceStreams
from geecs_bluesky.plans.strict import (
    BinCounter,
    _confirmed_down,
    _log_non_frame_failure,
    geecs_name,
    name_failed_status,
    plugin_reasons,
)

logger = logging.getLogger(__name__)

#: Seconds between progress reads while a slot waits.
POLL_S = 0.02
#: How long the witness may trail the devices that already reported, seconds
#: (its IOC polls the counter; ~25 ms measured).
WITNESS_GRACE_S = 0.25
#: Exhausted slots in a row that pause the run.
EXHAUSTED_PAUSE = 2
#: What an incomplete row leads to (:meth:`OnDemand._missed`).
_REFIRE, _PAUSED, _EXHAUSTED = "refire", "paused", "exhausted"
#: Seconds one counter read may take; a read that fails or outlasts it is
#: no reading (the device did not report), never a failed run.
READ_TIMEOUT_S = 1.0


async def _bounded(read: Any, failed: Any) -> Any:
    """``await read()`` within :data:`READ_TIMEOUT_S`; *failed* when it raises or outlasts it."""
    (result,) = await asyncio.gather(
        asyncio.wait_for(read(), READ_TIMEOUT_S), return_exceptions=True
    )
    if isinstance(result, asyncio.CancelledError):
        raise result
    if isinstance(result, BaseException):
        logger.debug("counter read failed: %r", result)
        return failed
    return result


class _Progress:
    """One per-shot device's progress counter: its frame counts, or its stamp."""

    def __init__(self, obj: Any, framed: bool) -> None:
        self.owner = owner_of(obj)
        self.name = self.owner.name
        self.framed = framed
        acquire = getattr(self.owner, "_acquire", None)
        self.timeout = float(getattr(acquire, "shot_timeout", DEFAULT_SHOT_TIMEOUT))

    async def read(self) -> tuple[float, ...]:
        if self.framed:
            ctx = self.owner._prepare_ctx
            providers = ctx.streamable_data_providers if ctx else ()
            return tuple(
                [
                    float(await p.collections_written_signal.get_value())
                    for p in providers
                ]
            )
        last = self.owner.last_acq_timestamp
        return (math.nan if last is None else float(last),)

    async def read_safe(self) -> tuple[float, ...]:
        """:meth:`read`, bounded; ``()`` when it fails."""
        return await _bounded(self.read, ())

    def moved(self, base: tuple[float, ...], now: tuple[float, ...]) -> bool:
        if not base or len(now) != len(base):
            return False  # an unreadable counter, either side: no evidence
        if self.framed:
            return all(n > b for n, b in zip(now, base, strict=True))
        return not math.isnan(now[0]) and now[0] != base[0]

    @staticmethod
    def recorded(base: tuple[float, ...]) -> float:
        return min(base) if base else math.nan


# The waits poll to a deadline and are never cancelled from outside: an
# outer ``asyncio.wait_for`` cancelling an inner one can lose the
# cancellation on Python 3.11 (the inner read finishing at that instant),
# and the poller would then run forever.


async def _until_moved(p: _Progress, base: tuple[float, ...], deadline: float) -> None:
    loop = asyncio.get_running_loop()
    while loop.time() < deadline:
        if p.moved(base, await p.read_safe()):
            return
        await asyncio.sleep(POLL_S)


async def _until_counted(witness: Any, base: Any, deadline: float) -> None:
    loop = asyncio.get_running_loop()
    while loop.time() < deadline:
        if await _bounded(witness.counts.get_value, None) not in (None, base):
            return
        await asyncio.sleep(POLL_S)


class SlotColumns:
    """The shot table's own columns of one row (a plain Readable, like ``bin_number``)."""

    name = "shot"
    parent = None

    def __init__(
        self, required: Sequence[str], *, scan_mode: bool, fly: bool = False
    ) -> None:
        self._required = list(required)
        self._scan_mode = scan_mode
        self._fly = fly
        self.shot_index = 0
        self.values: dict[str, Any] = {}
        self.reset(1, {})

    def reset(self, attempt: int, baselines: dict[str, float]) -> None:
        """A new row: its attempt, its baselines, every flag cleared.

        A fly row has no fire and no hardware witness: no ``fire_stamp``,
        ``unwitnessed`` or ``witness_late`` (an ``unwitnessed`` column would
        make the reader time the row by a fire it does not have).
        """
        fired = {} if self._fly else {"fire_stamp": math.nan}
        witness = {} if self._fly else {"unwitnessed": False, "witness_late": False}
        self.values = {
            "attempt": attempt,
            **fired,
            "complete": False,
            "missing": "",
            **witness,
            "interrupted": False,
            **{f"{n}-baseline": baselines.get(n, math.nan) for n in self._required},
        }
        if self._scan_mode:
            self.values["announce_stamp"] = math.nan

    def read(self) -> dict[str, Any]:
        """The row's columns as Bluesky readings."""
        now = time.time()
        data = {"shot_index": self.shot_index, **self.values}
        return {k: {"value": v, "timestamp": now} for k, v in data.items()}

    def describe(self) -> dict[str, Any]:
        """The data keys, fixed for the run."""
        dtypes = {bool: "boolean", int: "integer", float: "number", str: "string"}
        data = {"shot_index": 0, **self.values}
        return {
            k: {"source": f"plan:{k}", "dtype": dtypes[type(v)], "shape": []}
            for k, v in data.items()
        }


def start_keys(streams: DeviceStreams, required: Sequence[Any], witnessed: bool):
    """The shot-table start-document keys of a strict run."""
    return {
        LAYOUT_KEY: SHOT_TABLE_LAYOUT,
        "per_shot_streams": streams.names,
        "required": [owner_of(d).name for d in required],
        "shot_clock_column": "witness_stamp" if witnessed else "fire_stamp",
        STREAM_META_KEY: {
            name: {"sync_class": SYNCHRONOUS, "stale_after": None}
            for name in streams.names
        },
    }


class OnDemand:
    """The slot of one strict run; one instance per run.

    One slot (:meth:`take_reading`):

    1. throttle to ``shot_period``; read ``ShotControl.pause_count``;
    2. read every per-shot device's progress counter (a framed stream's frame
       count, a stamped device's ``acq_timestamp`` monitor) and the witness
       count — before the fire, in ARMED, so no edge is in flight; the
       required devices' reads are recorded in the row (``<device>-baseline``);
    3. fire: the announce under scan mode (``announce_stamp``), the
       SINGLESHOT put (``fire_stamp``; at the announce under a box-less
       profile), then a press of every Software-source native camera;
    4. wait, bounded by the devices' shot timeout, for every required counter
       to advance (with no required device, for the witness count);
    5. re-read every counter: the wait's return is never the verdict, so a
       pause or suspension that cut into the slot (``pause_count`` moved;
       ``interrupted``) costs nothing — a counter that advanced counts;
    6. the witness: only when neither the witness nor any per-shot device
       advanced is the fire reported as not fired — the row is kept and the
       run pauses with the reason; a witness that stayed put while devices
       advanced is ``witness_late``;
    7. the row: ``create``, read, and before ``save`` compare
       ``pause_count`` once more — a pause inside the bundle means ``drop``
       and read again, never a saved short row;
    8. collect every stream, each a contingency;
    9. a required device that did not report: refire (a new row, ``attempt``
       + 1) up to ``max_refires``; exhausted, the row is flagged
       (``complete`` false) and the scan continues — two exhausted slots in
       a row pause the run.  A required device whose liveness reads down
       pauses the run with the reason.  Non-required devices are never waited
       on and never refire anything.

    Parameters
    ----------
    shot_control :
        The trigger box (``ShotControl``).
    streams :
        The run's per-shot devices and their streams.
    required :
        The listed per-shot devices whose acquisition makes a shot complete.
    witness :
        The run's witness when it is read, else ``None`` (rows ``unwitnessed``).
    max_refires, shot_period, scan_mode :
        Extra shots after an incomplete one; the rep-rate throttle,
        seconds; the run's ``ScanModeControl`` under a scan-mode profile.
    """

    #: How an incomplete row's log line ends, and what a pause for exhausted
    #: rows adds to its reason (:meth:`_missed`).
    AGAIN = "taking another shot"
    EXHAUSTED_HINT = ""

    def __init__(
        self,
        shot_control: Any,
        streams: DeviceStreams,
        required: Sequence[Any],
        *,
        witness: Any | None = None,
        max_refires: int = 2,
        shot_period: float | None = None,
        scan_mode: Any | None = None,
    ) -> None:
        if shot_period is not None and shot_period <= 0:
            raise ValueError(f"shot_period must be positive seconds, got {shot_period}")
        self.sc = shot_control
        self.streams = streams
        framed = {id(d) for d in streams.framed}
        self.progress = {id(d): _Progress(d, id(d) in framed) for d in streams.devices}
        self.required = [self.progress[id(d)] for d in required]
        self.witness = witness
        self.attempts = max_refires + 1
        self.shot_period = shot_period
        self.scan_mode = scan_mode
        self.boxless = bool(getattr(shot_control, "boxless", False))
        self.presses = [
            d
            for d in streams.natives
            if d.trigger_source == NativeTriggerSource.SOFTWARE
        ]
        self.columns = SlotColumns(
            [p.name for p in self.required], scan_mode=scan_mode is not None
        )
        self._last_fire: float | None = None
        self._exhausted = 0
        self._reported_down: set[str] = set()
        self._refused_press: set[str] = set()

    # ------------------------------------------------------------------ plans
    def _counters(self):
        """Plan: every per-shot counter, and the witness count (``None`` without one)."""
        out: dict[str, Any] = {}

        async def read() -> None:
            for p in self.progress.values():
                out[p.name] = await p.read_safe()
            if self.witness is not None:
                out[""] = await _bounded(self.witness.counts.get_value, None)

        yield from bps.wait_for([read])
        return out

    def _wait(self, base: dict[str, Any]):
        """Plan: until every required counter advanced (or the witness, with none required)."""
        required, witness = self.required, self.witness

        async def wait() -> None:
            loop = asyncio.get_running_loop()
            if required:
                deadline = loop.time() + max(p.timeout for p in required)
                await asyncio.gather(
                    *(_until_moved(p, base[p.name], deadline) for p in required)
                )
            if witness is not None and base[""] is not None:
                grace = WITNESS_GRACE_S if required else DEFAULT_SHOT_TIMEOUT
                await _until_counted(witness, base[""], loop.time() + grace)

        if required or witness is not None:
            yield from bps.wait_for([wait])

    def _throttle(self):
        if self.shot_period is not None and self._last_fire is not None:
            remaining = self.shot_period - (time.monotonic() - self._last_fire)
            if remaining > 0:
                yield from bps.sleep(remaining)

    def _fire(self):
        values = self.columns.values
        self._last_fire = time.monotonic()
        try:
            if self.scan_mode is not None:
                values["announce_stamp"] = time.time()
                yield from bps.mv(self.scan_mode, ANNOUNCE)
            if self.boxless:
                values["fire_stamp"] = values["announce_stamp"]
            else:
                values["fire_stamp"] = time.time()
                yield from bps.mv(self.sc, TriggerState.SINGLESHOT.value)
        except FailedStatus as exc:
            _log_non_frame_failure(exc)
            raise
        if self.presses:
            yield from bps.wait_for([self._press])

    async def _press(self) -> None:
        """Press every Software-source native; a refused press is logged once per camera."""
        results = await asyncio.gather(
            *(cam.press() for cam in self.presses), return_exceptions=True
        )
        for cam, result in zip(self.presses, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            name = geecs_name(cam)
            if isinstance(result, BaseException) and name not in self._refused_press:
                self._refused_press.add(name)
                logger.warning(
                    "%s: the press failed (%s: %s) — its stream gaps for this "
                    "fire (logged once)",
                    name,
                    type(result).__name__,
                    result,
                )

    def _write_row(self, readers: Sequence[Any], name: str):
        """Plan: one event; a pause inside the bundle drops it and reads again."""
        while True:
            mark = self.sc.pause_count
            yield from bps.create(name)
            try:
                for obj in readers:
                    yield from bps.read(obj)
            except BaseException:
                yield from bps.drop()
                raise
            if self.sc.pause_count == mark:
                yield from bps.save()
                return
            self.columns.values["interrupted"] = True
            yield from bps.drop()

    def _down(self, quiet: Sequence[_Progress]):
        """Plan: the GEECS names of the *quiet* devices whose liveness reads down."""
        down: list[str] = []
        for p in quiet:
            if isinstance(p.owner, NativeCamera):
                down += yield from down_native_cameras([p.owner])
            elif (yield from _confirmed_down([p.owner], geecs_name(p.owner))):
                down.append(geecs_name(p.owner))
        return down

    def _moved(self, p: _Progress, base: dict[str, Any], now: dict[str, Any]) -> bool:
        """Whether *p* recorded in this slot: its counter moved from *base* to *now*."""
        return p.moved(base[p.name], now[p.name])

    def _settled(self, base: dict[str, Any]) -> None:
        """The slot's counters are judged (a hook: the strict slot keeps nothing)."""

    def _pause(self, reason: str):
        logger.error("pausing the run: %s", reason)
        yield from bps.pause()

    def _verdict(
        self, base: dict[str, Any], now: dict[str, Any]
    ) -> tuple[list[_Progress], bool]:
        """Set the row's flags from the counters: the required devices that missed, and fired."""
        values = self.columns.values
        moved = {p.name for p in self.progress.values() if self._moved(p, base, now)}
        missing = [p for p in self.required if p.name not in moved]
        values["missing"] = ",".join(p.name for p in missing)
        values["complete"] = not missing
        if self.witness is None:
            values["unwitnessed"] = True
        elif None in (now[""], base[""]):
            pass  # the witness did not answer: no evidence either way
        elif now[""] == base[""]:
            if not moved:
                return missing, False
            values["witness_late"] = True
        return missing, True

    def _slot(self, readers: Sequence[Any], name: str):
        # ``attempt`` counts the shot's fires (the row's column); ``spent``
        # only the incomplete ones the budget pays for — a fire a pause
        # followed (nothing saw it, or a required device is down) costs no
        # refire, so a pause never loses a shot.
        attempt = spent = 0
        while True:
            attempt += 1
            yield from self._throttle()
            mark = self.sc.pause_count
            base = yield from self._counters()
            self.columns.shot_index += 1
            self.columns.reset(
                attempt,
                {p.name: _Progress.recorded(base[p.name]) for p in self.required},
            )
            yield from self._fire()
            yield from self._wait(base)
            now = yield from self._counters()
            self.columns.values["interrupted"] = self.sc.pause_count != mark
            missing, fired = self._verdict(base, now)
            yield from self._write_row(readers, name)
            yield from self.streams.collect()
            yield from self._report_quiet_optional(base, now)
            self._settled(base)
            if not fired:
                yield from self._pause(
                    "the fire was not seen: neither the witness nor any per-shot "
                    "device advanced — check the trigger box, then resume (the "
                    "step fires again)"
                )
                continue
            if not missing:
                self._exhausted = 0
                return None
            outcome = yield from self._missed(missing, spent + 1)
            if outcome == _PAUSED:
                continue
            spent += 1
            if outcome == _EXHAUSTED:
                return None

    def _missed(self, missing: Sequence[_Progress], n: int):
        """Plan: after an incomplete row, its *n*-th paid attempt: refire, paused or exhausted."""
        labels = [geecs_name(p.owner) for p in missing]
        # A plugin refusing the frames says so only in its WriteMessage.
        reasons = yield from plugin_reasons([p.owner for p in missing])
        described = ", ".join(
            f"{label} ({reason})" if reason else label
            for label, reason in zip(labels, reasons, strict=True)
        )
        down = yield from self._down(missing)
        if down:
            yield from self._pause(
                f"required device(s) {', '.join(down)} report down — check them, "
                "then resume (the step fires again)"
            )
            return _PAUSED
        if n < self.attempts:
            logger.warning(
                "shot %d of %d incomplete: no record from %s — row kept, %s",
                n,
                self.attempts,
                described,
                self.AGAIN,
            )
            return _REFIRE
        self._exhausted += 1
        cause = (
            "a file plugin refused what its camera pushed — see its message"
            if any(reasons)
            else "known camera frame-drop intermittency, ~1% observed — or a "
            "device that stopped acquiring"
        )
        logger.warning(
            "no complete shot after %d attempt(s): no record from %s — the last "
            "row is flagged incomplete and the scan continues (%s)",
            self.attempts,
            described,
            cause,
        )
        if self._exhausted >= EXHAUSTED_PAUSE:
            self._exhausted = 0
            yield from self._pause(
                f"{EXHAUSTED_PAUSE} shots in a row exhausted their refires (no "
                f"record from {', '.join(labels)}){self.EXHAUSTED_HINT}"
            )
        return _EXHAUSTED

    def _report_quiet_optional(self, base: dict[str, Any], now: dict[str, Any]):
        """Plan: a non-required device that did not report and reads down is logged once."""
        required = {id(p) for p in self.required}
        quiet = [
            p
            for p in self.progress.values()
            if id(p) not in required
            and p.name not in self._reported_down
            and not self._moved(p, base, now)
        ]
        if not quiet:
            return
        down = yield from self._down(quiet)
        self._reported_down.update(p.name for p in quiet if geecs_name(p.owner) in down)
        for name in down:
            logger.warning(
                "%s (not required) reports down — its stream gaps; the run "
                "continues (logged once)",
                name,
            )

    def take_reading(self, devices: Sequence[Any], name: str = "primary"):
        """Plan: one slot over the row readers *devices* (conditions, motors, extras)."""
        readers = separate_devices([*devices, self.columns])
        # A scalars view beside its owner's own scanned child: one column copy.
        views = [d for d in readers if isinstance(d, ScalarsView)]
        readers = [d for d in readers if not any(v.covers(d) for v in views)]
        return (yield from rewindable_wrapper(self._slot(readers, name), False))


def on_demand_per_shot(run: OnDemand, *, scan_mode: Any | None = None) -> Callable:
    """``bp.count(..., per_shot=on_demand_per_shot(run))``: a count is one bin."""
    bins = BinCounter()
    bins.value = 1
    extras = [bins] if scan_mode is None else [bins, scan_mode]

    def per_shot(detectors: Sequence[Any], take_reading_: Any = None):
        return (yield from name_failed_status(run.take_reading([*detectors, *extras])))

    per_shot.__name__ = per_shot.__qualname__ = "on_demand_per_shot"
    return per_shot


def on_demand_per_step(
    run: OnDemand, *, shots_per_step: int = 1, scan_mode: Any | None = None
) -> Callable:
    """``scan_nd(..., per_step=on_demand_per_step(run))``: *shots_per_step* slots per position."""
    if shots_per_step < 1:
        raise ValueError(f"shots_per_step must be >= 1, got {shots_per_step}")
    bins = BinCounter()
    extras = [bins] if scan_mode is None else [bins, scan_mode]

    def one_step(detectors: Sequence[Any], step: Any, pos_cache: Any):
        motors = list(step.keys())
        yield from bps.move_per_step(step, pos_cache)
        bins.value += 1
        for _ in range(shots_per_step):
            # A checkpoint per shot: a deferred pause lands between shots.
            yield from bps.checkpoint()
            yield from run.take_reading([*detectors, *motors, *extras])

    def per_step(
        detectors: Sequence[Any], step: Any, pos_cache: Any, take_reading_: Any = None
    ):
        return (yield from name_failed_status(one_step(detectors, step, pos_cache)))

    per_step.__name__ = per_step.__qualname__ = "on_demand_per_step"
    return per_step


__all__ = [
    "OnDemand",
    "SlotColumns",
    "on_demand_per_shot",
    "on_demand_per_step",
    "start_keys",
]
