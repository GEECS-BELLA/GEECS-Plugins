"""BackgroundSnapshot — every logged scalar outside the run's devices, in every row, softly.

Master Control logged every ``get='yes'`` variable of the experiment into
every s-file row; this object restores that parity without ever blocking
a scan (#1016, #929).

- **Membership.**  The namespace's telemetry set
  (:meth:`~geecs_bluesky.namespace.GeecsNamespace.telemetry`) minus what
  the run records itself, so no event key is contributed twice.  Decided
  in two steps.  At :meth:`probe`, right before ``open_run``: every
  candidate rooted at one of the run's **own readers** (its detectors and
  non-essential devices, a ``.scalars`` view standing for its owner —
  handed over by the bound plan) is excluded outright, whatever else is
  scanned on that device; every candidate rooted at any *other* staged
  device is **parked** — the plan stages root devices (bluesky's
  ``stage_wrapper`` stages ``root_ancestor``), so a scanned
  ``U_S1H.current`` looks like ``U_S1H`` whole there.  At the first step,
  :meth:`admit` is handed the movers and lets each parked device back in
  minus the keys its mover describes (the readback, the motor's own
  column), so the scanned device's other logged variables stay in the
  rows — Master Control logged them, and they matter most on the device
  being varied.  A parked device was staged by the RunEngine: the
  snapshot never stages or unstages it.
- **Probe.**  Right before ``open_run`` every member is connected, staged,
  described and read once, concurrently, within :data:`PROBE_TIMEOUT_S`.
  One that does not answer is dropped **for this run only**, named in the
  log and in the start document (``background_dropped``); an INVALID
  member is kept and reads ``NaN``.  A probe error is recorded
  (``background_probe_error``) and the run opens without background
  columns.  :func:`warm_up` connects every candidate once at environment
  open, so the probe's budget covers only the stage and the first read.
- **Read.**  One monitor-cache reading per shot; INVALID, a failed read or
  one past :data:`READ_TIMEOUT_S` reads ``NaN``.  Every declared key is in
  every row and ``read`` never raises.
- **Headers.**  ``_column_headers`` is the active members' union; the
  ``scalar_headers`` preprocessor merges it into ``geecs_scalar_headers``.

A plain Bluesky readable (``read``/``describe``/``stage``/``unstage``): the
strict ``take_reading`` reads it as one more device of the row and the
gated batch samples it at the tick.  Not ``Triggerable`` and not a
``GeecsDetector``, so the fire, the liveness gate, the shot clock and the
native-saving switch all pass it by.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Sequence
from typing import Any

from bluesky.protocols import Reading
from bluesky.utils import maybe_await, root_ancestor
from event_model import DataKey
from ophyd_async.core import AsyncStatus

from geecs_bluesky.devices.ca._view import geecs_device_name, owner_of
from geecs_bluesky.utils import is_connected

logger = logging.getLogger(__name__)

#: The probe's budget per member, seconds: connect (when not yet), stage,
#: describe and the first cached reading — concurrent across members, so
#: the run pays it once, before ``open_run``, whatever the count.  A served
#: PV answers in milliseconds on the lab network; an unserved one never
#: does, and this is what it costs the run.
PROBE_TIMEOUT_S = 1.0
#: The per-shot read's backstop, seconds.  A member that passed the probe
#: reads from its monitor cache without waiting; this bounds the shot if
#: something went wrong meanwhile (its columns then read ``NaN``).
READ_TIMEOUT_S = 0.5
#: The environment-open warm-up's budget, seconds (:func:`warm_up`): every
#: candidate's first connect at once, paid once per environment open —
#: only an unserved PV runs it out, and never inside a run.
WARM_UP_TIMEOUT_S = 20.0

#: The probe's note on a member kept although every reading it gave was
#: INVALID: served, but the gateway marks the device down.
_STALE = "served but INVALID at the start"


def _invalid(reading: Reading) -> bool:
    """Whether the gateway marked *reading* INVALID.

    The CA gateway sets INVALID severity on every readback of a device
    whose stream to it is down (``GeecsCAGateway/PV_CONTRACT.md``): the
    value served is stale.  The aioca backend maps CA's INVALID (3) to
    ``-1``; NO_ALARM, MINOR and MAJOR (0, 1, 2) carry live values.
    """
    return reading.get("alarm_severity", 0) not in (0, 1, 2)


def _blank(datakey: DataKey) -> Reading:
    """The reading of a missing value: ``NaN`` for a number, the empty value otherwise."""
    dtype = datakey.get("dtype")
    value: Any = float("nan")
    if dtype == "string":
        value = ""
    elif dtype == "boolean":
        value = False
    elif dtype == "array":
        value = []
    return {"value": value, "timestamp": time.time()}


def _one_line(exc: BaseException) -> str:
    """``Type: message`` on one line (a NotConnectedError's text spans lines)."""
    return f"{type(exc).__name__}: {' '.join(str(exc).split())}"


async def _call(obj: Any, method: str) -> None:
    """Call ``obj.<method>()`` if it exists and await whatever it returns."""
    fn = getattr(obj, method, None)
    if fn is None:
        return
    status = fn()
    if inspect.isawaitable(status):
        await status


def _retrieved(task: asyncio.Task[Any]) -> None:
    """Mark a finished task's exception retrieved (asyncio's "never retrieved" noise)."""
    if not task.cancelled():
        task.exception()


class BackgroundSnapshot:
    """The run's background telemetry: the candidates outside the run, read per shot.

    Parameters
    ----------
    candidates :
        Everything the experiment logs that a run *might* read in the
        background (``GeecsNamespace.telemetry()``): scalar-only devices,
        detectors' scalar signals.  Which of them this run reads is
        decided at :meth:`probe`.
    mock :
        Connect a not-yet-connected member with a mock backend (hermetic
        tests); the worker leaves this off.
    probe_timeout, read_timeout :
        :data:`PROBE_TIMEOUT_S` and :data:`READ_TIMEOUT_S`.
    name :
        The Bluesky object name.  The event keys are the members' own.
    """

    parent = None

    def __init__(
        self,
        candidates: Sequence[Any],
        *,
        mock: bool = False,
        probe_timeout: float = PROBE_TIMEOUT_S,
        read_timeout: float = READ_TIMEOUT_S,
        name: str = "background",
    ) -> None:
        self.name = name
        self._candidates = list(candidates)
        self._mock = mock
        self.probe_timeout = float(probe_timeout)
        self.read_timeout = float(read_timeout)
        self._active: list[Any] = []
        self._datakeys: dict[int, dict[str, DataKey]] = {}
        self._parked: list[Any] = []
        self._staged_by_me: set[int] = set()
        self._dropped: list[str] = []
        self._probe_error = ""
        self._warned: set[int] = set()

    def __repr__(self) -> str:
        """The candidate and active counts (the stock plans ``repr`` their detectors)."""
        return (
            f"BackgroundSnapshot({len(self._candidates)} candidate(s), "
            f"{len(self._active)} active)"
        )

    # ----------------------------------------------------------- properties
    @property
    def candidates(self) -> list[Any]:
        """Everything that may be read in the background (before the probe)."""
        return list(self._candidates)

    @property
    def members(self) -> list[Any]:
        """The members read per shot this run (after :meth:`probe`)."""
        return list(self._active)

    @property
    def dropped(self) -> list[str]:
        """GEECS device names dropped this run, in order.

        The probe's first (the start document's ``background_dropped``),
        then any the step's :meth:`admit` dropped — logged only, the start
        document has gone out by then.
        """
        return list(self._dropped)

    @property
    def probe_error(self) -> str:
        """``Type: message`` of a probe that failed outright this run, else empty."""
        return self._probe_error

    @property
    def _column_headers(self) -> dict[str, str]:
        """Event key → legacy ``Device Variable`` header, for the active members."""
        headers: dict[str, str] = {}
        for member in self._active:
            source = getattr(member, "_column_headers", None)
            if source is None:  # a detector's signal: the header map is its owner's
                source = getattr(
                    getattr(member, "parent", None), "_column_headers", None
                )
            source = source or {}
            for key in self._datakeys.get(id(member), {}):
                if key in source:
                    headers[key] = source[key]
        return headers

    # ------------------------------------------------------------ lifecycle
    def stage(self) -> AsyncStatus:
        """A new run: nothing is active until :meth:`probe` says so."""

        async def do() -> None:
            self._active = []
            self._datakeys = {}
            self._parked = []
            self._staged_by_me = set()
            self._dropped = []
            self._probe_error = ""
            self._warned = set()

        return AsyncStatus(do())

    def unstage(self) -> AsyncStatus:
        """Release the monitor caches the probe staged (each member's own ``unstage``).

        A member admitted at the step was staged by the RunEngine and is
        left to it.
        """

        async def do() -> None:
            members, self._active = self._active, []
            mine, self._staged_by_me = self._staged_by_me, set()
            self._parked = []
            await asyncio.gather(
                *(self._unstage_one(m) for m in members if id(m) in mine)
            )

        return AsyncStatus(do())

    async def probe(
        self, staged: Sequence[Any] = (), *, own: Sequence[Any] = ()
    ) -> None:
        """Decide this run's members: drop the run's own devices, probe the rest.

        Never raises: an error in the probe itself (not in a member — a
        member's failure is its own drop) leaves the run without background
        columns, logged at ERROR and recorded in :attr:`probe_error` for
        the start document.

        Parameters
        ----------
        staged :
            Every object the plan staged before ``open_run`` — root
            devices, as bluesky stages them: its detectors, non-essential
            devices and the motors' owners.
        own :
            The run's own readers as the plan lists them: its detectors
            and non-essential devices (a ``.scalars`` view counts as its
            owner).  Every candidate rooted at one is the run's and is
            excluded outright; every candidate rooted at any other staged
            device is parked for :meth:`admit`; the rest are probed
            concurrently, and the ones that answer within the budget are
            this run's members.
        """
        self._probe_error = ""
        try:
            await self._probe(staged, own)
        except Exception as exc:  # noqa: BLE001 - the run opens regardless
            self._active, self._datakeys = [], {}
            self._probe_error = _one_line(exc)
            logger.exception(
                "background telemetry: the probe failed — no background columns "
                "this run"
            )

    async def _probe(self, staged: Sequence[Any], own: Sequence[Any]) -> None:
        own_roots = {id(root_ancestor(owner_of(obj))) for obj in own}
        staged_roots = {id(root_ancestor(owner_of(obj))) for obj in staged}
        excluded: list[Any] = []
        members: list[Any] = []
        self._parked = []
        for m in self._candidates:
            root = id(root_ancestor(owner_of(m)))
            if root in own_roots:
                excluded.append(m)
            elif root in staged_roots:
                self._parked.append(m)
            else:
                members.append(m)
        self._active, self._datakeys, self._staged_by_me = [], {}, set()
        _, reasons, stale = await self._activate(members, taken=set(), stage=True)
        self._dropped = list(reasons)
        logger.info(
            "background telemetry: %d device(s) read per shot (%d in the run "
            "already, %d parked for the first step's axes, %d dropped)",
            len({geecs_device_name(m) for m in self._active}),
            len({geecs_device_name(m) for m in excluded}),
            len({geecs_device_name(m) for m in self._parked}),
            len(reasons),
        )
        self._warn(reasons, stale)

    async def admit(self, movers: Sequence[Any]) -> None:
        """Let the movers' devices back in, minus the movers' own columns.

        Called by the GEECS per-step hooks with the step's motors, before
        anything is read.  A mover is a child of a device the probe parked
        whole (it saw the root the plan staged); its device returns as a
        member minus the keys the mover describes — the readback the row
        carries as the motor's own column — so a key is never read twice.
        A mover that is itself a root (a whole device moved) stays the
        run's, and so does one whose device is among the run's own readers
        — it was never parked, the row carries that device whole.  A
        mover that does not describe within the budget keeps its
        whole device out for this run (WARNING): a key read twice fails the
        run, a column missing from one scan does not.  Idempotent — a
        device admitted or refused once is not looked at again — and never
        raises.
        """
        try:
            await self._admit(movers)
        except Exception:  # noqa: BLE001 - the step goes on regardless
            logger.exception(
                "background telemetry: admitting the step's movers failed — the "
                "parked devices stay out of this run"
            )

    async def _admit(self, movers: Sequence[Any]) -> None:
        if not self._parked:
            return
        parked_roots = {id(root_ancestor(owner_of(m))) for m in self._parked}
        taken: set[str] = set()
        admitted: set[int] = set()
        refused: set[int] = set()
        for obj in movers:
            owner = owner_of(obj)
            root = root_ancestor(owner)
            if root is owner or id(root) not in parked_roots:
                continue  # a whole device moved, or nothing parked under it
            try:
                keys = await asyncio.wait_for(
                    maybe_await(owner.describe()), self.probe_timeout
                )
            except Exception as exc:  # noqa: BLE001 - never read a key twice
                refused.add(id(root))
                logger.warning(
                    "background telemetry: %s (%s) did not describe (%s) — its "
                    "whole device stays the run's this time",
                    geecs_device_name(obj),
                    getattr(obj, "name", obj),
                    _one_line(exc),
                )
                continue
            taken.update(keys)
            admitted.add(id(root))
        admitted -= refused
        decided = admitted | refused
        members = [
            m for m in self._parked if id(root_ancestor(owner_of(m))) in admitted
        ]
        self._parked = [
            m for m in self._parked if id(root_ancestor(owner_of(m))) not in decided
        ]
        if not members:
            return
        activated, reasons, stale = await self._activate(
            members, taken=taken, stage=False
        )
        self._dropped.extend(name for name in reasons if name not in self._dropped)
        logger.info(
            "background telemetry: %d scanned device(s) admitted minus the axes' "
            "own columns (%d dropped)",
            len({geecs_device_name(m) for m in activated}),
            len(reasons),
        )
        self._warn(reasons, stale)

    async def _activate(
        self, members: Sequence[Any], *, taken: set[str], stage: bool
    ) -> tuple[list[Any], dict[str, str], list[str]]:
        """Probe *members* concurrently; activate each that answers, minus the *taken* keys.

        *stage* says whether the probe stages a member (a candidate the
        RunEngine did not stage) or reads from the stage the RunEngine
        already did (a parked device admitted at the step — nothing here
        stages or unstages it).  A member whose every key is taken is the
        run's own: a stage of ours is released.  Returns the activated
        members, the dropped devices (name → why) and the devices kept
        although every reading was INVALID.
        """
        results = await asyncio.gather(
            *(self._probe_one(m, stage=stage) for m in members)
        )
        activated: list[Any] = []
        recorded: list[Any] = []  # answered, but every key of it is the run's own
        reasons: dict[str, str] = {}  # dropped device → why, the first reason seen
        stale: list[str] = []  # kept, but every reading INVALID
        for m, (keys, note) in zip(members, results):
            name = geecs_device_name(m)
            if keys is None:
                reasons.setdefault(name, note)
                continue
            mine = {key: datakey for key, datakey in keys.items() if key not in taken}
            if not mine:
                recorded.append(m)
                continue
            activated.append(m)
            self._datakeys[id(m)] = mine
            if note and name not in stale:
                stale.append(name)
        self._active.extend(activated)
        if stage:
            self._staged_by_me.update(id(m) for m in activated)
            await asyncio.gather(*(self._unstage_one(m) for m in recorded))
        return activated, reasons, stale

    def _warn(self, reasons: dict[str, str], stale: list[str]) -> None:
        if reasons:
            logger.warning(
                "background telemetry: left out of this run (probed again at the "
                "next): %s",
                "; ".join(f"{name} ({why})" for name, why in reasons.items()),
            )
        if stale:
            logger.warning(
                "background telemetry: %s — the gateway marks the device down; its "
                "columns read NaN until it recovers",
                ", ".join(f"{name} ({_STALE})" for name in stale),
            )

    async def _probe_one(
        self, member: Any, *, stage: bool = True
    ) -> tuple[dict[str, DataKey] | None, str]:
        """Connect, stage (unless the RunEngine did), describe and read *member* once, within the budget.

        Returns ``(data keys, note)``: the keys and ``""`` for a member that
        answered, the keys and :data:`_STALE` for one whose every reading is
        INVALID, ``(None, why)`` for one dropped.
        """

        async def go() -> tuple[dict[str, DataKey], str]:
            if not is_connected(member):
                # Shielded: the probe's timeout must not cancel the connect
                # itself.  ophyd-async caches the connect as a task on the
                # device, and a cancelled one poisons every later connect
                # and connected-check (review of #1018).  The connect keeps
                # its own timeout, so it ends on its own and the cache holds
                # a proper verdict for the next run's probe.
                task = asyncio.ensure_future(
                    member.connect(mock=self._mock, timeout=self.probe_timeout)
                )
                task.add_done_callback(_retrieved)
                await asyncio.shield(task)
            if stage:
                await _call(member, "stage")
            keys = dict(await maybe_await(member.describe()))
            readings = await maybe_await(member.read())  # the first cached value
            stale = bool(readings) and all(_invalid(r) for r in readings.values())
            return keys, _STALE if stale else ""

        try:
            return await asyncio.wait_for(go(), self.probe_timeout)
        except asyncio.TimeoutError:
            reason = f"no answer within {self.probe_timeout:.1f} s"
        except Exception as exc:  # noqa: BLE001 - a member never fails the run
            reason = _one_line(exc)
        logger.debug(
            "background telemetry: %s (%s) dropped — %s",
            geecs_device_name(member),
            getattr(member, "name", member),
            reason,
        )
        if stage:
            await self._unstage_one(member)  # a half-staged cache is released
        return None, reason

    async def _unstage_one(self, member: Any) -> None:
        try:
            await _call(member, "unstage")
        except Exception:  # noqa: BLE001 - best effort, after the run
            logger.debug(
                "background telemetry: %s did not unstage cleanly",
                geecs_device_name(member),
                exc_info=True,
            )

    # ------------------------------------------------------------- readable
    async def describe(self) -> dict[str, DataKey]:
        """The active members' data keys, as the probe cached them."""
        out: dict[str, DataKey] = {}
        for member in self._active:
            out.update(self._datakeys[id(member)])
        return out

    async def read(self) -> dict[str, Reading]:
        """Every declared key, from the members' caches; ``NaN`` where nothing live is."""
        if not self._active:
            return {}
        results = await asyncio.gather(*(self._read_one(m) for m in self._active))
        out: dict[str, Reading] = {}
        for member, readings in zip(self._active, results):
            for key, datakey in self._datakeys[id(member)].items():
                reading = None if readings is None else readings.get(key)
                if reading is None or _invalid(reading):
                    reading = _blank(datakey)
                out[key] = reading
        return out

    async def _read_one(self, member: Any) -> dict[str, Reading] | None:
        try:
            return await asyncio.wait_for(maybe_await(member.read()), self.read_timeout)
        except Exception as exc:  # noqa: BLE001 - never the run's failure
            if id(member) not in self._warned:
                self._warned.add(id(member))
                logger.warning(
                    "background telemetry: %s stopped answering (%s) — its columns "
                    "read NaN for the rest of the run",
                    geecs_device_name(member),
                    _one_line(exc),
                )
            return None


async def warm_up(
    candidates: Sequence[Any], *, mock: bool = False, timeout: float = WARM_UP_TIMEOUT_S
) -> list[str]:
    """Connect every candidate that is not yet, concurrently, within *timeout*.

    Returns the GEECS device names that did not connect (logged once, at
    WARNING).  Nothing is dropped: a run's probe retries an unconnected
    member every time, so a device the gateway starts serving later is
    back at the next run.  A member already connected is left alone (a
    mock's callbacks would be lost by a reconnect).
    """
    pending = [m for m in candidates if not is_connected(m)]
    started = time.monotonic()
    results = await asyncio.gather(
        *(m.connect(mock=mock, timeout=timeout) for m in pending),
        return_exceptions=True,
    )
    failed: list[str] = []
    for member, result in zip(pending, results):
        if isinstance(result, BaseException):
            name = geecs_device_name(member)
            if name not in failed:
                failed.append(name)
    logger.info(
        "background telemetry: %d of %d candidate device(s) connected at "
        "environment open in %.1f s",
        len({geecs_device_name(m) for m in candidates}) - len(failed),
        len({geecs_device_name(m) for m in candidates}),
        time.monotonic() - started,
    )
    if failed:
        logger.warning(
            "background telemetry: not connected at environment open (probed "
            "again at every run): %s",
            ", ".join(failed),
        )
    return failed


def warm_up_on(
    run_engine: Any,
    candidates: Sequence[Any],
    *,
    mock: bool = False,
    timeout: float = WARM_UP_TIMEOUT_S,
) -> list[str]:
    """:func:`warm_up` on the RunEngine's loop, from the thread building the worker."""
    return asyncio.run_coroutine_threadsafe(
        warm_up(candidates, mock=mock, timeout=timeout), run_engine._loop
    ).result(timeout + 10.0)


__all__ = [
    "PROBE_TIMEOUT_S",
    "READ_TIMEOUT_S",
    "WARM_UP_TIMEOUT_S",
    "BackgroundSnapshot",
    "warm_up",
    "warm_up_on",
]
