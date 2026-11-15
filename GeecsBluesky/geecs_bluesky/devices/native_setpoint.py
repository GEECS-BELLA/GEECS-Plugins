"""NativeSetpointDevice — a supply served by plain EPICS records (geecs-epics) as settable children.

The third native kind (``NATIVE_DEVICES.md`` §8, the registry's
``setpoint_device``, the CAEN Easy-Driver pattern): each settable is a
readback record written at its setpoint record (``<record>:SP``, or the
family's setpoint role for it) under ``<experiment>:<device>:``.  For
StreamDevice, put-completion means the supply **accepted** the command,
not that it converged, so a set is never done on a "put ok": it is
refused before the put (nothing written) for a value outside ``DRVL``/
``DRVH`` (an ``ao`` would clamp and write), a status bit at 1 or an
INVALID readback, and for no set tolerance; it fails at once when the
setpoint record goes INVALID on the write (the IOC's ``HIHI``/``LOLO``
refusal, a ``#NAK``) and while waiting when a bit rises or the readback
goes INVALID; it is done when the readback is within the set tolerance, and
times out in ``GeecsConfirmTimeoutError``'s words (``CaConfirmSettable``'s
topology).

The set tolerance is the IOC's: a record beside the readback (a supply's
``current:TOL``, geecs-epics decision 21), named by the family file (the
setpoint role's ``tolerance_by``: ``SetpointDevice.tolerance_records``),
read **at every set**, before the put, uncached — it changes only when its
IOC restarts onto a new configuration, so one read per set always confirms
against the value the IOC serves now, never a number kept from before.  It
is a tolerance only while the channel connects, its severity is not
INVALID and its value is finite and above zero; anything else refuses the
set before any write (:class:`~geecs_bluesky.exceptions.GeecsNoSetToleranceError`).
The bound scan plans read every native settable they will move once more
before the claim (:func:`native_set_tolerances`), so a scan with no
tolerance is refused with nothing claimed, and the values ride in the start
document (``set_tolerances``).  The record is not a child of the device: a
supply whose IOC does not serve it yet still connects, reads and is
recorded; only its sets are refused.

:class:`NativeSetpoint` is one settable (the Movable child ``mv``, a sweep
axis and an action's ``set`` step find); :class:`NativeSetpointDevice` is
the noun, a per-row scalar device.  Not here yet: the gate grown by the
movers, enum settables, a user offset for pseudos, the scanner's list.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

import bluesky.plan_stubs as bps
from bluesky.protocols import Location
from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    AsyncStatus,
    DeviceVector,
    SignalR,
    StandardReadable,
)
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw

from geecs_bluesky.devices.ca._pv import CA_TRANSPORT_PREFIX
from geecs_bluesky.devices.ca._view import ScalarsView, owner_of
from geecs_bluesky.devices.ca.gateway_put import GatewaySetpointPut
from geecs_bluesky.devices.ca.liveness import down_natives
from geecs_bluesky.devices.ca.motor import within_tolerance
from geecs_bluesky.devices.ca.settable import log_failed_set
from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    GeecsConfirmTimeoutError,
    GeecsNoSetToleranceError,
    GeecsSetpointRefusedError,
)
from geecs_bluesky.utils import is_connected, settable_attribute

logger = logging.getLogger(__name__)

#: The alarm severity that means the record's value is not to be trusted
#: (a refused write on a setpoint, a device not answering on a readback).
INVALID = "INVALID"

#: Seconds between readback reads while a set waits (the Easy-Driver's
#: readback scans at 1 s, so the arrival is seen within a fifth of a scan).
POLL_INTERVAL_S = 0.2

#: Seconds the liveness gate allows a setpoint device to connect and answer.
LIVENESS_TIMEOUT_S = 5.0

#: Seconds a set-tolerance read may take, the connect and the read each: a
#: record the IOC does not serve costs this once per set (and per run).
TOLERANCE_TIMEOUT_S = 2.0


def _ca(pv: str) -> str:
    return f"{CA_TRANSPORT_PREFIX}{pv}"


class _ToleranceRecord:
    """A settable's set-tolerance record: its value, severity and alarm — kept off the device tree.

    Not a ``Device``, so the signals are no children: the settable's
    connect, read and liveness never depend on the record (an IOC not yet
    serving it leaves the supply connected and recorded).  It connects
    itself at its first read (mock mode: with the settable's own connect,
    so a test can set its values), and again at the next read until it has.
    """

    def __init__(self, pv: str) -> None:
        self.pv = pv
        self.value = epics_signal_r(float, _ca(pv))
        self.severity = epics_signal_r(str, _ca(pv + ".SEVR"))
        self.alarm = epics_signal_r(str, _ca(pv + ".STAT"))

    def signals(self) -> tuple[SignalR[float], SignalR[str], SignalR[str]]:
        return self.value, self.severity, self.alarm

    async def connect(self, *, mock: bool = False, timeout: float) -> None:
        await asyncio.gather(
            *(
                s.connect(mock=mock, timeout=timeout)
                for s in self.signals()
                if not is_connected(s)  # a mock is never rebuilt
            )
        )

    async def read(self) -> tuple[float, str, str]:
        value, severity, alarm = await asyncio.gather(
            *(s.get_value(cached=False) for s in self.signals())
        )
        return float(value), str(severity), str(alarm)


class NativeSetpoint(StandardReadable):
    """One settable: write the setpoint record, confirm on the readback record.

    Parameters
    ----------
    device :
        The GEECS device name (the registry key).
    record :
        The readback record's name under the prefix — the registry's
        settable name, and the variable the s-file header spells.
    prefix :
        The device's PV prefix, ``<experiment>:<device>:``.
    setpoint :
        The setpoint record's name under the prefix (the family's setpoint
        role for *record*, ``SetpointDevice.setpoint_records``).
    tolerance :
        The set-tolerance record's name under the prefix (the family's
        ``tolerance_by`` role for that setpoint,
        ``SetpointDevice.tolerance_records``): how close the readback must
        come, read at every set.
    timeout :
        Seconds the set waits for the readback after the write: the
        client's patience (the registry's ``SetpointSettable.timeout``).
    status :
        The device's status bits (two-state records under the prefix,
        1 = tripped), checked before the put and while waiting.
    name :
        ophyd-async name (the parent renames the child ``<device>-<record>``).
    """

    def __init__(
        self,
        device: str,
        record: str,
        prefix: str,
        *,
        setpoint: str,
        tolerance: str,
        timeout: float,
        status: Iterable[str] = (),
        name: str = "",
    ) -> None:
        self._geecs_device_name = device
        self._variable = record
        self._timeout = float(timeout)
        self._readback_pv = prefix + record
        self._setpoint_pv = prefix + setpoint
        self._tolerance = _ToleranceRecord(prefix + tolerance)
        #: The last tolerance read (a change is said once, at the read that sees it).
        self._last_tolerance: float | None = None
        with self.add_children_as_readables():
            self.readback = epics_signal_r(float, _ca(self._readback_pv))
        # Not readables: the setpoint is written, never a column; the alarm
        # fields and the drive limits are read by the set alone.
        self.setpoint = epics_signal_rw(float, _ca(self._setpoint_pv))
        self.setpoint_severity = epics_signal_r(str, _ca(self._setpoint_pv + ".SEVR"))
        self.setpoint_alarm = epics_signal_r(str, _ca(self._setpoint_pv + ".STAT"))
        self.drive_high = epics_signal_r(float, _ca(self._setpoint_pv + ".DRVH"))
        self.drive_low = epics_signal_r(float, _ca(self._setpoint_pv + ".DRVL"))
        self.readback_severity = epics_signal_r(str, _ca(self._readback_pv + ".SEVR"))
        self.readback_alarm = epics_signal_r(str, _ca(self._readback_pv + ".STAT"))
        self._status: dict[str, SignalR[bool]] = {}
        for bit in status:
            signal = epics_signal_r(bool, _ca(prefix + bit))
            setattr(self, f"status_{settable_attribute(bit)}", signal)
            self._status[bit] = signal
        # The one put primitive (#868: a refused put must not read as success).
        self._put = GatewaySetpointPut(signal=self.setpoint, timeout=self._timeout)
        super().__init__(name=name)

    async def connect(
        self,
        mock: Any = False,
        timeout: float = DEFAULT_TIMEOUT,
        force_reconnect: bool = False,
    ) -> None:
        """Connect the settable; in mock mode its set-tolerance record too (real mode: at its first read)."""
        await super().connect(
            mock=mock, timeout=timeout, force_reconnect=force_reconnect
        )
        if mock:
            await self._tolerance.connect(mock=True, timeout=timeout)

    @property
    def tolerance_pv(self) -> str:
        """The set-tolerance record's PV (the family's ``tolerance_by`` role)."""
        return self._tolerance.pv

    def _no_tolerance(
        self, read: str, target: float | None
    ) -> GeecsNoSetToleranceError:
        return GeecsNoSetToleranceError(
            self._geecs_device_name,
            self._variable,
            self._tolerance.pv,
            read,
            target=target,
        )

    async def read_tolerance(self, target: float | None = None) -> float:
        """The IOC's set tolerance, read now (connected if needed, uncached).

        A tolerance only while the record connects, its severity is not
        INVALID and its value is finite and above zero; anything else raises
        (*target* is the set's, when a set asks).  A value other than the
        last one read is logged at WARNING once: its IOC restarted onto a new
        configuration, and the sets confirm within the new value.

        Raises
        ------
        GeecsNoSetToleranceError
            No tolerance, naming the record and what was read of it.
        """
        record = self._tolerance
        mock = getattr(self.readback, "_mock", None) is not None
        try:
            await record.connect(mock=mock, timeout=TOLERANCE_TIMEOUT_S)
        except Exception as exc:  # every failure is the same answer
            raise self._no_tolerance(
                f"does not connect within {TOLERANCE_TIMEOUT_S:g} s (the IOC does "
                f"not serve it, or is down: {type(exc).__name__})",
                target,
            ) from exc
        try:
            value, severity, alarm = await asyncio.wait_for(
                record.read(), TOLERANCE_TIMEOUT_S
            )
        except Exception as exc:  # every failure is the same answer
            raise self._no_tolerance(
                f"did not answer within {TOLERANCE_TIMEOUT_S:g} s "
                f"({type(exc).__name__}: {' '.join(str(exc).split())})",
                target,
            ) from exc
        if severity == INVALID:
            raise self._no_tolerance(f"reads {value:g} INVALID ({alarm})", target)
        if not math.isfinite(value):
            raise self._no_tolerance(f"reads {value} (not a finite number)", target)
        if value <= 0:
            raise self._no_tolerance(f"reads {value:g} (not above zero)", target)
        previous, self._last_tolerance = self._last_tolerance, value
        if previous is not None and previous != value:
            logger.warning(
                "%s: the set tolerance %s changed from %g to %g since it was last "
                "read (its IOC restarted onto a new configuration): sets now "
                "confirm within %g",
                self.name,
                self._tolerance.pv,
                previous,
                value,
                value,
            )
        return value

    @property
    def reading_key(self) -> str:
        """Event column containing this settable's readback."""
        return self.readback.name

    @property
    def _column_headers(self) -> dict[str, str]:
        """Readback event key → its ``Device Variable`` header (the s-file)."""
        return {self.readback.name: f"{self._geecs_device_name} {self._variable}"}

    def set(self, value: float) -> AsyncStatus:
        """Write *value*; done when the readback is within the IOC's set tolerance of it.

        Implements :class:`bluesky.protocols.Movable`.  A failure is
        ERROR-logged with its cause (the one failed-set line,
        :func:`~geecs_bluesky.devices.ca.settable.log_failed_set`), then raised.
        """
        return AsyncStatus(self._set_logged(value))

    async def _set_logged(self, value: float) -> None:
        try:
            await self._set_and_wait(value)
        except Exception as exc:
            log_failed_set(
                logger, self.name, self._variable, value, self._setpoint_pv, exc
            )
            raise

    async def locate(self) -> Location:
        """Where the device is: the readback, as both fields (``CaSettable.locate``'s reason)."""
        value = await self.readback.get_value()
        return Location(setpoint=value, readback=value)

    def _refused(self, value: float, reason: str, *, written: bool) -> Exception:
        return GeecsSetpointRefusedError(
            self._geecs_device_name, self._variable, value, reason, written=written
        )

    async def _raised_bits(self) -> list[str]:
        values = await asyncio.gather(
            *(s.get_value(cached=False) for s in self._status.values())
        )
        return [bit for bit, up in zip(self._status, values) if up]

    async def _readback_invalid(self) -> str | None:
        severity, alarm = await asyncio.gather(
            self.readback_severity.get_value(cached=False),
            self.readback_alarm.get_value(cached=False),
        )
        if severity != INVALID:
            return None
        return (
            f"the readback {self._readback_pv} is INVALID ({alarm}): the device "
            "is not answering"
        )

    async def _refuse_before_put(self, target: float) -> float:
        """The checks that write nothing: the value, the tolerance, the drive limits, the bits, the readback.

        Returns the set tolerance read for this set.
        """
        if not math.isfinite(target):
            raise self._refused(
                target, "the value is not a finite number", written=False
            )
        tolerance = await self.read_tolerance(target)
        high, low = await asyncio.gather(
            self.drive_high.get_value(cached=False),
            self.drive_low.get_value(cached=False),
        )
        # An ao record with DRVH <= DRVL drives anything (no limits set).
        if high > low and not low <= target <= high:
            raise self._refused(
                target,
                f"outside the drive limits {low:g} <= x <= {high:g} of "
                f"{self._setpoint_pv} (the record would clamp it and write)",
                written=False,
            )
        raised = await self._raised_bits()
        if raised:
            raise self._refused(
                target,
                f"status bit(s) {', '.join(raised)} read 1 (tripped)",
                written=False,
            )
        invalid = await self._readback_invalid()
        if invalid is not None:
            raise self._refused(target, invalid, written=False)
        return tolerance

    async def _set_and_wait(self, value: float) -> None:
        target = float(value)
        tolerance = await self._refuse_before_put(target)
        logger.info(
            "%s: setting %s → %s (confirm on %s within %g, the IOC's %s)",
            self.name,
            self._variable,
            value,
            self._readback_pv,
            tolerance,
            self._tolerance.pv,
        )
        await self._put.put(target)
        # The put-callback returns once the record has processed this write,
        # so its alarm is this write's: INVALID is the IOC's HIHI/LOLO
        # refusal (nothing reached the device) or the supply's #NAK.
        severity, alarm = await asyncio.gather(
            self.setpoint_severity.get_value(cached=False),
            self.setpoint_alarm.get_value(cached=False),
        )
        if severity == INVALID:
            raise self._refused(
                target,
                f"{self._setpoint_pv} went INVALID ({alarm}) on the write: the IOC "
                "or the supply refused it",
                written=True,
            )
        await self._confirm(target, tolerance)

    async def _poll_once(self, target: float) -> float:
        """One confirmation read: the bits, the readback's severity, its value."""
        raised = await self._raised_bits()
        if raised:
            raise self._refused(
                target,
                f"status bit(s) {', '.join(raised)} rose while waiting for "
                "the readback",
                written=True,
            )
        invalid = await self._readback_invalid()
        if invalid is not None:
            raise self._refused(target, invalid, written=True)
        return float(await self.readback.get_value(cached=False))

    async def _confirm(self, target: float, tolerance: float) -> None:
        """Wait for the readback within tolerance, every read inside one deadline.

        Each poll is bounded by the time left, so a slow or stalled CA read
        cannot carry the set past its timeout, and a value that arrives
        after the deadline is not accepted.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        current: float | str = "nothing (no read completed)"

        def expired() -> GeecsConfirmTimeoutError:
            return GeecsConfirmTimeoutError(
                self._geecs_device_name,
                self._variable,
                self._readback_pv,
                target=target,
                current=current,
                timeout=self._timeout,
            )

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise expired()
            try:
                current = await asyncio.wait_for(self._poll_once(target), remaining)
            except TimeoutError:
                # Ours once the deadline has passed; otherwise the read's own
                # CA timeout, which is not a confirmation timeout.
                if loop.time() >= deadline:
                    raise expired() from None
                raise
            if loop.time() > deadline:
                raise expired()
            if within_tolerance(current, target, tolerance):
                logger.debug(
                    "%s: confirmed %s = %r (target=%r, tol=%.4g)",
                    self.name,
                    self._readback_pv,
                    current,
                    target,
                    tolerance,
                )
                return
            await asyncio.sleep(min(POLL_INTERVAL_S, max(0.0, deadline - loop.time())))


class NativeSetpointDevice(StandardReadable):
    """A setpoint device's noun: its settables as Movable children, readable as one device.

    Parameters
    ----------
    device :
        The GEECS device name (the registry key; the namespace binding).
    prefix :
        The device's PV prefix, ``<experiment>:<device>:``.
    settables :
        Readback record → its timeout source: anything with a ``timeout``
        attribute (the registry's ``SetpointSettable``).  Each binds as
        :func:`~geecs_bluesky.utils.settable_attribute` of its record (the
        rule a ``Device:Variable`` reference expands by).
    setpoints :
        Readback record → its setpoint record
        (``SetpointDevice.setpoint_records``, the family's rule).
    tolerances :
        Readback record → its set-tolerance record
        (``SetpointDevice.tolerance_records``, the family's rule).
    readbacks :
        Further readback records, columns of the row.
    status :
        Status bits every settable checks.
    name :
        ophyd-async device name (the event keys' stem).

    Raises
    ------
    GeecsConfigurationError
        A record whose attribute would shadow one of the noun's own, or a
        settable with no setpoint or set-tolerance record.
    """

    def __init__(
        self,
        device: str,
        prefix: str,
        settables: Mapping[str, Any],
        *,
        setpoints: Mapping[str, str],
        tolerances: Mapping[str, str],
        readbacks: Iterable[str] = (),
        status: Iterable[str] = (),
        name: str = "",
    ) -> None:
        self._geecs_device_name = device
        self._settables: list[NativeSetpoint] = []
        self._readbacks: dict[str, SignalR[float]] = {}
        status = list(status)
        unrecorded = [r for r in settables if r not in setpoints or r not in tolerances]
        if unrecorded:
            raise GeecsConfigurationError(
                f"device namespace: {device}: settable(s) {unrecorded} have no "
                "setpoint or set-tolerance record (the family's setpoint role and "
                "its tolerance_by)"
            )
        # The scalars-only view every namespace device carries, and the
        # listed readbacks' severities (the liveness read; the settables
        # carry their own) — bound first, so a record cannot shadow them.
        self.scalars = ScalarsView(self)
        readbacks = list(readbacks)
        self.readback_severities = DeviceVector(
            {
                index: epics_signal_r(str, _ca(prefix + record + ".SEVR"))
                for index, record in enumerate(readbacks)
            }
        )
        with self.add_children_as_readables():
            for record, spec in settables.items():
                child = NativeSetpoint(
                    device,
                    record,
                    prefix,
                    setpoint=setpoints[record],
                    tolerance=tolerances[record],
                    timeout=spec.timeout,
                    status=status,
                )
                self._bind(device, record, child)
                self._settables.append(child)
            for record in readbacks:
                signal = epics_signal_r(float, _ca(prefix + record))
                self._bind(device, record, signal)
                self._readbacks[record] = signal
        super().__init__(name=name)

    def _bind(self, device: str, record: str, child: Any) -> None:
        attr = settable_attribute(record)
        if hasattr(type(self), attr) or hasattr(self, attr):
            raise GeecsConfigurationError(
                f"device namespace: {device}: record {record!r} would bind as "
                f"{attr!r}, a name of the device itself"
            )
        setattr(self, attr, child)  # ophyd-async registers + names it

    @property
    def settables(self) -> tuple[NativeSetpoint, ...]:
        """The settables, in the registry's order."""
        return tuple(self._settables)

    @property
    def _column_headers(self) -> dict[str, str]:
        """Every readback column → its ``Device Variable`` header."""
        headers: dict[str, str] = {}
        for child in self._settables:
            headers.update(child._column_headers)
        for record, signal in self._readbacks.items():
            headers[signal.name] = f"{self._geecs_device_name} {record}"
        return headers

    def severity_signals(self) -> list[SignalR[str]]:
        """Every readback's ``.SEVR`` — the settables' and the listed ones (the liveness read)."""
        return [
            *(c.readback_severity for c in self._settables),
            *self.readback_severities.values(),
        ]


def native_setpoint_devices(devices: Iterable[Any]) -> list[NativeSetpointDevice]:
    """The setpoint devices among *devices* (a ``.scalars`` view counts as its owner)."""
    return [d for d in map(owner_of, devices) if isinstance(d, NativeSetpointDevice)]


async def _down(device: NativeSetpointDevice) -> bool:
    if not is_connected(device):
        await device.connect(timeout=LIVENESS_TIMEOUT_S)
    severities = await asyncio.wait_for(
        asyncio.gather(*(s.get_value(cached=False) for s in device.severity_signals())),
        LIVENESS_TIMEOUT_S,
    )
    return any(s == INVALID for s in severities)


def down_native_setpoint_devices(devices: Iterable[Any]):
    """Plan: the setpoint devices among *devices* with a readback INVALID, or whose IOC is down.

    §10's per-kind rule: a StreamDevice readback whose device stops
    answering goes INVALID (TIMEOUT/COMM), so any readback at INVALID reads
    down; a connect or read that fails means the IOC is down
    (:func:`~geecs_bluesky.devices.ca.liveness.down_natives`), bounded by
    ``LIVENESS_TIMEOUT_S``.

    Returns
    -------
    list of str
        The GEECS names of the devices judged down.
    """
    return (yield from down_natives(native_setpoint_devices(devices), _down))


def native_set_tolerances(movers: Iterable[Any]):
    """Plan: read the set tolerance of every native settable a run will move; refuse the run if one has none.

    The bound scan plans call it before the claim and before the trigger
    box is driven (after the liveness gate), with the movers they know
    before the run (a sweep's axes, optimize's movables): a scan whose
    native settable has no tolerance is refused with nothing claimed, never
    at its first point.  Each set reads the record again (a value that
    changed since is used, and said); this read is the run's early answer
    and its record.  Movers that are not native settables are skipped; a
    pseudo positioner over native settables cannot be built yet (a
    ``NativeSetpoint`` has no user offset), and when one is, its components
    belong here too.

    Returns
    -------
    dict
        ``"Device:variable"`` → ``{"pv": <the tolerance record>, "value":
        <its value>}``, the start document's ``set_tolerances``; empty when
        the run moves no native settable.

    Raises
    ------
    GeecsNoSetToleranceError
        The first settable with no tolerance, every other one in its notes.
    """
    found = list({id(m): m for m in movers if isinstance(m, NativeSetpoint)}.values())
    if not found:
        return {}

    async def run() -> list[float | BaseException]:
        return await asyncio.gather(
            *(s.read_tolerance() for s in found), return_exceptions=True
        )

    (task,) = yield from bps.wait_for([run])
    results = task.result()
    failed = [r for r in results if isinstance(r, BaseException)]
    if failed:
        first = failed[0]
        for other in failed[1:]:
            first.add_note(str(other))
        first.add_note(
            "the run was refused before the trigger box was driven (nothing "
            "claimed, no scan folder): restart the IOC onto a configuration "
            "that serves the record, then resubmit"
        )
        raise first
    return {
        f"{s._geecs_device_name}:{s._variable}": {"pv": s.tolerance_pv, "value": v}
        for s, v in zip(found, results)
    }


__all__ = [
    "NativeSetpoint",
    "NativeSetpointDevice",
    "down_native_setpoint_devices",
    "native_set_tolerances",
    "native_setpoint_devices",
]
