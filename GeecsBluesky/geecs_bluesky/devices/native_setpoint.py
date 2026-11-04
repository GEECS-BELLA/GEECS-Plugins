"""NativeSetpointDevice — a supply served by plain EPICS records (geecs-epics) as settable children.

The third native kind (``NATIVE_DEVICES.md`` §8, the registry's
``setpoint_device``, the CAEN Easy-Driver pattern): each settable is a
readback record written at its setpoint record (``<record>:SP``, or the
family's setpoint role for it) under ``<experiment>:<device>:``.  For
StreamDevice, put-completion means the supply **accepted** the command,
not that it converged, so a set is never done on a "put ok": it is
refused before the put (nothing written) for a value outside ``DRVL``/
``DRVH`` (an ``ao`` would clamp and write), a status bit at 1 or an
INVALID readback; it fails at once when the setpoint record goes INVALID
on the write (the IOC's ``HIHI``/``LOLO`` refusal, a ``#NAK``) and while
waiting when a bit rises or the readback goes INVALID; it is done when the
readback is within tolerance, and times out in
``GeecsConfirmTimeoutError``'s words (``CaConfirmSettable``'s topology).

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

from bluesky.protocols import Location
from ophyd_async.core import AsyncStatus, DeviceVector, SignalR, StandardReadable
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw

from geecs_schemas import SETPOINT_SUFFIX

from geecs_bluesky.devices.ca._pv import CA_TRANSPORT_PREFIX
from geecs_bluesky.devices.ca._view import ScalarsView, owner_of
from geecs_bluesky.devices.ca.gateway_put import GatewaySetpointPut
from geecs_bluesky.devices.ca.liveness import down_natives
from geecs_bluesky.devices.ca.motor import within_tolerance
from geecs_bluesky.devices.ca.settable import log_failed_set
from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    GeecsConfirmTimeoutError,
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


def _ca(pv: str) -> str:
    return f"{CA_TRANSPORT_PREFIX}{pv}"


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
        The setpoint record's name under the prefix (default
        ``<record>:SP``).
    tolerance :
        How close the readback must come to the target for the set to be done.
    timeout :
        Seconds the set waits for that after the write.
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
        setpoint: str | None = None,
        tolerance: float,
        timeout: float,
        status: Iterable[str] = (),
        name: str = "",
    ) -> None:
        setpoint = setpoint or record + SETPOINT_SUFFIX
        self._geecs_device_name = device
        self._variable = record
        self._tolerance = float(tolerance)
        self._timeout = float(timeout)
        self._readback_pv = prefix + record
        self._setpoint_pv = prefix + setpoint
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

    @property
    def reading_key(self) -> str:
        """Event column containing this settable's readback."""
        return self.readback.name

    @property
    def _column_headers(self) -> dict[str, str]:
        """Readback event key → its ``Device Variable`` header (the s-file)."""
        return {self.readback.name: f"{self._geecs_device_name} {self._variable}"}

    def set(self, value: float) -> AsyncStatus:
        """Write *value*; done when the readback is within tolerance of it.

        Implements :class:`bluesky.protocols.Movable`.  A failure is
        ERROR-logged with its cause (the one failed-set line,
        :func:`~geecs_bluesky.devices.ca.settable.log_failed_set`), then raised.
        """
        logger.info(
            "%s: setting %s → %s (confirm on %s, tol=%.4g)",
            self.name,
            self._variable,
            value,
            self._readback_pv,
            self._tolerance,
        )
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

    async def _refuse_before_put(self, target: float) -> None:
        """The checks that write nothing: the value, the drive limits, the bits, the readback."""
        if not math.isfinite(target):
            raise self._refused(
                target, "the value is not a finite number", written=False
            )
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

    async def _set_and_wait(self, value: float) -> None:
        target = float(value)
        await self._refuse_before_put(target)
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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        while True:
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
            current = float(await self.readback.get_value(cached=False))
            if within_tolerance(current, target, self._tolerance):
                logger.debug(
                    "%s: confirmed %s = %r (target=%r, tol=%.4g)",
                    self.name,
                    self._readback_pv,
                    current,
                    target,
                    self._tolerance,
                )
                return
            if loop.time() > deadline:
                raise GeecsConfirmTimeoutError(
                    self._geecs_device_name,
                    self._variable,
                    self._readback_pv,
                    target=target,
                    current=current,
                    timeout=self._timeout,
                )
            await asyncio.sleep(POLL_INTERVAL_S)


class NativeSetpointDevice(StandardReadable):
    """A setpoint device's noun: its settables as Movable children, readable as one device.

    Parameters
    ----------
    device :
        The GEECS device name (the registry key; the namespace binding).
    prefix :
        The device's PV prefix, ``<experiment>:<device>:``.
    settables :
        Readback record → ``(tolerance, timeout)`` source: anything with
        ``tolerance`` and ``timeout`` attributes (the registry's
        ``SetpointSettable``).  Each binds as
        :func:`~geecs_bluesky.utils.settable_attribute` of its record (the
        rule a ``Device:Variable`` reference expands by).
    setpoints :
        Readback record → its setpoint record
        (``SetpointDevice.setpoint_records``); absent, ``<record>:SP``.
    readbacks :
        Further readback records, columns of the row.
    status :
        Status bits every settable checks.
    name :
        ophyd-async device name (the event keys' stem).

    Raises
    ------
    GeecsConfigurationError
        A record whose attribute would shadow one of the noun's own.
    """

    def __init__(
        self,
        device: str,
        prefix: str,
        settables: Mapping[str, Any],
        *,
        setpoints: Mapping[str, str] | None = None,
        readbacks: Iterable[str] = (),
        status: Iterable[str] = (),
        name: str = "",
    ) -> None:
        self._geecs_device_name = device
        self._settables: list[NativeSetpoint] = []
        self._readbacks: dict[str, SignalR[float]] = {}
        status = list(status)
        setpoints = dict(setpoints or {})
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
                    setpoint=setpoints.get(record),
                    tolerance=spec.tolerance,
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


__all__ = [
    "NativeSetpoint",
    "NativeSetpointDevice",
    "down_native_setpoint_devices",
    "native_setpoint_devices",
]
