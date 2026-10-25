"""NativeMotor — a positioner served by EPICS motor records (geecs-epics) as a scan axis per axis.

The second native device kind (``NATIVE_DEVICES.md`` §7; the registry's
``epics_motor``): one device, one stock motor record per axis, under
``<experiment>:<device>:<axis>``.  The stock ophyd-async
:class:`~ophyd_async.epics.motor.Motor` already speaks the record — the
move is a put-callback on ``.VAL`` the record completes when ``.DMOV``
returns to 1 (the device's own verdict), the soft limits ``.HLM``/``.LLM``
are checked **before the put** (``MotorLimitsError``), the timeout comes
from ``.VELO`` and ``.ACCL``, and the record retries to its own ``.RDBD``.
What GEECS adds is thin: :class:`NativeMotorAxis` (the post-move readback
check that names a tripped limit switch, the ERROR log of a failed set,
the GEECS names and the s-file header) and :class:`NativeMotor` (the
parent noun whose axes are the Movable children ``mv``, a sweep axis and an
action's ``set`` step find; readable as one per-row scalar device).

Not here yet (the design's later slices): the liveness gate grown by the
run's movers, a soft user offset for a pseudo positioner over native axes
(the stock ``offset`` signal is the record's own ``.OFF``), the scanner's
settables list and the archive set.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

from ophyd_async.core import (
    CALCULATE_TIMEOUT,
    CalculatableTimeout,
    SignalR,
    StandardReadable,
    TimeoutCalculator,
    WatchableAsyncStatus,
)
from ophyd_async.epics.motor import Motor, MotorMoveLogic

from geecs_bluesky.devices.ca._view import ScalarsView
from geecs_bluesky.devices.ca.motor import within_tolerance
from geecs_bluesky.devices.ca.settable import log_failed_set
from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    GeecsMotorStoppedShortError,
)
from geecs_bluesky.utils import settable_attribute

logger = logging.getLogger(__name__)


@dataclass
class NativeMotorMoveLogic(MotorMoveLogic):
    """The stock motor move, then the readback checked against the record's deadband.

    The stock ``move`` is the ``.VAL`` put with completion: done when the
    record says done.  The record says done short of the target too — a
    limit switch, a following error — and a refused put reads as success
    on this transport (#868), so the readback is read back once the put
    completes and must sit within ``.RDBD`` of the target.
    """

    deadband: SignalR[float]
    high_limit_switch: SignalR[int]
    low_limit_switch: SignalR[int]
    device: str
    axis: str
    #: Set by :meth:`stop`: a move the operator stopped is not "stopped short".
    stop_requested: bool = False

    async def stop(self) -> None:
        """Write ``.STOP``; the move in flight then ends as stopped, not as a fault."""
        self.stop_requested = True
        await super().stop()

    async def move(self, new_position: float, timeout: TimeoutCalculator) -> None:
        """Put ``.VAL`` with completion, then confirm the readback within ``.RDBD``.

        The confirm reads the record afresh (``cached=False``): ``set`` is
        observing the readback meanwhile, and the put-callback's completion
        can land before the monitor of the final ``.RBV`` — a cached read
        would be the previous update and fail a move that arrived.
        """
        self.stop_requested = False
        await super().move(new_position, timeout)
        if self.stop_requested:
            return  # ``StandardMovable.set`` reports the stop itself
        current, deadband, high, low = await asyncio.gather(
            self.readback.get_value(cached=False),
            self.deadband.get_value(),
            self.high_limit_switch.get_value(cached=False),
            self.low_limit_switch.get_value(cached=False),
        )
        if within_tolerance(float(current), float(new_position), float(deadband)):
            return
        raise GeecsMotorStoppedShortError(
            self.device,
            self.axis,
            target=float(new_position),
            current=float(current),
            deadband=float(deadband),
            high_limit_switch=bool(high),
            low_limit_switch=bool(low),
        )


class NativeMotorAxis(Motor):
    """One axis: the stock motor record device with the GEECS move verdict on top.

    Parameters
    ----------
    device :
        The GEECS device name (the registry key).
    axis :
        The axis — the motor record's name under the device's prefix, and
        the variable the s-file header spells (``U_ESP_JetXYZ x``).
    prefix :
        The device's PV prefix, ``<experiment>:<device>:``.
    name :
        ophyd-async name (the namespace renames the child ``<device>-<axis>``;
        the readback column carries the axis's own name, as a stock motor's).
    """

    def __init__(self, device: str, axis: str, prefix: str, name: str = "") -> None:
        self._geecs_device_name = device
        self._variable = axis
        super().__init__(prefix + axis, name=name)

    @cached_property
    def movable_logic(self) -> NativeMotorMoveLogic:
        """The stock move logic with the post-move readback check."""
        return NativeMotorMoveLogic(
            readback=self.user_readback,
            setpoint=self.user_setpoint,
            motor_stop=self.motor_stop,
            low_limit_travel=self.low_limit_travel,
            high_limit_travel=self.high_limit_travel,
            dial_low_limit_travel=self.dial_low_limit_travel,
            dial_high_limit_travel=self.dial_high_limit_travel,
            velocity=self.velocity,
            acceleration_time=self.acceleration_time,
            deadband=self.deadband,
            high_limit_switch=self.high_limit_switch,
            low_limit_switch=self.low_limit_switch,
            device=self._geecs_device_name,
            axis=self._variable,
        )

    @property
    def reading_key(self) -> str:
        """Event column containing this axis's readback (the motor's own name)."""
        return self.user_readback.name

    @property
    def _column_headers(self) -> dict[str, str]:
        """Readback event key → its ``Device Variable`` header (the s-file)."""
        return {self.user_readback.name: f"{self._geecs_device_name} {self._variable}"}

    def set(
        self, new_position: float, timeout: CalculatableTimeout = CALCULATE_TIMEOUT
    ) -> WatchableAsyncStatus:
        """Move to *new_position*; a failure is ERROR-logged with its cause, then raised.

        The stock status (limits checked before the put, progress watched
        on the readback, done on the record's completion), with the GEECS
        log line every settable writes before its status fails: the
        ``FailedStatus`` the RunEngine raises carries only the status repr.
        """
        logger.info("%s: moving %s → %s", self.name, self._variable, new_position)
        status = super().set(new_position, timeout)

        def log_failed(done: WatchableAsyncStatus) -> None:
            exc = done.exception()
            if exc is not None:
                log_failed_set(
                    logger,
                    self.name,
                    self._variable,
                    new_position,
                    self.user_setpoint.source,
                    exc,
                )

        status.add_callback(log_failed)
        return status


class NativeMotor(StandardReadable):
    """A positioner's noun: its axes as Movable children, readable as one device.

    Parameters
    ----------
    device :
        The GEECS device name (the registry key; the namespace binding).
    prefix :
        The device's PV prefix, ``<experiment>:<device>:``.
    axes :
        The registry's axes, in order: each binds as
        :func:`~geecs_bluesky.utils.settable_attribute` of its name (the
        rule a ``Device:Variable`` reference expands by).
    name :
        ophyd-async device name (the event keys' stem).

    Raises
    ------
    GeecsConfigurationError
        An axis whose attribute would shadow one of the noun's own.
    """

    def __init__(
        self, device: str, prefix: str, axes: Iterable[str], name: str = ""
    ) -> None:
        self._geecs_device_name = device
        self._axes: list[NativeMotorAxis] = []
        # The scalars-only view every namespace device carries (``X.scalars``
        # in a plan's detector list, a preset's ``save_images: false``) —
        # bound first, so an axis cannot shadow it.
        self.scalars = ScalarsView(self)
        with self.add_children_as_readables():
            for axis in axes:
                attr = settable_attribute(axis)
                if hasattr(type(self), attr) or hasattr(self, attr):
                    raise GeecsConfigurationError(
                        f"device namespace: {device}: axis {axis!r} would bind as "
                        f"{attr!r}, a name of the device itself"
                    )
                child = NativeMotorAxis(device, axis, prefix)
                setattr(self, attr, child)  # ophyd-async registers + names it
                self._axes.append(child)
        super().__init__(name=name)

    @property
    def axes(self) -> tuple[NativeMotorAxis, ...]:
        """The axes, in the registry's order."""
        return tuple(self._axes)

    @property
    def _column_headers(self) -> dict[str, str]:
        """Every axis's readback column → its ``Device Variable`` header."""
        headers: dict[str, str] = {}
        for axis in self._axes:
            headers.update(axis._column_headers)
        return headers


__all__ = ["NativeMotor", "NativeMotorAxis", "NativeMotorMoveLogic"]
