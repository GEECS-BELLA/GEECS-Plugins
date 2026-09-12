"""CaSettable — a writable GEECS variable driven through the CA gateway.

The gateway exposes a settable GEECS variable as two PVs: a readback
(``[experiment:]device:variable``, fed by the device stream) and a setpoint
(``…:SP``, forwarded to the device over UDP).  This device writes the
setpoint and reads back the real value.

The ``:SP`` put is not fire-and-forget: it rides GEECS's native **blocking**
set, which completes only when the device reports convergence (per the DB
tolerance) or failure — a plain CaSettable already waits.
:class:`~geecs_bluesky.devices.ca.motor.CaMotor` adds an independent
readback-tolerance poll on top; the decoupled set-X-confirm-Y case is
:class:`~geecs_bluesky.devices.ca.confirm.CaConfirmSettable`.
"""

from __future__ import annotations

import asyncio
import logging

from bluesky.protocols import Location
from ophyd_async.core import AsyncStatus, StandardReadable
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw

from geecs_bluesky.devices.ca._pv import ca_pv, setpoint_pv
from geecs_bluesky.devices.ca.gateway_put import GatewaySetpointPut

logger = logging.getLogger(__name__)


class CaSettable(StandardReadable):
    """A writable GEECS variable as a Bluesky ``Movable`` over CA.

    ``set(value)`` puts to the gateway's ``…:SP`` PV (which forwards to the
    device); ``read()`` returns the ``readback`` child (the real streamed value).

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"U_S1H"``).
    variable : str
        Writable variable name on that device (e.g. ``"Current"``).
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    settle_time : float
        Seconds to wait after the CA put resolves before completing the status.
    datatype : type or None
        Scalar CA datatype (default ``float``); ``None`` lets ophyd-async infer
        it from the PV at connect (enum / string / char-array setpoints).
    _readback_attr : str
        Name for the readback signal attribute on the device instance.
        Subclasses can override (e.g. ``"position"`` for motors).
    """

    def __init__(
        self,
        device: str,
        variable: str,
        *,
        experiment: str | None = None,
        name: str = "settable",
        settle_time: float = 0.0,
        datatype: type | None = float,
        _readback_attr: str = "readback",
    ) -> None:
        readback_pv = ca_pv(experiment, device, variable)
        # Setpoint is not a child readable (no feedback loop): set() writes it,
        # read() reflects the streamed readback instead.
        self._setpoint = epics_signal_rw(datatype, setpoint_pv(readback_pv))
        # Layer-1 puts ride the shared gateway-put primitive (the one owner
        # of addressing/coercion/timeout policy); the typed signal stays the
        # transport — connect-time dtype check and the mock-backend seam.
        # timeout=None defers to the signal's own default per put.
        self._put = GatewaySetpointPut(signal=self._setpoint, timeout=None)
        with self.add_children_as_readables():
            setattr(self, _readback_attr, epics_signal_r(datatype, readback_pv))
        super().__init__(name=name)
        self._readback_attr_name = _readback_attr
        self._geecs_device_name = device
        self._variable = variable
        self._settle_time = settle_time

    @property
    def _column_headers(self) -> dict[str, str]:
        """Readback event key → its legacy ``Device Variable`` header (the s-file).

        Computed from the readback signal's *current* name: a namespace
        child is renamed by its parent after construction.
        """
        readback = getattr(self, self._readback_attr_name)
        return {readback.name: f"{self._geecs_device_name} {self._variable}"}

    def set(self, value: float) -> AsyncStatus:
        """Put *value* to the setpoint PV; status resolves after ``settle_time``.

        Implements :class:`bluesky.protocols.Movable`.
        """
        logger.info("%s: setting %s → %s", self.name, self._variable, value)
        return AsyncStatus(self._set_and_wait(value))

    async def locate(self) -> Location:
        """Where the device is: the streamed readback, as both fields.

        Implements :class:`bluesky.protocols.Locatable` — what the ``rel_*``
        plans and ``reset_positions_wrapper`` stash before the first move
        and restore afterwards.  Without it bluesky falls back to
        ``obj.position``, which on a :class:`CaMotor` is the readback
        *signal*, not a number (2b acceptance, 2026-09-12: every ``rel_*``
        plan failed with ``unsupported operand type(s) for +: 'SignalR' and
        'float'``).  The readback stands in for the setpoint on purpose: the
        gateway's ``:SP`` PV is the last put *through the gateway*, not
        where the device is — a stage driven from LabVIEW since would make
        a relative scan run about the wrong point.
        """
        readback = getattr(self, self._readback_attr_name)
        value = await readback.get_value()
        return Location(setpoint=value, readback=value)

    async def _set_and_wait(self, value: float) -> None:
        """Write the setpoint and wait ``settle_time`` (subclasses may poll)."""
        await self._put.put(value)
        if self._settle_time > 0:
            await asyncio.sleep(self._settle_time)
