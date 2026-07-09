"""CaSettable — a writable GEECS variable driven through the CA gateway.

The gateway exposes a settable GEECS variable as two PVs: a readback
(``[Experiment:]Device:Variable``, fed by the device stream) and a setpoint
(``…:SP``, whose CA puts the gateway forwards to the device over UDP).  This
device writes the setpoint and reads back the real value — the CA counterpart of
:class:`~geecs_bluesky.devices.settable.GeecsSettable`.

Completion semantics — CaSettable already waits
-----------------------------------------------
The ``:SP`` put is **not** fire-and-forget: the gateway forwards it as a
GEECS UDP set, which itself blocks until the device reports the value
converged (per the DB tolerance) or failed.  So a plain CaSettable already
carries GEECS's native set-completion — it is a legitimate choice for a real
positioner, not a lesser one.

:class:`~geecs_bluesky.devices.ca.motor.CaMotor` adds *one* thing on top: a
second poll of the streamed readback until it is within tolerance of the
target.  That extra poll only carries information when the readback is an
independent measurement that can lag or overshoot the commanded value (a
stage encoder).  When the readback just echoes what was written, the poll is
satisfied trivially and CaMotor is near-cosmetic.  Choose CaMotor when you do
not fully trust GEECS's own set-completion for that axis — not because
CaSettable "doesn't wait."

The open question is empirical, not schema-shaped: how reliable is GEECS's
blocking-set convergence for real stages?  If it is solid, CaMotor is nearly
cosmetic; if it is not, CaSettable is the quietly-risky choice for anything
mechanical.  That — not which ``kind`` a config declares — is where the real
risk lives.

Neither class covers the case where the variable you *set* differs from the
variable that *measures* the result (the EMQ triplet's ``Current_Limit`` vs
its measured current): CaMotor polls the readback of the *same* variable it
set, so it cannot confirm a decoupled measurement.  The schema names that
case via ``ScanVariable.confirm``; the device that would act on it is a later
milestone.
"""

from __future__ import annotations

import asyncio
import logging

from ophyd_async.core import AsyncStatus, StandardReadable
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw

from geecs_bluesky.devices.ca._pv import ca_pv

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
    datatype : type
        Scalar CA datatype (default ``float``).
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
        datatype: type = float,
        _readback_attr: str = "readback",
    ) -> None:
        readback_pv = ca_pv(experiment, device, variable)
        # Setpoint is not a child readable (no feedback loop): set() writes it,
        # read() reflects the streamed readback instead.
        self._setpoint = epics_signal_rw(datatype, f"{readback_pv}:SP")
        with self.add_children_as_readables():
            setattr(self, _readback_attr, epics_signal_r(datatype, readback_pv))
        super().__init__(name=name)
        self._readback_attr_name = _readback_attr
        self._geecs_device_name = device
        self._variable = variable
        self._settle_time = settle_time
        # Map the readback event key to its legacy "Device Variable" header for
        # the Tiled→s-file exporter (the scan-device column in a step scan).
        self._column_headers = {f"{name}-{_readback_attr}": f"{device} {variable}"}

    async def disconnect(self) -> None:
        """Per-scan teardown hook (scanner bridge ``_disconnect_devices_sync``).

        This device holds no persistent monitor subscription, so there is
        nothing to unsubscribe — the method exists so scanner teardown is
        uniform across every CA device type (``CaMotor`` inherits it) instead
        of raising a (swallowed) ``AttributeError``.
        """

    def set(self, value: float) -> AsyncStatus:
        """Put *value* to the setpoint PV; status resolves after ``settle_time``.

        Implements :class:`bluesky.protocols.Movable`.
        """
        logger.info("%s: setting %s → %s", self.name, self._variable, value)
        return AsyncStatus(self._set_and_wait(value))

    async def _set_and_wait(self, value: float) -> None:
        """Write the setpoint and wait ``settle_time`` (subclasses may poll)."""
        await self._setpoint.set(value)
        if self._settle_time > 0:
            await asyncio.sleep(self._settle_time)
