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

A set that fails is **logged at ERROR by the device**, naming the ``:SP``
PV and the cause by ``str`` (:meth:`CaSettable._set_logged`, shared by
the subclasses) before the status fails: the ``FailedStatus`` the
RunEngine throws into the plan carries only the status repr, and the
engine's own ``Run aborted`` traceback lands in the journal after the
scan log has closed — so without this line a refused motor put left no
PV name in the scan folder (GEECS-Plugins#868).

Every settable also carries a **user offset** (:attr:`CaSettable.offset`,
a soft signal): the EPICS motor record's user/dial split (``.OFF``,
``user = dial + offset``) held in software, because a GEECS variable has
no offset field — the raw GEECS value is the dial.  ``set()``, ``read()``
and ``locate()`` stay in the dial frame; the offset is consumed by the
pseudo positioners (:mod:`geecs_bluesky.devices.ca.pseudo`), whose
``mode: relative`` entries zero their components' offsets at scan start
(:meth:`CaSettable.set_current_position`, ophyd's spelling).  An
operator-facing "set current position as zero" with persistence is the
follow-on arc (`09_pseudo_transform.md` §3).
"""

from __future__ import annotations

import asyncio
import logging

from bluesky.protocols import Location
from ophyd_async.core import AsyncStatus, StandardReadable, soft_signal_r_and_setter
from ophyd_async.epics.core import epics_signal_r, epics_signal_rw

from geecs_bluesky.devices.ca._pv import ca_pv, setpoint_pv
from geecs_bluesky.devices.ca.gateway_put import GatewaySetpointPut, bare_pv
from geecs_bluesky.exceptions import failure_cause_text

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
        self._setpoint_pv = bare_pv(setpoint_pv(readback_pv))  # for the log
        self._setpoint = epics_signal_rw(datatype, setpoint_pv(readback_pv))
        # Layer-1 puts ride the shared gateway-put primitive (the one owner
        # of addressing/coercion/timeout policy); the typed signal stays the
        # transport — connect-time dtype check and the mock-backend seam.
        # timeout=None defers to the signal's own default per put.
        self._put = GatewaySetpointPut(signal=self._setpoint, timeout=None)
        with self.add_children_as_readables():
            setattr(self, _readback_attr, epics_signal_r(datatype, readback_pv))
        #: The user offset (``user = dial + offset``, EPICS ``.OFF``); ``0.0``
        #: until :meth:`set_current_position` moves it.  Not a readable: the
        #: event column stays the raw (dial) readback.
        self.offset, self._set_offset = soft_signal_r_and_setter(
            float, initial_value=0.0
        )
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
        return AsyncStatus(self._set_logged(value))

    async def _set_logged(self, value: float) -> None:
        """Run :meth:`_set_and_wait`; on failure, ERROR-log the PV and the cause first.

        The one place a failed set is named for the scan log
        (GEECS-Plugins#868): the cause by ``str`` through the shared
        :func:`~geecs_bluesky.exceptions.failure_cause_text` — a refused
        ``aioca.CANothing`` is falsy and carries the CA message only
        through ``str``.  The exception then propagates untouched into
        the status.
        """
        try:
            await self._set_and_wait(value)
        except Exception as exc:
            logger.error(
                "%s: set %s → %s failed (%s): %s",
                self.name,
                self._variable,
                value,
                self._setpoint_pv,
                failure_cause_text(exc),
            )
            raise

    async def set_current_position(self, position: float) -> None:
        """Redefine the user frame so the current readback reads *position*.

        ophyd's ``PositionerBase.set_current_position`` / the EPICS motor
        record's ``.SET`` mode: nothing moves; :attr:`offset` becomes
        ``position − readback``.  ``set_current_position(0.0)`` is "zero
        here" — what a relative pseudo positioner does to each component
        at stage.
        """
        readback = getattr(self, self._readback_attr_name)
        current = float(await readback.get_value())
        self._set_offset(float(position) - current)
        logger.info(
            "%s: user frame set: %s reads %s here (offset %s)",
            self.name,
            self._variable,
            position,
            float(position) - current,
        )

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
