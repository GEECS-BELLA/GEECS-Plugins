"""CaMotor — position-feedback motor driven through the CA gateway.

Two layers of convergence:

1. The gateway's ``…:SP`` write rides the blocking GEECS UDP set (given the
   full ``move_timeout`` budget rather than the short CA default — a slow
   axis is not a dead one), so the put already carries GEECS's native
   convergence; a plain CaSettable is not "non-blocking."
2. A readback polling loop then confirms the *streamed* position is within
   ``tolerance`` of the target — belt-and-suspenders for devices whose UDP
   set-completion semantics are ambiguous.  It only adds information when
   the readback is an independent measurement (a stage encoder), not an
   echo of the command.

The poll reads the readback of the *same* variable it set; the decoupled
set-X-confirm-Y case is
:class:`~geecs_bluesky.devices.ca.confirm.CaConfirmSettable`.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from ophyd_async.core import AsyncStatus

from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.exceptions import GeecsMotorTimeoutError

logger = logging.getLogger(__name__)

_DEFAULT_MOVE_TIMEOUT = 30.0  # seconds

#: Fallback move-completion tolerance, used only when the GEECS DB records no
#: usable ``tolerance`` for the variable.  :meth:`GeecsSession.move_tolerance`
#: resolves the per-axis DB value; this is what it returns when it cannot.
DEFAULT_TOLERANCE = 0.005

#: A DB tolerance larger than this fraction of the variable's own travel span
#: is logged as suspect.  Span-relative, not absolute: tolerances are in each
#: variable's own units (µm on U_CompAeroTech, mm on the ESPs), so any fixed
#: threshold flags correctly-configured axes purely for their unit choice —
#: exactly the unit-blindness it is meant to detect.
TOLERANCE_SPAN_FRACTION = 0.01

# Binary floating point puts an exactly-on-tolerance arrival a few ULPs *over*
# the limit: |-10.505 - -10.5| evaluates to 0.005000000000000782, not 0.005.
# A stage that landed exactly on tolerance therefore polled for the full
# move_timeout and paused the scan for an operator (U_ModeImagerESP, Scan034).
#
# The error in |current - value| scales with the *operands*, not the tolerance
# (~ULP(|position|) = |x| * 2.2e-16), so the slack must too: a tolerance-relative
# epsilon under-covers exactly the large-coordinate axes (U_CompAeroTech reads
# ~4e4). Four ULPs of the larger operand covers the subtraction plus the
# comparison with room to spare, and stays far below any real tolerance.
ULP_SLACK = 4 * sys.float_info.epsilon


class CaMotor(CaSettable):
    """GEECS motor over gateway PVs, with position-feedback polling.

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"U_ESP_JetXYZ"``).
    variable : str
        Position variable name (e.g. ``"Position.Axis 1"``).
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    tolerance : float
        Move completion tolerance.  ``set()`` resolves when
        ``|readback − setpoint| ≤ tolerance``, with a few ULPs of the larger
        operand as slack so an arrival landing exactly on the tolerance is not
        lost to binary floating-point representation.  Defaults to
        :data:`DEFAULT_TOLERANCE`; :meth:`GeecsSession.motor` normally passes
        the GEECS DB's per-variable value instead.
    settle_time : float
        Extra seconds to wait after arrival before completing the status.
    move_timeout : float
        Maximum seconds for the whole move (CA put budget and polling
        deadline).  Default ``30.0``.
    """

    def __init__(
        self,
        device: str,
        variable: str,
        *,
        experiment: str | None = None,
        name: str = "motor",
        tolerance: float = DEFAULT_TOLERANCE,
        settle_time: float = 0.0,
        move_timeout: float = _DEFAULT_MOVE_TIMEOUT,
    ) -> None:
        super().__init__(
            device,
            variable,
            experiment=experiment,
            name=name,
            settle_time=settle_time,
            _readback_attr="position",
        )
        self._tolerance = tolerance
        self._move_timeout = move_timeout

    def set(self, value: float) -> AsyncStatus:
        """Move to *value* and return a Status that completes on arrival.

        Implements :class:`bluesky.protocols.Movable`.
        """
        logger.info(
            "%s: moving %s → %s (tol=%.4g)",
            self.name,
            self._variable,
            value,
            self._tolerance,
        )
        return AsyncStatus(self._set_and_wait(value))

    async def _set_and_wait(self, value: float) -> None:
        """Put the setpoint (blocks through GEECS convergence), then confirm.

        The readback poll reads the streamed position PV (updated at the
        device's ~5 Hz push rate through the gateway) until it is within
        tolerance of the target.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._move_timeout

        # Layer 1: the gateway :SP put rides the blocking GEECS UDP set,
        # through the shared put primitive with move_timeout as its budget.
        await self._put.put(value, timeout=self._move_timeout)

        # Layer 2: confirm the streamed readback converged.
        position = getattr(self, self._readback_attr_name)
        while True:
            current = float(await position.get_value())
            slack = ULP_SLACK * max(abs(current), abs(value))
            if abs(current - value) <= self._tolerance + slack:
                logger.debug(
                    "%s: arrived at %.6g (target=%.6g, tol=%.4g)",
                    self.name,
                    current,
                    value,
                    self._tolerance,
                )
                break
            if loop.time() > deadline:
                raise GeecsMotorTimeoutError(
                    self._geecs_device_name,
                    self._variable,
                    target=value,
                    current=current,
                    timeout=self._move_timeout,
                )
            await asyncio.sleep(0.1)

        if self._settle_time > 0:
            await asyncio.sleep(self._settle_time)
