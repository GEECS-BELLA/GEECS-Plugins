"""CaConfirmSettable — the topology-C device: set X, confirm on Y.

``CaSettable``/``CaMotor`` both poll (or trust GEECS to converge) the
readback of the **same** variable they wrote.  Some devices split the two:
you set one variable, but a *different* variable measures the physical
result.  The production example is the EMQ triplet: the catalog's "EMQ1
Current" writes ``U_EMQTripletBipolar:Current_Limit.Ch1`` (a software limit
GEECS's own blocking set trivially confirms — the limit register echoes what
was written), while the real current is the separate ``Current.Ch1``
variable, which the write never touches.

``ScanVariable.confirm`` names this second variable in the schema
(``geecs_schemas.scan_variables``); this device is what acts on it.

Completion semantics
---------------------
1. The gateway's ``…:SP`` write on ``variable`` forwards to the GEECS UDP set,
   which blocks until *that* variable's own convergence criterion is met (the
   same Layer-1 wait every ``CaSettable`` gets) — this says nothing about the
   confirming variable.
2. A poll on the **confirming variable's** streamed readback then waits for it
   to match the target: analog (float) match by tolerance, discrete
   (string/enum, e.g. a future ``CaShutter``) match by exact equality.

Defaults come from a live characterization of
``U_EMQTripletBipolar:Current.ChN`` (no beam, 2026-07-09,
``Planning/scan_variable_metadata/00_overview.md`` Deferred #5): jitter
0.01 A, response lag <1 s, settles within ~3 frames.  ``tolerance=0.05``
(5x the observed jitter) and ``timeout=10.0`` (comfortably past the settle
time) are sized from those numbers, not guessed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Union

from ophyd_async.core import AsyncStatus
from ophyd_async.epics.core import epics_signal_r

from geecs_bluesky.devices.ca._pv import ca_pv
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.exceptions import GeecsConfirmTimeoutError

logger = logging.getLogger(__name__)

_DEFAULT_TOLERANCE = 0.05
_DEFAULT_TIMEOUT = 10.0

ConfirmValue = Union[float, str]


class CaConfirmSettable(CaSettable):
    """Writes ``variable``, confirms completion on a *different* variable.

    Parameters
    ----------
    device : str
        GEECS device name (e.g. ``"U_EMQTripletBipolar"``).
    variable : str
        The variable this device writes (e.g. ``"Current_Limit.Ch1"``).
    confirm_device : str
        GEECS device name for the confirming variable.  Usually the same as
        *device* (the EMQ case), but named separately since the schema's
        ``confirm`` field is a full ``Device:Variable`` and need not match.
    confirm_variable : str
        The variable that measures the physical result (e.g.
        ``"Current.Ch1"``).
    experiment : str, optional
        Experiment PV-namespace prefix (e.g. ``"Undulator"``).
    name : str
        ophyd-async device name (namespaces the event keys).
    tolerance : float
        Analog (float) match tolerance for the confirming variable.
        ``set()`` resolves when ``|confirm_readback - target| <= tolerance``.
        Default ``0.05`` — see the module docstring for where this number
        comes from.  Ignored for a discrete (string) target, which matches by
        exact equality instead.
    timeout : float
        Maximum seconds to wait for the confirming variable to match, after
        the ``:SP`` write's own convergence.  Default ``10.0``.
    settle_time : float
        Extra seconds to wait after the confirming variable matches, before
        completing the status.
    datatype : type
        Scalar CA datatype for both ``variable`` and ``confirm_variable``
        (default ``float``, the analog case). Pass ``str`` for a discrete
        confirm target (e.g. a future ``CaShutter``'s limit-switch label).
    """

    def __init__(
        self,
        device: str,
        variable: str,
        *,
        confirm_device: str,
        confirm_variable: str,
        experiment: str | None = None,
        name: str = "confirm",
        tolerance: float = _DEFAULT_TOLERANCE,
        timeout: float = _DEFAULT_TIMEOUT,
        settle_time: float = 0.0,
        datatype: type = float,
    ) -> None:
        super().__init__(
            device,
            variable,
            experiment=experiment,
            name=name,
            settle_time=settle_time,
            datatype=datatype,
        )
        # A plain (non-child) readable: the confirming variable is not this
        # device's own data (it may live on a different GEECS device
        # entirely), so it must not appear as an event-row column here — the
        # confirming device's own save set, if any, is the place that reads it.
        # Same datatype as the target variable: for the common analog case
        # (float in, float confirm) that's the natural default; a discrete
        # confirm target (string/enum) passes datatype=str for both.
        self._confirm_readback = epics_signal_r(
            datatype, ca_pv(experiment, confirm_device, confirm_variable)
        )
        self._confirm_device_name = confirm_device
        self._confirm_variable = confirm_variable
        self._tolerance = tolerance
        self._timeout = timeout

    async def connect(
        self, mock: bool = False, timeout: float = 10.0, **kwargs
    ) -> None:
        """Connect this device's own signals plus the confirming readback."""
        await asyncio.gather(
            super().connect(mock=mock, timeout=timeout, **kwargs),
            self._confirm_readback.connect(mock=mock, timeout=timeout),
        )

    def set(self, value: float) -> AsyncStatus:
        """Put *value* to ``variable``; status resolves once ``confirm`` matches.

        Implements :class:`bluesky.protocols.Movable`.
        """
        logger.info(
            "%s: setting %s → %s (confirm on %s/%s, tol=%.4g)",
            self.name,
            self._variable,
            value,
            self._confirm_device_name,
            self._confirm_variable,
            self._tolerance,
        )
        return AsyncStatus(self._set_and_confirm(value))

    async def _set_and_confirm(self, value: float) -> None:
        """Put the setpoint (Layer 1), then poll the confirming variable."""
        loop = asyncio.get_running_loop()

        # Layer 1: the gateway :SP put rides the blocking GEECS UDP set on
        # `variable` — this only confirms *that* variable, not the target.
        await self._setpoint.set(value)

        # Layer 2: poll the confirming variable until it matches.
        deadline = loop.time() + self._timeout
        while True:
            current: ConfirmValue = await self._confirm_readback.get_value()
            if _matches(current, value, self._tolerance):
                logger.debug(
                    "%s: confirmed %s/%s = %r (target=%r, tol=%.4g)",
                    self.name,
                    self._confirm_device_name,
                    self._confirm_variable,
                    current,
                    value,
                    self._tolerance,
                )
                break
            if loop.time() > deadline:
                raise GeecsConfirmTimeoutError(
                    self._geecs_device_name,
                    self._variable,
                    f"{self._confirm_device_name}:{self._confirm_variable}",
                    target=value,
                    current=current,
                    timeout=self._timeout,
                )
            await asyncio.sleep(0.1)

        if self._settle_time > 0:
            await asyncio.sleep(self._settle_time)

    async def disconnect(self) -> None:
        """Per-scan teardown hook — no persistent subscription to release."""


def _matches(current: ConfirmValue, target: ConfirmValue, tolerance: float) -> bool:
    """Analog match by tolerance for numbers, exact equality otherwise.

    A discrete (string/enum) confirming variable — e.g. a future
    ``CaShutter``'s inserted/removed limit switch — matches by equality;
    an analog (float) one matches within ``tolerance``.
    """
    try:
        return abs(float(current) - float(target)) <= tolerance
    except (TypeError, ValueError):
        return current == target
